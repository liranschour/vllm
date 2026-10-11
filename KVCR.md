# KVCR

## Design

### Offloading to CPU memory

Two distinct paths exist, and which one a deployment uses is decided purely by
whether a KVCR-owned pool is configured:

| | **index-and-pin** | **deposit** |
| --- | --- | --- |
| Config | `secondary_g2_slots = 0`, no `g3`, no Guard (the default, and what KVCR's own quick-start uses) | `secondary_g2_slots > 0`, or `g3`, or Guard mode |
| Extra CPU copy | **none** — peers RDMA straight out of vLLM's own host tier | **yes** — one copy into a KVCR-owned pool |
| Who owns residency | vLLM's LRU; block disappears when vLLM evicts it | KVCR policy (admission + eviction + tiering) |
| Survives engine crash | no | only with Guard (pool owned by the sidecar) |

The adapter always registers vLLM's *existing* primary host pool with KVCR's
NIXL agent as one `RegionDescriptor(addr, size=block_size_bytes, count=n_chunks)`
(`manager.py:430`), and every `MemoryRef` it builds is just an element index into
that region (`_make_descriptor`, `manager.py:628`) — never a KVCR buffer. That is
what makes the no-copy path possible: KVCR can serve a peer by pinning a vLLM
chunk in place (`request_pin` → `_resolve_pin`, `manager.py:156`/`:179`) and
writing from it. `remote_fw_dram.py` states the source-side flow outright:
`start_write -> local claim/framework pin -> write -> write_done`.

#### `submit_store` — how a block actually lands in KVCR

vLLM side (`vllm/v1/kv_offload/tiering/kvcr/manager.py:494`) is thin and does no
I/O itself:

1. Map each `(OffloadKey, chunk_id)` of the `TransferJob` to
   `{BlockKey: [MemoryRef]}`, where the `MemoryRef` points at the chunk slot
   **inside vLLM's primary CPU tier** — so the "source" of the store is memory
   KVCR already has registered.
2. Empty batch → complete the job successfully inline.
3. Otherwise call `kvcr.deposit(blocks)` and record
   `op_handle → (job_id, n_blocks, success, None)` in `_jobs_by_op`. Completion
   is picked up later by `_poll_finished_jobs()` draining `kvcr.poll_completed()`,
   decrementing `remaining` per entry; the `JobResult` is only emitted once every
   key of the op has reported.

KVCR side (`kvcr @ main`, `api.py:168` → `core.py:556` → `local_dram.py:322`):

1. **No KVCR-owned pool → immediate failure.** `core.deposit` checks
   `self._local_dram` (which is `None` whenever `backend_configs.local_dram` is
   `None`, `core.py:283`) and, if absent, completes the op with
   `OpEntryStatus.FAILED` for *every* key (`core.py:570-574`). The adapter calls
   `deposit` unconditionally, so in a pure index-and-pin deployment every store
   job reports failure. `deposit` is only meaningful when KVCR owns DRAM.
2. **Per-key fast paths** (`local_dram.py:350`): if the block already has a
   `local_dram` residency — `READY` → SUCCESS (or a claim handle when
   `no_evict`), `DISCARDING` → FAILED, layout mismatch → FAILED. No copy.
3. **Admission policy** decides before any memory is reserved:
   `policy.decide_ingest(meta, CacheTier.FW_G2, required_local=no_evict, framework_hints=hints)`.
   `PlacementAction.DROP` → `OpEntryStatus.DROPPED` (a *refusal*, distinct from a
   failure).
4. **Slot allocation** `_allocate_slots(...)` reserves pool slots, possibly
   evicting other blocks. If capacity can't be freed now, the key is parked on a
   capacity waiter rather than failed (`eviction_pending`); evicted keys are
   published immediately as `InventoryEvent(LOCAL_G2, removed=True)`
   (`local_dram.py:398`).
5. **Residency goes `FILLING`**, and the actual bytes move on the progress
   thread: one batched `_LocalCopyOp` carrying `src_descriptors` (vLLM's chunks)
   and `dst_descriptors` (the reserved KVCR slots), executed through the
   configured backend (UCX by default) with the operation deadline attached.
   The engine thread never blocks on the copy.
6. **On completion** the residency flips to `READY` (`local_dram.py:702`), the
   block enters the policy-scored eviction queue, and KVCR publishes
   `InventoryEvent(LOCAL_G2, removed=False)` (`local_dram.py:762`) → the adapter's
   `_record_inventory` (`manager.py:611`) turns that into an
   `OffloadingEvent(medium=CPU, ownership="kvcr")`, which is how the router's
   global index learns KVCR now holds the block. (This is also why local G2/G3
   requires `self_describing_kv_events`.)

So the answer to "does it keep an extra CPU copy": only on the `deposit` path,
and that copy is the product — it is what gives retention beyond vLLM's LRU,
SSD tiering underneath it, and (with Guard) survival of the engine process.
On the default path KVCR stores nothing of its own; it indexes vLLM's host tier
and pins chunks in place for peer reads.

## llm-d support

No support yet.
