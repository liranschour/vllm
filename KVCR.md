# KVCR

## Design

### Offloading to CPU memory

Two **mechanisms**, not two deployment modes. Index-and-pin always exists; the
config only decides whether deposit exists *alongside* it:

| | **index-and-pin** | **deposit** |
| --- | --- | --- |
| Exists when | always — the fallback in every configuration, and the only mechanism in the default one (`secondary_g2_slots = 0`, no `g3`, no Guard — what KVCR's quick-start uses) | only when `secondary_g2_slots > 0`, or `g3`, or Guard mode |
| Extra CPU copy | **none** — peers RDMA straight out of vLLM's own host tier | **yes** — one copy into a KVCR-owned pool |
| Who owns residency | vLLM's LRU; block disappears when vLLM evicts it | KVCR policy (admission + eviction + tiering) |
| Survives engine crash | no | only with Guard (pool owned by the sidecar) |

**They coexist, and the choice is per block, not per deployment or per request.**
Configuring a pool does not switch index-and-pin off — it adds a second,
cheaper source of bytes that KVCR prefers when it happens to have them. At serve
time (producer side, `remote_fw_dram.py`):

- `_try_local_source_write` tries the pool first, all-or-nothing: it bails on the
  first key that is not `READY`, whose layout indices do not line up, or if
  `_state_lock` cannot be taken non-blocking.
- On any bail, `_resume_source_pin` (`:1503-1532`) claims whatever KVCR-owned
  residencies exist, computes `unresolved_keys`, pins **only the remainder** out
  of vLLM's host tier, and submits **one** write whose `src_descriptors` mix both
  kinds. A 16-block transfer can be 11 blocks from KVCR DRAM and 5 from pinned
  vLLM memory.

So with a pool configured, a block is served from the pool only if it is
currently resident there — anything the policy dropped, evicted, or never
ingested still goes out via a pin, exactly as in the pool-less configuration.
The consumer side is per-block in the same way: `core.deliver` splits one call
across `LOCAL_G2` / `G3` / `REMOTE_G2` (`core.py:525-553`).

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

### KV sharing and remote peers

Two planes, and they run in opposite directions: the **control** plane is ZMQ and
flows consumer → producer (a request), while the **data** plane is NIXL and flows
producer → consumer (an RDMA `WRITE` push). There is no remote *read*; the
consumer never pulls, it asks to be written into.

(KVCR line numbers below are `ai-dynamo/kvcr @ main`; vLLM CI pins `v0.1.1`, so
they may drift slightly.)

#### 1. Local registration — each process registers only its own memory

At startup, on the progress thread (`progress.py:630`):

- One NIXL agent per KVCR process:
  `nixl_agent(name, nixl_agent_config(num_threads=4, capture_telemetry=True,
  enable_listen_thread=True, listen_port=..., backends=...))`. The name is
  `KVCR-{uuid4}` and the listen port is an ephemeral port the adapter grabs by
  binding `:0` (`manager.py:353-356`, `:402`) — so the agent identity is
  decoupled from host:port.
- `_register_memory_regions()` (`progress.py:641`) walks every region — vLLM's
  framework regions, KVCR-owned DRAM pools, G3 — and calls
  `register_memory([(addr, extent, device_Id, "")], mem_type=...)` with
  `extent = size + (count - 1) * (stride or size)`. **One registration covers the
  whole strided pool**, not one per block; this is why a 50 GiB CPU tier costs a
  single `ibv_reg_mr`-class call rather than millions.
- Then `prepare_memory("", regions)` *and* `prepare_memory(agent_name, regions)`
  build the local prepped descriptor grids, and `get_agent_metadata()` is cached
  as `_nixl_agent_metadata` (`progress.py:659-661`) — the blob a peer needs.

#### 2. Mapping a remote peer — the consumer advertises itself

The peer handshake is carried by `ZmqPeerControlChannel` (msgpack frames,
`control_host` / `control_ports[dp_local_rank]` / `control_advertise_host`,
built at `manager.py:345-350`). Message types: `target_metadata`,
`target_metadata_ack`, `start_write`, `write_refused`, plus the `write_done`
NIXL notification.

Counter-intuitively it is the **target (consumer) that pushes its metadata to the
source**, because the source is the side that will issue the transfer.
`_send_control` (`remote_fw_dram.py:954-987`) attaches to any outbound message:

- `target_agent` — the consumer's NIXL agent name
- `target_agent_metadata` — its `get_agent_metadata()` blob
- `target_regions` — the label → `RegionDescriptor` catalogs (framework + KVCR)

attached only while the source hasn't acked, rate-limited by
`metadata_retry_interval_ms`. With `eager_ctrl_connect` the consumer sends a bare
`target_metadata` up front (`:861`) so the handshake isn't on the first request's
critical path.

The source then does, in `_remote_agent()` (`remote_fw_dram.py:1810-1900`):

1. validate that registration names match region labels,
2. `agent.add_remote_agent(target_metadata)` → remote agent name,
3. `progress.prepare_memory(remote_agent, regions, owner_name=target_agent)`,
4. cache it in `_remote_agents_by_target[target_agent]`, reused by later ops
   (telemetry reports `peer_setup … reused` vs `connected`).

`prepare_memory` (`progress.py:330-378`) is the **pool-and-index** machinery:
regions are grouped by `mem_type`, each label gets an offset in a per-mem-type
grid, and `prep_xfer_dlist(agent_name, rows, mem_type, backends)` is called with
`rows = (addr, size, device_Id, stride or size, count)`. A peer therefore resolves
a `MemoryRef(agent, label, index)` to `(prepped handle, offset[label] + index)`
locally — **block addresses are never sent per transfer**, only pool label plus
element index.

Invalidation: after a disconnect, `check_remote_metadata(remote_agent)` decides
whether NIXL actually dropped the native metadata; if so `release_prepared()`,
drop the cache entry, and bump `_route_generation[target]`. Changed metadata
means `_unload_remote_agent()` then re-add. The code's own TODO notes that a peer
*restart* is not distinguished from an additive metadata update — matching the
observed fragility when a peer is restarted under load.

#### 3. The transfer — `start_write` then one RDMA WRITE

Consumer side (`submit_load` → `kvcr.deliver` → non-resident keys →
`_remote_fw_dram.deliver` → `_TargetPullOp`, state `START_WRITE`):

1. Target sends `start_write` (`remote_fw_dram.py:128-140`) carrying `op_handle`,
   `remaining_timeout_ms`, `source_incarnation`, `keys`, **`dst_descriptors`**
   (its own `MemoryRef`s — where the bytes should land) and
   `allow_layout_subset`; state → `WAITING_WRITE_DONE`.
2. Source resolves the keys to sources. If they live in vLLM-owned memory it
   first acquires a framework pin (`_SourcePinOp`; the adapter's `_resolve_pin`
   takes a hold via `parent.create_store_job`), then builds a `_SourceWriteOp`
   in `READY_TO_WRITE`. The module header states the flow outright:
   `start_write -> local claim/framework pin -> write -> write_done`.
3. Before submitting, the route generation is re-checked **on the submitting
   thread** (`remote_fw_dram.py:334-343`) — NIXL reuses handles for a reused
   agent name, so a stale route must produce a refusal rather than bytes from a
   dead generation.
4. `progress.submit_transfer("WRITE", src_descriptors, dst_descriptors,
   remote_side_agent=..., backend=..., notif_msg=_write_done_notif(...))`
   (`:357`) → `_prepared_indices` on both sides →
   `agent.make_prepped_xfer("WRITE", local, local_idx, remote, remote_idx,
   notif_msg, backends)` → `agent.transfer(handle)`, whose post state is
   `DONE` / `PROC` / `PEND` / `ERR` (`progress.py:242-300`). G3 (`mem_type
   == "FILE"`) bypasses the prepped grids and uses `initialize_xfer` instead.
5. Completion rides the transfer: the notification is delivered to the target,
   whose `_poll_notifications` (`:1908`) decodes `write_done`, matches
   `op_handle`, and honors `completed_indices` / `terminal` for partial success.
   The source can also send `write_done(success=False)` over the control channel
   on deadline or cancel, or `write_refused` (`:1401`) when it won't serve.

Net effect: one RDMA WRITE straight from the producer's registered DRAM into the
consumer's registered DRAM, no staging copy at either end, with both engine
threads only enqueueing work. Default transport is UCX on both legs
(`local_dram_backend`, `remote_fw_dram_backend`); G3 defaults to `GDS_MT`.

## KVCR Guard

Resilience against failures. Based on shared memory.

## llm-d support

No support yet. Searched the whole `llm-d` org — code, all repos, every
versioned docs tree (0.7 → 0.10), PRs and issues: **zero** occurrences of
`kvcr` / "KV Cache Runner", and zero of `kv_hints` / `kvHints`. llm-d's KV
offloading docs name LMCache, Mooncake and NVIDIA **KVBM** as the out-of-tree
options, never KVCR. The asymmetry is notable: KVCR's own README lists
`llmd-router` under "Routers", but only the Dynamo KV Router is marked
*supported*, so treat that listing as aspirational.

Worth knowing that llm-d's founding architecture doc takes the opposite
position by design: it "prioritizes a strong operational boundary between
in-memory prefix cache tiers and local or remote storage tiers **rather than a
unified memory API like the KV Block Manager**" — an objection that transfers
directly to KVCR, KVBM's successor in this role.

### What it would take for llm-d's router to replace Dynamo's KV router

KVCR needs a KV-aware router to be useful at all: the quick-start states that
vLLM's own router or `consistent_hash` "does not produce those hints". Nothing
ties it to Dynamo though — the envelope is deliberately orchestrator-agnostic
(RFC #53421 is titled "First-Class, **Orchestrator-Agnostic** KV Hint
Envelope").

**llm-d already has the hard half.** The KV-Cache Indexer (moved into
`llm-d-router`, llm-d/llm-d-router#1886) keeps "a global, near-real-time view of
KV-Cache block locality across a fleet of vLLM pods" from vLLM KVEvents,
tracking which blocks sit on which node and which tier, and exposes
`Score(prompt, pods)`. That is functionally the inventory Dynamo's KV router
maintains. What is missing is not the index but the *emission*.

Four pieces of work, in dependency order:

1. **Expose `kv_hints` over HTTP — upstream vLLM, and the actual blocker.**
   Today the envelope is plumbed only on the Python path
   (`AsyncLLM.generate(kv_hints=...)` → `input_processor` → `EngineCoreRequest`
   → `Request` → `ReqContext`) and there are **zero** references to it anywhere
   under `vllm/entrypoints/`. That is why the quick-start runs
   `python3 -m dynamo.vllm` rather than `vllm serve` — Dynamo's worker wraps
   `AsyncLLM` in-process. An external HTTP gateway like llm-d's router cannot
   attach a hint at all. The fix has a clean precedent in the same files:
   `kv_transfer_params: dict[str, Any] | None`
   (`vllm/entrypoints/openai/completion/protocol.py:226`, plus chat, responses,
   anthropic), which is how NixlConnector P/D proxies pass params over HTTP.
   A `kv_hints` field would be the same shape, and the typed envelope already
   exists in `vllm/v1/kv_hints/protocol.py`. Note `vllm_xargs` is **not** a
   shortcut: the adapter reads the typed `req_context.kv_hints`, not
   `extra_args`. The only alternative is llm-d running its own Python frontend
   around `AsyncLLM`, which is heavier and against its gateway architecture.
2. **Map pod → KVCR control endpoint (llm-d-router).** The hint's
   `source_control_endpoint` is
   `tcp://<control_advertise_host>:<control_ports[dp_local_rank]>` — the ZMQ
   control endpoint, *not* the HTTP one, and indexed by local DP rank. KVCR's
   inventory events do not carry it, so the router must derive it from
   deployment config (the `POD_IP` + base-port convention), DP-rank aware.
3. **Block-hash and tier agreement — mostly free.** `block_hashes` are uint64
   block keys the indexer already ingests from KVEvents. Needs
   `self_describing_kv_events: true` (which KVCR requires for local G2/G3
   inventory anyway), and the router should also consume KVCR's own inventory
   events — they arrive tagged `ownership="kvcr"` with medium CPU/STORAGE — so
   it can price "in that pod's vLLM CPU tier" differently from "in that pod's
   KVCR DRAM/SSD".
4. **Emission policy and envelope conformance.** Decide when a remote fetch
   beats local recompute and emit only then; hints are advisory and a useless
   one merely degrades to `MISS`. The parser is strict: `protocol_version
   "0.1"`, `action_type "kv.fetch"`, `action_version "1.0"`, `mode` must be
   `"copy"`, `no_retain` absent, `block_hashes` non-empty. Each pod's tier
   config must also carry `"router_capabilities": ["router_hint"]` or the tier
   refuses to construct.

No NATS, etcd, or Dynamo runtime is needed. Item 1 is the only upstream change,
it is small and mechanical, and it would unblock *any* orchestrator — which
makes it the natural thing to propose first.
