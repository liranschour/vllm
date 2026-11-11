# Off-thread promotion for P2P lookups

Status: draft design, base for a PR
Branch: `worktree-p2p-offthread-promotion`
Code references are against `01fba62494` (upstream main, 2026-11-11).

## 1. Summary

A symmetric P2P pull (`remote_kv_source`) currently costs at least two engine
steps of the *pulling* engine before the request can be scheduled. The real
work fits in a fraction of one step: the probe round trip is ~2-6 ms and the
NIXL fetch ~10-77 ms. We measure ~497 ms of lookup wait per pull in
production, against ~6 ms in the unit/e2e test, because the wait is quantized
to the pulling engine's step time, and production steps are long.

The step quantum comes from one thing: the answer to a lookup (and the start of
the fetch it enables) is only acted on by the next `schedule()` call. This
design moves the *promotion* (allocate a primary CPU slot, submit the P2P
fetch) onto the tier-poll thread, so it is triggered when the
`LookupRespMsg` arrives, instead of at the next step. The scheduler is not
changed: its single lookup site finds the promotion already in flight or
done, through primary-tier state that already exists.

Expected effect: one step boundary removed from every symmetric pull
(`lookup_async_delay` drops from ~2 × T_step to ~1 × T_step on the probe path).

## 2. Background

### 2.1 Where the time goes today

Consumer side, one symmetric pull, in steps of the pulling engine:

```text
schedule(k)    lookup -> RETRY (register probe)        ; on_schedule_end: LookupMsg out
forward k      tier-poll thread receives LookupRespMsg, caches it      <- idle
schedule(k+1)  lookup -> HIT -> _initiate_promotion -> HIT_PENDING     ; on_schedule_end: FetchMsg out
forward k+1    NIXL fetch
schedule(k+2)  lookup -> primary HIT -> load scheduled                 <- lookup_async_delay ends
```

The producer is *not* step-gated in the warm case. It serves LookupMsgs on its
tier-poll thread (`poll_tiers` -> `serve_external_requests`, 1 ms interval,
since #58168). It only waits for its own step when a key is HIT_PENDING
(its own GPU->CPU store still in flight) or RETRY (promotion from a lower
tier). Those cases are out of scope here (see §10).

So both wasted boundaries are on the consumer, and both have the same cause:
nothing acts on the reply until the scheduler re-runs the lookup.

### 2.2 Relevant code

| What | Where |
| --- | --- |
| Only lookup call site in the scheduler cycle (WAITING loop) | `v1/core/sched/scheduler.py:1144` `get_num_new_matched_tokens` |
| Deferral: `None` -> `skip_request` | `scheduler.py:1150-1155` |
| Prefix scan; RETRY keeps scanning, MISS breaks | `offloading/scheduler.py:828` `_maximal_prefix_lookup` |
| Keys + positions recorded before lookup | `offloading/scheduler.py:414` `update_offload_keys` |
| Tiering lookup: primary first, then secondaries; HIT -> promotion | `tiering/manager.py:699` `lookup` |
| Promotion: allocate primary slot now, submit at step end | `tiering/manager.py:795` `_initiate_promotion`, `:849` `_flush_pending_promotions` |
| Promotion completion -> primary `complete_write` | `tiering/manager.py:539` `_complete_promotion` |
| Tier-poll round (no promotion flush, by design) | `tiering/manager.py:476` `poll_tiers` |
| Primary HIT_PENDING while slot not ready | `cpu/manager.py:214` `lookup` |
| P2P consumer lookup (probe, cached answers) | `tiering/p2p/manager.py:423` `lookup` |
| P2P submit_load -> FetchMsg | `tiering/p2p/manager.py:609` `submit_load` |
| Probe cache, reply handling, fetch clears probes | `tiering/p2p/session/client.py:447` `register_lookup`, `:528` `on_lookup_resp`, `:256` `request_blocks` (`probes.clear()` at `:308`) |

## 3. Goals and non-goals

Goals

- Remove the k -> k+1 boundary on symmetric P2P pulls by starting the
  promotion when the lookup reply arrives.
- No change to the scheduler or to the connector's lookup semantics.
- Identical behavior when the tier-poll thread is disabled
  (`tier_poll_interval_s = 0`) or the feature flag is off.

Non-goals (follow-ups, §10)

- Probing on request arrival (`on_new_request`) to remove the remaining
  boundary.
- Producer-side HIT_PENDING / RETRY step waits.
- FS / obj async lookups (`AsyncLookupManager`), which have the same RETRY
  quantum but don't opt into the tier-poll thread.
- PD path (`remote_prefiller`): it already returns an eager HIT and pays only
  the promotion boundary.

## 4. Design

### 4.1 New timeline

```text
schedule(k)    lookup -> RETRY (register probe)        ; LookupMsg out
forward k      poll thread: LookupRespMsg -> resolved -> eager promotion
                 (prepare_write + submit_load -> FetchMsg) -> NIXL fetch
                 -> TransferDone -> _complete_promotion (poll thread)
schedule(k+1)  lookup -> primary HIT                   -> load scheduled
               (or primary HIT_PENDING if the fetch is still in flight -> HIT at k+2,
                same as today, never worse)
```

With async scheduling (batch queue 2), `schedule(k+1)` runs right after
dispatching step k, before the reply is back. The saving still holds: today
the HIT lands at `schedule(k+3)`, with eager promotion at `schedule(k+2)`.

### 4.2 Coordination with the scheduler: primary-tier state

The scheduler needs no new flag to know that a promotion already happened.
`TieringOffloadingManager.lookup()` asks the primary tier first and
short-circuits on HIT / HIT_PENDING, before any secondary tier is asked. The
primary slot that `_initiate_promotion` allocates *is* the coordination
record:

| Stage (any thread, under `manager.lock`) | Primary state | Scheduler lookup returns |
| --- | --- | --- |
| `prepare_write` in `_initiate_promotion` | slot exists, `ref_cnt=-1`, not ready | HIT_PENDING (P2P tier not asked) |
| fetch in flight | same | HIT_PENDING -> defer |
| `_complete_promotion` -> `complete_write(True)` | ready, `ref_cnt=0` | HIT -> `prepare_load` takes a ref |
| fetch failed -> `complete_write(False)` | slot freed | primary MISS -> P2P `_failed_req_ids` -> MISS -> recompute |

`_request_load_sources` is filled by `_initiate_promotion` too, so
`get_load_source()` keeps attributing the hit to the P2P tier.

All transitions happen under `manager.lock`. `poll_tiers` holds it for the
whole round, and the scheduler holds it for the step region
(`_acquire_step` ... `build_connector_meta`). Whichever path reaches a key
second sees the primary slot and does not promote again. `prepare_write`
also skips keys that are already present.

### 4.3 Tier hook (`tiering/base.py`)

The promotion belongs to the tiering manager. A tier sees only
`ParentManager` and must not call `_initiate_promotion`. Add one optional
hook to `SecondaryTierManager`:

```python
def take_resolved_lookups(self) -> list[ResolvedLookup]:
    """Deferred lookups this tier resolved since the last call.

    Called by the tiering manager's tier-poll round, under the manager
    lock, only for tiers with tolerates_off_thread_polling. Each entry
    names a request and the keys the tier now answers HIT for, in the
    order the scheduler would accept them. Default: none.
    """
    return []

@dataclass(slots=True, frozen=True)
class ResolvedLookup:
    req_context: ReqContext
    keys: list[OffloadKey]   # leading HIT run(s), prefix order per group
```

### 4.4 P2P tier: building `ResolvedLookup`s

State (consumer side, `P2PSecondaryTierManager`):

- `_deferred_probes: dict[str, ReqContext]`, keyed by `kv_request_id`.
  Filled in `lookup()` when the probe path returns RETRY (only when eager
  promotion is enabled). Popped on resolve, on `on_request_finished`, on
  `_failed_req_ids`, and on session reap.

Resolution trigger: after each `_poll_once`, for every `kv_request_id` in
`_deferred_probes` whose client state has no `None` probe and no `unsent`
entry (i.e. every key the scheduler registered has its answer). A request
whose range was registered across several steps waits for all of its
LookupMsgs.

Key selection, which reproduces `_maximal_prefix_lookup` at step k+1:

1. Partition the request's probes by `get_offload_group_idx(key)`.
2. Within a group, sort by `req_context.get_offload_key_position(key)`.
3. Take keys while the probe is `True`, and stop at the first `False`.
4. Concatenate the groups. Skip the request if the result is empty.

Keys between probed keys that the P2P tier never saw are primary hits. The
scheduler didn't ask secondaries for them, so they don't break the run. This
matches the scheduler, which counts HIT and HIT_PENDING as hits and only
breaks on MISS.

Gating (see §6.2): this ordering is only exact for full-attention groups. The
tier queues a `ResolvedLookup` only when the manager was built with
`eager_promotion=True`, which the spec sets only for KV configs with no
sliding-window, Mamba or EAGLE groups.

### 4.5 Tiering manager: eager promotion in `poll_tiers`

```python
def poll_tiers(self) -> None:
    gate = self._processed_jobs_this_step
    self._processed_jobs_this_step = True
    try:
        self._process_finished_jobs(self._polled_tiers)
        for tier in self._polled_tiers:
            tier.serve_external_requests(self._tier_parents[tier])
        if self._eager_promotion:
            self._promote_resolved_lookups()        # new
    finally:
        self._processed_jobs_this_step = gate

def _promote_resolved_lookups(self) -> None:
    assert not self._pending_load_submissions   # empty outside the step region
    for tier in self._polled_tiers:
        tier_idx = self._tier_index[tier]
        for rl in tier.take_resolved_lookups():
            ctx = rl.req_context
            st = self._req_state.get(ctx.req_id)
            if st is None or st.is_finished or self._promotions_held:
                continue
            if not ctx.load_tier_filter.allows(tier.medium, tier.locality):
                continue
            for key in rl.keys:
                r = self.primary_tier.lookup(key, ctx)
                if r is not LookupResult.MISS:
                    continue                      # already there / in flight
                if not self._initiate_promotion(tier_idx, key, ctx):
                    break                         # primary full: scheduler decides
            self._metrics.on_eager_promotion(tier_idx, ...)
    self._flush_pending_promotions()
```

Notes

- `_pending_load_submissions` is empty whenever the poll thread holds the lock:
  it is filled only inside the scheduler's step region and drained by that
  step's `on_schedule_end`. The assert documents that, and the flush submits
  only this round's entries. The `poll_tiers` docstring says promotion flushes
  are excluded to keep per-step batches whole. Eager promotion does not
  fragment them: each `ResolvedLookup` is one request, which is already one
  batch (`PendingPromotion` per (tier, request)).
- It does not call `self.lookup()`. That would double-count lookup metrics
  (the scheduler looks the same keys up again at k+1) and could start FS/obj
  async probes off-thread.
- Tier preference: today, if an earlier secondary tier answers RETRY and
  P2P answers HIT, `lookup()` promotes from P2P anyway (RETRY doesn't stop the
  loop). Eager promotion promotes from P2P too, so source choice is unchanged.

### 4.6 Prerequisite: keep MISS answers across a fetch (suspected existing bug)

`request_blocks` clears *all* probes when the fetch closes a round
(`client.py:308`). The comment there is about stale `True`. A stale `True`
could point at a block the producer has since evicted, so clearing it is
right. Clearing `False` costs a round, though:

```text
k+1  scan: keys 0..h-1 HIT -> promote; key h cached False -> MISS -> break; FetchMsg (clears probes)
k+2  scan: keys 0..h-1 primary HIT; key h: primary MISS -> P2P -> not in probes -> RETRY  (!)
k+3  producer answers MISS again -> MISS -> request finally scheduled
```

So a partial hit seems to pay one extra lookup round today. With eager
promotion it gets worse: the fetch (and the clear) happens *before* the
scheduler's k+1 scan, so the scan meets key h as a fresh probe at k+1.

Fix: in `request_blocks`, drop only the `True` (and any `None`) probes, and
keep `False`. A kept `False` is conservative: at worst the request recomputes
a block that has become available since, which is today's MISS outcome
anyway.

This is unverified. PR step 1 starts with a failing test that reproduces the
extra round, and only then applies the fix.

### 4.7 Configuration

- P2P tier config key `eager_promotion: bool` (default `false` in the first
  PR). Add it to `config_info_keys`.
- The spec resolves an effective `TieringOffloadingManager(eager_promotion=...)`:
  the key is on, and `tier_poll_interval_s > 0`, and the KV config is
  full-attention only. If the key is set but the KV config is not eligible,
  log once at INFO and disable it.

## 5. Behavior and edge cases

| Case | Behavior |
| --- | --- |
| Scheduler lookup races the poll thread | Impossible. Both run under `manager.lock`, and the second sees the primary slot |
| Reply lands but request already scheduled / finished | `_deferred_probes` popped in `on_request_finished`; `_req_state` check skips it |
| Request aborted while eager fetch in flight | Existing path: `finish()` aborts loads -> promotion fails -> `complete_write(False)` frees the slot |
| Fetch fails / peer dies | Existing path: `_failed_req_ids` + failed `JobResult` -> slot freed -> scheduler sees MISS -> recompute |
| Primary full | `_initiate_promotion` returns False -> stop; the scheduler gets the same outcome (MISS) |
| `reset_cache` hold (`_promotions_held`) | Skip, same as `lookup()` |
| `load_tier_filter` excludes P2P | Skip, same as `lookup()` |
| Session not send-ready | `request_blocks` already queues `pending_fetch`; no change |
| Eviction between `complete_write` and the scheduler's `prepare_load` | Slot has `ref_cnt=0` and can be evicted (this window exists today too; eager promotion makes it longer). Next lookup: primary MISS -> P2P -> probes were cleared by the fetch -> new LookupMsg under `round_seq+1`. Producer pins again (one pin per round; the previous round's pin was already released at finalize) or answers MISS. Cost: one extra round + re-transfer. Nothing is double-pinned and nothing leaks |
| Partial fetch of a round's supply | Unchanged: the producer releases leftover pins at round finalize |
| PD path (`remote_prefiller`) | Never registers probes -> never queued |
| `tier_poll_interval_s = 0` | No poll thread -> hook never called -> today's behavior |
| Stats | `prepare_write` off-thread appends to the CPU tier's per-step allocation stats (`allocation_sizes_in_current_batch`). They are reported with the next step. Under the lock, so this is safe |

## 6. Risks

1. **Over-promotion.** We promote before the scheduler decides. Bounded by the
   keys the scheduler itself registered (its candidate range, already capped
   by `max_load_tokens`), cut at the first MISS. Wasted only if the request is
   finished or preempted in between.
2. **Non-full-attention groups.** Sliding-window groups scan from the end and
   don't break on MISS. EAGLE groups drop a volatile trailing chunk. Mamba
   aligns to a block size. The §4.4 ordering is wrong for these, so they are
   gated off (§4.7), and a follow-up can pass per-group scan rules down.
3. **Longer eviction window under CPU-tier pressure** (see §5). Measure
   re-probe rounds first. If they matter, add a short consumer pin released
   when the scheduler adopts the chunk (`prepare_load`) or at request finish.
4. **More work on the poll thread while holding the lock.** One
   `prepare_write` + `submit_load` per resolved request, which is the same
   cost the scheduler pays today, moved off the step region. Check scheduler
   lock-wait p99 with the existing lock-hold harness.

## 7. Observability

- New counter `vllm:kv_offload_eager_promoted_chunks_total{tier}`, plus a
  counter of requests promoted eagerly.
- Existing `vllm:kv_offload_lookup_async_delay_seconds` is the headline: it
  should fall by ~1 × T_step.
- Existing `vllm:kv_offload_p2p_lookup_rtt_seconds` should stay unchanged
  (it measures the wire).
- Debug log on eager promote: `kv_request_id`, keys, the first-MISS position.

## 8. Testing

Unit

- `tests/v1/kv_offload/tiering/p2p/test_sessions.py`
    - fetch keeps `False` probes and drops `True` ones (§4.6), and a
    re-scheduled request with a partial hit sends no second LookupMsg.
- `tests/v1/kv_offload/tiering/p2p/test_manager.py`
    - `take_resolved_lookups` returns nothing until every registered probe is
    answered (multi-step registration).
    - leading-run selection per group by position: stops at the first False,
    tolerates gaps that are primary hits, skips all-MISS.
    - PD requests and disabled flag never queue; `on_request_finished` / reap
    drops pending entries.
- `tests/v1/kv_offload/tiering/test_tiering_offloading.py` (fake polled tier)
    - `poll_tiers` promotes and submits immediately; the scheduler's next
    `lookup()` returns HIT_PENDING, then HIT after completion processed by
    `poll_tiers`.
    - guards: `_promotions_held`, tier filter, finished request, primary full
    stops the run, key already in primary is not promoted twice.
    - no double promotion when the scheduler looks up first in the same window.
- `tests/v1/kv_offload/tiering/test_tier_poll_thread.py`
    - real thread: reply -> promotion without any `on_schedule_end`.

E2E (llmd pods, Llama-3.2-1B and 3.1-8B, symmetric `remote_kv_source`)

- A/B on the `lock_api_conc` harness (`tests/v1/kv_offload/tiering/p2p/results/`
  scripts): idle and busy consumer, c ∈ {1,2,4,8,16}. Seed and pull per wave.
  Report `lookup_async_delay / T_step` (expect ~2 -> ~1), TTFT, and
  external prefix cache hits (must not drop).
- Busy consumer is the case that matters: load the *pulling* engine with
  8k-token prefills so T_step is a few hundred ms.
- Accuracy: `run_accuracy_test.sh` matrix with the flag on.

## 9. PR plan

1. **[Bugfix][KV Offload][P2P] Keep MISS probe answers across a fetch.** Test
   first, then the fix. Stands alone.
2. **[KV Offload][P2P] Eager promotion of resolved P2P lookups on the tier-poll
   thread.** Covers the tier hook, the manager path, the P2P implementation,
   config/gating, metrics and unit tests. Flag defaults off.
3. Flip the default to on for eligible configs after the e2e A/B.

## 10. Follow-ups

- **Probe on arrival.** Send the LookupMsg from `on_new_request` (the session
  is already opened there). Combined with eager promotion, a pull can finish
  while the request waits in the queue, so the first `schedule()` finds HIT.
  That removes the remaining boundary.
- **Producer "pending HIT".** Answer HIT_PENDING keys right away, park them
  like `pending_cascade_keys`, and fail the fetch if the store fails. The
  producer's `complete_store` wait then overlaps the fetch instead of coming
  before the reply.
- **Generalize to `AsyncLookupManager` tiers** (FS/obj) once they opt into
  off-thread polling.
- **Per-group scan rules** so sliding-window / hybrid models qualify.
