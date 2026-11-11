# Off-thread promotion for P2P lookups

Status: draft design, base for a PR (rev 2: connector-driven selection)
Branch: `worktree-p2p-offthread-promotion`
Code references are against `01fba62494` (upstream main, 2026-11-11).

## 1. Summary

A symmetric P2P pull (`remote_kv_source`) currently costs at least two engine
steps of the *pulling* engine before the request can be scheduled. The real
work fits in a fraction of one step: the probe round trip is ~2-6 ms and the
NIXL fetch ~10-77 ms. We measure ~497 ms of lookup wait per pull in
production, against ~6 ms in the unit/e2e test, because the wait is quantized
to the pulling engine's step time, and production steps are long.

The step quantum comes from one thing: the answer to a lookup, and the fetch
it enables, is only acted on by the next `schedule()` call. This design acts on
it as soon as the `LookupRespMsg` arrives. The tier-poll thread re-runs the
**connector's own lookup** for that request in a speculative mode. The
speculative run promotes (allocates a primary CPU slot, submits the P2P fetch)
exactly the keys the scheduler would promote at the next step, for any
attention type: full, sliding-window, Mamba, EAGLE, hybrid. Its result is
discarded. The scheduler's own lookup at the next step makes the load decision
as today, and finds the promotion in flight or done through primary-tier state
that already exists.

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

### 2.2 Which keys get promoted is decided by the connector's scan

Promotion is a side effect of `TieringOffloadingManager.lookup()` returning HIT
from a secondary tier. So the set of promoted keys is exactly the set of keys
the connector's scan *visits*, and the connector's scan is the one that knows
the KV layout (`_lookup_complete_chunks`, `offloading/scheduler.py:904`):

- full-attention groups: prefix scan; RETRY keeps scanning, MISS breaks
  (`_maximal_prefix_lookup`, `:828`);
- sliding-window groups: scan from the end until a full window of hits,
  with a wider first window for a partial rightmost chunk
  (`_sliding_window_lookup`); only the trailing section is promoted;
- Mamba alignment, EAGLE's volatile trailing chunk, `max_load_tokens`;
- the cross-group convergence loop (a group can tighten the boundary and
  force earlier groups to re-scan);
- the partial-tail lookup (`_lookup_partial_tail_hit`).

Rev 1 of this doc tried to reproduce the selection inside the P2P tier and had
to gate everything except full attention. Rev 2 does not reproduce it: it runs
the connector's scan.

### 2.3 Relevant code

| What | Where |
| --- | --- |
| Only lookup call site in the scheduler cycle (WAITING loop) | `v1/core/sched/scheduler.py:1144` `get_num_new_matched_tokens` |
| Deferral: `None` -> `skip_request` | `scheduler.py:1150-1155` |
| Connector lookup entry, preconditions | `offloading/scheduler.py:1364` `get_num_new_matched_tokens` (guards at `:1401-1417`) |
| Connector scan | `offloading/scheduler.py:1147` `_lookup` -> `:904` `_lookup_complete_chunks`, `_lookup_partial_tail_hit` |
| Connector side effects in the scan | `_events_tracker.record_lookup` (`:844`, `:1259`), `record_partial_lookup` (`:1324`), `_maybe_register_shared_prefix_junction` (`:1329`) |
| Connector lock discipline | `offloading/scheduler.py:752` `_acquire_step`, `:770` `_manager_locked` |
| Tiering lookup: primary first, then secondaries; HIT -> promotion | `tiering/manager.py:699` `lookup` |
| Promotion: allocate primary slot now, submit at step end | `tiering/manager.py:795` `_initiate_promotion`, `:849` `_flush_pending_promotions` |
| Promotion completion -> primary `complete_write` | `tiering/manager.py:539` `_complete_promotion` |
| Tier-poll round (no promotion flush, by design) | `tiering/manager.py:476` `poll_tiers` |
| Manager lock (plain, non-re-entrant `threading.Lock`) | `tiering/manager.py:299` |
| Primary HIT_PENDING while slot not ready | `cpu/manager.py:214` `lookup` |
| P2P consumer lookup (probe, cached answers) | `tiering/p2p/manager.py:423` `lookup` |
| P2P submit_load -> FetchMsg | `tiering/p2p/manager.py:609` `submit_load` |
| Probe cache, reply handling, fetch clears probes | `tiering/p2p/session/client.py:447` `register_lookup`, `:528` `on_lookup_resp`, `:256` `request_blocks` (`probes.clear()` at `:308`) |

## 3. Goals and non-goals

Goals

- Remove the k -> k+1 boundary on symmetric P2P pulls by starting the
  promotion when the lookup reply arrives.
- Promote exactly what the scheduler would promote at k+1, for every KV
  layout the connector supports, without duplicating its scan rules.
- No change to the scheduler, or to the connector's load decision.
- Identical behavior when the tier-poll thread is disabled
  (`tier_poll_interval_s = 0`), the feature flag is off, or the manager is
  not a `TieringOffloadingManager`.

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
forward k      poll thread: LookupRespMsg -> request fully answered
                 -> connector speculative _lookup(req) -> manager.lookup HITs
                 -> _initiate_promotion -> flush -> FetchMsg -> NIXL fetch
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

### 4.3 Components and call flow

```text
P2P tier (consumer)          TieringOffloadingManager            OffloadingConnectorScheduler
-------------------          ------------------------            ----------------------------
lookup() -> RETRY:
  remember kv_request_id
  -> req_id
_poll_once (poll thread):
  on_lookup_resp ...
  request fully answered
  -> queue req_id
                             poll_tiers() [holds lock]
                               _process_finished_jobs
                               serve_external_requests
                               for req_id in
                                 tier.take_resolved_lookups():
                                 handler(req_id)  ───────────────►  _speculative_lookup(req_id)
                                                                      guards (§4.6)
                                                                      with speculative mode:
                                                                        _lookup(req_status)
                                                  ◄── manager.lookup(key) per scanned key
                                 (secondary HIT -> _initiate_promotion)
                               _flush_pending_promotions()
                                 -> P2P submit_load -> FetchMsg
```

The tiering layer never imports the connector. The connector registers a
callback with the manager, and the manager calls it.

### 4.4 Tier hook (`tiering/base.py`)

```python
def take_resolved_lookups(self) -> list[str]:
    """req_ids whose deferred lookups this tier fully answered since the
    last call.

    Called from the tiering manager's tier-poll round, under the manager
    lock, only for tiers with tolerates_off_thread_polling. "Fully
    answered" means a re-run of the request's lookup would no longer get
    RETRY from this tier. Default: none.
    """
    return []
```

### 4.5 P2P tier: reporting fully-answered requests

State (consumer side, `P2PSecondaryTierManager`):

- `_deferred_probes: dict[str, str]`, `kv_request_id -> req_id`. Filled in
  `lookup()` when the probe path returns RETRY, and only when eager promotion
  is enabled. Popped when reported, in `on_request_finished`, when the id
  enters `_failed_req_ids`, and on session reap.

Trigger: at the end of `_poll_once`, for every entry whose client state has
no `None` probe and no `unsent` key, move `req_id` to the report list. Every
key the scheduler registered then has an answer. A request whose range was
registered across several steps waits for all of its LookupMsgs. Expose this
on `ClientRole` as `is_lookup_resolved(kv_request_id) -> bool`.

The tier no longer chooses keys. That is the connector's job (§4.6).

### 4.6 Connector: speculative lookup

Registration, in `OffloadingConnectorScheduler.__init__`:

```python
if eager_promotion_enabled:          # §4.9
    self.manager.set_resolved_lookup_handler(self._speculative_lookup)
```

`set_resolved_lookup_handler` is a new optional method on `OffloadingManager`
(`kv_offload/base.py`) with a no-op default. `TieringOffloadingManager`
implements it.

Handler (runs on the poll thread, **already under `manager.lock`**):

```python
def _speculative_lookup(self, req_id: str) -> None:
    req_status = self._req_status.get(req_id)
    if req_status is None:
        return
    req = req_status.req
    if (
        req.status != RequestStatus.WAITING          # scheduled / preempted / finished
        or req.skip_reading_external_kv
        or req_status.load_failed
        or req_status.transfer_jobs                  # same guard as get_num_new_matched_tokens
    ):
        return
    self._speculative = True
    try:
        with self.manager.speculative_lookups():     # suppress manager lookup metrics
            self._lookup(req_status)                 # result discarded
    finally:
        self._speculative = False
```

Inputs: it reuses what `get_num_new_matched_tokens` recorded at step k:
`req_status.num_locally_computed_tokens` and the offload keys. It does not
call `update_offload_keys()`, so it never touches `req.block_hashes`.
`max_num_new_tokens` is `None`, as in the scheduler's call.

Side effects of `_lookup`, and how each is handled:

| Side effect | Treatment in speculative mode |
| --- | --- |
| `manager.lookup()` -> secondary HIT -> `_initiate_promotion` | **Wanted.** This is the point |
| `manager.lookup()` -> new probes (P2P, partial-tail key) / FS/obj async lookups | Kept. Equivalent to what k+1 would register. Flushed at the next `on_schedule_end`, as today. The tiering docs already allow any tier to be reached from the poll thread through the manager |
| Manager lookup metrics (`_metrics.on_lookup`) | Suppressed via `manager.speculative_lookups()`, or every scanned key is counted twice |
| `_events_tracker.record_lookup` / `record_partial_lookup` | Suppressed when `self._speculative`, or KV events are emitted twice |
| `_maybe_register_shared_prefix_junction` | **Skipped** when `self._speculative`. It writes `Request.shared_prefix_boundary` on the scheduler's `Request`, which the manager lock does not protect |
| `group_state.lookup_hit_end_chunk_idx` | Kept. Every `_lookup` resets it first, and k+1 recomputes it |
| Reads `_chunks_being_loaded` | Kept. It is read-only here and mutated only in step-region hooks |

Lock discipline: the manager lock is a plain, non-re-entrant `threading.Lock`.
The handler must not take it, so it must not call `_manager_locked`, and
`_lookup` does not. Every other connector hook holds the same lock (step
region or `_manager_locked`), so the handler cannot interleave with them.
`self._req_status` and `req_status` are mutated only inside those hooks.
Verifying that claim is a PR checklist item (§8).

### 4.7 Tiering manager: `poll_tiers`

```python
def poll_tiers(self) -> None:
    gate = self._processed_jobs_this_step
    self._processed_jobs_this_step = True
    try:
        self._process_finished_jobs(self._polled_tiers)
        for tier in self._polled_tiers:
            tier.serve_external_requests(self._tier_parents[tier])
        if self._resolved_lookup_handler is not None:
            self._run_resolved_lookups()                  # new
    finally:
        self._processed_jobs_this_step = gate

def _run_resolved_lookups(self) -> None:
    assert not self._pending_load_submissions    # empty outside the step region
    try:
        for tier in self._polled_tiers:
            for req_id in tier.take_resolved_lookups():
                try:
                    self._resolved_lookup_handler(req_id)
                except Exception:
                    logger.exception("speculative lookup failed for %s", req_id)
    finally:
        self._flush_pending_promotions()          # submit what the scans queued
```

Notes

- The gate is held shut for the round, so the scan's `lookup()` calls don't
  re-poll tiers off-thread (the existing `poll_tiers` contract).
- `_pending_load_submissions` is empty whenever the poll thread holds the lock:
  it is filled only inside the scheduler's step region, and that step's
  `on_schedule_end` drains it. The flush therefore submits only this round's
  promotions. Batching is preserved: one `PendingPromotion` per (tier,
  request), so each speculative scan is already one batch per tier.
- A handler failure leaves at most a partly-queued request. The `finally`
  flush submits it, and the scheduler's k+1 scan treats it like any promotion
  in flight.

### 4.8 Prerequisite: keep MISS answers across a fetch (suspected existing bug)

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
promotion it matters more: the speculative scan sends the fetch, and clears
the probes, *before* the scheduler's k+1 scan, so that scan meets key h as a
fresh probe at k+1. The same applies to sliding-window scans, which skip over
MISS keys on their way to a window.

Fix: in `request_blocks`, drop only the `True` (and any `None`) probes, and
keep `False`. A kept `False` is conservative: at worst the request recomputes
a block that has become available since, which is today's MISS outcome
anyway.

This is unverified. PR step 1 starts with a failing test that reproduces the
extra round, and only then applies the fix.

### 4.9 Configuration

- P2P tier config key `eager_promotion: bool` (default `false` in the first
  PR). Add it to `config_info_keys`.
- It is effective when the key is on and `tier_poll_interval_s > 0`. The spec
  passes the effective value to the connector, which then registers the
  handler. There is no KV-layout gate: the connector's own scan handles every
  layout.

## 5. Behavior and edge cases

| Case | Behavior |
| --- | --- |
| Scheduler lookup races the poll thread | Impossible. Both run under `manager.lock`, and the second sees the primary slot |
| Inputs change between poll and k+1 (local prefix grew, more answers, eviction) | The k+1 scan decides on fresh inputs. A promotion the speculative scan already started is either used or simply unused (bandwidth only) |
| Request no longer WAITING (scheduled, preempted, finished) | Handler returns early; `_deferred_probes` is popped in `on_request_finished` |
| Request aborted while eager fetch in flight | Existing path: `finish()` aborts loads -> promotion fails -> `complete_write(False)` frees the slot |
| Fetch fails / peer dies | Existing path: `_failed_req_ids` + failed `JobResult` -> slot freed -> scheduler sees MISS -> recompute |
| Primary full | `_initiate_promotion` returns False inside the manager lookup -> MISS for that key, the same outcome k+1 would get |
| `reset_cache` hold (`_promotions_held`) | `manager.lookup` returns MISS, as on the scheduler thread |
| `load_tier_filter` excludes P2P | Enforced inside `manager.lookup`, as on the scheduler thread |
| Speculative scan hits RETRY from another tier (FS, or a new P2P probe) | Keeps scanning, as the scheduler does; that tier's lookup is flushed at the next step |
| Session not send-ready | `request_blocks` already queues `pending_fetch`; no change |
| Eviction between `complete_write` and the scheduler's `prepare_load` | Slot has `ref_cnt=0` and can be evicted (this window exists today too; eager promotion makes it longer). Next lookup: primary MISS -> P2P -> probes were cleared by the fetch -> new LookupMsg under `round_seq+1`. Producer pins again (one pin per round; the previous round's pin was already released at finalize) or answers MISS. Cost: one extra round + re-transfer. Nothing is double-pinned and nothing leaks |
| Partial fetch of a round's supply | Unchanged: the producer releases leftover pins at round finalize |
| PD path (`remote_prefiller`) | Never registers probes -> never reported |
| `tier_poll_interval_s = 0`, flag off, or non-tiering manager | No handler registered -> today's behavior |
| Stats | `prepare_write` off-thread appends to the CPU tier's per-step allocation stats (`allocation_sizes_in_current_batch`). They are reported with the next step. Under the lock, so this is safe |

## 6. Risks

1. **The speculative mode has to stay complete.** Any new side effect added
   to the connector scan later (events, metrics, writes to `Request`) must
   check `self._speculative`. Mitigation: one test runs `_lookup`
   speculatively and asserts no events, no metric deltas, and an unchanged
   `Request`. Put a docstring note on `_lookup` and on the scan helpers.
2. **Connector state touched off the scheduler thread.** This is safe only
   while every connector hook mutates `_req_status` / `req_status` under the
   manager lock. This is true today by the lock discipline in §4.6, and the
   PR must re-verify it hook by hook. Engine-owned `Request` fields are read
   only (`status`, `skip_reading_external_kv`), never written.
3. **Over-promotion.** The speculative scan promotes before the scheduler
   decides. It is bounded by the scan itself (the same keys k+1 would
   promote), and wasted only if the inputs change (§5) or the request leaves
   WAITING in between.
4. **Longer eviction window under CPU-tier pressure** (see §5). Measure
   re-probe rounds first. If they matter, add a short consumer pin released
   when the scheduler adopts the chunk (`prepare_load`) or at request finish.
5. **More work on the poll thread while holding the lock.** One connector
   scan per fully-answered request: the same cost the scheduler pays at k+1
   (which still runs), moved off the step region. Check scheduler lock-wait
   p99 with the existing lock-hold harness at high concurrency.

## 7. Observability

- New counter `vllm:kv_offload_speculative_lookups_total`.
- New counter `vllm:kv_offload_eager_promoted_chunks_total{tier}`: promotions
  initiated under the speculative mode.
- Existing `vllm:kv_offload_lookup_async_delay_seconds` is the headline: it
  should fall by ~1 × T_step.
- Existing `vllm:kv_offload_p2p_lookup_rtt_seconds` should stay unchanged
  (it measures the wire).
- Debug log per speculative scan: req_id, the hit tokens the scan computed,
  and the number of promotions queued.

## 8. Testing

Unit

- `tests/v1/kv_offload/tiering/p2p/test_sessions.py`
    - fetch keeps `False` probes and drops `True` ones (§4.8), and a
      re-scheduled request with a partial hit sends no second LookupMsg.
- `tests/v1/kv_offload/tiering/p2p/test_manager.py`
    - `take_resolved_lookups` reports nothing until every registered probe is
      answered (multi-step registration), then reports the request once.
    - PD requests and the disabled flag never report; `on_request_finished`,
      `_failed_req_ids` and reap drop pending entries.
- `tests/v1/kv_offload/tiering/test_tiering_offloading.py` (fake polled tier)
    - `poll_tiers` calls the handler for reported ids, then flushes: promotions
      are submitted with no `on_schedule_end`.
    - the scheduler's next `lookup()` returns HIT_PENDING, then HIT after
      `poll_tiers` processes the completion.
    - no double promotion when the scheduler looks up first in the same window.
    - a handler exception still flushes and doesn't kill the round.
    - `speculative_lookups()` suppresses lookup metrics.
- Connector tests (`tests/v1/kv_connector/unit/` offloading scheduler tests)
    - **equivalence**: for full-attention, sliding-window, and a
      full-attention + sliding-window hybrid (plus Mamba-aligned if a
      fixture exists), the keys promoted by the speculative scan equal the
      keys the step-k+1 scan promotes with the same answers. Sliding
      window must promote only the trailing window.
    - speculative mode emits no KV events, does not change
      `Request.shared_prefix_boundary`, and leaves `deferred_lookup_start_time`
      untouched.
    - guards: not WAITING, `skip_reading_external_kv`, `load_failed`,
      in-flight `transfer_jobs` -> no lookup.
- `tests/v1/kv_offload/tiering/test_tier_poll_thread.py`
    - real thread: reply -> speculative scan -> promotion submitted without
      any `on_schedule_end`.

PR checklist

- Audit every `OffloadingConnectorScheduler` method that mutates
  `_req_status` / `req_status`, and confirm it runs inside the step region or
  `_manager_locked`.

E2E (llmd pods, Llama-3.2-1B and 3.1-8B, symmetric `remote_kv_source`, plus
one sliding-window / hybrid model that fits a pod)

- A/B on the `lock_api_conc` harness (`tests/v1/kv_offload/tiering/p2p/results/`
  scripts): idle and busy consumer, c ∈ {1,2,4,8,16}. Seed and pull per wave.
  Report `lookup_async_delay / T_step` (expect ~2 -> ~1), TTFT, and
  external prefix cache hits (must not drop).
- Busy consumer is the case that matters: load the *pulling* engine with
  8k-token prefills so T_step is a few hundred ms.
- Promoted vs loaded chunks per pull (over-promotion), and scheduler lock-wait
  p99.
- Accuracy: `run_accuracy_test.sh` matrix with the flag on.

## 9. PR plan

1. **[Bugfix][KV Offload][P2P] Keep MISS probe answers across a fetch.** Test
   first, then the fix. Stands alone.
2. **[KV Offload] Speculative connector lookup driven by the tier-poll
   thread.** Covers `set_resolved_lookup_handler` / `speculative_lookups` on
   the manager, `take_resolved_lookups` tier hook, `poll_tiers` wiring, the
   connector's speculative mode and its equivalence tests. No tier reports
   yet, so this is a no-op.
3. **[KV Offload][P2P] Report fully-answered P2P lookups.** Adds the P2P
   `_deferred_probes` and `is_lookup_resolved`, the `eager_promotion` flag
   (default off), metrics and the e2e A/B.
4. Flip the default to on after the e2e A/B.

Steps 2 and 3 can be a single PR if reviewers prefer, but step 2 is the
generic part that FS/obj could reuse later.

## 10. Follow-ups

- **Probe on arrival.** Send the LookupMsg from `on_new_request` (the session
  is already opened there). Combined with the speculative scan, a pull can
  finish while the request waits in the queue, so the first `schedule()` finds
  HIT. That removes the remaining boundary.
- **Producer "pending HIT".** Answer HIT_PENDING keys right away, park them
  like `pending_cascade_keys`, and fail the fetch if the store fails. The
  producer's `complete_store` wait then overlaps the fetch instead of coming
  before the reply.
- **Generalize to `AsyncLookupManager` tiers** (FS/obj): they can report
  resolved requests through the same hook once they opt into off-thread
  polling.

## Revision history

- rev 1: the P2P tier chose keys (leading HIT run per group by position);
  full-attention-only gate.
- rev 2: the connector's own scan runs speculatively from the tier-poll
  thread, so selection is exact for every KV layout (including
  sliding-window, which promotes only the trailing section) and the layout
  gate is gone.
