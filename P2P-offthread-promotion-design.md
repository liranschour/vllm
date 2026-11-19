# Converged-poll trigger for request-level promotions

Status: draft design (rev 5: rebased on RFC #66782 v2)
Branch: `worktree-p2p-offthread-promotion`

Builds on:

- RFC #66782 (@orozery), "Request-level KV promotions: decide at promotion
  time, pin until load", v2.
- #67069 (draft): `SchedulerView.lookup_request()`, bound with
  `OffloadingManager.bind_scheduler_view()`. Part of RFC item 2.
- #60128 (approved): lazy `TransferJob.materialize()` / `cancel()`,
  `ParentContext`.

Existing-code references are against upstream `01fba62494` (2026-11-11). Names
from the RFC and the two PRs refer to their current text, not merged code.

## 1. Summary

A symmetric P2P pull (`remote_kv_source`) costs at least two engine steps of
the *pulling* engine before its load can start, although the real work fits in
a fraction of one step: the probe round trip is ~2-6 ms and the NIXL fetch
~10-77 ms. We measure ~497 ms of lookup wait per pull in production, against
~6 ms in tests, because the wait is quantized to the pulling engine's step.

RFC #66782 replaces per-key promotion with a request-level plan. That fixes
budgets, over-commit, wasted promotions and source choice, but **it does not
change when the plan is made**. The poll still runs only from
`get_num_new_matched_tokens`, once per step, so the two-step penalty remains.

This design adds one thing on top of the RFC: **when the last async lookup
answer for a PENDING request arrives, run that request's RFC poll on the
tier-poll thread**, instead of waiting for the next `schedule()`. The poll is
the RFC's own: a converged `lookup_request`, then a `RequestJob` and its views.
It runs under the manager lock, like the RFC's fill. The scheduler's next
`poll_load` then finds the plan already made, and often the transfer already
done.

Expected effect: one step boundary removed from every symmetric P2P pull
(`lookup_async_delay` ~2 × T_step -> ~1 × T_step). A separate, independent
fix to the P2P probe cache (§6) is required for the gain on sliding-window and
partial-tail models.

## 2. Background

### 2.1 Where the time goes today

Consumer side, one symmetric pull, in steps of the pulling engine:

```text
schedule(k)    lookup -> RETRY (register probe)        ; on_schedule_end: LookupMsg out
forward k      tier-poll thread receives LookupRespMsg, caches it      <- idle
schedule(k+1)  lookup -> HIT -> promotion -> HIT_PENDING               ; on_schedule_end: FetchMsg out
forward k+1    NIXL fetch
schedule(k+2)  lookup -> primary HIT -> load scheduled                 <- lookup_async_delay ends
```

The producer is not step-gated in the warm case. Since #58168 it serves
LookupMsgs on its tier-poll thread (`poll_tiers` -> `serve_external_requests`,
1 ms interval). It only waits for its own step when a key is HIT_PENDING (its
own store still in flight) or must be promoted from a lower tier. RFC item 1's
serve claims, which let a serve join an in-flight write or promotion, address
that side.

So both wasted boundaries are on the consumer, and both come from one rule:
nothing acts on a lookup answer until the next `schedule()`.

### 2.2 The same pull under RFC #66782 v2

```text
schedule(k)    poll_load: P2P probe unresolved -> walk not converged -> PENDING
forward k      LookupRespMsg lands on the tier-poll thread                    <- idle
schedule(k+1)  poll_load: converged -> RequestJob, views submitted
               P2P view materializes inside submit_load (fill) -> FetchMsg
forward k+1    NIXL fetch
schedule(k+2)  poll_load -> READY -> load scheduled
```

Same two boundaries. The RFC's open question about the tier-poll thread is about
lock contention with the fill, not about this latency.

### 2.3 What the RFC already provides that earlier revisions built themselves

| Rev 1-4 of this doc | RFC #66782 / #67069 / #60128 | Rev 5 |
| --- | --- | --- |
| Promotion as a side effect of `manager.lookup()`; speculative scan relies on it | `lookup()` no longer promotes; request-level `RequestJob` | dropped |
| Connector handler registered with the manager (`set_resolved_lookup_handler`) | `SchedulerView` + `bind_scheduler_view()` (#67069) | dropped, reuse |
| Speculative `_lookup` with events/metrics suppressed | `lookup_request(quiet=...)`; metrics once at poll | dropped, reuse |
| `lookup_settled` hook + finality rule, to keep the source tier | Walk not converged (RETRY, unresolved async probe) -> PENDING; key->tier from the converged walk | dropped |
| `_select_promotion_source` refactor | Fill records key->tier | dropped |
| Hybrid over-promotion (keys between the sliding-window cut and the full-attention boundary) accepted as existing waste | Decide at promotion time; commit only the converged set | fixed by the RFC |
| Eviction between promotion and `prepare_load`; follow-up consumer pin | Pins held until `prepare_load` | fixed by the RFC |
| Follow-up: producer answers "pending HIT" | Item 1: serve paths claim `HIT_PENDING`, join in-flight transfers | covered by the RFC |
| Off-thread trigger when lookup answers arrive | not in the RFC | **kept (§4)** |
| P2P probe-cache bugs A/B | not in the RFC | **kept (§6), independent** |
| `speculation_max_wait_s` | open question: "v2 has no wait deadline" | **offered to the RFC (§4.7)** |

## 3. Goals and non-goals

Goals

- Remove the k -> k+1 boundary from symmetric P2P pulls by running the RFC's
  poll as soon as the request's lookup can converge.
- No new planning logic: the off-thread poll is the RFC poll, with the same
  `lookup_request`, `should_admit`, `RequestJob` and fill.
- Identical behavior when the trigger is disabled, the tier-poll thread is off
  (`tier_poll_interval_s = 0`), or the manager is not a
  `TieringOffloadingManager`.

Non-goals

- Anything the RFC decides: key selection, source tier, admission, pins,
  serve claims, reset ordering.
- FS/obj triggers (`AsyncLookupManager` tiers): same hook, follow-up (§10).
- PD path (`remote_prefiller`): it never probes, so there is nothing to wait
  for. Its single promotion boundary is the RFC's poll-to-submit step.
- Probing on request arrival (§10).

## 4. Design

### 4.1 Idea

The RFC's poll is a pure function of live state, run under the manager lock:
the connector's walk via `lookup_request`, `should_admit`, then `RequestJob`
and view submission. Nothing in it requires the scheduler thread. The RFC
already runs `lookup_request` off-thread, in the fill on a tier worker.

So when the tier-poll thread observes that the async lookups of a PENDING
request have all resolved, it runs the same poll for that request right away.
If the walk converges, the plan exists before the next `schedule()`, the P2P
view materializes in `submit_load` on the poll thread, and the fetch overlaps
forward k.

### 4.2 New timeline

```text
schedule(k)    poll_load: P2P probe unresolved -> PENDING        ; LookupMsg out
forward k      tier-poll round: LookupRespMsg -> request's probes resolved
                 -> converged poll for the request (RFC poll, off-thread)
                 -> lookup_request converges -> should_admit -> RequestJob
                 -> P2P view: submit_load -> materialize (fill) -> FetchMsg
                 -> NIXL fetch -> TransferDone -> landing pins (poll thread)
schedule(k+1)  poll_load -> READY -> load scheduled
               (or PENDING if the fetch is still in flight -> READY at k+2,
                same as without the trigger, never worse)
```

With async scheduling (batch queue 2), `schedule(k+1)` runs right after step k
is dispatched, before the reply is back. The saving still holds: without the
trigger READY lands at `schedule(k+3)`, with it at `schedule(k+2)`.

### 4.3 Tier hook (`tiering/base.py`)

```python
def take_converged_requests(self) -> list[str]:
    """req_ids whose async lookups in this tier all resolved since the
    last call.

    A wake-up hint for the tier-poll thread, not a correctness signal: the
    manager re-runs the request's poll, and that decides convergence. Called
    under the manager lock, only for tiers with
    tolerates_off_thread_polling. Default: none.
    """
    return []
```

Because the poll itself decides convergence (by `lookup_request` returning
None or keys), the hook can be approximate. A false wake-up costs one poll that
returns PENDING. A missed wake-up costs what it costs today: the next
scheduler poll handles the request.

P2P implementation (consumer side, `P2PSecondaryTierManager`):

- `_deferred_probes: dict[str, str]`, `kv_request_id -> req_id`. Filled in
  `lookup()` when the probe path returns RETRY, only when the trigger is
  enabled. Popped when reported, in `on_request_finished`, when the id enters
  `_failed_req_ids`, and on session reap.
- At the end of `_poll_once`, report each entry whose client state has no
  unanswered probe (`None`) and no `unsent` key. Expose that test as
  `ClientRole.is_lookup_resolved(kv_request_id)`.

### 4.4 Manager: the off-thread poll (`poll_tiers`)

```python
def poll_tiers(self) -> None:
    gate = self._processed_jobs_this_step
    self._processed_jobs_this_step = True
    try:
        self._process_finished_jobs(self._polled_tiers)       # also lands RequestJob views
        for tier in self._polled_tiers:
            tier.serve_external_requests(self._tier_parents[tier])
        if self._converged_poll_enabled:
            for tier in self._polled_tiers:
                for req_id in tier.take_converged_requests():
                    self._poll_request_off_thread(req_id)    # new
    finally:
        self._processed_jobs_this_step = gate

def _poll_request_off_thread(self, req_id: str) -> None:
    st = self._req_state.get(req_id)
    if st is None or st.is_finished or st.request_job is not None:
        return                                   # gone, or already planned
    if not self._scheduler_view.is_waiting(req_id):   # new SchedulerView query
        return
    self._poll(st.req_context)                   # the RFC poll, unchanged
```

`self._poll` is the body of the RFC's `poll_load` for one request:
`lookup_request`, then classify, then `should_admit`, then `RequestJob` and
`submit_load` of each view. `poll_load` itself becomes "if a `RequestJob`
exists, report its state; else `_poll`".

Requirements this places on the RFC implementation (to confirm on #66782):

1. **The poll is idempotent per request.** A poll on a request with a live
   `RequestJob` reports PENDING/READY and does not re-plan. The RFC implies it
   ("register waiters, create one RequestJob"); the trigger makes it a hard
   requirement, because the scheduler's poll can follow the off-thread one in
   the same step window.
2. **The poll can run on any thread that holds the manager lock.** This is
   already true of the fill in v2. If the RFC adopts @javieratomic's point 9
   (move the fill to the scheduler thread at `on_schedule_end`), the trigger
   needs this property kept for the poll, even if the fill moves.
3. **The poll's view submission may run off the scheduler thread.** Today
   `_flush_pending_promotions` runs at `on_schedule_end` only, and
   `poll_tiers` leaves it out by design. Under the RFC a request's views are
   one batch already, so submitting them from the poll thread fragments
   nothing.

### 4.5 Connector side

Two small additions to the connector's `SchedulerView`, beyond #67069:

- `is_waiting(req_id) -> bool`: the request is still in the WAITING queue,
  has no in-flight `transfer_jobs`, no `load_failed`, and
  `skip_reading_external_kv` is unset. These are the same guards
  `get_num_new_matched_tokens` applies before it looks up.
- `lookup_request` must use the poll inputs the connector cached at its last
  `get_num_new_matched_tokens`: computed tokens and the token caps. The RFC
  already caches these for the fill, and the off-thread poll reuses them.

Thread safety is the RFC's: the connector's state is touched only under the
manager lock. That includes reads of `Request.block_hashes`, which the RFC lists
as a known limitation for resumable requests with hash lookahead and
streaming-session updates. The trigger adds a second off-thread caller, not a
new class of access. If the RFC chooses to bypass the fill for those requests,
the trigger must skip them the same way (`is_waiting` returns False).

### 4.6 What the scheduler sees

- `poll_load` at k+1 finds the `RequestJob`:
    - READY if the P2P fetch landed and the landing converted waiters into
      pins, which `_process_finished_jobs` does on the poll thread;
    - PENDING if it is still in flight.
- `NO_HIT` and the ready-only fallback happen exactly as in the RFC, just
  earlier.
- Lookup metrics are recorded once, at the poll that converged, whichever
  thread ran it. `lookup_async_delay` (connector, deferral to resolution) is
  unchanged as a metric and is the headline result.

### 4.7 Wait deadline (offered to the RFC)

The RFC's open question "v2 has no wait deadline; a stalled tier can leave a
request PENDING indefinitely" applies directly. A request woken by P2P may
still be waiting on an FS async probe. Proposed, as part of the RFC rather
than this trigger: `pending_max_wait_s` (tiering extra config, default 1 s)
from the request's first PENDING poll. Past it, the poll answers with the
ready-only fallback instead of PENDING. The trigger doesn't depend on it.

## 5. Multi-tier source choice and sliding window

These were the subject of rev 3 and rev 4. Under the RFC they are handled by the
plan itself:

- **Source tier.** The poll only plans after the connector's walk converges,
  i.e. no allowed tier is RETRY, and the fill records key->tier from a converged
  quiet walk. An early poll on the tier-poll thread therefore cannot pick a
  different tier than a later one with the same answers. If an FS probe is
  still unresolved, the off-thread poll returns PENDING and FS's own
  resolution (or the next scheduler poll) completes it. Rev 3's
  `lookup_settled` and finality rule are no longer needed.
- **Sliding window, Mamba, EAGLE, hybrid.** `lookup_request` is the connector's
  own walk, so the key set is exact for every layout, and "decide at promotion
  time" commits only the converged set. That removes rev 4's hybrid
  over-promotion.
- **What remains.** Sliding window still depends on the P2P probe cache keeping
  MISS answers (§6). The walk from the end crosses every trailing MISS key. If
  the fetch has dropped their answers, the next walk re-probes them and does
  not converge, so the request goes back to PENDING for another round.

## 6. Prerequisite: keep unconsumed probe answers across a fetch (two suspected bugs)

This part is unchanged in substance from rev 4, and independent of the RFC.

`ClientRole.request_blocks` (`p2p/session/client.py:256`) sends the FetchMsg
that closes the round, advances `round_seq`, and clears every probe of the
request (`:308`). Clearing cached `True` is right: once the producer serves the
fetch it releases its pins, so a stale `True` could point at an evicted block.
Clearing the rest is not:

- **Bug A: cached MISS answers are dropped.** A partial hit re-probes its first
  MISS key at the next lookup and pays an extra round. For sliding window,
  every MISS key the walk crosses from the end is re-probed.
- **Bug B: registered-but-unsent probes are orphaned.** A key registered in
  the same pass that sends the fetch (the partial-tail boundary key, for
  example) is still in `unsent`. It goes out under `round_seq+1`, but its
  answer arrives for a key no longer in `probes`, and `on_lookup_resp` drops
  it silently. The next lookup registers it again.

Under the RFC both show up as a walk that doesn't converge, so an extra PENDING
round. Under the trigger, the fetch goes out off-thread before the scheduler's
next walk, which then meets the dropped keys as fresh probes.

Fix, at round close:

| Probe state at fetch | Action | Why |
| --- | --- | --- |
| `True` | drop | the producer releases the pin after serving the fetch; re-probe under the next round |
| `False` | **keep** | a MISS stays a MISS; at worst the request recomputes a block that has become available since |
| `None`, still in `unsent` | **keep** | belongs to the next round, sent with `round_seq+1` |
| `None`, already sent under round r | drop | round r is closed on the producer; re-probe |

`can_send_terminal_fetch` (`client.py:325`) and `_maybe_prune`
(`client.py:179`) treat any entry in `probes` as live lookup state and must
ignore kept `False` entries. `finish()` still clears everything. Both bugs are
unverified: the PR starts with one failing test per bug.

## 7. Behavior and edge cases

| Case | Behavior |
| --- | --- |
| Off-thread poll and scheduler poll for the same request | Serialized by the manager lock; the second finds the `RequestJob` and reports its state (requirement 1) |
| Woken but walk still not converged (another tier RETRY, new partial-tail probe) | Off-thread poll returns PENDING; nothing is planned. The next wake-up or scheduler poll retries |
| Request left WAITING (scheduled, preempted, finished) | `is_waiting` is False; nothing runs |
| GPU prefix changed since the cached poll inputs | Same as the RFC fill (@javieratomic, point 2). The load fails if the committed set doesn't cover the scheduler's range, and the request prefills |
| Admission DEFER / REJECT | As in the RFC, earlier. REJECT takes the ready-only fallback at the next poll |
| Fetch fails / peer dies | RFC/#60128 paths: failed view -> `RequestJob` cancelled -> next poll re-plans or falls back |
| Reset in progress (`_resetting`) or promotions held | The RFC's poll answers as during a reset; the trigger adds no path around it |
| PD request | Never registers probes -> never woken |
| Trigger off / `tier_poll_interval_s = 0` / non-tiering manager | No wake-ups -> RFC behavior |

## 8. Risks

1. **Depends on the RFC landing in a compatible shape.** Requirements 1-3 in
   §4.4 must hold. Mitigation: raise them on #66782 before any code, and land
   the trigger only after RFC item 2.
2. **Lock hold on the poll thread.** One connector walk per woken request,
   under the manager lock. This is the cost the scheduler's poll pays anyway,
   moved off the step region, and the scheduler's later poll becomes a state
   check. Measure scheduler lock-wait p99 at high concurrency with the lock-hold
   harness. The RFC's open question on fill cost covers the same concern.
3. **Off-thread connector reads.** Same exposure as the RFC fill, including
   the `block_hashes` limitation (§4.5). The trigger must follow whatever the
   RFC decides for resumable and streaming requests.
4. **The sliding-window gain needs §6.** Without it the walk keeps
   re-probing trailing MISS keys and does not converge on the first wake-up.

## 9. Observability

- `vllm:kv_offload_offthread_polls_total{outcome}` with outcome one of
  `planned`, `pending`, `no_hit`, `skipped` (not waiting / already planned).
- Existing `vllm:kv_offload_lookup_async_delay_seconds`: should fall by ~1 ×
  T_step.
- Existing `vllm:kv_offload_p2p_lookup_rtt_seconds`: unchanged (wire).
- Debug log per off-thread poll: req_id, outcome, keys per tier.

## 10. Testing

Unit

- `tests/v1/kv_offload/tiering/p2p/test_sessions.py` (§6, the standalone PR):
    - bug A: a fetch keeps `False`, drops `True`; a re-scheduled partial hit
      sends no second LookupMsg.
    - bug B: a key registered but unsent at fetch time is sent under the next
      round and its answer is applied.
    - a key sent under the closed round is dropped and re-probed.
    - kept `False` entries don't block the terminal empty FetchMsg or pruning.
- `tests/v1/kv_offload/tiering/p2p/test_manager.py`:
    - `take_converged_requests` reports a request once, after all its probes
      are answered (multi-step registration included); PD requests and the
      disabled trigger never report; finish, failure and reap drop entries.
- `tests/v1/kv_offload/tiering/test_tiering_offloading.py` (fake tiers + fake
  `SchedulerView`):
    - a woken request is planned on `poll_tiers` with no `on_schedule_end`;
      the next `poll_load` returns READY once the fake view lands, PENDING
      before.
    - idempotency: scheduler poll after an off-thread plan does not re-plan;
      off-thread poll after a scheduler plan is skipped.
    - not converged (another tier RETRY) -> no plan; a later wake-up plans.
    - not waiting -> no poll.
    - the planned key set and key->tier are identical whether the plan was
      made off-thread or by `poll_load`, for full-attention, sliding-window and
      hybrid fixtures (reusing #67069's `lookup_request` tests).
- `tests/v1/kv_offload/tiering/test_tier_poll_thread.py`:
    - real thread: LookupRespMsg -> plan -> view submitted without a scheduler
      step.

E2E (llmd pods, Llama-3.2-1B and 3.1-8B, symmetric `remote_kv_source`, plus
one sliding-window or hybrid model that fits a pod), on the RFC branch:

- A/B, trigger off vs on, on the `lock_api_conc` harness: idle and busy
  pulling engine, c ∈ {1,2,4,8,16}, seed and pull per wave. Report
  `lookup_async_delay / T_step` (expect ~2 -> ~1), TTFT, external prefix
  cache hits (must not drop), and scheduler lock-wait p99.
- Busy pulling engine is the case that matters (8k-token prefills, so T_step
  is a few hundred ms).
- `[fs, p2p]` tier config: per-tier load counts must match trigger-off.
- Accuracy: `run_accuracy_test.sh` matrix with the trigger on.

## 11. Plan

1. **[Bugfix][KV Offload][P2P] Keep unconsumed probe answers across a
   fetch** (§6). Standalone, now: one failing test per bug, then the fix. It
   doesn't depend on the RFC and helps both today's path and the RFC's.
2. **Comment on #66782** (no code):
    - the measured penalty and the §2.2 timeline showing it survives v2;
    - the trigger and its three requirements (§4.4);
    - `pending_max_wait_s` for the wait-deadline open question;
    - bugs A/B, so item 1's P2P serve migration accounts for them.
3. **After RFC item 2 lands: [KV Offload][P2P] Plan converged requests on the
   tier-poll thread.** Covers `take_converged_requests` (base + P2P),
   `SchedulerView.is_waiting`, the off-thread poll in `poll_tiers`, a
   `converged_poll` flag (default off), metrics and tests.
4. E2E A/B on the RFC branch, then flip the default.

## 12. Follow-ups

- **FS/obj wake-ups.** `AsyncLookupManager` tiers implement
  `take_converged_requests` from their per-key phase and `_req_keys`, so an FS
  answer can also trigger the plan. This needs FS/obj to opt into off-thread
  polling.
- **Probe on arrival.** The RFC's `TieringPolicyContext.prefetch(req_context)`
  is the natural home: issue the P2P probe from `on_new_request`. With the
  trigger, the plan can be made and the fetch finished while the request is
  still queued, so the first `schedule()` sees READY. That removes the
  remaining boundary.
- **Probe sliding-window ranges from the end in batches**, so a long prompt
  doesn't probe keys below the window it will load.

## Revision history

- rev 1: the P2P tier chose keys (leading HIT run per group by position);
  full-attention-only gate.
- rev 2: the connector's own scan runs speculatively from the tier-poll
  thread; exact for every KV layout.
- rev 3: source choice across secondary tiers (`lookup_settled`, finality
  rule, `_select_promotion_source`).
- rev 4: sliding-window walkthrough; probe-cache prerequisite corrected (keep
  MISS and unsent probes) and made required.
- rev 5: rebased on RFC #66782 v2, #67069 and #60128. The speculative scan,
  handler, finality rule, source-selection refactor and consumer pin are
  dropped, because the RFC's request plan, `lookup_request(quiet)`, converged
  walk and pins-until-load cover them. What remains is the off-thread poll
  trigger (with three requirements on the RFC), the standalone probe-cache
  fix, and a wait-deadline proposal. Corrects rev 2-4's claim that the walk
  doesn't touch `Request.block_hashes`: it reads them, so the RFC's
  limitation applies.
