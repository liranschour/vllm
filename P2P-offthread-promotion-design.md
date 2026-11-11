# Off-thread promotion for P2P lookups

Status: draft design, base for a PR (rev 3: source choice across secondary tiers)
Branch: `worktree-p2p-offthread-promotion`
Code references are against `01fba62494` (upstream main, 2026-11-11).

## 1. Summary

A symmetric P2P pull (`remote_kv_source`) currently costs at least two engine
steps of the *pulling* engine before the request can be scheduled. The real
work fits in a fraction of one step: the probe round trip is ~2-6 ms and the
NIXL fetch ~10-77 ms. We measure ~497 ms of lookup wait per pull in
production, against ~6 ms in the unit/e2e test, because the wait is quantized
to the pulling engine's step time, and production steps are long.

The step quantum comes from one thing: the lookup answers, and the fetch they
enable, are only acted on by the next `schedule()` call. This design acts on
them as soon as every secondary tier has answered for the request. At that
point the tier-poll thread re-runs the **connector's own lookup** for the
request in a speculative mode. That run promotes (allocates a primary CPU
slot, submits the load) exactly the keys the scheduler would promote at the
next step:

- for any attention type: full, sliding-window, Mamba, EAGLE, hybrid;
- **from the secondary tier the tiering manager would have chosen.** A
  speculative promotion is only made when no other tier that could win the
  choice is still undecided, so speculation never changes the load source.

The result is discarded. The scheduler's own lookup at the next step makes the
load decision as today, and finds the promotion in flight or done through
primary-tier state that already exists.

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
to gate everything except full attention. Since rev 2, it runs the connector's
scan instead.

### 2.3 Which tier a key is loaded from is decided by the tiering manager

For each key, `TieringOffloadingManager.lookup()` (`tiering/manager.py:699`)
queries the primary tier first. On a primary miss, it walks the secondary tiers
in config order, skipping tiers that `load_tier_filter` excludes:

- **RETRY does not stop the walk.** Every tier with an async lookup (P2P
  probe, FS/obj `AsyncLookupManager`) registers its probe in the same pass, so
  the lookups to all secondary tiers run in parallel.
- **The first tier that answers HIT at that moment is the source.** The
  promotion is initiated from it, even if a tier ordered earlier is still
  RETRY.
- MISS everywhere -> MISS; any RETRY and no HIT -> RETRY.

So the source is "first HIT in config order *among the answers present when
the scan runs*". Today the scan runs at the scheduler's step k+1. By then the
answers of every tier that resolves within one forward pass are usually in, so
in practice the config order decides when several tiers hold the key.

Running the scan earlier changes that. A speculative scan fired the moment
P2P's answers land would usually run before a slower, earlier-ordered tier
(FS/obj) has answered, and would promote from P2P where the k+1 scan would have
promoted from FS. That is the same rule applied at a different moment, and it
would systematically override the configured tier order. Rev 3 makes this
impossible (§4.6, §4.8).

### 2.4 Relevant code

| What | Where |
| --- | --- |
| Only lookup call site in the scheduler cycle (WAITING loop) | `v1/core/sched/scheduler.py:1144` `get_num_new_matched_tokens` |
| Deferral: `None` -> `skip_request` | `scheduler.py:1150-1155` |
| Connector lookup entry, preconditions | `offloading/scheduler.py:1364` `get_num_new_matched_tokens` (guards at `:1401-1417`) |
| Connector scan | `offloading/scheduler.py:1147` `_lookup` -> `:904` `_lookup_complete_chunks`, `_lookup_partial_tail_hit` |
| Connector side effects in the scan | `_events_tracker.record_lookup` (`:844`, `:1259`), `record_partial_lookup` (`:1324`), `_maybe_register_shared_prefix_junction` (`:1329`) |
| Connector lock discipline | `offloading/scheduler.py:752` `_acquire_step`, `:770` `_manager_locked` |
| Tiering lookup: primary first, then secondaries in order; first HIT -> promotion | `tiering/manager.py:699` `lookup` (secondary walk at `:752-786`) |
| Promotion: allocate primary slot now, submit at step end | `tiering/manager.py:795` `_initiate_promotion`, `:849` `_flush_pending_promotions` |
| Promotion completion -> primary `complete_write` | `tiering/manager.py:539` `_complete_promotion` |
| Tier-poll round (no promotion flush, by design) | `tiering/manager.py:476` `poll_tiers` |
| Manager lock (plain, non-re-entrant `threading.Lock`) | `tiering/manager.py:299` |
| Primary HIT_PENDING while slot not ready | `cpu/manager.py:214` `lookup` |
| FS/obj async lookups: per-key phase, request index, result queue | `tiering/async_lookup.py:70` `AsyncLookupManager` (`_lookup_state`, `_req_keys`, `drain_results`) |
| P2P consumer lookup (probe, cached answers) | `tiering/p2p/manager.py:423` `lookup` |
| P2P submit_load -> FetchMsg | `tiering/p2p/manager.py:609` `submit_load` |
| Probe cache, reply handling, fetch clears probes | `tiering/p2p/session/client.py:447` `register_lookup`, `:528` `on_lookup_resp`, `:256` `request_blocks` (`probes.clear()` at `:308`) |

## 3. Goals and non-goals

Goals

- Remove the k -> k+1 boundary on symmetric P2P pulls by starting the
  promotion as soon as the request's lookups are answered.
- Promote exactly what the scheduler would promote at k+1, for every KV
  layout the connector supports, without duplicating its scan rules.
- **Never change the load source.** A speculative promotion comes from the
  tier the tiering manager's selection rule picks with every competing
  tier's answer in hand.
- No change to the scheduler, or to the connector's load decision.
- Identical behavior when the tier-poll thread is disabled
  (`tier_poll_interval_s = 0`), the feature flag is off, or the manager is
  not a `TieringOffloadingManager`.

Non-goals (follow-ups, §10)

- Probing on request arrival (`on_new_request`) to remove the remaining
  boundary.
- Producer-side HIT_PENDING / RETRY step waits.
- A new source-selection policy (cost-based, best-of-HITs). The design only
  routes selection through one helper so such a policy plugs in (§4.8).
- Speculation triggered by FS/obj answers alone (no P2P involved). The
  hooks allow it; enabling it is a follow-up.
- PD path (`remote_prefiller`): it already returns an eager HIT and pays only
  the promotion boundary.

## 4. Design

### 4.1 New timeline

```text
schedule(k)    lookup -> RETRY from P2P (and FS/obj, if configured)   ; LookupMsg out, FS batch out
forward k      FS/obj worker resolves its batch; poll thread receives LookupRespMsg
               poll round: request is a candidate (P2P answered)
                 and settled (every allowed tier answered for it)
                 -> connector speculative _lookup(req) -> manager.lookup per key
                 -> source = first HIT in tier order, final -> _initiate_promotion
                 -> flush -> submit_load (FetchMsg / FS read)
                 -> completion -> _complete_promotion (poll thread for P2P)
schedule(k+1)  lookup -> primary HIT                   -> load scheduled
               (or primary HIT_PENDING if the load is still in flight -> HIT at k+2,
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
| `prepare_write` in `_initiate_promotion` | slot exists, `ref_cnt=-1`, not ready | HIT_PENDING (secondaries not asked) |
| load in flight | same | HIT_PENDING -> defer |
| `_complete_promotion` -> `complete_write(True)` | ready, `ref_cnt=0` | HIT -> `prepare_load` takes a ref |
| load failed -> `complete_write(False)` | slot freed | primary MISS -> secondaries (P2P: `_failed_req_ids` -> MISS) -> recompute |

`_request_load_sources` is filled by `_initiate_promotion` too, so
`get_load_source()` keeps attributing the hit to the tier it came from.

All transitions happen under `manager.lock`. `poll_tiers` holds it for the
whole round, and the scheduler holds it for the step region
(`_acquire_step` ... `build_connector_meta`). Whichever path reaches a key
second sees the primary slot and does not promote again. `prepare_write`
also skips keys that are already present.

### 4.3 Components and call flow

```text
P2P tier / FS,obj tiers      TieringOffloadingManager            OffloadingConnectorScheduler
-----------------------      ------------------------            ----------------------------
P2P lookup() -> RETRY:
  remember kv_request_id
  -> req_id
_poll_once (poll thread):
  on_lookup_resp ...
  probes answered
  -> queue req_id
                             poll_tiers() [holds lock]
                               _process_finished_jobs
                               serve_external_requests
                               candidates |= tier.take_resolved_lookups()
                               for req_id in candidates:
                                 if every allowed tier
                                   .lookup_settled(ctx):
                                   handler(req_id)  ──────────────►  _speculative_lookup(req_id)
                                                                       guards (§4.7)
                                                                       with speculative mode:
                                                                         _lookup(req_status)
                                                   ◄── manager.lookup(key) per scanned key
                                 _select_promotion_source(...)
                                   final? -> _initiate_promotion
                                   not final -> RETRY (scheduler decides)
                               _flush_pending_promotions()
                                 -> tier.submit_load
```

The tiering layer never imports the connector. The connector registers a
callback with the manager, and the manager calls it.

### 4.4 Tier hooks (`tiering/base.py`)

Two optional hooks on `SecondaryTierManager`:

```python
def take_resolved_lookups(self) -> list[str]:
    """req_ids whose deferred lookups this tier fully answered since the
    last call, and which this tier wants speculated on.

    Called from the tiering manager's tier-poll round, under the manager
    lock, only for tiers with tolerates_off_thread_polling. Default: none.
    """
    return []

def lookup_settled(self, req_context: ReqContext) -> bool:
    """Whether this tier has answered every lookup the request issued to it,
    so a re-run of those lookups would not get RETRY from this tier.

    May be called from the tier-poll thread, under the manager lock. A tier
    without async lookups is always settled. Default: True.
    """
    return True
```

`take_resolved_lookups` nominates a request; `lookup_settled` says whether
the request's lookup in each tier is complete. Only P2P nominates in this
design. Every tier with async lookups answers `lookup_settled`.

### 4.5 Tier implementations

P2P (`P2PSecondaryTierManager`, consumer side):

- `_deferred_probes: dict[str, str]`, `kv_request_id -> req_id`. Filled in
  `lookup()` when the probe path returns RETRY, and only when eager promotion
  is enabled. Popped when reported, in `on_request_finished`, when the id
  enters `_failed_req_ids`, and on session reap.
- At the end of `_poll_once`, move each entry whose client state has no
  `None` probe and no `unsent` key to the report list
  (`take_resolved_lookups`). Expose `ClientRole.is_lookup_resolved(kv_request_id)`.
- `lookup_settled(ctx)`: true when the request has no P2P source, or when
  `is_lookup_resolved` is true for its `kv_request_id`.

FS / obj (through `AsyncLookupManager`, `tiering/async_lookup.py`):

```python
def lookup_settled(self, req_context: ReqContext) -> bool:
    self.drain_results()                    # apply worker results that are in
    for key in self._req_keys.get(req_context.req_id, ()):
        state = self._lookup_state.get(key)
        if state is not None and state.phase is not LookupPhase.RESOLVED:
            return False                    # PENDING (not flushed) or IN_FLIGHT
    return True
```

Draining from the poll thread is allowed: `AsyncLookupManager` state belongs
to whichever thread holds the manager lock (the module docstring already
covers the poll thread reaching a tier through the parent). A key still
PENDING (registered but not flushed) counts as unsettled. It is flushed at
the next `on_schedule_end`, as today.

Tiers with synchronous lookups (KVCR, example) keep the default `True`.

### 4.6 Speculation candidates and the settled check (`poll_tiers`)

The manager keeps `_speculation_candidates: dict[str, ReqContext]`. Each poll
round:

1. Add the ids each polled tier reports in `take_resolved_lookups()`.
2. Drop candidates that are finished or gone from `_req_state`.
3. For each remaining candidate, run the speculative scan only if
   `lookup_settled(ctx)` is true for **every** secondary tier that
   `ctx.load_tier_filter` allows. Otherwise keep the candidate for the next
   round.
4. Run each candidate at most once, then drop it. If the scheduler's own
   lookup gets there first, the candidate is harmless: the scan sees primary
   HIT / HIT_PENDING, or the handler's WAITING guard rejects it.

A candidate whose other tier never settles (for example a stuck FS worker) is
dropped when its request leaves WAITING, or when it has waited
`speculation_max_wait_s` (default 1 s, well above any healthy lookup). From
then on the normal path owns it, exactly as today.

### 4.7 Connector: speculative lookup

Registration, in `OffloadingConnectorScheduler.__init__`:

```python
if eager_promotion_enabled:          # §4.10
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
        with self.manager.speculative_lookups():     # metrics off + finality rule on
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
| `manager.lookup()` -> secondary HIT -> `_initiate_promotion` | **Wanted**, but only when the source is final (§4.8) |
| `manager.lookup()` -> new probes (P2P, partial-tail key) / FS/obj async lookups | Kept. Equivalent to what k+1 would register. Flushed at the next `on_schedule_end`, as today |
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

### 4.8 Tiering manager: source selection and finality

Pull source selection out of `lookup()` into one helper that both the
scheduler path and the speculative path use:

```python
def _select_promotion_source(
    self, key: OffloadKey, req_context: ReqContext, exclude_tier_idx: int | None
) -> tuple[int | None, bool]:
    """Walk allowed secondary tiers; return (source_tier_idx, any_retry).

    Policy (unchanged): the first tier in config order that answers HIT.
    In speculative mode, the choice must also be final: a HIT from tier i
    is only taken if no tier the policy could prefer over i (today: any
    lower index) answered RETRY. Otherwise return (None, True), so the
    caller reports RETRY and the scheduler makes the choice at k+1.
    """
    any_retry = False
    for i, tier in enumerate(self.secondary_tiers):
        if i == exclude_tier_idx:
            continue
        if not req_context.load_tier_filter.allows(tier.medium, tier.locality):
            continue
        result = tier.lookup(key, req_context)
        ...  # metrics (skipped when speculative)
        if result is LookupResult.HIT:
            if self._speculative and any_retry:
                return None, True            # an earlier-ordered tier is undecided
            return i, any_retry
        if result is LookupResult.RETRY:
            any_retry = True
    return None, any_retry
```

`lookup()` keeps its contract:

- source found: `_initiate_promotion(source, ...)` -> HIT_PENDING, or MISS if
  primary is full;
- no source and `any_retry`: RETRY;
- otherwise MISS.

Notes on finality

- With the settled check of §4.6, the finality rule should never fire. Every
  allowed tier has answered before the scan runs. It is a safety net for
  answers that change between the check and the scan, and for tiers whose
  `lookup_settled` is coarser than their per-key `lookup`. A metric counts
  how often it fires (§7).
- **Source equivalence.** With every allowed tier settled, the speculative
  scan sees the same set of answers the k+1 scan would see if no new answers
  arrived in between. It applies the same order, so it picks the same
  source. If an answer *changes* afterwards (a tier evicts the key), k+1
  would pick differently; that is the same timing exposure as today
  between `schedule(k+1)` and the load.
- **Future policies.** A policy that ranks HITs (cost, locality, bandwidth)
  replaces the body of `_select_promotion_source`. Its finality condition
  is "no RETRY tier could outrank the current best". For a full ranking that
  means no RETRY among allowed tiers. The settled check already provides it.

### 4.9 Tiering manager: `poll_tiers`

```python
def poll_tiers(self) -> None:
    gate = self._processed_jobs_this_step
    self._processed_jobs_this_step = True
    try:
        self._process_finished_jobs(self._polled_tiers)
        for tier in self._polled_tiers:
            tier.serve_external_requests(self._tier_parents[tier])
        if self._resolved_lookup_handler is not None:
            self._run_speculative_lookups()               # new
    finally:
        self._processed_jobs_this_step = gate

def _run_speculative_lookups(self) -> None:
    assert not self._pending_load_submissions    # empty outside the step region
    for tier in self._polled_tiers:
        for req_id in tier.take_resolved_lookups():
            st = self._req_state.get(req_id)
            if st is not None:
                self._speculation_candidates.setdefault(req_id, _Candidate(st.req_context))
    try:
        for req_id, cand in list(self._speculation_candidates.items()):
            st = self._req_state.get(req_id)
            if st is None or st.is_finished or cand.expired():
                del self._speculation_candidates[req_id]
                continue
            if not self._all_tiers_settled(cand.req_context):
                continue                          # retry next round
            del self._speculation_candidates[req_id]
            try:
                self._resolved_lookup_handler(req_id)
            except Exception:
                logger.exception("speculative lookup failed for %s", req_id)
    finally:
        self._flush_pending_promotions()          # submit what the scans queued

def _all_tiers_settled(self, ctx: ReqContext) -> bool:
    return all(
        tier.lookup_settled(ctx)
        for tier in self.secondary_tiers
        if ctx.load_tier_filter.allows(tier.medium, tier.locality)
    )
```

Notes

- The gate is held shut for the round, so the scan's `lookup()` calls don't
  re-poll tiers off-thread (the existing `poll_tiers` contract).
- `_pending_load_submissions` is empty whenever the poll thread holds the lock:
  it is filled only inside the scheduler's step region, and that step's
  `on_schedule_end` drains it. The flush therefore submits only this round's
  promotions. Batching is preserved: one `PendingPromotion` per (tier,
  request).
- Promotions from a non-polled tier (FS chosen as source): `submit_load` runs
  on the poll thread from the flush. Its completion is collected at the next
  step (FS is not polled off-thread), so the gain for that key is the
  submit-time advance only. That is still not worse than today.
- A handler failure leaves at most a partly-queued request. The `finally`
  flush submits it, and the scheduler's k+1 scan treats it like any promotion
  in flight.

### 4.10 Prerequisite: keep MISS answers across a fetch (suspected existing bug)

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
MISS keys on their way to a window. It also affects multi-tier selection: an
unanswered P2P probe makes P2P unsettled again.

Fix: in `request_blocks`, drop only the `True` (and any `None`) probes, and
keep `False`. A kept `False` is conservative: at worst the request recomputes
a block that has become available since, which is today's MISS outcome
anyway.

This is unverified. PR step 1 starts with a failing test that reproduces the
extra round, and only then applies the fix.

### 4.11 Configuration

- P2P tier config key `eager_promotion: bool` (default `false` in the first
  PR). Add it to `config_info_keys`.
- Tiering extra-config key `speculation_max_wait_s` (default `1.0`).
- Eager promotion is effective when the key is on and
  `tier_poll_interval_s > 0`. The spec passes the effective value to the
  connector, which then registers the handler. There is no KV-layout gate,
  because the connector's own scan handles every layout. There is no
  tier-order gate either, because the settled check and the finality rule
  keep the configured order.

## 5. Behavior and edge cases

| Case | Behavior |
| --- | --- |
| Scheduler lookup races the poll thread | Impossible. Both run under `manager.lock`, and the second sees the primary slot |
| P2P only (no other secondary tier) | Settled as soon as P2P answers -> full gain |
| FS ordered **before** P2P, FS answers MISS | Scan runs once FS has answered; source P2P, as k+1 would choose |
| FS before P2P, FS answers HIT | Scan runs once FS has answered; source FS (lower index), as k+1 would choose |
| FS before P2P, FS slow (not settled) | Candidate waits for FS. Gain shrinks to "promote when the slower tier answers"; source unchanged; never worse than today. Dropped after `speculation_max_wait_s` |
| FS ordered **after** P2P | Today the walk stops at P2P's HIT, so P2P is chosen. The settled check still waits for FS. Option: only require tiers that could outrank the HIT (§6, risk 4) |
| Answer changes after the settled check (eviction, failed load mark_miss) | The finality rule / MISS outcome handles it; the k+1 scan decides on fresh inputs |
| Inputs change between poll and k+1 (local prefix grew, more answers, eviction) | The k+1 scan decides on fresh inputs. A promotion already started is used, or is simply unused (bandwidth only) |
| Request no longer WAITING (scheduled, preempted, finished) | Handler returns early; candidate dropped; `_deferred_probes` popped in `on_request_finished` |
| Request aborted while eager load in flight | Existing path: P2P `finish()` aborts loads -> promotion fails -> `complete_write(False)` frees the slot |
| Load fails / peer dies | Existing path: failed `JobResult` -> slot freed -> scheduler sees MISS (P2P `_failed_req_ids`, FS `mark_miss`) -> recompute |
| Primary full | `_initiate_promotion` returns False inside the manager lookup -> MISS for that key, the same outcome k+1 would get |
| `reset_cache` hold (`_promotions_held`) | `manager.lookup` returns MISS, as on the scheduler thread |
| `load_tier_filter` excludes a tier | Excluded tiers are neither asked nor waited on, as on the scheduler thread |
| Session not send-ready | `request_blocks` already queues `pending_fetch`; no change |
| Eviction between `complete_write` and the scheduler's `prepare_load` | Slot has `ref_cnt=0` and can be evicted (this window exists today too; eager promotion makes it longer). Next lookup: primary MISS -> P2P -> probes were cleared by the fetch -> new LookupMsg under `round_seq+1`. Producer pins again (one pin per round; the previous round's pin was already released at finalize) or answers MISS. Cost: one extra round + re-transfer. Nothing is double-pinned and nothing leaks |
| Partial fetch of a round's supply | Unchanged: the producer releases leftover pins at round finalize |
| PD path (`remote_prefiller`) | Never registers probes -> never nominated |
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
   manager lock. This is true today by the lock discipline in §4.7, and the
   PR must re-verify it hook by hook. Engine-owned `Request` fields are read
   only (`status`, `skip_reading_external_kv`), never written.
3. **A slow earlier-ordered tier limits the gain.** When FS/obj is ordered
   before P2P and its lookup is slow, speculation waits for it, so the
   remaining gain depends on FS lookup latency, not P2P. The held-back
   counter (§7) makes this visible. Reordering tiers is the operator's call.
   Speculation does not override it.
4. **The settled check is stricter than needed.** It waits for every allowed
   tier, including tiers ordered *after* the eventual source, which can never
   win under the order policy. A refinement waits only for tiers that could
   outrank a HIT already seen. This is deferred: it needs per-key knowledge
   of which tier answered HIT, and it only helps configs with a slow tier
   placed after P2P.
5. **Over-promotion.** The speculative scan promotes before the scheduler
   decides. It is bounded by the scan itself (the same keys and source k+1
   would choose), and wasted only if the inputs change (§5) or the request
   leaves WAITING in between.
6. **Longer eviction window under CPU-tier pressure** (see §5). Measure
   re-probe rounds first. If they matter, add a short consumer pin released
   when the scheduler adopts the chunk (`prepare_load`) or at request finish.
7. **More work on the poll thread while holding the lock.** One connector
   scan per settled candidate, plus one `lookup_settled` sweep per candidate
   per round. That is the same cost the scheduler pays at k+1 (which still
   runs), moved off the step region. Check scheduler lock-wait p99 with the
   existing lock-hold harness at high concurrency.

## 7. Observability

- New counter `vllm:kv_offload_speculative_lookups_total`.
- New counter `vllm:kv_offload_eager_promoted_chunks_total{tier}`: promotions
  initiated under the speculative mode, labeled by **source** tier.
- New counter `vllm:kv_offload_speculation_held_back_total{tier}`: candidate
  rounds skipped because that tier was not settled. A high rate for an
  earlier-ordered tier means its lookup latency limits the gain.
- New counter `vllm:kv_offload_speculation_expired_total`: candidates dropped
  after `speculation_max_wait_s`.
- New counter `vllm:kv_offload_speculation_finality_retries_total`: HITs not
  taken because an earlier-ordered tier was RETRY (should stay ~0 with the
  settled check).
- Existing `vllm:kv_offload_lookup_async_delay_seconds` is the headline: it
  should fall by ~1 × T_step.
- Existing `vllm:kv_offload_p2p_lookup_rtt_seconds` should stay unchanged
  (it measures the wire).
- Debug log per speculative scan: req_id, the hit tokens the scan computed,
  the promotions queued per source tier.

## 8. Testing

Unit

- `tests/v1/kv_offload/tiering/p2p/test_sessions.py`
    - fetch keeps `False` probes and drops `True` ones (§4.10), and a
      re-scheduled request with a partial hit sends no second LookupMsg.
- `tests/v1/kv_offload/tiering/p2p/test_manager.py`
    - `take_resolved_lookups` reports nothing until every registered probe is
      answered (multi-step registration), then reports the request once.
    - `lookup_settled` follows the same condition; true for requests with no
      P2P source.
    - PD requests and the disabled flag never report; `on_request_finished`,
      `_failed_req_ids` and reap drop pending entries.
- `tests/v1/kv_offload/tiering/test_async_lookup.py`
    - `lookup_settled` is false while any of the request's keys is PENDING or
      IN_FLIGHT, true after the worker result is drained, and true for
      requests that never looked up in this tier.
- `tests/v1/kv_offload/tiering/test_tiering_offloading.py` (fake tiers)
    - `poll_tiers` runs the handler for nominated, settled ids, then flushes:
      promotions are submitted with no `on_schedule_end`.
    - **source preservation**, with tiers `[FS-like async, P2P-like polled]`:
        - FS MISS, P2P HIT -> source P2P;
        - FS HIT, P2P HIT -> source FS, same as the scheduler path;
        - FS not settled -> no scan; it runs on the round after FS settles,
          with the same source as the scheduler path;
        - FS never settles -> candidate expires, scheduler path unchanged.
    - finality rule: in speculative mode, a HIT after a RETRY tier returns
      RETRY and initiates no promotion; outside speculative mode, behavior
      is unchanged (first HIT wins).
    - `load_tier_filter` excludes a tier -> it is not waited on.
    - the scheduler's next `lookup()` returns HIT_PENDING, then HIT after
      `poll_tiers` processes the completion.
    - no double promotion when the scheduler looks up first in the same window.
    - a handler exception still flushes and doesn't kill the round.
    - `speculative_lookups()` suppresses lookup metrics.
- Connector tests (`tests/v1/kv_connector/unit/` offloading scheduler tests)
    - **equivalence**: for full-attention, sliding-window, and a
      full-attention + sliding-window hybrid (plus Mamba-aligned if a
      fixture exists), the keys *and source tiers* promoted by the
      speculative scan equal those the step-k+1 scan promotes with the
      same answers. Sliding window must promote only the trailing window.
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
- Confirm `_select_promotion_source` leaves `lookup()`'s non-speculative
  behavior byte-for-byte equivalent (existing tiering tests unchanged).

E2E (llmd pods, Llama-3.2-1B and 3.1-8B, symmetric `remote_kv_source`, plus
one sliding-window / hybrid model that fits a pod)

- A/B on the `lock_api_conc` harness (`tests/v1/kv_offload/tiering/p2p/results/`
  scripts): idle and busy consumer, c ∈ {1,2,4,8,16}. Seed and pull per wave.
  Report `lookup_async_delay / T_step` (expect ~2 -> ~1), TTFT, and
  external prefix cache hits (must not drop).
- Busy consumer is the case that matters: load the *pulling* engine with
  8k-token prefills so T_step is a few hundred ms.
- Two tier configs: P2P only, and `[fs, p2p]` with the key present in both
  tiers. For the latter, report the per-source promotion counts and confirm
  they match the flag-off run.
- Promoted vs loaded chunks per pull (over-promotion), and scheduler lock-wait
  p99.
- Accuracy: `run_accuracy_test.sh` matrix with the flag on.

## 9. PR plan

1. **[Bugfix][KV Offload][P2P] Keep MISS probe answers across a fetch.** Test
   first, then the fix. Stands alone.
2. **[KV Offload] Refactor: route secondary-tier source selection through
   `_select_promotion_source`.** No behavior change. Existing tests prove it.
3. **[KV Offload] Speculative connector lookup driven by the tier-poll
   thread.** Covers `set_resolved_lookup_handler` / `speculative_lookups`
   (including the finality rule), the `take_resolved_lookups` /
   `lookup_settled` tier hooks with the `AsyncLookupManager` implementation,
   candidates in `poll_tiers`, the connector's speculative mode, and the
   equivalence and source-preservation tests. No tier nominates yet, so
   this is a no-op.
4. **[KV Offload][P2P] Nominate fully-answered P2P lookups.** Adds the P2P
   `_deferred_probes`, `is_lookup_resolved`, `lookup_settled`, the
   `eager_promotion` flag (default off), metrics and the e2e A/B.
5. Flip the default to on after the e2e A/B.

## 10. Follow-ups

- **Probe on arrival.** Send the LookupMsg from `on_new_request` (the session
  is already opened there). Combined with the speculative scan, a pull can
  finish while the request waits in the queue, so the first `schedule()` finds
  HIT. That removes the remaining boundary.
- **Producer "pending HIT".** Answer HIT_PENDING keys right away, park them
  like `pending_cascade_keys`, and fail the fetch if the store fails. The
  producer's `complete_store` wait then overlaps the fetch instead of coming
  before the reply.
- **FS/obj nomination.** Let `AsyncLookupManager` tiers nominate requests
  through `take_resolved_lookups` once they opt into off-thread polling, so
  FS-only pulls get the same speedup.
- **Narrower settled check** (risk 4): wait only for tiers that could outrank
  a HIT already seen.
- **Ranked source selection** (cost / locality) as a policy behind
  `_select_promotion_source`.

## Revision history

- rev 1: the P2P tier chose keys (leading HIT run per group by position);
  full-attention-only gate.
- rev 2: the connector's own scan runs speculatively from the tier-poll
  thread, so selection is exact for every KV layout (including
  sliding-window, which promotes only the trailing section) and the layout
  gate is gone.
- rev 3: source choice across secondary tiers. Speculation runs only once
  every allowed tier has answered for the request (`lookup_settled`), and a
  speculative HIT is taken only when no earlier-ordered tier is still RETRY,
  so the configured tier order is never overridden. Selection moves into
  `_select_promotion_source`; adds candidates, expiry, metrics, tests and a
  refactor PR.
