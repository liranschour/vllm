# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Proactive fetch driven by the `p2p.fetch` kv_hints action.

A real TieringOffloadingManager over a real CPU primary tier, with a scripted
peer tier standing in for the remote source. The hint is delivered the way the
engine delivers it: as `ReqContext.kv_hints` of an unrelated carrier request.
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock

import pytest

from tests.v1.kv_offload.tiering.test_tiering_offloading import _mock_mmap_region
from vllm.distributed.kv_events import BlockStored
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.events import (
    OffloadingEventsTracker,
)
from vllm.v1.kv_hints import KvHintAction, KvHintsEnvelope
from vllm.v1.kv_offload.base import (
    LookupResult,
    OffloadingKVEventsConfig,
    OffloadKey,
    ReqContext,
    ScheduleEndContext,
    make_offload_key,
)
from vllm.v1.kv_offload.tiering.base import JobResult, TransferJob
from vllm.v1.kv_offload.tiering.example.manager import ExampleSecondaryTierManager
from vllm.v1.kv_offload.tiering.manager import (
    _FETCH_KEYS_PER_STEP,
    _MAX_ACTIVE_FETCHES,
    CPUPrimaryTierOffloadingManager,
    TieringOffloadingManager,
)
from vllm.v1.kv_offload.tiering.proactive_fetch import MAX_TRANSFER_ID_LEN

SOURCE_HOST = "10.0.0.9"
SOURCE_PORT = 5710


def _key(i: int) -> OffloadKey:
    return make_offload_key(f"block-hash-{i:04d}".encode(), 0)


K1, K2, K3 = _key(1), _key(2), _key(3)


class _PeerTier(ExampleSecondaryTierManager):
    """A secondary tier that, like P2P, probes a named peer before fetching.

    `lookup` answers RETRY (probe in flight) until the test resolves the key as
    held or missing on the peer. Loads complete on the next poll, unless
    `hold_jobs` keeps them in flight.
    """

    supports_proactive_fetch = True

    def __init__(self, primary_kv_view: memoryview):
        super().__init__(
            offloading_spec=MagicMock(),
            primary_kv_view=primary_kv_view,
            tier_type="peer",
        )
        self.held: set[OffloadKey] = set()
        self.absent: set[OffloadKey] = set()
        self.lookups: list[OffloadKey] = []
        self.hold_jobs = False
        self.in_flight: list[int] = []
        self.finished_requests: list[str] = []
        self.new_requests: list[ReqContext] = []
        self.source_ready: bool | None = True

    def is_fetch_source_ready(self, req_context):
        return self.source_ready

    def on_new_request(self, req_context):
        self.new_requests.append(req_context)
        return super().on_new_request(req_context)

    def lookup(self, key, req_context):
        self.lookups.append(key)
        if key in self.held:
            return LookupResult.HIT
        if key in self.absent:
            return LookupResult.MISS
        return LookupResult.RETRY

    def submit_load(self, job_metadata: TransferJob) -> None:
        if self.hold_jobs:
            self.in_flight.append(job_metadata.job_id)
        else:
            self.completed_jobs.append(JobResult(job_metadata.job_id, success=True))

    def drain_jobs(self) -> None:
        self.completed_jobs += [JobResult(j, success=False) for j in self.in_flight]
        self.in_flight = []

    def on_request_finished(self, req_context):
        self.finished_requests.append(req_context.req_id)


def _make(
    num_chunks: int = 16, fetching: bool = True
) -> tuple[TieringOffloadingManager, CPUPrimaryTierOffloadingManager, _PeerTier]:
    region = _mock_mmap_region(num_chunks)
    primary = CPUPrimaryTierOffloadingManager(
        num_chunks=num_chunks, mmap_region=region, enable_events=True
    )
    tier = _PeerTier(region.create_kv_memoryview())
    if not fetching:
        tier.supports_proactive_fetch = False  # type: ignore[misc]
    mgr = TieringOffloadingManager(primary_tier=primary, secondary_tiers=[tier])
    return mgr, primary, tier


def _step(mgr: TieringOffloadingManager) -> None:
    mgr.on_schedule_end(ScheduleEndContext(new_req_ids=[], preempted_req_ids=()))


def _fetch(
    keys,
    action_id: str = "m0",
    version: str = "1.0",
    host: str = SOURCE_HOST,
    port: int = SOURCE_PORT,
) -> KvHintAction:
    return KvHintAction(
        action_id=action_id,
        action_type="p2p.fetch",
        action_version=version,
        payload={
            "source": {"host": host, "port": port},
            "blocks": [k.hex() for k in keys],
        },
    )


def _carry(
    mgr: TieringOffloadingManager,
    *actions: KvHintAction,
    message_id: str = "msg-1",
    req_id: str = "carrier",
) -> ReqContext:
    """Deliver `actions` on a carrier request and finish it right away."""
    ctx = ReqContext(
        req_id=req_id,
        kv_hints=KvHintsEnvelope(
            protocol_version="0.1", message_id=message_id, actions=list(actions)
        ),
    )
    mgr.on_new_request(ctx)
    mgr.on_request_finished(ctx)
    return ctx


def test_hint_fetches_blocks_into_cpu_and_outlives_its_carrier():
    mgr, primary, tier = _make()
    _carry(mgr, _fetch([K1]))

    # The carrier already finished; the fetch it carried keeps running.
    assert tier.finished_requests == ["carrier"]
    assert mgr.poll_proactive_fetch("msg-1:m0")["state"] == "running"
    (fetch_ctx,) = (c for c in tier.new_requests if c.req_id != "carrier")
    assert fetch_ctx.kv_transfer_params == {
        "remote_kv_source": {
            "remote_host": SOURCE_HOST,
            "remote_port": SOURCE_PORT,
            "kv_request_id": "msg-1:m0",
        }
    }

    _step(mgr)  # probe registered; the peer has not answered
    assert mgr.poll_proactive_fetch("msg-1:m0")["state"] == "running"
    assert mgr.has_pending_work()

    tier.held.add(K1)
    _step(mgr)  # probe HIT -> promotion submitted
    assert primary.lookup(K1, fetch_ctx) is LookupResult.HIT_PENDING
    _step(mgr)  # load finished -> committed in CPU

    status = mgr.poll_proactive_fetch("msg-1:m0")
    assert status == {
        "state": "completed",
        "blocks_total": 1,
        "blocks_done": 1,
        "blocks_missing": 0,
        "blocks_failed": 0,
        "blocks_no_capacity": 0,
        "error": None,
    }
    assert primary.lookup(K1, fetch_ctx) is LookupResult.HIT
    assert "p2p-fetch:msg-1:m0" in tier.finished_requests
    assert not mgr.has_pending_work()


def test_fetched_block_serves_a_later_request_without_a_peer_round_trip():
    mgr, _, tier = _make()
    tier.held.add(K1)
    _carry(mgr, _fetch([K1]))
    _step(mgr)
    _step(mgr)
    assert mgr.poll_proactive_fetch("msg-1:m0")["state"] == "completed"

    tier.lookups.clear()
    real = ReqContext(req_id="real")
    mgr.on_new_request(real)
    assert mgr.lookup(K1, real) is LookupResult.HIT
    assert tier.lookups == []


def test_fetched_block_emits_a_cpu_block_stored_event(monkeypatch):
    """The orchestrator detects completion from this event; it is the whole
    status contract of the hint. Byte-mode hashes let it rebuild the
    OffloadKey it asked for."""
    monkeypatch.setenv("VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES", "0")
    mgr, _, tier = _make()
    tracker = OffloadingEventsTracker(
        OffloadingKVEventsConfig(
            enable_kv_cache_events=True, self_describing_kv_events=True
        )
    )
    tier.held.add(K2)
    _carry(mgr, _fetch([K2]))
    _step(mgr)
    _step(mgr)

    stored = [
        e for e in tracker.take_events(mgr.take_events()) if isinstance(e, BlockStored)
    ]
    assert len(stored) == 1
    (event,) = stored
    assert event.medium == "CPU"
    assert event.group_idx == 0
    assert event.block_hashes == [b"block-hash-0002"]


def test_probes_wait_while_the_source_is_connecting():
    """Before the handshake completes a P2P lookup answers MISS; a fetch
    must not read that as the source lacking every block."""
    mgr, _, tier = _make()
    tier.held.add(K1)
    tier.source_ready = None
    _carry(mgr, _fetch([K1]))
    _step(mgr)
    _step(mgr)
    assert tier.lookups == []
    assert mgr.poll_proactive_fetch("msg-1:m0")["state"] == "running"

    tier.source_ready = True
    _step(mgr)
    _step(mgr)
    assert mgr.poll_proactive_fetch("msg-1:m0")["blocks_done"] == 1


def test_unreachable_source_fails_the_fetch():
    mgr, _, tier = _make()
    tier.source_ready = False
    _carry(mgr, _fetch([K1, K2]))
    _step(mgr)
    status = mgr.poll_proactive_fetch("msg-1:m0")
    assert (status["state"], status["error"]) == ("failed", "source_unreachable")
    assert (status["blocks_failed"], status["blocks_missing"]) == (2, 0)
    assert tier.lookups == []


def test_block_already_in_cpu_completes_without_a_fetch():
    mgr, primary, tier = _make()
    seed = ReqContext(req_id="seed")
    mgr.on_new_request(seed)
    primary.prepare_store([K1], seed)
    primary.complete_store([K1], seed)

    _carry(mgr, _fetch([K1]))
    _step(mgr)
    assert mgr.poll_proactive_fetch("msg-1:m0")["blocks_done"] == 1
    assert tier.lookups == []


def test_source_miss_is_missing_and_still_completes():
    mgr, _, tier = _make()
    tier.absent.add(K2)
    _carry(mgr, _fetch([K2]))
    _step(mgr)
    status = mgr.poll_proactive_fetch("msg-1:m0")
    assert (status["state"], status["blocks_missing"]) == ("completed", 1)


def test_promotion_without_room_is_a_failure_not_a_miss():
    mgr, primary, tier = _make(num_chunks=1)
    tier.held.add(K1)
    _carry(mgr, _fetch([K1]))
    # Another request takes the only chunk (write-pending, not evictable)
    # between submission and the first probe.
    other = ReqContext(req_id="other")
    mgr.on_new_request(other)
    assert primary.prepare_store([K3], other) is not None

    _step(mgr)
    status = mgr.poll_proactive_fetch("msg-1:m0")
    assert status["state"] == "failed"
    assert status["error"] == "insufficient_capacity"
    assert (status["blocks_no_capacity"], status["blocks_missing"]) == (1, 0)


def test_unresolved_probe_at_the_deadline_fails():
    mgr, _, _ = _make()
    _carry(mgr, _fetch([K1]))
    _step(mgr)
    mgr._fetches["msg-1:m0"].deadline = time.monotonic() - 1.0
    _step(mgr)
    status = mgr.poll_proactive_fetch("msg-1:m0")
    assert (status["state"], status["error"]) == ("failed", "timeout")
    assert (status["blocks_failed"], status["blocks_missing"]) == (1, 0)


def test_reset_mid_fetch_fails_the_fetch_and_leaves_no_jobs():
    mgr, _, tier = _make()
    tier.held.add(K1)
    tier.hold_jobs = True
    _carry(mgr, _fetch([K1]))
    _step(mgr)
    _step(mgr)
    assert mgr._jobs

    mgr.reset_cache()

    status = mgr.poll_proactive_fetch("msg-1:m0")
    assert (status["state"], status["error"]) == ("failed", "reset")
    assert not mgr._jobs
    assert "p2p-fetch:msg-1:m0" not in mgr._req_state
    assert not mgr.has_pending_work()


def test_submission_beyond_cpu_capacity_is_rejected():
    mgr, _, _ = _make(num_chunks=1)
    _carry(mgr, _fetch([K1, K2]))
    assert mgr.poll_proactive_fetch("msg-1:m0") is None


def test_concurrent_fetches_are_bounded():
    mgr, _, _ = _make(num_chunks=1024)
    _carry(
        mgr,
        *(_fetch([_key(i)], action_id=f"a{i}") for i in range(_MAX_ACTIVE_FETCHES)),
    )
    _carry(mgr, _fetch([_key(999)], action_id="extra"), message_id="msg-2")
    assert len(mgr._fetches) == _MAX_ACTIVE_FETCHES
    assert mgr.poll_proactive_fetch("msg-2:extra") is None


def test_a_repeated_hint_does_not_restart_the_fetch():
    mgr, _, tier = _make()
    _carry(mgr, _fetch([K1]))
    first = mgr._fetches["msg-1:m0"]
    _carry(mgr, _fetch([K2]), req_id="carrier-2")
    assert mgr._fetches["msg-1:m0"] is first
    assert first.total == 1


def test_duplicate_keys_reconcile_against_the_total():
    mgr, _, tier = _make()
    tier.absent.update({K1, K2})
    _carry(mgr, _fetch([K1, K1, K2]))
    _step(mgr)
    status = mgr.poll_proactive_fetch("msg-1:m0")
    assert status["blocks_total"] == 2
    assert status["blocks_missing"] == 2


def test_keys_advanced_per_step_are_bounded():
    num_keys = _FETCH_KEYS_PER_STEP + 10
    mgr, _, tier = _make(num_chunks=num_keys * 2)
    _carry(mgr, _fetch([_key(i) for i in range(num_keys)]))
    _step(mgr)
    assert len(tier.lookups) == _FETCH_KEYS_PER_STEP
    _step(mgr)
    assert len(tier.lookups) == 2 * _FETCH_KEYS_PER_STEP


@pytest.mark.parametrize(
    "bad",
    [
        _fetch([K1], version="2.0"),
        _fetch([K1], host=""),
        _fetch([K1], port=0),
        _fetch([]),
        KvHintAction("m0", "p2p.fetch", "1.0", {"source": {"host": "h"}}),
        KvHintAction("m0", "p2p.fetch", "1.0", {"source": "h", "blocks": []}),
        KvHintAction(
            "m0",
            "p2p.fetch",
            "1.0",
            {"source": {"host": "h", "port": 1}, "blocks": ["not-hex"]},
        ),
        KvHintAction(
            "m0",
            "p2p.fetch",
            "1.0",
            {"source": {"host": "h", "port": 1}, "blocks": ["0011"]},
        ),
        _fetch([K1], action_id="x" * MAX_TRANSFER_ID_LEN),
        _fetch([K1], action_id="spaces are not allowed"),
    ],
)
def test_invalid_action_is_skipped_without_blocking_the_others(bad):
    mgr, _, _ = _make()
    _carry(mgr, bad, _fetch([K2], action_id="good"))
    assert list(mgr._fetches) == ["msg-1:good"]


def test_other_action_types_are_ignored():
    mgr, _, _ = _make()
    other = KvHintAction("k0", "kv.fetch", "1.0", {"block_hashes": [1]})
    _carry(mgr, other)
    assert not mgr._fetches


def test_tier_without_fetch_support_ignores_the_hint():
    mgr, _, tier = _make(fetching=False)
    _carry(mgr, _fetch([K1]))
    assert not mgr._fetches
    assert [c.req_id for c in tier.new_requests] == ["carrier"]
