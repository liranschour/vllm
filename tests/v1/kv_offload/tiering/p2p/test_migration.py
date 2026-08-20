# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for proactive P2P KV-cache migration.

Covers three layers without any real NIXL/RDMA transport:
  1. The control-RPC envelope codec (`handle_migration_rpc`).
  2. The migration driver in `TieringOffloadingManager`, wired to a stub
     secondary tier and a stub CPU primary tier that together simulate the
     probe -> promote -> commit lifecycle.
  3. The `on_rpc` -> scheduler -> manager dispatch and 501-gating.
"""

from __future__ import annotations

import time

import msgspec
import numpy as np

from vllm.distributed.kv_transfer.kv_connector.v1.offloading.migration import (
    MAX_TRANSFER_ID_LEN,
    handle_migration_rpc,
)
from vllm.v1.kv_offload.base import (
    LookupResult,
    OffloadKey,
    PrepareStoreOutput,
    ReqContext,
    RequestOffloadingContext,
    ScheduleEndContext,
)
from vllm.v1.kv_offload.cpu.common import CPULoadStoreSpec
from vllm.v1.kv_offload.tiering.base import JobResult
from vllm.v1.kv_offload.tiering.manager import (
    _MAX_ACTIVE_MIGRATIONS,
    _MIGRATION_KEYS_PER_STEP,
    TieringOffloadingManager,
)

K1 = OffloadKey(b"block-hash-0001\x00\x00\x00\x00")
K2 = OffloadKey(b"block-hash-0002\x00\x00\x00\x00")


def _key(i: int) -> OffloadKey:
    """A distinct OffloadKey in the same shape as K1/K2 (hash + group idx)."""
    return OffloadKey(f"block-hash-{i:04d}".encode() + b"\x00\x00\x00\x00")


# --------------------------------------------------------------------------
# Codec
# --------------------------------------------------------------------------


class _StubManager:
    """Records calls and returns canned results for codec tests."""

    def __init__(self):
        self.submitted: list[tuple] = []
        self.submit_result: tuple[bool, int, str | None] = (True, 0, None)
        self.poll_result: dict | None = None
        self.cancel_result = False

    def submit_migration(self, transfer_id, host, port, keys):
        self.submitted.append((transfer_id, host, port, list(keys)))
        accepted, _, error = self.submit_result
        return (accepted, len(keys), error)

    def poll_migration(self, transfer_id):
        return self.poll_result

    def cancel_migration(self, transfer_id):
        return self.cancel_result


def _decode(payload: bytes) -> dict:
    return msgspec.msgpack.decode(payload)


def test_codec_migrate_ok():
    mgr = _StubManager()
    req = msgspec.msgpack.encode(
        {
            "v": 1,
            "op": "migrate",
            "transfer_id": "t1",
            "source": {"host": "10.0.0.5", "port": 5710},
            "blocks": [bytes(K1), bytes(K2)],
        }
    )
    resp = _decode(handle_migration_rpc(mgr, req))
    assert resp == {"transfer_id": "t1", "accepted": True, "num_blocks": 2}
    assert mgr.submitted == [("t1", "10.0.0.5", 5710, [bytes(K1), bytes(K2)])]


def test_codec_migrate_rejected_propagates_error():
    mgr = _StubManager()
    mgr.submit_result = (False, 0, "duplicate_transfer_id")
    req = msgspec.msgpack.encode(
        {
            "op": "migrate",
            "transfer_id": "t1",
            "source": {"host": "h", "port": 1},
            "blocks": [bytes(K1)],
        }
    )
    resp = _decode(handle_migration_rpc(mgr, req))
    assert resp == {
        "transfer_id": "t1",
        "accepted": False,
        "error": "duplicate_transfer_id",
    }


def test_codec_poll_unknown_and_known():
    mgr = _StubManager()
    unknown = _decode(
        handle_migration_rpc(
            mgr, msgspec.msgpack.encode({"op": "poll", "transfer_id": "x"})
        )
    )
    assert unknown == {"transfer_id": "x", "state": "unknown"}

    mgr.poll_result = {
        "state": "completed",
        "blocks_total": 3,
        "blocks_done": 2,
        "blocks_missing": 1,
        "blocks_failed": 0,
        "error": None,
    }
    known = _decode(
        handle_migration_rpc(
            mgr, msgspec.msgpack.encode({"op": "poll", "transfer_id": "x"})
        )
    )
    assert known["transfer_id"] == "x"
    assert known["state"] == "completed"
    assert known["blocks_done"] == 2
    assert known["blocks_missing"] == 1


def test_codec_cancel():
    mgr = _StubManager()
    mgr.cancel_result = True
    resp = _decode(
        handle_migration_rpc(
            mgr, msgspec.msgpack.encode({"op": "cancel", "transfer_id": "t1"})
        )
    )
    assert resp == {"transfer_id": "t1", "cancelled": True}


def test_codec_malformed_and_bad_ops_do_not_raise():
    mgr = _StubManager()
    # not msgpack
    assert "error" in _decode(handle_migration_rpc(mgr, b"\xc1not-msgpack"))
    # not an object
    assert "error" in _decode(
        handle_migration_rpc(mgr, msgspec.msgpack.encode([1, 2, 3]))
    )
    # unknown op
    assert "error" in _decode(
        handle_migration_rpc(mgr, msgspec.msgpack.encode({"op": "frobnicate"}))
    )
    # migrate missing transfer_id
    assert "error" in _decode(
        handle_migration_rpc(
            mgr, msgspec.msgpack.encode({"op": "migrate", "source": {}, "blocks": []})
        )
    )
    # migrate bad source
    bad_src = _decode(
        handle_migration_rpc(
            mgr,
            msgspec.msgpack.encode(
                {"op": "migrate", "transfer_id": "t", "source": {}, "blocks": [b"x"]}
            ),
        )
    )
    assert bad_src == {"transfer_id": "t", "accepted": False, "error": "bad_source"}
    # migrate no blocks
    no_blocks = _decode(
        handle_migration_rpc(
            mgr,
            msgspec.msgpack.encode(
                {
                    "op": "migrate",
                    "transfer_id": "t",
                    "source": {"host": "h", "port": 1},
                    "blocks": [],
                }
            ),
        )
    )
    assert no_blocks["error"] == "no_blocks"
    assert mgr.submitted == []  # nothing reached the manager


# --------------------------------------------------------------------------
# Driver (real TieringOffloadingManager + stub tiers)
# --------------------------------------------------------------------------


class _FakePrimaryTier:
    """Duck-typed CPU primary tier: a dict of key -> "pending" | "ready"."""

    def __init__(self, num_blocks: int = 16):
        self._num_blocks = num_blocks
        self._state: dict[OffloadKey, str] = {}
        self._next_slot = 0
        self._buf = np.zeros((num_blocks, 8), dtype=np.uint8)
        # When True, prepare_write refuses like the real CPU tier does when it
        # cannot evict enough blocks to make room.
        self.fail_writes = False

    def _get_num_free_blocks(self) -> int:
        return self._num_blocks - self._next_slot

    @property
    def _num_evictable_cache_blocks(self) -> int:
        return sum(1 for state in self._state.values() if state == "ready")

    def get_kv_memoryview(self) -> memoryview:
        return memoryview(self._buf)

    def lookup(self, key, req_context) -> LookupResult:
        s = self._state.get(key)
        if s == "ready":
            return LookupResult.HIT
        if s == "pending":
            return LookupResult.HIT_PENDING
        return LookupResult.MISS

    def prepare_write(self, keys, req_context):
        if self.fail_writes:
            return None
        block_ids: list[int] = []
        for key in keys:
            self._state.setdefault(key, "pending")
            block_ids.append(self._next_slot)
            self._next_slot += 1
        return PrepareStoreOutput(
            keys_to_store=list(keys),
            store_spec=CPULoadStoreSpec(block_ids),
            evicted_keys=[],
        )

    # alias used by TieringOffloadingManager for secondary transfers
    prepare_read = prepare_write

    def complete_write(self, keys, req_context, success):
        for key in keys:
            if success:
                self._state[key] = "ready"
            else:
                self._state.pop(key, None)

    def complete_read(self, keys, req_context):
        pass

    def touch(self, keys, req_context):
        pass

    def on_request_finished(self, req_context):
        pass

    def take_events(self):
        return ()

    def get_stats(self):
        return None

    def reset_cache(self):
        self._state.clear()

    def shutdown(self):
        pass


class _FakeP2PTier:
    """Stub secondary tier that simulates the symmetric-P2P probe + fetch.

    - `lookup` returns RETRY until `resolve_hit`/`resolve_miss` flips a key.
    - `submit_load` records the job; `get_finished_jobs` reports it done once,
      one call later, simulating the async transfer completing.
    """

    tier_type = "p2p"
    medium = None
    supports_migration = True

    def __init__(self):
        self.locality = None
        self._hits: set[OffloadKey] = set()
        self._misses: set[OffloadKey] = set()
        self._pending_jobs: list[int] = []
        self._finished: list[JobResult] = []
        self.finished_requests: list[str] = []
        self.lookups: list[OffloadKey] = []
        # When True, submitted loads stay in flight instead of completing on
        # the next poll — the "fetch still running" state a cancel races with.
        self.hold_jobs = False

    def resolve_hit(self, key):
        self._hits.add(key)

    def resolve_miss(self, key):
        self._misses.add(key)

    # SecondaryTierManager surface used by the manager -------------------

    def on_new_request(self, req_context) -> RequestOffloadingContext:
        return RequestOffloadingContext()

    def lookup(self, key, req_context) -> LookupResult:
        self.lookups.append(key)
        if key in self._hits:
            return LookupResult.HIT
        if key in self._misses:
            return LookupResult.MISS
        return LookupResult.RETRY

    def submit_load(self, job_metadata):
        self._pending_jobs.append(job_metadata.job_id)

    def submit_store(self, job_metadata):
        self._finished.append(JobResult(job_id=job_metadata.job_id, success=True))

    def get_finished_jobs(self):
        # Promote last step's submitted loads to "finished" now.
        out = self._finished
        if not self.hold_jobs:
            out += [JobResult(job_id=j, success=True) for j in self._pending_jobs]
            self._pending_jobs = []
        self._finished = []
        return out

    def has_pending_work(self) -> bool:
        return True

    def serve_external_requests(self, parent):
        pass

    def on_schedule_end(self, context):
        pass

    def on_request_finished(self, req_context):
        self.finished_requests.append(req_context.req_id)
        # Mirror the real tier: a load aborted here can never be acked, so it
        # is reported as failed instead of being silently dropped.
        self._finished += [
            JobResult(job_id=j, success=False) for j in self._pending_jobs
        ]
        self._pending_jobs = []

    def drain_jobs(self):
        pass

    def shutdown(self):
        pass


def _make_manager():
    primary = _FakePrimaryTier()
    p2p = _FakeP2PTier()
    mgr = TieringOffloadingManager(primary_tier=primary, secondary_tiers=[p2p])
    return mgr, primary, p2p


def _step(mgr):
    mgr.on_schedule_end(ScheduleEndContext(new_req_ids=(), preempted_req_ids=()))


def test_migration_probe_promote_commit_completes():
    mgr, primary, p2p = _make_manager()

    accepted, num, error = mgr.submit_migration("m1", "10.0.0.9", 5710, [K1])
    assert (accepted, num, error) == (True, 1, None)
    assert mgr.poll_migration("m1")["state"] == "running"

    # Step 1: probe registered, peer not resolved yet -> still running.
    _step(mgr)
    assert mgr.poll_migration("m1")["state"] == "running"

    # Peer confirms it holds the block; step drives probe HIT -> promotion.
    p2p.resolve_hit(K1)
    _step(mgr)
    status = mgr.poll_migration("m1")
    assert status["state"] == "running"  # promotion in flight
    assert primary.lookup(K1, None) is LookupResult.HIT_PENDING

    # Next step: the load reports finished -> committed into CPU -> completed.
    _step(mgr)
    status = mgr.poll_migration("m1")
    assert status["state"] == "completed"
    assert status["blocks_done"] == 1
    assert status["blocks_missing"] == 0
    assert status["blocks_failed"] == 0
    assert primary.lookup(K1, None) is LookupResult.HIT
    assert "p2p-migration:m1" in p2p.finished_requests


def test_migration_already_resident_short_circuits():
    mgr, primary, p2p = _make_manager()
    primary._state[K1] = "ready"  # block already in CPU
    mgr.submit_migration("m2", "h", 5710, [K1])
    _step(mgr)
    status = mgr.poll_migration("m2")
    assert status["state"] == "completed"
    assert status["blocks_done"] == 1
    assert p2p._pending_jobs == []  # no fetch was issued


def test_migration_source_miss_marks_missing():
    mgr, primary, p2p = _make_manager()
    p2p.resolve_miss(K2)
    mgr.submit_migration("m3", "h", 5710, [K2])
    _step(mgr)
    status = mgr.poll_migration("m3")
    assert status["state"] == "completed"
    assert status["blocks_missing"] == 1
    assert status["blocks_done"] == 0


def test_migration_duplicate_and_empty_rejected():
    mgr, _, _ = _make_manager()
    assert mgr.submit_migration("m4", "h", 5710, [K1]) == (True, 1, None)
    assert mgr.submit_migration("m4", "h", 5710, [K2]) == (
        False,
        0,
        "duplicate_transfer_id",
    )
    assert mgr.submit_migration("m5", "h", 5710, []) == (False, 0, "no_blocks")


def test_migration_cancel_finalizes():
    mgr, _, p2p = _make_manager()
    mgr.submit_migration("m6", "h", 5710, [K1])
    _step(mgr)  # probe pending
    assert mgr.cancel_migration("m6") is True
    assert mgr.poll_migration("m6")["state"] == "cancelled"
    assert "p2p-migration:m6" in p2p.finished_requests
    # cancelling again is a no-op
    assert mgr.cancel_migration("m6") is False


def test_poll_unknown_returns_none():
    mgr, _, _ = _make_manager()
    assert mgr.poll_migration("nope") is None


# --------------------------------------------------------------------------
# on_rpc wiring / 501-gating
# --------------------------------------------------------------------------


def test_scheduler_on_rpc_gates_when_no_secondary_tiers():
    from vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler import (
        OffloadingConnectorScheduler,
    )

    sched = OffloadingConnectorScheduler.__new__(OffloadingConnectorScheduler)
    sched._data_parallel_size = 1
    # No secondary tiers -> not migratable -> None -> HTTP 501.
    sched.manager = TieringOffloadingManager(
        primary_tier=_FakePrimaryTier(), secondary_tiers=[]
    )
    assert sched.on_rpc(b"anything") is None


def test_scheduler_on_rpc_dispatches_when_migratable():
    from vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler import (
        OffloadingConnectorScheduler,
    )

    sched = OffloadingConnectorScheduler.__new__(OffloadingConnectorScheduler)
    sched._data_parallel_size = 1
    mgr, _, _ = _make_manager()
    sched.manager = mgr
    req = msgspec.msgpack.encode(
        {
            "op": "migrate",
            "transfer_id": "w1",
            "source": {"host": "h", "port": 5710},
            "blocks": [bytes(K1)],
        }
    )
    resp = sched.on_rpc(req)
    assert resp is not None
    assert msgspec.msgpack.decode(resp) == {
        "transfer_id": "w1",
        "accepted": True,
        "num_blocks": 1,
    }


# --------------------------------------------------------------------------
# Failure reporting: only a clean source miss may still report "completed"
# --------------------------------------------------------------------------


def test_migration_capacity_failure_reports_failed_not_missing():
    """A tier held the block but the primary tier had no room for it. That is
    our failure, not a source miss, so it must not terminate as "completed" —
    an orchestrator reading that would route a request to a cold cache."""
    mgr, primary, p2p = _make_manager()
    p2p.resolve_hit(K1)
    primary.fail_writes = True
    mgr.submit_migration("cap", "h", 5710, [K1])
    _step(mgr)
    status = mgr.poll_migration("cap")
    assert status["state"] == "failed"
    assert status["error"] == "insufficient_capacity"
    assert status["blocks_no_capacity"] == 1
    assert status["blocks_failed"] == 1
    assert status["blocks_missing"] == 0
    assert status["blocks_done"] == 0


def test_migration_timeout_reports_failed():
    """An unresolved probe past the deadline is failed, not filed as a source
    miss — we never learned whether the peer had the block."""
    mgr, _, _ = _make_manager()
    mgr.submit_migration("slow", "h", 5710, [K1])
    _step(mgr)
    assert mgr.poll_migration("slow")["state"] == "running"
    mgr._migrations["slow"].deadline = time.monotonic() - 1.0
    _step(mgr)
    status = mgr.poll_migration("slow")
    assert status["state"] == "failed"
    assert status["error"] == "timeout"
    assert status["blocks_failed"] == 1
    assert status["blocks_missing"] == 0


def test_migration_submit_rejects_beyond_primary_capacity():
    primary = _FakePrimaryTier(num_blocks=1)
    mgr = TieringOffloadingManager(
        primary_tier=primary, secondary_tiers=[_FakeP2PTier()]
    )
    assert mgr.submit_migration("big", "h", 5710, [K1, K2]) == (
        False,
        0,
        "insufficient_capacity",
    )
    assert mgr.poll_migration("big") is None


def test_migration_submit_rejects_too_many_concurrent():
    primary = _FakePrimaryTier(num_blocks=1024)
    mgr = TieringOffloadingManager(
        primary_tier=primary, secondary_tiers=[_FakeP2PTier()]
    )
    for i in range(_MAX_ACTIVE_MIGRATIONS):
        accepted, _, error = mgr.submit_migration(f"m{i}", "h", 5710, [_key(i)])
        assert (accepted, error) == (True, None)
    assert mgr.submit_migration("one-too-many", "h", 5710, [_key(9999)]) == (
        False,
        0,
        "too_many_migrations",
    )


def test_migration_duplicate_keys_reconcile_against_total():
    """Duplicate hashes are collapsed, so blocks_total always equals
    done + missing + failed once the migration terminates."""
    mgr, _, p2p = _make_manager()
    p2p.resolve_miss(K1)
    accepted, num_blocks, _ = mgr.submit_migration("dup", "h", 5710, [K1, K1, K2])
    assert (accepted, num_blocks) == (True, 2)
    p2p.resolve_miss(K2)
    _step(mgr)
    status = mgr.poll_migration("dup")
    assert status["blocks_total"] == 2
    assert (
        status["blocks_done"] + status["blocks_missing"] + status["blocks_failed"]
        == status["blocks_total"]
    )


# --------------------------------------------------------------------------
# Cost control and the actual payoff
# --------------------------------------------------------------------------


def test_migration_bounds_keys_advanced_per_step():
    """_drive_migrations runs inside the scheduler step, so a large prefetch
    must not turn into an unbounded number of tier lookups on that step."""
    num_keys = _MIGRATION_KEYS_PER_STEP + 10
    primary = _FakePrimaryTier(num_blocks=num_keys * 2)
    p2p = _FakeP2PTier()
    mgr = TieringOffloadingManager(primary_tier=primary, secondary_tiers=[p2p])
    keys = [_key(i) for i in range(num_keys)]
    assert mgr.submit_migration("wide", "h", 5710, keys)[0] is True

    _step(mgr)
    assert len(p2p.lookups) == _MIGRATION_KEYS_PER_STEP
    # Nothing resolved, so the rest are still owed and picked up next step.
    assert mgr.poll_migration("wide")["state"] == "running"
    _step(mgr)
    assert len(p2p.lookups) == 2 * _MIGRATION_KEYS_PER_STEP


def test_migrated_block_serves_a_later_request():
    """The point of the whole feature: once migrated, an ordinary request's
    lookup hits the block in CPU with no further remote fetch."""
    mgr, _, p2p = _make_manager()
    p2p.resolve_hit(K1)
    mgr.submit_migration("warm", "h", 5710, [K1])
    _step(mgr)
    _step(mgr)
    assert mgr.poll_migration("warm")["state"] == "completed"

    p2p.lookups.clear()
    real_ctx = ReqContext(req_id="real-request")
    mgr.on_new_request(real_ctx)
    assert mgr.lookup(K1, real_ctx) is LookupResult.HIT
    assert p2p.lookups == []  # served from CPU, no peer round-trip


def test_cancel_during_inflight_fetch_releases_primary_blocks():
    """Cancelling mid-fetch must drain the promotion job and free the CPU
    blocks it reserved. If it does not, those blocks stay write-pending
    forever and the next reset_cache()/sleep trips its assertions."""
    mgr, primary, p2p = _make_manager()
    p2p.resolve_hit(K1)
    p2p.hold_jobs = True  # the fetch stays in flight
    mgr.submit_migration("mid", "h", 5710, [K1])
    _step(mgr)
    assert mgr._jobs, "expected a promotion job in flight"
    assert primary.lookup(K1, None) is LookupResult.HIT_PENDING

    assert mgr.cancel_migration("mid") is True
    _step(mgr)

    assert mgr.poll_migration("mid")["state"] == "cancelled"
    assert not mgr._jobs
    assert primary.lookup(K1, None) is LookupResult.MISS
    # The invariants reset_cache() asserts on must hold.
    mgr._metrics.assert_idle()


# --------------------------------------------------------------------------
# on_rpc gating: capability and topology
# --------------------------------------------------------------------------


def test_scheduler_on_rpc_gates_when_no_tier_supports_migration():
    """A configured secondary tier that cannot pull from a named peer (e.g. a
    local store) must not accept migrations."""
    from vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler import (
        OffloadingConnectorScheduler,
    )

    class _LocalTier(_FakeP2PTier):
        supports_migration = False

    sched = OffloadingConnectorScheduler.__new__(OffloadingConnectorScheduler)
    sched._data_parallel_size = 1
    sched.manager = TieringOffloadingManager(
        primary_tier=_FakePrimaryTier(), secondary_tiers=[_LocalTier()]
    )
    assert sched.on_rpc(b"anything") is None


def test_scheduler_on_rpc_refuses_under_data_parallel():
    """The control RPC carries no DP-rank target and each rank owns its own CPU
    cache, so a migration cannot be aimed at the rank that will serve."""
    from vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler import (
        OffloadingConnectorScheduler,
    )

    sched = OffloadingConnectorScheduler.__new__(OffloadingConnectorScheduler)
    sched._data_parallel_size = 2
    mgr, _, _ = _make_manager()
    sched.manager = mgr
    resp = sched.on_rpc(
        msgspec.msgpack.encode(
            {
                "op": "migrate",
                "transfer_id": "dp1",
                "source": {"host": "h", "port": 5710},
                "blocks": [bytes(K1)],
            }
        )
    )
    assert resp is not None
    assert "unsupported_topology" in _decode(resp)["error"]
    assert mgr.poll_migration("dp1") is None


def test_codec_rejects_unsupported_version_and_bad_transfer_id():
    mgr = _StubManager()
    resp = handle_migration_rpc(
        mgr, msgspec.msgpack.encode({"v": 2, "op": "poll", "transfer_id": "x"})
    )
    assert "unsupported_version" in _decode(resp)["error"]

    resp = handle_migration_rpc(
        mgr, msgspec.msgpack.encode({"op": "poll", "transfer_id": "bad id!"})
    )
    assert "transfer_id" in _decode(resp)["error"]

    resp = handle_migration_rpc(
        mgr,
        msgspec.msgpack.encode(
            {"op": "poll", "transfer_id": "x" * (MAX_TRANSFER_ID_LEN + 1)}
        ),
    )
    assert "transfer_id" in _decode(resp)["error"]
    assert mgr.submitted == []
