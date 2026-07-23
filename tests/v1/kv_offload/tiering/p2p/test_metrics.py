# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for P2P secondary-tier metrics.

Covers the three layers that produce metrics:
  - definitions (build_p2p_metric_definitions / the manager hook),
  - client-role sample accumulation (lookup/fetch RTT, hits/misses, timeouts),
  - manager aggregation end to end (_poll_once + _reap_* + get_stats).
"""

from __future__ import annotations

from vllm.v1.kv_offload.base import (
    OffloadingCounterMetadata,
    OffloadingGaugeMetadata,
    OffloadingHistogramMetadata,
    OffloadKey,
)
from vllm.v1.kv_offload.tiering.p2p.data.base import TransferTelemetry
from vllm.v1.kv_offload.tiering.p2p.manager import (
    _UNBOUND_STORE_TIMEOUT_S,
    P2PSecondaryTierManager,
    _UnboundStoreBatch,
)
from vllm.v1.kv_offload.tiering.p2p.metrics import (
    P2PMetricNames,
    build_p2p_metric_definitions,
)
from vllm.v1.kv_offload.tiering.p2p.session import SessionCloseResult, SessionPollResult
from vllm.v1.kv_offload.tiering.p2p.session.client import ClientMetrics, ClientRole

# ---------------------------------------------------------------------------
# Metric definitions
# ---------------------------------------------------------------------------


class TestMetricDefinitions:
    def test_all_names_prefixed_and_present(self):
        defs = build_p2p_metric_definitions()
        expected = {
            v
            for k, v in vars(P2PMetricNames).items()
            if not k.startswith("_") and isinstance(v, str)
        }
        assert set(defs) == expected
        assert all(name.startswith("vllm:kv_offload_p2p_") for name in defs)

    def test_manager_hook_matches_module_definitions(self):
        assert P2PSecondaryTierManager.build_metric_definitions({}) == (
            build_p2p_metric_definitions()
        )

    def test_metric_types_and_buckets(self):
        defs = build_p2p_metric_definitions()
        # RTT + telemetry latencies/sizes are histograms with buckets.
        for name in (
            P2PMetricNames.LOOKUP_RTT,
            P2PMetricNames.FETCH_RTT,
            P2PMetricNames.TRANSFER_TIME,
            P2PMetricNames.POST_TIME,
            P2PMetricNames.TRANSFER_BYTES,
            P2PMetricNames.NUM_DESCRIPTORS,
        ):
            meta = defs[name]
            assert isinstance(meta, OffloadingHistogramMetadata)
            assert meta.buckets and list(meta.buckets) == sorted(meta.buckets)
        # Reliability signals are counters.
        for name in (
            P2PMetricNames.LOOKUP_HITS,
            P2PMetricNames.LOOKUP_MISSES,
            P2PMetricNames.LOAD_FAILURES,
            P2PMetricNames.STORE_FAILURES,
            P2PMetricNames.LOAD_TIMEOUTS,
            P2PMetricNames.UNBOUND_STORE_TIMEOUTS,
            P2PMetricNames.PEER_DISCONNECTS,
        ):
            assert isinstance(defs[name], OffloadingCounterMetadata)
        # Live-state signals are gauges.
        for name in (
            P2PMetricNames.ACTIVE_SESSIONS,
            P2PMetricNames.INFLIGHT_TRANSFERS,
        ):
            assert isinstance(defs[name], OffloadingGaugeMetadata)


# ---------------------------------------------------------------------------
# Client-role sample accumulation
# ---------------------------------------------------------------------------


class TestClientRoleMetrics:
    def _role(self) -> ClientRole:
        # Send callback is a no-op sink — we only inspect drained metrics.
        return ClientRole(peer_id="peer:1", send=lambda msg: None)

    def test_lookup_rtt_and_hits(self):
        role = self._role()
        role.register_lookup("req-1", b"k1")
        role.flush_pending_lookups()  # sends LookupMsg, stamps send time
        role.on_lookup_resp("req-1", [b"k1"], [True])

        m = role.drain_metrics()
        assert len(m.lookup_rtts) == 1
        assert m.lookup_rtts[0] >= 0.0
        assert m.lookup_hits == 1
        assert m.lookup_misses == 0

    def test_lookup_miss_counted(self):
        role = self._role()
        role.register_lookup("req-1", b"k1")
        role.register_lookup("req-1", b"k2")
        role.flush_pending_lookups()
        role.on_lookup_resp("req-1", [b"k1", b"k2"], [True, False])

        m = role.drain_metrics()
        assert m.lookup_hits == 1
        assert m.lookup_misses == 1

    def test_no_rtt_without_a_send(self):
        """A response with no matching outstanding LookupMsg records no RTT."""
        role = self._role()
        role.register_lookup("req-1", b"k1")  # registered but never flushed
        role.on_lookup_resp("req-1", [b"k1"], [True])

        m = role.drain_metrics()
        assert m.lookup_rtts == []

    def test_fetch_rtt(self):
        role = self._role()
        role.request_blocks(
            job_id=1,
            kv_request_id="req-2",
            keys=[OffloadKey(b"k")],
            block_ids=[0],
            send_ready=True,
        )
        role.on_transfer_done("req-2", success=True)

        m = role.drain_metrics()
        assert len(m.fetch_rtts) == 1
        assert m.fetch_rtts[0] >= 0.0

    def test_load_timeout_counted(self):
        role = self._role()
        role.request_blocks(
            job_id=1,
            kv_request_id="req-3",
            keys=[OffloadKey(b"k")],
            block_ids=[0],
            send_ready=True,
        )
        # Backdate the submit so collect_results sees a timed-out load.
        st = role._requests["req-3"]
        assert st.load is not None
        st.load.submitted_at -= 10_000.0
        role.collect_results()

        m = role.drain_metrics()
        assert m.load_timeouts == 1

    def test_drain_resets(self):
        role = self._role()
        role.register_lookup("req-1", b"k1")
        role.flush_pending_lookups()
        role.on_lookup_resp("req-1", [b"k1"], [True])
        role.drain_metrics()

        # Second drain with no new activity is empty.
        m = role.drain_metrics()
        assert m == ClientMetrics()


# ---------------------------------------------------------------------------
# Manager aggregation (end to end)
# ---------------------------------------------------------------------------


class _FakeCtrl:
    def poll(self):
        return []


class _FakeData:
    """Data transport stub exposing the metrics surface get_stats() uses."""

    def __init__(self, telemetry=None, inflight: int = 0) -> None:
        self._telemetry = list(telemetry or [])
        self._inflight = inflight

    def drain_telemetry(self):
        out = self._telemetry
        self._telemetry = []
        return out

    @property
    def inflight_count(self) -> int:
        return self._inflight

    def remove_remote_peer(self, peer_id):
        pass


class _FakeSession:
    def __init__(
        self,
        peer_id: str = "peer:1",
        *,
        alive: bool = True,
        connected: bool = True,
        loads=None,
        stores=None,
        metrics: ClientMetrics | None = None,
        close_result: SessionCloseResult | None = None,
    ) -> None:
        self.peer_id = peer_id
        self.alive = alive
        self.connected = connected
        self._loads = loads or []
        self._stores = stores or []
        self._metrics = metrics or ClientMetrics()
        self._close_result = close_result or SessionCloseResult([], [], [], [])

    def poll(self):
        return SessionPollResult(self._loads, self._stores, [])

    def drain_metrics(self):
        return self._metrics

    def close(self):
        return self._close_result


def _make_manager(data=None):
    from vllm.distributed.kv_transfer.kv_connector.v1.offloading.metrics import (
        OffloadingConnectorStats,
    )

    mgr = P2PSecondaryTierManager.__new__(P2PSecondaryTierManager)
    mgr._local_id = "127.0.0.1:7777"
    mgr._finished_jobs = []
    mgr._failed_req_ids = set()
    mgr._sessions = {}
    mgr._kv_to_session = {}
    mgr._unbound_stores = {}
    mgr._failed_serve_ctxs = []
    mgr._stats = OffloadingConnectorStats()
    mgr._control = _FakeCtrl()
    mgr._data = data or _FakeData()
    return mgr


class TestManagerMetrics:
    def test_client_metrics_and_failures_flow_to_stats(self):
        from vllm.v1.kv_offload.tiering.p2p.session import LoadResult, StoreResult

        mgr = _make_manager(data=_FakeData(inflight=2))
        session = _FakeSession(
            loads=[LoadResult(job_id=1, kv_request_id="r1", success=False)],
            stores=[StoreResult(job_id=2, success=False)],
            metrics=ClientMetrics(
                lookup_rtts=[0.001, 0.002],
                fetch_rtts=[0.05],
                lookup_hits=3,
                lookup_misses=1,
                load_timeouts=1,
            ),
        )
        mgr._sessions["peer:1"] = session

        mgr._poll_once()
        stats = mgr.get_stats()
        assert stats is not None
        reduced = stats.reduce()

        assert reduced[P2PMetricNames.LOAD_FAILURES] == 1
        assert reduced[P2PMetricNames.STORE_FAILURES] == 1
        assert reduced[P2PMetricNames.LOOKUP_HITS] == 3
        assert reduced[P2PMetricNames.LOOKUP_MISSES] == 1
        assert reduced[P2PMetricNames.LOAD_TIMEOUTS] == 1
        assert reduced[f"{P2PMetricNames.LOOKUP_RTT}_count"] == 2
        assert reduced[f"{P2PMetricNames.FETCH_RTT}_count"] == 1
        # Gauges snapshotted in get_stats.
        assert reduced[P2PMetricNames.ACTIVE_SESSIONS] == 1
        assert reduced[P2PMetricNames.INFLIGHT_TRANSFERS] == 2

    def test_transport_telemetry_folded_in_get_stats(self):
        telemetry = [
            TransferTelemetry(
                xfer_duration_s=0.01,
                post_duration_s=0.001,
                total_bytes=4096,
                desc_count=8,
            ),
            TransferTelemetry(
                xfer_duration_s=0.02,
                post_duration_s=0.002,
                total_bytes=8192,
                desc_count=16,
            ),
        ]
        mgr = _make_manager(data=_FakeData(telemetry=telemetry))
        stats = mgr.get_stats()
        assert stats is not None
        reduced = stats.reduce()

        assert reduced[f"{P2PMetricNames.TRANSFER_TIME}_count"] == 2
        assert reduced[f"{P2PMetricNames.POST_TIME}_count"] == 2
        assert reduced[f"{P2PMetricNames.TRANSFER_BYTES}_count"] == 2
        assert reduced[f"{P2PMetricNames.TRANSFER_BYTES}_sum"] == 4096 + 8192
        assert reduced[f"{P2PMetricNames.NUM_DESCRIPTORS}_count"] == 2

    def test_peer_disconnect_and_close_failures_counted(self):
        mgr = _make_manager()
        dead = _FakeSession(
            peer_id="dead:1",
            alive=False,
            connected=True,
            close_result=SessionCloseResult(
                failed_jobs=[10, 11],
                failed_req_ids=["r10", "r11"],
                failed_stores=[20],
                failed_serves=[],
            ),
        )
        mgr._sessions["dead:1"] = dead

        mgr._poll_once()
        reduced = mgr.get_stats().reduce()

        assert reduced[P2PMetricNames.PEER_DISCONNECTS] == 1
        assert reduced[P2PMetricNames.LOAD_FAILURES] == 2
        assert reduced[P2PMetricNames.STORE_FAILURES] == 1
        assert "dead:1" not in mgr._sessions

    def test_unbound_store_timeout_counted(self):
        import time

        mgr = _make_manager()
        old = time.monotonic() - _UNBOUND_STORE_TIMEOUT_S - 1.0
        batch = _UnboundStoreBatch(job_id=5, keys=[b"k"], block_ids=[0])
        batch.submitted_at = old
        mgr._unbound_stores["r-expired"] = [batch]

        mgr._poll_once()
        reduced = mgr.get_stats().reduce()

        assert reduced[P2PMetricNames.UNBOUND_STORE_TIMEOUTS] == 1

    def test_get_stats_resets_between_intervals(self):
        from vllm.v1.kv_offload.tiering.p2p.session import LoadResult

        mgr = _make_manager()
        session = _FakeSession(
            loads=[LoadResult(job_id=1, kv_request_id="r1", success=False)],
        )
        mgr._sessions["peer:1"] = session
        mgr._poll_once()
        first = mgr.get_stats().reduce()
        assert first[P2PMetricNames.LOAD_FAILURES] == 1

        # Next interval: no new failures, only the gauges carry over.
        session._loads = []
        mgr._poll_once()
        second = mgr.get_stats().reduce()
        assert P2PMetricNames.LOAD_FAILURES not in second
        assert second[P2PMetricNames.ACTIVE_SESSIONS] == 1
