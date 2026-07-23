# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prometheus metric names and definitions for the P2P secondary tier.

Kept separate from ``manager.py`` so the metric surface can be read (and
tested) in one place. ``build_p2p_metric_definitions`` is returned verbatim
from ``P2PSecondaryTierManager.build_metric_definitions`` and merged into the
tiering spec's definitions (see ``tiering/spec.py``). The manager records
observations into an ``OffloadingConnectorStats`` keyed by these names.
"""

from vllm.v1.kv_offload.base import (
    OffloadingCounterMetadata,
    OffloadingGaugeMetadata,
    OffloadingHistogramMetadata,
    OffloadingMetricMetadata,
)

# Sub-millisecond to 1s: control-plane lookup round trip (send LookupMsg →
# resolve LookupRespMsg). Mirrors the offloading lookup-delay layout.
_RTT_BUCKETS = (
    0.00001,
    0.00005,
    0.0001,
    0.0005,
    0.001,
    0.005,
    0.01,
    0.05,
    0.1,
    0.5,
    1.0,
)
# Fetch round trip (send FetchMsg → TransferDoneMsg) spans the data transfer
# and can approach the load timeout (30s), so extend the tail.
_FETCH_RTT_BUCKETS = (
    0.0001,
    0.0005,
    0.001,
    0.005,
    0.01,
    0.05,
    0.1,
    0.5,
    1.0,
    5.0,
    10.0,
    30.0,
)
# NIXL transfer/post durations. Copied from NixlPromMetrics.
_TIME_BUCKETS = (
    0.001,
    0.005,
    0.01,
    0.025,
    0.05,
    0.075,
    0.1,
    0.2,
    0.3,
    0.5,
    0.75,
    1.0,
    5.0,
)
# Bytes per transfer: uniform 2KB..16GB. Copied from NixlPromMetrics.
_BYTES_BUCKETS = tuple(2 ** (10 + i) for i in range(1, 25, 2))
# Descriptors per transfer. Copied from NixlPromMetrics.
_DESC_BUCKETS = (
    10,
    20,
    30,
    50,
    75,
    100,
    200,
    400,
    1000,
    2000,
    4000,
    10000,
    20000,
    50000,
)


class P2PMetricNames:
    """Flat ``vllm:kv_offload_p2p_*`` metric names emitted by the P2P tier."""

    # NIXL data-plane telemetry (from get_xfer_telemetry).
    TRANSFER_TIME = "vllm:kv_offload_p2p_transfer_time_seconds"
    POST_TIME = "vllm:kv_offload_p2p_post_time_seconds"
    TRANSFER_BYTES = "vllm:kv_offload_p2p_transfer_bytes"
    NUM_DESCRIPTORS = "vllm:kv_offload_p2p_num_descriptors"

    # Client-side latency.
    LOOKUP_RTT = "vllm:kv_offload_p2p_lookup_rtt_seconds"
    FETCH_RTT = "vllm:kv_offload_p2p_fetch_rtt_seconds"

    # Hit rate & reliability.
    LOOKUP_HITS = "vllm:kv_offload_p2p_lookup_hits"
    LOOKUP_MISSES = "vllm:kv_offload_p2p_lookup_misses"
    LOAD_FAILURES = "vllm:kv_offload_p2p_load_failures"
    STORE_FAILURES = "vllm:kv_offload_p2p_store_failures"
    LOAD_TIMEOUTS = "vllm:kv_offload_p2p_load_timeouts"
    UNBOUND_STORE_TIMEOUTS = "vllm:kv_offload_p2p_unbound_store_timeouts"

    # Session / control-plane health.
    ACTIVE_SESSIONS = "vllm:kv_offload_p2p_active_sessions"
    PEER_DISCONNECTS = "vllm:kv_offload_p2p_peer_disconnects"
    INFLIGHT_TRANSFERS = "vllm:kv_offload_p2p_inflight_transfers"


def build_p2p_metric_definitions() -> dict[str, OffloadingMetricMetadata]:
    """Return the Prometheus metric definitions emitted by the P2P tier."""
    return {
        P2PMetricNames.TRANSFER_TIME: OffloadingHistogramMetadata(
            documentation=(
                "Histogram of NIXL transfer duration for P2P KV block "
                "transfers, in seconds."
            ),
            buckets=_TIME_BUCKETS,
        ),
        P2PMetricNames.POST_TIME: OffloadingHistogramMetadata(
            documentation=(
                "Histogram of NIXL transfer post time for P2P KV block "
                "transfers, in seconds."
            ),
            buckets=_TIME_BUCKETS,
        ),
        P2PMetricNames.TRANSFER_BYTES: OffloadingHistogramMetadata(
            documentation="Histogram of bytes transferred per P2P KV block transfer.",
            buckets=_BYTES_BUCKETS,
        ),
        P2PMetricNames.NUM_DESCRIPTORS: OffloadingHistogramMetadata(
            documentation=("Histogram of NIXL descriptors per P2P KV block transfer."),
            buckets=_DESC_BUCKETS,
        ),
        P2PMetricNames.LOOKUP_RTT: OffloadingHistogramMetadata(
            documentation=(
                "Histogram of per-lookup-request round-trip time on the P2P "
                "secondary tier: from sending a LookupMsg to a peer until the "
                "matching LookupRespMsg resolves it, in seconds."
            ),
            buckets=_RTT_BUCKETS,
        ),
        P2PMetricNames.FETCH_RTT: OffloadingHistogramMetadata(
            documentation=(
                "Histogram of per-fetch round-trip time on the P2P secondary "
                "tier: from sending a FetchMsg to a peer until the matching "
                "TransferDoneMsg arrives, in seconds."
            ),
            buckets=_FETCH_RTT_BUCKETS,
        ),
        P2PMetricNames.LOOKUP_HITS: OffloadingCounterMetadata(
            documentation="Number of P2P lookup probes that hit on the peer.",
        ),
        P2PMetricNames.LOOKUP_MISSES: OffloadingCounterMetadata(
            documentation="Number of P2P lookup probes that missed on the peer.",
        ),
        P2PMetricNames.LOAD_FAILURES: OffloadingCounterMetadata(
            documentation="Number of failed P2P load (fetch) jobs.",
        ),
        P2PMetricNames.STORE_FAILURES: OffloadingCounterMetadata(
            documentation="Number of failed P2P store (serve) jobs.",
        ),
        P2PMetricNames.LOAD_TIMEOUTS: OffloadingCounterMetadata(
            documentation=(
                "Number of P2P loads that timed out awaiting a peer and were aborted."
            ),
        ),
        P2PMetricNames.UNBOUND_STORE_TIMEOUTS: OffloadingCounterMetadata(
            documentation=(
                "Number of P2P store jobs that expired without any peer fetching them."
            ),
        ),
        P2PMetricNames.ACTIVE_SESSIONS: OffloadingGaugeMetadata(
            documentation="Number of live P2P peer sessions.",
        ),
        P2PMetricNames.PEER_DISCONNECTS: OffloadingCounterMetadata(
            documentation="Number of P2P peer sessions reaped after disconnect.",
        ),
        P2PMetricNames.INFLIGHT_TRANSFERS: OffloadingGaugeMetadata(
            documentation="Number of P2P NIXL transfers currently in flight.",
        ),
    }
