# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A `p2p.fetch` kv_hints action pulls blocks through the real P2P tier.

Two TieringOffloadingManagers, each over a real CPU primary tier and a real
P2PSecondaryTierManager, are paired over the in-memory loopback control
transport used by test_manager.py; only the NIXL data plane is faked. The
destination receives the hint on an unrelated carrier request and must pull the
block from the source's CPU cache into its own.
"""

from __future__ import annotations

from tests.v1.kv_offload.tiering.p2p.test_manager import _build_paired_managers
from tests.v1.kv_offload.tiering.test_tiering_offloading import _mock_mmap_region
from vllm.v1.kv_hints import KvHintAction, KvHintsEnvelope
from vllm.v1.kv_offload.base import (
    LookupResult,
    ReqContext,
    ScheduleEndContext,
    make_offload_key,
)
from vllm.v1.kv_offload.tiering.manager import (
    CPUPrimaryTierOffloadingManager,
    TieringOffloadingManager,
)
from vllm.v1.kv_offload.tiering.p2p.manager import P2PSecondaryTierManager

NUM_CHUNKS = 8


def _tiering(p2p: P2PSecondaryTierManager) -> TieringOffloadingManager:
    p2p.locality = None
    primary = CPUPrimaryTierOffloadingManager(
        num_chunks=NUM_CHUNKS, mmap_region=_mock_mmap_region(NUM_CHUNKS)
    )
    # No tier-poll thread: the test drives every round on its own thread.
    return TieringOffloadingManager(
        primary_tier=primary, secondary_tiers=[p2p], tier_poll_interval_s=0
    )


def _step(*managers: TieringOffloadingManager) -> None:
    for mgr in managers:
        mgr.on_schedule_end(ScheduleEndContext(new_req_ids=[], preempted_req_ids=()))


def _carry(mgr: TieringOffloadingManager, blocks: list[str], host: str, port: int):
    ctx = ReqContext(
        req_id="carrier",
        kv_hints=KvHintsEnvelope(
            protocol_version="0.1",
            message_id="msg-1",
            actions=[
                KvHintAction(
                    action_id="m0",
                    action_type="p2p.fetch",
                    action_version="1.0",
                    payload={
                        "source": {"host": host, "port": port},
                        "blocks": blocks,
                    },
                )
            ],
        ),
    )
    mgr.on_new_request(ctx)
    mgr.on_request_finished(ctx)


def _run_until_terminal(dst, src, transfer_id: str, max_rounds: int = 50) -> dict:
    for _ in range(max_rounds):
        _step(dst, src)
        status = dst.poll_proactive_fetch(transfer_id)
        assert status is not None
        if status["state"] != "running":
            return status
    raise AssertionError(
        f"fetch still running: {dst.poll_proactive_fetch(transfer_id)}"
    )


def test_hint_pulls_a_block_from_the_source_cpu_cache():
    p2p_dst, p2p_src = _build_paired_managers()  # local ids "A:1" and "B:2"
    dst, src = _tiering(p2p_dst), _tiering(p2p_src)

    held, absent = make_offload_key(b"held", 0), make_offload_key(b"absent", 0)
    seed = ReqContext(req_id="seed")
    src.on_new_request(seed)
    src.primary_tier.prepare_store([held], seed)
    src.primary_tier.complete_store([held], seed)

    _carry(dst, [held.hex(), absent.hex()], host="B", port=2)
    status = _run_until_terminal(dst, src, "msg-1:m0")

    assert status["state"] == "completed"
    assert (status["blocks_done"], status["blocks_missing"]) == (1, 1)
    later = ReqContext(req_id="later")
    dst.on_new_request(later)
    assert dst.primary_tier.lookup(held, later) is LookupResult.HIT
    assert dst.primary_tier.lookup(absent, later) is LookupResult.MISS
    # The synthetic fetch request is gone on both sides.
    assert "p2p-fetch:msg-1:m0" not in dst._req_state
    assert not dst.has_pending_work() or dst._polled_tiers
