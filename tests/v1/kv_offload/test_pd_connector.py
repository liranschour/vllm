# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Unit tests for PDConnector.
"""

import itertools
import socket
import threading
import time
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from vllm.v1.kv_offload.abstract import (
    JobMetadata,
    OffloadKey,
    get_offload_block_hash,
)
from vllm.v1.kv_offload.mediums import CPULoadStoreSpec
import vllm.v1.kv_offload.secondary_tiers.pd_connector as _pd_mod
from vllm.v1.kv_offload.secondary_tiers.pd_connector import (
    PDConnector,
    _FetchJob,
    _PendingBlock,
    _StoreJob,
)
from vllm.v1.kv_offload.tiering.manager import (
    CPUPrimaryTierOffloadingManager,
    TieringOffloadingManager,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def free_port() -> int:
    """Return an unused TCP port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def make_key(i: int) -> OffloadKey:
    return OffloadKey(i.to_bytes(4, "big") + (0).to_bytes(4, "big"))


def make_primary_view() -> memoryview:
    tensor = torch.zeros((16, 8), dtype=torch.float32)
    return memoryview(tensor.numpy())


def _make_real_bufs(
    num_total: int, block_bytes: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return (prefiller_arr, decoder_arr) with unique per-block fill pattern."""
    p = np.empty((num_total, block_bytes), dtype=np.uint8)
    for i in range(num_total):
        p[i, :] = (i + 1) % 256
    d = np.zeros((num_total, block_bytes), dtype=np.uint8)
    return p, d


class _MinimalPrimary(CPUPrimaryTierOffloadingManager):
    def __init__(self):
        pass

    def get_primary_kv_tensor(self):
        return torch.zeros((16, 8), dtype=torch.int8)


# ---------------------------------------------------------------------------
# Skeleton tests
# ---------------------------------------------------------------------------

class TestPDConnectorSkeleton:

    def test_get_tier_name(self):
        p = free_port()
        c = PDConnector("127.0.0.1", p)
        try:
            assert c.get_tier_name() == "PDConnector"
        finally:
            c.close()

    def test_set_primary_view_stores_view(self):
        p = free_port()
        c = PDConnector("127.0.0.1", p)
        try:
            assert c._primary_view is None
            view = make_primary_view()
            c.set_primary_view(view)
            assert c._primary_view is view
        finally:
            c.close()

    def test_set_primary_view_builds_kv_blocks(self):
        p = free_port()
        c = PDConnector("127.0.0.1", p)
        try:
            view = make_primary_view()  # shape (16, 8)
            c.set_primary_view(view)
            arr = view.obj  # underlying numpy array
            assert len(c._kv_blocks) == arr.shape[0]  # 16 blocks
            assert bytes(c._kv_blocks[0]) == arr[0].tobytes()
        finally:
            c.close()

    def test_lookup_raises_not_implemented(self):
        p = free_port()
        c = PDConnector("127.0.0.1", p)
        try:
            with pytest.raises(NotImplementedError):
                c.lookup([make_key(0)])
        finally:
            c.close()

    def test_registered_with_tiering_manager(self):
        p = free_port()
        connector = PDConnector("127.0.0.1", p)
        try:
            primary = _MinimalPrimary()
            manager = TieringOffloadingManager(
                primary_tier=primary,
                secondary_tiers=[connector],
            )
            assert connector._primary_view is not None
            assert manager.secondary_tiers == [connector]
        finally:
            connector.close()


# ---------------------------------------------------------------------------
# ZMQ control channel tests
# ---------------------------------------------------------------------------

class _CapturingConnector(PDConnector):
    """PDConnector that records received messages and peer-down events."""

    def __init__(self, host: str, port: int):
        super().__init__(host, port)
        self.received: list[tuple[str, dict]] = []  # (sender_id, msg)
        self.peers_down: list[str] = []
        self._msg_event = threading.Event()
        self._down_event = threading.Event()

    def _handle_message(self, sender_id: str, msg: dict) -> None:
        self.received.append((sender_id, msg))
        self._msg_event.set()

    def _on_peer_down(self, peer_id: str) -> None:
        super()._on_peer_down(peer_id)
        self.peers_down.append(peer_id)
        self._down_event.set()

    def wait_for_message(self, timeout: float = 2.0) -> bool:
        got = self._msg_event.wait(timeout)
        self._msg_event.clear()
        return got

    def wait_for_peer_down(self, timeout: float = 5.0) -> bool:
        got = self._down_event.wait(timeout)
        self._down_event.clear()
        return got


class TestZMQControlChannel:

    def test_send_from_a_to_b(self):
        """A sends a message to B; B receives it."""
        pa, pb = free_port(), free_port()
        a = _CapturingConnector("127.0.0.1", pa)
        b = _CapturingConnector("127.0.0.1", pb)
        try:
            a._open_channel(b._peer_id, "127.0.0.1", pb)
            time.sleep(0.1)  # allow TCP connection to establish

            a._send(b._peer_id, {"type": "ping", "data": "hello"})

            assert b.wait_for_message(timeout=2.0), "B did not receive message from A"
            assert len(b.received) == 1
            sender_id, msg = b.received[0]
            assert sender_id == a._peer_id
            assert msg == {"type": "ping", "data": "hello"}
        finally:
            a.close()
            b.close()

    def test_send_from_b_to_a(self):
        """B sends a message to A; A receives it (both sides connect)."""
        pa, pb = free_port(), free_port()
        a = _CapturingConnector("127.0.0.1", pa)
        b = _CapturingConnector("127.0.0.1", pb)
        try:
            # Both sides connect to each other for full bidirectionality
            a._open_channel(b._peer_id, "127.0.0.1", pb)
            b._open_channel(a._peer_id, "127.0.0.1", pa)
            time.sleep(0.1)

            b._send(a._peer_id, {"type": "pong", "value": 42})

            assert a.wait_for_message(timeout=2.0), "A did not receive message from B"
            sender_id, msg = a.received[0]
            assert sender_id == b._peer_id
            assert msg == {"type": "pong", "value": 42}
        finally:
            a.close()
            b.close()

    def test_bidirectional_messaging(self):
        """A→B and B→A both work after mutual connect."""
        pa, pb = free_port(), free_port()
        a = _CapturingConnector("127.0.0.1", pa)
        b = _CapturingConnector("127.0.0.1", pb)
        try:
            a._open_channel(b._peer_id, "127.0.0.1", pb)
            b._open_channel(a._peer_id, "127.0.0.1", pa)
            time.sleep(0.1)

            a._send(b._peer_id, {"type": "from_a"})
            b._send(a._peer_id, {"type": "from_b"})

            assert b.wait_for_message(timeout=2.0)
            assert a.wait_for_message(timeout=2.0)

            assert b.received[0][1]["type"] == "from_a"
            assert a.received[0][1]["type"] == "from_b"
        finally:
            a.close()
            b.close()

    def test_peer_down_on_clean_disconnect(self):
        """When A closes, B's _on_peer_down is called."""
        pa, pb = free_port(), free_port()
        a = _CapturingConnector("127.0.0.1", pa)
        b = _CapturingConnector("127.0.0.1", pb)
        try:
            # A connects to B so A can send the disconnect message
            a._open_channel(b._peer_id, "127.0.0.1", pb)
            time.sleep(0.1)

            a.close()  # sends {"type": "disconnect"} before tearing down

            assert b.wait_for_peer_down(timeout=3.0), \
                "B was not notified of A's disconnect"
            assert a._peer_id in b.peers_down
        finally:
            b.close()

    def test_send_to_unknown_peer_raises(self):
        """_send to a peer that was never connected raises RuntimeError."""
        pa = free_port()
        a = _CapturingConnector("127.0.0.1", pa)
        try:
            with pytest.raises(RuntimeError, match="no connection"):
                a._send("127.0.0.1:9999", {"type": "test"})
        finally:
            a.close()

    def test_close_is_idempotent(self):
        """Calling close() twice does not raise."""
        p = free_port()
        c = PDConnector("127.0.0.1", p)
        c.close()
        c.close()  # should not raise


# ---------------------------------------------------------------------------
# submit_store() and get_finished() tests
# ---------------------------------------------------------------------------

class TestSubmitStoreJobTracking:

    def test_submit_store_creates_store_job(self):
        """submit_store adds a _StoreJob with correct remaining count."""
        p = free_port()
        c = PDConnector("127.0.0.1", p)
        try:
            c.set_primary_view(make_primary_view())
            keys = [make_key(0), make_key(1), make_key(2)]
            spec = CPULoadStoreSpec(block_ids=[0, 1, 2])
            c.submit_store(JobMetadata(job_id=42, keys=keys, spec=spec))

            assert 42 in c._store_jobs
            assert c._store_jobs[42].remaining == 3
        finally:
            c.close()

    def test_submit_store_populates_block_to_job(self):
        """Each block hash is recorded in _block_to_job with correct job_id and index."""
        p = free_port()
        c = PDConnector("127.0.0.1", p)
        try:
            c.set_primary_view(make_primary_view())
            keys = [make_key(0), make_key(1)]
            spec = CPULoadStoreSpec(block_ids=[5, 7])
            c.submit_store(JobMetadata(job_id=10, keys=keys, spec=spec))

            h0 = get_offload_block_hash(keys[0])
            h1 = get_offload_block_hash(keys[1])
            assert c._block_to_job[h0] == [(10, 5)]
            assert c._block_to_job[h1] == [(10, 7)]
        finally:
            c.close()

    def test_get_finished_returns_empty_when_no_transfers(self):
        """With empty _pending_blocks no transfers happen, get_finished returns []."""
        p = free_port()
        c = PDConnector("127.0.0.1", p)
        try:
            c.set_primary_view(make_primary_view())
            keys = [make_key(0)]
            spec = CPULoadStoreSpec(block_ids=[0])
            c.submit_store(JobMetadata(job_id=1, keys=keys, spec=spec))

            finished = list(c.get_finished())
            assert finished == []
        finally:
            c.close()

    def test_submit_store_job_remaining_matches_key_count(self):
        """remaining equals the number of keys submitted."""
        p = free_port()
        c = PDConnector("127.0.0.1", p)
        try:
            c.set_primary_view(make_primary_view())
            for n in (1, 5, 10):
                keys = [make_key(i) for i in range(n)]
                spec = CPULoadStoreSpec(block_ids=list(range(n)))
                job_id = 100 + n
                c.submit_store(JobMetadata(job_id=job_id, keys=keys, spec=spec))
                assert c._store_jobs[job_id].remaining == n
        finally:
            c.close()


# ---------------------------------------------------------------------------
# submit_load() and lookup_fetch tests
# ---------------------------------------------------------------------------

class TestSubmitLoadAndLookupFetch:

    def test_submit_load_creates_load_job(self):
        """submit_load adds a _LoadJob with correct peer_id."""
        pp, pd = free_port(), free_port()
        prefiller = PDConnector("127.0.0.1", pp)
        decoder = PDConnector("127.0.0.1", pd)
        try:
            prefiller.set_primary_view(make_primary_view())
            decoder.set_primary_view(make_primary_view())

            decoder.set_load_peer(42, prefiller._peer_id)
            keys = [make_key(0), make_key(1)]
            spec = CPULoadStoreSpec(block_ids=[0, 1])
            decoder.submit_load(JobMetadata(job_id=42, keys=keys, spec=spec))

            assert 42 in decoder._load_jobs
            assert decoder._load_jobs[42].peer_id == prefiller._peer_id
        finally:
            decoder.close()
            prefiller.close()

    def test_lookup_fetch_populates_pending_blocks(self):
        """When Prefiller has no stored blocks, lookup_fetch inserts into _pending_blocks."""
        pp, pd = free_port(), free_port()
        prefiller = PDConnector("127.0.0.1", pp)
        decoder = PDConnector("127.0.0.1", pd)
        try:
            prefiller.set_primary_view(make_primary_view())
            decoder.set_primary_view(make_primary_view())

            keys = [make_key(0), make_key(1)]
            spec = CPULoadStoreSpec(block_ids=[3, 5])
            decoder.set_load_peer(10, prefiller._peer_id)
            decoder.submit_load(JobMetadata(job_id=10, keys=keys, spec=spec))

            time.sleep(0.3)

            h0 = get_offload_block_hash(keys[0])
            h1 = get_offload_block_hash(keys[1])
            assert h0 in prefiller._pending_blocks
            assert h1 in prefiller._pending_blocks
            assert prefiller._pending_blocks[h0].remote_block_idx == 3
            assert prefiller._pending_blocks[h1].remote_block_idx == 5
            assert prefiller._pending_blocks[h0].peer_id == decoder._peer_id
        finally:
            decoder.close()
            prefiller.close()

    def test_lookup_fetch_matches_stored_blocks(self):
        """When Prefiller has stored blocks, lookup_fetch removes them from _block_to_job."""
        pp, pd = free_port(), free_port()
        prefiller = PDConnector("127.0.0.1", pp)
        decoder = PDConnector("127.0.0.1", pd)
        try:
            prefiller.set_primary_view(make_primary_view())
            decoder.set_primary_view(make_primary_view())

            store_keys = [make_key(0), make_key(1)]
            store_spec = CPULoadStoreSpec(block_ids=[2, 4])
            prefiller.submit_store(
                JobMetadata(job_id=1, keys=store_keys, spec=store_spec)
            )

            h0 = get_offload_block_hash(store_keys[0])
            h1 = get_offload_block_hash(store_keys[1])
            assert h0 in prefiller._block_to_job
            assert h1 in prefiller._block_to_job

            # Mock NIXL transport so the transfer call doesn't block.
            prefiller._agent.make_prepped_xfer = MagicMock(return_value=999)
            prefiller._agent.transfer = MagicMock()

            load_keys = [make_key(0), make_key(1)]
            load_spec = CPULoadStoreSpec(block_ids=[6, 8])
            decoder.set_load_peer(20, prefiller._peer_id)
            decoder.submit_load(
                JobMetadata(job_id=20, keys=load_keys, spec=load_spec)
            )

            time.sleep(0.3)

            assert h0 not in prefiller._block_to_job
            assert h1 not in prefiller._block_to_job
            assert len(prefiller._pending_blocks) == 0
        finally:
            decoder.close()
            prefiller.close()

    def test_submit_store_matches_pending_blocks(self):
        """submit_store matches _pending_blocks: blocks not added to _block_to_job."""
        p = free_port()
        prefiller = PDConnector("127.0.0.1", p)
        try:
            prefiller.set_primary_view(make_primary_view())

            # Directly inject pending blocks (as if a lookup_fetch arrived).
            keys = [make_key(0), make_key(1), make_key(2)]
            h0 = get_offload_block_hash(keys[0])
            h1 = get_offload_block_hash(keys[1])
            h2 = get_offload_block_hash(keys[2])
            prefiller._pending_blocks[h0] = _PendingBlock(
                peer_id="127.0.0.1:9999", remote_block_idx=3, decoder_job_id=0
            )
            prefiller._pending_blocks[h1] = _PendingBlock(
                peer_id="127.0.0.1:9999", remote_block_idx=5, decoder_job_id=0
            )
            prefiller._pending_blocks[h2] = _PendingBlock(
                peer_id="127.0.0.1:9999", remote_block_idx=7, decoder_job_id=0
            )

            # Mock NIXL transport and remote dlist.
            prefiller._agent.make_prepped_xfer = MagicMock(return_value=888)
            prefiller._agent.transfer = MagicMock()
            prefiller._remote_dlists["127.0.0.1:9999"] = MagicMock()

            store_keys = [make_key(0), make_key(1)]
            store_spec = CPULoadStoreSpec(block_ids=[0, 1])
            prefiller.submit_store(
                JobMetadata(job_id=2, keys=store_keys, spec=store_spec)
            )

            assert h0 not in prefiller._pending_blocks
            assert h1 not in prefiller._pending_blocks
            assert h0 not in prefiller._block_to_job
            assert h1 not in prefiller._block_to_job
            # key(2) was not in the store job — still pending
            assert h2 in prefiller._pending_blocks
        finally:
            prefiller.close()


# ---------------------------------------------------------------------------
# NIXL registration tests
# ---------------------------------------------------------------------------

pytest.importorskip("nixl._api", reason="NIXL not installed")


class TestNIXLRegistration:

    def test_nixl_agent_created_after_set_primary_view(self):
        """_agent is None before set_primary_view, not None after."""
        p = free_port()
        c = PDConnector("127.0.0.1", p)
        try:
            assert c._agent is None
            c.set_primary_view(make_primary_view())
            assert c._agent is not None
        finally:
            c.close()

    def test_reg_and_local_dlist_set_after_set_primary_view(self):
        """_reg and _local_dlist are set after set_primary_view."""
        p = free_port()
        c = PDConnector("127.0.0.1", p)
        try:
            assert c._reg is None
            assert c._local_dlist is None
            c.set_primary_view(make_primary_view())
            assert c._reg is not None
            assert c._local_dlist is not None
        finally:
            c.close()

    def test_close_releases_dlist_and_deregisters(self):
        """close() sets _local_dlist and _reg back to None."""
        p = free_port()
        c = PDConnector("127.0.0.1", p)
        c.set_primary_view(make_primary_view())
        assert c._local_dlist is not None
        c.close()
        assert c._local_dlist is None
        assert c._reg is None


# ---------------------------------------------------------------------------
# Connection establishment tests
# ---------------------------------------------------------------------------


class TestConnectionEstablishment:
    """
    Tests for the NIXL handshake between two PDConnector instances.

    The 'decoder' calls _ensure_connected(prefiller_peer_id); the 'prefiller'
    handles the 'connect' message and replies with 'connect_ack'.  After the
    handshake both sides should have the peer in _connections, and the
    prefiller should have a _remote_dlists entry for the decoder.
    """

    def test_ensure_connected_populates_connections_both_sides(self):
        """After _ensure_connected, both connector._connections contain the other."""
        pp, pd = free_port(), free_port()
        prefiller = PDConnector("127.0.0.1", pp)
        decoder   = PDConnector("127.0.0.1", pd)
        try:
            prefiller.set_primary_view(make_primary_view())
            decoder.set_primary_view(make_primary_view())

            decoder._ensure_connected(prefiller._peer_id)

            # Decoder must know it's connected to prefiller.
            assert prefiller._peer_id in decoder._connections
            # Prefiller must know it's connected to decoder.
            assert decoder._peer_id in prefiller._connections
        finally:
            decoder.close()
            prefiller.close()

    def test_ensure_connected_builds_remote_dlist_on_prefiller(self):
        """Prefiller has a _remote_dlists entry for the decoder after handshake."""
        pp, pd = free_port(), free_port()
        prefiller = PDConnector("127.0.0.1", pp)
        decoder   = PDConnector("127.0.0.1", pd)
        try:
            prefiller.set_primary_view(make_primary_view())
            decoder.set_primary_view(make_primary_view())

            decoder._ensure_connected(prefiller._peer_id)

            assert decoder._peer_id in prefiller._remote_dlists
            assert prefiller._remote_dlists[decoder._peer_id] is not None
        finally:
            decoder.close()
            prefiller.close()

    def test_ensure_connected_is_idempotent(self):
        """Calling _ensure_connected twice does not raise and does not duplicate state."""
        pp, pd = free_port(), free_port()
        prefiller = PDConnector("127.0.0.1", pp)
        decoder   = PDConnector("127.0.0.1", pd)
        try:
            prefiller.set_primary_view(make_primary_view())
            decoder.set_primary_view(make_primary_view())

            decoder._ensure_connected(prefiller._peer_id)
            decoder._ensure_connected(prefiller._peer_id)  # second call is a no-op

            assert len([p for p in decoder._connections
                        if p == prefiller._peer_id]) == 1
        finally:
            decoder.close()
            prefiller.close()

    def test_block_len_mismatch_raises(self):
        """_ensure_connected raises ValueError when block_len differs."""
        pp, pd = free_port(), free_port()
        # Prefiller gets a (16, 8) float32 view → block_len = 32 bytes
        # Decoder gets a (16, 4) float32 view → block_len = 16 bytes
        prefiller = PDConnector("127.0.0.1", pp)
        decoder   = PDConnector("127.0.0.1", pd)
        try:
            import torch
            prefiller.set_primary_view(
                memoryview(torch.zeros((16, 8), dtype=torch.float32).numpy())
            )
            decoder.set_primary_view(
                memoryview(torch.zeros((16, 4), dtype=torch.float32).numpy())
            )

            with pytest.raises((ValueError, Exception)):
                decoder._ensure_connected(prefiller._peer_id)
        finally:
            decoder.close()
            prefiller.close()

    def test_peer_down_removes_remote_dlist(self):
        """After decoder disconnects, prefiller's _remote_dlists entry is removed."""
        pp, pd = free_port(), free_port()
        prefiller = PDConnector("127.0.0.1", pp)
        decoder   = PDConnector("127.0.0.1", pd)
        try:
            prefiller.set_primary_view(make_primary_view())
            decoder.set_primary_view(make_primary_view())

            decoder._ensure_connected(prefiller._peer_id)
            assert decoder._peer_id in prefiller._remote_dlists

            decoder.close()
            # Allow the disconnect message / heartbeat to propagate.
            time.sleep(0.3)

            assert decoder._peer_id not in prefiller._remote_dlists
        finally:
            prefiller.close()


# ---------------------------------------------------------------------------
# Race-condition tests
# ---------------------------------------------------------------------------

class TestRaceConditions:
    """
    Verify that concurrent access between Thread 1 (submit_store / get_finished)
    and Thread 2 (_handle_message / lookup_fetch) does not produce TOCTOU
    races, iterator invalidation, or KeyErrors.
    """

    def test_concurrent_submit_store_and_lookup_fetch(self):
        """
        No block stranded in both maps; all N unique blocks produce transfers.

        The TOCTOU race: without _lock, submit_store and _handle_message can
        simultaneously decide a block "isn't in the other's map" and each insert
        into their own structure, leaving the block with no transfer ever fired.
        """
        N = 300
        p = free_port()
        prefiller = PDConnector("127.0.0.1", p)
        try:
            prefiller.set_primary_view(make_primary_view())

            decoder_peer = "127.0.0.1:9999"
            counter = itertools.count(1)
            prefiller._agent.make_prepped_xfer = MagicMock(
                side_effect=lambda *a, **kw: next(counter)
            )
            prefiller._agent.transfer = MagicMock()
            prefiller._remote_dlists[decoder_peer] = MagicMock()

            keys = [make_key(i) for i in range(N)]
            errors: list[Exception] = []

            def do_store() -> None:
                try:
                    for i, key in enumerate(keys):
                        spec = CPULoadStoreSpec(block_ids=[i % 16])
                        prefiller.submit_store(
                            JobMetadata(job_id=i, keys=[key], spec=spec)
                        )
                except Exception as exc:
                    errors.append(exc)

            def do_fetch() -> None:
                try:
                    for i, key in enumerate(keys):
                        bh = get_offload_block_hash(key)
                        prefiller._handle_message(decoder_peer, {
                            "type": "lookup_fetch",
                            "peer_id": decoder_peer,
                            "job_id": N + i,
                            "block_hashes": [bh],
                            "block_indexes": [i % 16],
                        })
                except Exception as exc:
                    errors.append(exc)

            t1 = threading.Thread(target=do_store)
            t2 = threading.Thread(target=do_fetch)
            t1.start()
            t2.start()
            t1.join(timeout=10.0)
            t2.join(timeout=10.0)

            assert not errors, f"Unexpected exceptions: {errors}"

            # Core invariant: no block in both maps simultaneously after completion.
            with prefiller._lock:
                both = set(prefiller._block_to_job) & set(prefiller._pending_blocks)
            assert not both, f"Block in both maps (TOCTOU race): {both}"

            # All N blocks must be accounted for: matched (transfer) or pending
            # in exactly one map (store arrived before its fetch, or vice-versa).
            with prefiller._lock:
                unmatched = (
                    sum(len(v) for v in prefiller._block_to_job.values())
                    + len(prefiller._pending_blocks)
                )
            transfers = prefiller._agent.make_prepped_xfer.call_count
            assert transfers + unmatched == N, (
                f"Accounting mismatch: {transfers} transfers + "
                f"{unmatched} unmatched != {N}"
            )
            # With correct locking every block should match.
            assert transfers == N, (
                f"Expected {N} transfers, got {transfers}; "
                f"{unmatched} blocks stranded"
            )
        finally:
            prefiller.close()

    def test_get_finished_concurrent_with_lookup_fetch(self):
        """
        get_finished() and _handle_message("lookup_fetch") run concurrently;
        no KeyError and every NIXL handle is released exactly once.
        """
        N = 300
        p = free_port()
        prefiller = PDConnector("127.0.0.1", p)
        try:
            prefiller.set_primary_view(make_primary_view())

            decoder_peer = "127.0.0.1:9999"
            prefiller._remote_dlists[decoder_peer] = MagicMock()

            # Pre-populate _block_to_job and _store_jobs before threads start.
            keys = [make_key(i) for i in range(N)]
            for i, key in enumerate(keys):
                bh = get_offload_block_hash(key)
                prefiller._block_to_job[bh] = [(i, i % 16)]
                prefiller._store_jobs[i] = _StoreJob(job_id=i, remaining=1, submitted_at=0.0)

            counter = itertools.count(1)
            prefiller._agent.make_prepped_xfer = MagicMock(
                side_effect=lambda *a, **kw: next(counter)
            )
            prefiller._agent.transfer = MagicMock()
            prefiller._agent.check_xfer_state = MagicMock(return_value="DONE")
            prefiller._agent.release_xfer_handle = MagicMock()

            errors: list[Exception] = []
            stop = threading.Event()

            def do_get_finished() -> None:
                try:
                    while not stop.is_set():
                        prefiller.get_finished()
                except Exception as exc:
                    errors.append(exc)

            def do_lookup_fetch() -> None:
                try:
                    for i, key in enumerate(keys):
                        bh = get_offload_block_hash(key)
                        prefiller._handle_message(decoder_peer, {
                            "type": "lookup_fetch",
                            "peer_id": decoder_peer,
                            "job_id": N + i,
                            "block_hashes": [bh],
                            "block_indexes": [i % 16],
                        })
                except Exception as exc:
                    errors.append(exc)

            t_gf = threading.Thread(target=do_get_finished)
            t_lf = threading.Thread(target=do_lookup_fetch)
            t_gf.start()
            t_lf.start()
            t_lf.join(timeout=10.0)
            stop.set()
            t_gf.join(timeout=5.0)

            # Drain any handles still in flight after stop.
            prefiller.get_finished()

            assert not errors, f"Unexpected exceptions: {errors}"

            # Every handle that was transferred must be released exactly once.
            transferred = prefiller._agent.transfer.call_count
            released = prefiller._agent.release_xfer_handle.call_count
            assert released == transferred, (
                f"release_xfer_handle called {released} times "
                f"but transfer called {transferred} times"
            )
        finally:
            prefiller.close()

    def test_peer_down_during_submit_store(self):
        """
        _remote_dlists cleared concurrently with submit_store does not raise
        KeyError; the missing dlist is silently skipped.
        """
        N = 300
        p = free_port()
        prefiller = PDConnector("127.0.0.1", p)
        try:
            prefiller.set_primary_view(make_primary_view())

            decoder_peer = "127.0.0.1:9999"
            prefiller._agent.make_prepped_xfer = MagicMock(return_value=1)
            prefiller._agent.transfer = MagicMock()
            prefiller._agent.release_xfer_handle = MagicMock()
            prefiller._agent.remove_remote_agent = MagicMock()
            prefiller._agent.release_dlist_handle = MagicMock()

            errors: list[Exception] = []

            def do_store() -> None:
                try:
                    for i in range(N):
                        key = make_key(i)
                        bh = get_offload_block_hash(key)
                        with prefiller._lock:
                            prefiller._pending_blocks[bh] = _PendingBlock(
                                peer_id=decoder_peer,
                                remote_block_idx=i % 16,
                                decoder_job_id=i,
                            )
                            prefiller._remote_dlists[decoder_peer] = MagicMock()
                        spec = CPULoadStoreSpec(block_ids=[i % 16])
                        prefiller.submit_store(
                            JobMetadata(job_id=i, keys=[key], spec=spec)
                        )
                except Exception as exc:
                    errors.append(exc)

            def do_clear_dlists() -> None:
                try:
                    for _ in range(N * 3):
                        with prefiller._lock:
                            prefiller._remote_dlists.pop(decoder_peer, None)
                except Exception as exc:
                    errors.append(exc)

            t1 = threading.Thread(target=do_store)
            t2 = threading.Thread(target=do_clear_dlists)
            t1.start()
            t2.start()
            t1.join(timeout=10.0)
            t2.join(timeout=10.0)

            assert not errors, f"Unexpected exceptions: {errors}"
        finally:
            prefiller.close()


# ---------------------------------------------------------------------------
# Load job completion tests
# ---------------------------------------------------------------------------

class TestLoadJobCompletion:
    """
    Verify the full transfer_done notification path:
    Prefiller detects NIXL completion → sends transfer_done → Decoder's
    get_finished() returns JobResult for the load job.
    """

    def _make_prefiller_with_mocked_nixl(self) -> PDConnector:
        """Return a PDConnector with a mocked NIXL agent."""
        p = free_port()
        c = PDConnector("127.0.0.1", p)
        c.set_primary_view(make_primary_view())
        counter = itertools.count(1)
        c._agent.make_prepped_xfer = MagicMock(
            side_effect=lambda *a, **kw: next(counter)
        )
        c._agent.transfer = MagicMock()
        c._agent.check_xfer_state = MagicMock(return_value="DONE")
        c._agent.release_xfer_handle = MagicMock()
        return c

    def test_transfer_done_sent_when_all_blocks_ready(self):
        """
        All N blocks are in _block_to_job at lookup_fetch time.
        After NIXL reports DONE, get_finished() sends transfer_done.
        """
        N = 4
        p = free_port()
        prefiller = PDConnector("127.0.0.1", p)
        try:
            prefiller.set_primary_view(make_primary_view())

            decoder_peer = "127.0.0.1:9999"
            prefiller._agent.make_prepped_xfer = MagicMock(return_value=1)
            prefiller._agent.transfer = MagicMock()
            prefiller._agent.check_xfer_state = MagicMock(return_value="DONE")
            prefiller._agent.release_xfer_handle = MagicMock()
            prefiller._remote_dlists[decoder_peer] = MagicMock()

            keys = [make_key(i) for i in range(N)]
            # Pre-populate _block_to_job (simulating a prior submit_store).
            for i, key in enumerate(keys):
                bh = get_offload_block_hash(key)
                prefiller._block_to_job[bh] = [(100 + i, i)]
                prefiller._store_jobs[100 + i] = _StoreJob(job_id=100 + i, remaining=1, submitted_at=0.0)

            sent_msgs: list[dict] = []
            orig_send = prefiller._send

            def capturing_send(pid: str, msg: dict) -> None:
                sent_msgs.append(msg)
                orig_send(pid, msg)

            prefiller._send = capturing_send  # type: ignore[method-assign]
            # Open a channel so _send doesn't raise (no real dealer needed for
            # capture; intercept before the real call which would fail).
            prefiller._dealers[decoder_peer] = MagicMock()
            prefiller._dealers[decoder_peer].send = MagicMock()

            decoder_job_id = 42
            prefiller._handle_message(decoder_peer, {
                "type": "lookup_fetch",
                "peer_id": decoder_peer,
                "job_id": decoder_job_id,
                "block_hashes": [get_offload_block_hash(k) for k in keys],
                "block_indexes": list(range(N)),
            })

            # All blocks matched → one NIXL handle in flight.
            assert 1 in prefiller._inflight_xfers

            # Trigger completion polling.
            prefiller.get_finished()

            # transfer_done must have been sent.
            td_msgs = [m for m in sent_msgs if m.get("type") == "transfer_done"]
            assert len(td_msgs) == 1, f"Expected 1 transfer_done, got {td_msgs}"
            assert td_msgs[0]["job_id"] == decoder_job_id
            assert td_msgs[0]["success"] is True
        finally:
            prefiller.close()

    def test_transfer_done_sent_after_pending_blocks_matched(self):
        """
        All N blocks go to _pending_blocks; a subsequent submit_store matches
        them.  transfer_done is sent only after those NIXL transfers complete.
        """
        N = 3
        p = free_port()
        prefiller = self._make_prefiller_with_mocked_nixl()
        try:
            decoder_peer = "127.0.0.1:9999"
            prefiller._remote_dlists[decoder_peer] = MagicMock()
            prefiller._dealers[decoder_peer] = MagicMock()
            prefiller._dealers[decoder_peer].send = MagicMock()

            keys = [make_key(i) for i in range(N)]
            decoder_job_id = 7

            # lookup_fetch arrives — no blocks stored yet.
            prefiller._handle_message(decoder_peer, {
                "type": "lookup_fetch",
                "peer_id": decoder_peer,
                "job_id": decoder_job_id,
                "block_hashes": [get_offload_block_hash(k) for k in keys],
                "block_indexes": list(range(N)),
            })

            # All blocks should be pending.
            assert len(prefiller._pending_blocks) == N
            # No inflight transfers yet.
            assert not prefiller._inflight_xfers

            sent_msgs: list[dict] = []
            orig_send = prefiller._send

            def capturing_send(pid: str, msg: dict) -> None:
                sent_msgs.append(msg)
                orig_send(pid, msg)

            prefiller._send = capturing_send  # type: ignore[method-assign]

            # submit_store matches all pending blocks.
            spec = CPULoadStoreSpec(block_ids=list(range(N)))
            prefiller.submit_store(JobMetadata(job_id=200, keys=keys, spec=spec))

            # Now there should be inflight handles.
            assert prefiller._inflight_xfers

            # Before get_finished: no transfer_done yet.
            td_before = [m for m in sent_msgs if m.get("type") == "transfer_done"]
            assert not td_before

            # Trigger completion.
            prefiller.get_finished()

            td_msgs = [m for m in sent_msgs if m.get("type") == "transfer_done"]
            assert len(td_msgs) == 1
            assert td_msgs[0]["job_id"] == decoder_job_id
        finally:
            prefiller.close()

    def test_transfer_done_only_after_all_split_stores_complete(self):
        """
        k blocks are ready at lookup_fetch time; the remaining N-k arrive via
        two separate submit_store calls.  transfer_done is sent only after all
        three NIXL transfers complete.
        """
        p = free_port()
        prefiller = self._make_prefiller_with_mocked_nixl()
        try:
            decoder_peer = "127.0.0.1:9999"
            prefiller._remote_dlists[decoder_peer] = MagicMock()
            prefiller._dealers[decoder_peer] = MagicMock()
            prefiller._dealers[decoder_peer].send = MagicMock()

            # 1 block ready, 2 blocks will be pending.
            ready_key = make_key(0)
            pending_keys = [make_key(1), make_key(2)]
            all_keys = [ready_key] + pending_keys
            decoder_job_id = 55

            # Pre-populate one ready block.
            bh0 = get_offload_block_hash(ready_key)
            prefiller._block_to_job[bh0] = [(300, 0)]
            prefiller._store_jobs[300] = _StoreJob(job_id=300, remaining=1, submitted_at=0.0)

            prefiller._handle_message(decoder_peer, {
                "type": "lookup_fetch",
                "peer_id": decoder_peer,
                "job_id": decoder_job_id,
                "block_hashes": [get_offload_block_hash(k) for k in all_keys],
                "block_indexes": [0, 1, 2],
            })

            sent_msgs: list[dict] = []
            orig_send = prefiller._send

            def capturing_send(pid: str, msg: dict) -> None:
                sent_msgs.append(msg)
                orig_send(pid, msg)

            prefiller._send = capturing_send  # type: ignore[method-assign]

            # First get_finished: completes the 1-block ready transfer.
            prefiller.get_finished()
            td = [m for m in sent_msgs if m.get("type") == "transfer_done"]
            assert not td, "Should not send transfer_done with 2 blocks still pending"

            # submit_store matches one of the two pending blocks.
            prefiller.submit_store(
                JobMetadata(job_id=301, keys=[pending_keys[0]],
                            spec=CPULoadStoreSpec(block_ids=[1]))
            )
            prefiller.get_finished()
            td = [m for m in sent_msgs if m.get("type") == "transfer_done"]
            assert not td, "Should not send transfer_done with 1 block still pending"

            # submit_store matches the last pending block.
            prefiller.submit_store(
                JobMetadata(job_id=302, keys=[pending_keys[1]],
                            spec=CPULoadStoreSpec(block_ids=[2]))
            )
            prefiller.get_finished()
            td = [m for m in sent_msgs if m.get("type") == "transfer_done"]
            assert len(td) == 1, f"Expected 1 transfer_done, got {td}"
            assert td[0]["job_id"] == decoder_job_id
        finally:
            prefiller.close()

    def test_decoder_get_finished_returns_load_job_result(self):
        """
        End-to-end: Decoder submits a load job; Prefiller sends transfer_done
        via the real ZMQ channel; Decoder's get_finished() returns a JobResult.
        """
        pp, pd = free_port(), free_port()
        prefiller = PDConnector("127.0.0.1", pp)
        decoder = PDConnector("127.0.0.1", pd)
        try:
            prefiller.set_primary_view(make_primary_view())
            decoder.set_primary_view(make_primary_view())

            keys = [make_key(0), make_key(1)]
            decoder_job_id = 99

            # Decoder submits the load (this sends lookup_fetch to prefiller).
            decoder.set_load_peer(decoder_job_id, prefiller._peer_id)
            decoder.submit_load(
                JobMetadata(job_id=decoder_job_id, keys=keys,
                            spec=CPULoadStoreSpec(block_ids=[0, 1]))
            )

            # Wait for lookup_fetch to arrive and populate pending_blocks.
            deadline = time.time() + 3.0
            while time.time() < deadline:
                with prefiller._lock:
                    n_pending = len(prefiller._pending_blocks)
                if n_pending == len(keys):
                    break
                time.sleep(0.05)
            assert n_pending == len(keys), "lookup_fetch did not arrive in time"

            # Mock NIXL on prefiller so submit_store triggers a transfer.
            prefiller._agent.make_prepped_xfer = MagicMock(return_value=77)
            prefiller._agent.transfer = MagicMock()
            prefiller._agent.check_xfer_state = MagicMock(return_value="DONE")
            prefiller._agent.release_xfer_handle = MagicMock()

            # submit_store matches all pending blocks and creates a handle.
            prefiller.submit_store(
                JobMetadata(job_id=400, keys=keys,
                            spec=CPULoadStoreSpec(block_ids=[0, 1]))
            )

            # get_finished on prefiller: completes handle, sends transfer_done.
            prefiller.get_finished()

            # Allow transfer_done to travel over ZMQ to decoder.
            deadline = time.time() + 3.0
            results: list[JobResult] = []
            while time.time() < deadline:
                results = list(decoder.get_finished())
                if results:
                    break
                time.sleep(0.05)

            assert len(results) == 1, f"Expected 1 JobResult, got {results}"
            assert results[0].job_id == decoder_job_id
            assert results[0].success is True
        finally:
            decoder.close()
            prefiller.close()

    def test_peer_down_fails_decoder_load_job(self):
        """
        When the Prefiller goes down while a load job is in flight, the
        Decoder's _on_peer_down fails the load job and get_finished() returns
        JobResult(success=False).
        """
        pp, pd = free_port(), free_port()
        prefiller = PDConnector("127.0.0.1", pp)
        decoder = PDConnector("127.0.0.1", pd)
        try:
            prefiller.set_primary_view(make_primary_view())
            decoder.set_primary_view(make_primary_view())

            decoder_job_id = 11
            decoder.set_load_peer(decoder_job_id, prefiller._peer_id)
            decoder.submit_load(
                JobMetadata(job_id=decoder_job_id, keys=[make_key(0)],
                            spec=CPULoadStoreSpec(block_ids=[0]))
            )

            # Wait until lookup_fetch arrives (decoder is connected).
            deadline = time.time() + 3.0
            while time.time() < deadline:
                with decoder._lock:
                    connected = prefiller._peer_id in decoder._connections
                if connected:
                    break
                time.sleep(0.05)

            # Prefiller drops off the network.
            prefiller.close()

            # Decoder's _on_peer_down should fire (via disconnect message or
            # heartbeat). Wait for it.
            deadline = time.time() + 5.0
            results: list[JobResult] = []
            while time.time() < deadline:
                results = list(decoder.get_finished())
                if results:
                    break
                time.sleep(0.1)

            assert len(results) == 1, f"Expected failure JobResult, got {results}"
            assert results[0].job_id == decoder_job_id
            assert results[0].success is False
        finally:
            decoder.close()


# ---------------------------------------------------------------------------
# End-to-end data integrity tests (real NIXL, no mocks)
# ---------------------------------------------------------------------------

class TestEndToEndDataIntegrity:
    """
    Real NIXL WRITE transfers — no mocking of make_prepped_xfer / transfer /
    check_xfer_state.  Verifies that bytes written by the Prefiller actually
    appear in the Decoder's memory buffer after get_finished() completes.
    """

    _NUM_TOTAL  = 8    # total blocks allocated per connector
    _NUM_XFER   = 6    # blocks actually transferred
    _BLOCK_BYTES = 64  # bytes per block

    def test_split_stores_data_integrity(self):
        """
        submit_store is called in 3 phases around submit_load:
          Phase 1 (2 blocks) before submit_load
          Phase 2 (2 blocks) after lookup_fetch arrives (blocks pending)
          Phase 3 (2 blocks) after lookup_fetch arrives (blocks pending)

        All 6 transferred blocks must contain the exact bytes from
        Prefiller's memory.
        """
        N = self._NUM_XFER
        prefiller_arr, decoder_arr = _make_real_bufs(
            self._NUM_TOTAL, self._BLOCK_BYTES
        )

        pp, pd = free_port(), free_port()
        prefiller = PDConnector("127.0.0.1", pp)
        decoder   = PDConnector("127.0.0.1", pd)
        try:
            prefiller.set_primary_view(memoryview(prefiller_arr))
            decoder.set_primary_view(memoryview(decoder_arr))

            keys = [make_key(i) for i in range(N)]

            # Phase 1: store blocks 0-1 before submit_load
            prefiller.submit_store(
                JobMetadata(
                    job_id=1, keys=keys[0:2],
                    spec=CPULoadStoreSpec(block_ids=[0, 1]),
                )
            )

            # Decoder requests all 6 blocks
            decoder.set_load_peer(100, prefiller._peer_id)
            decoder.submit_load(
                JobMetadata(
                    job_id=100, keys=keys,
                    spec=CPULoadStoreSpec(block_ids=list(range(N))),
                )
            )

            # Wait for lookup_fetch: blocks 0-1 matched, blocks 2-5 pending
            deadline = time.time() + 5.0
            n_pending = 0
            while time.time() < deadline:
                with prefiller._lock:
                    n_pending = len(prefiller._pending_blocks)
                if n_pending == 4:
                    break
                time.sleep(0.05)
            assert n_pending == 4, (
                f"lookup_fetch did not produce 4 pending blocks; got {n_pending}"
            )

            # Phase 2: blocks 2-3 match pending
            prefiller.submit_store(
                JobMetadata(
                    job_id=2, keys=keys[2:4],
                    spec=CPULoadStoreSpec(block_ids=[2, 3]),
                )
            )

            # Phase 3: blocks 4-5 match remaining pending
            prefiller.submit_store(
                JobMetadata(
                    job_id=3, keys=keys[4:6],
                    spec=CPULoadStoreSpec(block_ids=[4, 5]),
                )
            )

            # Poll until decoder's load job completes
            deadline = time.time() + 15.0
            load_results: list = []
            store_results: list = []
            while time.time() < deadline:
                store_results += list(prefiller.get_finished())
                load_results = list(decoder.get_finished())
                if load_results:
                    break
                time.sleep(0.05)

            assert len(load_results) == 1, (
                f"Expected 1 load result, got {load_results}"
            )
            assert load_results[0].job_id == 100
            assert load_results[0].success is True

            store_job_ids = {r.job_id for r in store_results}
            assert store_job_ids == {1, 2, 3}, (
                f"Expected store job IDs {{1,2,3}}, got {store_job_ids}"
            )
            assert all(r.success for r in store_results)

            # Data integrity: decoder memory must match prefiller memory
            np.testing.assert_array_equal(
                decoder_arr[:N], prefiller_arr[:N],
                err_msg="Decoder memory does not match Prefiller after NIXL transfer",
            )
        finally:
            decoder.close()
            prefiller.close()

    def test_all_blocks_stored_before_load_data_integrity(self):
        """
        Baseline: all 6 blocks are in Prefiller before submit_load.
        lookup_fetch matches all 6 → single NIXL handle.
        Verify data integrity after transfer completes.
        """
        N = self._NUM_XFER
        prefiller_arr, decoder_arr = _make_real_bufs(
            self._NUM_TOTAL, self._BLOCK_BYTES
        )

        pp, pd = free_port(), free_port()
        prefiller = PDConnector("127.0.0.1", pp)
        decoder   = PDConnector("127.0.0.1", pd)
        try:
            prefiller.set_primary_view(memoryview(prefiller_arr))
            decoder.set_primary_view(memoryview(decoder_arr))

            keys = [make_key(i) for i in range(N)]

            # Store all 6 blocks before the Decoder loads
            prefiller.submit_store(
                JobMetadata(
                    job_id=1, keys=keys,
                    spec=CPULoadStoreSpec(block_ids=list(range(N))),
                )
            )

            # Decoder loads all 6 blocks
            decoder.set_load_peer(100, prefiller._peer_id)
            decoder.submit_load(
                JobMetadata(
                    job_id=100, keys=keys,
                    spec=CPULoadStoreSpec(block_ids=list(range(N))),
                )
            )

            # Poll until load job completes
            deadline = time.time() + 15.0
            load_results: list = []
            while time.time() < deadline:
                prefiller.get_finished()
                load_results = list(decoder.get_finished())
                if load_results:
                    break
                time.sleep(0.05)

            assert len(load_results) == 1, (
                f"Expected 1 load result, got {load_results}"
            )
            assert load_results[0].job_id == 100
            assert load_results[0].success is True

            # Data integrity check
            np.testing.assert_array_equal(
                decoder_arr[:N], prefiller_arr[:N],
                err_msg="Decoder memory does not match Prefiller after NIXL transfer",
            )
        finally:
            decoder.close()
            prefiller.close()


# ---------------------------------------------------------------------------
# Error handling — store and load job timeouts
# ---------------------------------------------------------------------------

class TestErrorHandling:
    """
    Verify that store and load jobs fail gracefully when they exceed their
    timeouts. Timeouts are exercised by monkey-patching _pd_mod constants to
    very small values so tests don't sleep for 30 s.
    """

    def test_store_job_timeout(self, monkeypatch):
        """
        A store job that never gets a matching lookup_fetch times out and
        get_finished() returns JobResult(job_id, success=False).
        The corresponding _block_to_job entries must be cleaned up.
        """
        monkeypatch.setattr(_pd_mod, "_STORE_TIMEOUT_S", 0.05)

        p = free_port()
        prefiller = PDConnector("127.0.0.1", p)
        try:
            prefiller.set_primary_view(make_primary_view())

            keys = [make_key(0), make_key(1)]
            spec = CPULoadStoreSpec(block_ids=[0, 1])
            prefiller.submit_store(JobMetadata(job_id=10, keys=keys, spec=spec))

            # Blocks must be in _block_to_job before timeout
            h0 = get_offload_block_hash(keys[0])
            h1 = get_offload_block_hash(keys[1])
            assert h0 in prefiller._block_to_job
            assert h1 in prefiller._block_to_job

            # Wait for timeout to expire, then poll
            time.sleep(0.1)
            results = list(prefiller.get_finished())

            assert len(results) == 1
            assert results[0].job_id == 10
            assert results[0].success is False

            # _block_to_job must be cleaned up
            assert h0 not in prefiller._block_to_job
            assert h1 not in prefiller._block_to_job
        finally:
            prefiller.close()

    def test_load_job_timeout_sends_abort(self, monkeypatch):
        """
        End-to-end: Decoder load job times out → abort_lookup_fetch sent to
        Prefiller → Prefiller clears _pending_blocks and _fetch_jobs, sends
        abort_ack → Decoder's get_finished() returns JobResult(job_id, False).
        """
        monkeypatch.setattr(_pd_mod, "_LOAD_TIMEOUT_S", 0.05)

        pp, pd = free_port(), free_port()
        prefiller = PDConnector("127.0.0.1", pp)
        decoder   = PDConnector("127.0.0.1", pd)
        try:
            prefiller.set_primary_view(make_primary_view())
            decoder.set_primary_view(make_primary_view())

            keys = [make_key(0), make_key(1)]
            decoder_job_id = 77

            decoder.set_load_peer(decoder_job_id, prefiller._peer_id)
            decoder.submit_load(
                JobMetadata(job_id=decoder_job_id, keys=keys,
                            spec=CPULoadStoreSpec(block_ids=[0, 1]))
            )

            # Wait for lookup_fetch → both blocks go to _pending_blocks
            deadline = time.time() + 3.0
            while time.time() < deadline:
                with prefiller._lock:
                    n = len(prefiller._pending_blocks)
                if n == 2:
                    break
                time.sleep(0.02)
            assert n == 2, "lookup_fetch did not arrive in time"

            # Let the load job timeout expire, then poll decoder
            time.sleep(0.1)
            deadline = time.time() + 5.0
            results: list = []
            while time.time() < deadline:
                prefiller.get_finished()   # keep prefiller message loop alive
                results = list(decoder.get_finished())
                if results:
                    break
                time.sleep(0.05)

            assert len(results) == 1, f"Expected 1 result, got {results}"
            assert results[0].job_id == decoder_job_id
            assert results[0].success is False

            # Prefiller state must be clean
            time.sleep(0.1)   # allow abort_lookup_fetch to arrive
            with prefiller._lock:
                assert not prefiller._pending_blocks
                assert (decoder._peer_id, decoder_job_id) not in prefiller._fetch_jobs
        finally:
            decoder.close()
            prefiller.close()

    def test_load_abort_clears_prefiller_pending(self, monkeypatch):
        """
        After an abort_lookup_fetch, the Prefiller's _pending_blocks for that
        job are removed and the _fetch_jobs entry is gone.
        """
        monkeypatch.setattr(_pd_mod, "_LOAD_TIMEOUT_S", 0.05)

        pp, pd = free_port(), free_port()
        prefiller = PDConnector("127.0.0.1", pp)
        decoder   = PDConnector("127.0.0.1", pd)
        try:
            prefiller.set_primary_view(make_primary_view())
            decoder.set_primary_view(make_primary_view())

            keys = [make_key(i) for i in range(3)]
            decoder_job_id = 55
            decoder.set_load_peer(decoder_job_id, prefiller._peer_id)
            decoder.submit_load(
                JobMetadata(job_id=decoder_job_id, keys=keys,
                            spec=CPULoadStoreSpec(block_ids=[0, 1, 2]))
            )

            # Wait for all 3 blocks to land in _pending_blocks
            deadline = time.time() + 3.0
            while time.time() < deadline:
                with prefiller._lock:
                    n = len(prefiller._pending_blocks)
                if n == 3:
                    break
                time.sleep(0.02)
            assert n == 3

            # Trigger timeout on decoder, then drive the abort round-trip
            time.sleep(0.1)
            deadline = time.time() + 5.0
            while time.time() < deadline:
                decoder.get_finished()
                prefiller.get_finished()
                with prefiller._lock:
                    pending_gone = len(prefiller._pending_blocks) == 0
                if pending_gone:
                    break
                time.sleep(0.05)

            with prefiller._lock:
                assert len(prefiller._pending_blocks) == 0
                assert (decoder._peer_id, decoder_job_id) not in prefiller._fetch_jobs
        finally:
            decoder.close()
            prefiller.close()

    def test_load_abort_ack_timeout(self, monkeypatch):
        """
        Decoder sends abort_lookup_fetch but never receives abort_ack.
        After _ABORT_ACK_TIMEOUT_S elapses, get_finished() returns
        JobResult(job_id, success=False) anyway.
        """
        monkeypatch.setattr(_pd_mod, "_LOAD_TIMEOUT_S", 0.05)
        monkeypatch.setattr(_pd_mod, "_ABORT_ACK_TIMEOUT_S", 0.1)

        p = free_port()
        decoder = PDConnector("127.0.0.1", p)
        try:
            decoder.set_primary_view(make_primary_view())

            # Directly inject a timed-out load job (simulates expired state)
            decoder._load_jobs[99] = _pd_mod._LoadJob(
                job_id=99,
                peer_id="127.0.0.1:9999",
                submitted_at=time.monotonic() - 1.0,  # already expired
            )
            # Inject a fake dealer so _send doesn't raise
            decoder._dealers["127.0.0.1:9999"] = MagicMock()
            decoder._dealers["127.0.0.1:9999"].send = MagicMock()

            # First get_finished: detects timeout, sends abort, moves to _aborting_loads
            decoder.get_finished()
            assert 99 in decoder._aborting_loads

            # Wait for _ABORT_ACK_TIMEOUT_S to elapse
            time.sleep(0.15)

            # Second get_finished: abort_ack timeout fires, fails the job
            results = list(decoder.get_finished())
            assert len(results) == 1
            assert results[0].job_id == 99
            assert results[0].success is False
            assert 99 not in decoder._aborting_loads
        finally:
            decoder.close()

    def test_peer_down_fails_aborting_load(self, monkeypatch):
        """
        If the Prefiller goes down while a load job is in _aborting_loads
        (waiting for abort_ack), _on_peer_down immediately fails the job.
        """
        monkeypatch.setattr(_pd_mod, "_LOAD_TIMEOUT_S", 0.05)

        pp, pd = free_port(), free_port()
        prefiller = PDConnector("127.0.0.1", pp)
        decoder   = PDConnector("127.0.0.1", pd)
        try:
            prefiller.set_primary_view(make_primary_view())
            decoder.set_primary_view(make_primary_view())

            decoder_job_id = 33
            decoder.set_load_peer(decoder_job_id, prefiller._peer_id)
            decoder.submit_load(
                JobMetadata(job_id=decoder_job_id, keys=[make_key(0)],
                            spec=CPULoadStoreSpec(block_ids=[0]))
            )

            # Wait for connection to be established
            deadline = time.time() + 3.0
            while time.time() < deadline:
                with decoder._lock:
                    connected = prefiller._peer_id in decoder._connections
                if connected:
                    break
                time.sleep(0.02)

            # Let the load job timeout expire; move it to _aborting_loads
            time.sleep(0.1)
            decoder.get_finished()
            assert decoder_job_id in decoder._aborting_loads

            # Now prefiller drops off
            prefiller.close()

            # _on_peer_down should fire on decoder and fail the aborting job
            deadline = time.time() + 5.0
            results: list = []
            while time.time() < deadline:
                results = list(decoder.get_finished())
                if results:
                    break
                time.sleep(0.1)

            assert len(results) == 1
            assert results[0].job_id == decoder_job_id
            assert results[0].success is False
            assert decoder_job_id not in decoder._aborting_loads
        finally:
            decoder.close()
