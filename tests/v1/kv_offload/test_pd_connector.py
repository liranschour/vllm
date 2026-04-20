# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Unit tests for PDConnector.

Step 1: skeleton — tier name, primary view, TieringOffloadingManager wiring,
        unimplemented method stubs.
Step 2: ZMQ control channel — bidirectional messaging, clean-disconnect
        notification via _on_peer_down.
Step 6: submit_store() job tracking and get_finished().
Step 7: submit_load(), lookup_fetch handling, pending_blocks matching.
"""

import socket
import threading
import time
from unittest.mock import MagicMock

import pytest
import torch

from vllm.v1.kv_offload.abstract import (
    JobMetadata,
    OffloadKey,
    get_offload_block_hash,
)
from vllm.v1.kv_offload.mediums import CPULoadStoreSpec
from vllm.v1.kv_offload.secondary_tiers.pd_connector import (
    PDConnector,
    _PendingBlock,
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


class _MinimalPrimary(CPUPrimaryTierOffloadingManager):
    def __init__(self):
        pass

    def get_primary_kv_tensor(self):
        return torch.zeros((16, 8), dtype=torch.int8)


# ---------------------------------------------------------------------------
# Step 1: Skeleton tests
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
# Step 2: ZMQ control channel tests
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
# Step 6: submit_store() and get_finished() tests
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
# Step 7: submit_load() and lookup_fetch tests
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
                peer_id="127.0.0.1:9999", remote_block_idx=3
            )
            prefiller._pending_blocks[h1] = _PendingBlock(
                peer_id="127.0.0.1:9999", remote_block_idx=5
            )
            prefiller._pending_blocks[h2] = _PendingBlock(
                peer_id="127.0.0.1:9999", remote_block_idx=7
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
# Step 4: NIXL registration tests
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
# Step 5: Connection establishment tests
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
