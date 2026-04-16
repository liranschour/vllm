# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Unit tests for PDConnector.

Step 1: skeleton — tier name, primary view, TieringOffloadingManager wiring,
        unimplemented method stubs.
Step 2: ZMQ control channel — bidirectional messaging, clean-disconnect
        notification via _on_peer_down.
"""

import socket
import threading
import time

import pytest
import torch

from vllm.v1.kv_offload.abstract import JobMetadata, OffloadKey
from vllm.v1.kv_offload.secondary_tiers.pd_connector import PDConnector
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

    def get_primary_kv_tensors(self):
        return torch.zeros((16, 8), dtype=torch.float32)


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

    def test_lookup_raises_not_implemented(self):
        p = free_port()
        c = PDConnector("127.0.0.1", p)
        try:
            with pytest.raises(NotImplementedError):
                c.lookup([make_key(0)])
        finally:
            c.close()

    def test_submit_store_raises_not_implemented(self):
        p = free_port()
        c = PDConnector("127.0.0.1", p)
        try:
            with pytest.raises(NotImplementedError):
                c.submit_store(JobMetadata(job_id=0, keys=[], spec=None))
        finally:
            c.close()

    def test_submit_load_raises_not_implemented(self):
        p = free_port()
        c = PDConnector("127.0.0.1", p)
        try:
            with pytest.raises(NotImplementedError):
                c.submit_load(JobMetadata(job_id=0, keys=[], spec=None))
        finally:
            c.close()

    def test_get_finished_raises_not_implemented(self):
        p = free_port()
        c = PDConnector("127.0.0.1", p)
        try:
            with pytest.raises(NotImplementedError):
                list(c.get_finished())
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
            a._connect(b._peer_id, "127.0.0.1", pb)
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
            a._connect(b._peer_id, "127.0.0.1", pb)
            b._connect(a._peer_id, "127.0.0.1", pa)
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
            a._connect(b._peer_id, "127.0.0.1", pb)
            b._connect(a._peer_id, "127.0.0.1", pa)
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
            a._connect(b._peer_id, "127.0.0.1", pb)
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
