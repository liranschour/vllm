# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
PDConnector: Secondary tier implementation for PD disaggregation.

Handles KV cache transfer between Prefiller and Decoder nodes via NIXL.
The Prefiller stores blocks into the primary CPU tier and cascades them
to the Decoder. The Decoder promotes blocks from the Prefiller's CPU tier
into its own primary CPU tier via a NIXL WRITE transfer.
"""

import threading
import time
from collections.abc import Iterable

import msgspec
import zmq
import zmq.utils.monitor

from vllm.logger import init_logger
from vllm.v1.kv_offload.abstract import (
    JobMetadata,
    JobResult,
    OffloadKey,
    SecondaryTierManager,
)

logger = init_logger(__name__)

# ZMQ ZMTP keep-alive options (milliseconds)
_HEARTBEAT_IVL_MS = 2000
_HEARTBEAT_TIMEOUT_MS = 10000
_HEARTBEAT_TTL_MS = 10000


def _apply_heartbeat(sock: zmq.Socket) -> None:
    """Apply ZMTP heartbeat options to a socket."""
    sock.setsockopt(zmq.HEARTBEAT_IVL, _HEARTBEAT_IVL_MS)
    sock.setsockopt(zmq.HEARTBEAT_TIMEOUT, _HEARTBEAT_TIMEOUT_MS)
    sock.setsockopt(zmq.HEARTBEAT_TTL, _HEARTBEAT_TTL_MS)


class PDConnector(SecondaryTierManager):
    """
    Secondary tier for PD (Prefill-Decode) disaggregation.

    Each PDConnector instance runs a bidirectional ZMQ control channel:
      - One ROUTER socket bound to (host, port) — accepts connections from
        any number of remote peers.
      - One DEALER socket per remote peer — used to send messages to that
        peer's ROUTER.

    Keep-alive is handled by ZMQ's built-in ZMTP heartbeat. A background
    _monitor_loop thread watches each DEALER socket's ZMQ monitor for
    EVENT_DISCONNECTED and calls _on_peer_down(peer_id) when triggered.
    A clean disconnect sends {"type": "disconnect"} before closing so the
    remote side is notified immediately.

    The local peer identity is f"{host}:{port}". This identity is set on
    every outbound DEALER socket so the remote ROUTER can identify the sender.

    Control message dispatch (lookup_fetch, transfer_done, etc.) is added
    in later steps via _handle_message().
    """

    def __init__(self, host: str, port: int) -> None:
        self._peer_id = f"{host}:{port}"
        self._primary_view: memoryview | None = None
        self._closed = False
        self._lock = threading.Lock()

        self._zmq_ctx = zmq.Context()

        # ROUTER: single listener for all incoming peers
        self._router: zmq.Socket = self._zmq_ctx.socket(zmq.ROUTER)
        _apply_heartbeat(self._router)
        self._router.bind(f"tcp://{host}:{port}")

        # DEALER sockets and their ZMQ monitor PAIR sockets, keyed by remote peer_id
        self._dealers: dict[str, zmq.Socket] = {}
        self._monitor_sockets: dict[str, zmq.Socket] = {}

        self._listener_thread = threading.Thread(
            target=self._listener_loop,
            daemon=True,
            name=f"pd-ctrl-listener-{self._peer_id}",
        )
        self._listener_thread.start()

        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True,
            name=f"pd-ctrl-monitor-{self._peer_id}",
        )
        self._monitor_thread.start()

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    def _connect(self, remote_peer_id: str, host: str, port: int) -> None:
        """
        Open a DEALER socket to remote_peer_id's ROUTER and begin monitoring it.

        The DEALER's ZMQ identity is set to this connector's own peer_id so
        the remote ROUTER can identify the sender of every message.
        """
        dealer = self._zmq_ctx.socket(zmq.DEALER)
        _apply_heartbeat(dealer)
        dealer.identity = self._peer_id.encode()

        # Attach a ZMQ socket monitor *before* connecting so no events are missed.
        # Sanitize peer_id for use in an inproc address.
        safe_id = remote_peer_id.replace(":", "-").replace("/", "-")
        monitor_addr = f"inproc://pd-monitor-{safe_id}"
        dealer.monitor(monitor_addr, zmq.EVENT_DISCONNECTED)

        monitor_sock = self._zmq_ctx.socket(zmq.PAIR)
        monitor_sock.connect(monitor_addr)

        dealer.connect(f"tcp://{host}:{port}")

        with self._lock:
            self._dealers[remote_peer_id] = dealer
            self._monitor_sockets[remote_peer_id] = monitor_sock

    # ------------------------------------------------------------------
    # Messaging
    # ------------------------------------------------------------------

    def _send(self, remote_peer_id: str, msg: dict) -> None:
        """Send a msgpack-encoded message to a peer via its DEALER socket."""
        data = msgspec.msgpack.encode(msg)
        with self._lock:
            dealer = self._dealers.get(remote_peer_id)
        if dealer is None:
            raise RuntimeError(
                f"PDConnector: no connection to peer {remote_peer_id!r}"
            )
        dealer.send(data)

    def _handle_message(self, sender_id: str, msg: dict) -> None:
        """
        Dispatch an incoming control message.
        Extended in later steps (lookup_fetch, lookup_ack/nack, transfer_done).
        """
        pass

    # ------------------------------------------------------------------
    # Background threads
    # ------------------------------------------------------------------

    def _listener_loop(self) -> None:
        """Read messages from the ROUTER socket and dispatch them."""
        poller = zmq.Poller()
        poller.register(self._router, zmq.POLLIN)

        while not self._closed:
            try:
                ready = dict(poller.poll(timeout=100))
            except zmq.ZMQError:
                break
            if self._router not in ready:
                continue

            try:
                frames = self._router.recv_multipart()
            except zmq.ZMQError:
                break

            if len(frames) != 2:
                logger.warning(
                    "PDConnector %s: unexpected frame count %d, dropping",
                    self._peer_id,
                    len(frames),
                )
                continue

            identity, data = frames
            sender_id = identity.decode()

            try:
                msg = msgspec.msgpack.decode(data)
            except Exception as exc:
                logger.warning(
                    "PDConnector %s: failed to decode message from %s: %s",
                    self._peer_id,
                    sender_id,
                    exc,
                )
                continue

            if msg.get("type") == "disconnect":
                self._on_peer_down(sender_id)
            else:
                self._handle_message(sender_id, msg)

    def _monitor_loop(self) -> None:
        """
        Poll ZMQ monitor sockets on all DEALER connections.

        Calls _on_peer_down(peer_id) when EVENT_DISCONNECTED is detected,
        indicating the remote peer is unreachable.
        """
        while not self._closed:
            with self._lock:
                monitors = dict(self._monitor_sockets)

            if not monitors:
                time.sleep(0.1)
                continue

            poller = zmq.Poller()
            for sock in monitors.values():
                poller.register(sock, zmq.POLLIN)

            try:
                ready = dict(poller.poll(timeout=100))
            except zmq.ZMQError:
                break

            for peer_id, sock in monitors.items():
                if sock not in ready:
                    continue
                try:
                    event = zmq.utils.monitor.recv_monitor_message(
                        sock, zmq.NOBLOCK
                    )
                except zmq.Again:
                    continue
                except zmq.ZMQError:
                    # Socket was closed between the poll and the recv.
                    continue

                if event["event"] == zmq.EVENT_DISCONNECTED:
                    logger.info(
                        "PDConnector %s: ZMQ heartbeat detected peer %s is down",
                        self._peer_id,
                        peer_id,
                    )
                    self._on_peer_down(peer_id)

    # ------------------------------------------------------------------
    # Peer liveness
    # ------------------------------------------------------------------

    def _on_peer_down(self, peer_id: str) -> None:
        """
        Called when a peer connection is lost (clean disconnect or heartbeat timeout).

        Cleans up the DEALER and monitor sockets for the peer.
        Job cancellation logic is added in later steps.
        """
        with self._lock:
            if peer_id not in self._dealers:
                # Already handled (e.g., both monitor and disconnect message fired).
                return
            dealer = self._dealers.pop(peer_id)
            monitor = self._monitor_sockets.pop(peer_id)

        logger.warning("PDConnector %s: peer %s is down", self._peer_id, peer_id)

        monitor.close()
        dealer.close()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """
        Gracefully shut down the control channel.

        Sends {"type": "disconnect"} to each connected peer so they are
        notified immediately, then tears down all sockets.
        """
        if self._closed:
            return
        self._closed = True

        # Notify all peers of clean disconnect before closing sockets.
        with self._lock:
            peers = list(self._dealers.keys())
        for peer_id in peers:
            try:
                self._send(peer_id, {"type": "disconnect"})
            except Exception:
                pass

        # Tear down all DEALER sockets and their monitors with linger=0 so
        # pending messages are discarded immediately and no blocking occurs.
        with self._lock:
            for dealer in self._dealers.values():
                dealer.setsockopt(zmq.LINGER, 0)
                dealer.close()
            self._dealers.clear()
            for monitor in self._monitor_sockets.values():
                monitor.setsockopt(zmq.LINGER, 0)
                monitor.close()
            self._monitor_sockets.clear()

        self._router.setsockopt(zmq.LINGER, 0)
        self._router.close()

        # destroy() with linger=0 terminates the context immediately and
        # causes any blocking ZMQ calls in background threads to raise ZMQError,
        # allowing them to exit cleanly.
        self._zmq_ctx.destroy(linger=0)
        self._listener_thread.join(timeout=2.0)
        self._monitor_thread.join(timeout=2.0)

    # ------------------------------------------------------------------
    # SecondaryTierManager interface
    # ------------------------------------------------------------------

    def set_primary_view(self, view: memoryview) -> None:
        """
        Store the long-lived memoryview of the primary tier's CPU tensor.

        Called once by TieringOffloadingManager during initialisation.
        The view is used by submit_store and submit_load to read/write
        primary tier CPU memory directly (zero-copy).
        """
        self._primary_view = view

    def get_tier_name(self) -> str:
        return "PDConnector"

    def lookup(self, keys: Iterable[OffloadKey]) -> int | None:
        raise NotImplementedError

    def submit_store(self, job_metadata: JobMetadata) -> None:
        raise NotImplementedError

    def submit_load(self, job_metadata: JobMetadata) -> None:
        raise NotImplementedError

    def get_finished(self) -> Iterable[JobResult]:
        raise NotImplementedError
