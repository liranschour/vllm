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
from dataclasses import dataclass

import msgspec
import numpy as np
import zmq
import zmq.utils.monitor

from vllm.logger import init_logger
from vllm.v1.kv_offload.abstract import (
    JobId,
    JobMetadata,
    JobResult,
    OffloadKey,
    SecondaryTierManager,
    get_offload_block_hash,
)
from vllm.v1.kv_offload.mediums import CPULoadStoreSpec

logger = init_logger(__name__)

# Lazy NIXL import — optional dependency; None when not installed.
try:
    from nixl._api import nixl_agent as _NixlAgent
    from nixl._api import nixl_agent_config as _NixlAgentConfig
except ImportError:
    _NixlAgent = None  # type: ignore[assignment,misc]
    _NixlAgentConfig = None  # type: ignore[assignment,misc]

# ZMQ ZMTP keep-alive options (milliseconds)
_HEARTBEAT_IVL_MS = 2000
_HEARTBEAT_TIMEOUT_MS = 10000
_HEARTBEAT_TTL_MS = 10000


def _apply_heartbeat(sock: zmq.Socket) -> None:
    """Apply ZMTP heartbeat options to a socket."""
    sock.setsockopt(zmq.HEARTBEAT_IVL, _HEARTBEAT_IVL_MS)
    sock.setsockopt(zmq.HEARTBEAT_TIMEOUT, _HEARTBEAT_TIMEOUT_MS)
    sock.setsockopt(zmq.HEARTBEAT_TTL, _HEARTBEAT_TTL_MS)


@dataclass
class _StoreJob:
    job_id: JobId
    remaining: int


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
        self._kv_blocks: list[memoryview] = []
        self._agent = None
        self._reg = None
        self._local_dlist = None
        self._closed = False
        self._lock = threading.Lock()

        # Connection state (all accessed under _lock)
        self._connections: set[str] = set()
        self._connect_events: dict[str, threading.Event] = {}
        self._remote_dlists: dict[str, object] = {}
        self._peer_nixl_names: dict[str, str] = {}

        # Store job tracking (Step 6)
        self._store_jobs: dict[JobId, _StoreJob] = {}
        self._pending_blocks: dict[bytes, object] = {}
        self._block_to_job: dict[bytes, tuple[JobId, int]] = {}
        self._inflight_xfers: dict[object, tuple[JobId, int]] = {}
        self._finished_jobs: list[JobResult] = []

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

    def _open_channel(self, remote_peer_id: str, host: str, port: int) -> None:
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

    def _ensure_connected(self, peer_id: str) -> None:
        """
        Application-level NIXL handshake with a remote peer (Decoder side).

        Sends a 'connect' message containing this node's NIXL agent metadata
        and compact memory layout parameters (base_addr, num_blocks, block_len).
        Blocks until the Prefiller replies with 'connect_ack' (timeout = 10 s).

        Safe to call concurrently: a per-peer threading.Event serialises
        multiple callers waiting for the same handshake.
        """
        with self._lock:
            if peer_id in self._connections:
                return
            if peer_id in self._connect_events:
                event = self._connect_events[peer_id]
            else:
                event = threading.Event()
                self._connect_events[peer_id] = event

        host, port_str = peer_id.rsplit(":", 1)
        self._open_channel(peer_id, host, int(port_str))

        self._send(peer_id, {
            "type": "connect",
            "peer_id": self._peer_id,
            "agent_metadata": self._agent.get_agent_metadata(),
            "base_addr": int(np.asarray(self._kv_blocks[0]).ctypes.data),
            "num_blocks": len(self._kv_blocks),
            "block_len": self._kv_blocks[0].nbytes,
        })

        if not event.wait(timeout=10.0):
            raise TimeoutError(
                f"PDConnector: connect handshake timed out for {peer_id}"
            )

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
        """Dispatch an incoming control message."""
        msg_type = msg.get("type")

        if msg_type == "connect":
            # Prefiller side: Decoder is establishing a NIXL connection.
            decoder_peer_id = msg["peer_id"]
            local_block_len = self._kv_blocks[0].nbytes
            if msg["block_len"] != local_block_len:
                logger.error(
                    "PDConnector %s: block_len mismatch from %s: "
                    "remote=%d, local=%d — rejecting connect",
                    self._peer_id,
                    decoder_peer_id,
                    msg["block_len"],
                    local_block_len,
                )
                return
            nixl_name = self._agent.add_remote_agent(msg["agent_metadata"])
            base_addr = msg["base_addr"]
            num_blocks = msg["num_blocks"]
            block_len = msg["block_len"]
            block_descs = [
                (base_addr + i * block_len, block_len, 0)
                for i in range(num_blocks)
            ]
            xfer_dlist = self._agent.get_xfer_descs(block_descs, mem_type="DRAM")
            remote_dlist = self._agent.prep_xfer_dlist(nixl_name, xfer_dlist)

            with self._lock:
                self._peer_nixl_names[decoder_peer_id] = nixl_name
                self._remote_dlists[decoder_peer_id] = remote_dlist
                self._connections.add(decoder_peer_id)

            # Open reverse ZMQ channel if not already open.
            with self._lock:
                already_open = decoder_peer_id in self._dealers
            if not already_open:
                dhost, dport = decoder_peer_id.rsplit(":", 1)
                self._open_channel(decoder_peer_id, dhost, int(dport))

            self._send(decoder_peer_id, {
                "type": "connect_ack",
                "peer_id": self._peer_id,
            })

        elif msg_type == "connect_ack":
            # Decoder side: Prefiller has completed the handshake.
            with self._lock:
                self._connections.add(sender_id)
                event = self._connect_events.pop(sender_id, None)
            if event:
                event.set()

        else:
            logger.warning(
                "PDConnector %s: unknown message type %r from %s",
                self._peer_id,
                msg_type,
                sender_id,
            )

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

        Cleans up ZMQ sockets and any NIXL remote state for the peer.
        """
        with self._lock:
            if peer_id not in self._dealers:
                # Already handled (e.g., both monitor and disconnect message fired).
                return
            dealer = self._dealers.pop(peer_id)
            monitor = self._monitor_sockets.pop(peer_id)
            self._connections.discard(peer_id)
            event = self._connect_events.pop(peer_id, None)
            nixl_name = self._peer_nixl_names.pop(peer_id, None)
            dlist = self._remote_dlists.pop(peer_id, None)

        logger.warning("PDConnector %s: peer %s is down", self._peer_id, peer_id)

        monitor.close()
        dealer.close()

        if event:
            event.set()
        if nixl_name and self._agent is not None:
            self._agent.remove_remote_agent(nixl_name)
        if dlist and self._agent is not None:
            self._agent.release_dlist_handle(dlist)

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

        # Release remote NIXL descriptor lists and deregister remote agents.
        with self._lock:
            remote_items = list(self._remote_dlists.items())
            nixl_names = list(self._peer_nixl_names.values())
            self._remote_dlists.clear()
            self._peer_nixl_names.clear()
            self._connections.clear()
        if self._agent is not None:
            for _peer_id, dlist in remote_items:
                self._agent.release_dlist_handle(dlist)
            for nixl_name in nixl_names:
                self._agent.remove_remote_agent(nixl_name)

        # Release local NIXL resources before destroying the ZMQ context.
        if self._local_dlist is not None:
            self._agent.release_dlist_handle(self._local_dlist)
            self._local_dlist = None
        if self._reg is not None:
            self._agent.deregister_memory(self._reg)
            self._reg = None

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
        Store the long-lived memoryview of the primary tier's CPU tensor,
        build a per-block list, and register with NIXL for later transfers.

        Called once by TieringOffloadingManager during initialisation.
        view.shape[0] is num_blocks; each view[i] is a contiguous sub-view
        covering one block's bytes (zero-copy).
        """
        self._primary_view = view
        arr = np.asarray(view)
        self._kv_blocks = [memoryview(arr[i]) for i in range(arr.shape[0])]

        if _NixlAgent is None:
            return

        self._agent = _NixlAgent(self._peer_id, _NixlAgentConfig(backends=["UCX"]))

        # Register the entire contiguous KV buffer as one memory region.
        whole = np.asarray(self._primary_view)
        reg_desc = np.array(
            [[whole.ctypes.data, whole.nbytes, 0]], dtype=np.uint64
        )
        self._reg = self._agent.register_memory(reg_desc, mem_type="DRAM")

        # Build per-block list of (base_addr, nbytes, device_id=0) 3-tuples so
        # that future make_prepped_xfer() calls can address individual blocks by index.
        block_tuples = [
            (int(np.asarray(mv).ctypes.data), mv.nbytes, 0)
            for mv in self._kv_blocks
        ]
        xfer_dlist = self._agent.get_xfer_descs(block_tuples, mem_type="DRAM")
        self._local_dlist = self._agent.prep_xfer_dlist(
            "NIXL_INIT_AGENT", xfer_dlist
        )
        logger.debug(
            "PDConnector %s: registered %d blocks with NIXL",
            self._peer_id,
            len(self._kv_blocks),
        )

    def get_tier_name(self) -> str:
        return "PDConnector"

    def lookup(self, keys: Iterable[OffloadKey]) -> int | None:
        raise NotImplementedError

    def submit_store(self, job_metadata: JobMetadata) -> None:
        job_id = job_metadata.job_id
        keys = list(job_metadata.keys)
        spec = job_metadata.spec

        assert isinstance(spec, CPULoadStoreSpec), (
            f"Expected CPULoadStoreSpec, got {type(spec)}"
        )
        assert len(keys) == len(spec.block_ids), (
            f"Length mismatch: {len(keys)} keys but "
            f"{len(spec.block_ids)} block_ids in spec"
        )

        job = _StoreJob(job_id=job_id, remaining=len(keys))
        self._store_jobs[job_id] = job

        for key, block_idx in zip(keys, spec.block_ids):
            block_hash = get_offload_block_hash(key)
            self._block_to_job[block_hash] = (job_id, int(block_idx))

            if block_hash in self._pending_blocks:
                pass

    def submit_load(self, job_metadata: JobMetadata) -> None:
        raise NotImplementedError

    def get_finished(self) -> Iterable[JobResult]:
        for handle in list(self._inflight_xfers):
            state = self._agent.check_xfer_state(handle)
            if state == "DONE":
                self._agent.release_xfer_handle(handle)
                job_id, num_blocks = self._inflight_xfers.pop(handle)
                job = self._store_jobs[job_id]
                job.remaining -= num_blocks
                if job.remaining == 0:
                    del self._store_jobs[job_id]
                    self._finished_jobs.append(
                        JobResult(job_id=job_id, success=True)
                    )

        result = self._finished_jobs
        self._finished_jobs = []
        return result
