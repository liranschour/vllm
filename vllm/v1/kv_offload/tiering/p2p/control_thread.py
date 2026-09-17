# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
P2PControlThread: drives the P2P control plane independently of engine steps.

Why this exists
---------------
Every P2P control hop used to be serviced from a connector scheduling hook, so
each one waited for a scheduler-step boundary. On a saturated rank iterating
about once per second that put a full step between a peer's LookupMsg arriving
and its answer, another between the LookupRespMsg and the FetchMsg, and one more
per handshake hop — measured at 0.45-0.61 s each. The work itself is CPU-side and
takes microseconds; it was only ever waiting for a turn.

This thread takes that turn as soon as the scheduler frees it, which is for the
whole model-execution wait, so an inbound message is answered in about a
millisecond instead of half a second.

Threading contract
------------------
The tier's state is shared with the scheduler, so *everything* the sweep touches
must run inside a turn — not merely the calls back into the tiering manager.
That specifically includes the ZeroMQ sockets: the scheduler reaches the same
sockets through the drain path, and ``zmq.Socket`` is not thread-safe, so
polling one here while the scheduler receives on it aborts libzmq rather than
raising. Hence the wait between sweeps is on a ``threading.Event``, never on a
socket, and the sweep itself decides how long the next wait should be.
"""

import threading
from collections.abc import Callable

from vllm.logger import init_logger
from vllm.v1.kv_offload.tiering.base import ParentManager
from vllm.v1.kv_offload.tiering.executor_lock import ExecutorLockClosed

logger = init_logger(__name__)

# Long enough that a wedged sweep is reported rather than silently waited on,
# short enough not to stall engine teardown.
_JOIN_TIMEOUT_S = 5.0

# First wait, and the wait used after a failed sweep, before the sweep starts
# choosing for itself.
_INITIAL_WAIT_S = 0.001


class P2PControlThread:
    """Runs a bounded sweep of the P2P control plane, once per wakeup.

    Args:
        sweep: Performs one sweep and returns how long to wait before the next
            one, in seconds. Called inside a turn, so it must be bounded: the
            scheduler blocks behind it.
        parent: Handle used to open and close each turn.
        name: Thread name, for logs and for the lock's contention diagnostics.
    """

    def __init__(
        self,
        sweep: Callable[[], float],
        parent: ParentManager,
        name: str,
    ) -> None:
        self._sweep = sweep
        self._parent = parent
        self._stop = threading.Event()
        self._wakeup = threading.Event()
        self._thread = threading.Thread(target=self._run, name=name, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def wake(self) -> None:
        """Cut the current wait short, e.g. once the scheduler frees the lock."""
        self._wakeup.set()

    def stop(self) -> None:
        """Stop the thread and join it. Idempotent."""
        self._stop.set()
        self._wakeup.set()
        if self._thread.is_alive():
            self._thread.join(timeout=_JOIN_TIMEOUT_S)
            if self._thread.is_alive():
                logger.error(
                    "%s did not exit within %.1fs; its transports will be closed "
                    "while it may still be using them",
                    self._thread.name,
                    _JOIN_TIMEOUT_S,
                )

    def is_alive(self) -> bool:
        return self._thread.is_alive()

    def _run(self) -> None:
        wait_s = _INITIAL_WAIT_S
        while not self._stop.is_set():
            self._wakeup.wait(wait_s)
            self._wakeup.clear()
            if self._stop.is_set():
                return
            try:
                self._parent.step_begin()
                wait_s = self._sweep()
            except ExecutorLockClosed:
                # The manager is shutting down and will not hand out turns.
                return
            except Exception:
                # Never let one bad sweep kill the control plane: without this
                # thread the tier goes permanently deaf to every peer.
                logger.exception("P2P control sweep failed; continuing")
                wait_s = _INITIAL_WAIT_S
            finally:
                self._parent.step_done()
