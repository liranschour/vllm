# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
ExecutorLock: serializes all execution inside the tiering manager and its tiers.

The tiering manager and every tier under it form one body of state with no
internal locking. Historically that was safe because only the scheduler thread
ever entered it. A tier that drives its own thread (see
``SecondaryTierManager.bind_parent``) breaks that assumption, so this lock
restores it as an explicit guarantee: **at most one executor inside the subtree
at any moment**.

Two access patterns share one ``RLock``:

* ``with lock:`` — a lexical, re-entrant hold for the duration of one call.
  Exception safe, and beyond a depth bump it costs nothing when a turn is
  already open.

* ``open_turn()`` / ``close_turn()`` — one *extra* acquisition, held across
  several calls, for work whose steps must not interleave with another
  executor. Two such windows exist, and both are load-bearing:

  - the scheduler's ``lookup()`` -> ``prepare_load()`` window. ``lookup()``
    reports HIT while the chunk still sits at ``ref_cnt == 0``; only
    ``prepare_load()`` pins it. A tier thread calling ``parent.lookup`` in
    between can reach ``primary_tier.prepare_write()`` and evict that very
    chunk.
  - a tier thread's ``parent.lookup`` -> ``parent.create_store_job`` window,
    for the same reason in the other direction.

Nested ``with`` blocks never drop the depth below the turn's one acquisition,
so a turn survives them.

``_turn_owner`` is read outside the lock. That is safe because it is only ever
written to a thread's own id while that thread holds the lock, and cleared
before releasing: a non-owner can read ``None`` or some other thread's id, never
its own, so it always falls through to a real acquire. Reference loads and
stores are indivisible under the GIL, and remain so on a free-threaded build.
"""

import threading

from vllm.logger import init_logger

logger = init_logger(__name__)

# Nothing in the engine loop times out, so a lost release would otherwise be an
# unrecoverable silent hang: both FutureWrapper.result(timeout) and
# AsyncOutputFuture.result(timeout) raise "timeout not implemented". Bound every
# acquire so the victim reports who is holding the lock instead of wedging.
_ACQUIRE_TIMEOUT_S = 30.0


class ExecutorLockTimeout(RuntimeError):
    """Raised when the executor lock could not be acquired in time."""


class ExecutorLockClosed(RuntimeError):
    """Raised when a turn is requested after the lock has been closed."""


class ExecutorLock:
    def __init__(self, acquire_timeout_s: float = _ACQUIRE_TIMEOUT_S) -> None:
        self._lock = threading.RLock()
        self._acquire_timeout_s = acquire_timeout_s
        self._turn_owner: int | None = None
        self._holder: str | None = None
        self._closed = False

    def __enter__(self) -> "ExecutorLock":
        self._acquire()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self._lock.release()

    def _acquire(self) -> None:
        if not self._lock.acquire(timeout=self._acquire_timeout_s):
            raise ExecutorLockTimeout(
                f"could not acquire the tiering executor lock within "
                f"{self._acquire_timeout_s}s; held by {self._holder}"
            )
        self._holder = threading.current_thread().name

    def open_turn(self) -> None:
        """Claim a hold spanning several calls. Idempotent per thread."""
        if self._turn_owner == threading.get_ident():
            return
        if self._closed:
            raise ExecutorLockClosed(
                "tiering executor lock is closed; the manager is shutting down"
            )
        self._acquire()
        self._turn_owner = threading.get_ident()

    def close_turn(self) -> None:
        """Release this thread's turn. A no-op when it does not own one."""
        if self._turn_owner != threading.get_ident():
            return
        self._turn_owner = None
        self._lock.release()

    def owns_turn(self) -> bool:
        return self._turn_owner == threading.get_ident()

    def close(self) -> None:
        """Refuse further turns, so tier threads fail fast during shutdown.

        Scoped ``with`` access still works, letting the manager run its own
        teardown after the tier threads have been stopped and joined.
        """
        self._closed = True
