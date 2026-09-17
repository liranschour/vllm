# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for ExecutorLock.

The lock is what lets a secondary tier drive its own thread while the tiering
manager and its tiers stay free of internal locking. Two properties matter and
are what these tests pin down: a turn survives the nested scoped blocks that the
manager's own call paths create, and nothing can leave the lock held after it
returns, since no timeout anywhere in the engine loop would surface that.
"""

import threading

import pytest

from vllm.v1.kv_offload.tiering.executor_lock import (
    ExecutorLock,
    ExecutorLockClosed,
    ExecutorLockTimeout,
)


class TestScoped:
    def test_reentrant(self):
        lock = ExecutorLock()
        with lock, lock, lock:
            assert True
        # Fully released: another thread can take a turn.
        _assert_free(lock)

    def test_releases_on_exception(self):
        lock = ExecutorLock()
        with pytest.raises(ValueError), lock:
            raise ValueError("boom")
        _assert_free(lock)


class TestTurn:
    def test_idempotent_per_thread(self):
        """Repeated open_turn() must not stack acquisitions.

        The manager's hot path calls it on every lookup(), thousands of times
        per step, and a single close_turn() has to undo all of them.
        """
        lock = ExecutorLock()
        for _ in range(5):
            lock.open_turn()
        assert lock.owns_turn()
        lock.close_turn()
        assert not lock.owns_turn()
        _assert_free(lock)

    def test_survives_nested_scoped_blocks(self):
        """A scoped block inside a turn must not drop the turn's hold.

        This is the manager's real shape: a turn opened by lookup(), then
        further manager methods entered under it.
        """
        lock = ExecutorLock()
        lock.open_turn()
        with lock:
            pass
        assert lock.owns_turn()
        _assert_held(lock)
        lock.close_turn()
        _assert_free(lock)

    def test_close_turn_is_owner_only(self):
        """A thread that owns no turn must not release someone else's."""
        lock = ExecutorLock()
        lock.open_turn()

        def other():
            lock.close_turn()  # not the owner: must be a no-op

        t = threading.Thread(target=other)
        t.start()
        t.join()

        assert lock.owns_turn()
        _assert_held(lock)
        lock.close_turn()

    def test_close_turn_without_turn_is_noop(self):
        ExecutorLock().close_turn()


class TestMutualExclusion:
    def test_turn_excludes_other_thread(self):
        lock = ExecutorLock()
        entered = threading.Event()
        lock.open_turn()

        def other():
            lock.open_turn()
            entered.set()
            lock.close_turn()

        t = threading.Thread(target=other)
        t.start()
        assert not entered.wait(0.2), "second thread entered while a turn was open"
        lock.close_turn()
        assert entered.wait(5.0), "second thread never got its turn"
        t.join()

    def test_scoped_excludes_other_thread(self):
        lock = ExecutorLock()
        entered = threading.Event()

        def other():
            lock.open_turn()
            entered.set()
            lock.close_turn()

        with lock:
            t = threading.Thread(target=other)
            t.start()
            assert not entered.wait(0.2), "second thread entered a held lock"
        assert entered.wait(5.0), "second thread never acquired after release"
        t.join()


class TestFailFast:
    def test_timeout_raises_instead_of_hanging(self):
        """A lost release must surface as an error, not a wedged engine.

        Nothing in the engine loop times out — both FutureWrapper.result and
        AsyncOutputFuture.result reject a timeout argument — so a silent block
        here would be unrecoverable and undiagnosable.
        """
        lock = ExecutorLock(acquire_timeout_s=0.05)
        blocked = threading.Event()
        raised: list[BaseException] = []

        def holder():
            lock.open_turn()
            blocked.wait(5.0)
            lock.close_turn()

        t = threading.Thread(target=holder, name="lock-holder")
        t.start()
        try:
            while not lock.owns_turn() and t.is_alive():
                if lock._turn_owner is not None:
                    break
            try:
                lock.open_turn()
            except ExecutorLockTimeout as exc:
                raised.append(exc)
        finally:
            blocked.set()
            t.join()

        assert raised, "expected ExecutorLockTimeout"
        # The message must name the holder, so the log points at the culprit.
        assert "lock-holder" in str(raised[0])

    def test_closed_lock_refuses_turns_but_allows_scoped(self):
        """close() locks tier threads out while letting the manager tear down.

        shutdown() stops tier threads and then does its own work under the
        lock, so scoped access has to keep working after close().
        """
        lock = ExecutorLock()
        lock.close()

        with pytest.raises(ExecutorLockClosed):
            lock.open_turn()

        with lock:
            pass


def _assert_free(lock: ExecutorLock) -> None:
    """Assert the lock is fully released, from a thread that does not own it."""
    got = threading.Event()

    def probe():
        lock.open_turn()
        got.set()
        lock.close_turn()

    t = threading.Thread(target=probe)
    t.start()
    t.join(timeout=5.0)
    assert got.is_set(), "lock is still held"


def _assert_held(lock: ExecutorLock) -> None:
    """Assert another thread cannot acquire the lock right now.

    Probes the underlying lock with a bounded acquire rather than parking a
    thread in open_turn(): such a thread would still be waiting when the test
    later releases, and would then take the turn and never give it back.
    """
    acquired: list[bool] = []

    def probe():
        ok = lock._lock.acquire(timeout=0.2)
        acquired.append(ok)
        if ok:
            lock._lock.release()

    t = threading.Thread(target=probe)
    t.start()
    t.join(timeout=5.0)
    assert acquired == [False], "lock was not held"
