# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Abstract interfaces and data types for the secondary tiering layer.
"""

from abc import ABC, abstractmethod
from collections.abc import Collection, Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

from vllm.v1.kv_offload.base import (
    Locality,
    LookupResult,
    Medium,
    OffloadingEvent,
    OffloadingMetricMetadata,
    OffloadKey,
    ReqContext,
    RequestOffloadingContext,
    ScheduleEndContext,
)

if TYPE_CHECKING:
    from vllm.distributed.kv_transfer.kv_connector.v1.offloading.metrics import (
        OffloadingConnectorStats,
    )
    from vllm.v1.kv_offload.base import OffloadingSpec


# Type alias for job IDs used in async transfer tracking
JobId = int


class TieringOffloadingMetrics:
    """Metric names for TieringOffloadingManager."""

    LOOKUP_SYNC_DELAY = "vllm:kv_offload_tiering_lookup_sync_delay_seconds"
    LOOKUP_ASYNC_DELAY = "vllm:kv_offload_tiering_lookup_async_delay_seconds"
    READ_BYTES = "vllm:kv_offload_tiering_read_bytes"
    READ_TIME = "vllm:kv_offload_tiering_read_time"
    WRITE_BYTES = "vllm:kv_offload_tiering_write_bytes"
    WRITE_TIME = "vllm:kv_offload_tiering_write_time"
    PROMOTION_JOB_FAILURES = "vllm:kv_offload_tiering_promotion_job_failures"
    CASCADE_JOB_FAILURES = "vllm:kv_offload_tiering_cascade_job_failures"
    CHUNK_QUERIES = "vllm:kv_offload_tiering_chunk_queries"
    CHUNK_HITS = "vllm:kv_offload_tiering_chunk_hits"
    PRIMARY_WRITE_USAGE_PERC = "vllm:kv_offload_tiering_primary_write_usage_perc"
    PRIMARY_READ_USAGE_PERC = "vllm:kv_offload_tiering_primary_read_usage_perc"
    PROMOTION_ALLOCATION_FAILURES = (
        "vllm:kv_offload_tiering_promotion_allocation_failures"
    )
    ACTIVE_PROMOTION_JOBS = "vllm:kv_offload_tiering_active_promotion_jobs"
    ACTIVE_CASCADE_JOBS = "vllm:kv_offload_tiering_active_cascade_jobs"


@dataclass
class TransferJob:
    """Metadata for an in-flight async transfer job."""

    job_id: JobId
    keys: Collection[OffloadKey]
    chunk_ids: np.ndarray
    is_promotion: bool
    req_context: ReqContext


@dataclass
class JobResult:
    """Result of an async transfer job."""

    job_id: JobId
    # True if all keys succeeded; False if all or some failed.
    success: bool
    # Only applicable to promotion jobs. On partial failure, identifies the
    # keys that were successfully loaded. None means all keys share the fate
    # indicated by `success`. Must be a subset of the job's original keys.
    successful_keys: Collection[OffloadKey] | None = None
    transfer_time: float | None = None


class ParentManager(ABC):
    """Interface for secondary tiers to call back into the tiering manager.

    Bound once via bind_parent(), and valid for the tier's lifetime. The
    _SecondaryTierFacingParent wrapper implements this, automatically
    excluding the calling tier from fan-out operations.

    Required call sequence for each remote request:
        1. on_new_request(req_context)  — set up per-request state
        2. lookup(key, req_context)     — check chunk availability
           (repeat per chunk)
        3. create_store_job(keys, req_context) — pin chunks and get a
           job handle
        4. on_request_finished(req_context) — clean up per-request state

    Steps 2-3 may be interleaved. Step 4 must be called even if no
    chunks were found, to avoid leaking async lookup state (e.g. in
    the fs tier's AsyncLookupManager).

    Threading: the first call from a tier's own thread claims the tiering
    manager's executor lock, and step_done() releases it. A tier calling
    from a thread of its own MUST bracket its work with step_done(), or the
    scheduler blocks forever. Calls made from within a tier method that the
    manager itself invoked are already under the lock and need no bracket.
    """

    @abstractmethod
    def on_new_request(self, req_context: ReqContext) -> RequestOffloadingContext: ...

    @abstractmethod
    def lookup(self, key: OffloadKey, req_context: ReqContext) -> LookupResult: ...

    @abstractmethod
    def create_store_job(
        self,
        keys: Collection[OffloadKey],
        req_context: ReqContext,
    ) -> TransferJob: ...

    @abstractmethod
    def on_request_finished(self, req_context: ReqContext) -> None: ...

    def step_begin(self) -> None:
        """Begin this executor's turn, blocking until no one else holds one.

        Called by a tier that drives its own thread, before it touches any of
        its own state — not just before its first parent call, since the tier's
        own state is shared with the scheduler too. Idempotent per thread.

        Raises:
            ExecutorLockClosed: the manager is shutting down; the caller should
                stop its thread rather than retry.
        """
        return

    def step_done(self) -> None:
        """End this executor's turn: flush deferred work and release the lock.

        Must pair with step_begin(); a tier that fails to call it blocks the
        scheduler. A no-op when this thread does not own the turn, so it is also
        safe on a sweep the manager itself drove.
        """
        return


class SecondaryTierManager(ABC):
    """
    Abstract interface for managing a single non-primary offloading tier.

    Secondary tiers cannot directly access GPU memory. All data transfers
    must go through the CPU (primary) tier:
      - Store: GPU → CPU (primary) → secondary  (cascade)
      - Load:  secondary → CPU (primary) → GPU  (promotion)

    IMPORTANT: All methods run in the Scheduler process and must be
    lightweight and non-blocking. submit_load() and submit_store() submit
    async jobs; get_finished_jobs() polls for completion.

    Threading: the tiering manager guarantees a single executor inside itself
    and its tiers at any moment, so these methods never run concurrently with
    each other. They are called on the scheduler thread; a tier that runs its
    own thread (see bind_parent() and stop_executor()) also executes under that
    same guarantee, which means "not concurrent" rather than "always the
    scheduler thread". Anything a tier does under it delays the scheduler, so
    keep it bounded and never block on I/O.
    """

    medium: ClassVar[Medium | None] = None

    def __init__(
        self,
        offloading_spec: "OffloadingSpec",
        primary_kv_view: memoryview,
        tier_type: str,
    ) -> None:
        """
        Args:
            offloading_spec: Offloading configuration.
            primary_kv_view: Memoryview of the primary tier's CPU KV cache.
            tier_type: Tier type identifier, set by SecondaryTierFactory
                from the registered tier type.
        """
        self._offloading_spec = offloading_spec
        self._primary_kv_view: memoryview = primary_kv_view
        self.tier_type = tier_type
        self.locality: Locality | None = None

    @abstractmethod
    def lookup(self, key: OffloadKey, req_context: ReqContext) -> LookupResult:
        """
        Check whether a chunk exists in this secondary tier.

        Args:
            key: Offload key to look up.
            req_context: per-request context (e.g. kv_transfer_params).

        Returns:
            HIT if the chunk is present and ready,
            MISS if not found,
            or RETRY if the chunk is being transferred (retry later).
        """
        pass

    @abstractmethod
    def submit_store(self, job_metadata: TransferJob) -> None:
        """
        Submit an async job to store chunks from the primary tier to this
        secondary tier.

        This method must be lightweight and non-blocking: allocate metadata
        and submit the transfer, but do NOT perform the data copy on the
        calling thread.

        Preconditions (guaranteed by the framework):
          - ``job_metadata.chunk_ids`` are valid primary-tier slots, pinned
            (ref-counted) for the duration of the transfer.

        The implementation is responsible for:
          1. Filtering out chunks already present in this tier
          2. Evicting chunks if capacity is needed
          3. Allocating space in this tier
          4. Submitting the async transfer (read from primary via chunk_ids)

        Report completion via ``get_finished_jobs()``.

        Args:
            job_metadata: Job metadata including job_id, keys, and chunk_ids
                          identifying the primary-tier slots to read from.
        """
        pass

    @abstractmethod
    def submit_load(self, job_metadata: TransferJob) -> None:
        """
        Submit an async job to load chunks from this secondary tier to the
        primary tier.

        This method must be lightweight and non-blocking: mark chunks as
        in-flight and submit the transfer, but do NOT perform the data copy
        on the calling thread.

        Preconditions (guaranteed by the framework):
          - ``job_metadata.chunk_ids`` are allocated primary-tier slots
            ready to receive data.

        The implementation must copy data from this tier into the
        primary-tier slots identified by ``chunk_ids``.

        Report completion via ``get_finished_jobs()``.

        Args:
            job_metadata: Job metadata including job_id, keys, and chunk_ids
                          identifying the primary-tier slots to write into.
        """
        pass

    @abstractmethod
    def get_finished_jobs(self) -> Iterable[JobResult]:
        """
        Return all jobs (loads and stores) that completed since the last call.

        The framework uses these results to release resources and finalize
        transfers.

        Returns:
            Iterable of JobResult objects for jobs finished since the
            last call.
        """
        pass

    def has_pending_work(self) -> bool:
        """Whether this tier needs the engine to keep stepping.

        While True, on_schedule_end() and get_finished_jobs() continue
        to be called even when no requests are scheduled.
        """
        return False

    def take_events(self) -> Iterable[OffloadingEvent]:
        """Take KV events for storage state owned by this tier."""
        return ()

    def touch(self, keys: Collection[OffloadKey], req_context: ReqContext):
        """
        Mark chunks as recently used for eviction policy.

        Args:
            keys: Offload keys to mark as recently used.
            req_context: Per-request context.
        """
        return

    @abstractmethod
    def on_new_request(self, req_context: ReqContext) -> RequestOffloadingContext:
        """
        Called when a new request is first seen by the scheduler.

        Returns a RequestOffloadingContext expressing this tier's preference
        for how chunks should be offloaded for this request.

        Args:
            req_context: Per-request context.
        """
        pass

    def on_request_finished(self, req_context: ReqContext) -> None:
        """
        Called when a request has finished.

        By the time this is called, all per-request calls for this request
        (submit_store, submit_load, touch) have already been issued, and none
        will follow. Note this does NOT imply the tier's transfers have
        completed: jobs already submitted may still be in flight and will
        report via get_finished_jobs(). This is the right place to release
        per-request bookkeeping.

        Args:
            req_context: per-request context.
        """
        return

    def bind_parent(self, parent: ParentManager) -> None:
        """Receive a handle for calling back into the tiering manager.

        Called once during manager construction. The handle is valid for the
        tier's lifetime, so a tier that serves remotely-originated requests may
        use it from its own thread, bracketing each sweep with
        ``parent.step_done()``.

        Bind only — do not start threads here. This runs before the KV cache is
        configured, so anything started would also run through the profile run;
        start lazily on the first call the manager makes instead.

        Tiers that never call back into the manager leave this as a no-op.
        """
        return

    def stop_executor(self) -> None:
        """Stop any thread this tier started, and join it.

        Called by the manager before it takes its executor lock for shutdown,
        because a tier thread parked waiting for that lock could never be
        joined from inside it. Must be idempotent.
        """
        return

    def on_schedule_end(self, context: ScheduleEndContext) -> None:
        """Called once at the end of each scheduler step.

        Args:
            context: Per-step context from the scheduler.
        """
        return

    @abstractmethod
    def drain_jobs(self) -> None:
        """Block until every submitted load/store job has completed or failed.

        After this returns, no tier I/O is touching the primary memoryview,
        and every submitted job's result is available from `get_finished_jobs()`
        (yielded by a prior call or queued for the next one). Used by
        `TieringOffloadingManager.reset_cache` to release primary slots
        without racing with in-flight transfers.

        Implementations must not abort a mid-flight transfer: a partial copy
        would corrupt either the primary memoryview or the secondary backing
        store. Queued (not-yet-started) transfers may be cancelled, but their
        failure result must still appear in `get_finished_jobs()`.
        """
        pass

    def shutdown(self) -> None:
        """Release resources held by this tier (threads, connections, etc.)."""
        return

    @classmethod
    def build_metric_definitions(
        cls, extra_config: dict[str, Any]
    ) -> dict[str, OffloadingMetricMetadata]:
        """Return Prometheus metric definitions emitted by this tier."""
        return {}

    def get_stats(self) -> "OffloadingConnectorStats | None":
        """Return and reset metric observations collected by this tier."""
        return None
