# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
PDConnector: Secondary tier implementation for PD disaggregation.

Handles KV cache transfer between Prefiller and Decoder nodes via NIXL.
The Prefiller stores blocks into the primary CPU tier and cascades them
to the Decoder. The Decoder promotes blocks from the Prefiller's CPU tier
into its own primary CPU tier via a NIXL WRITE transfer.
"""

from collections.abc import Iterable

from vllm.v1.kv_offload.abstract import (
    JobMetadata,
    JobResult,
    OffloadKey,
    SecondaryTierManager,
)


class PDConnector(SecondaryTierManager):
    """
    Secondary tier for PD (Prefill-Decode) disaggregation.

    On the Prefiller side:
      - submit_store() registers block descriptors so they can be served
        when a lookup_fetch control message arrives from the Decoder.

    On the Decoder side:
      - submit_load() connects to the Prefiller peer, sends a
        CTRL:lookup_fetch message, and triggers a NIXL WRITE transfer
        that writes blocks directly into the primary CPU memory view.

    All methods are non-blocking. get_finished() polls for completed
    async transfers (both store cascades and load promotions).
    """

    def __init__(self) -> None:
        self._primary_view: memoryview | None = None

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
