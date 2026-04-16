# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Unit tests for PDConnector (Step 1: skeleton).

Verifies:
1. PDConnector instantiates and reports the correct tier name
2. set_primary_view() stores the memoryview
3. PDConnector can be registered with TieringOffloadingManager
4. Unimplemented abstract methods raise NotImplementedError
"""

import pytest
import torch

from vllm.v1.kv_offload.abstract import JobMetadata, OffloadKey
from vllm.v1.kv_offload.secondary_tiers.pd_connector import PDConnector
from vllm.v1.kv_offload.tiering.manager import (
    CPUPrimaryTierOffloadingManager,
    TieringOffloadingManager,
)


def make_key(i: int) -> OffloadKey:
    return OffloadKey(i.to_bytes(4, "big") + (0).to_bytes(4, "big"))


def make_primary_view() -> memoryview:
    tensor = torch.zeros((16, 8), dtype=torch.float32)
    return memoryview(tensor.numpy())


class TestPDConnectorSkeleton:

    def test_get_tier_name(self):
        connector = PDConnector()
        assert connector.get_tier_name() == "PDConnector"

    def test_set_primary_view_stores_view(self):
        connector = PDConnector()
        assert connector._primary_view is None

        view = make_primary_view()
        connector.set_primary_view(view)
        assert connector._primary_view is view

    def test_lookup_raises_not_implemented(self):
        connector = PDConnector()
        with pytest.raises(NotImplementedError):
            connector.lookup([make_key(0)])

    def test_submit_store_raises_not_implemented(self):
        connector = PDConnector()
        connector.set_primary_view(make_primary_view())
        with pytest.raises(NotImplementedError):
            connector.submit_store(JobMetadata(job_id=0, keys=[], spec=None))

    def test_submit_load_raises_not_implemented(self):
        connector = PDConnector()
        connector.set_primary_view(make_primary_view())
        with pytest.raises(NotImplementedError):
            connector.submit_load(JobMetadata(job_id=0, keys=[], spec=None))

    def test_get_finished_raises_not_implemented(self):
        connector = PDConnector()
        with pytest.raises(NotImplementedError):
            list(connector.get_finished())

    def test_registered_with_tiering_manager(self):
        """
        TieringOffloadingManager accepts PDConnector as a secondary tier
        and calls set_primary_view() during __init__.
        """
        connector = PDConnector()
        primary = CPUPrimaryTierOffloadingManager.__new__(
            CPUPrimaryTierOffloadingManager
        )

        # Provide a minimal get_primary_kv_tensors() via a mock primary
        class _MinimalPrimary(CPUPrimaryTierOffloadingManager):
            def __init__(self):
                pass  # skip CPUOffloadingManager.__init__

            def get_primary_kv_tensors(self):
                return torch.zeros((16, 8), dtype=torch.float32)

        primary = _MinimalPrimary()
        manager = TieringOffloadingManager(
            primary_tier=primary,
            secondary_tiers=[connector],
        )

        # After init, set_primary_view has been called
        assert connector._primary_view is not None
        assert manager.secondary_tiers == [connector]
