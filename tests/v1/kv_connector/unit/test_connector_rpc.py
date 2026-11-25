# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the generic connector control RPC (RFC #51639).

Covers the `bytes -> bytes` control channel at the layers below the HTTP
transport:

* `KVConnectorBase_V1.on_rpc` default (not-implemented sentinel).
* `MultiConnector.on_rpc` first-non-None dispatch, empty-bytes success, and
  exception propagation.
* `EngineCore.kv_connector_rpc` dispatch via `scheduler.get_kv_connector()`.
* `AsyncMPClient` DP-rank target resolution (single engine, never broadcast).
"""

import pytest

from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorBase_V1
from vllm.distributed.kv_transfer.kv_connector.v1.multi_connector import (
    MultiConnector,
)
from vllm.v1.engine.core import EngineCore
from vllm.v1.engine.core_client import AsyncMPClient


class _StubConnector(KVConnectorBase_V1):
    """Minimal concrete connector that skips the config-bound __init__.

    Only `on_rpc` behavior is under test, so the abstract methods are
    stubbed and no vllm_config is required.
    """

    def __init__(self):
        pass

    def start_load_kv(self, *args, **kwargs):
        pass

    def wait_for_layer_load(self, *args, **kwargs):
        pass

    def save_kv_layer(self, *args, **kwargs):
        pass

    def wait_for_save(self, *args, **kwargs):
        pass

    def get_num_new_matched_tokens(self, *args, **kwargs):
        return (0, False)

    def update_state_after_alloc(self, *args, **kwargs):
        pass

    def build_connector_meta(self, *args, **kwargs):
        return None

    def request_finished_all_groups(self, *args, **kwargs):
        return (False, None)

    def aggregate(self, *args, **kwargs):
        return None


class _EchoConnector(_StubConnector):
    def on_rpc(self, payload: bytes) -> bytes:
        return b"echo:" + payload


class _EmptyAckConnector(_StubConnector):
    def on_rpc(self, payload: bytes) -> bytes:
        return b""


class _RaisingConnector(_StubConnector):
    def on_rpc(self, payload: bytes) -> bytes:
        raise RuntimeError("boom")


def test_base_on_rpc_returns_none():
    # The base-class default signals "not implemented".
    assert _StubConnector().on_rpc(b"anything") is None


def test_on_rpc_override_returns_bytes():
    assert _EchoConnector().on_rpc(b"ping") == b"echo:ping"


def _make_multi(connectors) -> MultiConnector:
    mc = object.__new__(MultiConnector)
    mc._connectors = connectors
    return mc


def test_multi_connector_returns_first_non_none():
    mc = _make_multi([_StubConnector(), _EchoConnector()])
    assert mc.on_rpc(b"x") == b"echo:x"


def test_multi_connector_empty_bytes_is_success_not_decline():
    # An empty b"" from an earlier child is a valid reply and must win over a
    # later child that would have returned non-empty bytes.
    mc = _make_multi([_EmptyAckConnector(), _EchoConnector()])
    assert mc.on_rpc(b"x") == b""


def test_multi_connector_all_decline_returns_none():
    mc = _make_multi([_StubConnector(), _StubConnector()])
    assert mc.on_rpc(b"x") is None


def test_multi_connector_exception_propagates_without_fallthrough():
    # The raising child comes first; the exception must propagate rather than
    # falling through to the echo child.
    mc = _make_multi([_RaisingConnector(), _EchoConnector()])
    with pytest.raises(RuntimeError, match="boom"):
        mc.on_rpc(b"x")


class _FakeScheduler:
    def __init__(self, connector):
        self._connector = connector

    def get_kv_connector(self):
        return self._connector


def _make_engine_core(connector) -> EngineCore:
    core = object.__new__(EngineCore)
    core.scheduler = _FakeScheduler(connector)
    return core


def test_engine_core_rpc_no_connector_returns_none():
    assert _make_engine_core(None).kv_connector_rpc(b"x") is None


def test_engine_core_rpc_delegates_to_connector():
    assert _make_engine_core(_EchoConnector()).kv_connector_rpc(b"ping") == b"echo:ping"


def test_engine_core_rpc_propagates_connector_exception():
    with pytest.raises(RuntimeError, match="boom"):
        _make_engine_core(_RaisingConnector()).kv_connector_rpc(b"x")


def _make_client(ranks: list[int]) -> AsyncMPClient:
    client = object.__new__(AsyncMPClient)
    client.core_engines = [r.to_bytes(2, "little") for r in ranks]
    return client


def test_rpc_target_single_engine_default():
    # External LB: the server fronts one engine (here DP rank 2).
    client = _make_client([2])
    assert client._rpc_target_engine(None) == (2).to_bytes(2, "little")
    assert client._rpc_target_engine(2) == (2).to_bytes(2, "little")


def test_rpc_target_requires_dp_rank_with_multiple_engines():
    with pytest.raises(ValueError, match="dp_rank is required"):
        _make_client([0, 1])._rpc_target_engine(None)


@pytest.mark.parametrize("dp_rank", [1, 5, 70000])
def test_rpc_target_rejects_unserved_rank(dp_rank):
    with pytest.raises(ValueError, match="not served"):
        _make_client([0] if dp_rank == 1 else [0, 1])._rpc_target_engine(dp_rank)


@pytest.mark.asyncio
async def test_rpc_sent_to_exactly_one_engine():
    client = _make_client([0, 1, 2])
    calls = []

    async def _fake_call(method, *args, engine):
        calls.append((method, args, engine))
        return b"ok"

    client._call_utility_async = _fake_call
    assert await client.kv_connector_rpc_async(b"p", dp_rank=1) == b"ok"
    assert calls == [("kv_connector_rpc", (b"p",), (1).to_bytes(2, "little"))]
