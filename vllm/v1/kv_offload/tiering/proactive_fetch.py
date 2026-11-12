# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The `p2p.fetch` kv_hints action: proactive P2P KV-cache fetch.

An orchestrator asks a *destination* vLLM instance to pull a set of KV blocks
from a named *source* peer into its CPU KV cache, off any request's critical
path. The request is carried as a `KvHintAction` in the `kv_hints` envelope
(see `vllm/v1/kv_hints/protocol.py`) of any inference request sent to the
destination; the blocks need not belong to that request. A tiny warm-up
request (one output token) is enough to deliver it. It is a copy: the source
keeps its blocks.

Action fields::

    action_type: "p2p.fetch"
    action_version: "1.<minor>"
    payload: {
        "source": {"host": str, "port": int},  # the source's P2P endpoint
        "blocks": [str, ...],  # hex-encoded OffloadKeys
    }

The fetch is identified by `"<message_id>:<action_id>"`. Its lifecycle is
independent of the carrying request: finishing or aborting that request does
not cancel it. There is no reply channel; the outcome is observable through the
destination's KV cache events, where each block that lands emits a CPU
`BlockStored` (and a block that fails after a slot was reserved, a
`BlockRemoved`).

As kv_hints requires, a malformed or unsupported action is logged and skipped:
it never fails the request carrying it or the other actions in the envelope.
"""

from __future__ import annotations

from dataclasses import dataclass

import msgspec
import regex as re

from vllm.v1.kv_hints import KvHintAction
from vllm.v1.kv_offload.base import OffloadKey

ACTION_TYPE = "p2p.fetch"
# Payload schema major version this implementation understands. Minor versions
# only add optional fields, so any "1.x" is accepted.
ACTION_MAJOR_VERSION = "1"

# Upper bound on blocks per fetch action. vLLM does not validate action
# payloads, so the consumer bounds its own input. The real limit is the
# destination's CPU capacity, which submit_proactive_fetch() checks; this ceiling
# only keeps a malformed payload from allocating a huge dict before that check.
MAX_BLOCKS_PER_FETCH = 16_384

# The fetch id goes onto the wire verbatim as the P2P `kv_request_id` and
# into a request id and log lines, so bound its length and charset.
MAX_TRANSFER_ID_LEN = 128
_TRANSFER_ID_RE = re.compile(r"\A[A-Za-z0-9._:-]+\Z")

# An OffloadKey is a block hash followed by a 4-byte group index.
_GROUP_IDX_BYTES = 4


class FetchHintError(ValueError):
    """A `p2p.fetch` action that cannot be accepted; the message says why."""


class _Source(msgspec.Struct, forbid_unknown_fields=False):
    host: str
    port: int


class _Payload(msgspec.Struct, forbid_unknown_fields=False):
    source: _Source
    blocks: list[str]


@dataclass(frozen=True)
class FetchRequest:
    """A validated `p2p.fetch` action."""

    transfer_id: str
    host: str
    port: int
    keys: list[OffloadKey]


def is_fetch_action(action: KvHintAction) -> bool:
    return action.action_type == ACTION_TYPE


def parse_fetch_action(message_id: str, action: KvHintAction) -> FetchRequest:
    """Validate one `p2p.fetch` action.

    Args:
        message_id: The carrying envelope's message id.
        action: An action whose `action_type` is `p2p.fetch`.

    Returns:
        The validated fetch request.

    Raises:
        FetchHintError: If the version, id, or payload is not acceptable.

    """
    if action.action_version.split(".", 1)[0] != ACTION_MAJOR_VERSION:
        raise FetchHintError(f"unsupported_version: {action.action_version!r}")

    transfer_id = f"{message_id}:{action.action_id}"
    if len(transfer_id) > MAX_TRANSFER_ID_LEN or not _TRANSFER_ID_RE.match(transfer_id):
        raise FetchHintError(f"bad_id: {transfer_id!r}")

    try:
        payload = msgspec.convert(action.payload, type=_Payload)
    except msgspec.ValidationError as exc:
        raise FetchHintError(f"bad_payload: {exc}") from exc

    if not payload.source.host or not 0 < payload.source.port < 65536:
        raise FetchHintError("bad_source")
    if not payload.blocks:
        raise FetchHintError("no_blocks")
    if len(payload.blocks) > MAX_BLOCKS_PER_FETCH:
        raise FetchHintError("too_many_blocks")

    keys: list[OffloadKey] = []
    for block in payload.blocks:
        try:
            key = bytes.fromhex(block)
        except ValueError as exc:
            raise FetchHintError("bad_blocks: not hex") from exc
        if len(key) <= _GROUP_IDX_BYTES:
            raise FetchHintError("bad_blocks: too short for an OffloadKey")
        keys.append(OffloadKey(key))

    return FetchRequest(
        transfer_id=transfer_id,
        host=payload.source.host,
        port=payload.source.port,
        keys=keys,
    )
