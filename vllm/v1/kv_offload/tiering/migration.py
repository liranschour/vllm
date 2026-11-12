# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The `p2p.migrate` kv_hints action: proactive P2P KV-cache migration.

An orchestrator asks a *destination* vLLM instance to pull a set of KV blocks
from a named *source* peer into its CPU KV cache, off any request's critical
path. The request is carried as a `KvHintAction` in the `kv_hints` envelope
(see `vllm/v1/kv_hints/protocol.py`) of any inference request sent to the
destination; the blocks need not belong to that request. A tiny warm-up
request (one output token) is enough to deliver it.

Action fields::

    action_type: "p2p.migrate"
    action_version: "1.<minor>"
    payload: {
        "source": {"host": str, "port": int},  # the source's P2P endpoint
        "blocks": [str, ...],  # hex-encoded OffloadKeys
    }

The migration is identified by `"<message_id>:<action_id>"`. Its lifecycle is
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

ACTION_TYPE = "p2p.migrate"
# Payload schema major version this implementation understands. Minor versions
# only add optional fields, so any "1.x" is accepted.
ACTION_MAJOR_VERSION = "1"

# Upper bound on blocks per migrate action. vLLM does not validate action
# payloads, so the consumer bounds its own input. The real limit is the
# destination's CPU capacity, which submit_migration() checks; this ceiling
# only keeps a malformed payload from allocating a huge dict before that check.
MAX_BLOCKS_PER_MIGRATE = 16_384

# The migration id goes onto the wire verbatim as the P2P `kv_request_id` and
# into a request id and log lines, so bound its length and charset.
MAX_TRANSFER_ID_LEN = 128
_TRANSFER_ID_RE = re.compile(r"\A[A-Za-z0-9._:-]+\Z")

# An OffloadKey is a block hash followed by a 4-byte group index.
_GROUP_IDX_BYTES = 4


class MigrationHintError(ValueError):
    """A `p2p.migrate` action that cannot be accepted; the message says why."""


class _Source(msgspec.Struct, forbid_unknown_fields=False):
    host: str
    port: int


class _Payload(msgspec.Struct, forbid_unknown_fields=False):
    source: _Source
    blocks: list[str]


@dataclass(frozen=True)
class MigrationRequest:
    """A validated `p2p.migrate` action."""

    transfer_id: str
    host: str
    port: int
    keys: list[OffloadKey]


def is_migrate_action(action: KvHintAction) -> bool:
    return action.action_type == ACTION_TYPE


def parse_migrate_action(message_id: str, action: KvHintAction) -> MigrationRequest:
    """Validate one `p2p.migrate` action.

    Args:
        message_id: The carrying envelope's message id.
        action: An action whose `action_type` is `p2p.migrate`.

    Returns:
        The validated migration request.

    Raises:
        MigrationHintError: If the version, id, or payload is not acceptable.

    """
    if action.action_version.split(".", 1)[0] != ACTION_MAJOR_VERSION:
        raise MigrationHintError(f"unsupported_version: {action.action_version!r}")

    transfer_id = f"{message_id}:{action.action_id}"
    if len(transfer_id) > MAX_TRANSFER_ID_LEN or not _TRANSFER_ID_RE.match(transfer_id):
        raise MigrationHintError(f"bad_id: {transfer_id!r}")

    try:
        payload = msgspec.convert(action.payload, type=_Payload)
    except msgspec.ValidationError as exc:
        raise MigrationHintError(f"bad_payload: {exc}") from exc

    if not payload.source.host or not 0 < payload.source.port < 65536:
        raise MigrationHintError("bad_source")
    if not payload.blocks:
        raise MigrationHintError("no_blocks")
    if len(payload.blocks) > MAX_BLOCKS_PER_MIGRATE:
        raise MigrationHintError("too_many_blocks")

    keys: list[OffloadKey] = []
    for block in payload.blocks:
        try:
            key = bytes.fromhex(block)
        except ValueError as exc:
            raise MigrationHintError("bad_blocks: not hex") from exc
        if len(key) <= _GROUP_IDX_BYTES:
            raise MigrationHintError("bad_blocks: too short for an OffloadKey")
        keys.append(OffloadKey(key))

    return MigrationRequest(
        transfer_id=transfer_id,
        host=payload.source.host,
        port=payload.source.port,
        keys=keys,
    )
