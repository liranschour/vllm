# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Proactive P2P KV-cache migration: control-RPC payload schema and dispatch.

This is the connector-defined payload carried opaquely over the generic KV
connector control RPC (``POST /v1/kv_connector/rpc`` -> ``on_rpc``). An
orchestrator asks a *destination* vLLM instance to pull a set of KV blocks from
a named *source* peer into its CPU KV cache, off any request's critical path,
then polls for completion before routing the real request there.

The wire format is a ``msgspec.msgpack`` object with an ``op`` discriminator
(mirrors the P2P session protocol in
``vllm/v1/kv_offload/tiering/p2p/session/protocol.py``). Expected error
conditions (malformed payload, unknown op, unknown transfer_id, capacity) are
reported inside the response object (HTTP 200); only a connector that cannot
migrate at all returns ``None`` from ``on_rpc`` (HTTP 501), which is decided by
the caller, not here.

Request objects::

    {
        "v": 1,
        "op": "migrate",
        "transfer_id": str,
        "source": {"host": str, "port": int},
        "blocks": [bytes, ...],
    }
    {"v": 1, "op": "poll", "transfer_id": str}
    {"v": 1, "op": "cancel", "transfer_id": str}

Response objects::

    migrate -> {"transfer_id": str, "accepted": bool,
                "num_blocks": int}  # or {"accepted": False, "error": str}
    poll    -> {"transfer_id": str, "state": str, "blocks_total": int,
                "blocks_done": int, "blocks_missing": int,
                "blocks_failed": int, "error": str | None}
    cancel  -> {"transfer_id": str, "cancelled": bool}
    error   -> {"error": str}   # unparseable/invalid envelope
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import msgspec

from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.v1.kv_offload.tiering.manager import TieringOffloadingManager

logger = init_logger(__name__)

PROTOCOL_VERSION = 1

OP_MIGRATE = "migrate"
OP_POLL = "poll"
OP_CANCEL = "cancel"

# Upper bound on blocks per migrate request; the connector must bound its own
# input since vLLM performs no schema validation on opaque payloads.
MAX_BLOCKS_PER_MIGRATE = 100_000


def _encode(obj: dict[str, Any]) -> bytes:
    return msgspec.msgpack.encode(obj)


def _error(message: str) -> bytes:
    return _encode({"error": message})


def handle_migration_rpc(manager: TieringOffloadingManager, payload: bytes) -> bytes:
    """Decode, validate, and dispatch a migration control RPC.

    Args:
        manager: The tiering manager owning the migration registry.
        payload: The raw connector-defined request bytes.

    Returns:
        The response bytes. Always returns bytes (never None): the decision of
        whether this connector supports migration at all is made by the caller.
    """
    try:
        msg = msgspec.msgpack.decode(payload)
    except (msgspec.DecodeError, msgspec.ValidationError, ValueError) as exc:
        return _error(f"decode_error: {exc}")

    if not isinstance(msg, dict):
        return _error("malformed: expected a msgpack object")

    op = msg.get("op")
    if op == OP_MIGRATE:
        return _handle_migrate(manager, msg)
    if op == OP_POLL:
        return _handle_poll(manager, msg)
    if op == OP_CANCEL:
        return _handle_cancel(manager, msg)
    return _error(f"unknown_op: {op!r}")


def _require_transfer_id(msg: dict[str, Any]) -> str | None:
    transfer_id = msg.get("transfer_id")
    if isinstance(transfer_id, str) and transfer_id:
        return transfer_id
    return None


def _handle_migrate(manager: TieringOffloadingManager, msg: dict[str, Any]) -> bytes:
    transfer_id = _require_transfer_id(msg)
    if transfer_id is None:
        return _error("migrate: missing or invalid transfer_id")

    source = msg.get("source")
    if not isinstance(source, dict):
        return _encode(
            {"transfer_id": transfer_id, "accepted": False, "error": "bad_source"}
        )
    host = source.get("host")
    port = source.get("port")
    if not isinstance(host, str) or not host or not isinstance(port, int):
        return _encode(
            {"transfer_id": transfer_id, "accepted": False, "error": "bad_source"}
        )

    blocks = msg.get("blocks")
    if not isinstance(blocks, list) or not blocks:
        return _encode(
            {"transfer_id": transfer_id, "accepted": False, "error": "no_blocks"}
        )
    if len(blocks) > MAX_BLOCKS_PER_MIGRATE:
        return _encode(
            {"transfer_id": transfer_id, "accepted": False, "error": "too_many_blocks"}
        )
    if not all(isinstance(b, bytes) for b in blocks):
        return _encode(
            {"transfer_id": transfer_id, "accepted": False, "error": "bad_blocks"}
        )

    accepted, num_blocks, error = manager.submit_migration(
        transfer_id, host, port, blocks
    )
    if not accepted:
        return _encode({"transfer_id": transfer_id, "accepted": False, "error": error})
    return _encode(
        {"transfer_id": transfer_id, "accepted": True, "num_blocks": num_blocks}
    )


def _handle_poll(manager: TieringOffloadingManager, msg: dict[str, Any]) -> bytes:
    transfer_id = _require_transfer_id(msg)
    if transfer_id is None:
        return _error("poll: missing or invalid transfer_id")
    status = manager.poll_migration(transfer_id)
    if status is None:
        return _encode({"transfer_id": transfer_id, "state": "unknown"})
    return _encode({"transfer_id": transfer_id, **status})


def _handle_cancel(manager: TieringOffloadingManager, msg: dict[str, Any]) -> bytes:
    transfer_id = _require_transfer_id(msg)
    if transfer_id is None:
        return _error("cancel: missing or invalid transfer_id")
    cancelled = manager.cancel_migration(transfer_id)
    return _encode({"transfer_id": transfer_id, "cancelled": cancelled})
