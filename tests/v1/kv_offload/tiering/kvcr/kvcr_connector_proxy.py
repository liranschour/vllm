# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KVCR proxy server for OffloadingConnector + TieringOffloadingSpec.

The KVCR sibling of ``p2p/p2p_connector_proxy.py``. Same two-step PD flow —
prefill on the prefiller, then decode on the decoder — but the decode request
carries a router ``kv_hints`` envelope instead of ``kv_transfer_params``, because
KVCR only ever fetches from a peer the router names.

Differences from the P2P proxy, all forced by KVCR's design:

* The prefill request needs no injected metadata. A KVCR producer is passive:
  it holds KV in vLLM's own host tier and answers a peer's ``start_write`` by
  pinning those chunks, so nothing has to be told to it up front.
* Only prefiller-first makes sense. KVCR transfers are destination-initiated,
  so the decoder's fetch succeeds only once the prefiller already holds the
  blocks; there is no "push to a waiting decoder" mode to race.
* The hint names ``source_control_endpoint`` — the prefiller's **ZMQ control**
  endpoint (``control_ports[dp_local_rank]``), not its HTTP port.

Block hashes (``--hash-mode``):

* ``opportunistic`` (default) sends a placeholder hash list. KVCR's
  ``opportunistic_query`` — which the vLLM adapter enables by default — drops
  the hash-membership test, so the decoder queries the named prefiller for every
  key in the request. This exercises the real transfer path; it does not verify
  hash matching.
* ``exact`` computes the prefix block hashes with vLLM's own hashing and sends
  them. Two engine settings have to agree or every hint silently degrades to a
  miss: ``VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES`` must stay at its default of 1
  (at 0 the decoder's keys decode to raw bytes, which can never match the
  envelope's uint64 list), and ``--hash-algo`` here must match the engine's
  ``--prefix-caching-hash-algo``. The non-crypto algos (``xxhash*``) seed
  NONE_HASH per process unless ``PYTHONHASHSEED`` is shared, so exact mode
  needs ``sha256*`` or a pinned PYTHONHASHSEED everywhere. Exact mode also
  assumes no per-block extra keys, so ``cache_salt``, multimodal inputs and
  LoRA requests fall outside it — use ``opportunistic`` for those.

Usage:
    venv/bin/python kvcr_connector_proxy.py \
        --port 8192 \
        --prefiller-hosts 127.0.0.1 --prefiller-ports 8100 \
        --decoder-hosts   127.0.0.1 --decoder-ports  8200 \
        --kvcr-control-host 10.0.0.5 --kvcr-control-port 17771
"""

import argparse
import itertools
import logging
import os
import uuid
from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Envelope and action versions the KVCR hint parser accepts today. A mismatch
# is only warned about on the engine side, but keep them exact.
KV_HINT_PROTOCOL_VERSION = "0.1"
KV_FETCH_ACTION_TYPE = "kv.fetch"
KV_FETCH_ACTION_VERSION = "1.0"

# Stands in for the real prefix hashes under --hash-mode opportunistic. KVCR
# rejects an empty list, so the envelope needs at least one value; the value
# itself is never compared when opportunistic querying is on.
PLACEHOLDER_BLOCK_HASH = 1


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.prefill_clients = []
    app.state.decode_clients = []

    for i, (host, port) in enumerate(global_args.prefiller_instances):
        app.state.prefill_clients.append(
            {
                "client": httpx.AsyncClient(
                    timeout=None,
                    base_url=f"http://{host}:{port}/v1",
                    limits=httpx.Limits(
                        max_connections=None, max_keepalive_connections=None
                    ),
                ),
                "host": host,
                "port": port,
                "id": i,
            }
        )

    for i, (host, port) in enumerate(global_args.decoder_instances):
        app.state.decode_clients.append(
            {
                "client": httpx.AsyncClient(
                    timeout=None,
                    base_url=f"http://{host}:{port}/v1",
                    limits=httpx.Limits(
                        max_connections=None, max_keepalive_connections=None
                    ),
                ),
                "host": host,
                "port": port,
                "id": i,
            }
        )

    app.state.prefill_iterator = itertools.cycle(range(len(app.state.prefill_clients)))
    app.state.decode_iterator = itertools.cycle(range(len(app.state.decode_clients)))
    app.state.prefill_dp_iterator = itertools.cycle(
        range(global_args.prefiller_dp_size)
    )
    app.state.decode_dp_iterator = itertools.cycle(range(global_args.decoder_dp_size))

    print(
        f"KVCR proxy ready [hash-mode={global_args.hash_mode}]: "
        f"{len(app.state.prefill_clients)} prefiller(s) x "
        f"dp={global_args.prefiller_dp_size}, "
        f"{len(app.state.decode_clients)} decoder(s) x "
        f"dp={global_args.decoder_dp_size}. "
        f"Hint source endpoint base "
        f"{_source_endpoint(0)}"
    )
    if global_args.hash_mode == "exact":
        print(
            f"hash-mode=exact [algo={global_args.hash_algo}, "
            f"block_size={global_args.block_size}]: these must match the "
            "engine's --prefix-caching-hash-algo and --block-size, and the "
            "pods must keep VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES at its "
            "default of 1, or no hint hash can match."
        )
    yield

    for ci in app.state.prefill_clients:
        await ci["client"].aclose()
    for ci in app.state.decode_clients:
        await ci["client"].aclose()


app = FastAPI(lifespan=lifespan)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=8192)
    p.add_argument("--host", type=str, default="127.0.0.1")
    p.add_argument("--prefiller-hosts", type=str, nargs="+", default=["127.0.0.1"])
    p.add_argument("--prefiller-ports", type=int, nargs="+", default=[8100])
    p.add_argument("--decoder-hosts", type=str, nargs="+", default=["127.0.0.1"])
    p.add_argument("--decoder-ports", type=int, nargs="+", default=[8200])
    # The prefiller's KVCR peer control endpoint, injected into decode requests.
    # This is the tier's `control_advertise_host` and `control_ports` entry, not
    # the HTTP port.
    p.add_argument(
        "--kvcr-control-host",
        type=str,
        default="127.0.0.1",
        help="Host of the prefiller's KVCR ZMQ peer control socket "
        "(the tier's control_advertise_host)",
    )
    p.add_argument(
        "--kvcr-control-port",
        type=int,
        default=17771,
        help="Port of the prefiller's KVCR peer control socket for DP rank 0; "
        "rank N uses this + N, matching control_ports in rank order",
    )
    p.add_argument(
        "--hash-mode",
        choices=("opportunistic", "exact"),
        default="opportunistic",
        help="opportunistic: send a placeholder hash list and rely on KVCR's "
        "opportunistic_query (adapter default) to query the named peer for "
        "every key. exact: compute the prefix block hashes locally; needs "
        "VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES=1 on the decoder.",
    )
    p.add_argument(
        "--block-size",
        type=int,
        default=16,
        help="Engine block size, used by --hash-mode exact to chunk tokens.",
    )
    p.add_argument(
        "--hash-algo",
        choices=("sha256", "sha256_cbor", "xxhash", "xxhash_cbor"),
        default="sha256",
        help="Must match the engine's --prefix-caching-hash-algo for "
        "--hash-mode exact (default matches the engine default).",
    )
    p.add_argument(
        "--tokenizer",
        type=str,
        default=None,
        help="Tokenizer id or path for --hash-mode exact (defaults to the "
        "request's model field).",
    )
    p.add_argument(
        "--prefiller-dp-size",
        type=int,
        default=1,
        help="Data-parallel replica count of the prefiller. When >1 the proxy "
        "round-robins prefill across ranks via the X-data-parallel-rank header "
        "and points the hint at kvcr-control-port + rank, so the decoder "
        "fetches from the replica that produced the KV.",
    )
    p.add_argument(
        "--decoder-dp-size",
        type=int,
        default=1,
        help="Data-parallel replica count of the decoder.",
    )
    args = p.parse_args()
    if len(args.prefiller_hosts) != len(args.prefiller_ports):
        raise ValueError("Prefiller host/port count mismatch")
    if len(args.decoder_hosts) != len(args.decoder_ports):
        raise ValueError("Decoder host/port count mismatch")
    args.prefiller_instances = list(zip(args.prefiller_hosts, args.prefiller_ports))
    args.decoder_instances = list(zip(args.decoder_hosts, args.decoder_ports))
    return args


def _get_next(app, service: str):
    if service == "prefill":
        return app.state.prefill_clients[next(app.state.prefill_iterator)]
    return app.state.decode_clients[next(app.state.decode_iterator)]


def _next_dp_rank(app, service: str):
    """Advance the round-robin DP cursor for a role.

    Returns (rank, header_rank). ``rank`` drives the control-port arithmetic;
    ``header_rank`` is None when the role has dp_size==1 so no header is sent.
    """
    if service == "prefill":
        rank = next(app.state.prefill_dp_iterator)
        dp_size = global_args.prefiller_dp_size
    else:
        rank = next(app.state.decode_dp_iterator)
        dp_size = global_args.decoder_dp_size
    return rank, (rank if dp_size > 1 else None)


def _auth_headers(request_id: str) -> dict:
    headers: dict = {"X-Request-Id": request_id}
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def _source_endpoint(dp_rank: int) -> str:
    """The prefiller replica's KVCR control endpoint, as the hint names it."""
    host = global_args.kvcr_control_host
    port = global_args.kvcr_control_port + dp_rank
    return f"tcp://{host}:{port}"


_TOKENIZER_CACHE: dict[str, Any] = {}


def _block_hashes_for(req_data: dict) -> list[int]:
    """Prefix block hashes for this request, as uint64 values.

    Mirrors the engine: chain vLLM's block hashing over whole blocks of the
    prompt, then apply the same uint64 truncation the engine's key adapter
    applies through ``maybe_convert_block_hash``.
    """
    if global_args.hash_mode == "opportunistic":
        return [PLACEHOLDER_BLOCK_HASH]

    # Imported lazily so the proxy runs without vLLM in opportunistic mode.
    from vllm.utils.hashing import get_hash_fn_by_name
    from vllm.v1.core.kv_cache_utils import (
        hash_block_tokens,
        init_none_hash,
        maybe_convert_block_hash,
    )

    hash_fn = get_hash_fn_by_name(global_args.hash_algo)
    init_none_hash(hash_fn)
    if global_args.hash_algo.startswith("xxhash") and not os.getenv("PYTHONHASHSEED"):
        logger.warning(
            "hash-mode=exact with %s and no PYTHONHASHSEED: NONE_HASH is "
            "seeded per process, so these hashes cannot match the engine's.",
            global_args.hash_algo,
        )

    prompt = req_data.get("prompt")
    if isinstance(prompt, list) and prompt and isinstance(prompt[0], int):
        token_ids = list(prompt)
    else:
        name = global_args.tokenizer or req_data.get("model")
        if not isinstance(name, str) or not name:
            raise ValueError(
                "--hash-mode exact needs --tokenizer, a model field on the "
                "request, or a prompt given as token ids"
            )
        tokenizer = _TOKENIZER_CACHE.get(name)
        if tokenizer is None:
            from vllm.tokenizers import get_tokenizer

            tokenizer = get_tokenizer(name)
            _TOKENIZER_CACHE[name] = tokenizer
        text = prompt if isinstance(prompt, str) else _chat_text(req_data)
        token_ids = tokenizer.encode(text)

    block_size = global_args.block_size
    hashes: list[int] = []
    parent = None
    for start in range(0, len(token_ids) - block_size + 1, block_size):
        parent = hash_block_tokens(
            hash_fn, parent, token_ids[start : start + block_size]
        )
        external = maybe_convert_block_hash(parent)
        if isinstance(external, bytes):
            external = int.from_bytes(external, byteorder="big") & ((1 << 64) - 1)
        hashes.append(external)
    if not hashes:
        logger.warning(
            "prompt shorter than one block (%d tokens, block_size=%d); "
            "falling back to the placeholder hash",
            len(token_ids),
            block_size,
        )
        return [PLACEHOLDER_BLOCK_HASH]
    return hashes


def _chat_text(req_data: dict) -> str:
    """Flatten chat messages for tokenization in exact mode.

    The engine hashes the templated prompt, so chat hashes computed this way are
    approximate. Use /v1/completions, or pass prompt token ids, when the hint
    must match exactly.
    """
    parts = []
    for message in req_data.get("messages", []):
        content = message.get("content")
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            parts.extend(
                chunk.get("text", "") for chunk in content if isinstance(chunk, dict)
            )
    return "".join(parts)


def _kv_hints(request_id: str, prefill_rank: int, block_hashes: list[int]) -> dict:
    """Build the router hint envelope KVCR's hint_parser accepts."""
    return {
        "protocol_version": KV_HINT_PROTOCOL_VERSION,
        "message_id": request_id,
        "actions": [
            {
                "action_id": f"{request_id}-fetch",
                "action_type": KV_FETCH_ACTION_TYPE,
                "action_version": KV_FETCH_ACTION_VERSION,
                "payload": {
                    "source_control_endpoint": _source_endpoint(prefill_rank),
                    "block_hashes": block_hashes,
                    "mode": "copy",
                },
            }
        ],
    }


async def _prefill(client_info, endpoint, req_data, request_id, dp_rank=None):
    """Populate the prefiller's host tier: prefill-only, no injected metadata."""
    data = req_data.copy()
    # KVCR's producer side is passive, and a stray P2P param would be read by a
    # different tier in a MultiConnector setup.
    data.pop("kv_transfer_params", None)
    data.pop("kv_hints", None)
    data["stream"] = False
    data["max_tokens"] = 1
    data.pop("max_completion_tokens", None)
    data.pop("stream_options", None)
    data.pop("min_tokens", None)
    data.pop("min_completion_tokens", None)

    headers = _auth_headers(request_id)
    if dp_rank is not None:
        # Pin the prefill to the replica whose KVCR control socket the hint
        # will name.
        headers["X-data-parallel-rank"] = str(dp_rank)
    resp = await client_info["client"].post(endpoint, json=data, headers=headers)
    resp.raise_for_status()
    await resp.aread()
    return resp


async def _stream_decode(client_info, endpoint, req_data, request_id, dp_rank=None):
    headers = _auth_headers(request_id)
    if dp_rank is not None:
        headers["X-data-parallel-rank"] = str(dp_rank)
    async with client_info["client"].stream(
        "POST", endpoint, json=req_data, headers=headers
    ) as resp:
        resp.raise_for_status()
        async for chunk in resp.aiter_bytes():
            yield chunk


async def _handle_completions(api: str, request: Request):
    try:
        req_data = await request.json()
        request_id = str(uuid.uuid4())

        # Pick DP replicas for both roles up front (before any await) so the
        # two round-robin advances stay paired.
        prefill_rank, prefill_hdr = _next_dp_rank(request.app, "prefill")
        _decode_rank, decode_hdr = _next_dp_rank(request.app, "decode")

        prefill_client = _get_next(request.app, "prefill")
        await _prefill(prefill_client, api, req_data, request_id, dp_rank=prefill_hdr)

        # The prefiller now holds the KV in its host tier. Tell the decoder
        # which peer to fetch it from; KVCR does the rest on its first read.
        block_hashes = _block_hashes_for(req_data)
        req_data.pop("kv_transfer_params", None)
        req_data["kv_hints"] = _kv_hints(request_id, prefill_rank, block_hashes)

        decode_client = _get_next(request.app, "decode")
        logger.debug(
            "prefill=%s:%s dp=%s -> decode=%s:%s dp=%s hint_source=%s blocks=%d",
            prefill_client["host"],
            prefill_client["port"],
            prefill_rank,
            decode_client["host"],
            decode_client["port"],
            _decode_rank,
            _source_endpoint(prefill_rank),
            len(block_hashes),
        )

        async def generate():
            async for chunk in _stream_decode(
                decode_client, api, req_data, request_id, dp_rank=decode_hdr
            ):
                yield chunk

        return StreamingResponse(generate(), media_type="application/json")

    except Exception as e:
        import sys
        import traceback

        print(f"Proxy error on {api}: {e}")
        print("".join(traceback.format_exception(*sys.exc_info())))
        raise


@app.post("/v1/completions")
async def completions(request: Request):
    return await _handle_completions("/completions", request)


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    return await _handle_completions("/chat/completions", request)


@app.get("/healthcheck")
async def healthcheck():
    return {"status": "ok"}


if __name__ == "__main__":
    global_args = parse_args()

    import uvicorn

    uvicorn.run(app, host=global_args.host, port=global_args.port)
