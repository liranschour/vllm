# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
P2PConnector proxy server for OffloadingConnector + TieringOffloadingSpec.

Unlike NixlConnector (which returns remote_host/remote_port in the prefill
response), OffloadingConnector does not embed connector coordinates in its
response. This proxy injects the prefiller's P2PConnector address into
kv_transfer_params before forwarding the decode request so the decoder knows
where to pull KV blocks from.

Usage:
    .venv/bin/python p2p_connector_proxy.py \
        --port 8192 \
        --prefiller-host 127.0.0.1 --prefiller-port 8100 \
        --decoder-host   127.0.0.1 --decoder-port  8200 \
        --p2p-connector-host 127.0.0.1 --p2p-connector-port 5710
"""

import argparse
import asyncio
import itertools
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager, suppress

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Imports used only by --proactive mode (KV-events subscription + migrate RPC).
# Kept lazy-friendly at module scope; both are vLLM deps present in the venv.
try:
    import msgspec
    import zmq
    import zmq.asyncio

    from vllm.distributed.kv_events import BlockStored, KVEventBatch

    _PROACTIVE_IMPORTS_OK = True
except Exception as _exc:  # pragma: no cover - only hit on a broken env
    _PROACTIVE_IMPORTS_OK = False
    _PROACTIVE_IMPORT_ERR = _exc


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

    # Round-robin over each role's data-parallel replicas (independent of the
    # instance iterators above). For dp_size=1 these yield only rank 0.
    app.state.prefill_dp_iterator = itertools.cycle(
        range(global_args.prefiller_dp_size)
    )
    app.state.decode_dp_iterator = itertools.cycle(range(global_args.decoder_dp_size))

    # Proactive mode: subscribe to the prefiller's KV-cache events so we can
    # learn which OffloadKeys it stored and migrate them to the decoder.
    app.state.proactive = global_args.proactive
    app.state.collector = None  # set to a list while a migration is capturing
    app.state.migration_lock = asyncio.Lock()
    app.state.kv_sub_task = None
    app.state.kv_ctx = None
    if global_args.proactive:
        if not _PROACTIVE_IMPORTS_OK:
            raise RuntimeError(
                f"--proactive needs zmq + msgspec + vllm importable: "
                f"{_PROACTIVE_IMPORT_ERR}"
            )
        app.state.kv_ctx = zmq.asyncio.Context()
        sub = app.state.kv_ctx.socket(zmq.SUB)
        sub.connect(global_args.kv_events_endpoint)
        sub.setsockopt_string(zmq.SUBSCRIBE, global_args.kv_events_topic)
        app.state.kv_sub_task = asyncio.create_task(_kv_events_loop(app, sub))
        print(
            f"Proactive: subscribed to KV events at "
            f"{global_args.kv_events_endpoint} topic={global_args.kv_events_topic!r}"
        )

    mode = (
        "proactive"
        if global_args.proactive
        else ("decoder-first" if global_args.decoder_first else "prefiller-first")
    )
    if global_args.p2p:
        mode += ",p2p"
    pd_host = global_args.p2p_connector_host
    pd_port = global_args.p2p_connector_port
    n_pref = len(app.state.prefill_clients)
    n_dec = len(app.state.decode_clients)
    print(
        f"Proxy ready [{mode}]: "
        f"{n_pref} prefiller(s) x dp={global_args.prefiller_dp_size}, "
        f"{n_dec} decoder(s) x dp={global_args.decoder_dp_size}. "
        f"P2PConnector base at {pd_host}:{pd_port}"
    )
    yield

    if app.state.kv_sub_task is not None:
        app.state.kv_sub_task.cancel()
        with suppress(asyncio.CancelledError, Exception):
            await app.state.kv_sub_task
    if app.state.kv_ctx is not None:
        app.state.kv_ctx.destroy(linger=0)
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
    # P2PConnector coordinates of the prefiller — injected into decode requests.
    p.add_argument(
        "--p2p-connector-host",
        type=str,
        default="127.0.0.1",
        help="Host of the prefiller's P2PConnector ZMQ socket",
    )
    p.add_argument(
        "--p2p-connector-port",
        type=int,
        default=int(os.getenv("VLLM_P2P_SIDE_CHANNEL_PORT", "5710")),
        help="Port of the prefiller's P2PConnector ZMQ socket "
        "(default: $VLLM_P2P_SIDE_CHANNEL_PORT or 5710)",
    )
    # P2PConnector coordinates of the decoder — injected into prefill requests
    # so the prefiller's submit_store can resolve the peer to push KV to.
    p.add_argument(
        "--decoder-p2p-connector-host",
        type=str,
        default="127.0.0.1",
        help="Host of the decoder's P2PConnector ZMQ socket",
    )
    p.add_argument(
        "--decoder-p2p-connector-port",
        type=int,
        default=int(os.getenv("VLLM_P2P_SIDE_CHANNEL_PORT", "5710")) + 1,
        help="Port of the decoder's P2PConnector ZMQ socket "
        "(default: $VLLM_P2P_SIDE_CHANNEL_PORT + 1 or 5711)",
    )
    p.add_argument(
        "--decoder-first",
        action="store_true",
        help="Send decode request before prefill so decoder is already "
        "waiting when KV blocks arrive (decoder-first mode)",
    )
    p.add_argument(
        "--p2p",
        action="store_true",
        help="P2P mode: do not inject kv_transfer_params on the prefiller; "
        "on the decoder, inject kv_transfer_params with a top-level "
        "'remote_kv_source' block ({kv_request_id, remote_host, remote_port}) "
        "instead of the default 'remote_prefiller' block.",
    )
    p.add_argument(
        "--prefiller-dp-size",
        type=int,
        default=1,
        help="Data-parallel replica count of the prefiller. When >1 the proxy "
        "round-robins prefill across ranks via the X-data-parallel-rank header "
        "and injects remote_port = p2p-connector-port + rank into the decode "
        "request so the decoder pulls KV from that replica's control socket. "
        "Assumes a single prefiller instance (one host/port fronting N replicas).",
    )
    p.add_argument(
        "--decoder-dp-size",
        type=int,
        default=1,
        help="Data-parallel replica count of the decoder. When >1 the proxy "
        "round-robins decode across ranks via the X-data-parallel-rank header.",
    )
    p.add_argument(
        "--proactive",
        action="store_true",
        help="Proactive P2P mode: (1) prefill on the prefiller, (2) subscribe to "
        "the prefiller's KV-cache events to learn the stored blocks, trigger a "
        "proactive migration on the decoder via POST /v1/kv_connector/rpc so it "
        "pulls those blocks into its CPU cache, (3) send a plain decode request "
        "to the decoder, which serves it from the prefetched cache. Requires the "
        "prefiller to run with kv-cache events enabled and "
        "VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES=0.",
    )
    p.add_argument(
        "--kv-events-endpoint",
        type=str,
        default=None,
        help="ZMQ PUB endpoint of the prefiller's KV-cache events, e.g. "
        "tcp://127.0.0.1:5557 (proactive mode only).",
    )
    p.add_argument(
        "--kv-events-topic",
        type=str,
        default="kv-events",
        help="KV-cache events subscription topic (must match the prefiller's "
        "--kv-events-config topic; proactive mode only).",
    )
    p.add_argument(
        "--migration-timeout",
        type=float,
        default=30.0,
        help="Seconds to poll a proactive migration before giving up "
        "(proactive mode only).",
    )
    args = p.parse_args()
    if args.proactive and not args.kv_events_endpoint:
        raise ValueError("--proactive requires --kv-events-endpoint")
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

    Returns (rank, header_rank). ``rank`` (0..dp_size-1) is always used for the
    P2P remote_port arithmetic; ``header_rank`` is the same value when the role
    has dp_size>1 and None otherwise (so dp=1 sends no header, matching the
    single-replica behavior).
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


async def _prefill(client_info, endpoint, req_data, request_id, dp_rank=None):
    """Send a prefill-only request (max_tokens=1) to the prefiller."""
    data = req_data.copy()
    if global_args.p2p or global_args.proactive:
        # p2p: decoder pulls reactively. proactive: decoder pulls via an
        # explicit migrate RPC. Either way the prefiller just offloads to its
        # own CPU tier — no remote_decoder push target.
        data.pop("kv_transfer_params", None)
    else:
        data["kv_transfer_params"] = {
            "remote_decoder": {
                "kv_request_id": request_id,
            },
        }
    data["stream"] = False
    data["max_tokens"] = 1
    data.pop("max_completion_tokens", None)
    data.pop("stream_options", None)
    data.pop("min_tokens", None)
    data.pop("min_completion_tokens", None)

    headers = _auth_headers(request_id)
    if dp_rank is not None:
        # Pin this prefill to a specific DP replica so its P2PConnector (bound
        # at base_port + data_parallel_index) holds the produced KV blocks.
        headers["X-data-parallel-rank"] = str(dp_rank)
    resp = await client_info["client"].post(endpoint, json=data, headers=headers)
    resp.raise_for_status()
    await resp.aread()
    return resp


async def _stream_decode(client_info, endpoint, req_data, request_id, dp_rank=None):
    headers = _auth_headers(request_id)
    if dp_rank is not None:
        # Round-robin this decode onto a specific DP replica of the decoder.
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
        # two round-robin advances are atomic together. Header rank is None for
        # a role with dp_size==1 (no header → single replica, unchanged).
        prefill_rank, prefill_hdr = _next_dp_rank(request.app, "prefill")
        decode_rank, decode_hdr = _next_dp_rank(request.app, "decode")

        prefill_client = _get_next(request.app, "prefill")
        await _prefill(prefill_client, api, req_data, request_id, dp_rank=prefill_hdr)

        # Inject the prefiller's P2PConnector address so the decoder can pull
        # KV blocks from it. remote_port = base + prefill_rank targets the
        # replica that produced the KV (base+0 == base when dp=1).
        decoder_key = "remote_kv_source" if global_args.p2p else "remote_prefiller"
        req_data["kv_transfer_params"] = {
            decoder_key: {
                "kv_request_id": request_id,
                "remote_host": global_args.p2p_connector_host,
                "remote_port": global_args.p2p_connector_port + prefill_rank,
            },
        }

        decode_client = _get_next(request.app, "decode")
        logger.debug(
            "prefill=%s dp=%s decode=%s dp=%s",
            prefill_client,
            prefill_rank,
            decode_client,
            decode_rank,
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


async def _handle_completions_decoder_first(api: str, request: Request):
    """Decoder-first mode: send decode request before prefill.

    The decoder establishes its request and starts polling for KV blocks
    immediately. The prefill is then sent so the prefiller computes and
    pushes blocks to the already-waiting decoder.
    """
    try:
        req_data = await request.json()
        request_id = str(uuid.uuid4())

        # Pick DP replicas up front (see _handle_completions for rationale).
        prefill_rank, prefill_hdr = _next_dp_rank(request.app, "prefill")
        decode_rank, decode_hdr = _next_dp_rank(request.app, "decode")

        prefill_client = _get_next(request.app, "prefill")
        decode_client = _get_next(request.app, "decode")

        decoder_key = "remote_kv_source" if global_args.p2p else "remote_prefiller"
        decode_data = req_data.copy()
        decode_data["kv_transfer_params"] = {
            decoder_key: {
                "kv_request_id": request_id,
                "remote_host": global_args.p2p_connector_host,
                "remote_port": global_args.p2p_connector_port + prefill_rank,
            },
        }

        async def generate():
            queue: asyncio.Queue = asyncio.Queue()

            async def _run_decode():
                try:
                    async for chunk in _stream_decode(
                        decode_client, api, decode_data, request_id, dp_rank=decode_hdr
                    ):
                        await queue.put(("data", chunk))
                except Exception as exc:
                    await queue.put(("error", exc))
                finally:
                    await queue.put(("done", None))

            # 1. Start decode request — decoder is now waiting for KV blocks
            asyncio.create_task(_run_decode())

            # 2. Send prefill — blocks are computed and pushed to the decoder
            try:
                await _prefill(
                    prefill_client, api, req_data, request_id, dp_rank=prefill_hdr
                )
            except Exception as exc:
                logger.warning("decoder-first: prefill failed: %s", exc)

            logger.debug(
                "decoder-first: prefill done, streaming decode prefill=%s decode=%s",
                prefill_client,
                decode_client,
            )

            # 3. Stream the decode response
            while True:
                kind, value = await queue.get()
                if kind == "done":
                    break
                if kind == "error":
                    raise value  # type: ignore[misc]
                yield value

        return StreamingResponse(generate(), media_type="application/json")

    except Exception as e:
        import sys
        import traceback

        print(f"Proxy error on {api}: {e}")
        print("".join(traceback.format_exception(*sys.exc_info())))
        raise


# ---------------------------------------------------------------------------
# Proactive P2P migration
# ---------------------------------------------------------------------------


def _offload_key_from_event(ev) -> bytes | None:
    """Reconstruct the OffloadKey bytes from a CPU BlockStored event.

    OffloadKey == block_hash_bytes + group_idx.to_bytes(4, "big") (see
    make_offload_key in vllm/v1/kv_offload/base.py). The event's last block
    hash is the chunk's key hash (placeholder events carry exactly one). Only
    CPU-tier events are relevant; GPU prefix-cache events share the batch.
    Returns None for anything that isn't a usable CPU store key — including
    int-mode hashes (VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES must be 0).
    """
    if not isinstance(ev, BlockStored):
        return None
    if ev.medium != "CPU" or ev.group_idx is None or not ev.block_hashes:
        return None
    h = ev.block_hashes[-1]
    if not isinstance(h, (bytes, bytearray)):
        return None
    return bytes(h) + int(ev.group_idx).to_bytes(4, "big", signed=False)


async def _kv_events_loop(app, sub):
    """Background task: decode KV-events and, while a migration is capturing,
    append reconstructed CPU OffloadKeys to the active collector."""
    decoder = msgspec.msgpack.Decoder(type=KVEventBatch)
    try:
        while True:
            frames = await sub.recv_multipart()  # (topic, seq, payload)
            try:
                batch = decoder.decode(frames[-1])
            except Exception as exc:
                logger.warning("KV event decode failed: %s", exc)
                continue
            collector = app.state.collector
            if collector is None:
                continue
            for ev in batch.events:
                key = _offload_key_from_event(ev)
                if key is not None:
                    collector.append(key)
    except asyncio.CancelledError:
        pass
    finally:
        sub.close(linger=0)


async def _rpc_call(host: str, port, obj: dict) -> dict:
    """POST a msgpack control-RPC to a server's /v1/kv_connector/rpc."""
    url = f"http://{host}:{port}/v1/kv_connector/rpc"
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(
            url,
            content=msgspec.msgpack.encode(obj),
            headers={"Content-Type": "application/octet-stream"},
        )
        resp.raise_for_status()
        return msgspec.msgpack.decode(resp.content)


async def _poll_migration(host: str, port, transfer_id: str) -> bool:
    """Poll a migration to a terminal state. Returns True iff completed with
    at least one block landed."""
    deadline = time.monotonic() + global_args.migration_timeout
    while time.monotonic() < deadline:
        st = await _rpc_call(host, port, {"op": "poll", "transfer_id": transfer_id})
        state = st.get("state")
        if state in ("completed", "failed", "unknown"):
            done = st.get("blocks_done") or 0
            # print (not logger): the module logger's INFO records do not reach
            # the proxy's stdout log under uvicorn's logging config, and the e2e
            # test greps this line for the migration outcome.
            print(
                f"MIGRATION transfer_id={transfer_id} state={state} "
                f"blocks_done={done} blocks_missing={st.get('blocks_missing')} "
                f"blocks_failed={st.get('blocks_failed')}",
                flush=True,
            )
            return state == "completed" and done > 0
        await asyncio.sleep(0.1)
    print(f"MIGRATION transfer_id={transfer_id} state=timeout", flush=True)
    return False


async def _handle_completions_proactive(api: str, request: Request):
    """Proactive flow: prefill -> capture KV-event block keys -> migrate to the
    decoder -> poll -> plain decode (served from the prefetched CPU cache)."""
    try:
        req_data = await request.json()
        request_id = str(uuid.uuid4())

        prefill_rank, prefill_hdr = _next_dp_rank(request.app, "prefill")
        decode_rank, decode_hdr = _next_dp_rank(request.app, "decode")
        prefill_client = _get_next(request.app, "prefill")
        decode_client = _get_next(request.app, "decode")
        dhost, dport = decode_client["host"], decode_client["port"]

        # Serialize the capture+migrate window so KV events map to this request.
        async with request.app.state.migration_lock:
            collector: list[bytes] = []
            request.app.state.collector = collector
            try:
                await _prefill(
                    prefill_client, api, req_data, request_id, dp_rank=prefill_hdr
                )
                await asyncio.sleep(0.3)  # drain trailing events
            finally:
                request.app.state.collector = None

            keys = list(dict.fromkeys(collector))  # dedup, keep order
            print(
                f"proactive {request_id}: captured {len(keys)} CPU block key(s) "
                f"from prefill",
                flush=True,
            )

            if keys:
                source = {
                    "host": global_args.p2p_connector_host,
                    "port": global_args.p2p_connector_port + prefill_rank,
                }
                ack = await _rpc_call(
                    dhost,
                    dport,
                    {
                        "op": "migrate",
                        "transfer_id": request_id,
                        "source": source,
                        "blocks": keys,
                    },
                )
                if ack.get("accepted"):
                    await _poll_migration(dhost, dport, request_id)
                else:
                    print(
                        f"proactive {request_id}: migrate rejected: {ack}",
                        flush=True,
                    )
            else:
                print(
                    f"proactive {request_id}: no CPU blocks captured; "
                    f"decode will recompute",
                    flush=True,
                )

        # Plain decode: no kv_transfer_params, so the decoder serves from the
        # CPU cache the migration just populated (or recomputes on a miss).
        decode_data = req_data.copy()
        decode_data.pop("kv_transfer_params", None)

        async def generate():
            async for chunk in _stream_decode(
                decode_client, api, decode_data, request_id, dp_rank=decode_hdr
            ):
                yield chunk

        return StreamingResponse(generate(), media_type="application/json")

    except Exception as e:
        import sys
        import traceback

        print(f"Proxy error on {api}: {e}")
        print("".join(traceback.format_exception(*sys.exc_info())))
        raise


def _route_handler(api: str):
    if global_args.proactive:
        return lambda req: _handle_completions_proactive(api, req)
    if global_args.decoder_first:
        return lambda req: _handle_completions_decoder_first(api, req)
    return lambda req: _handle_completions(api, req)


@app.post("/v1/completions")
async def completions(request: Request):
    return await _route_handler("/completions")(request)


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    return await _route_handler("/chat/completions")(request)


@app.get("/healthcheck")
async def healthcheck():
    return {
        "status": "ok",
        "prefill_instances": len(app.state.prefill_clients),
        "decode_instances": len(app.state.decode_clients),
    }


if __name__ == "__main__":
    global global_args
    global_args = parse_args()
    import uvicorn

    uvicorn.run(app, host=global_args.host, port=global_args.port)
