# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A stand-in for llm-d's EPP, for the two-pod secondary-tier pull test.

This is deliberately *not* the real EPP. It replaces exactly one thing the EPP
does -- choosing which replica holds the most reusable prefix -- and nothing
else. The code under test is the routing **sidecar**: turning the source header
into a request field (``kv_transfer_params.remote_kv_source`` for the P2P tier,
a ``kv_hints`` envelope for KVCR), deriving the peer's tier endpoint, and
building an envelope the engine's parser accepts. None of that lives in the
EPP, and the real EPP needs InferencePool CRDs, a gateway and a KV-event
indexer to start at all -- far more moving parts than the thing being tested.

What it does, per request:

* pick a target sidecar, randomly by default (the scenario under test: the
  router picks a replica and tells it to pull the prefix from the other one),
* set ``x-llm-d-kv-cache-source-host-port`` to the *other* pod's sidecar,
* forward the body untouched.

Like the real EPP it strips any inbound value of that header, so a client
cannot choose its own source.

Test hooks, which the real EPP has no equivalent of:

* ``x-test-target: 0|1`` pins the target instead of sampling, so a test can
  seed one pod and pull from the other deterministically.
* ``x-test-no-hint: 1`` suppresses the source header, which is how a seed
  request is sent (compute locally, offload, advertise nothing).

Usage:
    python3 epp_stub.py --port 8080 \
        --backends 10.0.1.136:8101,10.0.3.38:8201 [--seed 1234]
"""

import argparse
import logging
import random
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

logger = logging.getLogger("epp_stub")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# The header the EPP sets and the sidecar consumes. Canonical spelling; the
# sidecar prefers the legacy alias but accepts either.
KV_CACHE_SOURCE_HEADER = "x-llm-d-kv-cache-source-host-port"
LEGACY_KV_CACHE_SOURCE_HEADER = "x-kv-cache-source-host-port"

TARGET_HEADER = "x-test-target"
NO_HINT_HEADER = "x-test-no-hint"

# Hop-by-hop and length headers must not be copied onto the upstream request.
_SKIP_REQUEST_HEADERS = {"host", "content-length", "connection", "transfer-encoding"}

global_args: argparse.Namespace
rng: random.Random


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.client = httpx.AsyncClient(timeout=httpx.Timeout(600.0))
    logger.info(
        "EPP stub ready: backends=%s mode=%s",
        global_args.backends,
        "pinned-or-random",
    )
    yield
    await app.state.client.aclose()


app = FastAPI(lifespan=lifespan)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument(
        "--backends",
        required=True,
        help="comma-separated sidecar endpoints, host:port,host:port (exactly 2)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="seed for target sampling; omit for a nondeterministic run",
    )
    args = parser.parse_args()
    args.backends = [b.strip() for b in args.backends.split(",") if b.strip()]
    if len(args.backends) != 2:
        parser.error(f"--backends needs exactly 2 endpoints, got {args.backends}")
    return args


def _choose_target(request: Request) -> tuple[int, bool]:
    """Return (target index, whether to attach a source hint)."""
    pinned = request.headers.get(TARGET_HEADER)
    if pinned is not None:
        try:
            index = int(pinned)
        except ValueError as exc:
            raise ValueError(f"{TARGET_HEADER} must be 0 or 1, got {pinned!r}") from exc
        if index not in (0, 1):
            raise ValueError(f"{TARGET_HEADER} must be 0 or 1, got {index}")
    else:
        index = rng.randrange(len(global_args.backends))

    attach_hint = request.headers.get(NO_HINT_HEADER, "") not in ("1", "true", "True")
    return index, attach_hint


def _upstream_headers(request: Request, source: str | None) -> dict[str, str]:
    headers = {
        key: value
        for key, value in request.headers.items()
        if key.lower() not in _SKIP_REQUEST_HEADERS
    }
    # Strip any client-supplied source under either spelling before deciding.
    headers.pop(KV_CACHE_SOURCE_HEADER, None)
    headers.pop(LEGACY_KV_CACHE_SOURCE_HEADER, None)
    headers.pop(TARGET_HEADER, None)
    headers.pop(NO_HINT_HEADER, None)
    if source is not None:
        headers[KV_CACHE_SOURCE_HEADER] = source
    return headers


async def _forward(path: str, request: Request):
    body = await request.body()
    try:
        index, attach_hint = _choose_target(request)
    except ValueError as exc:
        return JSONResponse(status_code=400, content={"error": str(exc)})

    target = global_args.backends[index]
    source = global_args.backends[1 - index] if attach_hint else None

    logger.info(
        "route path=%s target=%s source=%s",
        path,
        target,
        source if source is not None else "<none>",
    )

    client: httpx.AsyncClient = app.state.client
    url = f"http://{target}{path}"
    upstream = client.build_request(
        "POST", url, content=body, headers=_upstream_headers(request, source)
    )
    response = await client.send(upstream, stream=True)

    async def body_iter():
        try:
            async for chunk in response.aiter_raw():
                yield chunk
        finally:
            await response.aclose()

    # Echo the routing decision so a test can assert on it without log scraping.
    passthrough = {
        key: value
        for key, value in response.headers.items()
        if key.lower() not in ("content-length", "transfer-encoding", "connection")
    }
    passthrough["x-test-routed-to"] = target
    if source is not None:
        passthrough["x-test-routed-source"] = source

    return StreamingResponse(
        body_iter(), status_code=response.status_code, headers=passthrough
    )


@app.post("/v1/completions")
async def completions(request: Request):
    return await _forward("/v1/completions", request)


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    return await _forward("/v1/chat/completions", request)


@app.get("/healthz")
async def healthz():
    return {"status": "ok", "backends": global_args.backends}


if __name__ == "__main__":
    import uvicorn

    global_args = parse_args()
    rng = random.Random(global_args.seed)
    uvicorn.run(app, host=global_args.host, port=global_args.port, log_level="warning")
