# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from http import HTTPStatus

from fastapi import APIRouter, FastAPI, HTTPException, Query, Request
from fastapi.responses import Response

from vllm.engine.protocol import EngineClient
from vllm.logger import init_logger

logger = init_logger(__name__)

router = APIRouter()

OCTET_STREAM = "application/octet-stream"


def engine_client(request: Request) -> EngineClient:
    return request.app.state.engine_client


@router.post("/v1/kv_connector/rpc")
async def kv_connector_rpc(
    raw_request: Request,
    dp_rank: int | None = Query(default=None, ge=0),
):
    """Deliver a generic control RPC to the scheduler-side KV connector.

    The request body is the connector-defined request payload (raw bytes);
    vLLM never parses it. The bytes are routed unchanged to the connector's
    `on_rpc` hook running in the EngineCore scheduler process, and its
    response bytes are returned verbatim.

    The RPC runs on exactly one engine. With data parallelism, pass
    `?dp_rank=N` to choose it; omitting `dp_rank` targets the single engine
    this server fronts and is rejected if it fronts more than one.

    !!! warning "Experimental"
        This endpoint and the `KVConnectorBase_V1.on_rpc` hook are
        experimental. Their path, parameters, status codes, and semantics may
        change or be removed in any release without a deprecation period.

    Status codes:
        200: The connector handled the RPC. The body is the response bytes
            (an empty body is a valid success/ack).
        400: `dp_rank` is missing while this server fronts several engines,
            or names an engine this server does not front.
        404: No KV connector is configured.
        501: The connector does not implement the RPC hook.
        500: The connector's handler raised.
    """
    client = engine_client(raw_request)

    if client.vllm_config.kv_transfer_config is None:
        raise HTTPException(
            status_code=HTTPStatus.NOT_FOUND.value,
            detail="No KV connector is configured.",
        )

    payload = await raw_request.body()
    try:
        result = await client.kv_connector_rpc(payload, dp_rank=dp_rank)
    except ValueError as e:
        # Only target resolution raises ValueError locally; connector
        # handler errors arrive from the engine as a plain Exception (500).
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST.value, detail=str(e)
        ) from e

    if result is None:
        raise HTTPException(
            status_code=HTTPStatus.NOT_IMPLEMENTED.value,
            detail="The configured KV connector does not implement on_rpc().",
        )

    return Response(content=bytes(result), media_type=OCTET_STREAM)


def attach_router(app: FastAPI):
    app.include_router(router)
