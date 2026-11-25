// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::sync::Arc;

use axum::body::Bytes;
use axum::extract::rejection::QueryRejection;
use axum::extract::{Query, State};
use axum::http::{StatusCode, header};
use axum::response::IntoResponse;
use serde::Deserialize;
use vllm_engine_core_client::Error as EngineCoreError;

use crate::error::ApiError;
use crate::state::AppState;
use crate::utils::utility_call_error;

#[derive(Debug, Default, Deserialize)]
pub(crate) struct KvConnectorRpcParams {
    /// Data-parallel rank of the engine to run the RPC on.
    dp_rank: Option<u32>,
}

/// Deliver a generic control RPC to the scheduler-side KV connector.
///
/// The request body is the connector-defined request payload (raw bytes);
/// vLLM never parses it. The bytes are routed unchanged to the connector's
/// `on_rpc` hook running in the EngineCore scheduler process, and its
/// response bytes are returned verbatim.
///
/// The RPC runs on exactly one engine. With data parallelism, pass
/// `?dp_rank=N` to choose it; omitting `dp_rank` targets the single engine
/// this server fronts and is rejected if it fronts more than one.
///
/// Experimental: this endpoint and the `KVConnectorBase_V1.on_rpc` hook are
/// experimental. Their path, parameters, status codes, and semantics may
/// change or be removed in any release without a deprecation period.
///
/// Status codes:
/// - 200: The connector handled the RPC. The body is the response bytes (an
///   empty body is a valid success/ack).
/// - 400: `dp_rank` is missing while this server fronts several engines, or
///   names an engine this server does not front. A negative or non-integer
///   `dp_rank` is a query validation error, also 400.
/// - 404: No KV connector is configured.
/// - 501: The connector does not implement the RPC hook.
/// - 500: The connector's handler raised.
pub async fn kv_connector_rpc(
    State(state): State<Arc<AppState>>,
    params: Result<Query<KvConnectorRpcParams>, QueryRejection>,
    payload: Bytes,
) -> Result<impl IntoResponse, ApiError> {
    let Query(params) = params?;
    let client = state.engine_core_client();

    // TODO: engines report their `VllmConfig` only under
    // `VLLM_SERVER_DEV_MODE`. Without it this frontend cannot tell whether a
    // KV connector is configured, so the RPC is dispatched and an engine
    // without one answers 501 where Python answers 404.
    let no_kv_connector = client.ready_response().vllm_config.as_ref().is_some_and(|config| {
        config.pointer("/kv_transfer_config").is_none_or(serde_json::Value::is_null)
    });
    if no_kv_connector {
        return Err(ApiError::http_exception(
            StatusCode::NOT_FOUND,
            "No KV connector is configured.",
        ));
    }

    let reply = client.kv_connector_rpc(payload, params.dp_rank).await.map_err(|error| {
        match error {
            // Only target resolution fails locally; connector handler errors
            // arrive from the engine as a failed utility call (500).
            EngineCoreError::InvalidUtilityTarget { message } => {
                ApiError::http_exception(StatusCode::BAD_REQUEST, message)
            }
            error => utility_call_error("kv_connector_rpc", error),
        }
    })?;

    let Some(reply) = reply else {
        return Err(ApiError::http_exception(
            StatusCode::NOT_IMPLEMENTED,
            "The configured KV connector does not implement on_rpc().",
        ));
    };

    Ok(([(header::CONTENT_TYPE, "application/octet-stream")], reply))
}
