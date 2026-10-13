#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Deploy the two-pod router + secondary-tier pull environment.
#
#   ./deploy_router_env.sh p2p     # baseline: OffloadingConnector P2P tier
#   ./deploy_router_env.sh kvcr    # KVCR tier, driven by a kv_hints envelope
#   ./deploy_router_env.sh down    # tear everything down
#
# Both arms serve the same model from the same two pods and differ only in the
# engine's secondary tier and the sidecar's --kv-connector. That is the point:
# the baseline arm proves the environment works, so a failure in the kvcr arm
# is about kv_hints and not about the pods.
#
# This is NOT a P/D deployment. Each pod runs a full engine behind its own
# routing sidecar, and the EPP stand-in picks one per request and tells it to
# pull the prefix from the other. No prefiller header is ever set, so the
# sidecar takes its decoder-only path.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=router_env.sh
source "$SCRIPT_DIR/router_env.sh"

ARM="${1:-}"
case "$ARM" in
    p2p | kvcr | down) ;;
    *)
        echo "usage: $0 {p2p|kvcr|down}" >&2
        exit 2
        ;;
esac

log() { echo "[$(date +%H:%M:%S)] $*"; }
die() {
    echo "ERROR: $*" >&2
    exit 1
}

# --- Teardown ----------------------------------------------------------------
# pkill -f 'VLLM::' matches every engine process, not just EngineCore: a stale
# APIServer keeps the port alive through SO_REUSEPORT and makes the next deploy
# look half-up. The /dev/shm sweep matters because the offload mmap outlives
# the engine and the third restart otherwise dies on "Insufficient space".
#
# Every pattern here is bracketed ("vllm[ ]serve", not "vllm serve"). pkill -f
# matches against full command lines, and the remote shell's own command line
# contains this script text -- so an unbracketed pattern makes pkill SIGTERM its
# own shell mid-teardown. That fails silently through `oc exec`, leaves the old
# engine running, and the health probe then passes against the *previous* arm's
# config, which is a deeply confusing way to get a wrong result.
teardown_pod() {
    local pod="$1"
    log "tearing down $pod"
    oc exec "$pod" -- bash -lc '
        pkill -f "epp[_]stub.py" 2>/dev/null
        pkill -f "pd[-]sidecar" 2>/dev/null
        pkill -f "VLLM[:]:" 2>/dev/null
        pkill -f "vllm[ ]serve" 2>/dev/null
        for _ in $(seq 60); do
            pgrep -f "VLLM[:]:" >/dev/null 2>&1 || break
            sleep 1
        done
        if pgrep -f "VLLM[:]:" >/dev/null 2>&1; then
            pkill -9 -f "VLLM[:]:" 2>/dev/null
            pkill -9 -f "vllm[ ]serve" 2>/dev/null
            sleep 3
        fi
        rm -f /dev/shm/vllm_offload_*.mmap /dev/shm/psm_* 2>/dev/null
        true
    ' >/dev/null 2>&1
    return 0
}

# Surviving engine processes, as a bare integer. pgrep -c prints the count and
# exits non-zero when it is 0, hence the `|| true`.
count_vllm() {
    oc exec "$1" -- bash -lc 'pgrep -cf "VLLM[:]:" 2>/dev/null || true' 2>/dev/null |
        tr -cd '0-9'
}

if [[ "$ARM" == "down" ]]; then
    teardown_pod "$POD_A"
    teardown_pod "$POD_B"
    log "done"
    exit 0
fi

IP_A="$(pod_ip "$POD_A")" || die "cannot read $POD_A IP"
IP_B="$(pod_ip "$POD_B")" || die "cannot read $POD_B IP"
[[ -n "$IP_A" && -n "$IP_B" ]] || die "empty pod IP (A=$IP_A B=$IP_B)"
log "arm=$ARM  $POD_A=$IP_A  $POD_B=$IP_B"

# --- Engine KV config --------------------------------------------------------
# control_advertise_host must be the pod IP: KVCR defaults it to the hostname,
# which resolves only through a headless service, and the peer would then be
# told to connect somewhere it cannot reach.
kv_config() {
    local advertise_host="$1"
    if [[ "$ARM" == "p2p" ]]; then
        # The P2P tier takes its host/port from VLLM_P2P_SIDE_CHANNEL_*, set in
        # start_vllm, not from this config block.
        printf '%s' "{\"kv_connector\":\"OffloadingConnector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"spec_name\":\"TieringOffloadingSpec\",\"cpu_bytes_to_use\":${CPU_BYTES},\"secondary_tiers\":[{\"type\":\"p2p\"}]}}"
    else
        # opportunistic_query is deliberately left unset, i.e. the adapter
        # default of true. The router sends a placeholder block_hashes list, so
        # a pod with opportunistic_query=false would compare it and every hint
        # would miss silently.
        printf '%s' "{\"kv_connector\":\"OffloadingConnector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"spec_name\":\"TieringOffloadingSpec\",\"cpu_bytes_to_use\":${CPU_BYTES},\"secondary_tiers\":[{\"type\":\"kvcr\",\"router_capabilities\":[\"router_hint\"],\"control_host\":\"0.0.0.0\",\"control_ports\":[${KVCR_CONTROL_PORT}],\"control_advertise_host\":\"${advertise_host}\",\"secondary_g2_slots\":${KVCR_G2_SLOTS},\"enable_telemetry\":true,\"operation_timeout_ms\":5000,\"abandon_timeout_ms\":10000}]}}"
    fi
}

start_vllm() {
    local pod="$1" port="$2" advertise_host="$3"
    local cfg
    cfg="$(kv_config "$advertise_host")"

    # VLLM_P2P_SIDE_CHANNEL_HOST is the identity a peer dials back, used
    # verbatim. Its default is "localhost", which a remote pod can never
    # reach, so it has to be this pod's IP for a cross-pod pull to work at
    # all. Harmless in the kvcr arm, where the tier reads
    # control_advertise_host instead.
    log "starting vLLM on $pod:$port"
    oc exec "$pod" -- bash -lc "
        mkdir -p '$LOG_DIR'
        cd /
        PYTHONHASHSEED=$PYTHONHASHSEED_VALUE \
        VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES=1 \
        VLLM_P2P_SIDE_CHANNEL_HOST=$advertise_host \
        VLLM_P2P_SIDE_CHANNEL_PORT=$P2P_PORT \
        nohup vllm serve '$MODEL' \
            --port $port \
            --tensor-parallel-size 1 \
            --enforce-eager \
            --block-size $BLOCK_SIZE \
            --gpu-memory-utilization $GPU_MEM_UTIL \
            --max-model-len $MAX_MODEL_LEN \
            --kv-transfer-config '$cfg' \
            > '$LOG_DIR/vllm-$port.log' 2>&1 &
        echo started
    " >/dev/null 2>&1 || die "failed to launch vLLM on $pod"
}

wait_http() {
    local pod="$1" url="$2" what="$3" tries="${4:-180}"
    log "waiting for $what on $pod"
    for ((i = 0; i < tries; i++)); do
        if oc exec "$pod" -- bash -lc "curl -sf -o /dev/null '$url'" >/dev/null 2>&1; then
            log "$what is up"
            return 0
        fi
        sleep 2
    done
    return 1
}

# --- Sidecar -----------------------------------------------------------------
# --kv-connector picks the request field: offloading injects
# kv_transfer_params.remote_kv_source, kvcr injects a kv_hints envelope. The
# tier port flag is the PEER's listener; both pods use the same number, so one
# value serves both directions.
start_sidecar() {
    local pod="$1" sidecar_port="$2" vllm_port="$3" pod_ip="$4"
    local connector_flags
    if [[ "$ARM" == "p2p" ]]; then
        connector_flags="--kv-connector=offloading --p2p-connector-port=$P2P_PORT"
    else
        connector_flags="--kv-connector=kvcr --kvcr-control-port=$KVCR_CONTROL_PORT"
    fi

    log "starting sidecar on $pod:$sidecar_port -> vLLM:$vllm_port ($connector_flags)"
    oc exec "$pod" -- bash -lc "
        mkdir -p '$LOG_DIR'
        POD_IP=$pod_ip nohup '$SIDECAR_BIN' \
            --port=$sidecar_port \
            --model-server-port=$vllm_port \
            $connector_flags \
            --secure-serving=false \
            --enable-ssrf-protection=false \
            --zap-log-level=debug \
            > '$LOG_DIR/sidecar-$sidecar_port.log' 2>&1 &
        echo started
    " >/dev/null 2>&1 || die "failed to launch sidecar on $pod"
}

# --- Bring-up ----------------------------------------------------------------
teardown_pod "$POD_A"
teardown_pod "$POD_B"

# Assert the teardown actually worked. Without this, a surviving engine answers
# /health in seconds and the whole run silently measures the previous arm.
for pod in "$POD_A" "$POD_B"; do
    remaining="$(count_vllm "$pod")"
    remaining="${remaining:-0}"
    log "$pod: $remaining engine processes remain"
    [[ "$remaining" == "0" ]] ||
        die "$pod still has $remaining vLLM processes after teardown"
done

oc cp "$SCRIPT_DIR/epp_stub.py" "$POD_A:$EPP_STUB" || die "cannot stage epp_stub.py"
for pod in "$POD_A" "$POD_B"; do
    if ! oc exec "$pod" -- bash -lc "test -x '$SIDECAR_BIN'" >/dev/null 2>&1; then
        die "sidecar binary missing on $pod at $SIDECAR_BIN -- run stage_sidecar.sh first"
    fi
done

start_vllm "$POD_A" "$VLLM_PORT_A" "$IP_A"
start_vllm "$POD_B" "$VLLM_PORT_B" "$IP_B"
wait_http "$POD_A" "http://localhost:$VLLM_PORT_A/health" "vLLM $VLLM_PORT_A" ||
    die "vLLM on $POD_A never became healthy; see $LOG_DIR/vllm-$VLLM_PORT_A.log"
wait_http "$POD_B" "http://localhost:$VLLM_PORT_B/health" "vLLM $VLLM_PORT_B" ||
    die "vLLM on $POD_B never became healthy; see $LOG_DIR/vllm-$VLLM_PORT_B.log"

start_sidecar "$POD_A" "$SIDECAR_PORT_A" "$VLLM_PORT_A" "$IP_A"
start_sidecar "$POD_B" "$SIDECAR_PORT_B" "$VLLM_PORT_B" "$IP_B"
# The sidecar serves its own GET /health, so this probes the sidecar itself
# rather than proxying through to the engine (already checked above).
wait_http "$POD_A" "http://localhost:$SIDECAR_PORT_A/health" "sidecar $SIDECAR_PORT_A" 60 ||
    die "sidecar on $POD_A not answering; see $LOG_DIR/sidecar-$SIDECAR_PORT_A.log"
wait_http "$POD_B" "http://localhost:$SIDECAR_PORT_B/health" "sidecar $SIDECAR_PORT_B" 60 ||
    die "sidecar on $POD_B not answering; see $LOG_DIR/sidecar-$SIDECAR_PORT_B.log"

log "starting EPP stub on $POD_A:$EPP_PORT"
oc exec "$POD_A" -- bash -lc "
    mkdir -p '$LOG_DIR'
    cd /
    nohup python3 '$EPP_STUB' \
        --port $EPP_PORT \
        --backends $IP_A:$SIDECAR_PORT_A,$IP_B:$SIDECAR_PORT_B \
        --seed 1234 \
        > '$LOG_DIR/epp.log' 2>&1 &
    echo started
" >/dev/null 2>&1 || die "failed to launch EPP stub"
wait_http "$POD_A" "http://localhost:$EPP_PORT/healthz" "EPP stub" 60 ||
    die "EPP stub not answering; see $LOG_DIR/epp.log"

cat <<EOF

Environment is up (arm=$ARM).

  EPP stub      $POD_A:$EPP_PORT   (backends $IP_A:$SIDECAR_PORT_A, $IP_B:$SIDECAR_PORT_B)
  sidecar A     $POD_A:$SIDECAR_PORT_A -> vLLM $VLLM_PORT_A
  sidecar B     $POD_B:$SIDECAR_PORT_B -> vLLM $VLLM_PORT_B
  logs          <pod>:$LOG_DIR/

Run the test:
  ./run_router_test.sh $ARM
EOF
