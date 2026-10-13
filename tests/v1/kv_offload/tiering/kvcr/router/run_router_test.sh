#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Run the pull test against an environment deploy_router_env.sh brought up.
#
#   ./run_router_test.sh p2p            # deterministic seed/pull, both directions
#   ./run_router_test.sh kvcr
#   ./run_router_test.sh kvcr random    # router picks freely, hints the other pod
#
# The arm argument only labels the output and the saved results; which protocol
# is actually in play is decided by how the environment was deployed.
#
# MODEL and the pod/port variables come from router_env.sh, which shellcheck
# cannot follow through the computed source path.
# shellcheck disable=SC2153
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=router_env.sh
source "$SCRIPT_DIR/router_env.sh"

ARM="${1:-}"
MODE="${2:-pair}"
TRIALS="${TRIALS:-8}"
case "$ARM" in
    p2p | kvcr) ;;
    *)
        echo "usage: $0 {p2p|kvcr} [pair|random|control]" >&2
        exit 2
        ;;
esac

die() {
    echo "ERROR: $*" >&2
    exit 1
}

IP_B="$(pod_ip "$POD_B")" || die "cannot read $POD_B IP"
[[ -n "$IP_B" ]] || die "empty IP for $POD_B"

REMOTE_TEST="$POD_WORKDIR/router_pull_test.py"
oc cp "$SCRIPT_DIR/router_pull_test.py" "$POD_A:$REMOTE_TEST" ||
    die "cannot stage router_pull_test.py"

STAMP="$(date +%Y%m%d-%H%M%S)"
REMOTE_JSON="$LOG_DIR/results-$ARM-$MODE-$STAMP.json"

echo "=== arm=$ARM mode=$MODE trials=$TRIALS ==="
oc exec "$POD_A" -- bash -lc "
    cd / && python3 '$REMOTE_TEST' \
        --epp localhost:$EPP_PORT \
        --vllm-a localhost:$VLLM_PORT_A \
        --vllm-b $IP_B:$VLLM_PORT_B \
        --model '$MODEL' \
        --trials $TRIALS \
        --mode $MODE \
        --json-out '$REMOTE_JSON'
"
status=$?

RESULTS_DIR="$SCRIPT_DIR/results"
mkdir -p "$RESULTS_DIR"
if oc cp "$POD_A:$REMOTE_JSON" "$RESULTS_DIR/results-$ARM-$MODE-$STAMP.json" 2>/dev/null; then
    echo "results: $RESULTS_DIR/results-$ARM-$MODE-$STAMP.json"
fi

exit $status
