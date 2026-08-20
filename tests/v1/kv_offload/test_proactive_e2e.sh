#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# End-to-end correctness test for proactive P2P KV migration.
#
# Prereq: a deployment created with PROACTIVE_MODE=true, e.g.
#   PROACTIVE_MODE=true CONNECTOR=pd_connector bash deploy.sh --config <cluster_pd.env>
#
# What it checks:
#   1. Correctness — the proxy's proactive flow (prefill -> migrate KV to the
#      decoder -> decode from the prefetched CPU cache) produces the SAME greedy
#      output as a direct generation on the prefiller. The prefiller is the
#      ground-truth oracle: it holds the KV that was copied, and it computes
#      independently of the decoder's (possibly wrongly-populated) cache, so this
#      catches a corrupt migration that a decoder-vs-decoder compare would miss.
#   2. Efficacy — the proxy log shows the migration reached state=completed with
#      blocks_done > 0 (KV actually moved, not silently recomputed).
#
# Usage:
#   bash test_proactive_e2e.sh                 # uses /tmp/deploy_state.env
#   STATE_FILE=/path/state.env bash test_proactive_e2e.sh

set -euo pipefail

STATE_FILE="${STATE_FILE:-/tmp/deploy_state.env}"
if [[ ! -f "$STATE_FILE" ]]; then
    echo "ERROR: state file '$STATE_FILE' not found; run deploy.sh first." >&2
    exit 1
fi
# shellcheck disable=SC1090
source "$STATE_FILE"

PROXY_LOG="${PROXY_LOG:-/tmp/proxy.log}"
MAX_TOKENS="${MAX_TOKENS:-32}"

if [[ "${PROACTIVE_MODE:-false}" != "true" ]]; then
    echo "ERROR: deployment was not PROACTIVE_MODE=true (state: PROACTIVE_MODE=${PROACTIVE_MODE:-unset})." >&2
    exit 1
fi
command -v python3 >/dev/null 2>&1 || { echo "ERROR: python3 required locally." >&2; exit 1; }

# Unique prompt (epoch-seeded so it never hits a stale prefix cache) that is
# long enough to span several KV blocks so there is real KV to migrate.
SEED="$(date +%s)"
PROMPT="proactive-e2e run ${SEED}. "
for i in $(seq 1 60); do
    PROMPT+="the quick brown fox jumps over the lazy dog iteration ${i}. "
done
PROMPT="${PROMPT//\"/}"   # defensive: no embedded quotes

BODY="$(python3 -c '
import json, sys
print(json.dumps({
    "model": sys.argv[1],
    "prompt": sys.argv[2],
    "max_tokens": int(sys.argv[3]),
    "temperature": 0,
    "stream": False,
}))' "$MODEL" "$PROMPT" "$MAX_TOKENS")"

extract_text() {
    python3 -c 'import sys, json; print(json.load(sys.stdin)["choices"][0]["text"])'
}

echo "=== 1/3: proactive request via proxy (${PROXY_POD}) ==="
R_PROXY="$(oc exec "${PROXY_POD}" -- curl -sS --max-time 180 \
    "http://127.0.0.1:${PROXY_PORT}/v1/completions" \
    -H 'Content-Type: application/json' -d "${BODY}")"
O_PROXY="$(printf '%s' "$R_PROXY" | extract_text)"

echo "=== 2/3: reference generation direct on prefiller (${PREFILLER_POD}) ==="
R_REF="$(oc exec "${PREFILLER_POD}" -- curl -sS --max-time 180 \
    "http://127.0.0.1:${PREFILLER_HTTP_PORT}/v1/completions" \
    -H 'Content-Type: application/json' -d "${BODY}")"
O_REF="$(printf '%s' "$R_REF" | extract_text)"

echo "=== 3/3: migration outcome from proxy log ==="
MLINE="$(oc exec "${PROXY_POD}" -- grep 'MIGRATION ' "${PROXY_LOG}" 2>/dev/null | tail -1 || true)"

echo
echo "--- proxy (decoder, migrated KV): ${O_PROXY}"
echo "--- reference (prefiller):        ${O_REF}"
echo "--- migration:                    ${MLINE:-<none found>}"
echo

FAIL=0

if [[ -z "$O_PROXY" ]]; then
    echo "FAIL: empty proxy output"; FAIL=1
elif [[ "$O_PROXY" == "$O_REF" ]]; then
    echo "PASS: proactive output matches the prefiller reference (correct KV)."
else
    echo "FAIL: proactive output differs from the prefiller reference."; FAIL=1
fi

if [[ "$MLINE" == *"state=completed"* ]]; then
    DONE="$(sed -n 's/.*blocks_done=\([0-9]*\).*/\1/p' <<<"$MLINE")"
    if [[ "${DONE:-0}" -gt 0 ]]; then
        echo "PASS: migration completed with blocks_done=${DONE} (KV moved)."
    else
        echo "FAIL: migration completed but blocks_done=0 (nothing migrated)."; FAIL=1
    fi
else
    echo "FAIL: no completed migration found in ${PROXY_LOG}."; FAIL=1
fi

echo
if [[ $FAIL -eq 0 ]]; then
    echo "=== PASS ==="
else
    echo "=== FAIL ==="
fi
exit $FAIL
