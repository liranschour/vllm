#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Build the llm-d routing sidecar and stage it on both pods.
#
#   ROUTER_SRC=~/llm-d-router ./stage_sidecar.sh
#
# Built statically (CGO_ENABLED=0) so it runs on the vLLM image regardless of
# its glibc, and shipped with `oc cp` so the pods need no Go toolchain. Re-run
# this after changing sidecar code; deploy_router_env.sh refuses to start if
# the binary is missing but cannot tell a stale one from a fresh one.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=router_env.sh
source "$SCRIPT_DIR/router_env.sh"

: "${ROUTER_SRC:=$HOME/llm-d-router}"
BUILD_OUT="${BUILD_OUT:-${TMPDIR:-/tmp}/pd-sidecar}"

# go.mod requires a Go newer than most distros ship, so this prefers a
# user-local SDK and falls back to whatever is on PATH.
if [[ -z "${GO_BIN:-}" ]]; then
    if [[ -x "$HOME/.local/go-sdk/go/bin/go" ]]; then
        GO_BIN="$HOME/.local/go-sdk/go/bin/go"
    else
        GO_BIN="$(command -v go || true)"
    fi
fi

log() { echo "[$(date +%H:%M:%S)] $*"; }
die() {
    echo "ERROR: $*" >&2
    exit 1
}

[[ -d "$ROUTER_SRC" ]] || die "router source not found at $ROUTER_SRC (set ROUTER_SRC)"
[[ -n "$GO_BIN" && -x "$GO_BIN" ]] || die "no Go toolchain found. Install the version
go.mod requires and set GO_BIN, e.g.:
  curl -fsSL https://go.dev/dl/go1.27.2.linux-amd64.tar.gz -o /tmp/go.tar.gz
  mkdir -p ~/.local/go-sdk && tar -C ~/.local/go-sdk -xzf /tmp/go.tar.gz"
log "go: $("$GO_BIN" version)"

log "building $ROUTER_SRC/cmd/pd-sidecar -> $BUILD_OUT"
(
    cd "$ROUTER_SRC" || exit 1
    CGO_ENABLED=0 GOOS=linux GOARCH=amd64 "$GO_BIN" build -o "$BUILD_OUT" ./cmd/pd-sidecar
) || die "build failed"

log "sidecar commit: $(git -C "$ROUTER_SRC" rev-parse --short HEAD 2>/dev/null || echo unknown)"

for pod in "$POD_A" "$POD_B"; do
    log "staging to $pod:$SIDECAR_BIN"
    oc cp "$BUILD_OUT" "$pod:$SIDECAR_BIN" || die "oc cp to $pod failed"
    oc exec "$pod" -- bash -lc "chmod +x '$SIDECAR_BIN' && '$SIDECAR_BIN' --help >/dev/null 2>&1 && echo ok" |
        grep -q ok || die "staged binary does not run on $pod"
done

log "staged on both pods"
