# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Shared configuration for the two-pod router + secondary-tier pull test.
# Sourced by deploy_router_env.sh; override any value from the environment.

# --- Cluster -----------------------------------------------------------------
: "${KUBECONFIG:=$HOME/CWKubeconfig_waldorf_US-EAST-04A}"
export KUBECONFIG

# POD_A seeds the KV, POD_B pulls it (and vice versa in random mode). Both run
# an identical stack; the names are historical and carry no P/D role here.
: "${POD_A:=llmd-prefiller}"
: "${POD_B:=llmd-decoder}"

# --- Model -------------------------------------------------------------------
# Llama, not Qwen: Qwen3 emits garbled output on these pods even with healthy
# KV transport.
: "${MODEL:=meta-llama/Llama-3.2-1B}"
: "${BLOCK_SIZE:=16}"
: "${MAX_MODEL_LEN:=8192}"
: "${GPU_MEM_UTIL:=0.90}"

# --- Ports -------------------------------------------------------------------
: "${VLLM_PORT_A:=8100}"
: "${VLLM_PORT_B:=8200}"
# Routing sidecar, one per pod, in front of the local vLLM.
: "${SIDECAR_PORT_A:=8101}"
: "${SIDECAR_PORT_B:=8201}"
# The EPP stand-in; runs on POD_A and fans out to both sidecars.
: "${EPP_PORT:=8080}"
# Secondary-tier listeners. Same number on both pods, so one sidecar flag value
# serves both directions.
: "${P2P_PORT:=7777}"
: "${KVCR_CONTROL_PORT:=17771}"

# --- Tier sizing -------------------------------------------------------------
# 2 GiB of host tier on both pods. The source pod must still hold the seeded
# blocks when the puller asks for them, and a tier sized near the working set
# evicts them first (see the #58168 sweep).
: "${CPU_BYTES:=2147483648}"
# KVCR-owned pool slots. KVCR's deposit path fails outright with no pool
# configured, so give both pods one even though index-and-pin would serve peer
# reads without it.
: "${KVCR_G2_SLOTS:=4096}"

# Block hashes are computed independently on each pod and must agree, or a pull
# silently misses. The non-crypto hash algos seed NONE_HASH per process, so the
# seed has to be pinned identically on BOTH pods. Any fixed value works; what
# matters is that it is the same one.
: "${PYTHONHASHSEED_VALUE:=12345}"

# --- Paths on the pods -------------------------------------------------------
: "${POD_WORKDIR:=/vllm-workspace}"
: "${SIDECAR_BIN:=$POD_WORKDIR/pd-sidecar}"
: "${EPP_STUB:=$POD_WORKDIR/epp_stub.py}"
: "${LOG_DIR:=$POD_WORKDIR/router-test-logs}"

pod_ip() { oc get pod "$1" -o jsonpath='{.status.podIP}'; }
