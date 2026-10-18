#!/bin/bash
# usage: serve.sh <trace_prefix>  (MODEL, NO_KV, VLLM_LOCK_PROBE_* via env)
W=/workspace/lockhold
rm -rf $W/fs_tier; mkdir -p $W/fs_tier
export VLLM_LOCK_TRACE_FILE=$1
MODEL=${MODEL:-meta-llama/Llama-3.1-8B-Instruct}
KVCFG="{\"kv_connector\":\"OffloadingConnector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"spec_name\":\"TieringOffloadingSpec\",\"cpu_bytes_to_use\":64424509440,\"secondary_tiers\":[{\"type\":\"fs\",\"root_dir\":\"$W/fs_tier\"}]}}"
KVARGS=(--kv-transfer-config "$KVCFG")
[ -n "$NO_KV" ] && KVARGS=()
exec vllm serve "$MODEL" --port 8199 --gpu-memory-utilization 0.85 \
  --max-model-len 16384 --max-num-seqs 256 "${KVARGS[@]}" ${EXTRA_ARGS}
