#!/bin/bash
# usage: serve.sh <trace_prefix>   (probe knobs via VLLM_LOCK_PROBE_* env;
#        NO_KV=1 runs without the offloading connector as a control)
WT=/home/lirans/vllm/.claude/worktrees/lock-hold-measure
J=/home/lirans/.claude/jobs/0cef0ee0/tmp
rm -rf $J/fs_tier
mkdir -p $J/fs_tier $J/run
cd $J/run
export VLLM_USE_FLASHINFER_SAMPLER=0 PATH=$WT/.venv-measure/bin:$PATH VLLM_LOCK_TRACE_FILE=$1
KVCFG="{\"kv_connector\":\"OffloadingConnector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"spec_name\":\"TieringOffloadingSpec\",\"cpu_bytes_to_use\":4294967296,\"secondary_tiers\":[{\"type\":\"fs\",\"root_dir\":\"$J/fs_tier\"}]}}"
KVARGS=(--kv-transfer-config "$KVCFG")
[ -n "$NO_KV" ] && KVARGS=()
exec $WT/.venv-measure/bin/python -m vllm.entrypoints.cli.main serve unsloth/llama-3.2-1b-Instruct \
  --port 8199 --gpu-memory-utilization 0.55 --max-model-len 8192 --max-num-seqs 256 \
  "${KVARGS[@]}" ${EXTRA_ARGS}
