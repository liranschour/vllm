#!/bin/bash
cd /home/lirans/.claude/jobs/0cef0ee0/tmp
PY=/home/lirans/vllm/.claude/worktrees/lock-hold-measure/.venv-measure/bin/python
for n in 1 3 7; do
  VLLM_LOCK_PROBE_MODE=mirror VLLM_LOCK_PROBE_THREADS=$n VLLM_LOCK_PROBE_SPIN=1 $PY -I driver.py mirror_spin_n$n 16,64 1024
  VLLM_LOCK_PROBE_MODE=mirror VLLM_LOCK_PROBE_THREADS=$n VLLM_LOCK_PROBE_SPIN=0 $PY -I driver.py mirror_sleep_n$n 16,64 1024
done
echo SWEEP_DONE
