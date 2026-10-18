#!/bin/bash
# Full sweep on H200: baseline then N mirror contenders (spin + sleep).
cd /workspace/lockhold
python3 driver.py base 1,4,16,64,128,256 1024 || echo FAIL base
for n in 1 3 7; do
  for spin in 1 0; do
    VLLM_LOCK_PROBE_MODE=mirror VLLM_LOCK_PROBE_THREADS=$n VLLM_LOCK_PROBE_SPIN=$spin \
      python3 driver.py mirror_s${spin}_n$n 16,64,256 1024 || echo FAIL mirror_s${spin}_n$n
  done
done
echo SWEEP_DONE
