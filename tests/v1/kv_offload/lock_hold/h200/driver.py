"""Run lock-hold measurement sweeps.

usage: driver.py <run_name> [conc,...] [in_len]   (probe knobs: VLLM_LOCK_PROBE_* env)
Writes phases.csv (phase,t_start_ns,t_end_ns) and the lock trace under
J/run/<run_name>/.
"""

import glob
import os
import subprocess
import sys
import time
import urllib.request

J = "/workspace/lockhold"

PY = "python3"

name = sys.argv[1]
concs = [int(c) for c in (sys.argv[2] if len(sys.argv) > 2 else "1,4,16,64,128").split(",")]
in_len = int(sys.argv[3]) if len(sys.argv) > 3 else 1024
out_len = 128
out = f"{J}/run/{name}"
os.makedirs(out, exist_ok=True)
for f in glob.glob(f"{out}/trace*"):
    os.remove(f)

env = dict(os.environ)
log = open(f"{out}/serve.log", "w")
srv = subprocess.Popen(["bash", f"{J}/serve.sh", f"{out}/trace"], env=env,
                       stdout=log, stderr=subprocess.STDOUT, start_new_session=True)


def ready() -> bool:
    try:
        return urllib.request.urlopen("http://127.0.0.1:8199/health", timeout=2).status == 200
    except Exception:
        return False


try:
    deadline = time.time() + 1200
    while not ready():
        if srv.poll() is not None or time.time() > deadline:
            raise SystemExit(f"server failed, see {out}/serve.log")
        time.sleep(2)

    phases = open(f"{out}/phases.csv", "w")

    def mark(phase, t0):
        phases.write(f"{phase},{t0},{time.perf_counter_ns()}\n")
        phases.flush()

    t0 = time.perf_counter_ns()
    time.sleep(5)
    mark("idle", t0)

    seed = int(time.time())
    for i, c in enumerate(concs):
        n = max(32, c * 4)
        t0 = time.perf_counter_ns()
        subprocess.run(
            [PY, "-m", "vllm.entrypoints.cli.main", "bench", "serve",
             "--backend", "vllm", "--base-url", "http://127.0.0.1:8199",
             "--model", os.environ.get("MODEL", "meta-llama/Llama-3.1-8B-Instruct"),
             "--dataset-name", "random", "--random-input-len", str(in_len),
             "--random-output-len", str(out_len), "--ignore-eos",
             "--num-prompts", str(n), "--max-concurrency", str(c),
             "--seed", str(seed + i), "--save-result",
             "--result-dir", out, "--result-filename", f"bench_c{c}.json"],
            stdout=open(f"{out}/bench_c{c}.log", "w"), stderr=subprocess.STDOUT,
            check=True,
        )
        mark(f"c{c}", t0)
    time.sleep(2)  # let the 1 s trace flush catch up
finally:
    os.killpg(srv.pid, 2)
    try:
        srv.wait(60)
    except subprocess.TimeoutExpired:
        os.killpg(srv.pid, 9)
    time.sleep(3)
    for f in glob.glob("/dev/shm/vllm_offload_*.mmap"):
        os.remove(f)
print("done", out)
