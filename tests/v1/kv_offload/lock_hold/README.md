# OffloadingManager lock-hold measurement

Question: the scheduler thread holds `manager.lock` from the first
manager-touching hook of `schedule()` to the end of `build_connector_meta()`,
plus short regions for `update_connector_output`, `get_stats`, `take_events`,
etc. If N more threads shared this lock, would they contend?

Setup: branch `feat/offloading-manager-lock-api` (d54a11b5ee) + timing
instrumentation in `offloading/scheduler.py`. RTX A4500 16 GB (WSL),
Llama-3.2-1B, TieringOffloadingSpec (4 GiB CPU primary + FS secondary tier),
`vllm bench serve` random 1024 in / 128 out, ignore-eos, async scheduling (the default).

Instrumentation (`VLLM_LOCK_TRACE_FILE`): every lock region logs its request,
acquire and release timestamps; the step region also logs which hooks it covered.
Probe contenders (`VLLM_LOCK_PROBE_MODE=mirror`, `..._THREADS=N`): after each
real step, every probe thread waits a random offset inside the scheduler's last
free gap and then holds the lock for exactly as long as that step held it. That
models N more schedulers that each do the same work per step. `SPIN=1` holds the
GIL while holding the lock (pure-Python work); `SPIN=0` sleeps (GIL released).

Run: `sweep.sh` (or `driver.py <name> <concs> <in_len>`), then `analyze.py run/<name>`.

## Baseline: scheduler only (results/base2.txt)

| conc | steps/s | step-region hold p50 / p90 / p99 / max (ms) | lock duty | free gap between steps p50 / p10 (ms) | tok/step |
|---:|---:|---|---:|---|---:|
| 1   | 62  | 0.11 / 0.14 / 0.93 / 2.0   | 1.4% | 12.2 / 9.2  | 9   |
| 4   | 31  | 0.16 / 0.20 / 1.9 / 4.4    | 1.0% | 14.8 / 9.5  | 35  |
| 16  | 16  | 0.29 / 0.87 / 4.4 / 11.1   | 1.2% | 17.4 / 16.9 | 137 |
| 64  | 6.7 | 0.93 / 5.8 / 13.5 / 19.8   | 1.9% | 29.0 / 26.5 | 510 |
| 128 | 3.6 | 2.5 / 8.8 / 25.4 / 42.3    | 2.0% | 139 / 43    | 952 |

Short regions add little: `update_connector_output` p50 0.03 ms (p99 1.2 ms),
`get_stats` 0.06 ms, `take_events` 0.01 ms.

Step-region composition (c64): `build_connector_meta` 66%,
`get_num_new_matched_tokens` 23%, scheduler code running between hooks 10%,
`update_state_after_alloc` <1%. The long holds (p99 13-25 ms, max 42 ms) are
steps opened by `get_num_new_matched_tokens`, i.e. steps that admit new
prefills: lookup per block, then `prepare_store` for 1k-token prompts in `build_connector_meta`.

## With N mirror contenders (results/mirror_*.txt)

| N | GIL | conc | scheduler lock wait p99 / max (ms) | sched wait, % of wall | contender wait p99 / max (ms) | total lock duty | out tok/s vs base |
|---:|---|---:|---|---:|---|---:|---:|
| 1 | spin  | 16 | 0.02 / 0.09 | 0.04% | 0.01 / 0.04 | ~2%  | 414 vs 378 (noise) |
| 1 | spin  | 64 | 0.03 / 0.18 | 0.02% | 0.09 / 6.5  | ~3%  | 456 vs 450 |
| 3 | spin  | 64 | 0.04 / 1.7  | 0.02% | 0.87 / 16   | ~6%  | 452 vs 450 |
| 3 | sleep | 64 | 0.98 / 30   | 0.20% | 4.1 / 21    | ~6%  | — |
| 7 | spin  | 16 | 0.04 / 0.23 | 0.04% | 0.02 / 0.14 | ~5%  | 278 vs 378 (−26%) |
| 7 | spin  | 64 | 0.04 / 11   | 0.03% | 6.5 / 50    | ~9%  | 386 vs 450 (−14%) |
| 7 | sleep | 64 | 3.1 / 54    | 0.56% | 6.3 / 50    | ~10% | 382 vs 450 (−15%) |

## Conclusions

1. Lock duty is low. The scheduler holds the lock for 1-2% of wall time at
   every load tested. The typical step region is 0.1-1 ms against a 12-30 ms
   step, so 95%+ of each step is a free window (the lock is released before
   `future.result()` waits on the GPU).
2. Holds scale with the number of tokens a step admits, not with concurrency
   alone: p50 grows from 0.11 ms (c1) to 2.5 ms (c128), and p99/max is set by
   prefill-admitting steps (`get_num_new_matched_tokens` + `prepare_store`):
   about 25 ms p99 and 42 ms max at c128. That tail is what other threads queue behind.
3. With N extra lock users doing the same work, the scheduler thread barely
   waits: p99 under 0.05 ms in spin mode for N up to 7, and at most 0.6% of
   wall time even in the worst case (sleep mode, N=7). Contenders see their
   p99 wait grow from 0.01 ms (N=1) to about 6.5 ms (N=7, c64), because they
   queue behind the scheduler's long prefill steps and behind each other.
4. The throughput drop at N=7 (−14 to −26%) comes from the GIL, not the lock. In
   spin mode the scheduler's lock wait stays at p99 0.04 ms while tok/s falls,
   so the cost is CPU/GIL time stolen from the scheduler and the API process,
   not lock waits. In-process "more scheduler threads" would hit the GIL well
   before they hit this lock.
5. Bottom line: up to about 3 equal-weight sharers, the lock leaves a wide
   window (aggregate duty ~6%, every waiter's p99 under 1 ms in spin mode).
   At 7 sharers the aggregate duty is still only about 10%, but the tail
   (p99 ~6 ms, max ~50 ms) is set by the scheduler's long prefill steps.
   Splitting `build_connector_meta` / `get_num_new_matched_tokens` holds is
   the lever if that tail matters.

## H200 + Llama-3.1-8B re-measurement (results_h200/, h200/)

Pod `lock-hold-h200` (1×H200, image `vllm/vllm-openai:nightly` at 49cb2bbf7,
which already contains the lock API). The instrumented `scheduler.py` is that
file plus this branch's instrumentation patch. Tiering = 60 GiB CPU + FS
tier; random 1024 in / 128 out. Driver/serve/sweep variants are in `h200/`.

Baseline (scheduler only):

| conc | steps/s | hold p50 / p90 / p99 / max (ms) | lock duty | free gap p50 / p10 (ms) | tok/step | out tok/s |
|---:|---:|---|---:|---|---:|---:|
| 1   | 139 | 0.03 / 0.04 / 0.33 / 0.97 | 1.1%  | 4.9 / 4.8  | 9    | 196  |
| 4   | 71  | 0.05 / 0.16 / 0.79 / 1.8  | 1.0%  | 5.0 / 4.6  | 35   | 689  |
| 16  | 37  | 0.35 / 0.69 / 2.7 / 4.4   | 1.9%  | 5.8 / 5.4  | 138  | 1649 |
| 64  | 25  | 1.3 / 4.1 / 20 / 35       | 6.9%  | 9.8 / 6.9  | 536  | 2726 |
| 128 | 21  | 2.8 / 10 / 42 / 121       | 12.6% | 13.1 / 3.0 | 852  | 2992 |
| 256 | 17  | 5.2 / 17 / 131 / 342      | 17.1% | 20.0 / 4.4 | 1253 | 3022 |

Composition at c256: `build_connector_meta` 55%, `get_num_new_matched_tokens`
35% (lookups grow with the waiting queue), between-hooks 9%.

With N mirror contenders (scheduler wait = time the real scheduler thread
blocks on the lock; total duty = scheduler + contenders):

| N | GIL | conc | sched wait p99 / max (ms) | sched wait % wall | contender wait p99 / max (ms) | total duty | out tok/s vs base |
|---:|---|---:|---|---:|---|---:|---|
| 1 | spin  | 64  | 0.04 / 3.1   | 0.07% | 2.2 / 4.0    | ~10% | 2598 vs 2726 (−5%) |
| 1 | spin  | 256 | 0.79 / 38    | 0.56% | 13 / 280     | ~26% | 2946 vs 3022 (−3%) |
| 1 | sleep | 256 | 6.3 / 319    | 3.5%  | 17 / 135     | ~27% | 3037 vs 3022 |
| 3 | spin  | 64  | 0.04 / 2.8   | 0.06% | 3.7 / 49     | ~14% | 2507 (−8%) |
| 3 | spin  | 256 | 12 / 116     | 2.4%  | 68 / 691     | ~37% | 2787 (−8%) |
| 3 | sleep | 256 | 56 / 438     | 14.8% | 71 / 276     | ~48% | 2937 (−3%) |
| 7 | spin  | 16  | 0.03 / 0.04  | 0.04% | 0.01 / 0.04  | ~8%  | 1393 vs 1649 (−16%) |
| 7 | spin  | 64  | 0.04 / 51    | 0.73% | 17 / 277     | ~22% | 2340 (−14%) |
| 7 | spin  | 256 | 18 / 447     | 3.4%  | 124 / 389    | ~51% | 2454 (−19%) |
| 7 | sleep | 256 | 131 / 847    | 30%   | 230 / 880    | ~58% | 2495 (−17%) |

H200 conclusions:

1. With faster steps the lock's share of time grows. At low load (c ≤ 16) it
   is busy 1-2% and the free gap is about 5 ms per step, a wide window. From
   c64 on the scheduler alone holds it 7% (c64), 13% (c128) and 17% (c256),
   because work under the lock grows with tokens and waiting requests while
   GPU step time grows more slowly.
2. The tail is the problem, not the median. At c256 the p99 hold is 131 ms and
   the max 342 ms, all from prefill-admitting steps. Any other thread that
   arrives during one waits that long: with N=1 the contender's max wait is
   280 ms at c256.
3. The lock scales to N=1, and to N=3 at moderate load (c ≤ 64): the
   scheduler's p99 wait stays under 0.05 ms in spin mode and throughput is
   within 5-8%. At c256 with N ≥ 3 equal-weight sharers the lock saturates
   (37-58% total duty): the scheduler waits p99 12-131 ms and up to 30% of
   wall time (N=7, sleep). That is real contention.
4. As on the laptop GPU, the throughput cost at N=7 (−14 to −19%) appears even
   where the lock is idle (c16: lock wait 0.04 ms, tok/s −16%), so with
   in-process threads the GIL bites first. The lock becomes the limiter only
   at high load, where it adds to the GIL cost.
5. Levers if more sharers are planned: (a) shorten prefill-step holds, i.e.
   per-block `lookup()` in `get_num_new_matched_tokens` (35% at c256) and
   `prepare_store` in `build_connector_meta`; (b) release the region between
   requests rather than holding one step-long region (the HIT→pin invariant
   holds per request); (c) for sharers that do real work, use a process
   rather than a thread, since the GIL caps in-process sharers anyway.

Caveats: a single small GPU (1B model: fast steps, so the lock-free gap is
small in absolute terms; a larger model widens the gap and lowers duty).
The probes mirror hold durations but do no real manager work, so cache-line,
allocator and real tier-I/O effects are not modelled. The FS tier stores but
sees few lookup hits with random prompts.
