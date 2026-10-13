# Two-pod router test: secondary-tier KV pulls

Deploys two full vLLM engines, each behind an llm-d routing sidecar, and proves
that KV moves **between the pods** when the router tells one to pull a cached
prefix from the other. Runs the same test twice: once over the
OffloadingConnector **P2P tier** (the baseline, which establishes the
environment works) and once over **KVCR**, driven by a router `kv_hints`
envelope.

This is deliberately **not** a P/D deployment. Both pods are identical and
either can serve any request. No prefiller header is ever set, so the sidecar
takes its decoder-only path — the one that injects a cached-prefix pull.

```text
client ──> EPP stub ──> sidecar (pod X) ──> vLLM (pod X)
            picks X                             │
            hints Y                             │ secondary tier fetch
                                                ▼
                                          vLLM (pod Y) tier
```

## What is and is not under test

Under test is the **routing sidecar**: turning
`x-llm-d-kv-cache-source-host-port` into a request field, deriving the peer's
tier endpoint, and (for KVCR) building an envelope the engine's hint parser
accepts.

`epp_stub.py` stands in for llm-d's EPP and replaces exactly one thing it
does — choosing which replica holds the most reusable prefix. That choice is
protocol-agnostic and unchanged by the KVCR work, while the real EPP needs
InferencePool CRDs, a gateway and a KV-event indexer before it will start.
Swapping it for ~150 lines keeps the test about the sidecar. **A green run here
does not exercise the real EPP's source selection.**

## Prerequisites

- Both pods running, with a vLLM checkout that has the KVCR tier and accepts
  `kv_hints` on the HTTP APIs, and the `kvcr` package installed. Verified
  against engines at `d61c102f51`.
- A checkout of `llm-d-router` with the KVCR sidecar support, at `~/llm-d-router`
  or wherever `ROUTER_SRC` points.
- A Go toolchain matching `go.mod` (1.27.2), which is newer than most distros
  ship. `stage_sidecar.sh` looks in `~/.local/go-sdk/go/bin/go`, then `PATH`,
  and prints the install commands if it finds neither. Override with `GO_BIN`.
- `KUBECONFIG` pointing at the cluster; defaults to
  `~/CWKubeconfig_waldorf_US-EAST-04A`.

Nothing needs to be installed on the pods: the sidecar is built statically and
copied in with `oc cp`, and the test harness is stdlib-only Python.

Everything else — ports, model, tier sizes, pod names, `PYTHONHASHSEED` — is in
`router_env.sh` and overridable from the environment.

## Running it

```bash
# once per sidecar code change
ROUTER_SRC=~/llm-d-router ./stage_sidecar.sh

# baseline: does the environment work at all?
./deploy_router_env.sh p2p
./run_router_test.sh p2p pair
TRIALS=4 ./run_router_test.sh p2p control

# the feature under test
./deploy_router_env.sh kvcr
./run_router_test.sh kvcr pair
TRIALS=4 ./run_router_test.sh kvcr control     # negative control
TRIALS=10 ./run_router_test.sh kvcr random     # free routing

./deploy_router_env.sh down
```

Run the baseline first. If `p2p` fails, the problem is the environment, not
`kv_hints`, and debugging the KVCR arm would be wasted effort.

Run `control` whenever you care about the result. `pair` on its own only shows
that tier hits happened; the control is what attributes them to the hint.

## Measured results

`llmd-prefiller` + `llmd-decoder`, Llama-3.2-1B, `--block-size 16`, sidecar at
`llm-d-router@522fa0c5`, engines at `d61c102f51`.

| arm | mode | result |
| --- | --- | --- |
| p2p | pair | 8/8 trials transferred, 54 tier chunk hits + 864 external hits each |
| kvcr | pair | 8/8 trials transferred, same counts |
| kvcr | control (no hint) | 0/4 — nothing transferred |
| kvcr | random, 10 trials | 7/10 transferred, matching exactly the 7 routed across pods |

The two metrics corroborate each other: 54 chunks x 16 tokens per block = 864
tokens, so the chunk count and the token count describe the same transfer
rather than both merely being non-zero.

The sidecar log is the other half of the evidence — the kvcr arm logs
`running KVCR source protocol ... kvcr_control_endpoint=tcp://<peer>:17771`
and zero `running P2P source protocol`, confirming which code path ran.

## How the verdict is reached

`router_pull_test.py` runs two phases per trial:

1. **seed** — the prompt goes to pod S with no hint; S computes it and offloads
   the prompt blocks to its host tier.
2. **pull** — the *same* prompt goes to the other pod with a hint naming S.

Then it reads the **puller's** metrics around the pull:

| metric | meaning |
| --- | --- |
| `vllm:external_prefix_cache_hits_total` | blocks the engine took from the connector instead of recomputing — **the transfer criterion** |
| `vllm:kv_offload_tiering_chunk_hits_total` | chunks the secondary tier served; also moves on a plain store or local lookup |

A trial counts as a transfer on `external_prefix_cache_hits`, **not** on tier
chunk hits. Tier hits rise when a pod merely stores or looks up its own
blocks: a same-pod trial in random mode shows `tier_hits > 0` with
`external_hits == 0` and has moved nothing. Counting tier hits would have
scored those as successes.

`pair` mode requires **every** trial to transfer, since each one seeds on one
pod and pulls on the other, alternating direction. `control` mode requires
**zero**. `random` mode requires only that some do: with free routing a
prompt's two requests sometimes land on the same pod, which is a local hit.
Expect the transfer count there to equal the cross-pod routing count.

## Things that will silently cost you a transfer

Each of these produces a miss with no error anywhere — the request still
succeeds by recomputing, so only the metrics show it.

- **Reused prompts.** A prompt the puller has seen hits its own GPU prefix
  cache and transfers nothing. The test makes every prompt unique with a UUID;
  keep it that way.
- **Mismatched `PYTHONHASHSEED`.** Block hashes are computed independently on
  each pod and must agree. `router_env.sh` pins the same value on both;
  changing it on one pod only breaks every pull.
- **`opportunistic_query: false`** on the KVCR tier. The router sends a
  placeholder `block_hashes` list by design, so a pod that compares hashes
  misses everything. The deploy script leaves the setting unset, i.e. the
  adapter default of `true`.
- **A tier sized near the working set.** The source must still hold the seeded
  blocks when the puller asks. `CPU_BYTES` is 2 GiB for headroom.
- **A restarted peer.** Restarting one side mid-run can wedge the survivor
  rather than just failing the fetch. Redeploy both with
  `./deploy_router_env.sh <arm>` instead of bouncing one pod.

## Layout

| file | role |
| --- | --- |
| `router_env.sh` | shared config: pods, ports, model, tier sizing |
| `stage_sidecar.sh` | build the sidecar statically and `oc cp` it to both pods |
| `deploy_router_env.sh` | tear down and bring up both engines, sidecars, EPP stub |
| `epp_stub.py` | EPP stand-in: picks a replica, hints the other |
| `router_pull_test.py` | request generator and metric assertions (stdlib only) |
| `run_router_test.sh` | stage and run the test, copy results back |
| `results/` | saved JSON per run |
