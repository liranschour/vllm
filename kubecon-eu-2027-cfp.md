# KubeCon + CloudNativeCon Europe 2027 — CFP submission draft

- **Event:** 15–18 March 2027, Barcelona, Spain
- **CFP closes:** Sunday 11 October, 23:59 CEST — submit via Sessionize
- **Format:** Session Presentation (30 min, 1–2 speakers)
- **Track:** AI Inference and Infrastructure
- **Case study flag:** Yes — reports a real implementation and its measured outcomes
- **Previously presented at a CNCF/LF event:** No

---

## Title

> Enabling P2P CPU-Based KV Cache Sharing in Production Inference Clusters

<details>
<summary>Alternate titles</summary>

- Your KV Cache Is Stranded: Peer-to-Peer CPU Cache Sharing Across a vLLM Fleet
- Beyond HBM: Sharing KV Cache Peer-to-Peer Across vLLM Replicas in Production
- From Global Index to Moved Bytes: P2P KV Cache Sharing in vLLM and llm-d

</details>

---

## Description / Abstract

Every vLLM replica in a Kubernetes inference pool computes and then throws away KV
cache that its neighbours are about to recompute. Accelerator HBM is too small to
hold a useful working set, and the cache a replica does hold is unreachable from
any other replica, so cluster-wide prefix reuse collapses exactly when traffic
makes it most valuable. Prefill/decode disaggregation hits the same wall from the
other side: moving cache GPU-to-GPU forces prefiller and decoder to agree on
tensor-parallel width and block layout.

This session presents the peer-to-peer CPU KV cache tier now in vLLM, which turns
each replica's host memory into a shared, addressable cache layer. Blocks are
identified by content hash in a TP-agnostic CPU layout, a ZMQ control plane
negotiates what a peer actually holds, and NIXL/UCX RDMA moves the bytes without
touching GPU code. Peers are symmetric: any instance serves or pulls per request,
so a 4-way prefiller can feed an 8-way decoder and cache can flow in either
direction.

The talk then shows how llm-d, a CNCF Sandbox project, adopts this tier: its
prefix-aware router decides *which* peer to pull from while vLLM performs the
transfer, giving llm-d's global KV index an actuator rather than only a hint.

The session closes with production measurements across several workload shapes —
including where the feature pays for itself, where it is neutral, and where
enabling it is the wrong call.

---

## Benefits to the Ecosystem

Distributed KV cache management is where cloud native inference is heading, and
most of the public discussion is still architecture diagrams. This session gives
attendees the measured version: which workload shapes actually benefit from
cross-instance cache sharing, how much, and how to tell before committing
cluster memory to it.

Three things attendees take away. First, a concrete mental model of KV cache as a
tiered, cluster-addressable resource rather than per-pod scratch space — useful
whether or not they run vLLM. Second, the division of labour that makes this
work on Kubernetes: the orchestration layer owns placement and peer selection,
the inference engine owns the transfer, and the interface between them is small
enough to reason about. Third, honest negative results. Short prompts are faster
to recompute locally than to fetch, and some long-prompt workloads are so
prefill-compute-bound that transfer cost disappears into the noise. Operators who
know the boundary will not spend host memory on cases that cannot pay it back.

The talk also covers the failure modes that only appear in a real cluster —
peer restarts, stalled engines, transfer errors that strand a request until the
client times out — and the metrics added to make them diagnosable. Both the
vLLM tier and the llm-d integration are upstream and open source, so everything
shown is reproducible.

---

## Projects Involved

- **vLLM** — the P2P secondary tier under the tiering/offloading subsystem
- **llm-d** (CNCF Sandbox) — prefix-aware routing and peer selection over a vLLM pool
- **Kubernetes** — deployment substrate; LeaderWorkerSet for multi-node roles
- **NIXL / UCX** (open source, NVIDIA) — RDMA data plane
- **ZMQ** — control plane transport

---

## Talk Outline (30 min)

| Min | Section | Content |
| --- | --- | --- |
| 0–4 | The stranded cache problem | HBM ceiling; cache locality collapse under multi-replica routing; TP/layout reconciliation tax in GPU-to-GPU PD |
| 4–13 | The vLLM P2P tier | CPU tier as canonical TP-agnostic layout; content-hash block identity; ZMQ control plane (lookup → fetch → transfer-done); NIXL RDMA data plane; symmetric peers and the three role keys |
| 13–19 | llm-d adoption | Separation of concerns — router/EPP selects the peer, vLLM moves bytes; wiring the tier into a prefill/decode pool; DP-rank addressing; composing with prefix-aware routing |
| 19–27 | Measurements | Case 1 cross-replica prefix reuse; Case 2 PD over CPU with TP mismatch; Case 3 where it does not pay (short prompts, compute-bound long prefill); Case 4 control-plane placement as a first-order effect |
| 27–30 | Production lessons and what's next | Failure modes, observability, proactive/prefetch direction; Q&A |

---

## Measurement Inventory (speaker notes — not submitted)

Evidence already collected in-tree, to be re-run and tidied for the talk:

1. **Transfer cost vs request cost.** `fetch_rtt` (FetchMsg → TransferDone on the
   consumer) against TTFT, 4096-token inputs, concurrency 1→32 on two H-class pods:
   31 ms RTT at c1 (16% of TTFT), rising to ~606 ms at c16. Shows both the
   absolute cost and where consumer-side serialization begins to dominate.
   `tests/v1/kv_offload/tiering/p2p/results/fetch_rtt_*`

2. **Control-plane placement is a first-order effect.** Servicing the tiering
   control plane off the scheduler thread, measured on the lookup leg with a busy
   producer: lookup delay at c1 drops 0.156 s → 0.048 s (3.2x), and the
   busy/idle gating penalty collapses from 3.46x to 0.98x. The per-concurrency
   slope is unchanged (~19 ms/c, R² > 0.99), so this removes a constant penalty
   rather than fixing the scaling term — worth saying plainly rather than
   quoting the headline ratio alone.
   `tests/v1/kv_offload/tiering/p2p/results/lock_api_conc_compare_A_C_C2/summary.md`
   An earlier run of the same experiment (PR #58168) showed 0.220 s → 0.053 s
   (4.1x) with the gating penalty going 4.83x → 1.17x. Pick one run for the
   slide and cite it; do not blend the two.
   Caveat for the talk: this is the **lookup** leg only. The same change measured
   on the fetch leg (`P2P_FETCH_RTT`) came out at parity, so the claim must be
   scoped to lookup or it overstates the result.

3. **Where it does not pay.** 1024-token inputs: P2P pull is *slower* than
   ordinary GPU prefill (+27% median TTFT, worse at the tail). 65536-token inputs
   at c16/c100: the run saturates ~78k tok/s and every connector lands within
   0.5% — prefill compute dominates and transfer differences are invisible.
   This is the "know the boundary" slide.

4. **Cross-instance cache hit.** Two-pod prefill/decode with ~95% external cache
   hit rate; requires fully unique prompts, or the local GPU prefix cache absorbs
   the hit and no real transfer happens.

5. **Observability.** 15 `vllm:kv_offload_p2p_*` metrics (NIXL telemetry, fetch
   RTT, lookup RTT, counters/gauges), verified end-to-end.

### To do before submitting

- [ ] Decide on a co-speaker (llm-d side would strengthen the adoption section;
      note the CFP rule that no session is accepted with 3+ speakers who all
      identify as men, and that panels need more than one company)
- [ ] Link a recording of prior speaking, or record a 2-minute self-intro
- [ ] Re-run cases 1 and 3 on a clean cluster for citable, current numbers
- [ ] Confirm nothing substantially similar was delivered at an LF event in the
      past year (disqualifying)
