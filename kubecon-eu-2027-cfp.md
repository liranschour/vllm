# KubeCon + CloudNativeCon Europe 2027 — CFP submission draft

- **Event:** 15–18 March 2027, Barcelona, Spain
- **CFP closes:** Sunday 11 October, 23:59 CEST — submit via Sessionize
- **Format:** Session Presentation (30 min, 1–2 speakers)
- **Track:** AI Inference and Infrastructure
- **Case study flag:** Yes — reports a real implementation and its measured outcomes
- **Previously presented at a CNCF/LF event:** No
- **Open source projects to list:** Kubernetes, vLLM, llm-d, Gateway API Inference Extension, NIXL

---

## Title

Stop Recomputing Prefixes: Peer-to-Peer CPU KV Cache Sharing in vLLM

---

## Description

In a Kubernetes inference cluster, a repeated prefix is recomputed from scratch
whenever a request lands on a pod that did not serve the previous turn, even
though its KV cache is sitting in another pod's CPU memory, one RDMA hop away.

An llm-d cluster now closes that gap with a peer-to-peer KV cache tier in vLLM.
Any instance can match block hashes on a peer and pull the KV from that peer's
CPU memory instead of recomputing the prefill. Peers are symmetric: no fixed
prefill and decode roles, no shared filesystem, no central cache store.

This session reports what that enables, measured on multi-node H200 clusters under
both document-heavy and agentic workloads. Two placement strategies are
examined: hot-spot spill-over, where a saturated pod's prefix is pulled by a
cold one so load can be balanced without paying for prefill twice, and a split
between long-prefill and short-prefill pools, where the short pool pulls shared
context rather than recomputing it.

The session closes with a heuristic for when pulling beats recomputing. A
transfer can only win where the interconnect moves KV bytes faster than prefill
produces them, and the break-even prefix length is then roughly the pull
latency floor multiplied by prefill throughput — two figures an operator
measures once and routes against.

---

## Benefits to the Ecosystem

Prefix recomputation is one of the largest avoidable costs in production LLM
serving, and most Kubernetes operators pay it every time a request is routed
away from the pod holding the relevant cache. This session gives the community
a measured answer to when that cost can be removed by pulling cached state
instead of recomputing it, and when it cannot.

Every component involved is open source and reproducible. The peer-to-peer KV
cache tier is upstream in vLLM, peer selection is driven through the Gateway
API Inference Extension endpoint picker, and the deployment uses stock
Kubernetes primitives. The pattern needs no shared filesystem, no central cache
store, and no proprietary component, which keeps it reachable for teams running
modest clusters rather than only the largest ones.

---
