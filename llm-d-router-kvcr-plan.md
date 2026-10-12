# Enabling KVCR P2P sharing from llm-d-router, via `kv_hints`

Plan for teaching `llm-d-router` to drive **KVCR** peer-to-peer KV sharing by
emitting a `kv_hints` envelope, alongside (not replacing) the existing
OffloadingConnector **P2P tier** path that drives sharing with
`kv_transfer_params`.

Scope: the router only. The KVCR tier itself and the vLLM `kv_hints` plumbing
are prerequisites, tracked in [`KVCR.md`](KVCR.md). The P2P tier is **not**
being migrated or removed — the two mechanisms coexist behind one selector.

**Core design choice: the router names a peer, never a block list.** It always
targets KVCR's `opportunistic_query=True`, so the hint's required
`block_hashes` is a fixed placeholder and the engine does its own per-key
lookup — the same request-scoped semantics the P2P tier already has. There is
deliberately no exact-hash mode; §6 is the argument for that, and §3.1 the
mechanism that allows it.

Router baseline for every line reference below:
`llm-d/llm-d-router @ 46f3635` ("Keep IPv6 addresses intact in endpoint
subsets", #3305).

---

## 1. What the router already does for the P2P tier

Two legs, cleanly split. The EPP decides *who* holds the prefix; the sidecar
decides *how to ask for it*.

### Leg 1 — EPP: source selection (connector-agnostic)

`pkg/epp/framework/plugins/requestcontrol/dataproducer/p2psource/producer.go`
(`p2p-source-producer`):

1. In `DataProducer`, consumes per-endpoint `PrefixCacheMatchInfo` from an
   approximate or precise prefix-cache producer.
2. Samples among endpoints within one block of the longest eligible prefix,
   weighted by `1/(1+waiting queue)` and a request-ID hash, so pulls spread
   across replicas holding near-identical prefixes.
3. Publishes a `ReusablePrefixTokens` floor (`G = max(0, S - D + 1)`) for
   scheduling consumers, from the source's **confirmed contiguous CPU-tier**
   prefix only.
4. In `PreRequest`, emits `routing.KVCacheSourceHeader`
   (`x-llm-d-kv-cache-source-host-port`, `pkg/common/routing/common.go:44`)
   with the source's **HTTP** `host:port`, but only when the sampled source
   out-caches the computing endpoint by `minCachedTokenDelta` tokens. Any
   inbound value of the header is stripped.

### Leg 2 — sidecar: protocol injection (connector-specific)

`pkg/sidecar/proxy/dispatch.go:130-148` takes and validates the header —
`p2pPullAvailable()`, `isHostPort()`, then the SSRF `allowlistValidator` — and
then one of two paths runs:

- **PD** (`pkg/sidecar/proxy/connector_p2p.go:57`, `handleP2P`): prefill gets
  `kv_transfer_params.remote_decoder.kv_request_id` plus, optionally,
  `remote_kv_source` (`addP2PPullToPrefill:253`); decode gets
  `remote_prefiller{kv_request_id, remote_host, remote_port}`. Prefill is run
  to completion before decode is dispatched
  (`handleP2PSequentialRequests:128`).
- **Decode-only pull** (`decodeWithP2PSource:343`): rebuilds
  `kv_transfer_params` from scratch as `{remote_kv_source: {...}}` and
  re-dispatches.

The peer's tier port is derived locally, never trusted from the wire:
`p2pPortFor` (`:305`) reads the rank off the routed HTTP port
(`port - dpBasePort`), range-checks it, and returns `P2PConnectorPort + rank`.

---

## 2. P2P tier vs. KVCR: the delta the router must absorb

| | **P2P tier** (today) | **KVCR** (this plan) |
| --- | --- | --- |
| Request field | `kv_transfer_params` | top-level **`kv_hints`** envelope |
| Peer address | `remote_host` + `remote_port` ints | `source_control_endpoint`, a **`tcp://host:port` ZMQ URL** |
| Port meaning | P2P tier listener | KVCR **ZMQ control** port, `control_ports[dp_local_rank]` |
| Correlation | `kv_request_id` UUID per leg | none — fetch is keyed by block hashes; `message_id`/`action_id` are advisory |
| Prefill-side injection (PD) | required (`remote_decoder.kv_request_id`) | **none**: a KVCR producer is passive, it answers `start_write` by pinning its own host-tier chunks |
| Keys named | implicit, tier does its own lookup | same — `block_hashes` is a filter switched off by `opportunistic_query`, so it carries a **placeholder** (§3.1, §6) |
| Direction | consumer pulls | consumer asks, **producer RDMA-WRITEs** (destination-initiated) |
| Ordering | prefill-first already enforced | prefill-first **mandatory** — there is no push-to-waiting-decoder mode |
| Engine gate | `p2p` secondary tier | KVCR tier + `router_capabilities: ["router_hint"]` |

Target envelope, exactly as KVCR's parser accepts it (reference
implementation: `tests/v1/kv_offload/tiering/kvcr/kvcr_connector_proxy.py`,
`_kv_hints`):

```json
{
  "protocol_version": "0.1",
  "message_id": "<uuid>",
  "actions": [{
    "action_id": "<uuid>-fetch",
    "action_type": "kv.fetch",
    "action_version": "1.0",
    "payload": {
      "source_control_endpoint": "tcp://10.0.0.5:17771",
      "block_hashes": [12345678901234567890],
      "mode": "copy"
    }
  }]
}
```

What the parser actually enforces (`kvcr/src/kvcr/hint_parser.py:55-123`) —
worth separating, because only half of it is a hard requirement:

- **Rejected** (`ValueError`, the whole envelope is dropped): a non-list
  `actions`; no action whose `action_type` is exactly `"kv.fetch"` carrying a
  mapping `payload`; `no_retain` present at all; `mode` present with any value
  other than `"copy"`; a missing or non-string or empty
  `source_control_endpoint`; `block_hashes` not a list, or containing a
  non-`int`, a `bool`, or a value outside `[0, 2**64)`; and — importantly —
  **`block_hashes` resolving to an empty set**.
- **Warned only** (processed anyway with the supported schema):
  `protocol_version` and `action_version` mismatches, via
  `_warn_version_mismatch`. So sending `"0.1"` / `"1.0"` is correct but not
  load-bearing.
- `mode` is **optional** and defaults to `"copy"`.

---

## 3. Findings that shape the plan

These came out of reading both trees and are why the plan is small.

1. **`block_hashes` is a filter, not the work list — and opportunistic mode
   switches the filter off.** This is the finding the whole plan rests on, so
   the mechanics in full:

   - The hint is stored **per request**, `_request_hints[request_id] =
     _RequestHint(source, block_hashes, ...)`
     (`kvcr/src/kvcr/remote_fw_dram.py:547-560`).
   - The engine's lookup is per key and request-scoped:
     `lookup(key, req_context)` → `kvcr.query((block_key,), req_id)`
     (`manager.py:460-467`) → `remote_fw_dram.query(key, request_id)`
     (`remote_fw_dram.py:568`).
   - That `query` returns `False` when there is no hint, no source, the hint
     has `failed`, or the key is in `missing_keys`. Only **then** does it
     consult the hashes, and only if the flag is off:
     `if not self._options.opportunistic_query and adapter.decode(key) not in
     request_hint.block_hashes: return False` (`:580-583`).
   - **`block_hashes` appears nowhere else in the functional path.** It never
     scopes the fetch: `_start_target_pull` takes the keys of the engine's own
     load job minus the negative cache — `keys = tuple(key for key in blocks
     if key not in current_hint.missing_keys)` (`:611`) — and `missing_keys`
     grows by whatever an op asked for and did not get (`:811`), so
     over-asking self-corrects within a request.

   With `opportunistic_query=True` — the vLLM adapter's default
   (`manager.py:287`; note kvcr's own default is `False`, `config.py:49`) —
   the hint therefore degenerates to exactly the P2P tier's semantics:
   *this request may pull from this peer*, and KVCR runs its own per-key
   lookup and `submit_load` against it. **A v1 router does not need real block
   hashes**; a one-element placeholder satisfies the non-empty rule.

   **Design decision: the router always targets `opportunistic_query=True`,
   and there is no exact-hash mode** — rationale in §6. Because
   `opportunistic_query` is a per-pod engine setting and not a per-request
   hint field, this is a *deployment contract* rather than a runtime choice:
   a pod that disables it silently stops matching the router's hints.
   `block_hashes` is therefore always the placeholder constant.
2. **The EPP needs no change for v1.** `x-llm-d-kv-cache-source-host-port`
   already carries exactly what the hint needs — which peer holds the prefix.
   Selection, sampling, the `minCachedTokenDelta` gate and the
   `ReusablePrefixTokens` floor are all protocol-agnostic. **All v1 work is
   sidecar-side.**
3. **Where real hashes would live, if they were ever needed.** Recorded as an
   escape hatch only, since §6 rules them out. `PrefixCacheMatchInfo` exposes
   counts and no hashes
   (`pkg/epp/framework/plugins/datalayer/attribute/prefix/data_types.go:46-77`),
   but the precise producer already stashes
   `perPromptKeys [][]kvblock.BlockHash` under `blockKeysStateKey`
   (`preciseprefixcache/producer.go:436`, `prerequest.go:60`), and
   **`kvblock.BlockHash` is `uint64`** (`pkg/kvcache/kvblock/index.go:164`) —
   the exact type `block_hashes` wants. It would be a serialization job, not
   a recomputation job.
4. **`ownership` is ignored but harmless.** The router's vLLM event adapter
   decodes a fixed 11-field prefix and skips the remainder
   (`pkg/kvevents/engineadapter/vllm_msgpack.go:161-183`, `:201`). vLLM's
   `locality`/`ownership`/`session_id` trail that prefix
   (`vllm/distributed/kv_events.py:88` and `:123`), so KVCR-tagged events decode
   without misalignment — the router just cannot tell a KVCR-owned block from
   a vLLM-owned one.
5. **The CPU-tier gate works for KVCR events.** `medium` becomes `DeviceTier`
   and is lowercased (`pkg/kvevents/pool.go:52`), so vLLM's `Medium.CPU`
   (`"CPU"`) lands as `"cpu"` and matches p2psource's
   `cpuDeviceTier = "cpu"` (`producer.go:327`). KVCR's `medium=STORAGE` (G3)
   becomes `"storage"` and is correctly *excluded* from the CPU floor.
6. **Rank derivation is reusable verbatim.** KVCR indexes `control_ports` by
   `dp_local_rank`, the same pod-local rank `p2pPortFor` already recovers from
   the routed HTTP port. One shared helper serves both.
7. **`control_advertise_host` defaults to `socket.gethostname()`**
   (`manager.py:332`), i.e. the pod name — resolvable only via a headless
   service. Deployments must set it to `POD_IP` so the host the router names
   and the host KVCR advertises agree.

---

## 4. Phase 0 — prerequisites outside the router

Not router work, but the plan is blocked without them.

- **vLLM: `kv_hints` on the HTTP APIs.** Present locally as commit
  `9e1d6f2d75` ("[Frontend] Accept kv_hints on the generate HTTP APIs") —
  a top-level `kv_hints: dict[str, Any] | None` on the completion, chat,
  responses and anthropic protocols, validated into `KvHintsEnvelope` by
  `_get_kv_hints` (`vllm/entrypoints/generate/base/serving.py:274`). Must land
  upstream or ship in the serving image. `vllm_xargs` is **not** a shortcut:
  the adapter reads the typed `req_context.kv_hints`, not `extra_args`.
- **Engine config per pod**: KVCR secondary tier with
  `router_capabilities: ["router_hint"]` (the tier refuses to construct
  without it, `manager.py:310`), `self_describing_kv_events: true`,
  `control_advertise_host: $POD_IP`, a known `control_ports` base, and
  **`opportunistic_query` left at its adapter default of `True`** — disabling
  it makes every router hint stop matching (§6).
- **Pod-to-pod block-key agreement**, unchanged from the P2P tier's existing
  deployment requirements and *not* removed by dropping exact-hash mode: every
  peer needs the same `--block-size`, the same
  `--prefix-caching-hash-algo`, and — for the non-crypto `xxhash*` algos,
  which seed `NONE_HASH` per process — a shared `PYTHONHASHSEED`. The keys the
  decoder asks its peer for are vLLM's own block hashes, computed
  independently on each pod; if they disagree, nothing matches no matter what
  the hint says.
- **kvcr `v0.1.1`** (what vLLM CI pins).

---

## 5. Phase 1 — sidecar: `kv_hints` injection (the v1 deliverable)

New file `pkg/sidecar/proxy/connector_kvcr.go`, mirroring `connector_p2p.go`.

### 5.1 Configuration

`pkg/sidecar/proxy/options.go`, mirroring `p2p-connector-port` /
`enable-p2p-pull`:

- `--kvcr-control-port` (env `KVCR_CONTROL_PORT`, default e.g. `17771`) — the
  rank-0 ZMQ control port; rank *r* is `base + r`. Same validation as
  `P2PConnectorPort`: 1-65535, and `base + DataParallelSize - 1 <= 65535`
  (`options.go:722-729`).
- `--kv-cache-pull-protocol={p2p|kvcr}`, default `p2p` — which request field
  the sidecar injects for a named source.
- `--enable-kvcr-pull` — compose a KVCR pull alongside a NIXL PD connector,
  mirroring `--enable-p2p-pull`.
- Add `kvcr` to `supportedKVConnectors` (`options.go:175`) for the PD path.

No hint-mode flag: `block_hashes` is always the placeholder (§6), so there is
nothing to configure.

**Recommendation:** a single `--kv-cache-pull-protocol` selector plus a `kvcr`
connector value. It keeps one source-selection path and one validation path,
and makes the two tiers mutually exclusive per deployment — which they must be,
since a given pod runs one secondary tier. The alternative (a second,
independent `kvcrCacheSource` header and parallel plumbing) doubles the EPP
surface for no gain, because the header's content is identical.

### 5.2 Endpoint derivation

Refactor `p2pPortFor` (`connector_p2p.go:305`) into a shared rank helper, then:

```go
// rankFor recovers the pod-local DP rank from the routed HTTP port.
func (s *Server) rankFor(targetHostPort string) int

// kvcrControlEndpoint returns the peer's KVCR ZMQ control endpoint as the
// kv.fetch payload names it. Host comes from the validated routing header;
// the port is derived locally from config, so this adds no SSRF surface.
func (s *Server) kvcrControlEndpoint(targetHostPort string) string {
    host := extractHost(targetHostPort)
    port := s.config.KVCRControlPort + s.rankFor(targetHostPort)
    return "tcp://" + net.JoinHostPort(host, strconv.Itoa(port))
}
```

`net.JoinHostPort` keeps IPv6 literals bracketed — the concern behind #3305.
Keep `p2pPortFor`'s conservative fallbacks (unparsable port, or rank outside
`[0, DataParallelSize)`, falls back to rank 0).

### 5.3 Envelope construction

```go
const (
    kvHintProtocolVersion = "0.1"
    kvFetchActionType     = "kv.fetch"
    kvFetchActionVersion  = "1.0"
    kvFetchModeCopy       = "copy"

    // placeholderBlockHash satisfies the parser's non-empty block_hashes rule.
    // KVCR never compares it: opportunistic_query (the vLLM adapter default)
    // skips the membership test, so the hint means "this request may pull from
    // this peer" and KVCR runs its own per-key lookup. Any uint64 works.
    placeholderBlockHash = uint64(1)
)

func (s *Server) kvcrFetchHints(sourceHostPort string) map[string]any
```

Emits the §2 shape with `block_hashes: [placeholderBlockHash]`, `mode: "copy"`,
no `no_retain`. The signature takes no hash list — there is deliberately no way
to pass one.

### 5.4 Decode-only pull

`decodeWithKVCRSource(w, r, sourceHostPort)`, mirroring `decodeWithP2PSource`
(`:343`) including its self-source guard (comparing against
`POD_IP:config.Port`, so another rank on the same pod is still a valid peer):

- set `requestData["kv_hints"]` — the sidecar owns the field, so any
  client-supplied value is dropped, exactly as it owns `kv_transfer_params`.
- do **not** touch `kv_transfer_params`; a NIXL PD deployment composing a KVCR
  pull needs its NIXL params intact.
- re-dispatch through `dispatchDecode` so chunked decode still applies.

### 5.5 PD path

`handleKVCR(w, r, prefillPodHostPort, kvCacheSource, apiType)`:

- **prefill request**: body unchanged apart from `reqcommon.CapSingleToken`.
  No `kv_transfer_params`, no `kv_hints` — a KVCR producer needs no priming.
  *Optionally* (symmetry with `addP2PPullToPrefill`) inject a `kv_hints` fetch
  on the **prefill** request when `kvCacheSource` names a third peer that
  out-caches the prefiller, so the prefiller itself pulls cached prefix.
- **decode request**: original body plus `kv_hints` naming the prefiller's
  control endpoint.
- **reuse `handleP2PSequentialRequests` unchanged** (rename to
  `handleSequentialPD`). Prefill-to-completion is not merely preferable here;
  KVCR transfers are destination-initiated, so a decode fetch that arrives
  first finds nothing and burns its deadline.

Wire it at `proxy.go:520` alongside the existing `handleP2P` /
`handleNIXLV2` selection.

### 5.6 Dispatch and validation

`dispatch.go:130-148`: generalize `p2pPullAvailable()` to
`kvCachePullAvailable()` covering `KVConnector == kvcr`,
`EnableKVCRPull && KVConnector == nixlv2`, and the existing P2P cases. The
`isHostPort` and `allowlistValidator` checks are unchanged and still apply —
the header still carries an HTTP `host:port`, and only the port is re-derived.

Then branch on `--kv-cache-pull-protocol` at both call sites
(`dispatch.go:204` and `:222`).

### 5.7 Observability

- `pkg/sidecar/constants`: add `KVConnectorKVCR = "kvcr"`.
- Reuse `semconv.LLMDPDProxyKVCacheSource`; pass the new connector value to
  `semconv.LLMDPDProxyConnector`.
- Log the resolved control endpoint and hash count at `Info`, as
  `decodeWithP2PSource` logs `kv_request_id` / `p2p_connector_port`.

---

## 6. Why there is no exact-hash mode

**Decision: always `opportunistic_query=True`, always a placeholder
`block_hashes`.** The router names a *peer*, never a block list. Recorded here
because the alternative looks tempting and is a trap.

What ruling it out buys:

- **No hash plumbing in the EPP.** No new header, no ~1.4 KB of base64
  per request, no reaching across plugin state for `perPromptKeys`, no
  decision about which prefix of the match to publish.
- **No EPP↔engine hash agreement.** Exact mode would have required the EPP's
  token processor to reproduce the engine's hashes bit for bit — same
  `--prefix-caching-hash-algo`, `VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES=1`, and a
  `PYTHONHASHSEED` shared between the *EPP* and every pod. Every one of those
  is a silent-miss trap: a mismatch degrades to `MISS` with no error anywhere.
- **No correctness cliff on hard inputs.** Exact hashing has no answer for
  per-block `extra_keys` — `cache_salt`, multimodal inputs, LoRA — or for chat
  requests, where the engine hashes the *templated* prompt. Each would have
  needed a detect-and-downgrade path back to opportunistic anyway.

What it costs, stated plainly: a longer **lookup leg**. The decoder asks the
named peer about every key in the request, including keys that peer never
held, and only the `missing_keys` negative cache (§3.1) trims the set — after
the first miss. That is control-plane traffic, not data-plane: it does not move
KV, and it is bounded per request. Given the measured lookup-leg behaviour
under concurrency, this is the number to watch after the first deployment, and
the reason §5.7 logs the hash count and §8 asserts hit counters.

**What this does *not* remove:** pod-to-pod block-key agreement. The decoder
still asks its peer for vLLM's own block hashes, computed independently on each
pod, so matching `--block-size`, `--prefix-caching-hash-algo` and (for
`xxhash*`) `PYTHONHASHSEED` across peers remain hard requirements — see §4.
Dropping exact mode removes the EPP from that contract, not the pods.

If a deployment ever genuinely needs `opportunistic_query: false`, finding 3
records where the real hashes live; it is a serialization job, and this section
is the argument against doing it.

---

## 7. Phase 2 — optional refinements

- **Price KVCR-owned blocks separately.** Add `ownership` to the vLLM event
  decoder (`vllm_msgpack.go`; it is trailing, so appending a 12th decoder is
  backward compatible) and carry it into the index so a block in a pod's KVCR
  DRAM/SSD pool can be weighted differently from one in vLLM's own host tier.
  Only worth doing for deployments that configure `secondary_g2_slots > 0` or
  `g3`; in the default index-and-pin configuration both are the same bytes.
- **`storage` tier weight** in `pkg/kvcache/backend.go` so KVCR G3 blocks are
  scored as fetchable-but-slow rather than ignored.
- **Docs**: a `kvcrsource` section in the `dataproducer` README, deployment
  requirements mirroring the p2psource list (block sizes equal,
  `offload_prompt_only: false` for decode-phase blocks, shared
  `PYTHONHASHSEED` across peers, `opportunistic_query` left on), and a
  `docs/disaggregation.md` entry for the control-port convention.

---

## 8. Testing

- **Sidecar unit** (`connector_kvcr_test.go`, mirroring
  `connector_p2p_test.go`): envelope shape against every hard rule in §2,
  including a non-empty `block_hashes` and an absent `no_retain`;
  `tcp://` formatting incl. IPv6; rank derivation and every fallback;
  self-source skip; `kv_transfer_params` left untouched; client-supplied
  `kv_hints` dropped.
    - Worth a dedicated case: the placeholder is a plain `uint64` and must not
      serialize as a float, a string, or a JSON `true` — the parser rejects
      bools and non-ints outright (`hint_parser.py:112-118`), so a marshalling
      slip here silently disables every hint.
- **Dispatch** (`dispatch_test.go`): protocol selector routes to the right
  injector; `kvCachePullAvailable` gating; SSRF rejection still blanks the
  source.
- **PD ordering**: prefill completes before decode dispatch; prefill error is
  returned verbatim and decode is never dispatched.
- **End-to-end**: the two-pod KVCR PD setup already verified by hand
  (64/64 hits, 32 MiB transferred) with the PoC proxy replaced by the sidecar.
  Assert `vllm:kv_offload_*` hit counters on the decoder, and confirm a
  one-sided peer restart or an unmatched `PYTHONHASHSEED` shows up as a
  silent miss rather than an error.
- **Negative**: no `router_capabilities` → tier refuses to construct;
  hint naming an unreachable endpoint → request still succeeds by recompute;
  a pod with `opportunistic_query: false` → hints stop matching, which is the
  regression this contract exists to catch.

---

## 9. Risks and open questions

1. **Phase 0 is the real gate.** Without `kv_hints` on vLLM's HTTP surface
   there is no way for any gateway to attach a hint. Upstreaming that field is
   the single highest-value action and benefits every orchestrator, not just
   llm-d.
2. **Silent degradation is the dominant failure mode.** Every hint mismatch
   degrades to `MISS` with no error. Mitigation: a decoder-side hit-rate metric
   and a sidecar log of the emitted endpoint and hash count, so "hints are
   being ignored" is observable rather than invisible.
3. **Peer restart.** KVCR does not distinguish a peer *restart* from an
   additive metadata update (its own TODO in `remote_fw_dram.py`), matching the
   observed fragility when a peer restarts under load. The router cannot fix
   this; it should avoid naming a source whose pod has restarted, which the
   index's removal events already imply.
4. **One tier per pod.** A pod runs either the P2P tier or KVCR, so the
   selector is a deployment-wide choice. If mixed fleets are ever needed, the
   source's protocol has to become a per-endpoint property in the index rather
   than sidecar config.
5. **`opportunistic_query` is an unenforceable contract.** The router cannot
   read it, and a pod that disables it degrades to recompute silently (§6).
   The only defence is a pinned deployment manifest plus the decoder-side
   hit-rate metric in risk 2 — worth stating explicitly in the runbook rather
   than discovering as a performance mystery.
