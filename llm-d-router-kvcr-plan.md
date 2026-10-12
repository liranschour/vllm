# Enabling KVCR P2P sharing from llm-d-router, via `kv_hints`

Plan for teaching `llm-d-router` to drive **KVCR** peer-to-peer KV sharing by
emitting a `kv_hints` envelope, alongside (not replacing) the existing
OffloadingConnector **P2P tier** path that drives sharing with
`kv_transfer_params`.

Scope: the router only. The KVCR tier itself and the vLLM `kv_hints` plumbing
are prerequisites, tracked in [`KVCR.md`](KVCR.md). The P2P tier is **not**
being migrated or removed — the two mechanisms coexist behind one selector.

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
| Keys named | implicit, tier does its own lookup | **`block_hashes` explicit and non-empty** |
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

The parser is strict: `protocol_version "0.1"`, `action_type "kv.fetch"`,
`action_version "1.0"`, `mode` must be `"copy"`, `no_retain` absent,
`block_hashes` non-empty.

---

## 3. Findings that shape the plan

These came out of reading both trees and are why the plan is small.

1. **`opportunistic_query` defaults to `True`** in the vLLM adapter
   (`vllm/v1/kv_offload/tiering/kvcr/manager.py:287`). It drops the
   hash-membership test, so the decoder queries the named peer for *every* key
   in the request. **A v1 router does not need real block hashes** — a
   single-element placeholder list satisfies the non-empty rule and the real
   transfer path still runs. This is what makes a sidecar-only v1 possible.
2. **The EPP needs no change for v1.** `x-llm-d-kv-cache-source-host-port`
   already carries exactly what the hint needs — which peer holds the prefix.
   Selection, sampling, the `minCachedTokenDelta` gate and the
   `ReusablePrefixTokens` floor are all protocol-agnostic. **All v1 work is
   sidecar-side.**
3. **`PrefixCacheMatchInfo` carries counts only** — `matchBlocks`,
   `cachedBlockCount`, `confirmedCachedBlockCount`, `cachedBlocksByTier`
   (`pkg/epp/framework/plugins/datalayer/attribute/prefix/data_types.go:46-77`)
   — and no hashes. So exact-hash mode cannot read them from the attribute.
4. **But the keys are already in plugin state.** The precise producer stashes
   `perPromptKeys [][]kvblock.BlockHash` under `blockKeysStateKey`
   (`preciseprefixcache/producer.go:436`, `prerequest.go:60`), and
   **`kvblock.BlockHash` is `uint64`** (`pkg/kvcache/kvblock/index.go:164`) —
   the exact type `block_hashes` wants. Exact mode is a *serialization* job,
   not a recomputation job.
5. **`ownership` is ignored but harmless.** The router's vLLM event adapter
   decodes a fixed 11-field prefix and skips the remainder
   (`pkg/kvevents/engineadapter/vllm_msgpack.go:161-183`, `:201`). vLLM's
   `locality`/`ownership`/`session_id` trail that prefix
   (`vllm/distributed/kv_events.py:88` and `:123`), so KVCR-tagged events decode
   without misalignment — the router just cannot tell a KVCR-owned block from
   a vLLM-owned one.
6. **The CPU-tier gate works for KVCR events.** `medium` becomes `DeviceTier`
   and is lowercased (`pkg/kvevents/pool.go:52`), so vLLM's `Medium.CPU`
   (`"CPU"`) lands as `"cpu"` and matches p2psource's
   `cpuDeviceTier = "cpu"` (`producer.go:327`). KVCR's `medium=STORAGE` (G3)
   becomes `"storage"` and is correctly *excluded* from the CPU floor.
7. **Rank derivation is reusable verbatim.** KVCR indexes `control_ports` by
   `dp_local_rank`, the same pod-local rank `p2pPortFor` already recovers from
   the routed HTTP port. One shared helper serves both.
8. **`control_advertise_host` defaults to `socket.gethostname()`**
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
  `control_advertise_host: $POD_IP`, and a known `control_ports` base.
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
- `--kvcr-hint-mode={opportunistic|exact}`, default `opportunistic` (see
  Phase 2).

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
)

func (s *Server) kvcrFetchHints(sourceHostPort string, blockHashes []uint64) map[string]any
```

Emits the §2 shape. `mode: "copy"`, no `no_retain`. In `opportunistic` mode
`blockHashes` is the one-element placeholder.

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

## 6. Phase 2 — exact block hashes (`--kvcr-hint-mode=exact`)

Needed only to exercise real hash matching, or to run with
`opportunistic_query: false`. Opportunistic mode is correct but queries the
peer for every key, so exact mode is the efficiency play, not a correctness fix.

The sidecar has no tokenizer, so the hashes must come from the EPP, which
already has them (finding 4).

1. **EPP — publish the keys.** In `p2psource`'s `PreRequest`, read
   `blockKeysState.perPromptKeys` and emit the matched prefix keys for the
   *selected source* on a new header, e.g.
   `x-llm-d-kv-cache-source-block-hashes`. Register it in
   `pkg/common/routing/common.go` so it inherits `SetRoutingHeader` /
   `TakeRoutingHeaderValues` and the ingress strip.
   - The EPP must publish only the blocks the source *confirmably holds in a
     pull-servable tier* — `BlocksByTier["cpu"]`, the same prefix the
     `ReusablePrefixTokens` floor is computed from — not the whole prompt.
   - Cross-plugin state access (precise-cache state read from p2psource) may
     warrant promoting the keys into a declared data-layer attribute instead of
     reaching into another plugin's state key.
2. **Encoding.** A 10k-token prompt at `blockSizeTokens: 64` is ~156 hashes;
   decimal would be ~3 KB of header. Use base64 of little-endian `uint64`s
   (~1.4 KB) and cap the count, dropping to opportunistic mode past the cap.
3. **Sidecar.** Decode the header into `[]uint64` and pass it to
   `kvcrFetchHints`. Treat a malformed value as absent and fall back to the
   placeholder — never fail the request over an advisory hint.
4. **Agreement requirements** (silent-miss traps, all confirmed in `KVCR.md`
   and the PoC proxy docstring; a mismatch degrades to `MISS`, it does not
   error):
   - `VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES=1` (the default) — at `0` the
     decoder's keys decode to raw bytes and can never match a uint64 list.
   - the EPP token processor's hash algo must equal the engine's
     `--prefix-caching-hash-algo`.
   - `xxhash*` seeds `NONE_HASH` per process, so exact mode needs `sha256*` or
     a **shared `PYTHONHASHSEED`** across every pod *and* the EPP.
   - exact mode assumes no per-block extra keys: `cache_salt`, multimodal
     inputs and LoRA requests fall outside it — force opportunistic there.
     `extra_keys` is already decoded by the EPP adapter, so the producer can
     detect and downgrade.

---

## 7. Phase 3 — optional refinements

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
  `PYTHONHASHSEED` where exact mode is used), and a `docs/disaggregation.md`
  entry for the control-port convention.

---

## 8. Testing

- **Sidecar unit** (`connector_kvcr_test.go`, mirroring
  `connector_p2p_test.go`): envelope shape and strict-parser conformance;
  `tcp://` formatting incl. IPv6; rank derivation and every fallback;
  self-source skip; `kv_transfer_params` left untouched; client-supplied
  `kv_hints` dropped; malformed hash header ignored.
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
  hint naming an unreachable endpoint → request still succeeds by recompute.

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
5. **Chat-template hashing.** Exact hashes over chat requests depend on the EPP
   tokenizing exactly as the engine templates. The precise producer already
   owns this contract; exact mode inherits it, and `/v1/completions` or
   token-id prompts are the safe path.
