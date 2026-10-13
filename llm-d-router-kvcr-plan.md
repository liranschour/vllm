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

Router baseline for the line references below:
`llm-d/llm-d-router @ 46f3635` ("Keep IPv6 addresses intact in endpoint
subsets", #3305). The implementation landed against `f45f48b5`; the anchors
cited here were re-checked against that commit.

**Status.** Phase 0 (§4) is in place: the vLLM `kv_hints` HTTP change is
committed as `9e1d6f2d75` with its entrypoint tests passing, though **not yet
upstream**. Phase 1 (§5) is implemented and pushed — commit `522fa0c5` on
`feat/kvcr-kv-hints` in `liranschour/llm-d-router`. Phase 2 (§7) is explicitly
**not** being implemented.

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

**Status: implemented.** `feat(sidecar): drive KVCR cached-prefix pulls with a
kv_hints envelope`, commit `522fa0c5` on `feat/kvcr-kv-hints`
(`liranschour/llm-d-router`), against upstream `f45f48b5`. Build, `go vet`,
`gofmt` and `golangci-lint` (v2.14.0, the version `Dockerfile.builder` pins)
are clean, and the whole `./pkg/...` suite passes.

This section describes the code as it landed. Where it diverges from the design
as first written, the divergence is called out — §5.1 dropped a flag and §5.6 is
new.

New file `pkg/sidecar/proxy/connector_kvcr.go`, mirroring `connector_p2p.go`.

### 5.1 Configuration

`pkg/sidecar/proxy/options.go`, mirroring `p2p-connector-port` /
`enable-p2p-pull`:

- `--kvcr-control-port` (env `KVCR_CONTROL_PORT`, default `17771`) — the
  rank-0 ZMQ control port; rank *r* is `base + r`. Validated like
  `P2PConnectorPort`: 1-65535, and `base + DataParallelSize - 1 <= 65535`.
- `--enable-kvcr-pull` — compose a KVCR pull alongside a NIXL PD connector,
  mirroring `--enable-p2p-pull`. Rejected unless `--kv-connector=nixlv2`, and
  **rejected together with `--enable-p2p-pull`**.
- `kvcr` added to `supportedKVConnectors` for the PD path, so it also appears
  in the `--kv-connector` help text automatically.
- `KVCRControlPort` / `EnableKVCRPull` on `Config`, plus `yamlConfiguration`
  entries and merge arms so both are settable from the config file.

No hint-mode flag: `block_hashes` is always the placeholder (§6), so there is
nothing to configure.

**Dropped during implementation: `--kv-cache-pull-protocol={p2p|kvcr}`.** The
design called for an explicit selector. It turned out to be redundant and
unsafe: the protocol is already a total function of
`(kv-connector, enable-p2p-pull, enable-kvcr-pull)`, so a third knob could only
ever contradict the other two — `--kv-connector=kvcr
--kv-cache-pull-protocol=p2p` has no meaningful behaviour. Two predicates
derive it instead (§5.7), and the mutual-exclusion check above makes the
ambiguous combination unrepresentable rather than merely discouraged.

The rejected alternative is unchanged: a second, independent `kvcrCacheSource`
header with parallel EPP plumbing would double the EPP surface for no gain,
because the header's content is identical.

### 5.2 Endpoint derivation

`p2pPortFor`'s rank derivation was extracted into a shared helper, so both
tiers resolve a peer's rank through one implementation:

```go
// rankFor recovers the target's pod-local DP rank from its routed port,
// returning 0 whenever the rank cannot be trusted. kind names the caller for
// the debug logs ("P2P" / "KVCR").
func (s *Server) rankFor(targetHostPort, kind string) int

// p2pPortFor is now just the base plus that rank.
func (s *Server) p2pPortFor(targetHostPort string) int {
    return s.config.P2PConnectorPort + s.rankFor(targetHostPort, "P2P")
}

// kvcrControlEndpoint returns the peer's KVCR ZMQ control endpoint as the
// kv.fetch payload names it. Host comes from the validated routing header;
// the port is derived locally from config, so this adds no SSRF surface.
func (s *Server) kvcrControlEndpoint(targetHostPort string) string {
    host := extractHost(targetHostPort)
    port := s.config.KVCRControlPort + s.rankFor(targetHostPort, "KVCR")
    return "tcp://" + net.JoinHostPort(host, strconv.Itoa(port))
}
```

`rankFor` keeps every one of `p2pPortFor`'s conservative fallbacks — no data
parallelism, zero `dpBasePort`, an unparsable or non-numeric port, or a rank
outside `[0, DataParallelSize)` all yield rank 0, i.e. the caller's base port.
The refactor is behaviour-preserving for P2P; the existing `p2pPortFor` table
test still passes unchanged.

`net.JoinHostPort` keeps IPv6 literals bracketed — the concern behind #3305 —
and two test entries pin that, since an unbracketed `fd00::5:17771` cannot be
parsed back into host and port.

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

The envelope's own key names are constants too (`hintFieldProtocolVersion`,
`hintFieldActions`, `hintFieldSourceControlEndpoint`, …) rather than inline
strings, so the parser's vocabulary lives in one place and the tests assert
against the same symbols the builder uses. `action_id` is derived as
`message_id + "-fetch"`, which keeps a request's envelope and its single action
correlatable in logs without a second UUID.

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
  No `kv_transfer_params`, no `kv_hints` of its own — a KVCR producer needs no
  priming. The optional symmetry with `addP2PPullToPrefill` was implemented:
  `addKVCRPullToPrefill` injects a `kv_hints` fetch on the **prefill** request
  when `kvCacheSource` names a peer other than the prefiller, so the prefiller
  itself pulls cached prefix. It no-ops on an empty source or a source equal to
  the prefiller (compared through `normalizeEndpoint`, so a scheme-prefixed
  value still matches).
- **decode request**: original body plus `kv_hints` naming the prefiller's
  control endpoint.
- **`handleP2PSequentialRequests` became `handleSequentialPD`**, now taking a
  `connector string` so its two spans report `kvcr` rather than hard-coding
  `offloading`. That was the one change needed to share it; the body is
  otherwise untouched. Prefill-to-completion is not merely preferable here —
  KVCR transfers are destination-initiated, so a decode fetch that arrives
  first finds nothing and burns its deadline.

Wired into `setKVConnector` (`proxy.go`) as a new `case
constants.KVConnectorKVCR`, alongside the existing `handleP2P` /
`handleNIXLV2` arms.

### 5.6 The NIXLv2 compose path — a bug this design missed

Not in the original plan, and the one genuine defect implementation turned up.

`handleNIXLV2` composes a cached-prefix pull onto its prefill request, and it
did so by calling `addP2PPullToPrefill` **unconditionally**, at two sites (the
sequential and parallel-WRITE paths). That is correct only because
`dispatch.go` blanks `kvCacheSource` when no pull is available. Once
`--enable-kvcr-pull` makes a source available under `nixlv2`, those call sites
would have injected a P2P `remote_kv_source` block into a deployment whose
secondary tier is KVCR — the wrong protocol, silently, with the hint the engine
actually reads never sent at all.

The fix is a protocol-aware wrapper both sites now call:

```go
// addCachedPrefixPullToPrefill injects the prefill-side pull in whichever
// protocol this deployment's secondary tier speaks. Takes the whole request
// body, since kv_hints is top-level while remote_kv_source nests under
// kv_transfer_params.
func (s *Server) addCachedPrefixPullToPrefill(prefillRequest map[string]any, kvCacheSource, prefillPodHostPort string)
```

It routes on `kvcrPullAvailable()`, and its P2P branch uses a checked type
assertion on `kv_transfer_params` so a caller that has not built that field
gets a no-op instead of a panic. `handleP2P` keeps calling
`addP2PPullToPrefill` directly: it only runs under `--kv-connector=offloading`,
where `kvcrPullAvailable()` is false by construction.

The general lesson for the rest of this work: a second tier makes every
*unconditional* protocol choice in the existing code a latent wrong-protocol
bug, and those call sites are not where the new code is.

### 5.7 Dispatch and validation

`dispatch.go` gates the source header on `!s.p2pPullAvailable() &&
!s.kvcrPullAvailable()` and branches on `s.kvcrPullAvailable()` at the
decode-only call site. The `isHostPort` and `allowlistValidator` checks are
unchanged and still apply — the header still carries an HTTP `host:port`, and
only the port is re-derived locally.

**Divergence from the design:** there is no single `kvCachePullAvailable()`.
Two predicates read better than one that answers "some tier is available" and
then needs a second question to say which:

```go
func (s *Server) kvcrPullAvailable() bool {
    return s.config.KVConnector == constants.KVConnectorKVCR ||
        (s.config.EnableKVCRPull && s.config.KVConnector == constants.KVConnectorNIXLV2)
}
```

`p2pPullAvailable()` is untouched and stays correct, since `kvcr` is not
`offloading`. Validation (§5.1) keeps the two from being true at once.

### 5.8 Observability

- `pkg/sidecar/constants`: added `KVConnectorKVCR = "kvcr"`.
- `semconv.LLMDPDProxyKVCacheSource` reused as-is;
  `semconv.LLMDPDProxyConnector` now receives the connector from
  `handleSequentialPD`'s new parameter, so KVCR spans are attributed to `kvcr`.
- Both entry points log the resolved control endpoint at `Info`, where
  `decodeWithP2PSource` logs `kv_request_id` / `p2p_connector_port`.

Deliberately **not** logged: a hash count. The design asked for one, but with
`block_hashes` always a one-element placeholder it would be a constant, and
printing it would imply the router had chosen those hashes. The endpoint is the
only part of the hint the router actually decides, so it is the only part worth
logging.

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
under concurrency, this is the number to watch after the first deployment —
hence the decoder-side hit-rate assertion in §8, which is the signal that
actually distinguishes "hints working, lookups chatty" from "hints ignored".

**What this does *not* remove:** pod-to-pod block-key agreement. The decoder
still asks its peer for vLLM's own block hashes, computed independently on each
pod, so matching `--block-size`, `--prefix-caching-hash-algo` and (for
`xxhash*`) `PYTHONHASHSEED` across peers remain hard requirements — see §4.
Dropping exact mode removes the EPP from that contract, not the pods.

If a deployment ever genuinely needs `opportunistic_query: false`, finding 3
records where the real hashes live; it is a serialization job, and this section
is the argument against doing it.

---

## 7. Phase 2 — optional refinements (not being implemented)

Kept as a record of what was consciously left out, not as queued work. Note
that the docs bullet is the one with a user-visible cost: `--kv-connector=kvcr`
and `--enable-kvcr-pull` currently exist only in their flag help text.

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

### Written, in `connector_kvcr_test.go` (Ginkgo, mirroring `connector_p2p_test.go`)

- **`kvcrPullAvailable`** table: `kvcr` always on; `nixlv2` on only with the
  flag; `offloading` off (it is the P2P tier); the flag inert on `sglang`.
- **`kvcrControlEndpoint`** table, 12 entries: rank 0 / mid / last-rank
  offsets, single-DP, and every fallback (port below base, beyond the rank
  range, missing, non-numeric, zero `dpBasePort`), plus a scheme-prefixed
  target and two IPv6 literals asserting the brackets survive.
- **`kvcrFetchHints`**: the endpoint is offset by the source's rank; `mode` is
  `copy` and `no_retain` absent; `block_hashes` is a non-empty `[]uint64`;
  `message_id` is fresh per envelope and `action_id` derives from it; rank
  derivation survives `Clone()`, which rank servers depend on.
    - One case earns its place on its own: the placeholder must serialize as a
      JSON **integer**, asserted as `"block_hashes":[1]` on the marshalled
      body. The parser rejects bools and non-ints outright
      (`hint_parser.py:112-118`), so a float or string here would disable every
      hint silently rather than fail loudly.
- **`addKVCRPullToPrefill`**: injects across ranks on one pod; no-ops when the
  source is the prefiller, when the source is the same endpoint spelled with a
  scheme, and when no source is named.
- **`addCachedPrefixPullToPrefill`** (§5.6): emits `kv_hints` under
  `nixlv2 + --enable-kvcr-pull` **and leaves `kv_transfer_params` untouched**;
  emits `remote_kv_source` under `nixlv2` without it; does not panic when
  there are no `kv_transfer_params` to nest under.

A note on method, because the suite hides it: `go test` prints no Ginkgo spec
names, so a new spec that never runs looks exactly like one that passes. These
were confirmed to execute by breaking an assertion and watching it fail — which
also printed the real marshalled envelope,
`{"actions":[{"action_id":"…-fetch","action_type":"kv.fetch","action_version":"1.0","payload":{"block_hashes":[1],"mode":"copy","source_control_endpoint":"tcp://10.0.0.9:17773"}}],"message_id":"…","protocol_version":"0.1"}`,
checked by eye against §2.

### Not written, and worth knowing

- **Dispatch-level** (`dispatch_test.go`): that the gate admits a source under
  each tier and that the decode branch picks the matching injector. Covered
  indirectly — the predicates and both injectors are tested in isolation — but
  not the wiring between them.
- **PD ordering** for `handleKVCR`: that prefill completes before decode is
  dispatched, and that a prefill error returns verbatim with decode never
  dispatched. The existing P2P suite covers `handleSequentialPD`'s body, which
  is now shared and unchanged, so the risk is the `handleKVCR` call site only.
- **End-to-end**: the two-pod KVCR PD setup already verified by hand (64/64
  hits, 32 MiB transferred) with the PoC proxy replaced by the sidecar. Assert
  `vllm:kv_offload_*` hit counters on the decoder, and confirm a one-sided peer
  restart or an unmatched `PYTHONHASHSEED` shows up as a silent miss rather
  than an error. **This is the only test that would prove the feature works at
  all** — everything above proves only that the router emits a well-formed
  envelope, never that an engine acts on it.
- **Negative**: no `router_capabilities` → tier refuses to construct; hint
  naming an unreachable endpoint → request still succeeds by recompute; a pod
  with `opportunistic_query: false` → hints stop matching, which is the
  regression that contract exists to catch.

---

## 9. Risks and open questions

1. **Phase 0 is the real gate.** Without `kv_hints` on vLLM's HTTP surface
   there is no way for any gateway to attach a hint. Upstreaming that field is
   the single highest-value action and benefits every orchestrator, not just
   llm-d.
2. **Silent degradation is the dominant failure mode.** Every hint mismatch
   degrades to `MISS` with no error. The sidecar side of the mitigation is in
   place — both entry points log the emitted control endpoint (§5.8) — but the
   half that would actually detect the failure is not: a decoder-side hit-rate
   metric. Until that exists, "hints are being ignored" and "hints are working"
   look identical from the router.
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
