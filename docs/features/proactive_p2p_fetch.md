# Proactive P2P KV-cache fetch

An orchestrator can stage KV blocks on a vLLM instance **before** routing a
request to it. The destination pulls the named blocks from a source peer into
its CPU KV cache, off any request's critical path; the request that is later
routed there finds them through its normal lookup and promotes them to GPU,
instead of paying for a remote pull or a recompute.

A fetch copies: the source keeps its blocks and can keep serving them, to its
own requests and to other peers.

It builds on the [P2P secondary tier](https://github.com/vllm-project/vllm/tree/main/vllm/v1/kv_offload/tiering/p2p)
of the `OffloadingConnector` and on the `kv_hints` request envelope
([RFC #53421](https://github.com/vllm-project/vllm/issues/53421)). There is no
dedicated endpoint: the fetch is a `p2p.fetch` action carried by any
inference request.

## Requesting a fetch

Send the destination any request whose `kv_hints` carry a `p2p.fetch`
action. The blocks need not belong to that request, and the fetch outlives
it, so a one-token warm-up request is enough:

```bash
curl http://destination:8000/v1/completions -H 'Content-Type: application/json' -d '{
  "prompt": "Warm-up.",
  "max_tokens": 1,
  "kv_hints": {
    "protocol_version": "0.1",
    "message_id": "req-42",
    "actions": [{
      "action_id": "fetch",
      "action_type": "p2p.fetch",
      "action_version": "1.0",
      "payload": {
        "source": {"host": "10.0.0.5", "port": 5710},
        "blocks": ["<hex OffloadKey>", "..."]
      }
    }]
  }
}'
```

| Field | Meaning |
| ----- | ------- |
| `source.host`, `source.port` | The source instance's P2P endpoint (`VLLM_P2P_SIDE_CHANNEL_PORT` plus its DP rank). |
| `blocks` | Hex-encoded `OffloadKey`s: the block hash followed by the 4-byte big-endian KV-cache group index. At most 16,384. |
| `action_version` | `1.x`. Other major versions are ignored. |

The fetch id is `"<message_id>:<action_id>"`, and it is sent to the source
as the P2P `kv_request_id`. A hint repeating a live or recently finished id is
ignored rather than restarted.

Block identity is the content hash used everywhere in the offloading
subsystem, so the source and destination must agree on `PYTHONHASHSEED`, as
for any P2P sharing. The orchestrator can learn a source's block keys from its
KV-cache events: with `VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES=0`, each CPU
`BlockStored` event carries the block hash and `group_idx` of one stored
`OffloadKey`.

Because the hint rides a request, it reaches the DP rank that request is routed
to. Send the warm-up request to the rank that will serve the real request
(`X-data-parallel-rank`): each rank owns its own CPU cache.

## Observing completion

`kv_hints` has no reply channel. The destination reports each block through its
KV-cache events instead, so enable them (`--kv-events-config`) on the
destination as well:

- A block that lands in the CPU cache emits a CPU `BlockStored` event.
- A block whose transfer failed after a CPU slot was reserved emits a CPU
  `BlockRemoved` event.
- A block the source does not hold emits nothing, so wait with a timeout.

The destination also logs one line per fetch when it finishes, with the
counts of blocks done, missing on the source, failed, and not admitted for lack
of CPU capacity.

A malformed or unsupported action never fails the request carrying it: it is
logged and skipped, as `kv_hints` requires. The same applies to a fetch
the destination refuses: too many concurrent fetches (16), more blocks than
its CPU cache can hold, or no secondary tier that supports fetch.

## How it works

A fetch is a request-less symmetric-P2P consumer transaction, driven by the
destination's `TieringOffloadingManager` once per scheduler step:

1. When the carrying request is added, `on_new_request` validates the action
   and registers a fetch with a synthetic `remote_kv_source` request
   context, which opens the P2P session toward the source.
2. Each step, up to 256 keys across all fetches are probed through the
   normal `lookup()` path. The P2P tier batches them into a `LookupMsg`; the
   source answers from its own CPU cache and pins the hits.
3. Hits are promoted like any P2P hit: a CPU slot is reserved, a `FetchMsg`
   names it, and the source RDMA-writes the block into it.
4. The block is committed into the CPU cache index by hash, which emits the
   `BlockStored` event. It is never pulled to GPU.

The source needs nothing new: a fetch looks to it like any symmetric-P2P
consumer. Probes wait until the session handshake completes, since a live
request's lookup treats a still-connecting peer as a miss and would otherwise
report every block missing; a session that cannot be established fails the
fetch. Blocks unresolved after 300 s, or lost to a cache reset, fail.
