# PD Connector Design Document

## Abstract

In this design PD disaggregation is based on the vLLM CPU KV cache which is per vLLM instance and it is in canonical layout (single TP unified block size). The PD Connector is a secondary tier that is registered as such. Orchestration with the CPU cache is done via the PrimaryTier and it not known to the secondary tiers.

## Assumptions

- Decoder can be known or unknown to Prefiller when a request is submitted to the Prefiller (Deffered decode)
- A request can be submitted to the Decoder without any dependency on submitting the request to the prefiller
- Translating canonical layout to per GPU worker layout is done in the worker context (Secondary tiers are agnostic to that)
- A store operation to the CPU KV cache can fail only by:
  - Allocation failure
  - Prefiller crash

## Requirements

- Each Prefill node can service pull requests from any other node in the cluster
- Crash of Prefiller or Decoder should be handled gracefully without any resource leaks
- Support general P2P sharing pro-active and reactive
- Security?
- Different block size across nodes in the cluster?
- Performance of TTFT and Tok/Sec should be similar to the existing NixlConnector

## HLD

### Architecture

#### Components

- **OffloadingManager (CPU KV Cache)** — Manages the CPU KV cache per vLLM instance in canonical layout.

- **PrimaryTier** — The main component that drives the KV cache lifecycle. It interacts with the OffloadingManager to offload GPU memory to CPU and triggers secondary tiers accordingly.

- **SecondaryTier** — A pluggable component registered with the OffloadingManager that implements the actual KV cache transfer between nodes. The PD Connector is implemented as a secondary tier, handling load and store between peers.

#### Component Diagram

```mermaid
graph TD
    PP[PrimaryTier] -->|schedule load/store| OM["OffloadingManager<br/>CPU KV Cache"]
    PP -->|load/save| SP["SecondaryTier<br/>PD Connector"]
    PP -->|get_required_blocks| SP
    SP -->|CTRL:lookup_fetch| Remote[Remote Peer PD]
    SP -->|NIXL.Transfer| Remote
```

### Design Decisions

- P block IDs – how do we pass request's allocated blocks on Prefiller to Decoder.
  - **Option 1** - Add allocated blocks to request header on Prefiller.
    - When do we know for sure that blocks have been saved already
  - **Option 2** - Control message that will prepare the data operation and then trigger one-sided transfer.
- Allow streaming of saved KV blocks on the prefiller side to the Decoder.
  Implemented by allowing to submit the request to the decoder at once before KV blocks are computed on the Prefiller. This allows the prefiller to send KV blocks once they are in the CPU cache after receiving an allocate_fetch() control command from the decoder.
### API

#### Secondary Tier
- `register_secondary_tier()`
- `load(job_id, block_hashs, peer_id)` — Decoder initiates a load from Prefiller
- `get_required_blocks()` - Polled by the primary tier
- `save(job_id, block_descs)` — Prefiller saves blocks
- `get_finished()` — Async notification of load/save operation completion

### Flow

#### Sequence Diagram

```mermaid
sequenceDiagram
    participant Prefiller_OC as Prefiller PrimaryTier
    participant Prefiller_PD as Prefiller SecondaryTier PD
    participant Decoder_PD as Decoder SecondaryTier PD
    participant Decoder_OC as Decoder PrimaryTier

    Prefiller_PD->>Prefiller_PD: Open listener thread
    Decoder_PD->>Decoder_PD: Open listener thread

    Note over Prefiller_OC,Decoder_OC: ── Init time ──

    Decoder_OC->>Decoder_PD: load(job_id, block_hashs, peer_id)
    Note right of Prefiller_PD: If no connection to D exists,<br/>do handshake and create connection

    Decoder_PD->>Prefiller_PD: 𝗖𝗧𝗥𝗟:lookup_fetch(job_id, block_hashs, local_block_descs)

    Prefiller_OC->>Prefiller_PD: get_required_blocks()
    Note left of Prefiller_OC: Iterate over chunks till completion
    Prefiller_OC->>Prefiller_PD: save(job_id, block_descs)

    Prefiller_PD-)Decoder_PD: 𝗗𝗔𝗧𝗔:NIXL.Transfer(WRITE, local_block_descs, remote_block_descs)

    Prefiller_PD-->>Decoder_PD: Transfer complete
    Prefiller_PD-->>Prefiller_PD: Transfer complete

    Decoder_OC->>Decoder_PD: get_finished(job_id)
    Prefiller_OC->>Prefiller_PD: get_finished(job_id)

```
### Error Handling
#### Allocation Failure on Prefiller Side (valid only when a request is sent to the Decoder before the Prefiller completes it)
Temporary solution is based on timeout after lookup_fetch().
An abort request API can be considered that should be passed by the orchestrator layer.
#### vLLM Crash
- A lost control connection between the Prefiller and the Decoder should trigger an abort of all ongoing requests.
#### Submit lookup_fetch() before a request is submitted to the Prefiller
- lookup_fetch() should be constricted by a timeout to catch such a case.

## Implementation

### Step 1: SecondaryTiers

Introduce a `SecondaryTier` base class and a `SecondaryTiers` registry component. `PrimaryTier` depends on `SecondaryTiers` to dispatch operations. Individual `SecondaryTier` implementations register themselves into `SecondaryTiers` without any knowledge of `PrimaryTier`.

```
PrimaryTier --> SecondaryTiers --> [SecondaryTier, SecondaryTier, ...]
```

#### Tasks
- [ ] Define `SecondaryTier` abstract base class with `load`, `save`, `get_finished` methods
- [ ] Implement `SecondaryTiers` registry with `register(tier: SecondaryTier)` and dispatch methods
- [ ] `PrimaryTier` holds a reference to `SecondaryTiers` and calls it on `load`/`save`
- [ ] `SecondaryTiers` iterates registered tiers and calls each in sequence
- [ ] Add unit tests for registration and sequential dispatch

#### Tests

Tests are located in `tests/test_secondary_tiers.py`.

To run:
```bash
cd /home/lirans/my-utils/pd-connector
python3 -m pytest tests/test_secondary_tiers.py -v
```

### Step 2: PDConnector as a SecondaryTier

Implement `PDConnector` as a concrete `SecondaryTier`. It handles the actual KV cache transfer between Prefiller and Decoder nodes using NIXL.

On the **Prefiller side**, the PrimaryTier polls `get_required_blocks()` to determine which blocks the SecondaryTier needs, then calls `save()` to store KV block descriptors. The SecondaryTier waits for incoming `lookup_fetch` requests from the Decoder and initiates the transfer once blocks are available.
On the **Decoder side**, `load()` connects to the Prefiller peer, sends a `lookup_fetch` control message, and triggers a NIXL transfer to save the blocks into the local CPU cache.

#### Tasks
- [ ] Implement `PDConnector(SecondaryTier)` class in `src/pd_connector.py`
- [ ] `save(job_id, block_descs)` — register block descriptors, ready to serve `lookup_fetch`
- [ ] `load(job_id, block_hashes, peer_id)` — connect to peer, send `CTRL:lookup_fetch`, trigger NIXL transfer
- [ ] `get_finished(job_ids)` — poll NIXL transfer status and return completed job ids
- [ ] Add unit tests in `tests/test_pd_connector.py`

#### Tests

Tests are located in `tests/test_pd_connector.py`.

To run:
```bash
cd /home/lirans/my-utils/pd-connector
python3 -m pytest tests/test_pd_connector.py -v
```

### Step 3: ZMQ CtrlTransport

Implement `ZmqCtrlTransport` — a concrete `CtrlTransport` backed by ZMQ. The listener uses a ZMQ `ROUTER` socket so it can accept connections from multiple peers simultaneously. Each peer connects with a `DEALER` socket, and the router identifies peers by their socket identity.

Peer liveness is tracked via a heartbeat mechanism. If a peer stops sending heartbeats within the timeout window, the transport fires a `on_peer_down(peer_id)` callback so that `PDConnector` can abort all in-progress jobs for that peer.

```
Peer A (DEALER) ──┐
Peer B (DEALER) ──┼──► ZmqCtrlTransport (ROUTER) ──► PDConnector
Peer C (DEALER) ──┘         │
                        heartbeat monitor
                        → on_peer_down(peer_id)
```

#### Design decisions
- **Socket types**: `ROUTER` on the listener side, `DEALER` on the connecting side — allows multiplexing N peers on one port.
- **Heartbeat**: each peer sends a periodic `{"type": "heartbeat", "peer_id": "..."}` message. The listener tracks `last_seen[peer_id]` and fires `on_peer_down` if a peer exceeds `heartbeat_timeout_s`.
- **Message format**: MessagePack (msgspec) — consistent with the existing NIXL handshake wire format.
- **Graceful disconnect**: a `{"type": "disconnect", "peer_id": "..."}` message triggers immediate `on_peer_down` without waiting for timeout.

#### Keep-Alive Mechanism

Liveness is implemented using **ZMQ's built-in ZMTP heartbeat** — no application-level ping thread is needed. Both the ROUTER and each DEALER socket have the following socket options set at creation time:

| Option | Default | Description |
|---|---|---|
| `HEARTBEAT_IVL` | 2000 ms | How often ZMQ sends a PING to the peer |
| `HEARTBEAT_TIMEOUT` | 10000 ms | How long ZMQ waits for a PONG before closing the connection |
| `HEARTBEAT_TTL` | 10000 ms | How long the remote peer considers this side alive without a PING |

When ZMQ detects a dead connection (no PONG within `HEARTBEAT_TIMEOUT`), it closes the DEALER socket and fires `EVENT_DISCONNECTED` on that socket's monitor. A dedicated `_monitor_loop` background thread polls all DEALER monitor sockets using a ZMQ `Poller` and calls `on_peer_down(peer_id)` when `EVENT_DISCONNECTED` is received.

- **Startup**: heartbeats are activated automatically as soon as the socket is connected. No application-level handshake is required.
- **Shutdown**: on clean disconnect, `disconnect()` sends a `{"type": "disconnect"}` application message so the remote `_listener_loop` fires `on_peer_down` immediately without waiting for the heartbeat timeout to expire.

#### Tasks
- [ ] Implement `ZmqCtrlTransport(CtrlTransport)` in `src/zmq_ctrl_transport.py`
- [ ] Listener thread: ZMQ `ROUTER` socket, receives from any peer, dispatches to `recv()` queue
- [ ] Sender: ZMQ `DEALER` socket per peer (lazily created), sends messages to a specific peer
- [ ] Heartbeat sender: background thread sends heartbeat to each connected peer at `heartbeat_interval_s`
- [ ] Heartbeat monitor: background thread checks `last_seen` and calls `on_peer_down(peer_id)` on timeout
- [ ] Send `disconnect` message on `close()` before tearing down sockets
- [ ] Register `on_peer_down` callback in `PDConnector` to cancel in-progress jobs for the failed peer
- [ ] Add unit tests in `tests/test_zmq_ctrl_transport.py`

#### Tests

Tests are located in `tests/test_zmq_ctrl_transport.py`.

To run:
```bash
cd /home/lirans/my-utils/pd-connector
python3 -m pytest tests/test_zmq_ctrl_transport.py -v
```

### Step 4: Register Memory

Accept CPU KV block tensors at `PDConnector` init time and store them for later NIXL registration (which happens in Step 5). No NIXL calls are made in this step.

#### Design

`PDConnector.__init__()` receives a `kv_blocks: list[torch.Tensor]` argument representing the CPU KV cache blocks allocated by the OffloadingManager. The tensors are stored as `self._kv_blocks` for use in the NIXL registration step.

#### Tasks
- [ ] Add `kv_blocks: list[torch.Tensor]` parameter to `PDConnector.__init__()`
- [ ] Store as `self._kv_blocks = kv_blocks`
- [ ] Add unit test in `tests/test_pd_connector.py` verifying that tensors passed at init are accessible via `self._kv_blocks`

#### Tests

Tests are located in `tests/test_pd_connector.py`.

To run:
```bash
cd /home/lirans/my-utils/pd-connector
python3 -m pytest tests/test_pd_connector.py -v
```

### Step 5: NIXL Registration and Prepped Descriptor List

Create a `nixl_agent` inside `PDConnector` and register the CPU KV block tensors with NIXL at init time. Immediately prepare a local descriptor list handle (`nixl_prepped_dlist_handle`) from the registered memory so that future transfers can be initiated using only block indices — avoiding repeated descriptor preparation per transfer.

No metadata exchange with remote peers is performed in this step.

#### Design

At `PDConnector.__init__()` time, after storing `self._kv_blocks`:

1. Create `self._agent = nixl_agent(peer_id, nixl_agent_config(backends=["UCX"]))`
2. Register all KV blocks: `self._reg = self._agent.register_memory(self._kv_blocks)`
3. Prepare a local descriptor list: `self._local_dlist = self._agent.prep_xfer_dlist("NIXL_INIT_AGENT", self._kv_blocks)`

On `close()`, deregister memory and release the handle:
```
self._agent.release_dlist_handle(self._local_dlist)
self._agent.deregister_memory(self._reg)
```

#### NIXL API mapping

| PDConnector action | NIXL call |
|---|---|
| `__init__()` | `nixl_agent(peer_id, nixl_agent_config(backends=["UCX"]))` |
| `__init__()` | `agent.register_memory(self._kv_blocks)` → `self._reg` |
| `__init__()` | `agent.prep_xfer_dlist("NIXL_INIT_AGENT", self._kv_blocks)` → `self._local_dlist` |
| `close()` | `agent.release_dlist_handle(self._local_dlist)` |
| `close()` | `agent.deregister_memory(self._reg)` |

#### Tasks
- [ ] Create `nixl_agent` in `PDConnector.__init__()` with UCX backend; store as `self._agent`
- [ ] Call `self._agent.register_memory(self._kv_blocks)` and store result as `self._reg`
- [ ] Call `self._agent.prep_xfer_dlist("NIXL_INIT_AGENT", self._kv_blocks)` and store as `self._local_dlist`
- [ ] On `close()`, call `self._agent.release_dlist_handle(self._local_dlist)` then `self._agent.deregister_memory(self._reg)`
- [ ] Add unit test verifying `self._reg` and `self._local_dlist` are set after init
- [ ] Add unit test verifying deregistration is called on `close()`

### Step 6: Connection Establishment

When `load()` is called for a peer not yet connected, establish a ZMQ control channel connection. The Decoder sends its NIXL agent metadata and KV block descriptors to the Prefiller. The Prefiller uses these to prep a remote descriptor list (`remote_dlist`) so that future `make_prepped_xfer` transfers need only block indices.

The Prefiller does **not** send its metadata back. The Decoder is the WRITE target, not the initiator — `add_remote_agent` is only required on the side that initiates transfers (Prefiller).

#### peer_id format

`peer_id` encodes the remote ZMQ listener address as `"<host>:<port>"`. `load()` parses it to drive `_ctrl.connect()`.

#### Handshake

```
Decoder ──connect msg──► Prefiller
         {type: "connect",
          peer_id: decoder_id,
          agent_metadata: <bytes>,        # Decoder's NIXL metadata (needed by Prefiller to WRITE)
          block_descs: [[addr,len,dev_id], ...]}  # Decoder's kv_blocks as list of 3-tuples

Decoder ◄──connect_ack── Prefiller
         {type: "connect_ack",
          peer_id: prefiller_id}          # No metadata: Decoder does not initiate transfers
```

The Decoder blocks in `_connect()` until the `connect_ack` arrives (per-peer `threading.Event`, with timeout).

#### State added

| Field | Type | Description |
|---|---|---|
| `_connections` | `set[str]` | peer_ids with a completed handshake |
| `_connect_events` | `dict[str, Event]` | one Event per in-progress connect |
| `_remote_dlists` | `dict[str, nixl_prepped_dlist_handle]` | Prefiller's prepped dlist per Decoder peer |

#### Flow

**Decoder side** (`load()` → `_connect(peer_id)`):
1. Parse `host, port = peer_id.rsplit(":", 1)`
2. `self._ctrl.connect(peer_id, host, int(port))`
3. Send `connect` message with `self._agent.get_agent_metadata()` and kv_blocks as `[(addr, nbytes, dev_id), ...]`
4. Wait on `self._connect_events[peer_id]` (timeout = 10 s)
5. Mark `peer_id` in `self._connections`

**Prefiller side** (listener handles `connect`):
1. `self._agent.add_remote_agent(msg["agent_metadata"])` — Prefiller learns Decoder's transport endpoints
2. `self._remote_dlists[sender_id] = self._agent.prep_xfer_dlist(sender_id, block_descs, mem_type="cpu")`
3. Connect back to Decoder's ZMQ listener, mark `sender_id` in `self._connections`
4. Reply with `connect_ack` — **no metadata included**

**Decoder side** (listener handles `connect_ack`):
1. Mark `sender_id` in `self._connections`
2. Set `self._connect_events[sender_id]` — unblocks `_ensure_connected`

#### On peer down

`_on_peer_down()` removes the peer from `_connections` and releases its entry in `_remote_dlists`.

#### Tasks
- [ ] Change `peer_id` contract: format `"<host>:<port>"`; parse in `_connect()`
- [ ] Add `_connections: set[str]`, `_connect_events: dict[str, threading.Event]`, `_remote_dlists: dict[str, nixl_prepped_dlist_handle]`
- [ ] `load()`: if `peer_id` not in `_connections`, call `_connect(peer_id)` before sending `lookup_fetch`
- [ ] Implement `_connect(peer_id)`: parse host/port, ZMQ connect, send `connect` message, wait on event
- [ ] Listener handles `connect`: `add_remote_agent`, `prep_xfer_dlist` → `_remote_dlists`, send `connect_ack` (no metadata)
- [ ] Listener handles `connect_ack`: set event (no `add_remote_agent`)
- [ ] `_on_peer_down()`: clear `_connections` entry, release and remove `_remote_dlists` entry
- [ ] `close()`: release all `_remote_dlists` handles
- [ ] Add integration test: two connectors, `load()` triggers handshake, verify both sides have `_connections` populated and Prefiller has `_remote_dlists` entry

#### Tests

Tests are located in `tests/test_pd_connector.py`.

To run:
```bash
cd /home/lirans/my-utils/pd-connector
python3 -m pytest tests/test_pd_connector.py -v
```

### Step 7: Lookup-Fetch with Two-Phase Timeout

When the Decoder calls `load()`, it sends a `lookup_fetch` control message to the Prefiller. The Prefiller replies with `lookup_ack` to confirm the control message was received and the job exists. `lookup_ack` does **not** mean data has been transferred — it is only a control-plane confirmation.

This creates two distinct failure modes, which must be tracked separately:

| Failure | Cause |
|---|---|
| No `lookup_ack` before `ack_deadline` | Control message was lost, or Prefiller does not have the job registered |
| `lookup_ack` received but no transfer complete before `xfer_deadline` | Data transfer did not occur (NIXL not implemented yet in this step) |

No NIXL data transfer occurs in this step. Jobs always time out at the transfer phase.

#### Message format

`lookup_fetch` (Decoder → Prefiller):
```
{
  "type":          "lookup_fetch",
  "job_id":        <str>,
  "block_hashes":  [<int>, ...],
  "block_indexes": [<int>, ...]   # Decoder's kv_block slot for each hash (same length, same order)
}
```

`lookup_ack` (Prefiller → Decoder):
```
{
  "type":    "lookup_ack",
  "job_id":  <str>
}
```

#### Two-phase deadline tracking

```
load() called
    │
    ├─ ack_deadline = now + _LOOKUP_ACK_TIMEOUT_S
    │
    ▼
[waiting for lookup_ack]
    │
    ├── ack_deadline exceeded → ABORTED  (control failure: message lost or job not found)
    │
    ▼ lookup_ack received
    │
    ├─ xfer_deadline = now + _TRANSFER_TIMEOUT_S
    │
    ▼
[waiting for transfer complete]  ← future step will resolve this
    │
    ├── xfer_deadline exceeded → ABORTED  (transfer failure: data did not arrive)
    │
    └── transfer complete → DONE
```

#### Flow

**Decoder** (`load()`):
1. Verify an active connection via `_ensure_connected(peer_id)`. If handshake times out, raise immediately.
2. Create `_LoadJob` with `ack_deadline = now + _LOOKUP_ACK_TIMEOUT_S`.
3. Send `lookup_fetch` with `job_id`, `block_hashes`, and `block_indexes`.

> `SecondaryPillar.load()` signature: `load(job_id, block_hashes, block_indexes, peer_id)`.
> `block_hashes[i]` identifies the block; `block_indexes[i]` is the Decoder's kv_block slot (index into the descriptors registered at init) where that block should be written.

**Prefiller** (handles `lookup_fetch`):
1. Look up `_save_jobs[job_id]`. If not found or aborted, return (NACK is a future step).
2. Reply with `lookup_ack` to the Decoder.

**Decoder** (handles `lookup_ack`):
1. Record `ack_received = True`.
2. Set `xfer_deadline = now + _TRANSFER_TIMEOUT_S`.
3. Do **not** mark job as DONE — data transfer has not occurred yet.

**Decoder `get_finished()`**:
- If `not ack_received` and `now > ack_deadline`: abort — control failure.
- If `ack_received` and `now > xfer_deadline`: abort — transfer failure.
- Return jobs in DONE state (transfer completion is a future step).

#### State added

| Field | Type | Description |
|---|---|---|
| `_LoadJob.ack_received` | `bool` | True once `lookup_ack` has been received |
| `_LoadJob.ack_deadline` | `float` | Deadline for receiving `lookup_ack`; exceeded → control failure |
| `_LoadJob.xfer_deadline` | `float` | Set on ack receipt; exceeded → transfer failure |

#### Tasks
- [ ] `_LoadJob`: add `ack_received: bool = False`, `ack_deadline: float`, `xfer_deadline: float = 0.0`
- [ ] `load()`: set `ack_deadline = now + _LOOKUP_ACK_TIMEOUT_S`; send `lookup_fetch`
- [ ] `_handle_lookup_fetch` (Prefiller): send `lookup_ack` back to Decoder
- [ ] `_handle_lookup_ack` (Decoder): set `ack_received = True`, `xfer_deadline = now + _TRANSFER_TIMEOUT_S`; do NOT mark DONE
- [ ] `get_finished()`: abort on ack timeout (control failure) or xfer timeout (transfer failure); return DONE jobs
- [ ] Add `_LOOKUP_ACK_TIMEOUT_S` and `_TRANSFER_TIMEOUT_S` class constants
- [ ] Test: load with no matching save → no ack → ack_deadline fires → ABORTED (control failure)
- [ ] Test: load with matching save → ack received → no transfer → xfer_deadline fires → ABORTED (transfer failure)

#### Tests

Tests are located in `tests/test_pd_connector.py`.

To run:
```bash
cd /home/lirans/my-utils/pd-connector
python3 -m pytest tests/test_pd_connector.py -v
```

### Step 8: NIXL Transfer with Per-Chunk Completion

When `save(job_id, chunk_descs)` is called on the Prefiller, it checks whether a `lookup_fetch` is pending for that job. If one exists, it immediately initiates a NIXL WRITE for the newly saved blocks. The Prefiller polls its in-flight NIXL handles and, when a chunk completes, sends a `transfer_done` control message to the Decoder listing the block hashes that were just transferred. The Decoder accumulates these until all requested hashes are received, at which point the job transitions to DONE.

This supports the streaming case: `lookup_fetch` may arrive before all KV blocks are computed on the Prefiller. Each chunk is transferred as soon as it is saved.

#### Block index mapping (Prefiller ↔ Decoder)

`load()` accepts `block_indexes: list[int]` alongside `block_hashes`. Each `block_indexes[i]` is the Decoder's pre-allocated kv_block slot (index into the descriptors registered at init) where `block_hashes[i]` should be written. Both lists are forwarded verbatim in the `lookup_fetch` message.

On the Prefiller side, each `BlockDesc` in `save()` has an `addr`. Using `_addr_to_idx[addr]` gives the local kv_block index. The `block_hash` field identifies the block, and `hash_to_remote_idx[block_hash]` (built from the `lookup_fetch` payload as `dict(zip(block_hashes, block_indexes))`) gives the remote kv_block index to pass to `make_prepped_xfer`.

#### New message: `transfer_done`

`transfer_done` (Prefiller → Decoder):
```
{
  "type":         "transfer_done",
  "job_id":       <str>,
  "block_hashes": [<int>, ...]   # hashes of blocks transferred in this chunk
}
```

#### Prefiller: pending fetch tracking

`_pending_fetches: dict[str, _PendingFetch]` — keyed by job_id, created when `lookup_fetch` arrives.

```python
@dataclass
class _PendingFetch:
    job_id: str
    decoder_peer_id: str
    hash_to_remote_idx: dict[int, int]  # block_hash → Decoder's kv_block index; built as dict(zip(block_hashes, block_indexes))
    remaining_hashes: set[int]          # hashes not yet transferred
```

`_xfer_handles: list[tuple[object, str, list[int]]]` — `(nixl_handle, job_id, chunk_hashes)` — for in-flight transfers.

#### Flow

**Prefiller** (handles `lookup_fetch`):
1. Send `lookup_ack` (same as Step 7).
2. Build `hash_to_remote_idx = dict(zip(msg["block_hashes"], msg["block_indexes"]))`.
3. Create `_PendingFetch` with `hash_to_remote_idx` and `remaining_hashes = set(msg["block_hashes"])`.
4. Call `_try_transfer(job_id)` to transfer any blocks already saved.

**Prefiller** (`save(job_id, chunk_descs)`):
1. Store chunk in `_save_jobs` as before.
2. Call `_try_transfer(job_id)`.

**Prefiller** (`_try_transfer(job_id)`):
1. Intersect saved `block_descs` (by hash) with `pending_fetch.remaining_hashes`.
2. For each matching block: resolve `local_idx = _addr_to_idx[desc.addr]`, `remote_idx = hash_to_remote_idx[desc.block_hash]`.
3. Call `make_prepped_xfer("WRITE", local_dlist, local_indices, remote_dlists[decoder], remote_indices)` → `handle`.
4. Call `transfer(handle)`, append `(handle, job_id, chunk_hashes)` to `_xfer_handles`.
5. Remove chunk hashes from `remaining_hashes`.

**Prefiller** (`get_finished(job_ids)`):
1. Poll `_xfer_handles`: for each handle, call `check_xfer_state(handle)`.
2. On `"D"` (Done): release handle, send `transfer_done(job_id, chunk_hashes)` to Decoder.
3. If `remaining_hashes` for the job is now empty: mark `_save_jobs[job_id]` as DONE.

**Decoder** (handles `transfer_done`):
1. Add `msg["block_hashes"]` to `_load_jobs[job_id].received_hashes`.
2. If `received_hashes` is a superset of `load_job.block_hashes`: mark job DONE.

**Decoder** (`get_finished(job_ids)`):
- Unchanged from Step 7: returns DONE jobs; aborts timed-out ones.

#### State added

| Side | Field | Type | Description |
|---|---|---|---|
| Prefiller | `_pending_fetches` | `dict[str, _PendingFetch]` | Active fetch requests waiting for chunks |
| Prefiller | `_xfer_handles` | `list[tuple[handle, str, list[int]]]` | In-flight NIXL handles with job context |
| Prefiller | `_addr_to_idx` | `dict[int, int]` | `kv_block.data_ptr() → list index`, built at init |
| Decoder | `_LoadJob.received_hashes` | `set[int]` | Block hashes confirmed transferred so far |

#### Sequence (streaming example)

```
Decoder.load("job1", block_hashes=[h0,h1,h2], block_indexes=[3,7,2], prefiller_id)
    └─► lookup_fetch{block_hashes:[h0,h1,h2], block_indexes:[3,7,2]}
         → Prefiller: hash_to_remote_idx={h0:3, h1:7, h2:2}, remaining={h0,h1,h2}
         → sends lookup_ack

Prefiller.save("job1", [desc(h0, addr=A0)])     ← first chunk
    └─► _try_transfer: local_idx=_addr_to_idx[A0], remote_idx=3
         WRITE kv_blocks[local_idx] → Decoder.kv_blocks[3]
         remaining = {h1, h2}
    └─► NIXL completes → transfer_done(job1, [h0]) → Decoder.received = {h0}

Prefiller.save("job1", [desc(h1,A1), desc(h2,A2)])   ← second chunk
    └─► _try_transfer: WRITE kv_blocks[local(h1)]→D[7], kv_blocks[local(h2)]→D[2]
    └─► NIXL completes → transfer_done(job1, [h1,h2]) → Decoder.received={h0,h1,h2}
         == requested → job DONE
```

#### Tasks
- [ ] `SecondaryPillar.load()`: extend signature to `load(job_id, block_hashes, block_indexes, peer_id)`
- [ ] `PDConnector.load()`: store `block_indexes` in `_LoadJob`; forward in `lookup_fetch` message
- [ ] Add `_LoadJob.block_indexes: list[int]`
- [ ] Build `_addr_to_idx: dict[int, int]` at `__init__()` from `self._kv_blocks`
- [ ] Define `_PendingFetch` dataclass
- [ ] Add `_pending_fetches: dict[str, _PendingFetch]` and `_xfer_handles` to `__init__()`
- [ ] `_handle_lookup_fetch`: build `hash_to_remote_idx` from `zip(block_hashes, block_indexes)`; create `_PendingFetch`; call `_try_transfer()`
- [ ] `save()`: after storing, call `_try_transfer(job_id)` if `_pending_fetches` has the job
- [ ] Implement `_try_transfer(job_id)`: intersect, build index lists, call NIXL, update remaining
- [ ] `get_finished()` (Prefiller): poll `_xfer_handles`, on done send `transfer_done`, release handle
- [ ] Listener: handle `transfer_done` on Decoder side
- [ ] `_handle_transfer_done`: accumulate `received_hashes`, mark DONE when complete
- [ ] Add `_LoadJob.received_hashes: set[int]`
- [ ] `close()`: release all in-flight NIXL handles in `_xfer_handles`
- [ ] Test: save all blocks before load → single NIXL write → decoder gets transfer_done → job DONE
- [ ] Test: save in two chunks after load → two NIXL writes → decoder accumulates → job DONE after second chunk
- [ ] Test: partial save (only some hashes) → decoder never reaches DONE → xfer_deadline fires → ABORTED

#### Tests

Tests are located in `tests/test_pd_connector.py`.

To run:
```bash
cd /home/lirans/my-utils/pd-connector
python3 -m pytest tests/test_pd_connector.py -v
```
