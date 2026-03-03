# Node Trinity: Router / Storage / Compute

Three priority-tiered logical sidecars within a single Harmony node runtime.
Each tier is a sans-I/O state machine driven by the same tokio async runtime,
differentiated by CPU scheduling priority. All three are Zenoh queryables on
the same session — local and remote resources use the exact same interface.

---

## 1. Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                    Harmony Node Runtime                  │
│                                                         │
│  ┌─────────────────────────────────────────────────┐   │
│  │  Tier 1: ROUTER (highest priority)              │   │
│  │  harmony-reticulum Node + harmony-zenoh matching │   │
│  │  • Packet forwarding, path table, link table     │   │
│  │  • Zenoh pub/sub/queryable routing               │   │
│  │  • Never starved — gets CPU first, always        │   │
│  └─────────────────────────────────────────────────┘   │
│                         ▲                               │
│                    zero-copy                            │
│                         ▼                               │
│  ┌─────────────────────────────────────────────────┐   │
│  │  Tier 2: STORAGE (middle priority)              │   │
│  │  harmony-content ContentStore + Zenoh queryable  │   │
│  │  • W-TinyLFU cache, BlobStore backends           │   │
│  │  • Subscribes to content announcements           │   │
│  │  • Replies to content queries from any peer      │   │
│  │  • "Local librarian" — best use of local storage │   │
│  └─────────────────────────────────────────────────┘   │
│                         ▲                               │
│                    zero-copy                            │
│                         ▼                               │
│  ┌─────────────────────────────────────────────────┐   │
│  │  Tier 3: COMPUTE (lowest priority)              │   │
│  │  WASM execution + durable checkpointing          │   │
│  │  • wasmtime (desktop) or wasmi (embedded)        │   │
│  │  • Queryable: harmony/compute/activity/{type}    │   │
│  │  • Best-effort on excess resources               │   │
│  │  • Offloadable to more capable peers             │   │
│  └─────────────────────────────────────────────────┘   │
│                                                         │
│  ┌─────────────────────────────────────────────────┐   │
│  │  Zenoh Session (unified IPC bus)                 │   │
│  │  All three tiers are queryables on the same      │   │
│  │  Zenoh session — local and remote look identical │   │
│  └─────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

Priority enforcement:
- Router tasks drain their queue completely before yielding.
- Storage tasks process one query, then yield.
- Compute tasks execute N WASM instructions, then yield.
- When the router is busy, storage and compute get minimal CPU — correct
  behavior, the network must not bottleneck.

The Zenoh queryable interface enables future process separation without code
changes: since everything communicates via key expressions, splitting a tier
into a separate process is a Zenoh configuration change, not a code change.

---

## 2. Transparent Resource Resolution

Local and remote resources use the exact same Zenoh key expression interface.
The only difference is latency.

### Fetch flow

```
App: zenoh_get("harmony/content/a/abc123...")
  │
  ├──→ Local Tier 2 ContentStore     → Reply (~μs)
  ├──→ LAN peer's Tier 2             → Reply (~ms)
  └──→ Remote peer's Tier 2          → Reply (~100ms+)

Zenoh returns the FIRST reply (closest/fastest wins)
```

### Opportunistic caching along the path

Every Tier 2 subscribes to `harmony/content/transit/**`. When content passes
through this node's router (Tier 1), Tier 2 sees it via subscription. If the
cache has capacity and W-TinyLFU admission allows, the content is stored
locally. Future queries for the same CID are served locally. Popular content
naturally replicates toward the edges.

### Locality metadata (optional)

```rust
pub enum Locality {
    Local,      // Same node (Tier 2 cache hit)
    Lan,        // Same network segment
    Remote,     // Cross-network
}
```

Apps that care (e.g., transport quality UI indicators) can inspect locality.
Apps that don't care just get the bytes.

### Cache sizing is node-local policy

| Device type   | Typical cache | Backend          |
|---------------|--------------|------------------|
| Mobile phone  | 50–200 MB    | Memory-only      |
| Laptop        | 1–4 GB       | Memory + SQLite  |
| Desktop       | 4–32 GB      | Memory + RocksDB |
| Home server   | 100 GB+      | Memory + disk    |
| Dedicated node| Terabytes    | Memory + SSD     |

W-TinyLFU admission ensures a 50 MB phone cache with good admission beats a
1 TB cache with random eviction.

---

## 3. Tier 1 — The Router

Highest priority. Singular job: move packets as fast as possible, never block.

### Already built

- `harmony-reticulum::Node` — sans-I/O state machine for packet forwarding,
  path table maintenance, link management, announce rate limiting, packet
  deduplication.
- `harmony-zenoh::QueryableRouter` + `PubSubRouter` + `LivelinessRouter` —
  key expression matching, query dispatch, presence tokens.

### What the router does NOT do

- Store content (Tier 2).
- Execute application logic (Tier 3).
- Make caching decisions (Tier 2).
- Interpret payloads (forwards opaque bytes).

### Priority enforcement

```rust
loop {
    // Tier 1: drain ALL router events before yielding
    while let Ok(event) = router_rx.try_recv() {
        let actions = router.handle(event);
        dispatch_router_actions(actions).await;
    }
    yield_now().await;
}
```

### Router-to-Tier-2 handoff

When a packet arrives for a local queryable:
1. Router matches via `QueryableRouter`.
2. Emits `QueryableAction::DeliverQuery`.
3. Tier 2 picks up the query from a shared channel.
4. Router is already processing the next packet.

The router never blocks waiting for Tier 2 or Tier 3.

---

## 4. Tier 2 — The Storage Librarian

Middle priority. Job: make the best use of available local storage to serve
the network.

### Dual role

```
Subscriber role (incoming data):
  harmony/content/transit/**   → opportunistic caching of passing content
  harmony/content/publish/**   → explicit content storage requests
  harmony/announce/**          → content availability announcements

Queryable role (outgoing data):
  harmony/content/{prefix}/**  → serve cached content to any requester
  harmony/storage/stats        → cache metrics
```

### Storage tier structure

```rust
pub struct StorageTier<B: BlobStore> {
    cache: ContentStore<B>,      // W-TinyLFU admission + eviction
    backend: B,                  // Pluggable: memory, SQLite, RocksDB, disk
    budget: StorageBudget,       // How much memory/disk to use
    metrics: StorageMetrics,     // Hit rates, eviction counts, capacity
}
```

### Admission decisions

| Signal                              | Decision                                  |
|-------------------------------------|-------------------------------------------|
| Content passes through, never seen  | Admit to window segment if capacity allows|
| Content passes through frequently   | Promote to protected segment              |
| Content explicitly published        | Always admit                              |
| Content pinned by user/app          | Never evict, up to 50% cap                |
| Cache full, new content arrives     | W-TinyLFU frequency comparison            |
| Content not accessed in a long time | Demote protected → probation → evict      |

### Cross-application cache sharing

Every app on the same node shares the same Tier 2 cache. If the chat client
fetches a profile picture, the notification daemon gets it for free:

```
Chat app:         zenoh_get("harmony/content/a/abc...")  → miss → fetch → cache
Notification app: zenoh_get("harmony/content/a/abc...")  → HIT → instant
File browser:     zenoh_get("harmony/content/a/abc...")  → HIT → instant
```

No per-app caches, no duplicated fetches. One librarian serves them all.

### Budget negotiation

```
Total capacity:     4 GB
Pinned (reserved):  1.5 GB  (37.5% — under 50% cap)
Network cache:      2.5 GB  (available for opportunistic caching)
```

---

## 5. Tier 3 — The Compute Engine

Lowest priority. Job: execute WASM workloads on excess resources, with durable
checkpointing for crash recovery and offload.

### Queryable interface

```
App: zenoh_get("harmony/compute/activity/image-thumbnail", payload: cid)
  │
  ├──→ Local Tier 3 (spare CPU, AC power)        → executes locally
  ├──→ LAN desktop's Tier 3 (beefy, idle)        → executes there
  └──→ Remote peer's Tier 3 (dedicated compute)   → executes remotely
```

Same interface as content fetching. Zenoh routes to the best available host.

### WASM execution model

```rust
pub trait ComputeRuntime {
    fn execute(
        &mut self,
        module: &[u8],
        input: &[u8],
        budget: InstructionBudget,
    ) -> ComputeResult;
}

pub enum ComputeResult {
    Complete { output: Vec<u8> },
    Yielded { checkpoint: Checkpoint },
    NeedsIO { request: IORequest, checkpoint: Checkpoint },
    Failed { error: ComputeError },
}
```

Two implementations:
- `WasmtimeRuntime` — JIT, fast, desktop/server (requires std).
- `WasmiRuntime` — interpreter, portable, embedded/mobile (no_std).

### Instruction budgeting

```
Router drains queue     → yields
Storage: one query      → yields
Compute: 10K WASM insns → yields
Router drains queue     → yields
...
```

Budget scales with system load:
- Router idle → large budget (100K instructions per slice).
- Router busy → tiny budget (1K) or pause entirely.
- On battery → reduce budget further or offload.

### Durable checkpointing (Temporal-inspired)

```
WASM executing...
  │
  ├── needs external data
  │   → Checkpoint: save WASM memory + stack + PC
  │   → Issue Zenoh query for data
  │   → Store checkpoint: harmony/workflow/{id}/checkpoint
  │
  │   ... node may crash, restart, or offload ...
  │
  ├── Data arrives
  │   → Load checkpoint
  │   → Resume WASM from where it left off
  │
  └── Complete
      → Publish result: harmony/compute/result/{id}
      → Cache result in Tier 2
```

Any node can pick up a checkpoint and resume. WASM is deterministic —
replay from checkpoint produces identical results.

### Compute offload decision (automatic with hints)

```rust
pub enum ComputeHint {
    PreferLocal,
    PreferPowerful,
    LatencySensitive,
    DurabilityRequired,
}
```

System evaluates: battery level, CPU load, network latency to peers,
peer advertised capacity, power source, estimated computation size.

If `cost_local > cost_offload + transfer_overhead`, offload.

---

## 6. Unified Zenoh Key Expression Namespace

```
harmony/
├── reticulum/                          # Tier 1: Router
│   ├── announce/{dest_hash}            # Path announcements
│   ├── link/{link_id}                  # Bidirectional link data
│   └── diagnostics/{node_addr}         # Router stats (queryable)
│
├── content/                            # Tier 2: Storage
│   ├── {prefix}/**                     # Content fetch (16 sharded queryables)
│   ├── publish/{cid}                   # Explicit store request (pub/sub)
│   ├── transit/**                      # Opportunistic cache subscription
│   └── stats/{node_addr}               # Cache metrics (queryable)
│
├── announce/                           # Tier 2: Content availability
│   └── {cid}                           # "I have this content" (pub/sub)
│
├── compute/                            # Tier 3: Compute
│   ├── activity/{type}                 # Execute activity (queryable)
│   ├── result/{workflow_id}            # Computation results (pub/sub)
│   └── capacity/{node_addr}            # Advertised compute capacity
│
├── workflow/                           # Tier 3: Durable execution
│   └── {workflow_id}/
│       ├── checkpoint                  # Latest checkpoint
│       └── history/**                  # Event replay log
│
├── presence/                           # Cross-tier: Liveliness
│   └── {community_id}/{peer_addr}      # Online/offline tokens
│
├── community/                          # Application layer
│   └── {hub_id}/channels/{channel_name}
│
├── messages/                           # Application layer
│   └── {peer_addr}/inbox
│
└── profile/                            # Application layer
    └── {peer_addr}                     # Profile metadata + avatar CIDs
```

All queries work identically whether local, LAN, or remote:

```rust
zenoh_get("harmony/content/a/abc123...")        // Fetch content
zenoh_get("harmony/compute/activity/thumbnail") // Run computation
zenoh_get("harmony/profile/deadbeef...")        // Get peer profile
```

Cross-application sharing: every app on the node queries the same Zenoh
session. The local Tier 2 serves all of them from one cache.

---

## 7. Crate Structure & Implementation Scope

### Existing crates (extend)

| Crate              | Extension                                              |
|--------------------|--------------------------------------------------------|
| `harmony-content`  | `StorageTier<B>` wrapper, `StorageBudget`, transit sub |
| `harmony-zenoh`    | Namespace constants, `Locality` metadata on replies    |
| `harmony-reticulum`| Router-to-storage handoff channel type                 |
| `harmony-node`     | Full node runtime: instantiate all tiers, event loop   |

### New crates

| Crate              | Purpose                                                |
|--------------------|--------------------------------------------------------|
| `harmony-compute`  | `ComputeRuntime` trait, wasmtime + wasmi impls,        |
|                    | instruction budgeting, checkpoint serialization        |
| `harmony-workflow` | Durable execution orchestration, checkpoint storage    |
|                    | via Zenoh, replay logic, offload decision heuristics   |

### Updated dependency graph

```
harmony-crypto
  └── harmony-identity
      └── harmony-reticulum
harmony-zenoh (standalone)
harmony-content (depends on harmony-crypto)
harmony-compute (depends on harmony-content)
  └── harmony-workflow (depends on harmony-compute + harmony-zenoh)
harmony-node (depends on ALL — the orchestrator)
```

### harmony-node event loop

```rust
async fn run_node(config: NodeConfig) {
    // Tier 1: Router
    let mut router = Router::new(config.router);

    // Tier 2: Storage
    let mut storage = StorageTier::new(config.storage_budget, backend);
    storage.declare_queryables(&zenoh_session).await;
    storage.subscribe_transit(&zenoh_session).await;

    // Tier 3: Compute
    let mut compute = ComputeTier::new(config.compute_budget, wasm_runtime);
    compute.declare_queryables(&zenoh_session).await;

    // Priority event loop
    loop {
        // Tier 1: drain all (highest priority)
        while let Ok(event) = router_rx.try_recv() {
            dispatch(router.handle(event)).await;
        }

        // Tier 2: one event (middle priority)
        if let Ok(event) = storage_rx.try_recv() {
            storage.handle(event).await;
        }

        // Tier 3: one slice (lowest priority)
        if let Ok(task) = compute_rx.try_recv() {
            compute.execute_slice(task).await;
        }

        yield_now().await;
    }
}
```

Sans-I/O preserved: all tier logic is testable with loopback interfaces.
Only `harmony-node` does actual I/O.

---

## Implementation Beads

Suggested priority order:

1. **Extend harmony-content with StorageTier wrapper** — queryable + subscriber
   integration for Tier 2.
2. **Extend harmony-zenoh with namespace constants and Locality** — the shared
   vocabulary.
3. **Extend harmony-node to full runtime** — wire up Tier 1 + Tier 2 event loop.
4. **New: harmony-compute crate** — ComputeRuntime trait, wasmi impl, budgeting.
5. **New: harmony-workflow crate** — checkpoint model, replay, offload heuristics.
6. **Integrate Tier 3 into harmony-node** — complete the trinity event loop.
7. **Add wasmtime runtime** — JIT alternative for desktop/server.
