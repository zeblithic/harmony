# Multi-Platform Adapter Architecture — Design Spec

## Goal

Extract a reusable `harmony-runtime` library crate from the existing
`harmony-node` binary. This crate provides the sans-I/O runtime
orchestration (event/action pipeline, tier scheduling, peer management)
that every Harmony network participant needs, regardless of platform.
It also defines the `PlatformAdapter` trait boundary that Windows,
macOS, bare-metal, and future WASM consumers implement to join the
network.

## Background

Harmony currently has three consumer contexts:

- **harmony-node** — standalone Rust binary (Linux/macOS/Windows),
  wires harmony-reticulum's Node state machine + Zenoh + storage +
  compute into a tokio event loop. Contains ~6800 lines of runtime
  orchestration logic locked inside a binary crate with no lib target.
- **harmony-client** — Tauri v2 + Svelte 5 desktop app. Currently
  only does Zenoh capacity discovery; doesn't consume harmony core
  crates yet. Planned to embed a full node.
- **harmony-os** — bare-metal mesh OS (RPi5). Ring 1 unikernel uses
  harmony-reticulum's Node directly via a lightweight event loop.

The problem: the runtime orchestration logic in harmony-node can't be
reused. harmony-client would need to duplicate thousands of lines of
tier scheduling, peer management, and Zenoh routing to become a full
node. This spec solves that by extracting the shared runtime into a
library.

### Motivation: Heterogeneous Platforms

Research into deploying Harmony on consumer hardware (MSI Aegis RTX
5080 gaming PCs, Apple Silicon Macs) revealed that replacing the host
OS is impractical due to cryptographic boot restrictions (Microsoft
Pluton, UEFI Secure Boot), proprietary GPU firmware (NVIDIA GSP-RM),
and undocumented embedded controllers. The pragmatic path is a
**hosted adapter** — a Rust daemon or embedded library that runs
within the host OS and participates in the Harmony network through
standard OS APIs.

This requires a shared runtime library that abstracts away
platform-specific I/O while providing identical protocol behavior
across all deployment contexts.

## Design Decisions

- **Extract, don't rewrite**: Move existing battle-tested code from
  harmony-node into harmony-runtime. No behavioral changes. All
  existing harmony-node tests must pass after refactor.
- **Sans-I/O boundary at `tick()`**: `NodeRuntime::tick()` takes
  events, returns actions. All I/O is the caller's responsibility.
  This is the same pattern used by harmony-reticulum's `Node`.
- **PlatformAdapter composes existing traits**: Wraps EntropySource,
  PersistentState, and networking into a single trait that platforms
  implement once.
- **Compute is stubbed, not implemented**: The `ComputeBackend` trait
  and descriptor types exist to inform the design. HTTP bridging to
  LM Studio / ollama is the immediate follow-on, not part of this
  spec.
- **Attestation tiers, not binary trust**: Three-tier model
  (Sovereign, HardwareBound, Unattested) acknowledges that platforms
  differ in sovereignty without permanently penalizing hosted nodes.
- **harmony-os stays independent**: Ring 1 unikernel continues using
  harmony-reticulum's Node directly. harmony-runtime is heavier
  (Zenoh routing, storage tiers, compute scheduling) and not needed
  for the unikernel's current scope.

## Crate Structure

```
harmony/crates/
├── harmony-crypto/
├── harmony-identity/
├── harmony-platform/      # EntropySource, NetworkInterface, PersistentState
├── harmony-reticulum/     # Node state machine, packets, announces
├── harmony-zenoh/         # Key expressions, subscription tables
├── harmony-content/       # Content addressing, CAS
├── harmony-compute/       # WASM execution engine
├── harmony-workflow/       # Task orchestration
├── harmony-athenaeum/     # Content catalog
├── harmony-runtime/       # NEW — runtime orchestration library
│   ├── src/
│   │   ├── lib.rs         # Re-exports
│   │   ├── runtime.rs     # NodeRuntime state machine
│   │   ├── config.rs      # NodeConfig, tier scheduling, budgets
│   │   ├── adapter.rs     # PlatformAdapter + ComputeBackend traits
│   │   ├── attestation.rs # Trust tiers, AttestationReport
│   │   ├── discovery.rs   # PeerTable, PeerManager (sans-I/O)
│   │   └── contacts.rs    # ContactStore, tunnel peer tracking
│   └── Cargo.toml
└── harmony-node/          # REFACTORED — thin binary over harmony-runtime
```

### Dependency Graph

```
harmony-platform ─────────────────────────────────────┐
harmony-reticulum ────────────────────────┐            │
harmony-zenoh ──────────────┐             │            │
harmony-content ────────┐   │             │            │
                        ▼   ▼             ▼            ▼
                   harmony-runtime    harmony-unikernel (Ring 1)
                    ▲         ▲        (uses Node directly)
                    │         │
              harmony-node  harmony-client
              (daemon)      (Tauri app)
```

### Feature Flags

- `default = ["std"]` — for hosted platforms
- No-std support for potential future harmony-os consumption
- `compute` — enables ComputeBackend traits (gated since not all
  nodes offer compute)
- `attestation` — enables trust tier reporting

### License

MIT/Apache-2.0, same as all harmony core crates.

## PlatformAdapter Trait

The unification point — every platform implements this to host a
Harmony runtime.

```rust
pub trait PlatformAdapter {
    type Error: core::fmt::Debug;

    // --- Lifecycle ---

    /// Platform-specific initialization (open sockets, register
    /// service, etc.). Returns initial platform metadata.
    fn init(&mut self) -> Result<PlatformInfo, Self::Error>;

    /// Graceful shutdown. Release sockets, deregister service.
    fn shutdown(&mut self) -> Result<(), Self::Error>;

    // --- Networking ---

    /// Available network transports on this platform.
    fn available_interfaces(&self) -> Vec<InterfaceDescriptor>;

    /// Send raw bytes on a named interface.
    fn send(
        &mut self,
        interface: &str,
        data: &[u8],
    ) -> Result<(), Self::Error>;

    /// Poll for inbound data on any interface. Non-blocking.
    fn receive(&mut self) -> Option<(String, Vec<u8>)>;

    // --- Persistence ---

    /// Key-value blob storage.
    fn persistence(&mut self) -> &mut dyn PersistentState;

    // --- Entropy ---

    /// Cryptographic random bytes.
    fn entropy(&mut self) -> &mut dyn EntropySource;

    // --- Attestation ---

    /// Report trust tier and optional attestation evidence.
    fn attestation(&self) -> AttestationReport;

    // --- Compute ---

    /// Enumerate available compute backends.
    /// Empty vec if this node offers no compute.
    fn compute_backends(&self) -> Vec<ComputeBackendDescriptor>;
}
```

### Design Rationale

- **Composes** harmony-platform traits rather than replacing them.
  The adapter owns the implementations and hands out references.
- **Networking uses `send`/`receive` on the adapter** rather than
  separate trait objects per interface — simpler for platforms where
  socket management is centralized.
- **`available_interfaces()` returns descriptors, not trait objects**
  — the runtime uses interface names as string keys, matching Node's
  existing `interface_name: String` convention.
- **Attestation is read-only** — the runtime decides how to use it.
- **Compute backends are descriptors** — the runtime queries what's
  available and routes work through a separate ComputeBackend trait.

### InterfaceDescriptor

```rust
pub struct InterfaceDescriptor {
    pub name: String,
    pub kind: InterfaceKind,
    pub mtu: usize,
    /// Estimated bandwidth in bytes/sec. 0 = unknown.
    pub bandwidth_estimate: u64,
}

pub enum InterfaceKind {
    Udp,
    Tcp,
    Serial,
    LoRa,
    ZenohTunnel,
    IrohQuic,
    Other(String),
}
```

## Attestation Model

Three-tier trust model that acknowledges platform sovereignty
differences without permanently penalizing hosted nodes.

```rust
pub struct AttestationReport {
    pub tier: AttestationTier,
    /// Optional platform-specific evidence (TPM quote, Secure
    /// Enclave attestation, etc.). Opaque bytes.
    pub evidence: Option<Vec<u8>>,
}

pub enum AttestationTier {
    /// Hardware-rooted keys on a sovereign OS (e.g., RPi5 +
    /// harmony-os + TPM). Highest initial trust.
    Sovereign,
    /// Hardware-rooted keys but vendor-controlled OS (e.g.,
    /// Windows TPM, macOS Secure Enclave). Hardware binding
    /// verified but platform theoretically compromisable by
    /// vendor.
    HardwareBound,
    /// No hardware attestation. Trust earned through behavioral
    /// scoring (cooperation weight) only.
    Unattested,
}
```

The cooperation scoring system (`weight: Option<f32>` in
`NodeAction::SendOnInterface`) provides the behavioral trust
dimension. Over time, an Unattested node with consistent good
behavior can earn equivalent routing priority to a Sovereign node.
The attestation tier affects **initial** trust, not permanent status.

## Compute Backend

Pluggable compute providers behind a stable network-facing protocol.

### Descriptor Types (in harmony-runtime)

```rust
pub struct ComputeBackendDescriptor {
    pub id: String,
    pub name: String,
    pub kind: ComputeBackendKind,
    pub capabilities: Vec<ComputeCapability>,
}

pub enum ComputeBackendKind {
    /// HTTP inference server (LM Studio, ollama, vLLM, etc.).
    HttpInference { endpoint: String },
    /// Direct GPU access (future: CUDA, Metal, Vulkan compute).
    DirectGpu { device_id: String },
    /// CPU-only compute.
    Cpu,
}

pub enum ComputeCapability {
    /// LLM inference with a specific model.
    Inference { model_id: String, context_length: u32 },
    /// WASM module execution.
    WasmExecution { fuel_budget: u64 },
    /// Raw tensor compute (future).
    TensorCompute { flops_estimate: u64 },
}
```

### ComputeBackend Trait

```rust
pub trait ComputeBackend {
    type Error: core::fmt::Debug;

    fn run_inference(
        &mut self,
        model_id: &str,
        input: &[u8],
        params: &[u8],
    ) -> Result<Vec<u8>, Self::Error>;
}
```

The runtime never calls ComputeBackend directly — it emits
`RuntimeAction::RunInference` and the platform's event loop
dispatches to the appropriate backend. This preserves the sans-I/O
boundary.

### Compute Request Flow

```
Remote peer
  → Zenoh query
  → RuntimeEvent::ComputeQuery
  → NodeRuntime::tick() matches capability
  → RuntimeAction::RunInference
  → Platform event loop dispatches to backend (e.g., HTTP POST
    to localhost:1234/v1/completions)
  → RuntimeEvent (inference result)
  → NodeRuntime::tick()
  → RuntimeAction::SendReply
  → Platform sends Zenoh reply
```

## NodeRuntime Extraction

### What Moves to harmony-runtime

| Component | Approx Lines | Description |
|-----------|-------------|-------------|
| RuntimeEvent enum | ~100 | 28 inbound event variants |
| RuntimeAction enum | ~60 | 16+ outbound action variants |
| NodeConfig + sub-configs | ~120 | Budgets, scheduling, policy |
| NodeRuntime + tick() | ~4000 | Core state machine |
| Tier 1 routing logic | ~800 | Delegates to Node |
| Tier 2 storage logic | ~1200 | CAS, content policy, Zenoh keys |
| Tier 3 compute scheduling | ~400 | Dispatch, capacity tracking |
| PeerManager | ~300 | Announce scheduling, path expiry |
| PeerTable | ~200 | Peer discovery tracking |
| ContactStore | ~200 | Contact + tunnel peer state |

### What Stays in harmony-node

| Component | Why |
|-----------|-----|
| event_loop::run() | tokio::select! — platform-specific |
| UDP socket binding | OS-specific networking |
| Zenoh Session setup | Async runtime dependency |
| iroh Endpoint + QUIC tunnels | Async + platform-specific |
| mDNS daemon | Platform-specific discovery |
| disk_io.rs | Requires tokio spawn_blocking |
| S3 async fetches | Feature-gated async |
| Inference streaming | Feature-gated async |
| Signal handlers | Platform-specific |
| CLI + main() | Binary concerns |

### Core API

```rust
// harmony-runtime/src/runtime.rs

pub struct NodeRuntime {
    node: Node,                  // harmony-reticulum sans-I/O
    config: NodeConfig,
    book_store: MemoryBookStore, // in-memory CAS cache
    contacts: ContactStore,
    peer_manager: PeerManager,
    peer_table: PeerTable,
    // ... tier state, counters, queues
}

impl NodeRuntime {
    pub fn new(
        config: NodeConfig,
        identity: PrivateIdentity,
    ) -> Self;

    /// Process buffered events, return actions for the platform.
    /// Called each tick cycle (typically 250ms).
    pub fn tick(
        &mut self,
        events: &[RuntimeEvent],
    ) -> Vec<RuntimeAction>;

    pub fn register_interface(
        &mut self,
        name: &str,
        mode: InterfaceMode,
    );

    pub fn register_destination(
        &mut self,
        hash: [u8; 16],
    );
}
```

## Consumer Integration

### harmony-node (standalone daemon)

Pure code-movement refactor. The tokio event loop stays in the
binary, calls `NodeRuntime::tick()` instead of inline state machine
logic. All existing tests pass unchanged.

```rust
// harmony-node/src/event_loop.rs (after refactor)
async fn run(config: NodeConfig, identity: PrivateIdentity) {
    let mut runtime = NodeRuntime::new(config, identity);
    // ... setup UDP, Zenoh, iroh, mDNS as today

    loop {
        tokio::select! {
            _ = interval.tick() => {
                events.push(RuntimeEvent::TimerTick { now, unix_now });
                let actions = runtime.tick(&events);
                events.clear();
                for action in actions {
                    dispatch_action(&udp, &zenoh, action).await;
                }
            }
            Ok((len, addr)) = udp.recv_from(&mut buf) => {
                events.push(RuntimeEvent::InboundPacket { ... });
            }
            // ... other async arms push events
        }
    }
}
```

### harmony-client (Tauri embedded daemon)

Adds harmony-runtime as a dependency. Spawns an event loop task
alongside the Tauri app. Replaces the current ZenohState with the
full runtime.

```
harmony-client/src-tauri/
├── src/
│   ├── lib.rs        — Tauri setup, IPC commands
│   ├── daemon.rs     — embedded event loop driving NodeRuntime
│   └── adapters/
│       ├── common.rs — cross-platform defaults (UDP, getrandom, fs)
│       ├── windows.rs — future: Win32, Windows TPM, DirectStorage
│       └── macos.rs  — future: Secure Enclave, Network.framework
```

Starts with a cross-platform `common.rs` adapter using Rust std
library APIs. Platform-specific adapters are follow-on work.

### harmony-os (Ring 1 bare-metal)

Does NOT depend on harmony-runtime initially. Continues using
harmony-reticulum's Node directly via its lightweight
UnikernelRuntime. The unikernel doesn't need Zenoh routing, storage
tiers, or compute scheduling yet. When it does, it can adopt
harmony-runtime — the underlying Node state machine is the same.

## Follow-On Work

| Sub-project | Description | Depends on |
|-------------|-------------|------------|
| harmony-client daemon embedding | Wire harmony-runtime into Tauri, replace ZenohState | This spec |
| Windows platform adapter | Win32 networking, Windows TPM, service registration | Client embedding |
| macOS platform adapter | Network.framework, Secure Enclave, LaunchAgent | Client embedding |
| HTTP compute bridge | LM Studio / ollama discovery and request proxying | Client embedding |
| Standalone daemon mode | Extract embedded daemon to background service | Client embedding |
| Attestation protocol | Wire format for exchanging attestation reports | This spec (types) |
| Direct GPU compute | CUDA / Metal / Vulkan compute backends | HTTP bridge |
| harmony-os adoption | Ring 1 optionally adopts harmony-runtime | Independent |

## Testing

All tests use the existing harmony-node test suite. The refactor is
behavior-preserving — no new protocol features, no wire format
changes.

### Extraction validation
- All existing `cargo test -p harmony-node` tests pass after refactor
- `harmony-runtime` has its own unit tests for NodeRuntime, config
  parsing, peer management
- Doc tests demonstrating how to construct and tick a NodeRuntime

### Trait boundary validation
- Mock PlatformAdapter implementation in harmony-runtime's test suite
- Integration test: create NodeRuntime with mock adapter, feed
  events, verify expected actions

### Cross-crate validation
- harmony-node compiles and passes tests against harmony-runtime
- harmony-runtime compiles independently with `cargo test -p
  harmony-runtime`

## Out of Scope

- Per-platform adapter implementations (Windows, macOS, Linux-specific)
- Actual compute backend implementations (HTTP bridge, GPU)
- Attestation wire protocol (how nodes exchange/verify reports)
- harmony-os adopting harmony-runtime
- Zenoh session management (remains in platform-specific event loops)
- Tunnel/iroh infrastructure (remains in platform-specific code)
- Real hardware testing (this is a pure code-organization refactor)
