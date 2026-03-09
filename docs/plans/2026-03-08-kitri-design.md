# Kitri — Harmony's Native Programming Model for Durable Distributed Computation

## Summary

Kitri is Harmony's programming model for durable, distributed computation. It provides first-class native Rust execution under OS-level isolation, with WASM as a fallback for untrusted and cross-platform code. Programs declare intent through Rust proc-macros and a TOML manifest; the Harmony runtime handles durability, retries, pub/sub wiring, and capability enforcement.

The core guarantee: **your workflow completes exactly once, or you get a clear error with no side effects to clean up.**

Named after a person who embodies setting a goal and making it work — determined, considerate, tenacious — Kitri gives everyone the ability to create anything they can dream of, and the system makes sure it actually happens.

## Design Principles

- **Zero-ceremony defaults, opt-in power.** Works casually, rewards intentionality.
- **Native Rust first, WASM second.** The OS provides isolation; no need to tax trusted code with WASM overhead.
- **Durable by default.** All I/O is event-sourced at the 9P boundary. Explicit checkpoints for expensive compute.
- **AI-friendly.** Rust's compiler + Kitri's type system + capability declarations = machine-verifiable correctness. AI writes the logic; the toolchain keeps it honest.

## Architecture — The Layered Cake

Kitri follows the same concentric ring pattern as the rest of Harmony:

```
Layer 0: harmony-kitri       (no_std, Ring 0)
         Core traits, types, manifest parsing, checkpoint serialization.
         Sans-I/O. No runtime. The "protocol" of Kitri.

Layer 1: kitri-sdk + macros   (std, Ring 0+)
         Proc-macros (#[kitri::workflow], #[kitri::subscribes], etc.)
         Runtime library wiring to harmony-workflow + zenoh + content.
         Works anywhere Rust compiles. The developer experience.

Layer 2: kitri-service        (Ring 2, microkernel)
         OS-level process supervisor, automatic 9P event sourcing,
         DAG-aware scheduler, UCAN trust tiering.
         Progressive enhancement — Layer 1 programs gain these
         capabilities for free when deployed on Harmony OS.

Layer 3: kitri-wasm           (Ring 0+)
         WASM compilation target + adapter to harmony-compute runtime.
         For untrusted code, cross-platform portability, non-Rust languages.
```

A Kitri program written with Layer 1 macros works on a laptop (no OS integration), works better on Harmony OS (automatic durability + trust tiering), and can be compiled to WASM (universal portability). Same code, progressive enhancement.

### Execution Tiers

| Tier | Runtime | Isolation | Durability | Use case |
|------|---------|-----------|------------|----------|
| Native Rust | Direct process under Ring 2 | OS-level: capabilities, Lyll/Nakaiah, sandboxed user | 9P event sourcing + explicit checkpoints | Trusted code, performance-critical |
| WASM | harmony-compute (Wasmi/Wasmtime) | WASM sandbox + OS-level | Full memory checkpointing + event sourcing | Untrusted code, cross-language |

### Build Order

Layer 0 → Layer 1 → ship it → Layer 3 (WASM) → Layer 2 (OS integration, as microkernel matures).

## Programming Model

### Macro System

The `kitri-macros` proc-macro crate provides annotations that generate distributed plumbing:

```rust
use kitri_sdk::prelude::*;

#[kitri::workflow]
#[kitri::subscribes("shipments/incoming")]
#[kitri::publishes("shipments/verified")]
async fn verify_shipment(manifest: Manifest) -> KitriResult<Verified> {
    // Fetch content by CID — automatically event-sourced.
    // On replay after crash, the cached response is returned instantly.
    let scan = kitri::fetch(manifest.scan_cid).await?;

    // Invoke an AI model — also event-sourced, won't re-run on replay.
    let summary = kitri::infer("summarize scan findings", &scan).await?;

    // Explicit checkpoint — expensive work above is preserved.
    kitri::checkpoint(&summary).await?;

    // Query another Kitri program via Zenoh queryable.
    let proof = kitri::query("blockchain/verify", &scan.hash).await?;

    // Return value is published to "shipments/verified" automatically.
    Ok(Verified { manifest, proof, summary })
}
```

**What the macros generate:**
- 9P server registration for the workflow's namespace entry
- Zenoh subscriber declaration for input topics
- Zenoh publisher declaration for output topics
- Event-sourcing wrapper around every `kitri::fetch`, `kitri::query`, `kitri::infer` call
- Capability request declarations (derived from topics and CIDs accessed)
- Retry/backoff wrapper with defaults from `Kitri.toml`

### SDK Primitives

The `kitri_sdk` crate provides a small set of durable I/O primitives:

| Primitive | Purpose | Event-sourced? |
|-----------|---------|----------------|
| `kitri::fetch(cid)` | Retrieve content by CID | Yes |
| `kitri::store(data)` | Store content, returns CID | Yes |
| `kitri::query(topic, input)` | Zenoh queryable invocation | Yes |
| `kitri::publish(topic, data)` | Publish to Zenoh topic | Staged until audit passes |
| `kitri::subscribe(topic)` | Returns async stream of messages | N/A (trigger, not I/O) |
| `kitri::infer(prompt, context)` | AI model invocation | Yes (cached on replay) |
| `kitri::checkpoint(state)` | Explicit durability save point | Persisted to content store |
| `kitri::spawn(workflow, input)` | Launch child workflow | Yes (child gets own WorkflowId) |
| `kitri::seal(data, identity)` | Encrypt via kernel (no key exposure) | Yes |
| `kitri::open(sealed)` | Decrypt via kernel | Yes |

### Manifest (`Kitri.toml`)

Program-level configuration that doesn't belong in code:

```toml
[package]
name = "shipment-verifier"
version = "0.1.0"

[runtime]
max_retries = 3
retry_backoff = "exponential"
timeout = "5m"
fuel_budget = 1_000_000       # instruction budget for WASM tier

[capabilities]
subscribe = ["shipments/incoming"]
publish = ["shipments/verified"]
fetch = ["shipments/**"]       # CID namespace access
infer = true                   # allowed to invoke AI models

[trust]
signers = ["did:key:z6Mk..."]  # UCAN issuers whose signatures are accepted

[deploy]
prefer_native = true           # prefer native Rust over WASM when available
replicas = 3                   # desired redundancy
```

### Composition

**Implicit (pub/sub):** Wire programs together by matching Zenoh topics. Program A publishes to `shipments/verified`, Program B subscribes to `shipments/verified`. No coordination needed. Fully decoupled.

**Explicit (DAG):** Declare the same topology as a DAG in `Kitri.toml`:

```toml
[dag.supply-chain]
steps = [
    { workflow = "scan-receiver",       output = "shipments/scanned" },
    { workflow = "shipment-verifier",   input = "shipments/scanned", output = "shipments/verified" },
    { workflow = "notification-sender", input = "shipments/verified" },
]
backpressure = "bounded(1000)"
retry_subgraph = true
```

These are two views of the same topology. The DAG declaration lets the runtime optimize scheduling, provide end-to-end tracing, enforce ordering, and retry failed subgraphs intelligently.

## Durability and Failure Model

### Three Layers of Protection

**Layer 1: I/O Boundary Event Sourcing (automatic)**

Every durable I/O primitive is intercepted and logged:

```
WorkflowId = BLAKE3(program_cid || input)

Event Log:
  [0] IoRequested { op: Fetch, cid: abc123 }
  [1] IoResolved  { op: Fetch, cid: abc123, result: Ok(bytes) }
  [2] IoRequested { op: Infer, prompt: "summarize..." }
  [3] IoResolved  { op: Infer, result: Ok("The scan shows...") }
  [4] Checkpoint  { state: <serialized>, content_cid: def456 }
  [5] IoRequested { op: Publish, topic: "shipments/verified" }
  [6] IoCommitted { op: Publish, topic: "shipments/verified" }
```

On crash recovery: load event log, re-execute the program from scratch (or from last checkpoint), replay cached I/O responses. Execution reaches the same point deterministically, then continues forward.

**Layer 2: Explicit Checkpoints (opt-in)**

For compute-heavy workflows where replaying from scratch is expensive:

```rust
let result = heavy_computation(&data);
kitri::checkpoint(&result).await?;
// On crash, replay resumes HERE instead of re-running heavy_computation
```

Checkpoints serialize to content-addressed storage. The checkpoint CID is recorded in the event log.

**Layer 3: Staged Writes (automatic)**

All externally-visible side effects (`publish`, `store`) go to a staging area:

```
                  +-------------+
  kitri::publish  |  Staging    |---- audit -----> Zenoh topic
                  |  Area       |   (trust tier)
  kitri::store    |  (buffered) |---- audit -----> Content store
                  +-------------+
                        |
                   on failure:
                   discard all
```

- **Trusted programs** (valid UCAN chain): capability check only, commit immediately
- **Untrusted programs**: full synchronous content audit before commit
- **On kill/panic**: staged writes are discarded atomically — no partial side effects

### Failure Classification

| Failure Type | Kitri Behavior | Programmer Sees |
|---|---|---|
| Transient (network timeout, node crash) | Automatic retry with backoff | Nothing — transparent recovery |
| Resource exhaustion (fuel budget, memory) | Kill, discard staged writes, retry or escalate | `KitriError::ResourceExhausted` after max retries |
| Logic error (panic, assertion failure) | Kill, discard staged writes, no retry | `KitriError::WorkflowFailed(reason)` |
| Capability denied | Kill, discard staged writes, no retry | `KitriError::Unauthorized` |
| Content tampered (Lyll/Nakaiah) | Quarantine frame, kill process, retry on clean node | `KitriError::IntegrityViolation` |

### Saga Compensation (opt-in)

For multi-step workflows that coordinate across services:

```rust
#[kitri::workflow]
async fn transfer_funds(from: Account, to: Account, amount: u64) -> KitriResult<()> {
    let debit = kitri::query("accounts/debit", &(from, amount)).await?;

    kitri::on_rollback(|| async {
        kitri::query("accounts/credit", &(from, amount)).await
    });

    kitri::query("accounts/credit", &(to, amount)).await?;

    Ok(()) // success — compensators discarded
}
```

Compensators are themselves durable (event-sourced), so they survive crashes.

### Exactly-Once Semantics

`WorkflowId = BLAKE3(program_cid || input)`.

- Same program + same input = same WorkflowId
- Already complete → return cached result
- In progress → attach to existing execution
- No duplicate work, no duplicate side effects

## Security and Trust Model

### Trust Tiers

| Tier | Who | Isolation | I/O Audit |
|---|---|---|---|
| **Owner** | Programs signed by the node's own identity | Ring 2 sandbox + capabilities | Capability check only (fast path) |
| **Delegated** | Valid UCAN chain from a trusted issuer | Ring 2 sandbox + capabilities | Capability check + rate limiting |
| **Untrusted** | Anonymous, expired chain, unknown signer | WASM sandbox + Ring 2 + capabilities | Full synchronous content audit |

The trust tier is evaluated at deploy time by inspecting the UCAN chain. Programs cannot escalate their own tier.

### Capability Declarations

Programs declare required capabilities in `Kitri.toml`. The runtime issues a scoped UCAN token at deployment. Any I/O outside scope returns `KitriError::Unauthorized`.

### Encryption Boundary

All encryption operations are system calls into Ring 2 kernel space. Kitri programs never touch raw key material:

```rust
let sealed = kitri::seal(data, recipient_identity).await?;
let opened = kitri::open(sealed_data).await?;
```

Lyll/Nakaiah verify integrity of encrypted frames. The UCAN chain proves authorization.

### Audit Trail

Every workflow produces an immutable, content-addressed audit trail:

```
WorkflowId: abc789
Program:    shipment-verifier (CID: def123)
Signer:     did:key:z6Mk...
Trust Tier: Owner
Input CID:  aaa111
Event Log CID: bbb222
Output CID: ccc333
Duration:   1.2s
Retries:    0
```

The audit trail is a content-addressed bundle — verifiable, replicable, inspectable by authorized parties. Compliance for free.

## AI Integration

### AI as Consumer — Writing Kitri Programs

Kitri is AI-writable by construction:

- **Rust's compiler** catches memory and type errors
- **Kitri's macros** reduce generated code surface area
- **`Kitri.toml`** is declarative — generatable from natural language descriptions
- **Strong return types** (`KitriResult<Verified>`) act as machine-verifiable specifications

### AI as Participant — Durable Model Invocation

AI model calls are first-class durable I/O:

```rust
let analysis = kitri::infer(InferRequest {
    prompt: "Analyze this scan for anomalies",
    context: &scan_data,
    model: "default",
    max_tokens: 1000,
    deterministic: true,
}).await?;
```

- **Event-sourced:** cached on replay, never re-invokes the model
- **Content-addressed:** identical requests deduplicate across the mesh
- **Capability-gated:** requires `infer = true` in manifest
- **Model-agnostic:** node resolves model name at runtime

### AI as Orchestrator — Dynamic Workflow Assembly

AI agents can browse available Kitri programs via Zenoh, compose them into DAGs, submit and monitor execution:

```rust
#[kitri::workflow]
async fn ai_orchestrator(goal: String) -> KitriResult<Plan> {
    let available = kitri::query("kitri/registry/list", &()).await?;

    let plan = kitri::infer(InferRequest {
        prompt: &format!(
            "Given these available workflows: {available:?}\n\
             Compose a DAG to achieve: {goal}"
        ),
        context: &available,
        ..Default::default()
    }).await?;

    let dag: KitriDag = plan.parse()?;
    let execution = kitri::submit_dag(dag).await?;
    let result = execution.wait().await?;

    Ok(result)
}
```

### The Virtuous Cycle

```
AI writes Kitri code
  -> Rust compiler validates it
    -> Kitri runtime makes it durable
      -> Audit trail proves what happened
        -> AI learns from the audit trail
          -> AI writes better Kitri code
```

## Crate Layout

```
harmony/crates/
  harmony-kitri/         Layer 0: core traits, types, manifest (no_std)
  kitri-macros/          Layer 1: proc-macros
  kitri-sdk/             Layer 1: runtime library (std)

harmony-os/crates/
  harmony-microkernel/
    src/kitri/           Layer 2: OS-level Kitri service (future)

  kitri-wasm/            Layer 3: WASM adapter (future)
```

## Relationship to Existing Crates

| Existing Crate | Kitri's Relationship |
|---|---|
| `harmony-compute` | Kitri's WASM execution backend (Layer 3) |
| `harmony-workflow` | Kitri's durability engine — event sourcing, replay, scheduling |
| `harmony-zenoh` | Kitri's communication plane — pub/sub, queryables, liveliness |
| `harmony-content` | Kitri's storage plane — CIDs, blob store, caching |
| `harmony-identity` | Kitri's trust plane — UCAN capabilities, signing, encryption |
| `harmony-reticulum` | Kitri's network plane — mesh routing, delay-tolerant delivery |

Kitri is the programming model that unifies them. It is not a replacement for any of these crates — it is the developer-facing surface that makes them accessible.

## Open Questions for Future Design Iterations

1. **Checkpoint serialization format** — serde + bincode? Cap'n Proto for zero-copy? Content-addressed Merkle snapshots?
2. **DAG language** — is TOML expressive enough for complex DAGs, or does this need a richer format?
3. **Testing story** — how do you unit-test a Kitri workflow? Mock the I/O primitives? Replay recorded event logs?
4. **Versioning** — when a program CID changes (new version), how do in-flight workflows migrate?
5. **Observability** — beyond audit trails, what does real-time monitoring look like? Zenoh liveliness tokens per workflow?
6. **Resource accounting** — how do fuel budgets translate to native Rust execution? Wall-clock time? Instruction counting via perf counters?
