# harmony-runtime Extraction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extract the sans-I/O `NodeRuntime` state machine from the `harmony-node` binary into a reusable `harmony-runtime` library crate, and define the `PlatformAdapter` trait boundary for multi-platform Harmony network participation.

**Architecture:** Move `runtime.rs` (~6970 lines, 100 tests), `page_index.rs` (260 lines), and inference type definitions from `harmony-node` into a new `harmony-runtime` library crate. Refactor `harmony-node` to import from `harmony-runtime`. Add new trait definitions (`PlatformAdapter`, `AttestationReport`, `ComputeBackend`) via TDD.

**Tech Stack:** Rust, harmony core crate ecosystem (15 crate dependencies), `#[cfg(feature)]` gates for inference support.

**Spec:** `docs/superpowers/specs/2026-03-28-harmony-runtime-multi-platform-design.md`

---

### Task 1: Create harmony-runtime crate scaffold

**Files:**
- Create: `crates/harmony-runtime/Cargo.toml`
- Create: `crates/harmony-runtime/src/lib.rs`
- Modify: `Cargo.toml` (workspace root, add member + dependency)

- [ ] **Step 1: Create the crate directory**

```bash
mkdir -p crates/harmony-runtime/src
```

- [ ] **Step 2: Write Cargo.toml with all required dependencies**

Create `crates/harmony-runtime/Cargo.toml`:

```toml
[package]
name = "harmony-runtime"
description = "Sans-I/O runtime orchestration for Harmony network nodes"
version.workspace = true
edition.workspace = true
license.workspace = true
repository.workspace = true
rust-version.workspace = true

[features]
default = ["std"]
std = [
    "harmony-content/std",
    "harmony-contacts/std",
    "harmony-discovery/std",
    "harmony-memo/std",
    "harmony-peers/std",
    "harmony-crypto/std",
    "harmony-identity/std",
    "harmony-reticulum/std",
]
inference = [
    "harmony-compute/inference",
    "harmony-workflow/inference",
    "dep:harmony-inference",
    "dep:harmony-engram",
    "dep:harmony-agent",
    "dep:candle-core",
]

[dependencies]
harmony-agent = { workspace = true, optional = true }
harmony-athenaeum = { workspace = true, features = ["std"] }
harmony-compute.workspace = true
harmony-contacts = { workspace = true, features = ["std"] }
harmony-content = { workspace = true, features = ["std"] }
harmony-credential.workspace = true
harmony-crypto = { workspace = true, features = ["std"] }
harmony-discovery = { workspace = true, features = ["std"] }
harmony-engram = { workspace = true, optional = true }
harmony-identity = { workspace = true, features = ["std"] }
harmony-inference = { workspace = true, optional = true }
harmony-memo = { workspace = true, features = ["std"] }
harmony-peers = { workspace = true, features = ["std"] }
harmony-reticulum = { workspace = true, features = ["std"] }
harmony-speculative.workspace = true
harmony-workflow.workspace = true
harmony-zenoh.workspace = true
candle-core = { workspace = true, optional = true }
hex.workspace = true
postcard.workspace = true
tracing.workspace = true
```

- [ ] **Step 3: Write minimal lib.rs**

Create `crates/harmony-runtime/src/lib.rs`:

```rust
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Sans-I/O runtime orchestration for Harmony network nodes.
//!
//! This crate provides `NodeRuntime`, the core state machine that wires
//! Tier 1 (Reticulum routing), Tier 2 (content storage), and Tier 3
//! (compute scheduling) into a unified event/action pipeline.
//!
//! Consumers feed events via [`NodeRuntime::push_event`] and process
//! returned actions from [`NodeRuntime::tick`] through platform-specific
//! I/O (UDP sockets, Zenoh sessions, disk, etc.).
```

- [ ] **Step 4: Add harmony-runtime to workspace members and dependencies**

In the workspace root `Cargo.toml`, add `"crates/harmony-runtime"` to the `members` list. Add `harmony-runtime = { path = "crates/harmony-runtime", default-features = false }` to `[workspace.dependencies]`.

- [ ] **Step 5: Verify the empty crate compiles**

Run: `cargo check -p harmony-runtime`
Expected: Success (empty crate, no errors)

- [ ] **Step 6: Commit**

```bash
git add crates/harmony-runtime/ Cargo.toml
git commit -m "feat: scaffold harmony-runtime crate (empty)"
```

---

### Task 2: Move page_index.rs to harmony-runtime

**Files:**
- Move: `crates/harmony-node/src/page_index.rs` → `crates/harmony-runtime/src/page_index.rs`
- Modify: `crates/harmony-runtime/src/lib.rs`

`page_index.rs` (260 lines) is referenced by `runtime.rs` via `crate::page_index::PageIndex`. It depends only on `harmony-athenaeum` and `harmony-content` — no circular dependencies.

- [ ] **Step 1: Copy page_index.rs to harmony-runtime**

Copy `crates/harmony-node/src/page_index.rs` to `crates/harmony-runtime/src/page_index.rs`. Do NOT delete from harmony-node yet.

- [ ] **Step 2: Add module declaration to lib.rs**

Add to `crates/harmony-runtime/src/lib.rs`:

```rust
pub mod page_index;
```

- [ ] **Step 3: Verify harmony-runtime compiles with page_index**

Run: `cargo check -p harmony-runtime`
Expected: Success

- [ ] **Step 4: Commit**

```bash
git add crates/harmony-runtime/src/
git commit -m "feat(runtime): move page_index module to harmony-runtime"
```

---

### Task 3: Move inference types to harmony-runtime

**Files:**
- Create: `crates/harmony-runtime/src/inference_types.rs`
- Modify: `crates/harmony-runtime/src/lib.rs`

`runtime.rs` references ~15 items from `crate::inference::*`: `InferenceInput`, `InferenceRequest`, `TokenInferenceRequest`, `build_capacity_payload`, `DEFAULT_MAX_INFERENCE_TOKENS`, `CAPACITY_READY`, `CAPACITY_BUSY`, `INFERENCE_TAG`, `TOKEN_INFERENCE_TAG`. These are simple data types and parsing functions with no external I/O dependencies.

The `INFERENCE_RUNNER_WASM` constant (which requires `build.rs` + `include_bytes!`) stays in `harmony-node` — it's only used by `compute.rs` tests, not by `runtime.rs`.

- [ ] **Step 1: Read the full inference.rs to identify extractable types**

Read `crates/harmony-node/src/inference.rs` in full. The items referenced by `runtime.rs` that must move are:

**Must move (referenced by runtime.rs):**
- `InferenceInput` (enum: Text/TokenIds)
- `InferenceRequest` (struct + `parse()` method)
- `TokenInferenceRequest` (struct + `parse()` method)
- `InferenceOutput` (if referenced)
- `build_capacity_payload()` (function)
- `DEFAULT_MAX_INFERENCE_TOKENS` (constant)
- `CAPACITY_READY` (constant)
- `INFERENCE_TAG` (constant)
- `TOKEN_INFERENCE_TAG` (constant)

**Must stay in harmony-node (build.rs or inference-feature deps):**
- `INFERENCE_RUNNER_WASM` (requires `include_bytes!` from build.rs)
- `decode_sampling_params()` (returns `harmony_inference::SamplingParams`)
- `build_runner_input()` (depends on harmony-inference types)

- [ ] **Step 2: Create inference_types.rs with the extractable types**

Copy ONLY the items listed above from `crates/harmony-node/src/inference.rs` into `crates/harmony-runtime/src/inference_types.rs`. Preserve the exact type signatures so runtime.rs references compile unchanged after path updates.

- [ ] **Step 3: Add module declaration to lib.rs**

Add to `crates/harmony-runtime/src/lib.rs`:

```rust
pub mod inference_types;
```

- [ ] **Step 4: Verify harmony-runtime compiles**

Run: `cargo check -p harmony-runtime`
Expected: Success

- [ ] **Step 5: Commit**

```bash
git add crates/harmony-runtime/src/
git commit -m "feat(runtime): add inference type definitions for runtime extraction"
```

---

### Task 4: Move runtime.rs to harmony-runtime

**Files:**
- Move: `crates/harmony-node/src/runtime.rs` → `crates/harmony-runtime/src/runtime.rs`
- Modify: `crates/harmony-runtime/src/lib.rs`

This is the core extraction — ~6970 lines with 100 tests. The file's external crate imports are unchanged. Internal references need updating:
- `crate::page_index::PageIndex` → unchanged (page_index is now in harmony-runtime)
- `crate::inference::*` → `crate::inference_types::*`
- `crate::compute::tests::ADD_WAT` / `FETCH_WAT` → inline in test module

- [ ] **Step 1: Copy runtime.rs to harmony-runtime**

Copy `crates/harmony-node/src/runtime.rs` to `crates/harmony-runtime/src/runtime.rs`.

- [ ] **Step 2: Update internal crate references**

In `crates/harmony-runtime/src/runtime.rs`, replace all occurrences of `crate::inference::` with `crate::inference_types::`. There are approximately 15 references (grep showed them all). The `crate::page_index::` references remain unchanged since `page_index` is already in `harmony-runtime`.

- [ ] **Step 3: Handle test-only WAT constants**

In the `#[cfg(test)]` module in `runtime.rs` (around line 3900), there are references to `crate::compute::tests::ADD_WAT` and `crate::compute::tests::FETCH_WAT`. These are WAT source strings used by ~4 tests. Define them directly in the test module:

```rust
#[cfg(test)]
mod tests {
    // WAT source for test compute modules (originally in compute.rs)
    const ADD_WAT: &str = "(module ...)"; // Copy exact content from compute.rs
    const FETCH_WAT: &str = "(module ...)"; // Copy exact content from compute.rs
    // ... rest of tests, replacing crate::compute::tests::ADD_WAT with ADD_WAT
```

Read `crates/harmony-node/src/compute.rs` test module to find the exact WAT strings.

- [ ] **Step 4: Add module declaration and re-exports to lib.rs**

Update `crates/harmony-runtime/src/lib.rs`:

```rust
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Sans-I/O runtime orchestration for Harmony network nodes.

pub mod inference_types;
pub mod page_index;
pub mod runtime;

// Re-export primary types for ergonomic imports
pub use runtime::{NodeConfig, NodeRuntime, RuntimeAction, RuntimeEvent};
pub use runtime::{AdaptiveCompute, TierSchedule};
```

- [ ] **Step 5: Verify harmony-runtime compiles**

Run: `cargo check -p harmony-runtime`
Expected: Success (may need to fix import paths iteratively)

- [ ] **Step 6: Run harmony-runtime tests**

Run: `cargo test -p harmony-runtime`
Expected: All 100 runtime tests pass

- [ ] **Step 7: Commit**

```bash
git add crates/harmony-runtime/src/
git commit -m "feat(runtime): move NodeRuntime state machine to harmony-runtime

Moves runtime.rs (~6970 lines, 100 tests), page_index.rs, and
inference type definitions from harmony-node into the new
harmony-runtime library crate. All tests pass."
```

---

### Task 5: Refactor harmony-node to import from harmony-runtime

**Files:**
- Modify: `crates/harmony-node/Cargo.toml`
- Modify: `crates/harmony-node/src/main.rs`
- Modify: `crates/harmony-node/src/event_loop.rs`
- Modify: `crates/harmony-node/src/inference.rs`
- Delete: `crates/harmony-node/src/runtime.rs`
- Delete: `crates/harmony-node/src/page_index.rs`

- [ ] **Step 1: Add harmony-runtime dependency to harmony-node**

In `crates/harmony-node/Cargo.toml`, add:

```toml
harmony-runtime.workspace = true
```

Also add the inference feature forwarding:

```toml
[features]
inference = ["harmony-compute/inference", "harmony-workflow/inference", "dep:harmony-inference", "dep:candle-core", "dep:harmony-agent", "harmony-runtime/inference"]
```

- [ ] **Step 2: Replace runtime module with re-export**

Replace the contents of `crates/harmony-node/src/runtime.rs` with a re-export shim (to minimize changes to other files):

```rust
// Re-export from harmony-runtime for backwards compatibility within this crate.
pub use harmony_runtime::runtime::*;
```

Or alternatively, delete `runtime.rs` entirely and update `main.rs` to change `mod runtime;` to an `extern crate` / `use` import. Choose whichever approach minimizes the diff.

- [ ] **Step 3: Update main.rs imports**

In `crates/harmony-node/src/main.rs`, the `mod runtime;` declaration either stays (if using shim approach) or changes to:

```rust
use harmony_runtime::{NodeConfig, NodeRuntime, RuntimeAction, RuntimeEvent};
use harmony_runtime::runtime::TierSchedule; // if needed
```

Update the `NodeRuntime::new(config, MemoryBookStore::new())` call — it should compile unchanged since the API is identical.

- [ ] **Step 4: Update event_loop.rs imports**

In `crates/harmony-node/src/event_loop.rs`, update:

```rust
// Before:
use crate::runtime::{NodeRuntime, RuntimeAction, RuntimeEvent};

// After (if shim):
// unchanged

// After (if direct):
use harmony_runtime::{NodeRuntime, RuntimeAction, RuntimeEvent};
```

- [ ] **Step 5: Update inference.rs to export types from harmony-runtime**

In `crates/harmony-node/src/inference.rs`, re-export the types that were moved:

```rust
// Types moved to harmony-runtime; re-export for crate-internal compat
pub use harmony_runtime::inference_types::{
    InferenceInput, InferenceRequest, TokenInferenceRequest,
    build_capacity_payload, DEFAULT_MAX_INFERENCE_TOKENS,
    CAPACITY_READY, CAPACITY_BUSY, INFERENCE_TAG, TOKEN_INFERENCE_TAG,
};

// INFERENCE_RUNNER_WASM stays here (requires build.rs)
pub const INFERENCE_RUNNER_WASM: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/inference_runner.wasm"));
```

- [ ] **Step 6: Delete page_index.rs from harmony-node**

Delete `crates/harmony-node/src/page_index.rs`. Update `main.rs` to remove `mod page_index;` declaration if present. The runtime now uses `harmony_runtime::page_index::PageIndex` internally.

- [ ] **Step 7: Verify harmony-node compiles**

Run: `cargo check -p harmony-node`
Expected: Success (fix any remaining import issues iteratively)

- [ ] **Step 8: Run all harmony-node tests**

Run: `cargo test -p harmony-node`
Expected: All tests pass (config: 18, main: 32, integration: 15 = 65 tests in harmony-node. The 100 runtime tests now live in harmony-runtime.)

- [ ] **Step 9: Run full workspace tests**

Run: `cargo test --workspace`
Expected: All tests pass across all crates

- [ ] **Step 10: Commit**

```bash
git add crates/harmony-node/ crates/harmony-runtime/
git commit -m "refactor(node): import NodeRuntime from harmony-runtime

harmony-node now depends on harmony-runtime for the core state
machine. All 100 runtime tests live in harmony-runtime; harmony-node
retains its 65 CLI/config/integration tests."
```

---

### Task 6: Add attestation types (TDD)

**Files:**
- Create: `crates/harmony-runtime/src/attestation.rs`
- Modify: `crates/harmony-runtime/src/lib.rs`

- [ ] **Step 1: Write failing tests for attestation types**

Create `crates/harmony-runtime/src/attestation.rs`:

```rust
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Trust tier model for multi-platform attestation.

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn attestation_tier_variants_exist() {
        let _ = AttestationTier::Sovereign;
        let _ = AttestationTier::HardwareBound;
        let _ = AttestationTier::Unattested;
    }

    #[test]
    fn attestation_report_without_evidence() {
        let report = AttestationReport {
            tier: AttestationTier::Unattested,
            evidence: None,
        };
        assert!(report.evidence.is_none());
    }

    #[test]
    fn attestation_report_with_evidence() {
        let evidence = vec![0x01, 0x02, 0x03];
        let report = AttestationReport {
            tier: AttestationTier::Sovereign,
            evidence: Some(evidence.clone()),
        };
        assert_eq!(report.evidence.unwrap(), evidence);
    }

    #[test]
    fn attestation_tier_debug_impl() {
        // Verify Debug is derived
        let tier = AttestationTier::HardwareBound;
        let _ = format!("{:?}", tier);
    }
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test -p harmony-runtime attestation`
Expected: FAIL — types not defined

- [ ] **Step 3: Implement attestation types**

Add above the test module:

```rust
/// Report of a node's platform attestation status.
#[derive(Debug, Clone)]
pub struct AttestationReport {
    /// Trust tier based on key binding and platform sovereignty.
    pub tier: AttestationTier,
    /// Optional platform-specific attestation evidence (TPM quote,
    /// Secure Enclave attestation, etc.). Opaque bytes.
    pub evidence: Option<Vec<u8>>,
}

/// Trust tier reflecting how strongly a node's keys are bound to hardware.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttestationTier {
    /// Hardware-rooted keys on a sovereign OS (e.g., harmony-os + TPM).
    Sovereign,
    /// Hardware-rooted keys but vendor-controlled OS (e.g., Windows TPM).
    HardwareBound,
    /// No hardware attestation. Trust via behavioral scoring only.
    Unattested,
}
```

- [ ] **Step 4: Add module declaration to lib.rs**

Add to `crates/harmony-runtime/src/lib.rs`:

```rust
pub mod attestation;
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cargo test -p harmony-runtime attestation`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add crates/harmony-runtime/src/attestation.rs crates/harmony-runtime/src/lib.rs
git commit -m "feat(runtime): add AttestationReport and AttestationTier types"
```

---

### Task 7: Add compute backend types (TDD)

**Files:**
- Create: `crates/harmony-runtime/src/compute_backend.rs`
- Modify: `crates/harmony-runtime/src/lib.rs`

- [ ] **Step 1: Write failing tests for compute backend types**

Create `crates/harmony-runtime/src/compute_backend.rs`:

```rust
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Pluggable compute backend traits and descriptors.

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compute_backend_kind_variants_exist() {
        let _ = ComputeBackendKind::HttpInference {
            endpoint: "http://localhost:1234".to_string(),
        };
        let _ = ComputeBackendKind::DirectGpu {
            device_id: "gpu0".to_string(),
        };
        let _ = ComputeBackendKind::Cpu;
    }

    #[test]
    fn compute_capability_variants_exist() {
        let _ = ComputeCapability::Inference {
            model_id: "llama-3.1".to_string(),
            context_length: 8192,
        };
        let _ = ComputeCapability::WasmExecution { fuel_budget: 100_000 };
        let _ = ComputeCapability::TensorCompute { flops_estimate: 1_000_000 };
    }

    #[test]
    fn compute_backend_descriptor_construction() {
        let desc = ComputeBackendDescriptor {
            id: "lmstudio-0".to_string(),
            name: "LM Studio".to_string(),
            kind: ComputeBackendKind::HttpInference {
                endpoint: "http://localhost:1234".to_string(),
            },
            capabilities: vec![ComputeCapability::Inference {
                model_id: "qwen-2.5".to_string(),
                context_length: 32768,
            }],
        };
        assert_eq!(desc.id, "lmstudio-0");
        assert_eq!(desc.capabilities.len(), 1);
    }

    /// Verify a mock backend can implement the trait.
    struct MockBackend;

    impl ComputeBackend for MockBackend {
        type Error = String;

        fn run_inference(
            &mut self,
            _model_id: &str,
            _input: &[u8],
            _params: &[u8],
        ) -> Result<Vec<u8>, Self::Error> {
            Ok(vec![0x42])
        }
    }

    #[test]
    fn mock_backend_implements_trait() {
        let mut backend = MockBackend;
        let result = backend.run_inference("test", b"hello", b"").unwrap();
        assert_eq!(result, vec![0x42]);
    }
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test -p harmony-runtime compute_backend`
Expected: FAIL

- [ ] **Step 3: Implement compute backend types**

Add above the test module:

```rust
/// Descriptor for an available compute backend.
#[derive(Debug, Clone)]
pub struct ComputeBackendDescriptor {
    /// Unique identifier for this backend instance.
    pub id: String,
    /// Human-readable name.
    pub name: String,
    /// What kind of backend this is.
    pub kind: ComputeBackendKind,
    /// Capabilities this backend advertises.
    pub capabilities: Vec<ComputeCapability>,
}

/// Classification of compute backend type.
#[derive(Debug, Clone)]
pub enum ComputeBackendKind {
    /// HTTP inference server (LM Studio, ollama, vLLM, etc.).
    HttpInference { endpoint: String },
    /// Direct GPU access (future: CUDA, Metal, Vulkan compute).
    DirectGpu { device_id: String },
    /// CPU-only compute.
    Cpu,
}

/// A specific compute capability offered by a backend.
#[derive(Debug, Clone)]
pub enum ComputeCapability {
    /// LLM inference with a specific model.
    Inference { model_id: String, context_length: u32 },
    /// WASM module execution.
    WasmExecution { fuel_budget: u64 },
    /// Raw tensor compute (future).
    TensorCompute { flops_estimate: u64 },
}

/// Execute compute work on a specific backend.
///
/// Implementations are platform-specific. The runtime never calls
/// this directly — it emits `RuntimeAction::RunInference` and the
/// platform's event loop dispatches to the appropriate backend.
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

- [ ] **Step 4: Add module declaration to lib.rs**

Add to `crates/harmony-runtime/src/lib.rs`:

```rust
pub mod compute_backend;
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cargo test -p harmony-runtime compute_backend`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add crates/harmony-runtime/src/compute_backend.rs crates/harmony-runtime/src/lib.rs
git commit -m "feat(runtime): add ComputeBackend trait and descriptor types"
```

---

### Task 8: Add PlatformAdapter trait (TDD)

**Files:**
- Create: `crates/harmony-runtime/src/adapter.rs`
- Modify: `crates/harmony-runtime/src/lib.rs`

Now that attestation (Task 6) and compute_backend (Task 7) exist, the PlatformAdapter trait can reference them without circular dependency issues.

- [ ] **Step 1: Write failing tests for PlatformAdapter types**

Create `crates/harmony-runtime/src/adapter.rs` with tests first:

```rust
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Platform adapter traits for multi-platform Harmony participation.

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn interface_descriptor_construction() {
        let desc = InterfaceDescriptor {
            name: "udp0".to_string(),
            kind: InterfaceKind::Udp,
            mtu: 500,
            bandwidth_estimate: 1_000_000,
        };
        assert_eq!(desc.name, "udp0");
        assert_eq!(desc.mtu, 500);
    }

    #[test]
    fn interface_kind_variants_exist() {
        let _ = InterfaceKind::Udp;
        let _ = InterfaceKind::Tcp;
        let _ = InterfaceKind::Serial;
        let _ = InterfaceKind::LoRa;
        let _ = InterfaceKind::ZenohTunnel;
        let _ = InterfaceKind::IrohQuic;
        let _ = InterfaceKind::Other("custom".to_string());
    }

    #[test]
    fn platform_info_construction() {
        let info = PlatformInfo {
            hostname: "test-node".to_string(),
            interfaces: vec![],
        };
        assert_eq!(info.hostname, "test-node");
        assert!(info.interfaces.is_empty());
    }

    /// Verify a mock adapter can implement the full trait.
    struct MockAdapter;

    impl PlatformAdapter for MockAdapter {
        type Error = String;

        fn init(&mut self) -> Result<PlatformInfo, Self::Error> {
            Ok(PlatformInfo {
                hostname: "mock".to_string(),
                interfaces: vec![],
            })
        }

        fn shutdown(&mut self) -> Result<(), Self::Error> {
            Ok(())
        }

        fn available_interfaces(&self) -> Vec<InterfaceDescriptor> {
            vec![]
        }

        fn send(&mut self, _interface: &str, _data: &[u8]) -> Result<(), Self::Error> {
            Ok(())
        }

        fn receive(&mut self) -> Option<(String, Vec<u8>)> {
            None
        }

        fn attestation(&self) -> crate::attestation::AttestationReport {
            crate::attestation::AttestationReport {
                tier: crate::attestation::AttestationTier::Unattested,
                evidence: None,
            }
        }

        fn compute_backends(&self) -> Vec<crate::compute_backend::ComputeBackendDescriptor> {
            vec![]
        }
    }

    #[test]
    fn mock_adapter_implements_trait() {
        let mut adapter = MockAdapter;
        let info = adapter.init().unwrap();
        assert_eq!(info.hostname, "mock");
        assert!(adapter.receive().is_none());
        adapter.shutdown().unwrap();
    }
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test -p harmony-runtime adapter`
Expected: FAIL — types not defined yet

- [ ] **Step 3: Implement PlatformAdapter trait and types**

Add the type definitions above the test module in `adapter.rs`:

```rust
/// Metadata returned by platform initialization.
#[derive(Debug, Clone)]
pub struct PlatformInfo {
    /// Platform hostname or node name.
    pub hostname: String,
    /// Interfaces discovered at init time.
    pub interfaces: Vec<InterfaceDescriptor>,
}

/// Descriptor for a network transport available on the platform.
#[derive(Debug, Clone)]
pub struct InterfaceDescriptor {
    /// Interface name (matches Node's interface_name convention).
    pub name: String,
    /// Transport type.
    pub kind: InterfaceKind,
    /// Maximum transmission unit in bytes.
    pub mtu: usize,
    /// Estimated bandwidth in bytes/sec. 0 = unknown.
    pub bandwidth_estimate: u64,
}

/// Network transport type.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InterfaceKind {
    Udp,
    Tcp,
    Serial,
    LoRa,
    ZenohTunnel,
    IrohQuic,
    Other(String),
}

/// What a platform must provide to host a Harmony runtime.
///
/// Extends the foundation traits (EntropySource, PersistentState)
/// with lifecycle, networking, attestation, and compute discovery.
pub trait PlatformAdapter {
    type Error: core::fmt::Debug;

    // --- Lifecycle ---
    fn init(&mut self) -> Result<PlatformInfo, Self::Error>;
    fn shutdown(&mut self) -> Result<(), Self::Error>;

    // --- Networking ---
    fn available_interfaces(&self) -> Vec<InterfaceDescriptor>;
    fn send(&mut self, interface: &str, data: &[u8]) -> Result<(), Self::Error>;
    fn receive(&mut self) -> Option<(String, Vec<u8>)>;

    // --- Attestation ---
    fn attestation(&self) -> crate::attestation::AttestationReport;

    // --- Compute ---
    fn compute_backends(&self) -> Vec<crate::compute_backend::ComputeBackendDescriptor>;
}
```

- [ ] **Step 4: Add module declaration to lib.rs**

Add to `crates/harmony-runtime/src/lib.rs`:

```rust
pub mod adapter;
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cargo test -p harmony-runtime adapter`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add crates/harmony-runtime/src/adapter.rs crates/harmony-runtime/src/lib.rs
git commit -m "feat(runtime): add PlatformAdapter trait and interface types"
```

---

### Task 9: Final verification

- [ ] **Step 1: Run all harmony-runtime tests**

Run: `cargo test -p harmony-runtime`
Expected: All tests pass (100 runtime + adapter + attestation + compute_backend tests)

- [ ] **Step 2: Run all harmony-node tests**

Run: `cargo test -p harmony-node`
Expected: All tests pass (config, CLI, integration)

- [ ] **Step 3: Run full workspace tests**

Run: `cargo test --workspace`
Expected: All tests pass

- [ ] **Step 4: Run clippy**

Run: `cargo clippy --workspace --all-targets -- -D warnings`
Expected: No warnings

- [ ] **Step 5: Run nightly rustfmt**

Run: `rustup run nightly cargo fmt --all -- --check`
Expected: No formatting issues

- [ ] **Step 6: Fix any issues found in steps 3-5**

Address clippy warnings, formatting issues, or test failures iteratively.

- [ ] **Step 7: Final commit (if any fixes needed)**

```bash
git add -A
git commit -m "chore: fix clippy/fmt issues from harmony-runtime extraction"
```

- [ ] **Step 8: Verify git status is clean**

Run: `git status`
Expected: Clean working tree, all changes committed
