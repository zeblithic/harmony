# Kitri Layer 0 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build `harmony-kitri`, the sans-I/O core crate that defines Kitri's type system, I/O primitives, event log, manifest, trust model, failure handling, and engine trait.

**Architecture:** Layer 0 follows the same pattern as every other Harmony Ring 0 crate: no_std-compatible, sans-I/O, pure types and traits. It extends `harmony-workflow`'s event-sourcing model with richer I/O operations (Query, Publish, Infer, Checkpoint, Seal, Open), adds manifest parsing (Kitri.toml), trust tiering (Owner/Delegated/Untrusted), staged writes, failure classification, DAG composition, and saga compensation. The `KitriEngine` is a sans-I/O state machine that wraps `WorkflowEngine` with Kitri-specific semantics.

**Tech Stack:** Rust (no_std + alloc), serde (serialization), toml (manifest parsing, std-only feature), harmony-workflow, harmony-content, harmony-identity.

**Design doc:** `docs/plans/2026-03-08-kitri-design.md`

**Repo:** `zeblithic/harmony` (Ring 0 core)

---

### Task 1: Create `harmony-kitri` crate skeleton

**Files:**
- Create: `crates/harmony-kitri/Cargo.toml`
- Create: `crates/harmony-kitri/src/lib.rs`
- Modify: `Cargo.toml` (workspace root — add member + workspace dep)

**Step 1: Add crate to workspace**

Add `"crates/harmony-kitri"` to the `members` array in the root `Cargo.toml`. Add `harmony-kitri` to `[workspace.dependencies]`:

```toml
harmony-kitri = { path = "crates/harmony-kitri" }
```

**Step 2: Create `Cargo.toml`**

```toml
[package]
name = "harmony-kitri"
version.workspace = true
edition.workspace = true
license.workspace = true
repository.workspace = true
rust-version.workspace = true
description = "Kitri — Harmony's native programming model for durable distributed computation (Layer 0: types & traits)"

[features]
default = ["std"]
std = [
    "harmony-workflow/std",
    "harmony-content/std",
    "harmony-identity/std",
    "serde/std",
    "toml",
]

[dependencies]
harmony-workflow = { workspace = true, default-features = false }
harmony-content = { workspace = true, default-features = false }
harmony-identity = { workspace = true, default-features = false }
serde = { workspace = true, default-features = false, features = ["derive", "alloc"] }
toml = { workspace = true, optional = true }

[dev-dependencies]
```

Check workspace `Cargo.toml` for which deps already exist at workspace level. `serde` and `toml` likely need adding if not present. Look at existing crates (e.g., `harmony-workflow/Cargo.toml`) for the exact pattern.

**Step 3: Create `lib.rs`**

```rust
// SPDX-License-Identifier: Apache-2.0 OR MIT
//! Kitri — Harmony's native programming model for durable distributed computation.
//!
//! Layer 0: sans-I/O core types, traits, and manifest parsing.
//! This crate defines the *protocol* of Kitri — no runtime, no I/O.

#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

pub mod error;
pub mod io;
pub mod event;
pub mod checkpoint;
pub mod manifest;
pub mod trust;
pub mod retry;
pub mod program;
pub mod dag;
pub mod staging;
pub mod compensation;

pub use error::{KitriError, KitriResult};
```

**Step 4: Create stub modules**

Create each module file (`error.rs`, `io.rs`, `event.rs`, `checkpoint.rs`, `manifest.rs`, `trust.rs`, `retry.rs`, `program.rs`, `dag.rs`, `staging.rs`, `compensation.rs`) with just a comment header:

```rust
// SPDX-License-Identifier: Apache-2.0 OR MIT
//! <module description>
```

**Step 5: Verify it compiles**

Run: `cargo check -p harmony-kitri`
Expected: compiles with no errors (empty modules are valid)

**Step 6: Commit**

```bash
git add crates/harmony-kitri/ Cargo.toml
git commit -m "feat(kitri): add harmony-kitri crate skeleton — Layer 0 types & traits"
```

---

### Task 2: Error types

**Files:**
- Modify: `crates/harmony-kitri/src/error.rs`
- Test: inline `#[cfg(test)]` module

**Step 1: Write the failing test**

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kitri_error_variants_exist() {
        let e1 = KitriError::WorkflowFailed { reason: "boom".into() };
        assert!(matches!(e1, KitriError::WorkflowFailed { .. }));

        let e2 = KitriError::Unauthorized { resource: "topic/foo".into() };
        assert!(matches!(e2, KitriError::Unauthorized { .. }));

        let e3 = KitriError::ResourceExhausted { detail: "fuel".into() };
        assert!(matches!(e3, KitriError::ResourceExhausted { .. }));

        let e4 = KitriError::IntegrityViolation;
        assert!(matches!(e4, KitriError::IntegrityViolation));

        let e5 = KitriError::ContentNotFound { cid: [0xAA; 32] };
        assert!(matches!(e5, KitriError::ContentNotFound { .. }));

        let e6 = KitriError::InferFailed { reason: "model unavailable".into() };
        assert!(matches!(e6, KitriError::InferFailed { .. }));
    }

    #[test]
    fn kitri_result_alias_works() {
        let ok: KitriResult<u32> = Ok(42);
        assert_eq!(ok.unwrap(), 42);

        let err: KitriResult<u32> = Err(KitriError::IntegrityViolation);
        assert!(err.is_err());
    }
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p harmony-kitri error::tests -v`
Expected: FAIL — types not defined

**Step 3: Write implementation**

```rust
// SPDX-License-Identifier: Apache-2.0 OR MIT
//! Error types for Kitri workflows.

use alloc::string::String;
use core::fmt;

/// Errors surfaced to Kitri workflow authors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KitriError {
    /// Workflow logic failed (panic, assertion, unrecoverable).
    WorkflowFailed { reason: String },
    /// I/O operation denied — capability not granted.
    Unauthorized { resource: String },
    /// Resource budget exceeded after max retries.
    ResourceExhausted { detail: String },
    /// Lyll/Nakaiah detected memory corruption.
    IntegrityViolation,
    /// Content not found in the mesh.
    ContentNotFound { cid: [u8; 32] },
    /// AI model invocation failed.
    InferFailed { reason: String },
    /// Checkpoint serialization/deserialization failed.
    CheckpointFailed { reason: String },
    /// Manifest parsing error.
    ManifestInvalid { reason: String },
    /// DAG validation error (cycles, missing steps).
    DagInvalid { reason: String },
    /// Compensation (saga rollback) failed.
    CompensationFailed { reason: String },
}

impl fmt::Display for KitriError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::WorkflowFailed { reason } => write!(f, "workflow failed: {reason}"),
            Self::Unauthorized { resource } => write!(f, "unauthorized: {resource}"),
            Self::ResourceExhausted { detail } => write!(f, "resource exhausted: {detail}"),
            Self::IntegrityViolation => write!(f, "integrity violation detected"),
            Self::ContentNotFound { cid } => write!(f, "content not found: {cid:02x?}"),
            Self::InferFailed { reason } => write!(f, "inference failed: {reason}"),
            Self::CheckpointFailed { reason } => write!(f, "checkpoint failed: {reason}"),
            Self::ManifestInvalid { reason } => write!(f, "invalid manifest: {reason}"),
            Self::DagInvalid { reason } => write!(f, "invalid DAG: {reason}"),
            Self::CompensationFailed { reason } => write!(f, "compensation failed: {reason}"),
        }
    }
}

/// Convenience alias for Kitri workflow results.
pub type KitriResult<T> = Result<T, KitriError>;
```

**Step 4: Run tests**

Run: `cargo test -p harmony-kitri error::tests -v`
Expected: PASS

**Step 5: Commit**

```bash
git add crates/harmony-kitri/src/error.rs
git commit -m "feat(kitri): add KitriError and KitriResult types"
```

---

### Task 3: I/O operation types

**Files:**
- Modify: `crates/harmony-kitri/src/io.rs`
- Test: inline `#[cfg(test)]` module

These are the durable I/O primitives — the operations that get event-sourced.

**Step 1: Write the failing test**

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn io_op_variants() {
        let fetch = KitriIoOp::Fetch { cid: [0xAA; 32] };
        assert!(matches!(fetch, KitriIoOp::Fetch { .. }));

        let store = KitriIoOp::Store { data: vec![1, 2, 3] };
        assert!(matches!(store, KitriIoOp::Store { .. }));

        let query = KitriIoOp::Query {
            topic: "blockchain/verify".into(),
            payload: vec![42],
        };
        assert!(matches!(query, KitriIoOp::Query { .. }));

        let publish = KitriIoOp::Publish {
            topic: "shipments/verified".into(),
            payload: vec![1],
        };
        assert!(matches!(publish, KitriIoOp::Publish { .. }));

        let infer = KitriIoOp::Infer {
            prompt: "summarize".into(),
            context: vec![],
            model: None,
            max_tokens: Some(1000),
        };
        assert!(matches!(infer, KitriIoOp::Infer { .. }));

        let seal = KitriIoOp::Seal {
            data: vec![1],
            recipient: [0xBB; 16],
        };
        assert!(matches!(seal, KitriIoOp::Seal { .. }));

        let open = KitriIoOp::Open { sealed: vec![1, 2] };
        assert!(matches!(open, KitriIoOp::Open { .. }));

        let spawn = KitriIoOp::Spawn {
            program_cid: [0xCC; 32],
            input: vec![1],
        };
        assert!(matches!(spawn, KitriIoOp::Spawn { .. }));
    }

    #[test]
    fn io_result_variants() {
        let fetched = KitriIoResult::Fetched { data: vec![1] };
        assert!(matches!(fetched, KitriIoResult::Fetched { .. }));

        let stored = KitriIoResult::Stored { cid: [0xAA; 32] };
        assert!(matches!(stored, KitriIoResult::Stored { .. }));

        let replied = KitriIoResult::QueryReply { payload: vec![42] };
        assert!(matches!(replied, KitriIoResult::QueryReply { .. }));

        let published = KitriIoResult::Published;
        assert!(matches!(published, KitriIoResult::Published));

        let inferred = KitriIoResult::Inferred { output: "result".into() };
        assert!(matches!(inferred, KitriIoResult::Inferred { .. }));

        let sealed = KitriIoResult::Sealed { ciphertext: vec![1] };
        assert!(matches!(sealed, KitriIoResult::Sealed { .. }));

        let opened = KitriIoResult::Opened { plaintext: vec![1] };
        assert!(matches!(opened, KitriIoResult::Opened { .. }));

        let spawned = KitriIoResult::Spawned { workflow_id: [0xDD; 32] };
        assert!(matches!(spawned, KitriIoResult::Spawned { .. }));

        let failed = KitriIoResult::Failed { error: "timeout".into() };
        assert!(matches!(failed, KitriIoResult::Failed { .. }));
    }

    #[test]
    fn io_op_is_side_effect() {
        // Publish, Store, Seal, Spawn are side effects (need staging).
        assert!(KitriIoOp::Publish { topic: "t".into(), payload: vec![] }.is_side_effect());
        assert!(KitriIoOp::Store { data: vec![] }.is_side_effect());
        assert!(KitriIoOp::Seal { data: vec![], recipient: [0; 16] }.is_side_effect());
        assert!(KitriIoOp::Spawn { program_cid: [0; 32], input: vec![] }.is_side_effect());

        // Fetch, Query, Infer, Open are reads (no staging needed).
        assert!(!KitriIoOp::Fetch { cid: [0; 32] }.is_side_effect());
        assert!(!KitriIoOp::Query { topic: "t".into(), payload: vec![] }.is_side_effect());
        assert!(!KitriIoOp::Infer { prompt: "p".into(), context: vec![], model: None, max_tokens: None }.is_side_effect());
        assert!(!KitriIoOp::Open { sealed: vec![] }.is_side_effect());
    }
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p harmony-kitri io::tests -v`
Expected: FAIL — types not defined

**Step 3: Write implementation**

```rust
// SPDX-License-Identifier: Apache-2.0 OR MIT
//! Durable I/O operation types — the primitives that get event-sourced.

use alloc::string::String;
use alloc::vec::Vec;

/// A durable I/O operation requested by a Kitri workflow.
///
/// Each variant corresponds to one of the `kitri::*` SDK primitives.
/// The runtime intercepts these, logs them in the event log, and
/// replays cached responses on crash recovery.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KitriIoOp {
    /// Retrieve content by CID from the mesh.
    Fetch { cid: [u8; 32] },
    /// Store content, returning its CID.
    Store { data: Vec<u8> },
    /// Invoke a Zenoh queryable.
    Query { topic: String, payload: Vec<u8> },
    /// Publish to a Zenoh topic (staged until audit passes).
    Publish { topic: String, payload: Vec<u8> },
    /// Invoke an AI model.
    Infer {
        prompt: String,
        context: Vec<u8>,
        model: Option<String>,
        max_tokens: Option<u32>,
    },
    /// Encrypt data via kernel (no key exposure to the workflow).
    Seal { data: Vec<u8>, recipient: [u8; 16] },
    /// Decrypt sealed data via kernel.
    Open { sealed: Vec<u8> },
    /// Spawn a child workflow by program CID.
    Spawn { program_cid: [u8; 32], input: Vec<u8> },
}

impl KitriIoOp {
    /// Returns `true` if this operation has externally-visible side effects
    /// and must be staged before commit.
    pub fn is_side_effect(&self) -> bool {
        matches!(
            self,
            Self::Publish { .. } | Self::Store { .. } | Self::Seal { .. } | Self::Spawn { .. }
        )
    }
}

/// The result of a durable I/O operation, returned by the runtime.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KitriIoResult {
    Fetched { data: Vec<u8> },
    Stored { cid: [u8; 32] },
    QueryReply { payload: Vec<u8> },
    Published,
    Inferred { output: String },
    Sealed { ciphertext: Vec<u8> },
    Opened { plaintext: Vec<u8> },
    Spawned { workflow_id: [u8; 32] },
    Failed { error: String },
}
```

**Step 4: Run tests**

Run: `cargo test -p harmony-kitri io::tests -v`
Expected: PASS

**Step 5: Commit**

```bash
git add crates/harmony-kitri/src/io.rs
git commit -m "feat(kitri): add durable I/O operation types (KitriIoOp, KitriIoResult)"
```

---

### Task 4: Event log and checkpoint types

**Files:**
- Modify: `crates/harmony-kitri/src/event.rs`
- Modify: `crates/harmony-kitri/src/checkpoint.rs`
- Test: inline `#[cfg(test)]` modules

These extend `harmony-workflow`'s `HistoryEvent` with Kitri's richer I/O model.

**Step 1: Write failing tests for event.rs**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::{KitriIoOp, KitriIoResult};

    #[test]
    fn kitri_event_io_roundtrip() {
        let op = KitriIoOp::Fetch { cid: [0xAA; 32] };
        let result = KitriIoResult::Fetched { data: vec![1, 2, 3] };

        let requested = KitriEvent::IoRequested { seq: 0, op: op.clone() };
        let resolved = KitriEvent::IoResolved { seq: 0, result: result.clone() };

        assert!(matches!(requested, KitriEvent::IoRequested { seq: 0, .. }));
        assert!(matches!(resolved, KitriEvent::IoResolved { seq: 0, .. }));
    }

    #[test]
    fn kitri_event_checkpoint() {
        let event = KitriEvent::CheckpointSaved {
            seq: 5,
            state_cid: [0xBB; 32],
        };
        assert!(matches!(event, KitriEvent::CheckpointSaved { seq: 5, .. }));
    }

    #[test]
    fn event_log_append_and_len() {
        let mut log = KitriEventLog::new([0xCC; 32]);
        assert_eq!(log.len(), 0);
        assert!(log.is_empty());

        log.append(KitriEvent::IoRequested {
            seq: 0,
            op: KitriIoOp::Fetch { cid: [0; 32] },
        });
        assert_eq!(log.len(), 1);
        assert!(!log.is_empty());
    }

    #[test]
    fn event_log_workflow_id() {
        let id = [0xDD; 32];
        let log = KitriEventLog::new(id);
        assert_eq!(log.workflow_id(), &id);
    }

    #[test]
    fn event_log_replay_cache() {
        let mut log = KitriEventLog::new([0; 32]);
        let op = KitriIoOp::Fetch { cid: [0xAA; 32] };
        let result = KitriIoResult::Fetched { data: vec![42] };

        log.append(KitriEvent::IoRequested { seq: 0, op });
        log.append(KitriEvent::IoResolved { seq: 0, result: result.clone() });

        // Replay should find the cached result for seq 0.
        assert_eq!(log.cached_result(0), Some(&result));
        assert_eq!(log.cached_result(1), None);
    }
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p harmony-kitri event::tests -v`
Expected: FAIL

**Step 3: Write event.rs implementation**

```rust
// SPDX-License-Identifier: Apache-2.0 OR MIT
//! Event log — the durable record of a Kitri workflow's I/O history.

use alloc::vec::Vec;

use crate::io::{KitriIoOp, KitriIoResult};

/// A single event in a Kitri workflow's history.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KitriEvent {
    /// An I/O operation was requested by the workflow.
    IoRequested { seq: u64, op: KitriIoOp },
    /// An I/O operation completed (success or failure).
    IoResolved { seq: u64, result: KitriIoResult },
    /// An explicit checkpoint was saved.
    CheckpointSaved { seq: u64, state_cid: [u8; 32] },
    /// A compensator was registered for saga rollback.
    CompensatorRegistered { seq: u64, step_id: u64 },
}

/// The full event log for a workflow instance.
///
/// Content-addressable: the log can be stored and replicated via CID.
/// On crash recovery, the runtime replays this log to skip completed I/O.
#[derive(Debug, Clone)]
pub struct KitriEventLog {
    workflow_id: [u8; 32],
    events: Vec<KitriEvent>,
}

impl KitriEventLog {
    pub fn new(workflow_id: [u8; 32]) -> Self {
        Self {
            workflow_id,
            events: Vec::new(),
        }
    }

    pub fn workflow_id(&self) -> &[u8; 32] {
        &self.workflow_id
    }

    pub fn append(&mut self, event: KitriEvent) {
        self.events.push(event);
    }

    pub fn events(&self) -> &[KitriEvent] {
        &self.events
    }

    pub fn len(&self) -> usize {
        self.events.len()
    }

    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }

    /// Look up the cached result for a given sequence number.
    /// Used during replay to skip completed I/O.
    pub fn cached_result(&self, seq: u64) -> Option<&KitriIoResult> {
        self.events.iter().find_map(|e| match e {
            KitriEvent::IoResolved {
                seq: s,
                result,
            } if *s == seq => Some(result),
            _ => None,
        })
    }

    /// Find the most recent checkpoint, if any.
    pub fn last_checkpoint(&self) -> Option<(u64, [u8; 32])> {
        self.events.iter().rev().find_map(|e| match e {
            KitriEvent::CheckpointSaved { seq, state_cid } => Some((*seq, *state_cid)),
            _ => None,
        })
    }
}
```

**Step 4: Run event tests**

Run: `cargo test -p harmony-kitri event::tests -v`
Expected: PASS

**Step 5: Write failing tests for checkpoint.rs**

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checkpoint_creation() {
        let cp = KitriCheckpoint {
            workflow_id: [0xAA; 32],
            seq: 5,
            state: vec![1, 2, 3],
            state_cid: [0xBB; 32],
            fuel_consumed: 42_000,
        };
        assert_eq!(cp.seq, 5);
        assert_eq!(cp.fuel_consumed, 42_000);
        assert_eq!(cp.state, vec![1, 2, 3]);
    }

    #[test]
    fn checkpoint_is_newer_than() {
        let cp1 = KitriCheckpoint {
            workflow_id: [0; 32],
            seq: 3,
            state: vec![],
            state_cid: [0; 32],
            fuel_consumed: 100,
        };
        let cp2 = KitriCheckpoint {
            workflow_id: [0; 32],
            seq: 7,
            state: vec![],
            state_cid: [0; 32],
            fuel_consumed: 200,
        };
        assert!(cp2.is_newer_than(&cp1));
        assert!(!cp1.is_newer_than(&cp2));
        assert!(!cp1.is_newer_than(&cp1));
    }
}
```

**Step 6: Write checkpoint.rs implementation**

```rust
// SPDX-License-Identifier: Apache-2.0 OR MIT
//! Checkpoint types — explicit durability save points for expensive computations.

use alloc::vec::Vec;

/// A serialized checkpoint of workflow state.
///
/// Created by `kitri::checkpoint()`. Stored in content-addressed storage.
/// On crash recovery, the runtime deserializes this and skips forward
/// instead of replaying from scratch.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KitriCheckpoint {
    /// Which workflow this checkpoint belongs to.
    pub workflow_id: [u8; 32],
    /// The event log sequence number at which this checkpoint was taken.
    pub seq: u64,
    /// Serialized workflow state (opaque bytes — the workflow chooses format).
    pub state: Vec<u8>,
    /// CID of the serialized state in content-addressed storage.
    pub state_cid: [u8; 32],
    /// Total fuel consumed up to this checkpoint.
    pub fuel_consumed: u64,
}

impl KitriCheckpoint {
    /// Returns true if this checkpoint is more recent than `other`.
    pub fn is_newer_than(&self, other: &Self) -> bool {
        self.seq > other.seq
    }
}
```

**Step 7: Run all tests**

Run: `cargo test -p harmony-kitri -v`
Expected: PASS (error, io, event, checkpoint tests)

**Step 8: Commit**

```bash
git add crates/harmony-kitri/src/event.rs crates/harmony-kitri/src/checkpoint.rs
git commit -m "feat(kitri): add event log and checkpoint types"
```

---

### Task 5: Trust model types

**Files:**
- Modify: `crates/harmony-kitri/src/trust.rs`
- Test: inline `#[cfg(test)]` module

**Step 1: Write failing tests**

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trust_tier_ordering() {
        // Owner > Delegated > Untrusted
        assert!(TrustTier::Owner > TrustTier::Delegated);
        assert!(TrustTier::Delegated > TrustTier::Untrusted);
        assert!(TrustTier::Owner > TrustTier::Untrusted);
    }

    #[test]
    fn trust_tier_requires_audit() {
        assert!(!TrustTier::Owner.requires_content_audit());
        assert!(!TrustTier::Delegated.requires_content_audit());
        assert!(TrustTier::Untrusted.requires_content_audit());
    }

    #[test]
    fn trust_tier_requires_rate_limit() {
        assert!(!TrustTier::Owner.requires_rate_limit());
        assert!(TrustTier::Delegated.requires_rate_limit());
        assert!(TrustTier::Untrusted.requires_rate_limit());
    }

    #[test]
    fn trust_tier_allows_native() {
        assert!(TrustTier::Owner.allows_native_execution());
        assert!(TrustTier::Delegated.allows_native_execution());
        assert!(!TrustTier::Untrusted.allows_native_execution());
    }

    #[test]
    fn capability_declaration_variants() {
        let cap = CapabilityDecl::Subscribe { topic: "foo/**".into() };
        assert!(matches!(cap, CapabilityDecl::Subscribe { .. }));

        let cap = CapabilityDecl::Infer;
        assert!(matches!(cap, CapabilityDecl::Infer));
    }
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p harmony-kitri trust::tests -v`
Expected: FAIL

**Step 3: Write implementation**

```rust
// SPDX-License-Identifier: Apache-2.0 OR MIT
//! Trust model — UCAN-based tiering for Kitri program execution.

use alloc::string::String;
use alloc::vec::Vec;

/// Trust tier assigned to a Kitri program at deploy time.
///
/// Determined by inspecting the program's UCAN capability chain.
/// Higher tiers get lower-overhead execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum TrustTier {
    /// Anonymous, expired chain, or unknown signer.
    /// Forced into WASM sandbox with full synchronous audit.
    Untrusted = 0,
    /// Valid UCAN chain from a trusted issuer.
    /// Native execution with capability check + rate limiting.
    Delegated = 1,
    /// Signed by the node's own identity.
    /// Native execution with capability check only (fast path).
    Owner = 2,
}

impl TrustTier {
    /// Whether I/O content must be audited before commit.
    pub fn requires_content_audit(self) -> bool {
        self == Self::Untrusted
    }

    /// Whether I/O operations are rate-limited.
    pub fn requires_rate_limit(self) -> bool {
        self != Self::Owner
    }

    /// Whether this tier can run as native Rust (vs. forced WASM).
    pub fn allows_native_execution(self) -> bool {
        self != Self::Untrusted
    }
}

/// A declared capability requirement from `Kitri.toml`.
///
/// At deployment, the runtime issues a UCAN token scoped to exactly
/// these permissions. Any I/O outside scope returns Unauthorized.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CapabilityDecl {
    /// Can subscribe to messages on this topic pattern.
    Subscribe { topic: String },
    /// Can publish messages to this topic pattern.
    Publish { topic: String },
    /// Can fetch content under this CID namespace.
    Fetch { namespace: String },
    /// Can store content under this CID namespace.
    Store { namespace: String },
    /// Can invoke AI models.
    Infer,
    /// Can spawn a specific child workflow by name.
    Spawn { workflow: String },
    /// Can encrypt/decrypt via kernel.
    Seal,
}

/// The set of capabilities declared by a Kitri program.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CapabilitySet {
    pub declarations: Vec<CapabilityDecl>,
}

impl CapabilitySet {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add(&mut self, decl: CapabilityDecl) {
        self.declarations.push(decl);
    }

    pub fn is_empty(&self) -> bool {
        self.declarations.is_empty()
    }
}
```

**Step 4: Run tests**

Run: `cargo test -p harmony-kitri trust::tests -v`
Expected: PASS

**Step 5: Commit**

```bash
git add crates/harmony-kitri/src/trust.rs
git commit -m "feat(kitri): add trust model types (TrustTier, CapabilityDecl)"
```

---

### Task 6: Failure classification and retry policy

**Files:**
- Modify: `crates/harmony-kitri/src/retry.rs`
- Test: inline `#[cfg(test)]` module

**Step 1: Write failing tests**

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn failure_kind_is_retryable() {
        assert!(FailureKind::Transient.is_retryable());
        assert!(FailureKind::ResourceExhausted.is_retryable());
        assert!(!FailureKind::LogicError.is_retryable());
        assert!(!FailureKind::Unauthorized.is_retryable());
        assert!(!FailureKind::IntegrityViolation.is_retryable());
    }

    #[test]
    fn retry_policy_default() {
        let policy = RetryPolicy::default();
        assert_eq!(policy.max_retries, 3);
        assert!(matches!(policy.backoff, BackoffStrategy::Exponential { .. }));
    }

    #[test]
    fn retry_policy_no_retries() {
        let policy = RetryPolicy::none();
        assert_eq!(policy.max_retries, 0);
    }

    #[test]
    fn backoff_delay_exponential() {
        let backoff = BackoffStrategy::Exponential {
            initial_ms: 100,
            max_ms: 10_000,
        };
        assert_eq!(backoff.delay_ms(0), 100);   // 100 * 2^0
        assert_eq!(backoff.delay_ms(1), 200);   // 100 * 2^1
        assert_eq!(backoff.delay_ms(2), 400);   // 100 * 2^2
        assert_eq!(backoff.delay_ms(10), 10_000); // capped at max
    }

    #[test]
    fn backoff_delay_fixed() {
        let backoff = BackoffStrategy::Fixed { interval_ms: 500 };
        assert_eq!(backoff.delay_ms(0), 500);
        assert_eq!(backoff.delay_ms(5), 500);
    }
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p harmony-kitri retry::tests -v`
Expected: FAIL

**Step 3: Write implementation**

```rust
// SPDX-License-Identifier: Apache-2.0 OR MIT
//! Failure classification and retry policy.

use core::cmp;

/// Classification of a workflow failure.
///
/// Determines whether automatic retry is appropriate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FailureKind {
    /// Network timeout, node crash, temporary unavailability.
    Transient,
    /// Fuel budget or memory limit exceeded.
    ResourceExhausted,
    /// Panic, assertion failure, unrecoverable logic error.
    LogicError,
    /// Capability denied — missing or invalid UCAN.
    Unauthorized,
    /// Lyll/Nakaiah detected memory corruption.
    IntegrityViolation,
}

impl FailureKind {
    /// Whether this failure kind should trigger automatic retry.
    pub fn is_retryable(self) -> bool {
        matches!(self, Self::Transient | Self::ResourceExhausted)
    }
}

/// Backoff strategy between retries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackoffStrategy {
    /// Fixed delay between retries.
    Fixed { interval_ms: u64 },
    /// Exponential backoff: `initial_ms * 2^attempt`, capped at `max_ms`.
    Exponential { initial_ms: u64, max_ms: u64 },
}

impl BackoffStrategy {
    /// Compute the delay in milliseconds for a given retry attempt (0-indexed).
    pub fn delay_ms(&self, attempt: u32) -> u64 {
        match self {
            Self::Fixed { interval_ms } => *interval_ms,
            Self::Exponential { initial_ms, max_ms } => {
                let delay = initial_ms.saturating_mul(1u64.saturating_shl(attempt));
                cmp::min(delay, *max_ms)
            }
        }
    }
}

/// Retry policy for a Kitri workflow.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RetryPolicy {
    /// Maximum number of retry attempts (0 = no retries).
    pub max_retries: u32,
    /// Backoff strategy between retries.
    pub backoff: BackoffStrategy,
    /// Total timeout in milliseconds (0 = no timeout).
    pub timeout_ms: u64,
}

impl RetryPolicy {
    /// No retries — fail immediately on any error.
    pub fn none() -> Self {
        Self {
            max_retries: 0,
            backoff: BackoffStrategy::Fixed { interval_ms: 0 },
            timeout_ms: 0,
        }
    }
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self {
            max_retries: 3,
            backoff: BackoffStrategy::Exponential {
                initial_ms: 100,
                max_ms: 30_000,
            },
            timeout_ms: 300_000, // 5 minutes
        }
    }
}
```

**Step 4: Run tests**

Run: `cargo test -p harmony-kitri retry::tests -v`
Expected: PASS

**Step 5: Commit**

```bash
git add crates/harmony-kitri/src/retry.rs
git commit -m "feat(kitri): add failure classification and retry policy types"
```

---

### Task 7: Program descriptor and DAG composition

**Files:**
- Modify: `crates/harmony-kitri/src/program.rs`
- Modify: `crates/harmony-kitri/src/dag.rs`
- Test: inline `#[cfg(test)]` modules

**Step 1: Write failing tests for program.rs**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::retry::RetryPolicy;
    use crate::trust::{CapabilitySet, TrustTier};

    #[test]
    fn program_descriptor_creation() {
        let desc = KitriProgram {
            cid: [0xAA; 32],
            name: "shipment-verifier".into(),
            version: "0.1.0".into(),
            trust_tier: TrustTier::Owner,
            capabilities: CapabilitySet::new(),
            retry_policy: RetryPolicy::default(),
            prefer_native: true,
        };
        assert_eq!(desc.name, "shipment-verifier");
        assert!(desc.prefer_native);
    }

    #[test]
    fn workflow_id_deterministic() {
        let id1 = kitri_workflow_id(&[0xAA; 32], &[1, 2, 3]);
        let id2 = kitri_workflow_id(&[0xAA; 32], &[1, 2, 3]);
        assert_eq!(id1, id2);

        // Different input → different id.
        let id3 = kitri_workflow_id(&[0xAA; 32], &[4, 5, 6]);
        assert_ne!(id1, id3);

        // Different program → different id.
        let id4 = kitri_workflow_id(&[0xBB; 32], &[1, 2, 3]);
        assert_ne!(id1, id4);
    }
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p harmony-kitri program::tests -v`
Expected: FAIL

**Step 3: Write program.rs implementation**

```rust
// SPDX-License-Identifier: Apache-2.0 OR MIT
//! Program descriptor — metadata about a deployed Kitri program.

use alloc::string::String;

use crate::retry::RetryPolicy;
use crate::trust::{CapabilitySet, TrustTier};

/// Metadata describing a deployed Kitri program.
///
/// The `cid` is the content-address of the compiled binary (native or WASM).
/// The rest is derived from `Kitri.toml` + UCAN chain inspection.
#[derive(Debug, Clone)]
pub struct KitriProgram {
    /// Content-address of the compiled binary.
    pub cid: [u8; 32],
    /// Human-readable name from `Kitri.toml`.
    pub name: String,
    /// Version from `Kitri.toml`.
    pub version: String,
    /// Trust tier assigned at deploy time.
    pub trust_tier: TrustTier,
    /// Declared capabilities from `Kitri.toml`.
    pub capabilities: CapabilitySet,
    /// Retry policy from `Kitri.toml`.
    pub retry_policy: RetryPolicy,
    /// Whether to prefer native Rust execution over WASM.
    pub prefer_native: bool,
}

/// Compute a deterministic workflow ID: `BLAKE3(program_cid || input)`.
///
/// Same program + same input = same WorkflowId. This enables exactly-once
/// semantics: duplicate submissions return the cached result.
pub fn kitri_workflow_id(program_cid: &[u8; 32], input: &[u8]) -> [u8; 32] {
    use harmony_crypto::hash::blake3_hash;

    let mut preimage = alloc::vec::Vec::with_capacity(32 + input.len());
    preimage.extend_from_slice(program_cid);
    preimage.extend_from_slice(input);
    blake3_hash(&preimage)
}
```

Note: Check if `harmony-crypto` exposes `blake3_hash`. If it's named differently (e.g., `blake3` or a wrapper), adjust the call. Look at `harmony-workflow/src/types.rs` for how `WorkflowId::new` computes its hash — follow the same pattern.

**Step 4: Run program tests**

Run: `cargo test -p harmony-kitri program::tests -v`
Expected: PASS

**Step 5: Write failing tests for dag.rs**

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dag_step_creation() {
        let step = DagStep {
            workflow: "scan-receiver".into(),
            input_topic: None,
            output_topic: Some("shipments/scanned".into()),
        };
        assert_eq!(step.workflow, "scan-receiver");
        assert!(step.input_topic.is_none());
        assert!(step.output_topic.is_some());
    }

    #[test]
    fn dag_creation_and_validation() {
        let dag = KitriDag {
            name: "supply-chain".into(),
            steps: vec![
                DagStep {
                    workflow: "scan-receiver".into(),
                    input_topic: None,
                    output_topic: Some("shipments/scanned".into()),
                },
                DagStep {
                    workflow: "shipment-verifier".into(),
                    input_topic: Some("shipments/scanned".into()),
                    output_topic: Some("shipments/verified".into()),
                },
            ],
            retry_subgraph: true,
        };
        assert_eq!(dag.steps.len(), 2);
        assert!(dag.validate().is_ok());
    }

    #[test]
    fn dag_empty_is_invalid() {
        let dag = KitriDag {
            name: "empty".into(),
            steps: vec![],
            retry_subgraph: false,
        };
        assert!(dag.validate().is_err());
    }

    #[test]
    fn dag_broken_chain_is_invalid() {
        let dag = KitriDag {
            name: "broken".into(),
            steps: vec![
                DagStep {
                    workflow: "step-a".into(),
                    input_topic: None,
                    output_topic: Some("topic-a".into()),
                },
                DagStep {
                    workflow: "step-b".into(),
                    input_topic: Some("topic-MISSING".into()),
                    output_topic: None,
                },
            ],
            retry_subgraph: false,
        };
        assert!(dag.validate().is_err());
    }
}
```

**Step 6: Write dag.rs implementation**

```rust
// SPDX-License-Identifier: Apache-2.0 OR MIT
//! DAG composition — explicit workflow topology declarations.

use alloc::collections::BTreeSet;
use alloc::string::String;
use alloc::vec::Vec;

use crate::error::KitriError;

/// A single step in a Kitri DAG.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DagStep {
    /// Name of the Kitri workflow to execute.
    pub workflow: String,
    /// Zenoh topic this step consumes from (None = entry point).
    pub input_topic: Option<String>,
    /// Zenoh topic this step publishes to (None = terminal).
    pub output_topic: Option<String>,
}

/// A declared DAG of Kitri workflows.
///
/// This is the explicit version of implicit pub/sub composition.
/// When the runtime sees a DAG, it can optimize scheduling, provide
/// end-to-end tracing, and retry failed subgraphs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KitriDag {
    /// Human-readable name for this DAG.
    pub name: String,
    /// Ordered list of steps.
    pub steps: Vec<DagStep>,
    /// Whether to retry from the failed step (true) or from scratch (false).
    pub retry_subgraph: bool,
}

impl KitriDag {
    /// Validate the DAG structure.
    ///
    /// Checks:
    /// - At least one step exists
    /// - Every input_topic is produced by a preceding step's output_topic
    pub fn validate(&self) -> Result<(), KitriError> {
        if self.steps.is_empty() {
            return Err(KitriError::DagInvalid {
                reason: "DAG has no steps".into(),
            });
        }

        let mut available_topics = BTreeSet::new();
        for step in &self.steps {
            if let Some(ref input) = step.input_topic {
                if !available_topics.contains(input) {
                    return Err(KitriError::DagInvalid {
                        reason: alloc::format!(
                            "step '{}' requires topic '{}' which is not produced by a preceding step",
                            step.workflow, input
                        ),
                    });
                }
            }
            if let Some(ref output) = step.output_topic {
                available_topics.insert(output.clone());
            }
        }

        Ok(())
    }

    /// Return the entry points (steps with no input topic).
    pub fn entry_points(&self) -> Vec<&DagStep> {
        self.steps.iter().filter(|s| s.input_topic.is_none()).collect()
    }

    /// Return the terminal steps (steps with no output topic).
    pub fn terminal_steps(&self) -> Vec<&DagStep> {
        self.steps.iter().filter(|s| s.output_topic.is_none()).collect()
    }
}
```

**Step 7: Run all tests**

Run: `cargo test -p harmony-kitri -v`
Expected: PASS

**Step 8: Commit**

```bash
git add crates/harmony-kitri/src/program.rs crates/harmony-kitri/src/dag.rs
git commit -m "feat(kitri): add program descriptor and DAG composition types"
```

---

### Task 8: Staged writes and compensation

**Files:**
- Modify: `crates/harmony-kitri/src/staging.rs`
- Modify: `crates/harmony-kitri/src/compensation.rs`
- Test: inline `#[cfg(test)]` modules

**Step 1: Write failing tests for staging.rs**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::KitriIoOp;

    #[test]
    fn staging_buffer_empty_initially() {
        let buf = StagingBuffer::new();
        assert!(buf.is_empty());
        assert_eq!(buf.len(), 0);
    }

    #[test]
    fn staging_buffer_stage_and_commit() {
        let mut buf = StagingBuffer::new();
        buf.stage(StagedWrite {
            seq: 0,
            op: KitriIoOp::Publish {
                topic: "test/topic".into(),
                payload: vec![1, 2, 3],
            },
        });
        assert_eq!(buf.len(), 1);

        let committed = buf.commit_all();
        assert_eq!(committed.len(), 1);
        assert!(buf.is_empty());
    }

    #[test]
    fn staging_buffer_discard() {
        let mut buf = StagingBuffer::new();
        buf.stage(StagedWrite {
            seq: 0,
            op: KitriIoOp::Store { data: vec![1] },
        });
        buf.stage(StagedWrite {
            seq: 1,
            op: KitriIoOp::Publish {
                topic: "t".into(),
                payload: vec![2],
            },
        });
        assert_eq!(buf.len(), 2);

        buf.discard_all();
        assert!(buf.is_empty());
    }
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p harmony-kitri staging::tests -v`
Expected: FAIL

**Step 3: Write staging.rs implementation**

```rust
// SPDX-License-Identifier: Apache-2.0 OR MIT
//! Staged writes — buffer externally-visible side effects until audit passes.

use alloc::vec::Vec;

use crate::io::KitriIoOp;

/// A side-effect operation waiting in the staging buffer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StagedWrite {
    /// Event log sequence number that produced this write.
    pub seq: u64,
    /// The I/O operation to commit.
    pub op: KitriIoOp,
}

/// Buffer for externally-visible side effects.
///
/// All Publish, Store, Seal, and Spawn operations are staged here
/// before being committed. On workflow failure, the buffer is
/// discarded atomically — no partial side effects.
#[derive(Debug, Clone, Default)]
pub struct StagingBuffer {
    writes: Vec<StagedWrite>,
}

impl StagingBuffer {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn stage(&mut self, write: StagedWrite) {
        self.writes.push(write);
    }

    pub fn len(&self) -> usize {
        self.writes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.writes.is_empty()
    }

    /// Commit all staged writes. Returns the committed operations
    /// for the caller to execute. Clears the buffer.
    pub fn commit_all(&mut self) -> Vec<StagedWrite> {
        core::mem::take(&mut self.writes)
    }

    /// Discard all staged writes (workflow failed or was killed).
    pub fn discard_all(&mut self) {
        self.writes.clear();
    }

    /// View the staged writes without committing.
    pub fn pending(&self) -> &[StagedWrite] {
        &self.writes
    }
}
```

**Step 4: Run staging tests**

Run: `cargo test -p harmony-kitri staging::tests -v`
Expected: PASS

**Step 5: Write failing tests for compensation.rs**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::KitriIoOp;

    #[test]
    fn compensator_registration() {
        let comp = Compensator {
            step_id: 0,
            rollback_op: KitriIoOp::Query {
                topic: "accounts/credit".into(),
                payload: vec![1, 2, 3],
            },
        };
        assert_eq!(comp.step_id, 0);
    }

    #[test]
    fn compensation_log_empty_initially() {
        let log = CompensationLog::new();
        assert!(log.is_empty());
    }

    #[test]
    fn compensation_log_register_and_drain() {
        let mut log = CompensationLog::new();
        log.register(Compensator {
            step_id: 0,
            rollback_op: KitriIoOp::Query {
                topic: "accounts/credit".into(),
                payload: vec![1],
            },
        });
        log.register(Compensator {
            step_id: 1,
            rollback_op: KitriIoOp::Query {
                topic: "inventory/restock".into(),
                payload: vec![2],
            },
        });
        assert_eq!(log.len(), 2);

        // Drain returns compensators in REVERSE order (LIFO for saga rollback).
        let drained = log.drain_reverse();
        assert_eq!(drained.len(), 2);
        assert_eq!(drained[0].step_id, 1); // most recent first
        assert_eq!(drained[1].step_id, 0);
        assert!(log.is_empty());
    }

    #[test]
    fn compensation_log_discard() {
        let mut log = CompensationLog::new();
        log.register(Compensator {
            step_id: 0,
            rollback_op: KitriIoOp::Query {
                topic: "t".into(),
                payload: vec![],
            },
        });
        // On success, discard all compensators.
        log.discard_all();
        assert!(log.is_empty());
    }
}
```

**Step 6: Write compensation.rs implementation**

```rust
// SPDX-License-Identifier: Apache-2.0 OR MIT
//! Saga compensation — opt-in rollback for multi-step workflows.

use alloc::vec::Vec;

use crate::io::KitriIoOp;

/// A registered compensator that runs if the workflow fails after this step.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Compensator {
    /// Identifies which step this compensator rolls back.
    pub step_id: u64,
    /// The I/O operation to execute for rollback.
    pub rollback_op: KitriIoOp,
}

/// Log of registered compensators for a workflow.
///
/// On success, all compensators are discarded.
/// On failure, compensators execute in reverse order (LIFO)
/// to undo completed steps.
#[derive(Debug, Clone, Default)]
pub struct CompensationLog {
    compensators: Vec<Compensator>,
}

impl CompensationLog {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register(&mut self, compensator: Compensator) {
        self.compensators.push(compensator);
    }

    pub fn len(&self) -> usize {
        self.compensators.len()
    }

    pub fn is_empty(&self) -> bool {
        self.compensators.is_empty()
    }

    /// Drain all compensators in reverse order (LIFO) for saga rollback.
    /// The log is empty after this call.
    pub fn drain_reverse(&mut self) -> Vec<Compensator> {
        let mut result = core::mem::take(&mut self.compensators);
        result.reverse();
        result
    }

    /// Discard all compensators (workflow completed successfully).
    pub fn discard_all(&mut self) {
        self.compensators.clear();
    }
}
```

**Step 7: Run all tests**

Run: `cargo test -p harmony-kitri -v`
Expected: PASS

**Step 8: Commit**

```bash
git add crates/harmony-kitri/src/staging.rs crates/harmony-kitri/src/compensation.rs
git commit -m "feat(kitri): add staged writes and saga compensation types"
```

---

### Task 9: Manifest types

**Files:**
- Modify: `crates/harmony-kitri/src/manifest.rs`
- Test: inline `#[cfg(test)]` module

The manifest represents the parsed `Kitri.toml`. Parsing from TOML is feature-gated behind `std` (since `toml` needs std). The types themselves are no_std.

**Step 1: Write failing tests**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::retry::{BackoffStrategy, RetryPolicy};
    use crate::trust::{CapabilityDecl, CapabilitySet, TrustTier};

    #[test]
    fn manifest_creation() {
        let manifest = KitriManifest {
            name: "shipment-verifier".into(),
            version: "0.1.0".into(),
            runtime: RuntimeConfig {
                max_retries: 3,
                retry_policy: RetryPolicy::default(),
                fuel_budget: 1_000_000,
            },
            capabilities: CapabilitySet::new(),
            trust: TrustConfig {
                signers: vec![],
            },
            deploy: DeployConfig {
                prefer_native: true,
                replicas: 3,
            },
        };
        assert_eq!(manifest.name, "shipment-verifier");
        assert!(manifest.deploy.prefer_native);
        assert_eq!(manifest.deploy.replicas, 3);
    }

    #[test]
    fn runtime_config_defaults() {
        let config = RuntimeConfig::default();
        assert_eq!(config.max_retries, 3);
        assert_eq!(config.fuel_budget, 1_000_000);
    }

    #[test]
    fn deploy_config_defaults() {
        let config = DeployConfig::default();
        assert!(config.prefer_native);
        assert_eq!(config.replicas, 1);
    }

    #[cfg(feature = "std")]
    #[test]
    fn parse_kitri_toml() {
        let toml_str = r#"
[package]
name = "test-workflow"
version = "0.1.0"

[runtime]
max_retries = 5
fuel_budget = 2000000

[deploy]
prefer_native = false
replicas = 2
"#;
        let manifest = KitriManifest::from_toml(toml_str).unwrap();
        assert_eq!(manifest.name, "test-workflow");
        assert_eq!(manifest.version, "0.1.0");
        assert_eq!(manifest.runtime.max_retries, 5);
        assert_eq!(manifest.runtime.fuel_budget, 2_000_000);
        assert!(!manifest.deploy.prefer_native);
        assert_eq!(manifest.deploy.replicas, 2);
    }
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p harmony-kitri manifest::tests -v`
Expected: FAIL

**Step 3: Write implementation**

```rust
// SPDX-License-Identifier: Apache-2.0 OR MIT
//! Manifest types — parsed representation of `Kitri.toml`.

use alloc::string::String;
use alloc::vec::Vec;

use crate::retry::RetryPolicy;
use crate::trust::CapabilitySet;

/// Parsed `Kitri.toml` manifest.
#[derive(Debug, Clone)]
pub struct KitriManifest {
    pub name: String,
    pub version: String,
    pub runtime: RuntimeConfig,
    pub capabilities: CapabilitySet,
    pub trust: TrustConfig,
    pub deploy: DeployConfig,
}

/// Runtime configuration from `[runtime]` section.
#[derive(Debug, Clone)]
pub struct RuntimeConfig {
    pub max_retries: u32,
    pub retry_policy: RetryPolicy,
    pub fuel_budget: u64,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            max_retries: 3,
            retry_policy: RetryPolicy::default(),
            fuel_budget: 1_000_000,
        }
    }
}

/// Trust configuration from `[trust]` section.
#[derive(Debug, Clone, Default)]
pub struct TrustConfig {
    /// DID keys of trusted UCAN issuers.
    pub signers: Vec<String>,
}

/// Deployment preferences from `[deploy]` section.
#[derive(Debug, Clone)]
pub struct DeployConfig {
    pub prefer_native: bool,
    pub replicas: u32,
}

impl Default for DeployConfig {
    fn default() -> Self {
        Self {
            prefer_native: true,
            replicas: 1,
        }
    }
}

// ── TOML parsing (std-only) ─────────────────────────────────────────

#[cfg(feature = "std")]
mod parsing {
    use super::*;
    use crate::error::KitriError;
    use serde::Deserialize;

    #[derive(Deserialize)]
    struct RawManifest {
        package: RawPackage,
        runtime: Option<RawRuntime>,
        deploy: Option<RawDeploy>,
    }

    #[derive(Deserialize)]
    struct RawPackage {
        name: String,
        version: String,
    }

    #[derive(Deserialize)]
    struct RawRuntime {
        max_retries: Option<u32>,
        fuel_budget: Option<u64>,
    }

    #[derive(Deserialize)]
    struct RawDeploy {
        prefer_native: Option<bool>,
        replicas: Option<u32>,
    }

    impl KitriManifest {
        /// Parse a `Kitri.toml` string into a manifest.
        pub fn from_toml(toml_str: &str) -> Result<Self, KitriError> {
            let raw: RawManifest = toml::from_str(toml_str).map_err(|e| {
                KitriError::ManifestInvalid {
                    reason: e.to_string(),
                }
            })?;

            let mut runtime = RuntimeConfig::default();
            if let Some(rt) = raw.runtime {
                if let Some(mr) = rt.max_retries {
                    runtime.max_retries = mr;
                }
                if let Some(fb) = rt.fuel_budget {
                    runtime.fuel_budget = fb;
                }
            }

            let mut deploy = DeployConfig::default();
            if let Some(dep) = raw.deploy {
                if let Some(pn) = dep.prefer_native {
                    deploy.prefer_native = pn;
                }
                if let Some(r) = dep.replicas {
                    deploy.replicas = r;
                }
            }

            Ok(KitriManifest {
                name: raw.package.name,
                version: raw.package.version,
                runtime,
                capabilities: CapabilitySet::new(),
                trust: TrustConfig::default(),
                deploy,
            })
        }
    }
}
```

**Step 4: Run tests**

Run: `cargo test -p harmony-kitri manifest::tests -v`
Expected: PASS

**Step 5: Commit**

```bash
git add crates/harmony-kitri/src/manifest.rs
git commit -m "feat(kitri): add manifest types and TOML parsing"
```

---

### Task 10: KitriEngine — the sans-I/O state machine

**Files:**
- Create: `crates/harmony-kitri/src/engine.rs`
- Modify: `crates/harmony-kitri/src/lib.rs` (add `pub mod engine` + re-exports)
- Test: inline `#[cfg(test)]` module

This is the central state machine that ties everything together. It follows the same Event/Action pattern as `WorkflowEngine`, `PubSubRouter`, and `Session`.

**Step 1: Write failing tests**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::{KitriIoOp, KitriIoResult};
    use crate::retry::RetryPolicy;
    use crate::trust::{CapabilitySet, TrustTier};

    fn test_program() -> KitriProgram {
        KitriProgram {
            cid: [0xAA; 32],
            name: "test-workflow".into(),
            version: "0.1.0".into(),
            trust_tier: TrustTier::Owner,
            capabilities: CapabilitySet::new(),
            retry_policy: RetryPolicy::default(),
            prefer_native: true,
        }
    }

    #[test]
    fn engine_submit_returns_workflow_id() {
        let mut engine = KitriEngine::new();
        let program = test_program();
        let input = vec![1, 2, 3];

        let actions = engine.handle(KitriEvent::Submit {
            program: program.clone(),
            input: input.clone(),
        });

        // Should produce a WorkflowAccepted action with deterministic ID.
        let expected_id = kitri_workflow_id(&program.cid, &input);
        assert!(actions.iter().any(|a| matches!(
            a,
            KitriAction::WorkflowAccepted { workflow_id } if *workflow_id == expected_id
        )));
    }

    #[test]
    fn engine_duplicate_submit_deduplicates() {
        let mut engine = KitriEngine::new();
        let program = test_program();
        let input = vec![1, 2, 3];

        engine.handle(KitriEvent::Submit {
            program: program.clone(),
            input: input.clone(),
        });
        let actions = engine.handle(KitriEvent::Submit {
            program,
            input,
        });

        // Duplicate should produce Deduplicated, not a new WorkflowAccepted.
        assert!(actions.iter().any(|a| matches!(a, KitriAction::Deduplicated { .. })));
    }

    #[test]
    fn engine_io_requested_stages_side_effects() {
        let mut engine = KitriEngine::new();
        let program = test_program();
        let input = vec![1, 2, 3];
        let wf_id = kitri_workflow_id(&program.cid, &input);

        engine.handle(KitriEvent::Submit { program, input });

        // Simulate workflow requesting a publish (side effect).
        let actions = engine.handle(KitriEvent::IoRequested {
            workflow_id: wf_id,
            op: KitriIoOp::Publish {
                topic: "test/out".into(),
                payload: vec![42],
            },
        });

        // Should produce ExecuteIo for the runtime to handle,
        // but the write should be staged internally.
        assert!(actions.iter().any(|a| matches!(a, KitriAction::ExecuteIo { .. })));
    }

    #[test]
    fn engine_workflow_complete_commits_staged_writes() {
        let mut engine = KitriEngine::new();
        let program = test_program();
        let input = vec![1, 2, 3];
        let wf_id = kitri_workflow_id(&program.cid, &input);

        engine.handle(KitriEvent::Submit { program, input });

        // Stage a side effect.
        engine.handle(KitriEvent::IoRequested {
            workflow_id: wf_id,
            op: KitriIoOp::Publish {
                topic: "test/out".into(),
                payload: vec![42],
            },
        });
        engine.handle(KitriEvent::IoResolved {
            workflow_id: wf_id,
            result: KitriIoResult::Published,
        });

        // Complete the workflow.
        let actions = engine.handle(KitriEvent::WorkflowComplete {
            workflow_id: wf_id,
            output: vec![99],
        });

        // Should commit staged writes and emit completion.
        assert!(actions.iter().any(|a| matches!(a, KitriAction::CommitStagedWrite { .. })));
        assert!(actions.iter().any(|a| matches!(a, KitriAction::Complete { .. })));
    }

    #[test]
    fn engine_workflow_failed_discards_staged_writes() {
        let mut engine = KitriEngine::new();
        let program = test_program();
        let input = vec![1, 2, 3];
        let wf_id = kitri_workflow_id(&program.cid, &input);

        engine.handle(KitriEvent::Submit { program, input });

        // Stage a side effect.
        engine.handle(KitriEvent::IoRequested {
            workflow_id: wf_id,
            op: KitriIoOp::Store { data: vec![1] },
        });

        // Fail the workflow.
        let actions = engine.handle(KitriEvent::WorkflowFailed {
            workflow_id: wf_id,
            error: "boom".into(),
        });

        // Should NOT commit staged writes. Should emit failure.
        assert!(!actions.iter().any(|a| matches!(a, KitriAction::CommitStagedWrite { .. })));
        assert!(actions.iter().any(|a| matches!(a, KitriAction::Failed { .. })));
    }

    #[test]
    fn engine_status_tracking() {
        let mut engine = KitriEngine::new();
        let program = test_program();
        let input = vec![1, 2, 3];
        let wf_id = kitri_workflow_id(&program.cid, &input);

        assert_eq!(engine.status(&wf_id), None);

        engine.handle(KitriEvent::Submit { program, input });
        assert_eq!(engine.status(&wf_id), Some(KitriWorkflowStatus::Pending));

        engine.handle(KitriEvent::WorkflowComplete {
            workflow_id: wf_id,
            output: vec![],
        });
        assert_eq!(engine.status(&wf_id), Some(KitriWorkflowStatus::Complete));
    }
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p harmony-kitri engine::tests -v`
Expected: FAIL

**Step 3: Write implementation**

```rust
// SPDX-License-Identifier: Apache-2.0 OR MIT
//! KitriEngine — the sans-I/O state machine for Kitri workflow orchestration.
//!
//! Follows the same Event → handle() → Vec<Action> pattern as
//! WorkflowEngine, PubSubRouter, Session, and all other Harmony state machines.

use alloc::collections::BTreeMap;
use alloc::string::String;
use alloc::vec::Vec;

use crate::compensation::CompensationLog;
use crate::event::KitriEventLog;
use crate::io::{KitriIoOp, KitriIoResult};
use crate::program::{kitri_workflow_id, KitriProgram};
use crate::staging::{StagedWrite, StagingBuffer};

/// Inbound events to the Kitri engine.
#[derive(Debug, Clone)]
pub enum KitriEvent {
    /// Submit a new workflow for execution.
    Submit { program: KitriProgram, input: Vec<u8> },
    /// A workflow requested a durable I/O operation.
    IoRequested { workflow_id: [u8; 32], op: KitriIoOp },
    /// An I/O operation completed.
    IoResolved { workflow_id: [u8; 32], result: KitriIoResult },
    /// A workflow completed successfully.
    WorkflowComplete { workflow_id: [u8; 32], output: Vec<u8> },
    /// A workflow failed.
    WorkflowFailed { workflow_id: [u8; 32], error: String },
    /// An explicit checkpoint was requested.
    CheckpointRequested { workflow_id: [u8; 32], state: Vec<u8>, state_cid: [u8; 32] },
    /// A compensator was registered.
    CompensatorRegistered { workflow_id: [u8; 32], step_id: u64, rollback_op: KitriIoOp },
}

/// Outbound actions emitted by the Kitri engine.
#[derive(Debug, Clone)]
pub enum KitriAction {
    /// A workflow was accepted and assigned an ID.
    WorkflowAccepted { workflow_id: [u8; 32] },
    /// A duplicate submission was detected — attach to existing.
    Deduplicated { workflow_id: [u8; 32] },
    /// Execute an I/O operation (caller handles the actual I/O).
    ExecuteIo { workflow_id: [u8; 32], seq: u64, op: KitriIoOp },
    /// Commit a staged write (workflow succeeded, audit passed).
    CommitStagedWrite { workflow_id: [u8; 32], write: StagedWrite },
    /// Persist the event log for durability.
    PersistEventLog { workflow_id: [u8; 32] },
    /// Workflow completed successfully.
    Complete { workflow_id: [u8; 32], output: Vec<u8> },
    /// Workflow failed.
    Failed { workflow_id: [u8; 32], error: String },
    /// Execute a compensator for saga rollback.
    ExecuteCompensator { workflow_id: [u8; 32], step_id: u64, rollback_op: KitriIoOp },
}

/// Status of a Kitri workflow.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KitriWorkflowStatus {
    Pending,
    Executing,
    WaitingForIo,
    Complete,
    Failed,
    Compensating,
}

/// Internal state for a single workflow.
struct WorkflowState {
    program: KitriProgram,
    status: KitriWorkflowStatus,
    event_log: KitriEventLog,
    staging: StagingBuffer,
    compensation: CompensationLog,
    next_seq: u64,
    output: Option<Vec<u8>>,
}

/// The Kitri engine — manages workflow lifecycle with durable execution.
pub struct KitriEngine {
    workflows: BTreeMap<[u8; 32], WorkflowState>,
}

impl KitriEngine {
    pub fn new() -> Self {
        Self {
            workflows: BTreeMap::new(),
        }
    }

    pub fn status(&self, workflow_id: &[u8; 32]) -> Option<KitriWorkflowStatus> {
        self.workflows.get(workflow_id).map(|w| w.status)
    }

    pub fn workflow_count(&self) -> usize {
        self.workflows.len()
    }

    pub fn handle(&mut self, event: KitriEvent) -> Vec<KitriAction> {
        match event {
            KitriEvent::Submit { program, input } => self.handle_submit(program, input),
            KitriEvent::IoRequested { workflow_id, op } => {
                self.handle_io_requested(workflow_id, op)
            }
            KitriEvent::IoResolved {
                workflow_id,
                result,
            } => self.handle_io_resolved(workflow_id, result),
            KitriEvent::WorkflowComplete {
                workflow_id,
                output,
            } => self.handle_complete(workflow_id, output),
            KitriEvent::WorkflowFailed {
                workflow_id,
                error,
            } => self.handle_failed(workflow_id, error),
            KitriEvent::CheckpointRequested {
                workflow_id,
                state: _,
                state_cid,
            } => self.handle_checkpoint(workflow_id, state_cid),
            KitriEvent::CompensatorRegistered {
                workflow_id,
                step_id,
                rollback_op,
            } => self.handle_compensator(workflow_id, step_id, rollback_op),
        }
    }

    fn handle_submit(
        &mut self,
        program: KitriProgram,
        input: Vec<u8>,
    ) -> Vec<KitriAction> {
        let wf_id = kitri_workflow_id(&program.cid, &input);

        if self.workflows.contains_key(&wf_id) {
            return vec![KitriAction::Deduplicated { workflow_id: wf_id }];
        }

        self.workflows.insert(
            wf_id,
            WorkflowState {
                program,
                status: KitriWorkflowStatus::Pending,
                event_log: KitriEventLog::new(wf_id),
                staging: StagingBuffer::new(),
                compensation: CompensationLog::new(),
                next_seq: 0,
                output: None,
            },
        );

        vec![KitriAction::WorkflowAccepted { workflow_id: wf_id }]
    }

    fn handle_io_requested(
        &mut self,
        workflow_id: [u8; 32],
        op: KitriIoOp,
    ) -> Vec<KitriAction> {
        let Some(state) = self.workflows.get_mut(&workflow_id) else {
            return vec![];
        };

        let seq = state.next_seq;
        state.next_seq += 1;
        state.status = KitriWorkflowStatus::WaitingForIo;

        // Log the request.
        state.event_log.append(crate::event::KitriEvent::IoRequested {
            seq,
            op: op.clone(),
        });

        // Stage side effects.
        if op.is_side_effect() {
            state.staging.stage(StagedWrite { seq, op: op.clone() });
        }

        vec![
            KitriAction::ExecuteIo {
                workflow_id,
                seq,
                op,
            },
            KitriAction::PersistEventLog { workflow_id },
        ]
    }

    fn handle_io_resolved(
        &mut self,
        workflow_id: [u8; 32],
        result: KitriIoResult,
    ) -> Vec<KitriAction> {
        let Some(state) = self.workflows.get_mut(&workflow_id) else {
            return vec![];
        };

        let seq = if state.next_seq > 0 {
            state.next_seq - 1
        } else {
            0
        };
        state.status = KitriWorkflowStatus::Executing;

        state.event_log.append(crate::event::KitriEvent::IoResolved {
            seq,
            result,
        });

        vec![KitriAction::PersistEventLog { workflow_id }]
    }

    fn handle_complete(
        &mut self,
        workflow_id: [u8; 32],
        output: Vec<u8>,
    ) -> Vec<KitriAction> {
        let Some(state) = self.workflows.get_mut(&workflow_id) else {
            return vec![];
        };

        state.status = KitriWorkflowStatus::Complete;
        state.output = Some(output.clone());
        state.compensation.discard_all();

        // Commit all staged writes.
        let mut actions: Vec<KitriAction> = state
            .staging
            .commit_all()
            .into_iter()
            .map(|w| KitriAction::CommitStagedWrite {
                workflow_id,
                write: w,
            })
            .collect();

        actions.push(KitriAction::PersistEventLog { workflow_id });
        actions.push(KitriAction::Complete {
            workflow_id,
            output,
        });

        actions
    }

    fn handle_failed(
        &mut self,
        workflow_id: [u8; 32],
        error: String,
    ) -> Vec<KitriAction> {
        let Some(state) = self.workflows.get_mut(&workflow_id) else {
            return vec![];
        };

        // Discard staged writes — no partial side effects.
        state.staging.discard_all();

        // Run compensators in reverse order (saga rollback).
        let compensators = state.compensation.drain_reverse();
        let mut actions: Vec<KitriAction> = compensators
            .into_iter()
            .map(|c| KitriAction::ExecuteCompensator {
                workflow_id,
                step_id: c.step_id,
                rollback_op: c.rollback_op,
            })
            .collect();

        state.status = if actions.is_empty() {
            KitriWorkflowStatus::Failed
        } else {
            KitriWorkflowStatus::Compensating
        };

        actions.push(KitriAction::Failed {
            workflow_id,
            error,
        });

        actions
    }

    fn handle_checkpoint(
        &mut self,
        workflow_id: [u8; 32],
        state_cid: [u8; 32],
    ) -> Vec<KitriAction> {
        let Some(state) = self.workflows.get_mut(&workflow_id) else {
            return vec![];
        };

        let seq = state.next_seq;
        state.event_log.append(crate::event::KitriEvent::CheckpointSaved {
            seq,
            state_cid,
        });

        vec![KitriAction::PersistEventLog { workflow_id }]
    }

    fn handle_compensator(
        &mut self,
        workflow_id: [u8; 32],
        step_id: u64,
        rollback_op: KitriIoOp,
    ) -> Vec<KitriAction> {
        let Some(state) = self.workflows.get_mut(&workflow_id) else {
            return vec![];
        };

        state.compensation.register(crate::compensation::Compensator {
            step_id,
            rollback_op,
        });

        let seq = state.next_seq;
        state.event_log.append(crate::event::KitriEvent::CompensatorRegistered {
            seq,
            step_id,
        });

        vec![]
    }
}

impl Default for KitriEngine {
    fn default() -> Self {
        Self::new()
    }
}
```

**Step 4: Update lib.rs** to add `pub mod engine` and re-exports:

```rust
pub mod engine;
// ... existing modules ...

pub use engine::{KitriEngine, KitriEvent, KitriAction, KitriWorkflowStatus};
pub use program::{KitriProgram, kitri_workflow_id};
```

**Step 5: Run all tests**

Run: `cargo test -p harmony-kitri -v`
Expected: PASS

**Step 6: Commit**

```bash
git add crates/harmony-kitri/src/engine.rs crates/harmony-kitri/src/lib.rs
git commit -m "feat(kitri): add KitriEngine — sans-I/O workflow state machine"
```

---

### Task 11: Final integration and workspace wiring

**Files:**
- Modify: `crates/harmony-kitri/src/lib.rs` (final re-exports)
- Verify: workspace builds clean

**Step 1: Finalize lib.rs re-exports**

Ensure `lib.rs` re-exports all public types so consumers can do `use harmony_kitri::*`:

```rust
pub use checkpoint::KitriCheckpoint;
pub use compensation::{CompensationLog, Compensator};
pub use dag::{DagStep, KitriDag};
pub use engine::{KitriAction, KitriEngine, KitriEvent, KitriWorkflowStatus};
pub use error::{KitriError, KitriResult};
pub use event::KitriEventLog;
pub use io::{KitriIoOp, KitriIoResult};
pub use manifest::{DeployConfig, KitriManifest, RuntimeConfig, TrustConfig};
pub use program::{kitri_workflow_id, KitriProgram};
pub use retry::{BackoffStrategy, FailureKind, RetryPolicy};
pub use staging::{StagedWrite, StagingBuffer};
pub use trust::{CapabilityDecl, CapabilitySet, TrustTier};
```

**Step 2: Run full workspace build and test**

Run: `cargo test --workspace`
Expected: All tests pass including harmony-kitri

Run: `cargo clippy --workspace`
Expected: No warnings

Run: `cargo fmt --all -- --check`
Expected: No formatting issues

**Step 3: Commit**

```bash
git add crates/harmony-kitri/
git commit -m "feat(kitri): finalize Layer 0 — all types, traits, and engine complete"
```

---

## Phase 2: Layer 1 (future — separate plan)

Layer 1 (`kitri-macros` + `kitri-sdk`) will be planned in a separate document after Layer 0 ships. It depends on:

1. **`kitri-macros`** — proc-macro crate generating 9P registration, Zenoh wiring, event-sourcing wrappers
2. **`kitri-sdk`** — runtime library providing the `kitri::fetch`, `kitri::store`, `kitri::query`, `kitri::publish`, `kitri::infer`, `kitri::checkpoint`, `kitri::spawn`, `kitri::seal`, `kitri::open` async functions
3. **Integration with `harmony-node`** — wiring the KitriEngine into the node's event loop
4. **CLI tooling** — `kitri build`, `kitri run`, `kitri deploy`

These require design decisions about async runtime integration and the proc-macro API surface that should be explored in a dedicated brainstorming session after Layer 0 is validated.
