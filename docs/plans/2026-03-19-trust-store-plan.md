# Trust Store Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement `harmony-trust` crate with canonical 8-bit TrustScore, directional TrustEdge, TrustStore with local + received edges, and TrustLookup trait.

**Architecture:** Pure-data crate depending only on `harmony-identity` for `IdentityHash`. `no_std`-compatible using `hashbrown` + `postcard`. Same patterns as `harmony-contacts`. TrustLookup trait decouples consumers from store internals.

**Tech Stack:** Rust 1.85+, no_std + alloc, hashbrown, postcard, serde, TDD

**Spec:** `docs/plans/2026-03-19-trust-store-design.md`

---

## File Structure

### New files

```
crates/harmony-trust/Cargo.toml         — crate manifest
crates/harmony-trust/src/lib.rs         — re-exports
crates/harmony-trust/src/score.rs       — TrustScore newtype with dimension accessors
crates/harmony-trust/src/edge.rs        — TrustEdge struct
crates/harmony-trust/src/store.rs       — TrustStore: local edges, received edges, resolution
crates/harmony-trust/src/lookup.rs      — TrustLookup trait + TrustStore impl
crates/harmony-trust/src/error.rs       — TrustError enum
Cargo.toml                              — add workspace member + dep (modify)
```

---

### Task 1: Scaffold `harmony-trust` crate with `TrustScore`

**Files:**
- Create: `crates/harmony-trust/Cargo.toml`
- Create: `crates/harmony-trust/src/lib.rs`
- Create: `crates/harmony-trust/src/score.rs`
- Create: `crates/harmony-trust/src/error.rs`
- Modify: `Cargo.toml` (workspace)

- [ ] **Step 1: Create Cargo.toml**

```toml
[package]
name = "harmony-trust"
description = "Canonical trust scoring for the Harmony network"
version.workspace = true
edition.workspace = true
license.workspace = true
repository.workspace = true
rust-version.workspace = true

[dependencies]
harmony-identity = { workspace = true }
hashbrown = { workspace = true, features = ["serde"] }
postcard = { workspace = true }
serde = { workspace = true, default-features = false, features = ["derive", "alloc"] }

[features]
default = ["std"]
std = [
    "harmony-identity/std",
    "postcard/use-std",
    "serde/std",
]

[dev-dependencies]
```

- [ ] **Step 2: Create error.rs**

```rust
/// Errors returned by TrustStore operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TrustError {
    /// Serialization failed.
    SerializeError(&'static str),
    /// Deserialization failed.
    DeserializeError(&'static str),
}

impl core::fmt::Display for TrustError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::SerializeError(msg) => write!(f, "serialize error: {msg}"),
            Self::DeserializeError(msg) => write!(f, "deserialize error: {msg}"),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for TrustError {}
```

- [ ] **Step 3: Create score.rs with tests FIRST, then implementation**

Tests:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unknown_is_zero() {
        assert_eq!(TrustScore::UNKNOWN.raw(), 0x00);
        assert_eq!(TrustScore::UNKNOWN.identity(), 0);
        assert_eq!(TrustScore::UNKNOWN.compliance(), 0);
        assert_eq!(TrustScore::UNKNOWN.association(), 0);
        assert_eq!(TrustScore::UNKNOWN.endorsement(), 0);
    }

    #[test]
    fn from_dimensions_round_trip() {
        let score = TrustScore::from_dimensions(3, 2, 1, 0);
        assert_eq!(score.identity(), 3);
        assert_eq!(score.compliance(), 2);
        assert_eq!(score.association(), 1);
        assert_eq!(score.endorsement(), 0);
    }

    #[test]
    fn bit_layout_identity_is_msb() {
        // Identity=3 in bits [7:6] = 0b11_00_00_00 = 0xC0
        let score = TrustScore::from_dimensions(3, 0, 0, 0);
        assert_eq!(score.raw(), 0xC0);
    }

    #[test]
    fn bit_layout_endorsement_is_lsb() {
        // Endorsement=3 in bits [1:0] = 0b00_00_00_11 = 0x03
        let score = TrustScore::from_dimensions(0, 0, 0, 3);
        assert_eq!(score.raw(), 0x03);
    }

    #[test]
    fn bit_layout_all_max() {
        // All dimensions at 3 = 0b11_11_11_11 = 0xFF
        let score = TrustScore::from_dimensions(3, 3, 3, 3);
        assert_eq!(score.raw(), 0xFF);
    }

    #[test]
    fn bit_layout_known_byte() {
        // 0xF0 = 0b11_11_00_00 = identity=3, compliance=3, association=0, endorsement=0
        let score = TrustScore::new(0xF0);
        assert_eq!(score.identity(), 3);
        assert_eq!(score.compliance(), 3);
        assert_eq!(score.association(), 0);
        assert_eq!(score.endorsement(), 0);
    }

    #[test]
    fn from_dimensions_clamps_overflow() {
        // Values > 3 should be masked to 2 bits
        let score = TrustScore::from_dimensions(0xFF, 0xFF, 0xFF, 0xFF);
        assert_eq!(score.raw(), 0xFF); // all 3s
        assert_eq!(score.identity(), 3);
        assert_eq!(score.endorsement(), 3);
    }

    #[test]
    fn new_raw_round_trip() {
        for byte in 0u8..=255 {
            assert_eq!(TrustScore::new(byte).raw(), byte);
        }
    }

    #[test]
    fn serde_round_trip() {
        let score = TrustScore::from_dimensions(2, 3, 1, 0);
        let bytes = postcard::to_allocvec(&score).unwrap();
        let decoded: TrustScore = postcard::from_bytes(&bytes).unwrap();
        assert_eq!(decoded, score);
    }
}
```

Implementation:

```rust
use serde::{Deserialize, Serialize};

/// An 8-bit trust score encoding four orthogonal 2-bit dimensions.
///
/// Bit layout (MSB → LSB):
///   [7:6] Identity    — Is this entity verifiably who they claim to be?
///   [5:4] Compliance  — What is their standing? Clean record or evidence
///                       of rule-breaking? (default: unknown, cooperation assumed)
///   [3:2] Association — Do we actively want to include/associate with them?
///   [1:0] Endorsement — Do we invest social capital in them? Elevate them?
///
/// Ordering rationale: exclusion before inclusion (the inclusion-exclusion
/// paradox). Cooperation-by-default, aligned with long-term prisoner's dilemma.
///
/// 0x00 means "no information" (not "distrust").
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TrustScore(u8);

impl TrustScore {
    pub fn new(raw: u8) -> Self {
        Self(raw)
    }

    pub fn raw(self) -> u8 {
        self.0
    }

    pub fn identity(self) -> u8 {
        (self.0 >> 6) & 0x03
    }

    pub fn compliance(self) -> u8 {
        (self.0 >> 4) & 0x03
    }

    pub fn association(self) -> u8 {
        (self.0 >> 2) & 0x03
    }

    pub fn endorsement(self) -> u8 {
        self.0 & 0x03
    }

    pub fn from_dimensions(
        identity: u8,
        compliance: u8,
        association: u8,
        endorsement: u8,
    ) -> Self {
        Self(
            ((identity & 0x03) << 6)
                | ((compliance & 0x03) << 4)
                | ((association & 0x03) << 2)
                | (endorsement & 0x03),
        )
    }

    /// No information — default for unknown identities.
    pub const UNKNOWN: Self = Self(0x00);
}
```

- [ ] **Step 4: Create lib.rs**

```rust
#![cfg_attr(not(feature = "std"), no_std)]
extern crate alloc;

pub mod error;
pub mod score;

pub use error::TrustError;
pub use score::TrustScore;
```

- [ ] **Step 5: Add workspace entries**

In root `Cargo.toml`, add `"crates/harmony-trust"` to `members` and add to `[workspace.dependencies]`:

```toml
harmony-trust = { path = "crates/harmony-trust", default-features = false }
```

- [ ] **Step 6: Run tests**

Run: `cargo test -p harmony-trust`
Expected: 9 tests pass.

- [ ] **Step 7: Commit**

```bash
git add crates/harmony-trust/ Cargo.toml Cargo.lock
git commit -m "feat(trust): scaffold harmony-trust crate with TrustScore"
```

---

### Task 2: Add `TrustEdge` type

**Files:**
- Create: `crates/harmony-trust/src/edge.rs`
- Modify: `crates/harmony-trust/src/lib.rs`

- [ ] **Step 1: Create edge.rs with tests first**

Tests:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::score::TrustScore;

    #[test]
    fn edge_creation() {
        let edge = TrustEdge {
            truster: [0xAA; 16],
            trustee: [0xBB; 16],
            score: TrustScore::from_dimensions(3, 3, 2, 1),
            updated_at: 1000,
        };
        assert_eq!(edge.truster, [0xAA; 16]);
        assert_eq!(edge.trustee, [0xBB; 16]);
        assert_eq!(edge.score.identity(), 3);
        assert_eq!(edge.updated_at, 1000);
    }

    #[test]
    fn edge_equality() {
        let a = TrustEdge {
            truster: [0x01; 16],
            trustee: [0x02; 16],
            score: TrustScore::new(0xFF),
            updated_at: 500,
        };
        let b = a.clone();
        assert_eq!(a, b);
    }

    #[test]
    fn edge_serde_round_trip() {
        let edge = TrustEdge {
            truster: [0xCC; 16],
            trustee: [0xDD; 16],
            score: TrustScore::from_dimensions(1, 2, 3, 0),
            updated_at: 99999,
        };
        let bytes = postcard::to_allocvec(&edge).unwrap();
        let decoded: TrustEdge = postcard::from_bytes(&bytes).unwrap();
        assert_eq!(decoded, edge);
    }
}
```

Implementation:

```rust
use harmony_identity::IdentityHash;
use serde::{Deserialize, Serialize};

use crate::score::TrustScore;

/// A directional trust assessment from one identity to another.
/// Trust is directional: Alice's score for Bob ≠ Bob's score for Alice.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TrustEdge {
    /// The identity assigning the score.
    pub truster: IdentityHash,
    /// The identity being scored.
    pub trustee: IdentityHash,
    /// The trust score.
    pub score: TrustScore,
    /// When this score was last updated (unix timestamp seconds).
    pub updated_at: u64,
}
```

- [ ] **Step 2: Update lib.rs**

Add:

```rust
pub mod edge;
pub use edge::TrustEdge;
```

- [ ] **Step 3: Run tests**

Run: `cargo test -p harmony-trust`
Expected: 12 tests pass (9 score + 3 edge).

- [ ] **Step 4: Commit**

```bash
git add crates/harmony-trust/
git commit -m "feat(trust): add TrustEdge directional trust assessment"
```

---

### Task 3: Implement `TrustStore`

**Files:**
- Create: `crates/harmony-trust/src/store.rs`
- Modify: `crates/harmony-trust/src/lib.rs`

- [ ] **Step 1: Create store.rs with tests first**

Tests:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::score::TrustScore;

    const LOCAL: [u8; 16] = [0x01; 16];
    const ALICE: [u8; 16] = [0xAA; 16];
    const BOB: [u8; 16] = [0xBB; 16];
    const CAROL: [u8; 16] = [0xCC; 16];

    fn score(i: u8, c: u8, a: u8, e: u8) -> TrustScore {
        TrustScore::from_dimensions(i, c, a, e)
    }

    // --- Local edges ---

    #[test]
    fn set_and_get_local_score() {
        let mut store = TrustStore::new(LOCAL);
        store.set_score(&ALICE, score(3, 3, 2, 1), 1000);
        let s = store.local_score(&ALICE).unwrap();
        assert_eq!(s.identity(), 3);
        assert_eq!(s.endorsement(), 1);
    }

    #[test]
    fn overwrite_local_score() {
        let mut store = TrustStore::new(LOCAL);
        store.set_score(&ALICE, score(1, 1, 1, 1), 1000);
        store.set_score(&ALICE, score(3, 3, 3, 3), 2000);
        assert_eq!(store.local_score(&ALICE).unwrap(), score(3, 3, 3, 3));
    }

    #[test]
    fn remove_local_score() {
        let mut store = TrustStore::new(LOCAL);
        store.set_score(&ALICE, score(2, 2, 2, 2), 1000);
        let removed = store.remove_score(&ALICE);
        assert_eq!(removed, Some(score(2, 2, 2, 2)));
        assert!(store.local_score(&ALICE).is_none());
    }

    #[test]
    fn remove_nonexistent_returns_none() {
        let mut store = TrustStore::new(LOCAL);
        assert!(store.remove_score(&ALICE).is_none());
    }

    #[test]
    fn local_score_unknown_returns_none() {
        let store = TrustStore::new(LOCAL);
        assert!(store.local_score(&ALICE).is_none());
    }

    #[test]
    fn iterate_local_edges() {
        let mut store = TrustStore::new(LOCAL);
        store.set_score(&ALICE, score(3, 3, 0, 0), 1000);
        store.set_score(&BOB, score(1, 1, 0, 0), 2000);

        let edges: Vec<_> = store.local_edges().collect();
        assert_eq!(edges.len(), 2);
        assert!(edges.iter().all(|e| e.truster == LOCAL));
        assert!(edges.iter().any(|e| e.trustee == ALICE));
        assert!(edges.iter().any(|e| e.trustee == BOB));
    }

    // --- Received edges ---

    #[test]
    fn receive_and_query_edge() {
        let mut store = TrustStore::new(LOCAL);
        store.receive_edge(TrustEdge {
            truster: ALICE,
            trustee: BOB,
            score: score(2, 2, 1, 0),
            updated_at: 1000,
        });
        assert_eq!(
            store.received_score(&ALICE, &BOB),
            Some(score(2, 2, 1, 0))
        );
    }

    #[test]
    fn newer_received_edge_wins() {
        let mut store = TrustStore::new(LOCAL);
        store.receive_edge(TrustEdge {
            truster: ALICE,
            trustee: BOB,
            score: score(1, 1, 0, 0),
            updated_at: 1000,
        });
        store.receive_edge(TrustEdge {
            truster: ALICE,
            trustee: BOB,
            score: score(3, 3, 3, 3),
            updated_at: 2000,
        });
        assert_eq!(
            store.received_score(&ALICE, &BOB),
            Some(score(3, 3, 3, 3))
        );
    }

    #[test]
    fn stale_received_edge_rejected() {
        let mut store = TrustStore::new(LOCAL);
        store.receive_edge(TrustEdge {
            truster: ALICE,
            trustee: BOB,
            score: score(3, 3, 3, 3),
            updated_at: 2000,
        });
        // Older edge should be ignored
        store.receive_edge(TrustEdge {
            truster: ALICE,
            trustee: BOB,
            score: score(0, 0, 0, 0),
            updated_at: 1000,
        });
        assert_eq!(
            store.received_score(&ALICE, &BOB),
            Some(score(3, 3, 3, 3))
        );
    }

    #[test]
    fn self_edge_ignored() {
        let mut store = TrustStore::new(LOCAL);
        store.receive_edge(TrustEdge {
            truster: LOCAL,
            trustee: ALICE,
            score: score(3, 3, 3, 3),
            updated_at: 1000,
        });
        // Should not be stored in received_edges
        assert!(store.received_score(&LOCAL, &ALICE).is_none());
    }

    #[test]
    fn received_edges_for_trustee() {
        let mut store = TrustStore::new(LOCAL);
        store.receive_edge(TrustEdge {
            truster: ALICE,
            trustee: CAROL,
            score: score(3, 3, 0, 0),
            updated_at: 1000,
        });
        store.receive_edge(TrustEdge {
            truster: BOB,
            trustee: CAROL,
            score: score(1, 1, 0, 0),
            updated_at: 2000,
        });
        // Alice also scored Bob — should NOT appear in Carol's results
        store.receive_edge(TrustEdge {
            truster: ALICE,
            trustee: BOB,
            score: score(2, 2, 0, 0),
            updated_at: 1500,
        });

        let edges = store.received_edges_for(&CAROL);
        assert_eq!(edges.len(), 2);
        assert!(edges.iter().all(|e| e.trustee == CAROL));
    }

    #[test]
    fn received_edges_for_unknown_trustee_empty() {
        let store = TrustStore::new(LOCAL);
        assert!(store.received_edges_for(&ALICE).is_empty());
    }

    // --- Resolution ---

    #[test]
    fn effective_score_local_wins() {
        let mut store = TrustStore::new(LOCAL);
        store.set_score(&ALICE, score(3, 3, 3, 3), 1000);
        store.receive_edge(TrustEdge {
            truster: BOB,
            trustee: ALICE,
            score: score(0, 0, 0, 0),
            updated_at: 2000,
        });
        // Local edge should take precedence
        assert_eq!(store.effective_score(&ALICE), score(3, 3, 3, 3));
    }

    #[test]
    fn effective_score_unknown_when_no_local() {
        let mut store = TrustStore::new(LOCAL);
        // Only received edges, no local
        store.receive_edge(TrustEdge {
            truster: BOB,
            trustee: ALICE,
            score: score(3, 3, 3, 3),
            updated_at: 1000,
        });
        assert_eq!(store.effective_score(&ALICE), TrustScore::UNKNOWN);
    }

    #[test]
    fn effective_score_completely_unknown() {
        let store = TrustStore::new(LOCAL);
        assert_eq!(store.effective_score(&ALICE), TrustScore::UNKNOWN);
    }

    #[test]
    fn effective_score_self_trust() {
        // Querying trust for your own identity — no local edge set
        let store = TrustStore::new(LOCAL);
        assert_eq!(store.effective_score(&LOCAL), TrustScore::UNKNOWN);

        // Can also explicitly score yourself
        let mut store2 = TrustStore::new(LOCAL);
        store2.set_score(&LOCAL, score(3, 3, 3, 3), 1000);
        assert_eq!(store2.effective_score(&LOCAL), score(3, 3, 3, 3));
    }

    // --- Persistence ---

    #[test]
    fn serialize_round_trip() {
        let mut store = TrustStore::new(LOCAL);
        store.set_score(&ALICE, score(3, 3, 2, 1), 1000);
        store.set_score(&BOB, score(1, 1, 0, 0), 2000);
        store.receive_edge(TrustEdge {
            truster: CAROL,
            trustee: ALICE,
            score: score(2, 2, 1, 0),
            updated_at: 3000,
        });

        let bytes = store.serialize().unwrap();
        let restored = TrustStore::deserialize(&bytes).unwrap();
        assert_eq!(restored.local_score(&ALICE), Some(score(3, 3, 2, 1)));
        assert_eq!(restored.local_score(&BOB), Some(score(1, 1, 0, 0)));
        assert_eq!(
            restored.received_score(&CAROL, &ALICE),
            Some(score(2, 2, 1, 0))
        );
    }

    #[test]
    fn deserialize_empty_data() {
        assert!(matches!(
            TrustStore::deserialize(&[]),
            Err(TrustError::DeserializeError(_))
        ));
    }

    #[test]
    fn deserialize_bad_version() {
        assert!(matches!(
            TrustStore::deserialize(&[0xFF, 0x00]),
            Err(TrustError::DeserializeError(_))
        ));
    }

    #[test]
    fn empty_store_serialize_round_trip() {
        let store = TrustStore::new(LOCAL);
        let bytes = store.serialize().unwrap();
        let restored = TrustStore::deserialize(&bytes).unwrap();
        assert_eq!(restored.effective_score(&ALICE), TrustScore::UNKNOWN);
    }
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test -p harmony-trust`
Expected: FAIL — `TrustStore` not defined.

- [ ] **Step 3: Implement TrustStore**

```rust
use alloc::vec::Vec;
use hashbrown::HashMap;
use harmony_identity::IdentityHash;
use serde::{Deserialize, Serialize};

use crate::edge::TrustEdge;
use crate::error::TrustError;
use crate::score::TrustScore;

const FORMAT_VERSION: u8 = 1;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrustStore {
    local_identity: IdentityHash,
    local_edges: HashMap<IdentityHash, LocalEdge>,
    received_edges: HashMap<IdentityHash, HashMap<IdentityHash, ReceivedEdge>>,
}

// Intentionally separate types despite identical fields.
// ReceivedEdge may gain a `signature` field for Zenoh publishing.

#[derive(Debug, Clone, Serialize, Deserialize)]
struct LocalEdge {
    score: TrustScore,
    updated_at: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ReceivedEdge {
    score: TrustScore,
    updated_at: u64,
}

impl TrustStore {
    pub fn new(local_identity: IdentityHash) -> Self {
        Self {
            local_identity,
            local_edges: HashMap::new(),
            received_edges: HashMap::new(),
        }
    }

    // --- Local edges ---

    pub fn set_score(&mut self, trustee: &IdentityHash, score: TrustScore, now: u64) {
        self.local_edges.insert(
            *trustee,
            LocalEdge {
                score,
                updated_at: now,
            },
        );
    }

    pub fn remove_score(&mut self, trustee: &IdentityHash) -> Option<TrustScore> {
        self.local_edges.remove(trustee).map(|e| e.score)
    }

    pub fn local_score(&self, trustee: &IdentityHash) -> Option<TrustScore> {
        self.local_edges.get(trustee).map(|e| e.score)
    }

    pub fn local_edges(&self) -> impl Iterator<Item = TrustEdge> + '_ {
        self.local_edges.iter().map(move |(trustee, edge)| TrustEdge {
            truster: self.local_identity,
            trustee: *trustee,
            score: edge.score,
            updated_at: edge.updated_at,
        })
    }

    // --- Received edges ---

    pub fn receive_edge(&mut self, edge: TrustEdge) {
        // Ignore our own scores echoed back from the network.
        if edge.truster == self.local_identity {
            return;
        }
        let inner = self.received_edges.entry(edge.truster).or_default();
        match inner.get(&edge.trustee) {
            Some(existing) if existing.updated_at >= edge.updated_at => {
                // Stale — ignore.
            }
            _ => {
                inner.insert(
                    edge.trustee,
                    ReceivedEdge {
                        score: edge.score,
                        updated_at: edge.updated_at,
                    },
                );
            }
        }
    }

    pub fn received_score(
        &self,
        truster: &IdentityHash,
        trustee: &IdentityHash,
    ) -> Option<TrustScore> {
        self.received_edges
            .get(truster)
            .and_then(|inner| inner.get(trustee))
            .map(|e| e.score)
    }

    pub fn received_edges_for(&self, trustee: &IdentityHash) -> Vec<TrustEdge> {
        let mut result = Vec::new();
        for (truster, inner) in &self.received_edges {
            if let Some(edge) = inner.get(trustee) {
                result.push(TrustEdge {
                    truster: *truster,
                    trustee: *trustee,
                    score: edge.score,
                    updated_at: edge.updated_at,
                });
            }
        }
        result
    }

    // --- Resolution ---

    pub fn effective_score(&self, trustee: &IdentityHash) -> TrustScore {
        match self.local_score(trustee) {
            Some(score) => score,
            None => TrustScore::UNKNOWN,
        }
    }

    // --- Persistence ---

    pub fn serialize(&self) -> Result<Vec<u8>, TrustError> {
        let mut buf = Vec::new();
        buf.push(FORMAT_VERSION);
        let inner = postcard::to_allocvec(self)
            .map_err(|_| TrustError::SerializeError("postcard encode failed"))?;
        buf.extend_from_slice(&inner);
        Ok(buf)
    }

    pub fn deserialize(data: &[u8]) -> Result<Self, TrustError> {
        if data.is_empty() {
            return Err(TrustError::DeserializeError("empty data"));
        }
        if data[0] != FORMAT_VERSION {
            return Err(TrustError::DeserializeError("unsupported format version"));
        }
        postcard::from_bytes(&data[1..])
            .map_err(|_| TrustError::DeserializeError("postcard decode failed"))
    }
}
```

- [ ] **Step 4: Update lib.rs**

Add:

```rust
pub mod store;
pub use store::TrustStore;
```

- [ ] **Step 5: Run tests**

Run: `cargo test -p harmony-trust`
Expected: All 32 tests pass (9 score + 3 edge + 20 store).

- [ ] **Step 6: Commit**

```bash
git add crates/harmony-trust/
git commit -m "feat(trust): implement TrustStore with local/received edges and resolution"
```

---

### Task 4: Add `TrustLookup` trait

**Files:**
- Create: `crates/harmony-trust/src/lookup.rs`
- Modify: `crates/harmony-trust/src/lib.rs`

- [ ] **Step 1: Create lookup.rs with tests first**

Tests:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::score::TrustScore;
    use crate::store::TrustStore;

    const LOCAL: [u8; 16] = [0x01; 16];
    const ALICE: [u8; 16] = [0xAA; 16];

    #[test]
    fn trust_store_implements_lookup() {
        let mut store = TrustStore::new(LOCAL);
        store.set_score(&ALICE, TrustScore::from_dimensions(3, 3, 2, 1), 1000);

        // Use via trait
        let lookup: &dyn TrustLookup = &store;
        let score = lookup.score_for(&ALICE);
        assert_eq!(score.identity(), 3);
        assert_eq!(score.endorsement(), 1);
    }

    #[test]
    fn lookup_unknown_returns_zero() {
        let store = TrustStore::new(LOCAL);
        let lookup: &dyn TrustLookup = &store;
        assert_eq!(lookup.score_for(&ALICE), TrustScore::UNKNOWN);
    }
}
```

Implementation:

```rust
use harmony_identity::IdentityHash;

use crate::score::TrustScore;
use crate::store::TrustStore;

/// Trait for anything that can provide a trust score for an identity.
/// Consumers depend on this trait, not on TrustStore directly.
pub trait TrustLookup {
    /// Returns the effective trust score for an identity.
    /// Returns TrustScore::UNKNOWN for unscored identities.
    fn score_for(&self, trustee: &IdentityHash) -> TrustScore;
}

impl TrustLookup for TrustStore {
    fn score_for(&self, trustee: &IdentityHash) -> TrustScore {
        self.effective_score(trustee)
    }
}
```

- [ ] **Step 2: Update lib.rs**

Add:

```rust
pub mod lookup;
pub use lookup::TrustLookup;
```

- [ ] **Step 3: Run tests**

Run: `cargo test -p harmony-trust`
Expected: All 34 tests pass (9 + 3 + 20 + 2).

- [ ] **Step 4: Commit**

```bash
git add crates/harmony-trust/
git commit -m "feat(trust): add TrustLookup trait for consumer decoupling"
```

---

### Task 5: Final quality gate

- [ ] **Step 1: Full workspace test**

Run: `cargo test --workspace`
Expected: All tests pass (365+ existing + ~34 new).

- [ ] **Step 2: Format check**

Run: `cargo fmt --all -- --check`
Expected: No formatting issues.

- [ ] **Step 3: Clippy**

Run: `cargo clippy --workspace`
Expected: Clean.
