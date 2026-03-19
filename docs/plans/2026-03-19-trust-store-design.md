# Trust Store Design: Canonical Trust Scoring for Harmony

**Date:** 2026-03-19
**Status:** Approved
**Scope:** `harmony-trust` crate
**Supersedes:** `docs/plans/2026-03-07-trust-network-design.md` (client-side trust design — this spec defines the canonical core types that design was anticipating)

## Overview

Harmony needs a canonical, identity-agnostic trust scoring system that any
crate in the ecosystem can consume. Currently, trust is scattered across
domain-specific implementations (harmony-browser's `TrustPolicy`,
harmony-mail's `TrustMetrics`, harmony-kitri's `TrustTier`) with
incompatible score formats and no shared types.

This design introduces `harmony-trust`: a pure-data, `no_std`-compatible
crate providing:

1. **`TrustScore`** — an 8-bit type encoding four 2-bit dimensions
2. **`TrustEdge`** — a directional trust assessment between two identities
3. **`TrustStore`** — local + received trust graph with resolution chain
4. **`TrustLookup`** — a trait for consumers to query scores without
   depending on the store internals

## Design Philosophy

### Identity-Agnostic

Trust scores apply uniformly to any identity — human, AI agent, organization,
infrastructure node, distributed consciousness. What matters is authenticity
and behavior, not entity type. An identity is an identity.

### Cooperation by Default

Aligned with long-term prisoner's dilemma dynamics: the default posture is
cooperation. A verified identity with no evidence of defection should be
included. Score `0x00` means "no information" (not "distrust"), and the
system does not penalize the unknown.

### Exclusion Before Inclusion

The bit ordering (MSB to LSB: Identity → Compliance → Association →
Endorsement) follows the inclusion-exclusion paradox: groups are more
inclusive when you first ask "who should be kept out?" before "who should
be included?" This produces better outcomes for community health.

### Identity as MSB Enables Revocation

Identity occupying the most-significant bits serves a second purpose:
key compromise revocation. If evidence emerges that a key has been
compromised (e.g., a revocation published by a co-identity or root key),
the Identity dimension drops to `00` — and because it's the MSB, this
dominates the numerical score regardless of the other dimensions.

A compromised key with an excellent Compliance/Association/Endorsement
record is actually the most dangerous — it's a "wolf in sheep's clothing,"
the first target an attacker would puppet. The good reputation makes the
compromise MORE threatening, not less. Zeroing the MSBs reflects this:
the score collapses even if the other dimensions remain high, because
you can no longer be sure *who* is behind the actions.

Systems that compare trust scores numerically (e.g., thresholds, sorting)
automatically get revocation-aware behavior: a score of `0b00_11_11_11`
(0x3F — compromised key, exemplary everything else) sorts below
`0b01_00_00_00` (0x40 — weakly verified, nothing else). This is the
correct ordering.

### Domain-Specific Systems Are Consumers, Not Owners

harmony-mail's SMTP trust metrics, harmony-kitri's execution tiers, and
harmony-browser's rendering decisions are all domain-specific policies
that may consume `TrustScore` as one input among many. They do not dictate
the trust store's design — particularly mail, which is a Web2 adapter and
should flex to Harmony's model, not the other way around.

## Architecture

```
harmony-identity
    └── harmony-trust  (TrustScore, TrustEdge, TrustStore, TrustLookup trait)
```

Pure data crate. No I/O, no networking. `no_std`-compatible using
`hashbrown` for maps and `postcard` for serialization. Depends only on
`harmony-identity` for `IdentityHash`.

### Relationship to Other Crates

| Crate | Relationship to harmony-trust |
|-------|-------------------------------|
| harmony-contacts | Queries trust via `TrustLookup` trait. Does NOT embed scores. |
| harmony-peers | May factor trust into reconnection priority (future). |
| harmony-browser | Can replace ad hoc `HashMap<[u8;16], u8>` with `TrustLookup` (future). |
| harmony-kitri | Can derive `TrustTier` from `TrustScore` dimensions (future). |
| harmony-mail | Independent domain. May consume `TrustScore` for Harmony-origin mail, keeps its own `TrustMetrics` for SMTP gateways. |

No existing crate refactoring is in scope. The `TrustLookup` trait enables
incremental adoption.

## Core Types

### TrustScore

```rust
/// An 8-bit trust score encoding four orthogonal 2-bit dimensions.
///
/// Bit layout (MSB → LSB):
///   [7:6] Identity    — Is this entity verifiably who they claim to be?
///                       Have you verified them personally?
///   [5:4] Compliance  — What is their standing? Clean record, or
///                       evidence of rule-breaking without atonement?
///                       (default: unknown, cooperation assumed)
///   [3:2] Association — Do we actively want to include/associate with
///                       them? How much do we like them?
///   [1:0] Endorsement — Do we invest social capital in them? Defer to
///                       them? Elevate them to leadership/privileges?
///
/// Ordering rationale: exclusion before inclusion (the inclusion-exclusion
/// paradox — groups are more inclusive when you ask "who should be kept
/// out?" before "who should be included?"). This supports a
/// cooperation-by-default posture aligned with long-term prisoner's
/// dilemma dynamics.
///
/// A score of 0x00 means "no information" (not "distrust"). Unpublished
/// scores are presumed 0x00 by other nodes on the network.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TrustScore(u8);

impl TrustScore {
    pub fn new(raw: u8) -> Self { Self(raw) }
    pub fn raw(self) -> u8 { self.0 }

    /// Dimension accessors (each returns 0-3).
    pub fn identity(self) -> u8    { (self.0 >> 6) & 0x03 }
    pub fn compliance(self) -> u8  { (self.0 >> 4) & 0x03 }
    pub fn association(self) -> u8 { (self.0 >> 2) & 0x03 }
    pub fn endorsement(self) -> u8 { self.0 & 0x03 }

    /// Build a score from individual dimensions (each clamped to 0-3).
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
            | (endorsement & 0x03)
        )
    }

    /// No information — default for unknown identities.
    pub const UNKNOWN: Self = Self(0x00);
}
```

### What the dimension values mean

Each dimension is 2 bits (0-3):

| Value | Identity | Compliance | Association | Endorsement |
|-------|----------|------------|-------------|-------------|
| 0 | Unknown/unverified | Unknown | No preference | No endorsement |
| 1 | Weakly verified (indirect) | Minor concerns | Mild preference | Mild endorsement |
| 2 | Verified (checked credentials) | Good standing | Want to associate | Active endorsement |
| 3 | Personally verified | Exemplary record | Strong bond | Full social capital investment |

### TrustEdge

```rust
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

## TrustStore

The local node's view of the trust graph. Stores two categories of edges:

### Local edges

Scores assigned by the node's own identity — the user's subjective
assessments. These are authoritative and mutable.

### Received edges

Scores published by other identities on the network. Read-only locally,
ingested via `receive_edge()`. Used for future transitive trust computation
(EigenTrust). When multiple updates arrive for the same (truster, trustee)
pair, the latest `updated_at` wins.

### Internal structure

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrustStore {
    /// This node's identity (the truster for local edges).
    local_identity: IdentityHash,
    /// Scores this node assigns to other identities.
    local_edges: HashMap<IdentityHash, LocalEdge>,
    /// Scores received from other identities.
    /// Outer key: truster. Inner key: trustee.
    received_edges: HashMap<IdentityHash, HashMap<IdentityHash, ReceivedEdge>>,
}

// LocalEdge and ReceivedEdge are intentionally separate types despite
// identical fields. ReceivedEdge may gain a `signature` field when
// Zenoh publishing is implemented; LocalEdge will not.

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
```

### API

```rust
impl TrustStore {
    pub fn new(local_identity: IdentityHash) -> Self;

    // --- Local edges (your own assessments) ---

    /// Set or update your trust score for an identity.
    pub fn set_score(&mut self, trustee: &IdentityHash, score: TrustScore, now: u64);

    /// Remove your trust score for an identity. Returns the old score.
    pub fn remove_score(&mut self, trustee: &IdentityHash) -> Option<TrustScore>;

    /// Get your local score for an identity, if set.
    pub fn local_score(&self, trustee: &IdentityHash) -> Option<TrustScore>;

    /// Iterate all local edges (for publishing to the network).
    pub fn local_edges(&self) -> impl Iterator<Item = TrustEdge>;

    // --- Received edges (other identities' published scores) ---

    /// Ingest a trust edge received from the network.
    /// Updates existing edge if newer, ignores if stale.
    /// Silently ignores edges where `truster == self.local_identity`
    /// (your own scores echoed back from the network).
    pub fn receive_edge(&mut self, edge: TrustEdge);

    /// Get a specific received score.
    pub fn received_score(
        &self,
        truster: &IdentityHash,
        trustee: &IdentityHash,
    ) -> Option<TrustScore>;

    /// All received scores for a given trustee (from all trusters).
    /// Allocates a Vec because TrustEdge must be constructed from the
    /// nested map structure. For the EigenTrust future work, an inverted
    /// index (trustee → trusters) could make this O(k) instead of O(n).
    pub fn received_edges_for(
        &self,
        trustee: &IdentityHash,
    ) -> Vec<TrustEdge>;

    // --- Resolution ---

    /// The effective trust score for an identity.
    /// Resolution chain:
    ///   1. Local edge (if exists)
    ///   2. TrustScore::UNKNOWN (0x00)
    /// Future: step 2 becomes EigenTrust transitive computation
    /// using received edges.
    pub fn effective_score(&self, trustee: &IdentityHash) -> TrustScore;

    // --- Persistence ---

    pub fn serialize(&self) -> Result<Vec<u8>, TrustError>;
    pub fn deserialize(data: &[u8]) -> Result<Self, TrustError>;
}
```

## TrustLookup Trait

```rust
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

Returns `TrustScore` directly (not `Option`). Unknown identities get
`TrustScore::UNKNOWN`. This keeps consumer code simple.

## Error Types

```rust
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TrustError {
    /// Serialization failed.
    SerializeError(&'static str),
    /// Deserialization failed.
    DeserializeError(&'static str),
}
```

`Display` impl. `std::error::Error` under `std` feature. Same pattern as
`ContactError`.

## Zenoh Publishing (future, informing the design)

The namespace `/trust/<truster-id>/<trustee-id>` maps directly to the
data model:

- **Publishing:** Iterate `local_edges()`, publish each to
  `/trust/<my-id>/<trustee-id>` with the `TrustScore` byte + timestamp
  as payload.
- **Subscribing:** Subscribe to `/trust/*/<my-id>` to see what others
  think of you, or `/trust/<friend-id>/*` to collect a friend's scores
  for transitive computation.
- **Unpublished = 0x00:** If no message exists at a topic, the score is
  presumed `TrustScore::UNKNOWN`.

The store's `receive_edge()` is the ingestion point. A future Zenoh
integration would subscribe, deserialize, and call `receive_edge()`.
None of this is in scope for this bead — the data model is ready for it.

## `no_std` Compatibility

Same pattern as `harmony-contacts`:
- `#![cfg_attr(not(feature = "std"), no_std)]` with `extern crate alloc`
- `hashbrown` for HashMap (with `serde` feature)
- `postcard` for serialization with format version byte prefix
- `serde` with `derive` + `alloc` features

## What This Crate Does NOT Do

- No EigenTrust computation (future bead — data model supports it via
  `received_edges`)
- No Zenoh publishing/subscribing (future integration)
- No contact store integration (contacts queries trust store, not vice versa)
- No domain-specific policy (browser rendering decisions, SMTP metrics,
  WASM sandboxing — those stay in their respective crates)
- No identity verification (harmony-identity's job)
- No existing crate refactoring (consumers adopt `TrustLookup` incrementally)

## Testing Strategy

- **TrustScore:** dimension accessors round-trip, bit ordering verification,
  `from_dimensions` boundary clamping, `UNKNOWN` constant is 0x00,
  specific known byte values map to expected dimensions
- **TrustStore local edges:** set/get/remove scores, overwrite existing,
  iterate local edges
- **TrustStore received edges:** ingest edge, newer timestamp wins,
  stale edge rejected, self-edge ignored (truster == local_identity),
  query by (truster, trustee), query all edges for a trustee
- **TrustStore resolution:** local edge wins over received, unknown
  returned when no local edge, self-trust (truster == trustee)
- **TrustLookup:** TrustStore implements trait correctly
- **Persistence:** serialize/deserialize round-trip, corrupt data handling,
  empty store, format version check
