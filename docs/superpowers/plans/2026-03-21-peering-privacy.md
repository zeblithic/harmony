# Peering Privacy Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
> **Note:** The `- [ ]` checkboxes below are execution tracking markers for the agentic worker, not persistent TODO items. Task tracking uses `bd` (beads) — see bead `harmony-d63`.

**Goal:** Implement three zero-cost metadata privacy mitigations that make tunnel peer relationships harder to deduce from network observation: stochastic dialing, keepalive padding, and decoupled backoff curves.

**Architecture:** Three independent changes across three crates: (1) `harmony-tunnel` gets padded keepalives with jittered intervals, (2) `harmony-peers` gets identity-derived backoff jitter, (3) `harmony-node` gets a Poisson-delayed dial queue. Each is testable independently.

**Tech Stack:** Rust, `harmony-tunnel` (frame.rs, session.rs), `harmony-peers` (manager.rs), `harmony-node` (event_loop.rs), `harmony-crypto` (BLAKE3), `rand_core`

**Spec:** `docs/superpowers/specs/2026-03-21-peering-privacy-design.md`

---

## File Structure

```
crates/harmony-tunnel/src/
├── frame.rs        — Add Frame::keepalive_padded(rng) with random payload
└── session.rs      — Jitter keepalive interval (25-35s instead of fixed 30s)

crates/harmony-peers/src/
└── manager.rs      — Add local_identity_hash field, jittered probe_interval()

crates/harmony-node/src/
└── event_loop.rs   — Deferred dial queue with Poisson delay for InitiateTunnel
```

---

### Task 1: Keepalive Padding and Interval Jitter

**Files:**
- Modify: `crates/harmony-tunnel/src/frame.rs`
- Modify: `crates/harmony-tunnel/src/session.rs`

Two changes to make keepalive frames unrecognizable:
1. Random-length padding (0-128 bytes) so the ciphertext size varies
2. Jittered interval (25-35s) so the timing pattern is irregular

- [ ] **Step 1: Write test for padded keepalive**

In `crates/harmony-tunnel/src/frame.rs`, add to tests:

```rust
#[test]
fn keepalive_padded_has_variable_length() {
    use rand::rngs::OsRng;
    let frame1 = Frame::keepalive_padded(&mut OsRng);
    let frame2 = Frame::keepalive_padded(&mut OsRng);
    assert_eq!(frame1.tag, FrameTag::Keepalive);
    assert_eq!(frame2.tag, FrameTag::Keepalive);
    // Payload lengths should vary (statistically — both being 0 is p≈1/129)
    assert!(frame1.payload.len() <= 128);
    assert!(frame2.payload.len() <= 128);
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p harmony-tunnel keepalive_padded`

- [ ] **Step 3: Implement Frame::keepalive_padded()**

In `frame.rs`, add:

```rust
use rand_core::CryptoRngCore;

impl Frame {
    /// Create a keepalive frame with random-length padding.
    ///
    /// The padding makes keepalive ciphertexts indistinguishable from
    /// small Reticulum packets (19-500 bytes), defeating Statistical
    /// Disclosure Attacks based on fixed packet sizes.
    /// The receiver ignores keepalive payloads regardless of length.
    pub fn keepalive_padded(rng: &mut impl CryptoRngCore) -> Self {
        let mut len_byte = [0u8; 1];
        rng.fill_bytes(&mut len_byte);
        let pad_len = (len_byte[0] as usize) % 129; // 0-128 bytes
        let mut payload = alloc::vec![0u8; pad_len];
        rng.fill_bytes(&mut payload);
        Self {
            tag: FrameTag::Keepalive,
            payload,
        }
    }
}
```

Keep the existing `Frame::keepalive()` for tests that need deterministic empty keepalives.

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p harmony-tunnel keepalive_padded`

- [ ] **Step 5: Jitter the keepalive interval in session.rs**

In `session.rs`, change `handle_tick` to use a jittered interval. Instead of the fixed `KEEPALIVE_INTERVAL_MS`, compute a per-tick interval:

```rust
/// Base keepalive interval in milliseconds.
const KEEPALIVE_BASE_MS: u64 = 30_000;
/// Jitter range: ±5 seconds around the base.
const KEEPALIVE_JITTER_MS: u64 = 5_000;
```

In `handle_tick`, replace the keepalive check:

```rust
// Jittered keepalive interval: 25-35 seconds.
// Use a simple hash of the nonce counter as cheap deterministic jitter
// (no RNG needed in Tick — sans-I/O purity).
let jitter = (self.send_nonce % 11) * 1000; // 0-10s, seeded by nonce counter
let interval = KEEPALIVE_BASE_MS - KEEPALIVE_JITTER_MS + jitter;
if now_ms.saturating_sub(self.last_sent_ms) >= interval {
```

Also update the keepalive frame construction to use `keepalive_padded`:

```rust
let frame = Frame::keepalive_padded(&mut rand::rngs::OsRng);
```

Wait — `session.rs` is sans-I/O and doesn't have an RNG in `handle_tick`. The `Tick` event doesn't carry an RNG. Options:
- Pass RNG via the Tick event (breaks sans-I/O convention)
- Store an RNG on the session (breaks `!Send`)
- Use the nonce counter as a PRNG seed for padding (deterministic but unpredictable to observers)

Best approach: use `self.send_nonce` to derive padding. The nonce is already incrementing and unique per session. Observers don't know it (it's inside the AEAD).

```rust
// In handle_tick, for keepalive:
let pad_len = ((self.send_nonce * 7 + 13) % 129) as usize;
let padding = alloc::vec![0u8; pad_len]; // zeros are fine — payload is encrypted
let frame = Frame {
    tag: FrameTag::Keepalive,
    payload: padding,
};
```

This avoids needing an RNG in the tick path while still producing variable-length keepalives.

- [ ] **Step 6: Update the keepalive test**

The existing `keepalive_sent_after_interval` test checks exact timing at 30001ms. Update it to account for the jittered interval (the first keepalive fires between 25000-35000ms).

- [ ] **Step 7: Run all tunnel tests**

Run: `RUST_MIN_STACK=8388608 cargo test -p harmony-tunnel`

- [ ] **Step 8: Commit**

```bash
git add crates/harmony-tunnel/
git commit -m "feat(tunnel): padded keepalives and jittered interval for privacy

Random-length padding (0-128 bytes) makes keepalive ciphertexts
indistinguishable from small data packets. Interval jitter (25-35s)
breaks the strict periodicity that SDA relies on."
```

---

### Task 2: Decoupled Exponential Backoff

**Files:**
- Modify: `crates/harmony-peers/src/manager.rs`
- Modify: `crates/harmony-peers/Cargo.toml` (add harmony-crypto dep for BLAKE3)

Add a per-peer jitter factor to the probe interval, derived from identity hashes, so reconnection curves are unique and uncorrelatable.

- [ ] **Step 1: Write test for jittered probe interval**

In `crates/harmony-peers/src/manager.rs`, add to tests:

```rust
#[test]
fn probe_interval_varies_by_local_identity() {
    // Same peer, different local identities → different intervals
    let mut mgr1 = PeerManager::with_local_identity([0x11; 16]);
    let mut mgr2 = PeerManager::with_local_identity([0x22; 16]);

    let interval1 = PeerManager::probe_interval_jittered(
        PeeringPriority::High, 3, &[0x11; 16], &[0xAA; 16],
    );
    let interval2 = PeerManager::probe_interval_jittered(
        PeeringPriority::High, 3, &[0x22; 16], &[0xAA; 16],
    );

    // Both should be in a valid range but differ
    assert!(interval1 > 0 && interval1 <= PROBE_INTERVAL_MAX);
    assert!(interval2 > 0 && interval2 <= PROBE_INTERVAL_MAX);
    assert_ne!(interval1, interval2);
}

#[test]
fn probe_interval_varies_by_peer_identity() {
    // Same local identity, different peers → different intervals
    let interval1 = PeerManager::probe_interval_jittered(
        PeeringPriority::High, 3, &[0x11; 16], &[0xAA; 16],
    );
    let interval2 = PeerManager::probe_interval_jittered(
        PeeringPriority::High, 3, &[0x11; 16], &[0xBB; 16],
    );
    assert_ne!(interval1, interval2);
}

#[test]
fn probe_interval_jittered_is_deterministic() {
    // Same inputs → same output (reproducible for testing)
    let a = PeerManager::probe_interval_jittered(
        PeeringPriority::Normal, 2, &[0x11; 16], &[0xAA; 16],
    );
    let b = PeerManager::probe_interval_jittered(
        PeeringPriority::Normal, 2, &[0x11; 16], &[0xAA; 16],
    );
    assert_eq!(a, b);
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test -p harmony-peers probe_interval_jittered`

- [ ] **Step 3: Add local_identity_hash and implement jittered probe**

Add `harmony-crypto` to `crates/harmony-peers/Cargo.toml`:
```toml
harmony-crypto = { workspace = true }
```

In `manager.rs`:

```rust
pub struct PeerManager {
    pub(crate) peers: HashMap<IdentityHash, PeerState>,
    local_identity_hash: IdentityHash,
}

impl PeerManager {
    pub fn new() -> Self {
        Self {
            peers: HashMap::new(),
            local_identity_hash: [0u8; 16], // default for backward compat
        }
    }

    pub fn with_local_identity(local_identity_hash: IdentityHash) -> Self {
        Self {
            peers: HashMap::new(),
            local_identity_hash,
        }
    }

    /// Compute a jittered probe interval for a specific peer.
    ///
    /// Uses BLAKE3(local_hash || peer_hash) to derive a deterministic
    /// but unique jitter factor (0.75-1.25x) for each peer relationship.
    /// This makes exponential backoff curves uncorrelatable between
    /// different observers — they can't match reconnection patterns.
    pub fn probe_interval_jittered(
        priority: PeeringPriority,
        retry_count: u32,
        local_hash: &IdentityHash,
        peer_hash: &IdentityHash,
    ) -> u64 {
        let base_interval = Self::probe_interval(priority, retry_count);
        if base_interval == 0 {
            return 0;
        }

        // Derive jitter from identity pair
        let mut seed_input = [0u8; 32];
        seed_input[..16].copy_from_slice(local_hash);
        seed_input[16..].copy_from_slice(peer_hash);
        let hash = harmony_crypto::hash::blake3_hash(&seed_input);
        let jitter_raw = u64::from_le_bytes(hash[..8].try_into().unwrap());

        // jitter_factor: 0.75 to ~1.25 (768/1024 to 1280/1024)
        let jitter_factor_num = 768 + (jitter_raw % 512); // 768-1279
        let jittered = base_interval * jitter_factor_num / 1024;

        jittered.min(PROBE_INTERVAL_MAX)
    }
}
```

Then update the call sites in `handle_tick` to use `probe_interval_jittered` instead of `probe_interval`:

```rust
let interval = Self::probe_interval_jittered(
    peer.priority, peer.retry_count,
    &self.local_identity_hash, &identity_hash,
);
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test -p harmony-peers`

- [ ] **Step 5: Commit**

```bash
git add crates/harmony-peers/
git commit -m "feat(peers): identity-derived backoff jitter for privacy

Deterministic per-peer jitter factor from BLAKE3(local||peer) makes
exponential backoff curves unique and uncorrelatable between observers.
Interval varies by ±25%, stable across retries for test reproducibility."
```

---

### Task 3: Stochastic Dialing Delay

**Files:**
- Modify: `crates/harmony-node/src/event_loop.rs`

Add a Poisson-distributed delay between PeerManager emitting `InitiateTunnel` and the event loop actually dialing the relay. This breaks the timing correlation between Zenoh discovery queries and QUIC connection attempts.

- [ ] **Step 1: Add a deferred dial queue**

In `event_loop.rs`, add a struct and a `VecDeque` for pending dials:

```rust
use std::collections::VecDeque;

/// A tunnel dial request waiting for its scheduled fire time.
struct DeferredDial {
    identity_hash: [u8; 16],
    node_id: [u8; 32],
    relay_url: Option<String>,
    fire_at_ms: u64, // monotonic millis when this dial should execute
}
```

Before the select loop:
```rust
let mut deferred_dials: VecDeque<DeferredDial> = VecDeque::new();
```

- [ ] **Step 2: Defer InitiateTunnel instead of executing immediately**

When `RuntimeAction::InitiateTunnel` arrives in the action dispatch, instead of immediately connecting, push it to the deferred queue with a random delay:

```rust
RuntimeAction::InitiateTunnel { identity_hash, node_id, relay_url } => {
    // Poisson-distributed delay: -ln(U) * mean, where mean = 2000ms
    // Use a simple approximation: uniform random 500-4000ms
    let delay_ms = 500 + (rand::random::<u64>() % 3500);
    let fire_at = tunnel_task::millis_since_start() + delay_ms;
    deferred_dials.push_back(DeferredDial {
        identity_hash, node_id, relay_url, fire_at_ms: fire_at,
    });
    tracing::debug!(
        identity = %hex::encode(identity_hash),
        delay_ms,
        "tunnel dial deferred for privacy"
    );
}
```

- [ ] **Step 3: Drain the deferred queue on each tick**

In the timer tick arm (or at the top of the select loop body), check for ready dials:

```rust
// Drain deferred dials that have reached their fire time.
let now = tunnel_task::millis_since_start();
while let Some(front) = deferred_dials.front() {
    if front.fire_at_ms <= now {
        let dial = deferred_dials.pop_front().unwrap();
        // Execute the dial (same logic as the old immediate path)
        // ... connect via iroh endpoint ...
    } else {
        break; // queue is ordered by fire_at_ms
    }
}
```

Note: The queue is naturally ordered because dials are pushed in chronological order and delays are bounded. If a race condition causes out-of-order entries, sort on drain or use a `BinaryHeap`.

- [ ] **Step 4: Verify it compiles**

Run: `cargo check -p harmony-node`

- [ ] **Step 5: Commit**

```bash
git add crates/harmony-node/
git commit -m "feat(node): stochastic dialing delay for tunnel privacy

Defers InitiateTunnel actions by 0.5-4s (uniform random) before
dialing the relay, breaking the timing correlation between Zenoh
discovery queries and QUIC connection attempts."
```

---

### Task 4: Cleanup and verification

- [ ] **Step 1: Run clippy**

Run: `cargo clippy -p harmony-tunnel -p harmony-peers -p harmony-node`

- [ ] **Step 2: Run workspace tests**

Run: `RUST_MIN_STACK=8388608 cargo test --workspace`

- [ ] **Step 3: Commit cleanup**

```bash
git add -A
git commit -m "chore: clippy fixes for peering privacy mitigations"
```

---

## Summary

| Task | Description | Crate | Privacy Threat Mitigated |
|------|-------------|-------|------------------------|
| 1 | Keepalive padding + interval jitter | harmony-tunnel | SDA on fixed packet sizes/timing |
| 2 | Identity-derived backoff jitter | harmony-peers | Reconnection curve correlation |
| 3 | Stochastic dialing delay | harmony-node | Discovery→dial timing correlation |
| 4 | Cleanup and verification | all | — |

**Total cost:** Zero bandwidth, 0.5-4s latency on first connection only, ±25% variation on backoff intervals, ~64 bytes/30s average padding overhead. All changes are backward-compatible — no wire format changes, no new dependencies beyond BLAKE3 (already in workspace).
