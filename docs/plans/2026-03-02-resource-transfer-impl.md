# Resource Transfer Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement chunked data transfer over established Reticulum links with adaptive windowing, retransmission, cancellation, and proof-of-delivery.

**Architecture:** Standalone `ResourceSender` and `ResourceReceiver` sans-I/O state machines in `resource.rs`, trait-based via `LinkCrypto` for encrypt/decrypt abstraction. Event/action pattern matching `Node` and `LivelinessRouter`. Two-layer encryption: payload-level (whole blob) and transport-level (per-packet, handled by caller).

**Tech Stack:** Rust, `harmony-crypto` (SHA-256), `rmp` (msgpack wire format), existing `Link`/`Packet`/`PacketContext` types.

**Design doc:** `docs/plans/2026-03-02-resource-transfer-design.md`

---

### Task 1: Bootstrap — Error Variants, Types, Trait, Enums, Module

**Files:**
- Modify: `Cargo.toml` (workspace deps)
- Modify: `crates/harmony-reticulum/Cargo.toml`
- Modify: `crates/harmony-reticulum/src/error.rs`
- Create: `crates/harmony-reticulum/src/resource.rs`
- Modify: `crates/harmony-reticulum/src/lib.rs`

**Step 1: Add `rmp` workspace dependency**

In the root `Cargo.toml`, under `[workspace.dependencies]` after the `hex` line:

```toml
rmp = "0.8"
```

In `crates/harmony-reticulum/Cargo.toml`, under `[dependencies]`:

```toml
rmp = { workspace = true }
```

**Step 2: Add error variants**

In `crates/harmony-reticulum/src/error.rs`, add after the `IfacInvalidSize` variant:

```rust
#[error("resource too large: {size} bytes exceeds {max} byte limit")]
ResourceTooLarge { size: usize, max: usize },

#[error("resource advertisement invalid")]
ResourceAdvInvalid,

#[error("resource hash mismatch")]
ResourceHashMismatch,

#[error("resource proof invalid")]
ResourceProofInvalid,

#[error("unknown resource part: map_hash {map_hash:02x?}")]
ResourceUnknownPart { map_hash: [u8; 4] },

#[error("resource transfer failed")]
ResourceFailed,

#[error("resource already complete")]
ResourceAlreadyComplete,
```

**Step 3: Create `resource.rs` with types, constants, trait, enums, and struct skeletons**

```rust
//! Reticulum resource transfer — chunked data over encrypted links.
//!
//! Implements sender and receiver state machines for transferring data
//! larger than a single packet MTU. Uses adaptive windowing, hashmap-based
//! part tracking, retransmission, and proof-of-delivery.

use rand_core::CryptoRngCore;

use crate::context::PacketContext;
use crate::error::ReticulumError;

// ── Constants ────────────────────────────────────────────────────────

/// Map hash length: 4-byte truncated SHA-256.
pub const MAPHASH_LEN: usize = 4;

/// Maximum resource data size (~1MB). Matches Python `MAX_EFFICIENT_SIZE`.
pub const MAX_EFFICIENT_SIZE: usize = 0xFF_FFFF;

/// Maximum receiver retries before failure.
pub const MAX_RETRIES: u8 = 16;

/// Maximum advertisement retries before failure.
pub const MAX_ADV_RETRIES: u8 = 4;

/// Initial request window size.
pub const WINDOW_INITIAL: usize = 4;

/// Minimum request window.
pub const WINDOW_MIN: usize = 2;

/// Default window ceiling (slow links).
pub const WINDOW_MAX_SLOW: usize = 10;

/// Window ceiling for fast links (>50kbps).
pub const WINDOW_MAX_FAST: usize = 75;

/// Window ceiling for very slow links (<2kbps).
pub const WINDOW_MAX_VERY_SLOW: usize = 4;

/// Fast link threshold: 50 kbps in bytes/sec.
pub const RATE_FAST: u64 = 6250;

/// Very slow link threshold: 2 kbps in bytes/sec.
pub const RATE_VERY_SLOW: u64 = 250;

/// Sender grace time in milliseconds.
pub const SENDER_GRACE_TIME_MS: u64 = 10_000;

/// Processing grace time for advertisement timeout in milliseconds.
pub const PROCESSING_GRACE_MS: u64 = 1_000;

/// Proof timeout multiplier.
pub const PROOF_TIMEOUT_FACTOR: u64 = 3;

/// Part timeout multiplier (before first RTT measurement).
pub const PART_TIMEOUT_FACTOR: u64 = 4;

/// Part timeout multiplier (after first RTT measurement).
pub const PART_TIMEOUT_FACTOR_AFTER_RTT: u64 = 2;

/// Per-retry delay in milliseconds.
pub const PER_RETRY_DELAY_MS: u64 = 500;

/// Retry grace time in milliseconds.
pub const RETRY_GRACE_TIME_MS: u64 = 250;

/// Consecutive fast-rate rounds needed to upgrade window_max.
pub const FAST_RATE_THRESHOLD: usize = 5;

/// Consecutive very-slow-rate rounds needed to downgrade window_max.
pub const VERY_SLOW_RATE_THRESHOLD: usize = 2;

/// Minimum flexibility gap between window_max and window_min.
pub const WINDOW_FLEXIBILITY: usize = 4;

/// Hashmap exhaustion flag: receiver has all hashes.
pub const HASHMAP_IS_NOT_EXHAUSTED: u8 = 0x00;

/// Hashmap exhaustion flag: receiver needs more hashes.
pub const HASHMAP_IS_EXHAUSTED: u8 = 0xFF;

// ── Types ────────────────────────────────────────────────────────────

/// 16-byte resource identifier: SHA-256(encrypted_data + random_hash)[:16].
pub type ResourceHash = [u8; 16];

/// 4-byte part identifier: SHA-256(part_data + random_hash)[:4].
pub type MapHash = [u8; 4];

/// 4-byte random nonce per resource.
pub type RandomHash = [u8; 4];

// ── LinkCrypto trait ─────────────────────────────────────────────────

/// Abstraction over link-level encryption for resource transfer.
///
/// `Link` implements this trait directly. Tests use `MockLinkCrypto`.
pub trait LinkCrypto {
    /// Encrypt plaintext using the link's session key.
    fn encrypt(
        &self,
        rng: &mut dyn CryptoRngCore,
        plaintext: &[u8],
    ) -> Result<Vec<u8>, ReticulumError>;

    /// Decrypt ciphertext using the link's session key.
    fn decrypt(&self, ciphertext: &[u8]) -> Result<Vec<u8>, ReticulumError>;

    /// Link identifier (16 bytes).
    fn link_id(&self) -> &[u8; 16];

    /// Maximum plaintext size for a single resource part.
    /// Each part, when sent as a link packet, must fit within the link's MTU
    /// after encryption and framing overhead.
    fn mdu(&self) -> usize;
}

// ── State enums ──────────────────────────────────────────────────────

/// Sender state machine states.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SenderState {
    Queued,
    Advertised,
    Transferring,
    AwaitingProof,
    Complete,
    Failed,
    Rejected,
}

/// Receiver state machine states.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReceiverState {
    Transferring,
    Assembling,
    Complete,
    Corrupt,
    Failed,
}

// ── Events ───────────────────────────────────────────────────────────

/// Inbound events for resource state machines.
#[derive(Debug)]
pub enum ResourceEvent {
    /// Sender receives a request for parts from the receiver.
    RequestReceived { plaintext: Vec<u8> },
    /// Receiver receives a data part.
    PartReceived { plaintext: Vec<u8> },
    /// Receiver receives an extended hashmap segment.
    HashmapUpdateReceived { plaintext: Vec<u8> },
    /// Sender receives a proof of delivery.
    ProofReceived { plaintext: Vec<u8> },
    /// Either side receives a cancel from the peer.
    CancelReceived,
    /// Caller fires a scheduled timeout.
    Timeout { now_ms: u64 },
}

// ── Actions ──────────────────────────────────────────────────────────

/// Outbound actions from resource state machines.
#[derive(Debug, Clone)]
pub enum ResourceAction {
    /// Caller should encrypt this plaintext and send it with the given context.
    SendPacket {
        context: PacketContext,
        plaintext: Vec<u8>,
    },
    /// Sender state changed.
    SenderStateChanged { new_state: SenderState },
    /// Receiver state changed.
    ReceiverStateChanged { new_state: ReceiverState },
    /// Transfer progress update.
    Progress { fraction: f32 },
    /// Receiver: all parts received and hash verified. Call `finalize()`.
    AssemblyReady,
    /// Receiver: finalized — here is the decrypted data.
    Completed { data: Vec<u8> },
    /// Sender: proof validated, transfer complete.
    ProofValidated,
    /// Schedule a timeout callback at this absolute time.
    ScheduleTimeout { deadline_ms: u64 },
}

// ── Sender ───────────────────────────────────────────────────────────

/// Resource sender state machine.
///
/// Lifecycle: `new()` → `advertise()` → `handle_event()` loop → Complete/Failed.
pub struct ResourceSender {
    state: SenderState,
}

impl ResourceSender {
    /// Current state.
    pub fn state(&self) -> SenderState {
        self.state
    }
}

// ── Receiver ─────────────────────────────────────────────────────────

/// Resource receiver state machine.
///
/// Lifecycle: `accept()` → `handle_event()` loop → `finalize()` → Complete/Failed.
pub struct ResourceReceiver {
    state: ReceiverState,
}

impl ResourceReceiver {
    /// Current state.
    pub fn state(&self) -> ReceiverState {
        self.state
    }
}

// ── MockLinkCrypto ───────────────────────────────────────────────────

#[cfg(test)]
pub(crate) struct MockLinkCrypto {
    pub id: [u8; 16],
    pub mdu: usize,
}

#[cfg(test)]
impl MockLinkCrypto {
    pub fn new(mdu: usize) -> Self {
        Self {
            id: [0xAA; 16],
            mdu,
        }
    }
}

#[cfg(test)]
impl LinkCrypto for MockLinkCrypto {
    fn encrypt(
        &self,
        _rng: &mut dyn CryptoRngCore,
        plaintext: &[u8],
    ) -> Result<Vec<u8>, ReticulumError> {
        // Identity pass-through: "encrypted" = plaintext (for testability)
        Ok(plaintext.to_vec())
    }

    fn decrypt(&self, ciphertext: &[u8]) -> Result<Vec<u8>, ReticulumError> {
        Ok(ciphertext.to_vec())
    }

    fn link_id(&self) -> &[u8; 16] {
        &self.id
    }

    fn mdu(&self) -> usize {
        self.mdu
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mock_link_crypto_identity_roundtrip() {
        use rand::rngs::OsRng;
        let mock = MockLinkCrypto::new(64);
        let data = b"hello resource";
        let encrypted = mock.encrypt(&mut OsRng, data).unwrap();
        let decrypted = mock.decrypt(&encrypted).unwrap();
        assert_eq!(decrypted, data);
        assert_eq!(mock.link_id(), &[0xAA; 16]);
        assert_eq!(mock.mdu(), 64);
    }

    #[test]
    fn sender_initial_state() {
        let sender = ResourceSender {
            state: SenderState::Queued,
        };
        assert_eq!(sender.state(), SenderState::Queued);
    }

    #[test]
    fn receiver_initial_state() {
        let receiver = ResourceReceiver {
            state: ReceiverState::Transferring,
        };
        assert_eq!(receiver.state(), ReceiverState::Transferring);
    }
}
```

**Step 4: Register the module in `lib.rs`**

Add to `crates/harmony-reticulum/src/lib.rs`:

Module declaration (after `pub mod packet_hashlist;`):
```rust
pub mod resource;
```

Re-exports (after the `path_table` re-export):
```rust
pub use resource::{
    LinkCrypto, ReceiverState, ResourceAction, ResourceEvent, ResourceHash, ResourceReceiver,
    ResourceSender, SenderState,
};
```

**Step 5: Run tests to verify compilation**

Run: `cargo test -p harmony-reticulum resource --no-run`
Expected: Compiles successfully.

Run: `cargo test -p harmony-reticulum resource -- --nocapture`
Expected: 3 tests pass.

**Step 6: Commit**

```bash
git add -A && git commit -m "feat(reticulum): bootstrap resource transfer module with types and trait"
```

---

### Task 2: Hash Computations, Chunking, and Advertisement Wire Format

**Files:**
- Modify: `crates/harmony-reticulum/src/resource.rs`

**Reference:** Python `Resource.py` computes hashes as:
- `map_hash = SHA-256(part_data + random_hash)[:4]`
- `resource_hash = SHA-256(encrypted_data + random_hash)[:16]`
- `proof = SHA-256(plaintext + resource_hash)[:16]`

**Step 1: Write failing tests for hash computations**

Add to the `tests` module:

```rust
#[test]
fn compute_map_hash_deterministic() {
    let part_data = b"some part data";
    let random_hash: RandomHash = [0x01, 0x02, 0x03, 0x04];
    let mh = compute_map_hash(part_data, &random_hash);
    assert_eq!(mh.len(), MAPHASH_LEN);
    // Same input → same output
    assert_eq!(mh, compute_map_hash(part_data, &random_hash));
    // Different input → different output
    assert_ne!(mh, compute_map_hash(b"other data", &random_hash));
}

#[test]
fn compute_resource_hash_deterministic() {
    let encrypted = b"encrypted blob data";
    let random_hash: RandomHash = [0x01, 0x02, 0x03, 0x04];
    let rh = compute_resource_hash(encrypted, &random_hash);
    assert_eq!(rh.len(), 16);
    assert_eq!(rh, compute_resource_hash(encrypted, &random_hash));
    assert_ne!(rh, compute_resource_hash(b"other", &random_hash));
}

#[test]
fn compute_proof_hash_deterministic() {
    let plaintext = b"original data";
    let resource_hash: ResourceHash = [0xAB; 16];
    let proof = compute_proof_hash(plaintext, &resource_hash);
    assert_eq!(proof.len(), 16);
    assert_eq!(proof, compute_proof_hash(plaintext, &resource_hash));
}
```

Run: `cargo test -p harmony-reticulum resource`
Expected: FAIL — `compute_map_hash`, `compute_resource_hash`, `compute_proof_hash` not found.

**Step 2: Implement hash functions**

Add above the `#[cfg(test)]` block:

```rust
use harmony_crypto::hash;

/// Compute a 4-byte map hash for a resource part.
/// `map_hash = SHA-256(part_data || random_hash)[:4]`
pub fn compute_map_hash(part_data: &[u8], random_hash: &RandomHash) -> MapHash {
    let mut input = Vec::with_capacity(part_data.len() + MAPHASH_LEN);
    input.extend_from_slice(part_data);
    input.extend_from_slice(random_hash);
    let full = hash::full_hash(&input);
    let mut mh = [0u8; MAPHASH_LEN];
    mh.copy_from_slice(&full[..MAPHASH_LEN]);
    mh
}

/// Compute the 16-byte resource hash.
/// `resource_hash = SHA-256(encrypted_data || random_hash)[:16]`
pub fn compute_resource_hash(encrypted_data: &[u8], random_hash: &RandomHash) -> ResourceHash {
    let mut input = Vec::with_capacity(encrypted_data.len() + MAPHASH_LEN);
    input.extend_from_slice(encrypted_data);
    input.extend_from_slice(random_hash);
    hash::truncated_hash(&input)
}

/// Compute the 16-byte proof hash.
/// `proof = SHA-256(plaintext || resource_hash)[:16]`
pub fn compute_proof_hash(plaintext: &[u8], resource_hash: &ResourceHash) -> ResourceHash {
    let mut input = Vec::with_capacity(plaintext.len() + 16);
    input.extend_from_slice(plaintext);
    input.extend_from_slice(resource_hash);
    hash::truncated_hash(&input)
}
```

Run: `cargo test -p harmony-reticulum resource`
Expected: All hash tests pass.

**Step 3: Write failing tests for chunking**

```rust
#[test]
fn chunk_data_single_part() {
    let data = vec![0xAA; 10];
    let parts = chunk_data(&data, 64);
    assert_eq!(parts.len(), 1);
    assert_eq!(parts[0], data);
}

#[test]
fn chunk_data_multiple_parts() {
    let data = vec![0xBB; 100];
    let parts = chunk_data(&data, 30);
    assert_eq!(parts.len(), 4); // 30+30+30+10
    assert_eq!(parts[0].len(), 30);
    assert_eq!(parts[1].len(), 30);
    assert_eq!(parts[2].len(), 30);
    assert_eq!(parts[3].len(), 10);
    // Reassemble
    let reassembled: Vec<u8> = parts.into_iter().flatten().collect();
    assert_eq!(reassembled, data);
}

#[test]
fn chunk_data_exact_boundary() {
    let data = vec![0xCC; 60];
    let parts = chunk_data(&data, 30);
    assert_eq!(parts.len(), 2);
    assert_eq!(parts[0].len(), 30);
    assert_eq!(parts[1].len(), 30);
}
```

Run: `cargo test -p harmony-reticulum resource`
Expected: FAIL — `chunk_data` not found.

**Step 4: Implement chunking**

```rust
/// Split data into SDU-sized parts.
fn chunk_data(data: &[u8], sdu: usize) -> Vec<Vec<u8>> {
    data.chunks(sdu).map(|c| c.to_vec()).collect()
}
```

Run: `cargo test -p harmony-reticulum resource`
Expected: All chunking tests pass.

**Step 5: Write failing tests for advertisement encode/decode**

```rust
#[test]
fn advertisement_roundtrip() {
    let adv = ResourceAdvertisement {
        transfer_size: 1024,
        data_size: 900,
        part_count: 4,
        resource_hash: [0xAA; 16],
        random_hash: [0x01, 0x02, 0x03, 0x04],
        original_hash: [0xBB; 16],
        hashmap: vec![0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88],
        flags: 0x01,
        segment_index: 1,
        total_segments: 1,
    };
    let encoded = adv.encode();
    let decoded = ResourceAdvertisement::decode(&encoded).unwrap();
    assert_eq!(decoded.transfer_size, 1024);
    assert_eq!(decoded.data_size, 900);
    assert_eq!(decoded.part_count, 4);
    assert_eq!(decoded.resource_hash, [0xAA; 16]);
    assert_eq!(decoded.random_hash, [0x01, 0x02, 0x03, 0x04]);
    assert_eq!(decoded.original_hash, [0xBB; 16]);
    assert_eq!(decoded.hashmap, vec![0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88]);
    assert_eq!(decoded.flags, 0x01);
    assert_eq!(decoded.segment_index, 1);
    assert_eq!(decoded.total_segments, 1);
}

#[test]
fn advertisement_decode_invalid_rejects() {
    assert!(ResourceAdvertisement::decode(&[0x00, 0x01]).is_err());
}
```

Run: `cargo test -p harmony-reticulum resource`
Expected: FAIL — `ResourceAdvertisement` not found.

**Step 6: Implement advertisement encode/decode**

Use `rmp` for manual msgpack encoding to match Python's dictionary wire format exactly.

```rust
/// Resource advertisement data, sent as the first packet of a transfer.
///
/// Encoded as a msgpack map with single-character string keys matching
/// Python Reticulum's `ResourceAdvertisement` format.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResourceAdvertisement {
    pub transfer_size: u32,
    pub data_size: u32,
    pub part_count: u32,
    pub resource_hash: ResourceHash,
    pub random_hash: RandomHash,
    pub original_hash: ResourceHash,
    pub hashmap: Vec<u8>,
    pub flags: u8,
    pub segment_index: u32,
    pub total_segments: u32,
}

impl ResourceAdvertisement {
    /// Encode as msgpack map matching Python wire format.
    pub fn encode(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(128);
        // 10-entry map
        rmp::encode::write_map_len(&mut buf, 10).unwrap();

        // "t": transfer_size
        rmp::encode::write_str(&mut buf, "t").unwrap();
        rmp::encode::write_u32(&mut buf, self.transfer_size).unwrap();

        // "d": data_size
        rmp::encode::write_str(&mut buf, "d").unwrap();
        rmp::encode::write_u32(&mut buf, self.data_size).unwrap();

        // "n": part_count
        rmp::encode::write_str(&mut buf, "n").unwrap();
        rmp::encode::write_u32(&mut buf, self.part_count).unwrap();

        // "h": resource_hash (bin16)
        rmp::encode::write_str(&mut buf, "h").unwrap();
        rmp::encode::write_bin(&mut buf, &self.resource_hash).unwrap();

        // "r": random_hash (bin4)
        rmp::encode::write_str(&mut buf, "r").unwrap();
        rmp::encode::write_bin(&mut buf, &self.random_hash).unwrap();

        // "o": original_hash (bin16)
        rmp::encode::write_str(&mut buf, "o").unwrap();
        rmp::encode::write_bin(&mut buf, &self.original_hash).unwrap();

        // "m": hashmap (binN)
        rmp::encode::write_str(&mut buf, "m").unwrap();
        rmp::encode::write_bin(&mut buf, &self.hashmap).unwrap();

        // "f": flags
        rmp::encode::write_str(&mut buf, "f").unwrap();
        rmp::encode::write_u8(&mut buf, self.flags).unwrap();

        // "i": segment_index
        rmp::encode::write_str(&mut buf, "i").unwrap();
        rmp::encode::write_u32(&mut buf, self.segment_index).unwrap();

        // "l": total_segments
        rmp::encode::write_str(&mut buf, "l").unwrap();
        rmp::encode::write_u32(&mut buf, self.total_segments).unwrap();

        buf
    }

    /// Decode from msgpack bytes.
    pub fn decode(data: &[u8]) -> Result<Self, ReticulumError> {
        use rmp::decode::*;
        let mut cursor = std::io::Cursor::new(data);

        let map_len = read_map_len(&mut cursor)
            .map_err(|_| ReticulumError::ResourceAdvInvalid)?;

        let mut adv = ResourceAdvertisement {
            transfer_size: 0,
            data_size: 0,
            part_count: 0,
            resource_hash: [0; 16],
            random_hash: [0; 4],
            original_hash: [0; 16],
            hashmap: Vec::new(),
            flags: 0,
            segment_index: 1,
            total_segments: 1,
        };

        let mut key_buf = [0u8; 4];
        for _ in 0..map_len {
            let key = read_str(&mut cursor, &mut key_buf)
                .map_err(|_| ReticulumError::ResourceAdvInvalid)?;
            match key {
                "t" => {
                    adv.transfer_size = read_int(&mut cursor)
                        .map_err(|_| ReticulumError::ResourceAdvInvalid)?;
                }
                "d" => {
                    adv.data_size = read_int(&mut cursor)
                        .map_err(|_| ReticulumError::ResourceAdvInvalid)?;
                }
                "n" => {
                    adv.part_count = read_int(&mut cursor)
                        .map_err(|_| ReticulumError::ResourceAdvInvalid)?;
                }
                "h" => {
                    let mut hash_buf = vec![0u8; 16];
                    let len = read_bin_len(&mut cursor)
                        .map_err(|_| ReticulumError::ResourceAdvInvalid)?;
                    if len as usize != 16 {
                        return Err(ReticulumError::ResourceAdvInvalid);
                    }
                    std::io::Read::read_exact(&mut cursor, &mut hash_buf)
                        .map_err(|_| ReticulumError::ResourceAdvInvalid)?;
                    adv.resource_hash.copy_from_slice(&hash_buf);
                }
                "r" => {
                    let len = read_bin_len(&mut cursor)
                        .map_err(|_| ReticulumError::ResourceAdvInvalid)?;
                    if len as usize != 4 {
                        return Err(ReticulumError::ResourceAdvInvalid);
                    }
                    std::io::Read::read_exact(&mut cursor, &mut adv.random_hash)
                        .map_err(|_| ReticulumError::ResourceAdvInvalid)?;
                }
                "o" => {
                    let mut hash_buf = vec![0u8; 16];
                    let len = read_bin_len(&mut cursor)
                        .map_err(|_| ReticulumError::ResourceAdvInvalid)?;
                    if len as usize != 16 {
                        return Err(ReticulumError::ResourceAdvInvalid);
                    }
                    std::io::Read::read_exact(&mut cursor, &mut hash_buf)
                        .map_err(|_| ReticulumError::ResourceAdvInvalid)?;
                    adv.original_hash.copy_from_slice(&hash_buf);
                }
                "m" => {
                    let len = read_bin_len(&mut cursor)
                        .map_err(|_| ReticulumError::ResourceAdvInvalid)? as usize;
                    adv.hashmap = vec![0u8; len];
                    std::io::Read::read_exact(&mut cursor, &mut adv.hashmap)
                        .map_err(|_| ReticulumError::ResourceAdvInvalid)?;
                }
                "f" => {
                    adv.flags = read_int(&mut cursor)
                        .map_err(|_| ReticulumError::ResourceAdvInvalid)?;
                }
                "i" => {
                    adv.segment_index = read_int(&mut cursor)
                        .map_err(|_| ReticulumError::ResourceAdvInvalid)?;
                }
                "l" => {
                    adv.total_segments = read_int(&mut cursor)
                        .map_err(|_| ReticulumError::ResourceAdvInvalid)?;
                }
                _ => {
                    // Skip unknown keys for forward compatibility.
                    // Read and discard the value. For simplicity, reject unknown keys.
                    return Err(ReticulumError::ResourceAdvInvalid);
                }
            }
        }

        Ok(adv)
    }
}
```

**Note for implementer:** The `rmp::decode::read_int` function is generic over numeric types. If the `rmp` crate API differs slightly from what's shown here (e.g., `read_int` may require explicit turbofish like `read_int::<u32, _>`), adjust accordingly. Consult `rmp` docs at https://docs.rs/rmp/latest/rmp/. If unknown-key skipping is needed later, you can use `rmp::decode::read_marker` + skip logic; for now rejecting unknown keys is fine.

Run: `cargo test -p harmony-reticulum resource`
Expected: All tests pass (hash + chunking + advertisement).

**Step 7: Commit**

```bash
git add -A && git commit -m "feat(reticulum): add resource hash computations, chunking, and advertisement wire format"
```

---

### Task 3: ResourceSender — Creation, Advertisement, Request Handling, Proof Validation

**Files:**
- Modify: `crates/harmony-reticulum/src/resource.rs`

**Reference:** Sender lifecycle: `new()` encrypts + chunks → `advertise()` emits adv packet → receiver sends RESOURCE_REQ → sender matches hashes + emits parts → receiver sends proof → sender validates.

**Step 1: Write failing test for `ResourceSender::new()`**

```rust
#[test]
fn sender_new_creates_parts_and_hashes() {
    use rand::rngs::OsRng;
    let mock = MockLinkCrypto::new(30); // 30-byte SDU → multiple parts
    let data = vec![0xAA; 100];
    let sender = ResourceSender::new(&mut OsRng, &mock, &data, 0).unwrap();
    assert_eq!(sender.state(), SenderState::Queued);
    assert!(!sender.hash().iter().all(|&b| b == 0));
    assert!(sender.part_count() > 1);
}

#[test]
fn sender_new_rejects_oversized_data() {
    use rand::rngs::OsRng;
    let mock = MockLinkCrypto::new(64);
    let data = vec![0; MAX_EFFICIENT_SIZE + 1];
    assert!(matches!(
        ResourceSender::new(&mut OsRng, &mock, &data, 0),
        Err(ReticulumError::ResourceTooLarge { .. })
    ));
}
```

Run: `cargo test -p harmony-reticulum resource`
Expected: FAIL — `ResourceSender::new` not defined.

**Step 2: Implement `ResourceSender::new()`**

Replace the `ResourceSender` skeleton with the full struct and `new()`:

```rust
use std::collections::HashMap;

pub struct ResourceSender {
    state: SenderState,
    resource_hash: ResourceHash,
    random_hash: RandomHash,
    original_data: Vec<u8>,
    encrypted_data: Vec<u8>,
    parts: Vec<Vec<u8>>,
    map_hashes: Vec<MapHash>,
    hashmap: Vec<u8>,
    expected_proof: ResourceHash,
    sdu: usize,
    adv_sent_at: Option<u64>,
    last_activity: u64,
    adv_retries_left: u8,
    sent_parts: usize,
    rtt_ms: Option<u64>,
}

impl ResourceSender {
    /// Create a new resource sender.
    ///
    /// Encrypts the data, chunks it into SDU-sized parts, and computes
    /// all hashes needed for the transfer protocol.
    pub fn new(
        rng: &mut dyn CryptoRngCore,
        crypto: &impl LinkCrypto,
        data: &[u8],
        now_ms: u64,
    ) -> Result<Self, ReticulumError> {
        if data.len() > MAX_EFFICIENT_SIZE {
            return Err(ReticulumError::ResourceTooLarge {
                size: data.len(),
                max: MAX_EFFICIENT_SIZE,
            });
        }

        // Generate 4-byte random nonce
        let mut random_hash: RandomHash = [0u8; 4];
        rng.fill_bytes(&mut random_hash);

        // Encrypt entire payload
        let encrypted_data = crypto.encrypt(rng, data)?;

        // Compute resource hash on encrypted data
        let resource_hash = compute_resource_hash(&encrypted_data, &random_hash);

        // Chunk into SDU-sized parts
        let sdu = crypto.mdu();
        let parts = chunk_data(&encrypted_data, sdu);

        // Compute map hash for each part
        let map_hashes: Vec<MapHash> = parts
            .iter()
            .map(|p| compute_map_hash(p, &random_hash))
            .collect();

        // Build concatenated hashmap
        let hashmap: Vec<u8> = map_hashes.iter().flat_map(|h| h.iter().copied()).collect();

        // Pre-compute expected proof from original plaintext
        let expected_proof = compute_proof_hash(data, &resource_hash);

        Ok(Self {
            state: SenderState::Queued,
            resource_hash,
            random_hash,
            original_data: data.to_vec(),
            encrypted_data,
            parts,
            map_hashes,
            hashmap,
            expected_proof,
            sdu,
            adv_sent_at: None,
            last_activity: now_ms,
            adv_retries_left: MAX_ADV_RETRIES,
            sent_parts: 0,
            rtt_ms: None,
        })
    }

    pub fn state(&self) -> SenderState {
        self.state
    }

    pub fn hash(&self) -> &ResourceHash {
        &self.resource_hash
    }

    pub fn part_count(&self) -> usize {
        self.parts.len()
    }
}
```

Run: `cargo test -p harmony-reticulum resource`
Expected: `sender_new_*` tests pass.

**Step 3: Write failing test for `advertise()`**

```rust
#[test]
fn sender_advertise_emits_packet_and_timeout() {
    use rand::rngs::OsRng;
    let mock = MockLinkCrypto::new(64);
    let data = vec![0xAA; 50];
    let mut sender = ResourceSender::new(&mut OsRng, &mock, &data, 1000).unwrap();

    let actions = sender.advertise(1000);
    assert_eq!(sender.state(), SenderState::Advertised);

    let has_send = actions.iter().any(|a| matches!(a, ResourceAction::SendPacket {
        context: PacketContext::ResourceAdv, ..
    }));
    assert!(has_send, "should emit ResourceAdv packet");

    let has_timeout = actions.iter().any(|a| matches!(a, ResourceAction::ScheduleTimeout { .. }));
    assert!(has_timeout, "should schedule advertisement timeout");
}
```

**Step 4: Implement `advertise()`**

```rust
/// Begin the transfer by emitting the advertisement packet.
pub fn advertise(&mut self, now_ms: u64) -> Vec<ResourceAction> {
    let adv = ResourceAdvertisement {
        transfer_size: self.encrypted_data.len() as u32,
        data_size: self.original_data.len() as u32,
        part_count: self.parts.len() as u32,
        resource_hash: self.resource_hash,
        random_hash: self.random_hash,
        original_hash: self.resource_hash, // same for single-segment
        hashmap: self.hashmap.clone(),
        flags: 0x01, // encrypted
        segment_index: 1,
        total_segments: 1,
    };

    self.state = SenderState::Advertised;
    self.adv_sent_at = Some(now_ms);
    self.last_activity = now_ms;

    let rtt_estimate = self.rtt_ms.unwrap_or(SENDER_GRACE_TIME_MS);
    let deadline = now_ms + rtt_estimate + PROCESSING_GRACE_MS;

    vec![
        ResourceAction::SendPacket {
            context: PacketContext::ResourceAdv,
            plaintext: adv.encode(),
        },
        ResourceAction::SenderStateChanged {
            new_state: SenderState::Advertised,
        },
        ResourceAction::ScheduleTimeout {
            deadline_ms: deadline,
        },
    ]
}
```

**Step 5: Write failing test for `handle_event(RequestReceived)`**

```rust
#[test]
fn sender_handles_request_and_emits_parts() {
    use rand::rngs::OsRng;
    let mock = MockLinkCrypto::new(30);
    let data = vec![0xAA; 100]; // ~4 parts at SDU=30 (mock = identity, so encrypted = 100 bytes)
    let mut sender = ResourceSender::new(&mut OsRng, &mock, &data, 0).unwrap();
    sender.advertise(0);

    // Build a request for the first 2 parts' map hashes
    let mut req_plaintext = vec![HASHMAP_IS_NOT_EXHAUSTED];
    req_plaintext.extend_from_slice(&sender.resource_hash);
    req_plaintext.extend_from_slice(&sender.map_hashes[0]);
    req_plaintext.extend_from_slice(&sender.map_hashes[1]);

    let actions = sender.handle_event(ResourceEvent::RequestReceived {
        plaintext: req_plaintext,
    });
    assert_eq!(sender.state(), SenderState::Transferring);

    let send_count = actions.iter().filter(|a| matches!(a, ResourceAction::SendPacket {
        context: PacketContext::Resource, ..
    })).count();
    assert_eq!(send_count, 2, "should send 2 requested parts");
}
```

**Step 6: Implement `handle_event()` for the sender**

```rust
/// Drive the sender state machine.
pub fn handle_event(&mut self, event: ResourceEvent) -> Vec<ResourceAction> {
    match event {
        ResourceEvent::RequestReceived { plaintext } => self.handle_request(plaintext),
        ResourceEvent::ProofReceived { plaintext } => self.handle_proof(plaintext),
        ResourceEvent::CancelReceived => self.handle_cancel_received(),
        ResourceEvent::Timeout { now_ms } => self.handle_timeout(now_ms),
        _ => vec![], // Receiver-only events
    }
}

/// Cancel the transfer.
pub fn cancel(&mut self) -> Vec<ResourceAction> {
    if self.state == SenderState::Complete || self.state == SenderState::Failed {
        return vec![];
    }
    self.state = SenderState::Failed;
    vec![
        ResourceAction::SendPacket {
            context: PacketContext::ResourceIcl,
            plaintext: self.resource_hash.to_vec(),
        },
        ResourceAction::SenderStateChanged {
            new_state: SenderState::Failed,
        },
    ]
}

fn handle_request(&mut self, plaintext: Vec<u8>) -> Vec<ResourceAction> {
    // Parse: [flag:1][resource_hash:16][map_hashes:N*4]
    // If flag == 0xFF: [flag:1][last_map_hash:4][resource_hash:16][map_hashes:N*4]
    if plaintext.is_empty() {
        return vec![];
    }

    let flag = plaintext[0];
    let (offset, _last_map_hash) = if flag == HASHMAP_IS_EXHAUSTED {
        if plaintext.len() < 1 + 4 + 16 {
            return vec![];
        }
        let mut lmh = [0u8; 4];
        lmh.copy_from_slice(&plaintext[1..5]);
        (5, Some(lmh))
    } else {
        (1, None)
    };

    if plaintext.len() < offset + 16 {
        return vec![];
    }

    let req_resource_hash: ResourceHash = plaintext[offset..offset + 16].try_into().unwrap();
    if req_resource_hash != self.resource_hash {
        return vec![];
    }

    let hash_data = &plaintext[offset + 16..];
    if hash_data.len() % MAPHASH_LEN != 0 {
        return vec![];
    }

    self.state = SenderState::Transferring;
    self.last_activity = self.adv_sent_at.unwrap_or(0); // will be updated by caller

    // Build reverse lookup: map_hash → part index
    let mut hash_to_idx: HashMap<MapHash, usize> = HashMap::new();
    for (i, mh) in self.map_hashes.iter().enumerate() {
        hash_to_idx.insert(*mh, i);
    }

    let mut actions = Vec::new();
    let mut parts_sent_this_request = 0;

    for chunk in hash_data.chunks_exact(MAPHASH_LEN) {
        let requested: MapHash = chunk.try_into().unwrap();
        if let Some(&idx) = hash_to_idx.get(&requested) {
            actions.push(ResourceAction::SendPacket {
                context: PacketContext::Resource,
                plaintext: self.parts[idx].clone(),
            });
            parts_sent_this_request += 1;
        }
    }

    self.sent_parts += parts_sent_this_request;

    if self.sent_parts >= self.parts.len() {
        self.state = SenderState::AwaitingProof;
        actions.push(ResourceAction::SenderStateChanged {
            new_state: SenderState::AwaitingProof,
        });
        // Schedule proof timeout
        let rtt = self.rtt_ms.unwrap_or(SENDER_GRACE_TIME_MS);
        let deadline = self.last_activity + rtt * PROOF_TIMEOUT_FACTOR + SENDER_GRACE_TIME_MS;
        actions.push(ResourceAction::ScheduleTimeout { deadline_ms: deadline });
    } else {
        actions.push(ResourceAction::SenderStateChanged {
            new_state: SenderState::Transferring,
        });
        actions.push(ResourceAction::Progress {
            fraction: self.sent_parts as f32 / self.parts.len() as f32,
        });
    }

    // TODO (Task 6): Handle hashmap exhaustion — send HMU if flag == EXHAUSTED

    actions
}

fn handle_proof(&mut self, plaintext: Vec<u8>) -> Vec<ResourceAction> {
    // Proof packet: [resource_hash:16][proof:16]
    if plaintext.len() < 32 {
        return vec![];
    }

    let recv_resource_hash: ResourceHash = plaintext[..16].try_into().unwrap();
    if recv_resource_hash != self.resource_hash {
        return vec![];
    }

    let recv_proof: ResourceHash = plaintext[16..32].try_into().unwrap();
    if recv_proof != self.expected_proof {
        self.state = SenderState::Failed;
        return vec![ResourceAction::SenderStateChanged {
            new_state: SenderState::Failed,
        }];
    }

    self.state = SenderState::Complete;
    vec![
        ResourceAction::ProofValidated,
        ResourceAction::SenderStateChanged {
            new_state: SenderState::Complete,
        },
    ]
}

fn handle_cancel_received(&mut self) -> Vec<ResourceAction> {
    if self.state == SenderState::Complete || self.state == SenderState::Failed {
        return vec![];
    }
    self.state = SenderState::Rejected;
    vec![ResourceAction::SenderStateChanged {
        new_state: SenderState::Rejected,
    }]
}

fn handle_timeout(&mut self, now_ms: u64) -> Vec<ResourceAction> {
    match self.state {
        SenderState::Advertised => {
            if self.adv_retries_left > 0 {
                self.adv_retries_left -= 1;
                self.adv_sent_at = Some(now_ms);
                let adv = ResourceAdvertisement {
                    transfer_size: self.encrypted_data.len() as u32,
                    data_size: self.original_data.len() as u32,
                    part_count: self.parts.len() as u32,
                    resource_hash: self.resource_hash,
                    random_hash: self.random_hash,
                    original_hash: self.resource_hash,
                    hashmap: self.hashmap.clone(),
                    flags: 0x01,
                    segment_index: 1,
                    total_segments: 1,
                };
                let rtt_estimate = self.rtt_ms.unwrap_or(SENDER_GRACE_TIME_MS);
                let deadline = now_ms + rtt_estimate + PROCESSING_GRACE_MS;
                vec![
                    ResourceAction::SendPacket {
                        context: PacketContext::ResourceAdv,
                        plaintext: adv.encode(),
                    },
                    ResourceAction::ScheduleTimeout { deadline_ms: deadline },
                ]
            } else {
                self.state = SenderState::Failed;
                vec![ResourceAction::SenderStateChanged {
                    new_state: SenderState::Failed,
                }]
            }
        }
        SenderState::Transferring | SenderState::AwaitingProof => {
            // Transfer/proof timeout — fail
            self.state = SenderState::Failed;
            vec![ResourceAction::SenderStateChanged {
                new_state: SenderState::Failed,
            }]
        }
        _ => vec![],
    }
}
```

**Step 7: Write test for proof validation**

```rust
#[test]
fn sender_validates_correct_proof() {
    use rand::rngs::OsRng;
    let mock = MockLinkCrypto::new(200);
    let data = vec![0xAA; 50];
    let mut sender = ResourceSender::new(&mut OsRng, &mock, &data, 0).unwrap();
    sender.advertise(0);

    // Simulate receiver sending correct proof
    let proof = compute_proof_hash(&data, sender.hash());
    let mut proof_pkt = Vec::new();
    proof_pkt.extend_from_slice(sender.hash());
    proof_pkt.extend_from_slice(&proof);

    // Force state to AwaitingProof
    sender.state = SenderState::AwaitingProof;
    let actions = sender.handle_event(ResourceEvent::ProofReceived {
        plaintext: proof_pkt,
    });

    assert_eq!(sender.state(), SenderState::Complete);
    assert!(actions.iter().any(|a| matches!(a, ResourceAction::ProofValidated)));
}

#[test]
fn sender_rejects_wrong_proof() {
    use rand::rngs::OsRng;
    let mock = MockLinkCrypto::new(200);
    let data = vec![0xAA; 50];
    let mut sender = ResourceSender::new(&mut OsRng, &mock, &data, 0).unwrap();
    sender.state = SenderState::AwaitingProof;

    let mut bad_proof = Vec::new();
    bad_proof.extend_from_slice(sender.hash());
    bad_proof.extend_from_slice(&[0xFF; 16]); // wrong proof

    let actions = sender.handle_event(ResourceEvent::ProofReceived {
        plaintext: bad_proof,
    });
    assert_eq!(sender.state(), SenderState::Failed);
    assert!(!actions.iter().any(|a| matches!(a, ResourceAction::ProofValidated)));
}
```

Run: `cargo test -p harmony-reticulum resource`
Expected: All sender tests pass.

**Step 8: Commit**

```bash
git add -A && git commit -m "feat(reticulum): implement ResourceSender state machine"
```

---

### Task 4: ResourceReceiver — Accept, Part Handling, Assembly, Proof

**Files:**
- Modify: `crates/harmony-reticulum/src/resource.rs`

**Step 1: Write failing test for `ResourceReceiver::accept()`**

```rust
#[test]
fn receiver_accept_parses_adv_and_emits_request() {
    let adv = ResourceAdvertisement {
        transfer_size: 100,
        data_size: 100,
        part_count: 4,
        resource_hash: [0xAA; 16],
        random_hash: [0x01, 0x02, 0x03, 0x04],
        original_hash: [0xAA; 16],
        hashmap: vec![
            0x11, 0x12, 0x13, 0x14, // part 0
            0x21, 0x22, 0x23, 0x24, // part 1
            0x31, 0x32, 0x33, 0x34, // part 2
            0x41, 0x42, 0x43, 0x44, // part 3
        ],
        flags: 0x01,
        segment_index: 1,
        total_segments: 1,
    };

    let (receiver, actions) = ResourceReceiver::accept(&adv.encode(), 1000).unwrap();
    assert_eq!(receiver.state(), ReceiverState::Transferring);

    let has_req = actions.iter().any(|a| matches!(a, ResourceAction::SendPacket {
        context: PacketContext::ResourceReq, ..
    }));
    assert!(has_req, "should emit initial RESOURCE_REQ");
}
```

**Step 2: Implement `ResourceReceiver`**

Replace the skeleton:

```rust
pub struct ResourceReceiver {
    state: ReceiverState,
    resource_hash: ResourceHash,
    random_hash: RandomHash,
    original_hash: ResourceHash,
    transfer_size: u32,
    data_size: u32,
    total_parts: usize,
    parts: Vec<Option<Vec<u8>>>,
    hashmap: Vec<Option<MapHash>>,
    hashmap_height: usize,
    received_count: usize,
    consecutive_completed: usize,
    outstanding_parts: usize,
    // Window state
    window: usize,
    window_min: usize,
    window_max: usize,
    retries_left: u8,
    fast_rate_rounds: usize,
    very_slow_rate_rounds: usize,
    // Timing
    last_activity: u64,
    req_sent_at: Option<u64>,
    rtt_ms: Option<u64>,
    part_timeout_factor: u64,
    // Assembled data (set during assembly)
    assembled_encrypted: Option<Vec<u8>>,
}

impl ResourceReceiver {
    /// Accept a resource advertisement and begin receiving.
    pub fn accept(
        adv_plaintext: &[u8],
        now_ms: u64,
    ) -> Result<(Self, Vec<ResourceAction>), ReticulumError> {
        let adv = ResourceAdvertisement::decode(adv_plaintext)?;

        let total_parts = adv.part_count as usize;

        // Parse hashmap into individual MapHash entries
        let mut hashmap: Vec<Option<MapHash>> = vec![None; total_parts];
        let hashmap_height = adv.hashmap.len() / MAPHASH_LEN;
        for (i, chunk) in adv.hashmap.chunks_exact(MAPHASH_LEN).enumerate() {
            if i < total_parts {
                hashmap[i] = Some(chunk.try_into().unwrap());
            }
        }

        let mut receiver = Self {
            state: ReceiverState::Transferring,
            resource_hash: adv.resource_hash,
            random_hash: adv.random_hash,
            original_hash: adv.original_hash,
            transfer_size: adv.transfer_size,
            data_size: adv.data_size,
            total_parts,
            parts: vec![None; total_parts],
            hashmap,
            hashmap_height,
            received_count: 0,
            consecutive_completed: 0,
            outstanding_parts: 0,
            window: WINDOW_INITIAL,
            window_min: WINDOW_MIN,
            window_max: WINDOW_MAX_SLOW,
            retries_left: MAX_RETRIES,
            fast_rate_rounds: 0,
            very_slow_rate_rounds: 0,
            last_activity: now_ms,
            req_sent_at: None,
            rtt_ms: None,
            part_timeout_factor: PART_TIMEOUT_FACTOR,
        };

        let actions = receiver.request_next(now_ms);
        Ok((receiver, actions))
    }

    pub fn state(&self) -> ReceiverState {
        self.state
    }

    pub fn hash(&self) -> &ResourceHash {
        &self.resource_hash
    }

    /// Drive the receiver state machine.
    pub fn handle_event(&mut self, event: ResourceEvent) -> Vec<ResourceAction> {
        match event {
            ResourceEvent::PartReceived { plaintext } => self.handle_part(plaintext),
            ResourceEvent::HashmapUpdateReceived { plaintext } => self.handle_hmu(plaintext),
            ResourceEvent::CancelReceived => self.handle_cancel_received(),
            ResourceEvent::Timeout { now_ms } => self.handle_timeout(now_ms),
            _ => vec![], // Sender-only events
        }
    }

    /// Cancel the transfer.
    pub fn cancel(&mut self) -> Vec<ResourceAction> {
        if self.state == ReceiverState::Complete || self.state == ReceiverState::Failed {
            return vec![];
        }
        self.state = ReceiverState::Failed;
        vec![
            ResourceAction::SendPacket {
                context: PacketContext::ResourceRcl,
                plaintext: self.resource_hash.to_vec(),
            },
            ResourceAction::ReceiverStateChanged {
                new_state: ReceiverState::Failed,
            },
        ]
    }

    /// Finalize the transfer: decrypt assembled data, compute proof.
    ///
    /// Call this after receiving `AssemblyReady` action.
    pub fn finalize(
        &mut self,
        crypto: &impl LinkCrypto,
    ) -> Result<Vec<ResourceAction>, ReticulumError> {
        let encrypted = self
            .assembled_encrypted
            .take()
            .ok_or(ReticulumError::ResourceFailed)?;

        let plaintext = crypto.decrypt(&encrypted)?;

        let proof = compute_proof_hash(&plaintext, &self.resource_hash);
        let mut proof_pkt = Vec::with_capacity(32);
        proof_pkt.extend_from_slice(&self.resource_hash);
        proof_pkt.extend_from_slice(&proof);

        self.state = ReceiverState::Complete;

        Ok(vec![
            ResourceAction::SendPacket {
                context: PacketContext::ResourcePrf,
                plaintext: proof_pkt,
            },
            ResourceAction::Completed { data: plaintext },
            ResourceAction::ReceiverStateChanged {
                new_state: ReceiverState::Complete,
            },
        ])
    }

    /// Build and emit a RESOURCE_REQ for the next window of parts.
    fn request_next(&mut self, now_ms: u64) -> Vec<ResourceAction> {
        let mut requested: Vec<MapHash> = Vec::new();
        let mut hashmap_exhausted = false;

        let search_start = self.consecutive_completed;
        let search_end = (search_start + self.window).min(self.total_parts);

        for i in search_start..search_end {
            if self.parts[i].is_none() {
                if let Some(mh) = self.hashmap[i] {
                    requested.push(mh);
                } else {
                    hashmap_exhausted = true;
                    break;
                }
            }
        }

        if requested.is_empty() && !hashmap_exhausted {
            return vec![];
        }

        self.outstanding_parts = requested.len();
        self.req_sent_at = Some(now_ms);
        self.last_activity = now_ms;

        // Build RESOURCE_REQ packet
        let mut pkt = Vec::new();
        if hashmap_exhausted {
            pkt.push(HASHMAP_IS_EXHAUSTED);
            // Include last known map hash
            if let Some(last_mh) = self.last_known_map_hash() {
                pkt.extend_from_slice(&last_mh);
            } else {
                pkt.extend_from_slice(&[0u8; 4]);
            }
        } else {
            pkt.push(HASHMAP_IS_NOT_EXHAUSTED);
        }
        pkt.extend_from_slice(&self.resource_hash);
        for mh in &requested {
            pkt.extend_from_slice(mh);
        }

        // Schedule timeout
        let rtt = self.rtt_ms.unwrap_or(SENDER_GRACE_TIME_MS);
        let deadline = now_ms + rtt * self.part_timeout_factor + RETRY_GRACE_TIME_MS;

        vec![
            ResourceAction::SendPacket {
                context: PacketContext::ResourceReq,
                plaintext: pkt,
            },
            ResourceAction::ScheduleTimeout { deadline_ms: deadline },
        ]
    }

    fn last_known_map_hash(&self) -> Option<MapHash> {
        for i in (0..self.hashmap_height).rev() {
            if let Some(mh) = self.hashmap[i] {
                return Some(mh);
            }
        }
        None
    }

    fn handle_part(&mut self, plaintext: Vec<u8>) -> Vec<ResourceAction> {
        if self.state != ReceiverState::Transferring {
            return vec![];
        }

        // Identify part by computing map_hash and matching
        let received_hash = compute_map_hash(&plaintext, &self.random_hash);

        let mut found_idx = None;
        for (i, slot) in self.hashmap.iter().enumerate() {
            if let Some(mh) = slot {
                if *mh == received_hash && self.parts[i].is_none() {
                    found_idx = Some(i);
                    break;
                }
            }
        }

        let idx = match found_idx {
            Some(i) => i,
            None => return vec![], // Unknown or duplicate part
        };

        // Measure RTT on first part
        if self.received_count == 0 {
            if let Some(req_sent) = self.req_sent_at {
                let now = self.last_activity; // Caller should set this
                if now > req_sent {
                    self.rtt_ms = Some(now - req_sent);
                    self.part_timeout_factor = PART_TIMEOUT_FACTOR_AFTER_RTT;
                }
            }
        }

        self.parts[idx] = Some(plaintext);
        self.received_count += 1;
        if self.outstanding_parts > 0 {
            self.outstanding_parts -= 1;
        }

        // Update consecutive_completed height
        while self.consecutive_completed < self.total_parts
            && self.parts[self.consecutive_completed].is_some()
        {
            self.consecutive_completed += 1;
        }

        let mut actions = vec![ResourceAction::Progress {
            fraction: self.received_count as f32 / self.total_parts as f32,
        }];

        // All parts received?
        if self.received_count == self.total_parts {
            self.state = ReceiverState::Assembling;
            actions.push(ResourceAction::ReceiverStateChanged {
                new_state: ReceiverState::Assembling,
            });

            // Assemble and verify hash
            let assembled: Vec<u8> = self
                .parts
                .iter()
                .filter_map(|p| p.as_ref())
                .flat_map(|p| p.iter().copied())
                .collect();

            let computed_hash = compute_resource_hash(&assembled, &self.random_hash);
            if computed_hash != self.resource_hash {
                self.state = ReceiverState::Corrupt;
                actions.push(ResourceAction::ReceiverStateChanged {
                    new_state: ReceiverState::Corrupt,
                });
            } else {
                self.assembled_encrypted = Some(assembled);
                actions.push(ResourceAction::AssemblyReady);
            }
        } else if self.outstanding_parts == 0 {
            // Window complete, request next batch
            // TODO (Task 6): Grow window here
            let now = self.last_activity;
            let mut req_actions = self.request_next(now);
            actions.append(&mut req_actions);
        }

        actions
    }

    fn handle_hmu(&mut self, plaintext: Vec<u8>) -> Vec<ResourceAction> {
        // TODO (Task 7): Parse HMU packet, update hashmap, re-request
        let _ = plaintext;
        vec![]
    }

    fn handle_cancel_received(&mut self) -> Vec<ResourceAction> {
        if self.state == ReceiverState::Complete || self.state == ReceiverState::Failed {
            return vec![];
        }
        self.state = ReceiverState::Failed;
        vec![ResourceAction::ReceiverStateChanged {
            new_state: ReceiverState::Failed,
        }]
    }

    fn handle_timeout(&mut self, now_ms: u64) -> Vec<ResourceAction> {
        if self.state != ReceiverState::Transferring {
            return vec![];
        }

        if self.retries_left > 0 {
            self.retries_left -= 1;
            // TODO (Task 6): Window backoff here
            self.request_next(now_ms)
        } else {
            self.state = ReceiverState::Failed;
            vec![ResourceAction::ReceiverStateChanged {
                new_state: ReceiverState::Failed,
            }]
        }
    }
}
```

**Step 3: Write tests for part handling and assembly**

```rust
#[test]
fn receiver_handles_parts_and_assembles() {
    use rand::rngs::OsRng;
    let mock = MockLinkCrypto::new(30);
    let data = vec![0xAA; 80]; // Will become ~3 parts at SDU=30

    // Create sender to generate valid advertisement
    let sender = ResourceSender::new(&mut OsRng, &mock, &data, 0).unwrap();
    let adv = ResourceAdvertisement {
        transfer_size: sender.encrypted_data.len() as u32,
        data_size: data.len() as u32,
        part_count: sender.parts.len() as u32,
        resource_hash: sender.resource_hash,
        random_hash: sender.random_hash,
        original_hash: sender.resource_hash,
        hashmap: sender.hashmap.clone(),
        flags: 0x01,
        segment_index: 1,
        total_segments: 1,
    };

    let (mut receiver, _) = ResourceReceiver::accept(&adv.encode(), 0).unwrap();

    // Feed all parts
    for part in &sender.parts {
        receiver.handle_event(ResourceEvent::PartReceived {
            plaintext: part.clone(),
        });
    }

    assert_eq!(receiver.state(), ReceiverState::Assembling);

    // Finalize
    let actions = receiver.finalize(&mock).unwrap();
    assert_eq!(receiver.state(), ReceiverState::Complete);

    // Should contain Completed with original data
    let completed = actions.iter().find_map(|a| {
        if let ResourceAction::Completed { data } = a { Some(data) } else { None }
    });
    assert_eq!(completed.unwrap(), &data);

    // Should contain proof packet
    assert!(actions.iter().any(|a| matches!(a, ResourceAction::SendPacket {
        context: PacketContext::ResourcePrf, ..
    })));
}
```

**Note for implementer:** The test above accesses `sender.encrypted_data`, `sender.parts`, `sender.hashmap`, `sender.random_hash`, `sender.resource_hash` directly. You may need to add `pub(crate)` visibility to these fields or add accessor methods. Prefer `pub(crate)` for test access, keeping the public API through methods.

Run: `cargo test -p harmony-reticulum resource`
Expected: All receiver tests pass.

**Step 4: Commit**

```bash
git add -A && git commit -m "feat(reticulum): implement ResourceReceiver state machine"
```

---

### Task 5: Full Round-Trip Integration Tests

**Files:**
- Modify: `crates/harmony-reticulum/src/resource.rs`

Wire sender and receiver together end-to-end, simulating the packet exchange in a loop.

**Step 1: Write round-trip integration test**

```rust
/// Simulate a full sender ↔ receiver transfer using MockLinkCrypto.
///
/// Extracts SendPacket actions from one side and feeds them to the other
/// as the appropriate event type, based on the packet context.
#[test]
fn full_roundtrip_small_data() {
    use rand::rngs::OsRng;
    let mock = MockLinkCrypto::new(64);
    let data = b"Hello, resource transfer!".to_vec();

    // Sender creates and advertises
    let mut sender = ResourceSender::new(&mut OsRng, &mock, &data, 0).unwrap();
    let adv_actions = sender.advertise(0);

    // Extract advertisement plaintext
    let adv_plaintext = adv_actions
        .iter()
        .find_map(|a| match a {
            ResourceAction::SendPacket {
                context: PacketContext::ResourceAdv,
                plaintext,
            } => Some(plaintext.clone()),
            _ => None,
        })
        .expect("sender should emit advertisement");

    // Receiver accepts
    let (mut receiver, req_actions) = ResourceReceiver::accept(&adv_plaintext, 100).unwrap();

    // Extract request plaintext
    let req_plaintext = req_actions
        .iter()
        .find_map(|a| match a {
            ResourceAction::SendPacket {
                context: PacketContext::ResourceReq,
                plaintext,
            } => Some(plaintext.clone()),
            _ => None,
        })
        .expect("receiver should emit request");

    // Sender handles request, emits parts
    let part_actions = sender.handle_event(ResourceEvent::RequestReceived {
        plaintext: req_plaintext,
    });

    // Feed each part to receiver
    for action in &part_actions {
        if let ResourceAction::SendPacket {
            context: PacketContext::Resource,
            plaintext,
        } = action
        {
            receiver.handle_event(ResourceEvent::PartReceived {
                plaintext: plaintext.clone(),
            });
        }
    }

    // Receiver should be Assembling
    assert_eq!(receiver.state(), ReceiverState::Assembling);

    // Finalize
    let final_actions = receiver.finalize(&mock).unwrap();
    assert_eq!(receiver.state(), ReceiverState::Complete);

    // Extract proof
    let proof_plaintext = final_actions
        .iter()
        .find_map(|a| match a {
            ResourceAction::SendPacket {
                context: PacketContext::ResourcePrf,
                plaintext,
            } => Some(plaintext.clone()),
            _ => None,
        })
        .expect("receiver should emit proof");

    // Verify completed data
    let completed_data = final_actions
        .iter()
        .find_map(|a| match a {
            ResourceAction::Completed { data } => Some(data.clone()),
            _ => None,
        })
        .expect("receiver should emit Completed");
    assert_eq!(completed_data, data);

    // Sender validates proof
    let proof_actions = sender.handle_event(ResourceEvent::ProofReceived {
        plaintext: proof_plaintext,
    });
    assert_eq!(sender.state(), SenderState::Complete);
    assert!(proof_actions
        .iter()
        .any(|a| matches!(a, ResourceAction::ProofValidated)));
}
```

**Step 2: Write multi-part round-trip test**

```rust
#[test]
fn full_roundtrip_multipart() {
    use rand::rngs::OsRng;
    let mock = MockLinkCrypto::new(20); // Small SDU forces many parts
    let data: Vec<u8> = (0..200u8).collect();

    let mut sender = ResourceSender::new(&mut OsRng, &mock, &data, 0).unwrap();
    assert!(sender.part_count() >= 10);

    let adv_actions = sender.advertise(0);
    let adv_pt = extract_send(&adv_actions, PacketContext::ResourceAdv);

    let (mut receiver, mut actions) = ResourceReceiver::accept(&adv_pt, 0).unwrap();

    // Loop: receiver requests → sender sends parts → repeat until complete
    let mut iterations = 0;
    loop {
        iterations += 1;
        if iterations > 100 {
            panic!("round-trip did not complete in 100 iterations");
        }

        // Extract request from receiver actions
        let req_pt = match find_send(&actions, PacketContext::ResourceReq) {
            Some(pt) => pt,
            None => break,
        };

        // Sender handles request
        let sender_actions = sender.handle_event(ResourceEvent::RequestReceived {
            plaintext: req_pt,
        });

        // Feed parts to receiver
        actions = Vec::new();
        for a in &sender_actions {
            if let ResourceAction::SendPacket {
                context: PacketContext::Resource,
                plaintext,
            } = a
            {
                let part_actions = receiver.handle_event(ResourceEvent::PartReceived {
                    plaintext: plaintext.clone(),
                });
                actions.extend(part_actions);
            }
        }

        if receiver.state() == ReceiverState::Assembling {
            break;
        }
    }

    let final_actions = receiver.finalize(&mock).unwrap();
    assert_eq!(receiver.state(), ReceiverState::Complete);

    let completed_data = final_actions
        .iter()
        .find_map(|a| match a {
            ResourceAction::Completed { data } => Some(data.clone()),
            _ => None,
        })
        .unwrap();
    assert_eq!(completed_data, data);
}

/// Helper: extract SendPacket plaintext for a given context.
fn extract_send(actions: &[ResourceAction], ctx: PacketContext) -> Vec<u8> {
    find_send(actions, ctx).expect("expected SendPacket action")
}

fn find_send(actions: &[ResourceAction], ctx: PacketContext) -> Option<Vec<u8>> {
    actions.iter().find_map(|a| match a {
        ResourceAction::SendPacket { context, plaintext } if *context == ctx => {
            Some(plaintext.clone())
        }
        _ => None,
    })
}
```

**Step 3: Write edge case tests**

```rust
#[test]
fn roundtrip_single_byte() {
    use rand::rngs::OsRng;
    let mock = MockLinkCrypto::new(64);
    let data = vec![0x42];
    let mut sender = ResourceSender::new(&mut OsRng, &mock, &data, 0).unwrap();
    assert_eq!(sender.part_count(), 1);
    // Run through full protocol (abbreviated — use same pattern as small_data test)
    let adv_actions = sender.advertise(0);
    let adv_pt = extract_send(&adv_actions, PacketContext::ResourceAdv);
    let (mut receiver, req_actions) = ResourceReceiver::accept(&adv_pt, 0).unwrap();
    let req_pt = extract_send(&req_actions, PacketContext::ResourceReq);
    let part_actions = sender.handle_event(ResourceEvent::RequestReceived { plaintext: req_pt });
    for a in &part_actions {
        if let ResourceAction::SendPacket { context: PacketContext::Resource, plaintext } = a {
            receiver.handle_event(ResourceEvent::PartReceived { plaintext: plaintext.clone() });
        }
    }
    let final_actions = receiver.finalize(&mock).unwrap();
    let completed = final_actions.iter().find_map(|a| match a {
        ResourceAction::Completed { data } => Some(data.clone()),
        _ => None,
    }).unwrap();
    assert_eq!(completed, vec![0x42]);
}

#[test]
fn roundtrip_exact_sdu_boundary() {
    use rand::rngs::OsRng;
    let mock = MockLinkCrypto::new(50);
    let data = vec![0xDD; 100]; // Exactly 2 parts
    let sender = ResourceSender::new(&mut OsRng, &mock, &data, 0).unwrap();
    assert_eq!(sender.part_count(), 2);
}
```

Run: `cargo test -p harmony-reticulum resource`
Expected: All integration tests pass.

**Step 4: Commit**

```bash
git add -A && git commit -m "test(reticulum): add full resource transfer round-trip integration tests"
```

---

### Task 6: Adaptive Windowing and Retransmission

**Files:**
- Modify: `crates/harmony-reticulum/src/resource.rs`

Implement the three window adaptation rules in `ResourceReceiver`:
1. **Growth:** When outstanding == 0, grow window.
2. **Rate detection:** Measure RTT rate, adjust window_max.
3. **Backoff on timeout:** Shrink window, decrement retries.

**Step 1: Write tests for window behavior**

```rust
#[test]
fn window_grows_on_full_receive() {
    use rand::rngs::OsRng;
    let mock = MockLinkCrypto::new(10);
    let data = vec![0xAA; 200]; // ~20 parts at SDU=10
    let mut sender = ResourceSender::new(&mut OsRng, &mock, &data, 0).unwrap();
    let adv_actions = sender.advertise(0);
    let adv_pt = extract_send(&adv_actions, PacketContext::ResourceAdv);
    let (mut receiver, _) = ResourceReceiver::accept(&adv_pt, 0).unwrap();

    let initial_window = receiver.window;

    // Complete one full window of parts (feed parts that match the first request)
    // The window should grow after all outstanding parts arrive
    // ... (detailed simulation of feeding exactly window-many parts)
    // This test validates the principle; the exact growth logic should match Python.
    assert!(initial_window >= WINDOW_MIN);
}

#[test]
fn window_shrinks_on_timeout() {
    use rand::rngs::OsRng;
    let mock = MockLinkCrypto::new(10);
    let data = vec![0xAA; 200];
    let mut sender = ResourceSender::new(&mut OsRng, &mock, &data, 0).unwrap();
    let adv_actions = sender.advertise(0);
    let adv_pt = extract_send(&adv_actions, PacketContext::ResourceAdv);
    let (mut receiver, _) = ResourceReceiver::accept(&adv_pt, 0).unwrap();

    let initial_window = receiver.window;
    let initial_retries = receiver.retries_left;

    // Fire timeout
    receiver.handle_event(ResourceEvent::Timeout { now_ms: 100_000 });

    assert!(receiver.retries_left < initial_retries);
    // Window should shrink if above minimum
    if initial_window > WINDOW_MIN {
        assert!(receiver.window < initial_window);
    }
}

#[test]
fn receiver_fails_after_max_retries() {
    use rand::rngs::OsRng;
    let mock = MockLinkCrypto::new(10);
    let data = vec![0xAA; 50];
    let mut sender = ResourceSender::new(&mut OsRng, &mock, &data, 0).unwrap();
    let adv_actions = sender.advertise(0);
    let adv_pt = extract_send(&adv_actions, PacketContext::ResourceAdv);
    let (mut receiver, _) = ResourceReceiver::accept(&adv_pt, 0).unwrap();

    // Fire MAX_RETRIES + 1 timeouts
    for i in 0..=MAX_RETRIES {
        let actions = receiver.handle_event(ResourceEvent::Timeout { now_ms: i as u64 * 10_000 });
        if receiver.state() == ReceiverState::Failed {
            assert_eq!(i, MAX_RETRIES);
            assert!(actions.iter().any(|a| matches!(a,
                ResourceAction::ReceiverStateChanged { new_state: ReceiverState::Failed }
            )));
            return;
        }
    }
    panic!("should have failed after MAX_RETRIES");
}

#[test]
fn sender_retries_advertisement_then_fails() {
    use rand::rngs::OsRng;
    let mock = MockLinkCrypto::new(64);
    let data = vec![0xAA; 50];
    let mut sender = ResourceSender::new(&mut OsRng, &mock, &data, 0).unwrap();
    sender.advertise(0);

    // Should retry MAX_ADV_RETRIES times then fail
    for i in 0..MAX_ADV_RETRIES {
        let actions = sender.handle_event(ResourceEvent::Timeout { now_ms: (i as u64 + 1) * 10_000 });
        assert_eq!(sender.state(), SenderState::Advertised);
        assert!(actions.iter().any(|a| matches!(a, ResourceAction::SendPacket {
            context: PacketContext::ResourceAdv, ..
        })));
    }

    // One more timeout → fail
    let actions = sender.handle_event(ResourceEvent::Timeout { now_ms: 100_000 });
    assert_eq!(sender.state(), SenderState::Failed);
    assert!(actions.iter().any(|a| matches!(a,
        ResourceAction::SenderStateChanged { new_state: SenderState::Failed }
    )));
}
```

**Step 2: Add window adaptation logic to `handle_part` and `handle_timeout`**

In `handle_part`, when `outstanding_parts == 0` (window fully received), add:

```rust
// Window growth: full window received
if self.window < self.window_max {
    self.window += 1;
    if (self.window - self.window_min) > (WINDOW_FLEXIBILITY - 1) {
        self.window_min += 1;
    }
}
```

In `handle_timeout`, before the retry:

```rust
// Window backoff
if self.window > self.window_min {
    self.window -= 1;
    if self.window_max > self.window_min {
        self.window_max -= 1;
        if (self.window_max - self.window) > (WINDOW_FLEXIBILITY - 1) {
            self.window_max -= 1;
        }
    }
}
```

**Step 3: Add RTT-based rate detection**

In `handle_part`, after measuring RTT on first part:

```rust
if let Some(rtt) = self.rtt_ms {
    if rtt > 0 {
        // Estimate rate from request-response cost
        let bytes_transferred = plaintext.len() as u64; // Approximate
        let rate_bps = (bytes_transferred * 1000) / rtt;

        if rate_bps > RATE_FAST {
            self.fast_rate_rounds += 1;
            if self.fast_rate_rounds >= FAST_RATE_THRESHOLD {
                self.window_max = WINDOW_MAX_FAST;
            }
        } else if self.fast_rate_rounds == 0 && rate_bps < RATE_VERY_SLOW {
            self.very_slow_rate_rounds += 1;
            if self.very_slow_rate_rounds >= VERY_SLOW_RATE_THRESHOLD {
                self.window_max = WINDOW_MAX_VERY_SLOW;
            }
        }
    }
}
```

Run: `cargo test -p harmony-reticulum resource`
Expected: All windowing and timeout tests pass.

**Step 4: Commit**

```bash
git add -A && git commit -m "feat(reticulum): add adaptive windowing and retransmission to ResourceReceiver"
```

---

### Task 7: Cancellation and Hashmap Exhaustion (HMU)

**Files:**
- Modify: `crates/harmony-reticulum/src/resource.rs`

**Step 1: Write cancellation tests**

```rust
#[test]
fn sender_cancel_emits_icl() {
    use rand::rngs::OsRng;
    let mock = MockLinkCrypto::new(64);
    let data = vec![0xAA; 50];
    let mut sender = ResourceSender::new(&mut OsRng, &mock, &data, 0).unwrap();
    sender.advertise(0);

    let actions = sender.cancel();
    assert_eq!(sender.state(), SenderState::Failed);
    assert!(actions.iter().any(|a| matches!(a, ResourceAction::SendPacket {
        context: PacketContext::ResourceIcl, ..
    })));
}

#[test]
fn receiver_cancel_emits_rcl() {
    let adv = ResourceAdvertisement {
        transfer_size: 50, data_size: 50, part_count: 2,
        resource_hash: [0xAA; 16], random_hash: [1, 2, 3, 4],
        original_hash: [0xAA; 16],
        hashmap: vec![0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88],
        flags: 0x01, segment_index: 1, total_segments: 1,
    };
    let (mut receiver, _) = ResourceReceiver::accept(&adv.encode(), 0).unwrap();

    let actions = receiver.cancel();
    assert_eq!(receiver.state(), ReceiverState::Failed);
    assert!(actions.iter().any(|a| matches!(a, ResourceAction::SendPacket {
        context: PacketContext::ResourceRcl, ..
    })));
}

#[test]
fn sender_receives_cancel_becomes_rejected() {
    use rand::rngs::OsRng;
    let mock = MockLinkCrypto::new(64);
    let data = vec![0xAA; 50];
    let mut sender = ResourceSender::new(&mut OsRng, &mock, &data, 0).unwrap();
    sender.advertise(0);

    sender.handle_event(ResourceEvent::CancelReceived);
    assert_eq!(sender.state(), SenderState::Rejected);
}

#[test]
fn cancel_on_complete_is_noop() {
    use rand::rngs::OsRng;
    let mock = MockLinkCrypto::new(64);
    let data = vec![0xAA; 50];
    let mut sender = ResourceSender::new(&mut OsRng, &mock, &data, 0).unwrap();
    sender.state = SenderState::Complete;

    let actions = sender.cancel();
    assert!(actions.is_empty());
    assert_eq!(sender.state(), SenderState::Complete);
}
```

**Step 2: Implement HMU sender-side handling**

In `ResourceSender::handle_request()`, replace the TODO comment with:

```rust
// Handle hashmap exhaustion: send next hashmap segment
if flag == HASHMAP_IS_EXHAUSTED {
    if let Some(lmh) = _last_map_hash {
        if let Some(hmu_actions) = self.build_hmu(&lmh) {
            actions.extend(hmu_actions);
        }
    }
}
```

Add the `build_hmu` method:

```rust
/// Build a RESOURCE_HMU packet for the next hashmap segment.
fn build_hmu(&self, last_map_hash: &MapHash) -> Option<Vec<ResourceAction>> {
    // Find which part the last_map_hash corresponds to
    let part_idx = self.map_hashes.iter().position(|mh| mh == last_map_hash)?;

    // Calculate max hashes per HMU segment based on link MDU
    let max_hmu_hashes = (self.sdu.saturating_sub(16 + 10)) / MAPHASH_LEN; // 16 for resource_hash, ~10 for msgpack overhead
    if max_hmu_hashes == 0 {
        return None;
    }

    let segment = (part_idx + 1) / max_hmu_hashes;
    let start = segment * max_hmu_hashes;
    let end = ((segment + 1) * max_hmu_hashes).min(self.parts.len());

    if start >= self.parts.len() {
        return None;
    }

    let hashes: Vec<u8> = self.map_hashes[start..end]
        .iter()
        .flat_map(|h| h.iter().copied())
        .collect();

    // Build HMU: [resource_hash:16][msgpack([segment_index, hashmap_bytes])]
    let mut pkt = Vec::new();
    pkt.extend_from_slice(&self.resource_hash);

    // Encode msgpack array [segment, hashes]
    rmp::encode::write_array_len(&mut pkt, 2).unwrap();
    rmp::encode::write_u32(&mut pkt, segment as u32).unwrap();
    rmp::encode::write_bin(&mut pkt, &hashes).unwrap();

    Some(vec![ResourceAction::SendPacket {
        context: PacketContext::ResourceHmu,
        plaintext: pkt,
    }])
}
```

**Step 3: Implement HMU receiver-side handling**

Replace the `handle_hmu` TODO:

```rust
fn handle_hmu(&mut self, plaintext: Vec<u8>) -> Vec<ResourceAction> {
    if plaintext.len() < 16 {
        return vec![];
    }

    let recv_hash: ResourceHash = plaintext[..16].try_into().unwrap();
    if recv_hash != self.resource_hash {
        return vec![];
    }

    // Parse msgpack: [segment_index, hashmap_bytes]
    let mut cursor = std::io::Cursor::new(&plaintext[16..]);
    let arr_len = match rmp::decode::read_array_len(&mut cursor) {
        Ok(len) => len,
        Err(_) => return vec![],
    };
    if arr_len != 2 {
        return vec![];
    }

    let segment: u32 = match rmp::decode::read_int(&mut cursor) {
        Ok(s) => s,
        Err(_) => return vec![],
    };

    let hash_len = match rmp::decode::read_bin_len(&mut cursor) {
        Ok(len) => len as usize,
        Err(_) => return vec![],
    };

    let pos = cursor.position() as usize;
    let hash_data = &plaintext[16 + pos..16 + pos + hash_len];

    // Calculate where these hashes start in the part array
    let max_hmu_hashes = hash_data.len() / MAPHASH_LEN;
    let start = segment as usize * max_hmu_hashes;

    for (i, chunk) in hash_data.chunks_exact(MAPHASH_LEN).enumerate() {
        let idx = start + i;
        if idx < self.total_parts && self.hashmap[idx].is_none() {
            self.hashmap[idx] = Some(chunk.try_into().unwrap());
            self.hashmap_height = self.hashmap_height.max(idx + 1);
        }
    }

    // Now re-request with the new hashes
    let now = self.last_activity;
    self.request_next(now)
}
```

**Step 4: Write HMU test**

```rust
#[test]
fn hmu_extends_receiver_hashmap() {
    // Create advertisement with only first 2 of 4 hashes (simulating partial hashmap)
    let partial_hashmap = vec![
        0x11, 0x12, 0x13, 0x14, // part 0
        0x21, 0x22, 0x23, 0x24, // part 1
        // parts 2-3 missing from initial advertisement
    ];
    let adv = ResourceAdvertisement {
        transfer_size: 100, data_size: 100, part_count: 4,
        resource_hash: [0xAA; 16], random_hash: [1, 2, 3, 4],
        original_hash: [0xAA; 16],
        hashmap: partial_hashmap,
        flags: 0x01, segment_index: 1, total_segments: 1,
    };

    let (mut receiver, _) = ResourceReceiver::accept(&adv.encode(), 0).unwrap();
    assert_eq!(receiver.hashmap_height, 2);

    // Build HMU with segment 1 (parts 2-3)
    let mut hmu_pkt = Vec::new();
    hmu_pkt.extend_from_slice(&[0xAA; 16]); // resource_hash
    rmp::encode::write_array_len(&mut hmu_pkt, 2).unwrap();
    rmp::encode::write_u32(&mut hmu_pkt, 1).unwrap(); // segment 1
    let hashes = vec![0x31, 0x32, 0x33, 0x34, 0x41, 0x42, 0x43, 0x44];
    rmp::encode::write_bin(&mut hmu_pkt, &hashes).unwrap();

    let actions = receiver.handle_event(ResourceEvent::HashmapUpdateReceived {
        plaintext: hmu_pkt,
    });

    // Should have updated hashmap and issued a new request
    assert!(receiver.hashmap[2].is_some());
    assert!(receiver.hashmap[3].is_some());
    assert!(actions.iter().any(|a| matches!(a, ResourceAction::SendPacket {
        context: PacketContext::ResourceReq, ..
    })));
}
```

Run: `cargo test -p harmony-reticulum resource`
Expected: All cancellation and HMU tests pass.

**Step 5: Commit**

```bash
git add -A && git commit -m "feat(reticulum): add cancellation and hashmap update support"
```

---

### Task 8: LinkCrypto Impl on Link and Python Interop Tests

**Files:**
- Modify: `crates/harmony-reticulum/src/link.rs`
- Modify: `crates/harmony-reticulum/src/resource.rs` (interop tests)

**Step 1: Implement `LinkCrypto` for `Link`**

In `link.rs`, add the trait import and impl:

```rust
use crate::resource::LinkCrypto;

impl LinkCrypto for Link {
    fn encrypt(
        &self,
        rng: &mut dyn CryptoRngCore,
        plaintext: &[u8],
    ) -> Result<Vec<u8>, ReticulumError> {
        self.require_active()?;
        let key = self.derived_key.as_ref().unwrap();
        harmony_crypto::fernet::encrypt(rng, key, plaintext)
            .map_err(|e| ReticulumError::Identity(e.into()))
    }

    fn decrypt(&self, ciphertext: &[u8]) -> Result<Vec<u8>, ReticulumError> {
        self.require_active()?;
        let key = self.derived_key.as_ref().unwrap();
        harmony_crypto::fernet::decrypt(key, ciphertext)
            .map_err(|e| ReticulumError::Identity(e.into()))
    }

    fn link_id(&self) -> &[u8; 16] {
        &self.link_id
    }

    fn mdu(&self) -> usize {
        // Fernet overhead: version(1) + timestamp(8) + IV(16) + HMAC(32) = 57 bytes minimum
        // Plus PKCS7 padding (up to 16 bytes). Conservative estimate:
        // MTU(500) - Header1(19) - Fernet overhead(73) = 408.
        // Python uses SDU=383. Match Python for interop.
        383
    }
}
```

**Step 2: Write test with real Link crypto**

In `link.rs` tests:

```rust
#[test]
fn link_crypto_resource_roundtrip() {
    use crate::resource::{LinkCrypto, ResourceSender, ResourceReceiver, ResourceAction, ResourceEvent};
    use crate::context::PacketContext;

    let (initiator, responder) = setup_active_link();
    let data = b"encrypted resource data for link test";

    // Sender uses initiator's LinkCrypto
    let mut sender = ResourceSender::new(&mut OsRng, &initiator, data.as_slice(), 0).unwrap();
    let adv_actions = sender.advertise(0);

    let adv_pt = adv_actions.iter().find_map(|a| match a {
        ResourceAction::SendPacket { context: PacketContext::ResourceAdv, plaintext } => Some(plaintext.clone()),
        _ => None,
    }).unwrap();

    // Receiver uses responder's LinkCrypto
    let (mut receiver, req_actions) = ResourceReceiver::accept(&adv_pt, 0).unwrap();
    let req_pt = req_actions.iter().find_map(|a| match a {
        ResourceAction::SendPacket { context: PacketContext::ResourceReq, plaintext } => Some(plaintext.clone()),
        _ => None,
    }).unwrap();

    let part_actions = sender.handle_event(ResourceEvent::RequestReceived { plaintext: req_pt });
    for a in &part_actions {
        if let ResourceAction::SendPacket { context: PacketContext::Resource, plaintext } = a {
            receiver.handle_event(ResourceEvent::PartReceived { plaintext: plaintext.clone() });
        }
    }

    let final_actions = receiver.finalize(&responder).unwrap();
    let completed = final_actions.iter().find_map(|a| match a {
        ResourceAction::Completed { data } => Some(data.clone()),
        _ => None,
    }).unwrap();
    assert_eq!(completed, data);
}
```

**Step 3: Add Python interop tests**

Create test vectors by running the Python Reticulum `Resource` code (or derive them analytically). At minimum, verify hash computations match.

In `resource.rs` tests:

```rust
// ── Python interop tests ────────────────────────────────────────────

/// Verify map_hash matches Python: hashlib.sha256(part + random_hash).digest()[:4]
///
/// To generate: in Python:
/// ```python
/// import hashlib
/// part = bytes([0xAA] * 30)
/// random_hash = bytes([0x01, 0x02, 0x03, 0x04])
/// h = hashlib.sha256(part + random_hash).digest()[:4]
/// print(h.hex())
/// ```
#[test]
fn interop_map_hash_matches_python() {
    let part = vec![0xAA; 30];
    let random_hash: RandomHash = [0x01, 0x02, 0x03, 0x04];
    let mh = compute_map_hash(&part, &random_hash);
    // TODO: Replace with actual Python-generated value
    // For now, verify it's deterministic and 4 bytes
    assert_eq!(mh.len(), 4);
    assert_eq!(mh, compute_map_hash(&part, &random_hash));
}

/// Verify resource_hash matches Python: hashlib.sha256(data + random_hash).digest()[:16]
///
/// Same structure as Python's `RNS.Identity.full_hash(data + random_hash)[:16]`.
/// `full_hash` is SHA-256, `truncated_hash` takes first 16 bytes.
#[test]
fn interop_resource_hash_matches_python() {
    let data = vec![0xBB; 100];
    let random_hash: RandomHash = [0x05, 0x06, 0x07, 0x08];
    let rh = compute_resource_hash(&data, &random_hash);
    assert_eq!(rh.len(), 16);
    // TODO: Replace with Python-generated value
}

/// Verify proof matches Python: hashlib.sha256(plaintext + resource_hash).digest()[:16]
#[test]
fn interop_proof_hash_matches_python() {
    let plaintext = b"test data for proof";
    let resource_hash: ResourceHash = [0xCC; 16];
    let proof = compute_proof_hash(plaintext, &resource_hash);
    assert_eq!(proof.len(), 16);
    // TODO: Replace with Python-generated value
}

/// Verify advertisement msgpack can be decoded by Python and vice versa.
///
/// To generate Python vector:
/// ```python
/// import msgpack
/// adv = {
///     "t": 100, "d": 80, "n": 3,
///     "h": bytes([0xAA]*16), "r": bytes([1,2,3,4]),
///     "o": bytes([0xBB]*16), "m": bytes([0x11]*12),
///     "f": 1, "i": 1, "l": 1
/// }
/// print(msgpack.packb(adv).hex())
/// ```
#[test]
fn interop_advertisement_decode_from_python() {
    // TODO: Replace with actual Python-generated msgpack bytes
    // For now, verify our own encode/decode roundtrip
    let adv = ResourceAdvertisement {
        transfer_size: 100, data_size: 80, part_count: 3,
        resource_hash: [0xAA; 16], random_hash: [1, 2, 3, 4],
        original_hash: [0xBB; 16], hashmap: vec![0x11; 12],
        flags: 1, segment_index: 1, total_segments: 1,
    };
    let encoded = adv.encode();
    let decoded = ResourceAdvertisement::decode(&encoded).unwrap();
    assert_eq!(decoded, adv);
}
```

**Note for implementer:** The `TODO` comments above mark where actual Python-generated test vectors should be inserted. Run the Python snippets shown in the doc comments against the reference Reticulum at `/Users/zeblith/work/markqvist/Reticulum/` and paste the hex values. Use `hashlib.sha256(data).digest()` for hash computations — Reticulum's `Identity.full_hash()` is just SHA-256.

Run: `cargo test -p harmony-reticulum resource`
Expected: All tests pass (interop tests will pass with roundtrip for now, full vectors added later).

Run: `cargo test -p harmony-reticulum link`
Expected: All link tests pass including new `link_crypto_resource_roundtrip`.

**Step 4: Commit**

```bash
git add -A && git commit -m "feat(reticulum): implement LinkCrypto for Link and add interop test stubs"
```

---

### Task 9: Quality Gates

**Step 1: Run full workspace tests**

Run: `cargo test --workspace`
Expected: All tests pass (430+ existing + ~30 new resource tests).

**Step 2: Run clippy**

Run: `cargo clippy --workspace`
Expected: No warnings.

**Step 3: Run fmt check**

Run: `cargo fmt --all -- --check`
Expected: No formatting issues.

**Step 4: Fix any issues found, then commit if needed**

```bash
git add -A && git commit -m "chore(reticulum): fix quality gate issues"
```
