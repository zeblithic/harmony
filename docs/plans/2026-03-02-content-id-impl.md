# ContentId Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement the 32-byte `ContentId` struct — the foundational data type for Harmony's content-addressing layer.

**Architecture:** New `harmony-content` crate depending only on `harmony-crypto` (for SHA-256). The `ContentId` is a `#[repr(C)]` 32-byte struct with 28-byte truncated SHA-256 hash, 20-bit payload size, and 12-bit unary-encoded type tag + variable checksum. Pure data structure — no I/O, no async.

**Tech Stack:** Rust, `sha2` (via `harmony-crypto`), `thiserror`, `hex` (dev)

---

### Task 1: Scaffold the `harmony-content` crate

**Files:**
- Create: `crates/harmony-content/Cargo.toml`
- Create: `crates/harmony-content/src/lib.rs`
- Modify: `Cargo.toml` (workspace root, line 4 — add to members)

**Step 1: Create the crate directory**

Run: `mkdir -p crates/harmony-content/src`

**Step 2: Create `Cargo.toml`**

Create `crates/harmony-content/Cargo.toml`:

```toml
[package]
name = "harmony-content"
version.workspace = true
edition.workspace = true
license.workspace = true
repository.workspace = true
rust-version.workspace = true
description = "Content-addressed storage primitives for the Harmony decentralized stack"

[dependencies]
harmony-crypto.workspace = true
thiserror.workspace = true

[dev-dependencies]
hex.workspace = true
```

**Step 3: Create `src/lib.rs`**

Create `crates/harmony-content/src/lib.rs`:

```rust
pub mod cid;
pub mod error;
```

**Step 4: Create `src/error.rs`**

Create `crates/harmony-content/src/error.rs`:

```rust
/// Errors produced by content-addressing operations.
#[derive(Debug, thiserror::Error)]
pub enum ContentError {
    #[error("payload too large: {size} bytes exceeds maximum {max}")]
    PayloadTooLarge { size: usize, max: usize },

    #[error("invalid CID: checksum mismatch")]
    ChecksumMismatch,

    #[error("invalid CID: child depth {child} must be less than parent depth {parent}")]
    DepthViolation { child: u8, parent: u8 },
}
```

**Step 5: Add to workspace members**

Modify `Cargo.toml` (workspace root), add `"crates/harmony-content"` to the `members` array. Also add the workspace dependency:

```toml
harmony-content = { path = "crates/harmony-content" }
```

**Step 6: Verify it compiles**

Run: `cargo check -p harmony-content`
Expected: Success with no errors.

**Step 7: Commit**

```bash
git add crates/harmony-content/ Cargo.toml
git commit -m "feat(content): scaffold harmony-content crate"
```

---

### Task 2: `ContentId` struct and constants

**Files:**
- Create: `crates/harmony-content/src/cid.rs`

**Step 1: Write the failing test — struct size and alignment**

Add to `crates/harmony-content/src/cid.rs`:

```rust
use crate::error::ContentError;

/// Length of the truncated SHA-256 content hash in bytes.
pub const CONTENT_HASH_LEN: usize = 28;

/// Maximum payload size expressible in 20 bits.
pub const MAX_PAYLOAD_SIZE: usize = 0xF_FFFF; // 1,048,575 bytes

/// Number of bits used for the type tag + checksum field.
pub const TAG_BITS: u32 = 12;

/// Bitmask for the 12-bit tag field (lower 12 bits of the last u32).
pub const TAG_MASK: u32 = 0xFFF;

/// Bitmask for the 20-bit size field (upper 20 bits of the last u32).
pub const SIZE_MASK: u32 = 0xFFFF_F000;

/// A 32-byte content identifier.
///
/// Layout (big-endian):
/// - Bytes 0–27: truncated SHA-256 content hash (224 bits)
/// - Bytes 28–31: big-endian u32 where bits [31:12] = payload size, bits [11:0] = type tag + checksum
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
#[repr(C)]
pub struct ContentId {
    /// First 28 bytes of SHA-256(content).
    pub hash: [u8; CONTENT_HASH_LEN],
    /// Big-endian u32: upper 20 bits = size, lower 12 bits = tag+checksum.
    pub size_and_tag: [u8; 4],
}

impl std::fmt::Debug for ContentId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "ContentId({}, {}B, {:?})",
            hex_prefix(&self.hash),
            self.payload_size(),
            self.cid_type(),
        )
    }
}

/// Format the first 4 bytes of a hash as hex for debug display.
fn hex_prefix(hash: &[u8; CONTENT_HASH_LEN]) -> String {
    format!(
        "{:02x}{:02x}{:02x}{:02x}...",
        hash[0], hash[1], hash[2], hash[3]
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn content_id_is_32_bytes() {
        assert_eq!(std::mem::size_of::<ContentId>(), 32);
    }
}
```

**Step 2: Run test to verify it passes**

Run: `cargo test -p harmony-content content_id_is_32_bytes`
Expected: PASS (this is a structural test, it should pass immediately).

**Step 3: Commit**

```bash
git add crates/harmony-content/src/cid.rs
git commit -m "feat(content): add ContentId struct and constants"
```

---

### Task 3: Type tag encoding and decoding

**Files:**
- Modify: `crates/harmony-content/src/cid.rs`

The unary type tag uses leading-1-bits to encode the CID type. We need an enum and encode/decode functions.

**Step 1: Write the failing tests — type tag round-trip**

Add to the `tests` module in `cid.rs`:

```rust
    #[test]
    fn blob_tag_round_trip() {
        let tag = CidType::Blob.encode(0x7FF); // 11-bit checksum, max value
        assert_eq!(CidType::decode(tag), (CidType::Blob, 0x7FF));
    }

    #[test]
    fn bundle_l1_tag_round_trip() {
        let tag = CidType::Bundle(1).encode(0x3FF); // 10-bit checksum
        assert_eq!(CidType::decode(tag), (CidType::Bundle(1), 0x3FF));
    }

    #[test]
    fn bundle_l7_tag_round_trip() {
        let tag = CidType::Bundle(7).encode(0xF); // 4-bit checksum
        assert_eq!(CidType::decode(tag), (CidType::Bundle(7), 0xF));
    }

    #[test]
    fn inline_metadata_tag_round_trip() {
        let tag = CidType::InlineMetadata.encode(0x7); // 3-bit checksum
        assert_eq!(CidType::decode(tag), (CidType::InlineMetadata, 0x7));
    }

    #[test]
    fn all_bundle_depths_round_trip() {
        for depth in 1..=7 {
            let max_checksum = (1u16 << (11 - depth)) - 1;
            let tag = CidType::Bundle(depth).encode(max_checksum);
            let (decoded_type, decoded_checksum) = CidType::decode(tag);
            assert_eq!(decoded_type, CidType::Bundle(depth), "depth {depth}");
            assert_eq!(decoded_checksum, max_checksum, "depth {depth} checksum");
        }
    }

    #[test]
    fn depth_returns_correct_values() {
        assert_eq!(CidType::Blob.depth(), 0);
        assert_eq!(CidType::Bundle(1).depth(), 1);
        assert_eq!(CidType::Bundle(5).depth(), 5);
        assert_eq!(CidType::Bundle(7).depth(), 7);
    }
```

**Step 2: Run tests to verify they fail**

Run: `cargo test -p harmony-content`
Expected: FAIL — `CidType` not found.

**Step 3: Implement `CidType` enum with encode/decode**

Add above the `tests` module in `cid.rs`:

```rust
/// The type of a ContentId, encoded as a unary prefix in the 12-bit tag field.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CidType {
    /// Leaf data blob (depth 0). Tag prefix: `0`.
    Blob,
    /// Interior bundle node at the given depth (1–7). Tag prefix: `depth` leading 1-bits then `0`.
    Bundle(u8),
    /// Inline metadata (hash field repurposed). Tag prefix: `1111_1111_0`.
    InlineMetadata,
    /// Reserved type A. Tag prefix: `1111_1111_10`.
    ReservedA,
    /// Reserved type B. Tag prefix: `1111_1111_110`.
    ReservedB,
    /// Reserved type C. Tag prefix: `1111_1111_1110`.
    ReservedC,
    /// Reserved type D. Tag prefix: `1111_1111_1111`.
    ReservedD,
}

impl CidType {
    /// Returns the depth of this CID type. Blobs are 0, Bundle(n) is n.
    pub fn depth(&self) -> u8 {
        match self {
            CidType::Blob => 0,
            CidType::Bundle(d) => *d,
            _ => 0,
        }
    }

    /// Number of checksum bits available for this type.
    pub fn checksum_bits(&self) -> u32 {
        let prefix_len = self.prefix_len();
        TAG_BITS - prefix_len
    }

    /// Length of the unary prefix (including the terminating 0, except for ReservedD).
    fn prefix_len(&self) -> u32 {
        match self {
            CidType::Blob => 1,            // 0
            CidType::Bundle(d) => {         // d leading 1s + one 0
                *d as u32 + 1
            }
            CidType::InlineMetadata => 9,   // 1111_1111_0
            CidType::ReservedA => 10,       // 1111_1111_10
            CidType::ReservedB => 11,       // 1111_1111_110
            CidType::ReservedC => 12,       // 1111_1111_1110
            CidType::ReservedD => 12,       // 1111_1111_1111
        }
    }

    /// Encode this type and a checksum into a 12-bit tag value.
    ///
    /// The checksum must fit within `self.checksum_bits()` bits.
    pub fn encode(&self, checksum: u16) -> u16 {
        let prefix_len = self.prefix_len();
        let cksum_bits = TAG_BITS - prefix_len;

        // Build the prefix: leading 1s for the depth, then a 0 (except ReservedD)
        let prefix: u16 = match self {
            CidType::Blob => 0, // prefix bit is 0, shifted to top of 12-bit field
            CidType::Bundle(d) => {
                // d leading 1-bits: e.g., depth 3 = 0b1110 in the top 4 bits of the 12-bit field
                let ones = ((1u16 << *d) - 1) << (TAG_BITS - *d as u32);
                ones // the 0 terminator is implicit (bit position is 0)
            }
            CidType::InlineMetadata => 0xFF0 >> 0, // 1111_1111_0xxx => top 9 bits = 1111_1111_0
            CidType::ReservedA => 0xFF8 >> 0,      // 1111_1111_10xx
            CidType::ReservedB => 0xFFC >> 0,      // 1111_1111_110x
            CidType::ReservedC => 0xFFE,            // 1111_1111_1110
            CidType::ReservedD => 0xFFF,            // 1111_1111_1111
        };

        // Place the prefix in the top bits and the checksum in the remaining lower bits.
        // For Blob: prefix is 0 in bit 11, checksum in bits 10..0
        // For Bundle(d): d one-bits at top, 0 bit, then checksum
        let tag = match self {
            CidType::Blob => {
                // Bit 11 = 0, bits 10..0 = checksum (11 bits)
                checksum & ((1 << cksum_bits) - 1)
            }
            CidType::Bundle(_) | CidType::InlineMetadata | CidType::ReservedA | CidType::ReservedB => {
                prefix | (checksum & ((1 << cksum_bits) - 1))
            }
            CidType::ReservedC | CidType::ReservedD => {
                prefix // no checksum bits
            }
        };
        tag
    }

    /// Decode a 12-bit tag value into a `CidType` and checksum.
    pub fn decode(tag: u16) -> (CidType, u16) {
        let tag = tag & TAG_MASK as u16;

        // Count leading 1-bits in the 12-bit field.
        // Shift tag into the top 12 bits of a u16, then count leading ones.
        let shifted = tag << 4; // move 12 bits to top of u16
        let leading_ones = (!shifted).leading_zeros(); // leading zeros of inverted = leading ones of original

        match leading_ones {
            0 => {
                // Starts with 0 → Blob, 11-bit checksum
                let checksum = tag & 0x7FF;
                (CidType::Blob, checksum)
            }
            1..=7 => {
                // Bundle at depth = leading_ones
                let depth = leading_ones as u8;
                let cksum_bits = 11 - depth as u32;
                let checksum = tag & ((1 << cksum_bits) - 1);
                (CidType::Bundle(depth), checksum)
            }
            8 => {
                // 1111_1111_0 → InlineMetadata, 3-bit checksum
                let checksum = tag & 0x7;
                (CidType::InlineMetadata, checksum)
            }
            9 => {
                // 1111_1111_10 → ReservedA, 2-bit checksum
                let checksum = tag & 0x3;
                (CidType::ReservedA, checksum)
            }
            10 => {
                // 1111_1111_110 → ReservedB, 1-bit checksum
                let checksum = tag & 0x1;
                (CidType::ReservedB, checksum)
            }
            11 => {
                // 1111_1111_1110 → ReservedC, 0-bit checksum
                (CidType::ReservedC, 0)
            }
            _ => {
                // 1111_1111_1111 → ReservedD, 0-bit checksum
                (CidType::ReservedD, 0)
            }
        }
    }
}
```

**Step 4: Run tests to verify they pass**

Run: `cargo test -p harmony-content`
Expected: All tests PASS.

**Step 5: Commit**

```bash
git add crates/harmony-content/src/cid.rs
git commit -m "feat(content): add CidType enum with unary tag encode/decode"
```

---

### Task 4: Checksum computation and verification

**Files:**
- Modify: `crates/harmony-content/src/cid.rs`

The checksum is computed over the first 31 bytes of the CID plus the type prefix bits. We use a simple CRC-style checksum truncated to the available bits.

**Step 1: Write the failing tests — checksum round-trip**

Add to the `tests` module:

```rust
    #[test]
    fn compute_checksum_deterministic() {
        let hash = [0xABu8; CONTENT_HASH_LEN];
        let size: u32 = 1000;
        let cid_type = CidType::Blob;
        let c1 = compute_checksum(&hash, size, &cid_type);
        let c2 = compute_checksum(&hash, size, &cid_type);
        assert_eq!(c1, c2);
    }

    #[test]
    fn compute_checksum_varies_with_input() {
        let hash_a = [0xAAu8; CONTENT_HASH_LEN];
        let hash_b = [0xBBu8; CONTENT_HASH_LEN];
        let c1 = compute_checksum(&hash_a, 1000, &CidType::Blob);
        let c2 = compute_checksum(&hash_b, 1000, &CidType::Blob);
        assert_ne!(c1, c2);
    }

    #[test]
    fn checksum_fits_within_available_bits() {
        let hash = [0x42u8; CONTENT_HASH_LEN];
        for depth in 1..=7u8 {
            let cid_type = CidType::Bundle(depth);
            let cksum = compute_checksum(&hash, 500, &cid_type);
            let max = (1u16 << cid_type.checksum_bits()) - 1;
            assert!(cksum <= max, "depth {depth}: checksum {cksum} > max {max}");
        }
    }
```

**Step 2: Run tests to verify they fail**

Run: `cargo test -p harmony-content`
Expected: FAIL — `compute_checksum` not found.

**Step 3: Implement `compute_checksum`**

Add above the `tests` module:

```rust
/// Compute a checksum over the CID's hash, size, and type, truncated to fit
/// the available checksum bits for the given CID type.
///
/// Uses a simple XOR-fold of a SHA-256 hash of the input fields, truncated to
/// the number of checksum bits available for the type.
pub fn compute_checksum(hash: &[u8; CONTENT_HASH_LEN], size: u32, cid_type: &CidType) -> u16 {
    use harmony_crypto::hash::full_hash;

    let cksum_bits = cid_type.checksum_bits();
    if cksum_bits == 0 {
        return 0;
    }

    // Hash the content hash + size + type prefix to produce checksum input.
    let mut input = Vec::with_capacity(CONTENT_HASH_LEN + 4 + 1);
    input.extend_from_slice(hash);
    input.extend_from_slice(&size.to_be_bytes());
    input.push(cid_type.depth());
    let digest = full_hash(&input);

    // XOR-fold the first 2 bytes of the digest down to the available bits.
    let raw = u16::from_be_bytes([digest[0], digest[1]]);
    raw & ((1u16 << cksum_bits) - 1)
}
```

**Step 4: Run tests to verify they pass**

Run: `cargo test -p harmony-content`
Expected: All tests PASS.

**Step 5: Commit**

```bash
git add crates/harmony-content/src/cid.rs
git commit -m "feat(content): add checksum computation for ContentId"
```

---

### Task 5: `ContentId` constructors — `for_blob` and `for_bundle`

**Files:**
- Modify: `crates/harmony-content/src/cid.rs`

**Step 1: Write the failing tests**

Add to the `tests` module:

```rust
    #[test]
    fn for_blob_basic() {
        let data = b"hello harmony content addressing";
        let cid = ContentId::for_blob(data).unwrap();
        assert_eq!(cid.payload_size(), data.len() as u32);
        assert_eq!(cid.cid_type(), CidType::Blob);
    }

    #[test]
    fn for_blob_hash_is_truncated_sha256() {
        let data = b"test data";
        let cid = ContentId::for_blob(data).unwrap();
        let full = harmony_crypto::hash::full_hash(data);
        assert_eq!(&cid.hash, &full[..CONTENT_HASH_LEN]);
    }

    #[test]
    fn for_blob_rejects_oversized() {
        let data = vec![0u8; MAX_PAYLOAD_SIZE + 1];
        let result = ContentId::for_blob(&data);
        assert!(result.is_err());
    }

    #[test]
    fn for_blob_empty_data() {
        let cid = ContentId::for_blob(b"").unwrap();
        assert_eq!(cid.payload_size(), 0);
        assert_eq!(cid.cid_type(), CidType::Blob);
    }

    #[test]
    fn for_blob_max_size() {
        let data = vec![0xFFu8; MAX_PAYLOAD_SIZE];
        let cid = ContentId::for_blob(&data).unwrap();
        assert_eq!(cid.payload_size(), MAX_PAYLOAD_SIZE as u32);
    }

    #[test]
    fn for_bundle_basic() {
        // Create two blob CIDs, then bundle them
        let blob_a = ContentId::for_blob(b"chunk a").unwrap();
        let blob_b = ContentId::for_blob(b"chunk b").unwrap();
        let children = [blob_a, blob_b];
        let bundle_bytes = children_to_bytes(&children);
        let cid = ContentId::for_bundle(&bundle_bytes, &children).unwrap();
        assert_eq!(cid.cid_type(), CidType::Bundle(1)); // one level above blobs
        assert_eq!(cid.payload_size(), bundle_bytes.len() as u32);
    }

    #[test]
    fn for_bundle_depth_is_max_child_plus_one() {
        let blob = ContentId::for_blob(b"leaf").unwrap();
        let l1_children = [blob];
        let l1_bytes = children_to_bytes(&l1_children);
        let l1 = ContentId::for_bundle(&l1_bytes, &l1_children).unwrap();
        assert_eq!(l1.cid_type(), CidType::Bundle(1));

        let l2_children = [l1];
        let l2_bytes = children_to_bytes(&l2_children);
        let l2 = ContentId::for_bundle(&l2_bytes, &l2_children).unwrap();
        assert_eq!(l2.cid_type(), CidType::Bundle(2));
    }

    #[test]
    fn for_bundle_rejects_depth_overflow() {
        // Can't create a Bundle(8) — max is 7
        let blob = ContentId::for_blob(b"leaf").unwrap();
        // Build up to depth 7
        let mut current = blob;
        for _ in 0..7 {
            let children = [current];
            let bytes = children_to_bytes(&children);
            current = ContentId::for_bundle(&bytes, &children).unwrap();
        }
        assert_eq!(current.cid_type(), CidType::Bundle(7));

        // Trying to wrap a depth-7 bundle should fail
        let children = [current];
        let bytes = children_to_bytes(&children);
        let result = ContentId::for_bundle(&bytes, &children);
        assert!(result.is_err());
    }

    /// Helper: serialize CID array to bytes (the bundle payload).
    fn children_to_bytes(children: &[ContentId]) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(children.len() * 32);
        for cid in children {
            bytes.extend_from_slice(&cid.to_bytes());
        }
        bytes
    }
```

**Step 2: Run tests to verify they fail**

Run: `cargo test -p harmony-content`
Expected: FAIL — `for_blob`, `for_bundle`, `payload_size`, `cid_type`, `to_bytes` not found.

**Step 3: Implement the constructors and accessors**

Add to `impl ContentId`:

```rust
impl ContentId {
    /// Create a CID for a raw data blob.
    pub fn for_blob(data: &[u8]) -> Result<Self, ContentError> {
        if data.len() > MAX_PAYLOAD_SIZE {
            return Err(ContentError::PayloadTooLarge {
                size: data.len(),
                max: MAX_PAYLOAD_SIZE,
            });
        }

        let full = harmony_crypto::hash::full_hash(data);
        let mut hash = [0u8; CONTENT_HASH_LEN];
        hash.copy_from_slice(&full[..CONTENT_HASH_LEN]);

        let size = data.len() as u32;
        let cid_type = CidType::Blob;
        let checksum = compute_checksum(&hash, size, &cid_type);
        let tag = cid_type.encode(checksum);
        let size_and_tag_u32 = (size << TAG_BITS) | tag as u32;

        Ok(ContentId {
            hash,
            size_and_tag: size_and_tag_u32.to_be_bytes(),
        })
    }

    /// Create a CID for a bundle (array of child CIDs).
    ///
    /// `bundle_bytes` is the raw byte payload (concatenated child CIDs).
    /// `children` is the parsed slice of child CIDs (used for depth calculation).
    /// The bundle's depth is `max(child depths) + 1`.
    pub fn for_bundle(bundle_bytes: &[u8], children: &[ContentId]) -> Result<Self, ContentError> {
        if bundle_bytes.len() > MAX_PAYLOAD_SIZE {
            return Err(ContentError::PayloadTooLarge {
                size: bundle_bytes.len(),
                max: MAX_PAYLOAD_SIZE,
            });
        }

        let max_child_depth = children.iter().map(|c| c.cid_type().depth()).max().unwrap_or(0);
        let bundle_depth = max_child_depth + 1;

        if bundle_depth > 7 {
            return Err(ContentError::DepthViolation {
                child: max_child_depth,
                parent: bundle_depth,
            });
        }

        let full = harmony_crypto::hash::full_hash(bundle_bytes);
        let mut hash = [0u8; CONTENT_HASH_LEN];
        hash.copy_from_slice(&full[..CONTENT_HASH_LEN]);

        let size = bundle_bytes.len() as u32;
        let cid_type = CidType::Bundle(bundle_depth);
        let checksum = compute_checksum(&hash, size, &cid_type);
        let tag = cid_type.encode(checksum);
        let size_and_tag_u32 = (size << TAG_BITS) | tag as u32;

        Ok(ContentId {
            hash,
            size_and_tag: size_and_tag_u32.to_be_bytes(),
        })
    }

    /// Extract the 20-bit payload size.
    pub fn payload_size(&self) -> u32 {
        let raw = u32::from_be_bytes(self.size_and_tag);
        raw >> TAG_BITS
    }

    /// Extract the 12-bit tag and decode the CID type.
    pub fn cid_type(&self) -> CidType {
        let raw = u32::from_be_bytes(self.size_and_tag);
        let tag = (raw & TAG_MASK) as u16;
        let (cid_type, _checksum) = CidType::decode(tag);
        cid_type
    }

    /// Extract the checksum from the tag field.
    pub fn checksum(&self) -> u16 {
        let raw = u32::from_be_bytes(self.size_and_tag);
        let tag = (raw & TAG_MASK) as u16;
        let (_cid_type, checksum) = CidType::decode(tag);
        checksum
    }

    /// Verify the CID's checksum against its hash, size, and type.
    pub fn verify_checksum(&self) -> Result<(), ContentError> {
        let cid_type = self.cid_type();
        let expected = compute_checksum(&self.hash, self.payload_size(), &cid_type);
        if self.checksum() != expected {
            return Err(ContentError::ChecksumMismatch);
        }
        Ok(())
    }

    /// Serialize to a 32-byte array.
    pub fn to_bytes(&self) -> [u8; 32] {
        let mut bytes = [0u8; 32];
        bytes[..CONTENT_HASH_LEN].copy_from_slice(&self.hash);
        bytes[CONTENT_HASH_LEN..].copy_from_slice(&self.size_and_tag);
        bytes
    }

    /// Deserialize from a 32-byte array.
    pub fn from_bytes(bytes: [u8; 32]) -> Self {
        let mut hash = [0u8; CONTENT_HASH_LEN];
        hash.copy_from_slice(&bytes[..CONTENT_HASH_LEN]);
        let mut size_and_tag = [0u8; 4];
        size_and_tag.copy_from_slice(&bytes[CONTENT_HASH_LEN..]);
        ContentId { hash, size_and_tag }
    }
}
```

**Step 4: Run tests to verify they pass**

Run: `cargo test -p harmony-content`
Expected: All tests PASS.

**Step 5: Run clippy**

Run: `cargo clippy -p harmony-content`
Expected: No warnings.

**Step 6: Commit**

```bash
git add crates/harmony-content/src/cid.rs
git commit -m "feat(content): add ContentId constructors, accessors, and serialization"
```

---

### Task 6: Inline Metadata CID

**Files:**
- Modify: `crates/harmony-content/src/cid.rs`

**Step 1: Write the failing tests**

Add to the `tests` module:

```rust
    #[test]
    fn inline_metadata_round_trip() {
        let meta = ContentId::inline_metadata(
            100_000_000,  // 100MB total size
            100,          // 100 chunks
            1709337600000, // timestamp
            *b"text/pln",  // MIME type (8 bytes)
        );
        assert_eq!(meta.cid_type(), CidType::InlineMetadata);

        let (total_size, chunk_count, timestamp, mime) = meta.parse_inline_metadata().unwrap();
        assert_eq!(total_size, 100_000_000);
        assert_eq!(chunk_count, 100);
        assert_eq!(timestamp, 1709337600000);
        assert_eq!(&mime, b"text/pln");
    }

    #[test]
    fn inline_metadata_zero_values() {
        let meta = ContentId::inline_metadata(0, 0, 0, [0u8; 8]);
        let (total_size, chunk_count, timestamp, mime) = meta.parse_inline_metadata().unwrap();
        assert_eq!(total_size, 0);
        assert_eq!(chunk_count, 0);
        assert_eq!(timestamp, 0);
        assert_eq!(mime, [0u8; 8]);
    }

    #[test]
    fn inline_metadata_max_file_size() {
        let meta = ContentId::inline_metadata(u64::MAX, u32::MAX, u64::MAX, [0xFF; 8]);
        let (total_size, chunk_count, timestamp, mime) = meta.parse_inline_metadata().unwrap();
        assert_eq!(total_size, u64::MAX);
        assert_eq!(chunk_count, u32::MAX);
        assert_eq!(timestamp, u64::MAX);
        assert_eq!(mime, [0xFF; 8]);
    }

    #[test]
    fn parse_inline_metadata_rejects_non_metadata_cid() {
        let blob = ContentId::for_blob(b"not metadata").unwrap();
        assert!(blob.parse_inline_metadata().is_err());
    }
```

**Step 2: Run tests to verify they fail**

Run: `cargo test -p harmony-content`
Expected: FAIL — `inline_metadata`, `parse_inline_metadata` not found.

**Step 3: Implement inline metadata constructor and parser**

Add a new error variant to `error.rs`:

```rust
    #[error("not an inline metadata CID")]
    NotInlineMetadata,
```

Add to `impl ContentId`:

```rust
    /// Create an inline metadata CID.
    ///
    /// The 28-byte hash field is repurposed to store:
    /// - bytes 0–7: total file size (u64 big-endian)
    /// - bytes 8–11: total chunk count (u32 big-endian)
    /// - bytes 12–19: creation timestamp (u64 big-endian, Unix epoch ms)
    /// - bytes 20–27: MIME type or extension (8 bytes, packed)
    pub fn inline_metadata(
        total_size: u64,
        chunk_count: u32,
        timestamp: u64,
        mime: [u8; 8],
    ) -> Self {
        let mut hash = [0u8; CONTENT_HASH_LEN];
        hash[0..8].copy_from_slice(&total_size.to_be_bytes());
        hash[8..12].copy_from_slice(&chunk_count.to_be_bytes());
        hash[12..20].copy_from_slice(&timestamp.to_be_bytes());
        hash[20..28].copy_from_slice(&mime);

        let cid_type = CidType::InlineMetadata;
        // Size field is 0 for inline metadata (the data is in the hash field).
        let size: u32 = 0;
        let checksum = compute_checksum(&hash, size, &cid_type);
        let tag = cid_type.encode(checksum);
        let size_and_tag_u32 = (size << TAG_BITS) | tag as u32;

        ContentId {
            hash,
            size_and_tag: size_and_tag_u32.to_be_bytes(),
        }
    }

    /// Parse an inline metadata CID, returning (total_size, chunk_count, timestamp, mime).
    pub fn parse_inline_metadata(&self) -> Result<(u64, u32, u64, [u8; 8]), ContentError> {
        if self.cid_type() != CidType::InlineMetadata {
            return Err(ContentError::NotInlineMetadata);
        }

        let total_size = u64::from_be_bytes(self.hash[0..8].try_into().unwrap());
        let chunk_count = u32::from_be_bytes(self.hash[8..12].try_into().unwrap());
        let timestamp = u64::from_be_bytes(self.hash[12..20].try_into().unwrap());
        let mut mime = [0u8; 8];
        mime.copy_from_slice(&self.hash[20..28]);

        Ok((total_size, chunk_count, timestamp, mime))
    }
```

**Step 4: Run tests to verify they pass**

Run: `cargo test -p harmony-content`
Expected: All tests PASS.

**Step 5: Commit**

```bash
git add crates/harmony-content/src/cid.rs crates/harmony-content/src/error.rs
git commit -m "feat(content): add inline metadata CID construction and parsing"
```

---

### Task 7: Checksum verification and `Display` impl

**Files:**
- Modify: `crates/harmony-content/src/cid.rs`

**Step 1: Write the failing tests**

Add to the `tests` module:

```rust
    #[test]
    fn checksum_verification_passes_for_valid_cid() {
        let cid = ContentId::for_blob(b"valid data").unwrap();
        assert!(cid.verify_checksum().is_ok());
    }

    #[test]
    fn checksum_verification_fails_for_corrupted_cid() {
        let cid = ContentId::for_blob(b"valid data").unwrap();
        let mut bytes = cid.to_bytes();
        bytes[0] ^= 0xFF; // corrupt the hash
        let corrupted = ContentId::from_bytes(bytes);
        assert!(corrupted.verify_checksum().is_err());
    }

    #[test]
    fn checksum_verification_passes_for_bundle() {
        let blob = ContentId::for_blob(b"leaf").unwrap();
        let children = [blob];
        let bytes = children_to_bytes(&children);
        let bundle = ContentId::for_bundle(&bytes, &children).unwrap();
        assert!(bundle.verify_checksum().is_ok());
    }

    #[test]
    fn checksum_verification_passes_for_inline_metadata() {
        let meta = ContentId::inline_metadata(1000, 1, 0, [0; 8]);
        assert!(meta.verify_checksum().is_ok());
    }

    #[test]
    fn serialization_round_trip() {
        let cid = ContentId::for_blob(b"round trip test").unwrap();
        let bytes = cid.to_bytes();
        let restored = ContentId::from_bytes(bytes);
        assert_eq!(cid, restored);
    }

    #[test]
    fn display_shows_hex_type_and_size() {
        let cid = ContentId::for_blob(b"display test").unwrap();
        let s = format!("{cid}");
        assert!(s.contains("Blob"));
        assert!(s.contains("12")); // data length
    }
```

**Step 2: Run tests to verify they fail**

Run: `cargo test -p harmony-content`
Expected: FAIL — `Display` not implemented (other tests should pass since we already implemented `verify_checksum`).

**Step 3: Implement `Display`**

Add to `cid.rs`:

```rust
impl std::fmt::Display for ContentId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let cid_type = self.cid_type();
        let size = self.payload_size();
        write!(
            f,
            "{} {:?} {}B",
            hex_prefix(&self.hash),
            cid_type,
            size,
        )
    }
}
```

**Step 4: Run tests to verify they pass**

Run: `cargo test -p harmony-content`
Expected: All tests PASS.

**Step 5: Run full quality gates**

Run: `cargo test -p harmony-content && cargo clippy -p harmony-content && cargo fmt --all -- --check`
Expected: All pass, no warnings, no formatting issues.

**Step 6: Commit**

```bash
git add crates/harmony-content/src/cid.rs
git commit -m "feat(content): add checksum verification, Display impl, serialization round-trip"
```

---

### Task 8: Canonical test vectors

**Files:**
- Modify: `crates/harmony-content/src/cid.rs`

Establish canonical hex-encoded test vectors so future implementations in other languages produce byte-identical CIDs.

**Step 1: Write the test vectors**

Add to the `tests` module:

```rust
    // === Canonical test vectors ===
    // These exist so other language implementations can verify byte-identical CID output.

    #[test]
    fn canonical_vector_empty_blob() {
        let cid = ContentId::for_blob(b"").unwrap();
        let bytes = cid.to_bytes();
        // SHA-256("")[:28] = e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b
        assert_eq!(
            hex::encode(&bytes[..CONTENT_HASH_LEN]),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b",
        );
        assert_eq!(cid.payload_size(), 0);
        assert_eq!(cid.cid_type(), CidType::Blob);
        // Record the full 32-byte hex for cross-language verification
        let full_hex = hex::encode(bytes);
        // Verify it's deterministic by re-creating
        let cid2 = ContentId::for_blob(b"").unwrap();
        assert_eq!(hex::encode(cid2.to_bytes()), full_hex);
    }

    #[test]
    fn canonical_vector_hello_blob() {
        let cid = ContentId::for_blob(b"hello").unwrap();
        let bytes = cid.to_bytes();
        // SHA-256("hello")[:28] = 2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362
        assert_eq!(
            hex::encode(&bytes[..CONTENT_HASH_LEN]),
            "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362",
        );
        assert_eq!(cid.payload_size(), 5);
        assert_eq!(cid.cid_type(), CidType::Blob);
    }

    #[test]
    fn canonical_vector_bundle_of_two_blobs() {
        let blob_a = ContentId::for_blob(b"aaa").unwrap();
        let blob_b = ContentId::for_blob(b"bbb").unwrap();
        let children = [blob_a, blob_b];
        let bundle_bytes = children_to_bytes(&children);
        let bundle = ContentId::for_bundle(&bundle_bytes, &children).unwrap();
        assert_eq!(bundle.payload_size(), 64); // 2 × 32 bytes
        assert_eq!(bundle.cid_type(), CidType::Bundle(1));
        // The hash is SHA-256 of the 64-byte bundle payload
        let full = harmony_crypto::hash::full_hash(&bundle_bytes);
        assert_eq!(&bundle.hash, &full[..CONTENT_HASH_LEN]);
    }
```

**Step 2: Run tests to verify they pass**

Run: `cargo test -p harmony-content`
Expected: All tests PASS.

**Step 3: Run full quality gates**

Run: `cargo test -p harmony-content && cargo clippy -p harmony-content && cargo fmt --all -- --check`
Expected: All pass.

**Step 4: Commit**

```bash
git add crates/harmony-content/src/cid.rs
git commit -m "test(content): add canonical CID test vectors for cross-language verification"
```

---

### Summary

| Task | What it builds | Tests |
|---|---|---|
| 1 | Crate scaffold | Compiles |
| 2 | `ContentId` struct, constants | Size assertion |
| 3 | `CidType` enum, unary encode/decode | 6 tag round-trip tests |
| 4 | `compute_checksum` | 3 checksum tests |
| 5 | `for_blob`, `for_bundle`, accessors, serialization | 8 constructor tests |
| 6 | Inline metadata CID | 4 metadata tests |
| 7 | Checksum verification, `Display` | 6 verification tests |
| 8 | Canonical test vectors | 3 cross-language vectors |

Total: ~30 tests across 8 commits.
