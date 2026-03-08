# ContentId Signal Flags Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add 3 signal bits (encrypted, ephemeral, hash algorithm) to the top 3 bits of ContentId's hash field, enabling storage-class routing from the address alone.

**Architecture:** Add `sha224_hash()` to harmony-crypto (the `sha2` crate already includes SHA-224). Add `ContentFlags` struct to harmony-content's cid.rs. Modify `for_blob`/`for_bundle` to accept flags, mask hash[0], and set flag bits. Modify `verify_hash` to select algorithm and mask bits. Update all ~70 call sites to pass `ContentFlags::default()`.

**Tech Stack:** Rust, sha2 crate (SHA-224 already available), harmony-crypto, harmony-content

**Design doc:** `docs/plans/2026-03-07-content-flags-design.md`

---

### Task 1: Add SHA-224 to harmony-crypto

**Files:**
- Modify: `crates/harmony-crypto/src/hash.rs`

**Step 1: Write the failing test**

Add to the `tests` module in `hash.rs`:

```rust
#[test]
fn sha224_known_vector_empty() {
    // SHA-224("") = d14a028c2a3a2bc9476102bb288234c415a2b01f828ea62ac5b3e42f
    let hash = sha224_hash(b"");
    assert_eq!(
        hex::encode(hash),
        "d14a028c2a3a2bc9476102bb288234c415a2b01f828ea62ac5b3e42f"
    );
}

#[test]
fn sha224_known_vector_abc() {
    // SHA-224("abc") = 23097d223405d8228642a477bda255b32aadbce4bda0b3f7e36c9da7
    let hash = sha224_hash(b"abc");
    assert_eq!(
        hex::encode(hash),
        "23097d223405d8228642a477bda255b32aadbce4bda0b3f7e36c9da7"
    );
}

#[test]
fn sha224_deterministic() {
    let h1 = sha224_hash(b"hello harmony");
    let h2 = sha224_hash(b"hello harmony");
    assert_eq!(h1, h2);
}

#[test]
fn sha224_differs_from_sha256() {
    let s256 = full_hash(b"test");
    let s224 = sha224_hash(b"test");
    // Different lengths, but also the overlapping prefix should differ
    // (SHA-224 uses different initial values than SHA-256).
    assert_ne!(&s256[..28], &s224[..]);
}
```

**Step 2: Run tests to verify they fail**

Run: `cargo test -p harmony-crypto sha224 -- --nocapture`
Expected: FAIL — `sha224_hash` not found.

**Step 3: Implement sha224_hash**

Add to `hash.rs`, after the existing `use` statement on line 1:

```rust
use sha2::{Digest, Sha256, Sha224};
```

(Replace the existing `use sha2::{Digest, Sha256};`)

Then add the function after `name_hash`:

```rust
/// SHA-224 hash length in bytes.
pub const SHA224_HASH_LENGTH: usize = 28;

/// Compute the SHA-224 hash of `data` (28 bytes).
///
/// SHA-224 uses different initialization vectors from SHA-256, making them
/// independent hash families. Used by ContentId for power-of-choice collision
/// resistance (the `alt_hash` flag).
pub fn sha224_hash(data: &[u8]) -> [u8; SHA224_HASH_LENGTH] {
    let mut hasher = Sha224::new();
    hasher.update(data);
    hasher.finalize().into()
}
```

**Step 4: Run tests to verify they pass**

Run: `cargo test -p harmony-crypto`
Expected: All tests pass including the 4 new SHA-224 tests.

**Step 5: Commit**

```bash
git add crates/harmony-crypto/src/hash.rs
git commit -m "feat(crypto): add sha224_hash for ContentId power-of-choice"
```

---

### Task 2: Add ContentFlags struct

**Files:**
- Modify: `crates/harmony-content/src/cid.rs`

**Step 1: Write the failing tests**

Add to the `tests` module in `cid.rs`:

```rust
// -----------------------------------------------------------------------
// ContentFlags
// -----------------------------------------------------------------------

#[test]
fn content_flags_default_is_cleartext_durable_sha256() {
    let flags = ContentFlags::default();
    assert!(!flags.encrypted);
    assert!(!flags.ephemeral);
    assert!(!flags.alt_hash);
}

#[test]
fn content_flags_to_bits_round_trip() {
    for e in [false, true] {
        for d in [false, true] {
            for a in [false, true] {
                let flags = ContentFlags { encrypted: e, ephemeral: d, alt_hash: a };
                let bits = flags.to_bits();
                let restored = ContentFlags::from_bits(bits);
                assert_eq!(flags, restored, "round-trip failed for e={e}, d={d}, a={a}");
            }
        }
    }
}

#[test]
fn content_flags_bits_are_top_3() {
    // encrypted=1, ephemeral=0, alt_hash=0 → bit 7 set → 0b1000_0000 = 0x80
    let flags = ContentFlags { encrypted: true, ephemeral: false, alt_hash: false };
    assert_eq!(flags.to_bits(), 0x80);

    // encrypted=0, ephemeral=1, alt_hash=0 → bit 6 set → 0b0100_0000 = 0x40
    let flags = ContentFlags { encrypted: false, ephemeral: true, alt_hash: false };
    assert_eq!(flags.to_bits(), 0x40);

    // encrypted=0, ephemeral=0, alt_hash=1 → bit 5 set → 0b0010_0000 = 0x20
    let flags = ContentFlags { encrypted: false, ephemeral: false, alt_hash: true };
    assert_eq!(flags.to_bits(), 0x20);

    // all set → 0b1110_0000 = 0xE0
    let flags = ContentFlags { encrypted: true, ephemeral: true, alt_hash: true };
    assert_eq!(flags.to_bits(), 0xE0);
}
```

**Step 2: Run tests to verify they fail**

Run: `cargo test -p harmony-content content_flags -- --nocapture`
Expected: FAIL — `ContentFlags` not found.

**Step 3: Implement ContentFlags**

Add to `cid.rs` after the constants (after line 14, before the `ContentId` struct):

```rust
/// Storage-class flags embedded in the top 3 bits of a ContentId's hash byte 0.
///
/// These flags are part of the content's identity: same bytes with different
/// flags produce different ContentIds.
///
/// ```text
/// hash[0]: [E:1][D:1][A:1][hash_bits:5]
///           │    │    └─ 0=SHA-256, 1=SHA-224
///           │    └────── 0=durable, 1=ephemeral
///           └─────────── 0=cleartext, 1=encrypted
/// ```
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct ContentFlags {
    /// Cleartext (false) or encrypted (true).
    /// Encrypted blobs are not scannable for indexing/training and are
    /// quota-attributed to the key owner rather than the network.
    pub encrypted: bool,
    /// Durable (false) or ephemeral (true).
    /// Ephemeral blobs skip LFU caching, never migrate to cold storage,
    /// and are evicted eagerly.
    pub ephemeral: bool,
    /// SHA-256 (false) or SHA-224 (true).
    /// Provides power-of-choice collision resistance: if A=0 collides,
    /// the blob can be re-addressed with A=1 using an independent hash.
    pub alt_hash: bool,
}

impl ContentFlags {
    /// Encode flags into the top 3 bits of a byte.
    pub fn to_bits(self) -> u8 {
        let mut bits = 0u8;
        if self.encrypted { bits |= 0x80; } // bit 7
        if self.ephemeral { bits |= 0x40; } // bit 6
        if self.alt_hash  { bits |= 0x20; } // bit 5
        bits
    }

    /// Decode flags from the top 3 bits of a byte.
    pub fn from_bits(byte: u8) -> Self {
        ContentFlags {
            encrypted: byte & 0x80 != 0,
            ephemeral: byte & 0x40 != 0,
            alt_hash:  byte & 0x20 != 0,
        }
    }
}
```

**Step 4: Run tests to verify they pass**

Run: `cargo test -p harmony-content content_flags`
Expected: All 3 tests pass.

**Step 5: Commit**

```bash
git add crates/harmony-content/src/cid.rs
git commit -m "feat(content): add ContentFlags struct with bit encoding"
```

---

### Task 3: Add flags() accessor to ContentId

**Files:**
- Modify: `crates/harmony-content/src/cid.rs`

**Step 1: Write the failing test**

Add to tests:

```rust
#[test]
fn flags_accessor_reads_top_3_bits() {
    // Manually construct a ContentId with known flag bits in hash[0]
    let mut hash = [0u8; CONTENT_HASH_LEN];
    hash[0] = 0xE0; // all 3 flags set (bits 7,6,5)
    let cid = ContentId {
        hash,
        size_and_tag: [0; 4],
    };
    let flags = cid.flags();
    assert!(flags.encrypted);
    assert!(flags.ephemeral);
    assert!(flags.alt_hash);
}

#[test]
fn flags_accessor_no_flags() {
    let mut hash = [0xFF; CONTENT_HASH_LEN];
    hash[0] = 0x1F; // top 3 bits clear, lower 5 set
    let cid = ContentId {
        hash,
        size_and_tag: [0; 4],
    };
    let flags = cid.flags();
    assert!(!flags.encrypted);
    assert!(!flags.ephemeral);
    assert!(!flags.alt_hash);
}
```

**Step 2: Run tests to verify they fail**

Run: `cargo test -p harmony-content flags_accessor -- --nocapture`
Expected: FAIL — no method `flags` on `ContentId`.

**Step 3: Implement flags()**

Add to the `impl ContentId` block (e.g. after `cid_type()`):

```rust
    /// Extract the storage-class flags from the top 3 bits of hash[0].
    pub fn flags(&self) -> ContentFlags {
        ContentFlags::from_bits(self.hash[0])
    }
```

**Step 4: Run tests**

Run: `cargo test -p harmony-content flags_accessor`
Expected: PASS.

**Step 5: Commit**

```bash
git add crates/harmony-content/src/cid.rs
git commit -m "feat(content): add ContentId::flags() accessor"
```

---

### Task 4: Modify for_blob to accept ContentFlags

This is the core change. Modifies the signature and hash computation, then fixes all call sites.

**Files:**
- Modify: `crates/harmony-content/src/cid.rs` (definition + ~25 test call sites)
- Modify: `crates/harmony-content/src/blob.rs` (line 54 production, ~3 test call sites)
- Modify: `crates/harmony-content/src/reticulum_bridge.rs` (line 43 production, ~1 test call site)
- Modify: `crates/harmony-content/src/storage_tier.rs` (~15 test call sites)
- Modify: `crates/harmony-content/src/cache.rs` (~6 test call sites)
- Modify: `crates/harmony-content/src/zenoh_bridge.rs` (~2 test call sites)
- Modify: `crates/harmony-content/src/sketch.rs` (~1 test call site)
- Modify: `crates/harmony-content/src/lru.rs` (~1 test call site)
- Modify: `crates/harmony-content/src/delta.rs` (~5 test call sites)

**Step 1: Write the failing test**

Add to the test module in `cid.rs`:

```rust
#[test]
fn for_blob_with_default_flags_matches_sha256() {
    let data = b"test flags";
    let cid = ContentId::for_blob(data, ContentFlags::default()).unwrap();
    let flags = cid.flags();
    assert!(!flags.encrypted);
    assert!(!flags.ephemeral);
    assert!(!flags.alt_hash);
    // Hash should be SHA-256[:28] with top 3 bits cleared
    let full = harmony_crypto::hash::full_hash(data);
    assert_eq!(cid.hash[0] & 0x1F, full[0] & 0x1F);
    assert_eq!(&cid.hash[1..], &full[1..CONTENT_HASH_LEN]);
}

#[test]
fn for_blob_with_alt_hash_uses_sha224() {
    let data = b"test alt hash";
    let flags = ContentFlags { encrypted: false, ephemeral: false, alt_hash: true };
    let cid = ContentId::for_blob(data, flags).unwrap();
    assert!(cid.flags().alt_hash);
    // Hash should be SHA-224 with top 3 bits set to flags
    let sha224 = harmony_crypto::hash::sha224_hash(data);
    assert_eq!(cid.hash[0] & 0x1F, sha224[0] & 0x1F);
    assert_eq!(&cid.hash[1..], &sha224[1..CONTENT_HASH_LEN]);
}

#[test]
fn for_blob_same_data_different_flags_different_cid() {
    let data = b"same bytes, different identity";
    let cleartext = ContentId::for_blob(data, ContentFlags::default()).unwrap();
    let encrypted = ContentId::for_blob(data, ContentFlags {
        encrypted: true, ephemeral: false, alt_hash: false,
    }).unwrap();
    let ephemeral = ContentId::for_blob(data, ContentFlags {
        encrypted: false, ephemeral: true, alt_hash: false,
    }).unwrap();
    // All three must be different ContentIds
    assert_ne!(cleartext, encrypted);
    assert_ne!(cleartext, ephemeral);
    assert_ne!(encrypted, ephemeral);
}
```

**Step 2: Run to verify failure**

Run: `cargo test -p harmony-content for_blob_with -- --nocapture`
Expected: FAIL — `for_blob` takes 1 argument but 2 were supplied.

**Step 3: Modify for_blob signature and implementation**

Change `for_blob` in `cid.rs`:

```rust
    /// Create a CID for a raw data blob.
    pub fn for_blob(data: &[u8], flags: ContentFlags) -> Result<Self, ContentError> {
        if data.len() > MAX_PAYLOAD_SIZE {
            return Err(ContentError::PayloadTooLarge {
                size: data.len(),
                max: MAX_PAYLOAD_SIZE,
            });
        }

        let mut hash = if flags.alt_hash {
            harmony_crypto::hash::sha224_hash(data)
        } else {
            let full = harmony_crypto::hash::full_hash(data);
            let mut trunc = [0u8; CONTENT_HASH_LEN];
            trunc.copy_from_slice(&full[..CONTENT_HASH_LEN]);
            trunc
        };

        // Embed flags in top 3 bits of hash[0].
        hash[0] = (hash[0] & 0x1F) | flags.to_bits();

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
```

**Step 4: Fix all call sites**

Every existing `ContentId::for_blob(data)` becomes `ContentId::for_blob(data, ContentFlags::default())`.

**Production call sites (3):**
- `crates/harmony-content/src/blob.rs:54` — `ContentId::for_blob(data)?` → `ContentId::for_blob(data, ContentFlags::default())?`
- `crates/harmony-content/src/reticulum_bridge.rs:43` — same pattern

  Note for `reticulum_bridge.rs:43`: This re-computes the CID to verify against the received CID. It must use the same flags as the received CID. Change to:
  ```rust
  let computed_cid = ContentId::for_blob(blob_data, received_cid.flags())?;
  ```

- `crates/harmony-content/src/bundle.rs:118` — this calls `for_bundle`, not `for_blob` (handled in Task 5)

**Test call sites (~55):** In every test file listed above, find-and-replace `ContentId::for_blob(` with `ContentId::for_blob(` adding `, ContentFlags::default()` before the `)`. This is mechanical — every instance gets the same change.

The following files need this mechanical update (add `ContentFlags::default()` as second arg):
- `cid.rs` — all existing `for_blob` calls in tests (~25 call sites)
- `blob.rs` — test call sites (~3)
- `storage_tier.rs` — test call sites (~15)
- `cache.rs` — test call sites (~6)
- `zenoh_bridge.rs` — test call sites (~2)
- `sketch.rs` — test call sites (~1)
- `lru.rs` — test call sites (~1)
- `delta.rs` — test call sites (~5)

Each file also needs `use crate::cid::ContentFlags;` (or the appropriate import) if not already imported.

**Step 5: Run tests**

Run: `cargo test -p harmony-content`
Expected: All tests pass. The default flags produce identical CIDs to the old behavior — **EXCEPT** the canonical test vectors, because `hash[0]` now has its top 3 bits cleared. These will fail and are fixed in Task 7.

**Important:** The canonical vector tests (`canonical_vector_empty_blob`, `canonical_vector_hello_blob`, `canonical_vector_bundle_of_two_blobs`) will fail at this point because the hash now masks the top 3 bits. That's expected — they'll be updated in Task 7 after all constructors are modified.

**Step 6: Commit**

```bash
git add -A
git commit -m "feat(content): for_blob accepts ContentFlags, update all call sites"
```

---

### Task 5: Modify for_bundle to accept ContentFlags

**Files:**
- Modify: `crates/harmony-content/src/cid.rs` (definition + ~8 test call sites)
- Modify: `crates/harmony-content/src/bundle.rs` (line 118 production + ~10 test call sites)
- Modify: `crates/harmony-content/src/storage_tier.rs` (~1 test call site)

**Step 1: Write the failing test**

Add to tests in `cid.rs`:

```rust
#[test]
fn for_bundle_with_flags() {
    let blob = ContentId::for_blob(b"leaf", ContentFlags::default()).unwrap();
    let children = [blob];
    let bytes = children_to_bytes(&children);

    let encrypted_flags = ContentFlags { encrypted: true, ephemeral: false, alt_hash: false };
    let bundle = ContentId::for_bundle(&bytes, &children, encrypted_flags).unwrap();
    assert!(bundle.flags().encrypted);
    assert!(!bundle.flags().ephemeral);
    assert!(!bundle.flags().alt_hash);
    assert_eq!(bundle.cid_type(), CidType::Bundle(1));
}
```

**Step 2: Verify failure**

Run: `cargo test -p harmony-content for_bundle_with_flags -- --nocapture`
Expected: FAIL — `for_bundle` takes 2 arguments but 3 were supplied.

**Step 3: Modify for_bundle**

Change `for_bundle` in `cid.rs`:

```rust
    /// Create a CID for a bundle (array of child CIDs).
    ///
    /// `bundle_bytes` is the raw byte payload (concatenated child CIDs).
    /// `children` is the parsed slice of child CIDs (used for depth calculation).
    /// The bundle's depth is `max(child depths) + 1`.
    pub fn for_bundle(
        bundle_bytes: &[u8],
        children: &[ContentId],
        flags: ContentFlags,
    ) -> Result<Self, ContentError> {
        if bundle_bytes.len() > MAX_PAYLOAD_SIZE {
            return Err(ContentError::PayloadTooLarge {
                size: bundle_bytes.len(),
                max: MAX_PAYLOAD_SIZE,
            });
        }

        let max_child_depth = children
            .iter()
            .map(|c| c.cid_type().depth())
            .max()
            .unwrap_or(0);
        let bundle_depth = max_child_depth + 1;

        if bundle_depth > 7 {
            return Err(ContentError::DepthViolation {
                child: max_child_depth,
                parent: bundle_depth,
            });
        }

        let mut hash = if flags.alt_hash {
            harmony_crypto::hash::sha224_hash(bundle_bytes)
        } else {
            let full = harmony_crypto::hash::full_hash(bundle_bytes);
            let mut trunc = [0u8; CONTENT_HASH_LEN];
            trunc.copy_from_slice(&full[..CONTENT_HASH_LEN]);
            trunc
        };

        // Embed flags in top 3 bits of hash[0].
        hash[0] = (hash[0] & 0x1F) | flags.to_bits();

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
```

**Step 4: Fix all call sites**

**Production (1):**
- `bundle.rs:118` — `ContentId::for_bundle(&bundle_bytes, &entries)?` → `ContentId::for_bundle(&bundle_bytes, &entries, ContentFlags::default())?`

**Test call sites (~18):** Same mechanical update — add `, ContentFlags::default()` as the third argument in:
- `cid.rs` (~8 calls in tests)
- `bundle.rs` (~10 calls in tests)
- `storage_tier.rs` (~1 call in tests)

Each test file needs the `ContentFlags` import.

**Step 5: Run tests**

Run: `cargo test -p harmony-content`
Expected: All pass (except canonical vectors — fixed in Task 7).

**Step 6: Commit**

```bash
git add -A
git commit -m "feat(content): for_bundle accepts ContentFlags, update all call sites"
```

---

### Task 6: Modify verify_hash to respect algorithm flag

**Files:**
- Modify: `crates/harmony-content/src/cid.rs`

**Step 1: Write the failing tests**

Add to tests:

```rust
#[test]
fn verify_hash_sha256_default() {
    let data = b"verify sha256";
    let cid = ContentId::for_blob(data, ContentFlags::default()).unwrap();
    assert!(cid.verify_hash(data));
    assert!(!cid.verify_hash(b"wrong data"));
}

#[test]
fn verify_hash_sha224_alt() {
    let data = b"verify sha224";
    let flags = ContentFlags { encrypted: false, ephemeral: false, alt_hash: true };
    let cid = ContentId::for_blob(data, flags).unwrap();
    assert!(cid.verify_hash(data));
    assert!(!cid.verify_hash(b"wrong data"));
}

#[test]
fn verify_hash_ignores_flag_bits() {
    // A CID with encrypted flag should still verify against the data,
    // because verify_hash masks the top 3 bits before comparing.
    let data = b"flag bits ignored during verify";
    let flags = ContentFlags { encrypted: true, ephemeral: true, alt_hash: false };
    let cid = ContentId::for_blob(data, flags).unwrap();
    assert!(cid.verify_hash(data));
}
```

**Step 2: Verify current behavior**

Run: `cargo test -p harmony-content verify_hash_sha224 -- --nocapture`
Expected: FAIL — the current `verify_hash` uses `full_hash` unconditionally, won't match SHA-224 CIDs.

**Step 3: Modify verify_hash**

Replace the existing `verify_hash` method:

```rust
    /// Verify that this CID's hash matches the hash of the given data.
    ///
    /// Selects SHA-256 or SHA-224 based on the `alt_hash` flag, and masks
    /// the top 3 flag bits before comparing.
    pub fn verify_hash(&self, data: &[u8]) -> bool {
        let flags = self.flags();
        let digest = if flags.alt_hash {
            harmony_crypto::hash::sha224_hash(data)
        } else {
            let full = harmony_crypto::hash::full_hash(data);
            let mut trunc = [0u8; CONTENT_HASH_LEN];
            trunc.copy_from_slice(&full[..CONTENT_HASH_LEN]);
            trunc
        };
        // Compare lower 5 bits of byte 0 (flag bits excluded), then bytes 1-27.
        (self.hash[0] & 0x1F) == (digest[0] & 0x1F) && self.hash[1..] == digest[1..]
    }
```

**Step 4: Run tests**

Run: `cargo test -p harmony-content verify_hash`
Expected: All verify_hash tests pass.

**Step 5: Commit**

```bash
git add crates/harmony-content/src/cid.rs
git commit -m "feat(content): verify_hash respects alt_hash flag, masks flag bits"
```

---

### Task 7: Update canonical test vectors

**Files:**
- Modify: `crates/harmony-content/src/cid.rs` (3 existing canonical tests + 5 new ones)

The top 3 bits of hash[0] are now cleared for default flags, changing the canonical hex output.

**Step 1: Compute new canonical values**

The empty blob SHA-256 hash starts with `e3` = `0b1110_0011`. After masking top 3 bits: `0b0000_0011` = `0x03`. So the first byte changes from `e3` to `03`.

The "hello" blob SHA-256 hash starts with `2c` = `0b0010_1100`. After masking: `0b0000_1100` = `0x0c`. First byte changes from `2c` to `0c`.

The bundle hash starts with `5c` = `0b0101_1100`. After masking: `0b0001_1100` = `0x1c`. First byte changes from `5c` to `1c`.

**Step 2: Update existing canonical tests**

Update `canonical_vector_empty_blob`:
```rust
#[test]
fn canonical_vector_empty_blob() {
    let cid = ContentId::for_blob(b"", ContentFlags::default()).unwrap();
    let bytes = cid.to_bytes();
    // SHA-256("")[:28] with top 3 bits cleared:
    // e3 → 03 (0b1110_0011 & 0x1F = 0b0000_0011)
    assert_eq!(
        hex::encode(&bytes[..CONTENT_HASH_LEN]),
        "03b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b",
    );
    assert_eq!(cid.payload_size(), 0);
    assert_eq!(cid.cid_type(), CidType::Blob);
    assert_eq!(cid.flags(), ContentFlags::default());
}
```

Update `canonical_vector_hello_blob`:
```rust
#[test]
fn canonical_vector_hello_blob() {
    let cid = ContentId::for_blob(b"hello", ContentFlags::default()).unwrap();
    let bytes = cid.to_bytes();
    // SHA-256("hello")[:28] with top 3 bits cleared:
    // 2c → 0c (0b0010_1100 & 0x1F = 0b0000_1100)
    assert_eq!(
        hex::encode(&bytes[..CONTENT_HASH_LEN]),
        "0cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362",
    );
    assert_eq!(cid.payload_size(), 5);
    assert_eq!(cid.cid_type(), CidType::Blob);
    assert_eq!(cid.flags(), ContentFlags::default());
}
```

Update `canonical_vector_bundle_of_two_blobs`:
```rust
#[test]
fn canonical_vector_bundle_of_two_blobs() {
    let blob_a = ContentId::for_blob(b"aaa", ContentFlags::default()).unwrap();
    let blob_b = ContentId::for_blob(b"bbb", ContentFlags::default()).unwrap();
    let children = [blob_a, blob_b];
    let bundle_bytes = children_to_bytes(&children);
    let bundle = ContentId::for_bundle(&bundle_bytes, &children, ContentFlags::default()).unwrap();
    assert_eq!(bundle.payload_size(), 64);
    assert_eq!(bundle.cid_type(), CidType::Bundle(1));
    assert_eq!(bundle.flags(), ContentFlags::default());
}
```

**Important:** The full 32-byte hex assertions in these tests must be recomputed after the implementation because the hash change affects the checksum. **Do not hardcode the hex values in the plan** — compute them by running the test once with just the hash prefix assertion, then capture the full output.

The approach:
1. Remove the full 32-byte hex `assert_eq!` temporarily
2. Run the test, print the actual bytes with `eprintln!("{}", hex::encode(bytes))`
3. Paste the real value back as the canonical vector

**Step 3: Add new canonical vectors for flag combinations**

Add tests for each non-default flag combination against the "hello" blob:

```rust
#[test]
fn canonical_vector_hello_encrypted() {
    let flags = ContentFlags { encrypted: true, ephemeral: false, alt_hash: false };
    let cid = ContentId::for_blob(b"hello", flags).unwrap();
    assert!(cid.flags().encrypted);
    assert!(!cid.flags().ephemeral);
    assert!(!cid.flags().alt_hash);
    // First byte: SHA-256("hello")[0] = 0x2c, masked = 0x0c, | 0x80 = 0x8c
    assert_eq!(cid.hash[0], 0x8c);
}

#[test]
fn canonical_vector_hello_ephemeral() {
    let flags = ContentFlags { encrypted: false, ephemeral: true, alt_hash: false };
    let cid = ContentId::for_blob(b"hello", flags).unwrap();
    assert!(cid.flags().ephemeral);
    // First byte: 0x0c | 0x40 = 0x4c
    assert_eq!(cid.hash[0], 0x4c);
}

#[test]
fn canonical_vector_hello_alt_hash() {
    let flags = ContentFlags { encrypted: false, ephemeral: false, alt_hash: true };
    let cid = ContentId::for_blob(b"hello", flags).unwrap();
    assert!(cid.flags().alt_hash);
    // Hash is SHA-224, completely different from SHA-256
    let sha224 = harmony_crypto::hash::sha224_hash(b"hello");
    assert_eq!(cid.hash[0] & 0x1F, sha224[0] & 0x1F);
    assert_eq!(&cid.hash[1..], &sha224[1..]);
    // First byte: sha224[0] masked | 0x20
    assert_eq!(cid.hash[0], (sha224[0] & 0x1F) | 0x20);
}

#[test]
fn canonical_vector_hello_private_ephemeral() {
    let flags = ContentFlags { encrypted: true, ephemeral: true, alt_hash: false };
    let cid = ContentId::for_blob(b"hello", flags).unwrap();
    assert!(cid.flags().encrypted);
    assert!(cid.flags().ephemeral);
    // First byte: 0x0c | 0xC0 = 0xCC
    assert_eq!(cid.hash[0], 0xcc);
}

#[test]
fn canonical_vector_hello_all_flags() {
    let flags = ContentFlags { encrypted: true, ephemeral: true, alt_hash: true };
    let cid = ContentId::for_blob(b"hello", flags).unwrap();
    assert!(cid.flags().encrypted);
    assert!(cid.flags().ephemeral);
    assert!(cid.flags().alt_hash);
    let sha224 = harmony_crypto::hash::sha224_hash(b"hello");
    // First byte: sha224[0] masked | 0xE0
    assert_eq!(cid.hash[0], (sha224[0] & 0x1F) | 0xE0);
}
```

**Step 4: Run tests**

Run: `cargo test -p harmony-content canonical_vector`
Expected: All canonical vector tests pass.

**Step 5: Also verify checksum still works**

Run: `cargo test -p harmony-content checksum`
Expected: All pass — checksums are computed over the hash (which now includes flag bits), so different flags → different checksums automatically.

**Step 6: Commit**

```bash
git add crates/harmony-content/src/cid.rs
git commit -m "feat(content): update canonical vectors for flag bits, add flag combination vectors"
```

---

### Task 8: Update BlobStore and BundleBuilder APIs

The `BlobStore::insert()` trait and `BundleBuilder::build()` currently don't pass flags. They need to either accept flags or default to cleartext/durable/SHA-256.

**Files:**
- Modify: `crates/harmony-content/src/blob.rs` (trait + impl)
- Modify: `crates/harmony-content/src/bundle.rs` (BundleBuilder)

**Step 1: Write the failing test**

Add to `blob.rs` tests:

```rust
#[test]
fn insert_with_flags() {
    use crate::cid::ContentFlags;
    let mut store = MemoryBlobStore::new();
    let flags = ContentFlags { encrypted: true, ephemeral: false, alt_hash: false };
    let cid = store.insert_with_flags(b"secret data", flags).unwrap();
    assert!(cid.flags().encrypted);
    assert!(store.contains(&cid));
}
```

**Step 2: Verify failure**

Run: `cargo test -p harmony-content insert_with_flags -- --nocapture`
Expected: FAIL — no method `insert_with_flags`.

**Step 3: Add insert_with_flags to trait and impl**

Add to `BlobStore` trait:

```rust
    /// Insert raw blob data with explicit storage-class flags.
    fn insert_with_flags(
        &mut self,
        data: &[u8],
        flags: ContentFlags,
    ) -> Result<ContentId, ContentError>;
```

Add a default implementation for `insert` that calls `insert_with_flags`:

```rust
    /// Insert raw blob data with default flags (cleartext, durable, SHA-256).
    fn insert(&mut self, data: &[u8]) -> Result<ContentId, ContentError> {
        self.insert_with_flags(data, ContentFlags::default())
    }
```

Implement for `MemoryBlobStore`:

```rust
    fn insert_with_flags(&mut self, data: &[u8], flags: ContentFlags) -> Result<ContentId, ContentError> {
        let cid = ContentId::for_blob(data, flags)?;
        self.data.entry(cid).or_insert_with(|| data.to_vec());
        Ok(cid)
    }
```

And keep the existing `insert` as the default trait method (remove the direct impl if it exists).

For `BundleBuilder::build()`, add a `build_with_flags` method:

```rust
    pub fn build_with_flags(&self, flags: ContentFlags) -> Result<(Vec<u8>, ContentId), ContentError> {
        // ... same as build() but passes flags to for_bundle
        let bundle_cid = ContentId::for_bundle(&bundle_bytes, &entries, flags)?;
        Ok((bundle_bytes, bundle_cid))
    }

    pub fn build(&self) -> Result<(Vec<u8>, ContentId), ContentError> {
        self.build_with_flags(ContentFlags::default())
    }
```

**Step 4: Run tests**

Run: `cargo test -p harmony-content`
Expected: All pass.

**Step 5: Commit**

```bash
git add crates/harmony-content/src/blob.rs crates/harmony-content/src/bundle.rs
git commit -m "feat(content): add insert_with_flags and build_with_flags APIs"
```

---

### Task 9: Export ContentFlags from lib.rs and update reticulum_bridge

**Files:**
- Modify: `crates/harmony-content/src/lib.rs`
- Modify: `crates/harmony-content/src/reticulum_bridge.rs`
- Verify: `crates/harmony-browser/src/core.rs` (verify_hash should work unchanged)

**Step 1: Export ContentFlags**

In `lib.rs`, ensure `ContentFlags` is re-exported wherever `ContentId` is:

Check how `ContentId` is currently exported and add `ContentFlags` alongside it.

**Step 2: Verify reticulum_bridge uses received CID's flags**

The change in Task 4 should have updated `reticulum_bridge.rs:43` to use `received_cid.flags()`. Verify this is correct:

```rust
let computed_cid = ContentId::for_blob(blob_data, received_cid.flags())?;
```

This ensures that when verifying a received blob, we use the same hash algorithm as the sender encoded in the CID.

**Step 3: Verify harmony-browser still compiles**

`harmony-browser/src/core.rs:110` calls `cid.verify_hash(&data)`. Since `verify_hash` now reads the `alt_hash` flag from the CID itself, this call site needs no changes — it automatically selects the right algorithm.

Run: `cargo test --workspace`
Expected: All tests across all crates pass.

**Step 4: Commit**

```bash
git add -A
git commit -m "feat(content): export ContentFlags, verify cross-crate compatibility"
```

---

### Task 10: Final quality gates

**Step 1: Run full test suite**

```bash
cargo test --workspace
```

Expected: All tests pass (365+ tests).

**Step 2: Run clippy**

```bash
cargo clippy --workspace
```

Expected: Zero warnings.

**Step 3: Run fmt check**

```bash
cargo fmt --all -- --check
```

Expected: No formatting issues.

**Step 4: Verify design compliance**

Check against `docs/plans/2026-03-07-content-flags-design.md`:
- [ ] Top 3 bits of hash[0] encode E, D, A flags
- [ ] SHA-256 (A=0) and SHA-224 (A=1) both work
- [ ] Same data + different flags → different CID
- [ ] verify_hash respects algorithm flag
- [ ] Checksum covers flags (different flags → different checksum)
- [ ] All existing call sites use ContentFlags::default()
- [ ] Canonical test vectors updated
- [ ] New flag-combination test vectors added
- [ ] No changes to harmony-athenaeum or harmony-os

**Step 5: Commit any final fixes**

```bash
git add -A
git commit -m "chore: final quality gates for content flags"
```
