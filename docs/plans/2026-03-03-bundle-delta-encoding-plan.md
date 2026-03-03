# Bundle Delta-Encoding Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement COPY/INSERT delta-encoding for bundle updates so that when a file changes, only the differing CIDs need to be transferred.

**Architecture:** A new `delta.rs` module with three free functions (`compute_delta`, `apply_delta`, `encode_update`) operating on raw bundle bytes. CID-aligned opcodes — COPY and INSERT work in units of 32-byte CIDs. Linear scan algorithm with suffix matching. Pure functions, no state, sans-I/O.

**Tech Stack:** Rust, existing `harmony-content` crate primitives (`bundle::parse_bundle`, `bundle::CID_SIZE`, `ContentError`)

---

### Task 1: Error Variant + Module Scaffolding

**Files:**
- Create: `crates/harmony-content/src/delta.rs`
- Modify: `crates/harmony-content/src/error.rs`
- Modify: `crates/harmony-content/src/lib.rs`

**Context:** We need one new error variant and the delta module registered. `ContentError` is in `error.rs` and currently has variants up through `MissingContent`. `lib.rs` currently declares: blob, bundle, chunker, cid, dag, error.

**Step 1: Add error variant to `crates/harmony-content/src/error.rs`**

Add after the `MissingContent` variant (line 31):

```rust
    #[error("invalid delta: {reason}")]
    InvalidDelta { reason: &'static str },
```

**Step 2: Create `crates/harmony-content/src/delta.rs` with constants and stub test**

```rust
use crate::bundle::{self, CID_SIZE};
use crate::error::ContentError;

/// Opcode byte for COPY: copy CIDs from the old bundle.
const OP_COPY: u8 = 0x00;

/// Opcode byte for INSERT: inline new CID data.
const OP_INSERT: u8 = 0x01;

/// Tag byte for encode_update: full bundle follows.
const UPDATE_FULL: u8 = 0x00;

/// Tag byte for encode_update: delta follows.
const UPDATE_DELTA: u8 = 0x01;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cid::ContentId;

    /// Build raw bundle bytes from a slice of ContentIds.
    fn bundle_bytes(cids: &[ContentId]) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(cids.len() * CID_SIZE);
        for cid in cids {
            bytes.extend_from_slice(&cid.to_bytes());
        }
        bytes
    }

    /// Create N distinct blob CIDs for testing.
    fn make_cids(n: usize) -> Vec<ContentId> {
        (0..n)
            .map(|i| {
                let data = format!("test-blob-{i}");
                ContentId::for_blob(data.as_bytes()).unwrap()
            })
            .collect()
    }

    #[test]
    fn module_compiles() {
        let _ = OP_COPY;
        let _ = OP_INSERT;
        let cids = make_cids(3);
        let _ = bundle_bytes(&cids);
    }
}
```

**Step 3: Add `pub mod delta;` to `crates/harmony-content/src/lib.rs`**

Insert between `dag` and `error`:

```rust
pub mod blob;
pub mod bundle;
pub mod chunker;
pub mod cid;
pub mod dag;
pub mod delta;
pub mod error;
```

**Step 4: Run tests to verify compilation**

Run: `cargo test -p harmony-content delta::tests::module_compiles`
Expected: PASS

Run: `cargo clippy -p harmony-content`
Expected: Clean (unused constant warnings in delta.rs are OK — they'll resolve in Task 2)

**Step 5: Commit**

```bash
git add crates/harmony-content/src/delta.rs crates/harmony-content/src/error.rs crates/harmony-content/src/lib.rs
git commit -m "feat(content): add delta module scaffolding and InvalidDelta error variant"
```

---

### Task 2: `compute_delta()` — Linear Scan with Suffix Matching

**Files:**
- Modify: `crates/harmony-content/src/delta.rs`

**Context:** `compute_delta` takes two raw bundle byte slices (old and new), validates they're multiples of 32, parses them as CID slices, then emits COPY/INSERT opcodes via a linear scan. The algorithm:

1. Parse both as CID slices via `bundle::parse_bundle`.
2. Find the longest common prefix (matching CIDs from the start).
3. Find the longest common suffix (matching CIDs from the end, after the prefix).
4. Emit: COPY for prefix (if any) → INSERT for middle (if any) → COPY for suffix (if any).

Wire format:
- COPY: `[0x00] [cid_offset:u16 BE] [cid_count:u16 BE]` — 5 bytes
- INSERT: `[0x01] [cid_count:u16 BE] [cid_data...]` — 3 + count*32 bytes

**Step 1: Write the implementation and tests**

Add to `crates/harmony-content/src/delta.rs` (before `#[cfg(test)]`):

```rust
/// Compute a CID-aligned delta between an old and new bundle.
///
/// Returns a compact byte sequence of COPY/INSERT opcodes. Both inputs
/// must be valid bundle byte arrays (length is a multiple of 32).
///
/// The algorithm finds the longest common prefix and suffix of CIDs,
/// then emits COPY for shared regions and INSERT for changed regions.
pub fn compute_delta(old_bundle: &[u8], new_bundle: &[u8]) -> Result<Vec<u8>, ContentError> {
    let old_cids = bundle::parse_bundle(old_bundle)?;
    let new_cids = bundle::parse_bundle(new_bundle)?;

    let mut delta = Vec::new();

    if new_cids.is_empty() {
        return Ok(delta);
    }

    // Find longest common prefix.
    let prefix_len = old_cids
        .iter()
        .zip(new_cids.iter())
        .take_while(|(a, b)| a == b)
        .count();

    // Find longest common suffix (after the prefix).
    let old_remaining = old_cids.len() - prefix_len;
    let new_remaining = new_cids.len() - prefix_len;
    let suffix_len = old_cids[prefix_len..]
        .iter()
        .rev()
        .zip(new_cids[prefix_len..].iter().rev())
        .take_while(|(a, b)| a == b)
        .count();

    let insert_count = new_remaining - suffix_len;

    // Emit COPY for prefix.
    if prefix_len > 0 {
        emit_copy(&mut delta, 0, prefix_len);
    }

    // Emit INSERT for middle (new CIDs not in old).
    if insert_count > 0 {
        emit_insert(&mut delta, &new_cids[prefix_len..prefix_len + insert_count]);
    }

    // Emit COPY for suffix.
    if suffix_len > 0 {
        let suffix_offset = old_cids.len() - suffix_len;
        emit_copy(&mut delta, suffix_offset, suffix_len);
    }

    Ok(delta)
}

/// Emit a COPY opcode: copy `count` CIDs starting at `offset` in the old bundle.
fn emit_copy(delta: &mut Vec<u8>, offset: usize, count: usize) {
    delta.push(OP_COPY);
    delta.extend_from_slice(&(offset as u16).to_be_bytes());
    delta.extend_from_slice(&(count as u16).to_be_bytes());
}

/// Emit an INSERT opcode: inline `cids` as new data.
fn emit_insert(delta: &mut Vec<u8>, cids: &[ContentId]) {
    delta.push(OP_INSERT);
    delta.extend_from_slice(&(cids.len() as u16).to_be_bytes());
    for cid in cids {
        delta.extend_from_slice(&cid.to_bytes());
    }
}
```

This requires adding `use crate::cid::ContentId;` to the top-level imports in delta.rs.

Add tests:

```rust
    #[test]
    fn delta_identical_bundles() {
        let cids = make_cids(5);
        let old = bundle_bytes(&cids);
        let new = bundle_bytes(&cids);
        let delta = compute_delta(&old, &new).unwrap();
        // Single COPY opcode = 5 bytes, much smaller than 160-byte bundle.
        assert_eq!(delta.len(), 5);
        assert!(delta.len() < old.len());
    }

    #[test]
    fn delta_single_cid_change() {
        let mut cids = make_cids(5);
        let old = bundle_bytes(&cids);
        // Change middle CID.
        cids[2] = ContentId::for_blob(b"replacement").unwrap();
        let new = bundle_bytes(&cids);
        let delta = compute_delta(&old, &new).unwrap();
        // COPY(0,2) + INSERT(1) + COPY(3,2) = 5 + 35 + 5 = 45 bytes.
        assert_eq!(delta.len(), 45);
    }

    #[test]
    fn delta_append() {
        let cids = make_cids(5);
        let old = bundle_bytes(&cids[..3]);
        let new = bundle_bytes(&cids);
        let delta = compute_delta(&old, &new).unwrap();
        // COPY(0,3) + INSERT(2) = 5 + 67 = 72 bytes.
        assert_eq!(delta.len(), 72);
    }

    #[test]
    fn delta_prepend() {
        let cids = make_cids(5);
        let old = bundle_bytes(&cids[2..]);
        let new = bundle_bytes(&cids);
        let delta = compute_delta(&old, &new).unwrap();
        // INSERT(2) + COPY(0,3) = 67 + 5 = 72 bytes.
        assert_eq!(delta.len(), 72);
    }

    #[test]
    fn delta_empty_old_bundle() {
        let cids = make_cids(3);
        let old: Vec<u8> = vec![];
        let new = bundle_bytes(&cids);
        let delta = compute_delta(&old, &new).unwrap();
        // All INSERT: 3 + 96 = 99 bytes.
        assert_eq!(delta.len(), 99);
    }
```

**Step 2: Run tests**

Run: `cargo test -p harmony-content delta`
Expected: 6 tests PASS (module_compiles + 5 new)

**Step 3: Commit**

```bash
git add crates/harmony-content/src/delta.rs
git commit -m "feat(content): add compute_delta() — CID-aligned bundle diffing"
```

---

### Task 3: `apply_delta()` — Reconstruct New Bundle from Old + Delta

**Files:**
- Modify: `crates/harmony-content/src/delta.rs`

**Context:** `apply_delta` takes old bundle bytes and a delta byte sequence, parses the opcodes, and reconstructs the new bundle. Must validate: opcode bytes are 0x00 or 0x01, COPY offsets/counts are within bounds of old bundle, INSERT data is complete, and result is a valid bundle length.

**Step 1: Write the implementation and tests**

Add to `crates/harmony-content/src/delta.rs` (after `emit_insert`, before `#[cfg(test)]`):

```rust
/// Apply a delta to an old bundle, producing the new bundle bytes.
///
/// The old bundle must be a valid bundle (length multiple of 32).
/// The delta is a sequence of COPY/INSERT opcodes produced by `compute_delta`.
pub fn apply_delta(old_bundle: &[u8], delta: &[u8]) -> Result<Vec<u8>, ContentError> {
    let old_cids = bundle::parse_bundle(old_bundle)?;
    let mut result = Vec::new();
    let mut pos = 0;

    while pos < delta.len() {
        match delta[pos] {
            OP_COPY => {
                if pos + 5 > delta.len() {
                    return Err(ContentError::InvalidDelta {
                        reason: "truncated COPY opcode",
                    });
                }
                let offset =
                    u16::from_be_bytes([delta[pos + 1], delta[pos + 2]]) as usize;
                let count =
                    u16::from_be_bytes([delta[pos + 3], delta[pos + 4]]) as usize;
                pos += 5;

                if offset + count > old_cids.len() {
                    return Err(ContentError::InvalidDelta {
                        reason: "COPY range exceeds old bundle",
                    });
                }

                for cid in &old_cids[offset..offset + count] {
                    result.extend_from_slice(&cid.to_bytes());
                }
            }
            OP_INSERT => {
                if pos + 3 > delta.len() {
                    return Err(ContentError::InvalidDelta {
                        reason: "truncated INSERT opcode",
                    });
                }
                let count =
                    u16::from_be_bytes([delta[pos + 1], delta[pos + 2]]) as usize;
                pos += 3;

                let data_len = count * CID_SIZE;
                if pos + data_len > delta.len() {
                    return Err(ContentError::InvalidDelta {
                        reason: "truncated INSERT data",
                    });
                }

                result.extend_from_slice(&delta[pos..pos + data_len]);
                pos += data_len;
            }
            _ => {
                return Err(ContentError::InvalidDelta {
                    reason: "unknown opcode",
                });
            }
        }
    }

    Ok(result)
}
```

Add tests:

```rust
    #[test]
    fn apply_delta_round_trip() {
        let old_cids = make_cids(5);
        let mut new_cids = old_cids.clone();
        new_cids[2] = ContentId::for_blob(b"changed").unwrap();

        let old = bundle_bytes(&old_cids);
        let new = bundle_bytes(&new_cids);
        let delta = compute_delta(&old, &new).unwrap();
        let reconstructed = apply_delta(&old, &delta).unwrap();
        assert_eq!(reconstructed, new);
    }

    #[test]
    fn apply_delta_round_trip_prepend() {
        let cids = make_cids(5);
        let old = bundle_bytes(&cids[2..]);
        let new = bundle_bytes(&cids);
        let delta = compute_delta(&old, &new).unwrap();
        let reconstructed = apply_delta(&old, &delta).unwrap();
        assert_eq!(reconstructed, new);
    }

    #[test]
    fn apply_delta_invalid_opcode() {
        let old = bundle_bytes(&make_cids(1));
        let bad_delta = vec![0xFF, 0x00, 0x00, 0x00, 0x01];
        let result = apply_delta(&old, &bad_delta);
        assert!(matches!(result, Err(ContentError::InvalidDelta { .. })));
    }

    #[test]
    fn apply_delta_truncated_copy() {
        let old = bundle_bytes(&make_cids(1));
        let bad_delta = vec![OP_COPY, 0x00]; // Only 2 bytes, need 5.
        let result = apply_delta(&old, &bad_delta);
        assert!(matches!(result, Err(ContentError::InvalidDelta { .. })));
    }

    #[test]
    fn apply_delta_copy_out_of_bounds() {
        let old = bundle_bytes(&make_cids(2));
        // COPY offset=0, count=5 — but old only has 2 CIDs.
        let bad_delta = vec![OP_COPY, 0x00, 0x00, 0x00, 0x05];
        let result = apply_delta(&old, &bad_delta);
        assert!(matches!(result, Err(ContentError::InvalidDelta { .. })));
    }
```

**Step 2: Run tests**

Run: `cargo test -p harmony-content delta`
Expected: 11 tests PASS

**Step 3: Commit**

```bash
git add crates/harmony-content/src/delta.rs
git commit -m "feat(content): add apply_delta() — reconstruct bundle from delta"
```

---

### Task 4: `encode_update()` + Full Suite Verification

**Files:**
- Modify: `crates/harmony-content/src/delta.rs`

**Context:** `encode_update` computes the delta, compares its size to the full new bundle, and returns whichever is smaller. The return format: first byte `0x00` = full bundle follows, `0x01` = delta follows.

**Step 1: Write the implementation and tests**

Add to `crates/harmony-content/src/delta.rs` (after `apply_delta`, before `#[cfg(test)]`):

```rust
/// Encode a bundle update, choosing the most compact representation.
///
/// Compares the delta against the full new bundle and returns whichever
/// is smaller. The first byte of the result indicates the format:
/// - `0x00` — full bundle bytes follow (delta wasn't worth it)
/// - `0x01` — delta bytes follow
pub fn encode_update(old_bundle: &[u8], new_bundle: &[u8]) -> Result<Vec<u8>, ContentError> {
    let delta = compute_delta(old_bundle, new_bundle)?;

    // +1 for the tag byte in both cases.
    if delta.len() + 1 < new_bundle.len() + 1 {
        let mut result = Vec::with_capacity(1 + delta.len());
        result.push(UPDATE_DELTA);
        result.extend_from_slice(&delta);
        Ok(result)
    } else {
        let mut result = Vec::with_capacity(1 + new_bundle.len());
        result.push(UPDATE_FULL);
        result.extend_from_slice(new_bundle);
        Ok(result)
    }
}
```

Add tests:

```rust
    #[test]
    fn encode_update_uses_delta_when_smaller() {
        let cids = make_cids(10);
        let mut changed = cids.clone();
        changed[5] = ContentId::for_blob(b"one-change").unwrap();
        let old = bundle_bytes(&cids);
        let new = bundle_bytes(&changed);
        let update = encode_update(&old, &new).unwrap();
        assert_eq!(update[0], UPDATE_DELTA);
        assert!(update.len() < new.len());
    }

    #[test]
    fn encode_update_uses_full_when_delta_larger() {
        let old_cids = make_cids(3);
        let new_cids = make_cids(3); // Completely different (different indices produce different CIDs)
        // Actually make_cids(3) is deterministic, so we need truly different CIDs:
        let new_cids: Vec<ContentId> = (100..103)
            .map(|i| ContentId::for_blob(format!("different-{i}").as_bytes()).unwrap())
            .collect();
        let old = bundle_bytes(&old_cids);
        let new = bundle_bytes(&new_cids);
        let update = encode_update(&old, &new).unwrap();
        // Complete replacement: delta = INSERT(3) = 3 + 96 = 99 bytes.
        // Full bundle = 96 bytes. Full is smaller.
        assert_eq!(update[0], UPDATE_FULL);
        assert_eq!(update.len(), 1 + new.len());
    }
```

**Step 2: Run the full test suite**

Run: `cargo test -p harmony-content`
Expected: All tests pass (91 existing + ~13 new delta tests)

Run: `cargo test --workspace`
Expected: All workspace tests pass

Run: `cargo clippy -p harmony-content`
Expected: No warnings

Run: `cargo fmt --all -- --check`
Expected: No formatting issues

**Step 3: Commit**

```bash
git add crates/harmony-content/src/delta.rs
git commit -m "feat(content): add encode_update() — compact bundle update encoding"
```
