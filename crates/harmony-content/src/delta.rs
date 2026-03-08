use crate::bundle::{self, CID_SIZE};
use crate::cid::ContentId;
use crate::error::ContentError;
use alloc::vec::Vec;

/// Opcode byte for COPY: copy CIDs from the old bundle.
const OP_COPY: u8 = 0x00;

/// Opcode byte for INSERT: inline new CID data.
const OP_INSERT: u8 = 0x01;

/// Tag byte for encode_update: full bundle follows.
const UPDATE_FULL: u8 = 0x00;

/// Tag byte for encode_update: delta follows.
const UPDATE_DELTA: u8 = 0x01;

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
                let offset = u16::from_be_bytes([delta[pos + 1], delta[pos + 2]]) as usize;
                let count = u16::from_be_bytes([delta[pos + 3], delta[pos + 4]]) as usize;
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
                let count = u16::from_be_bytes([delta[pos + 1], delta[pos + 2]]) as usize;
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

/// Encode a bundle update, choosing the most compact representation.
///
/// Compares the delta against the full new bundle and returns whichever
/// is smaller. The first byte of the result indicates the format:
/// - `0x00` — full bundle bytes follow (delta wasn't worth it)
/// - `0x01` — delta bytes follow
pub fn encode_update(old_bundle: &[u8], new_bundle: &[u8]) -> Result<Vec<u8>, ContentError> {
    let delta = compute_delta(old_bundle, new_bundle)?;

    if delta.len() < new_bundle.len() {
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

#[cfg(test)]
mod tests {
    use super::*;

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
                ContentId::for_blob(data.as_bytes(), crate::cid::ContentFlags::default()).unwrap()
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
        cids[2] = ContentId::for_blob(b"replacement", crate::cid::ContentFlags::default()).unwrap();
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

    #[test]
    fn apply_delta_round_trip() {
        let old_cids = make_cids(5);
        let mut new_cids = old_cids.clone();
        new_cids[2] = ContentId::for_blob(b"changed", crate::cid::ContentFlags::default()).unwrap();

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

    #[test]
    fn encode_update_uses_delta_when_smaller() {
        let cids = make_cids(10);
        let mut changed = cids.clone();
        changed[5] =
            ContentId::for_blob(b"one-change", crate::cid::ContentFlags::default()).unwrap();
        let old = bundle_bytes(&cids);
        let new = bundle_bytes(&changed);
        let update = encode_update(&old, &new).unwrap();
        assert_eq!(update[0], UPDATE_DELTA);
        assert!(update.len() < new.len());
    }

    #[test]
    fn encode_update_uses_full_when_delta_larger() {
        let old_cids = make_cids(3);
        let new_cids: Vec<ContentId> = (100..103)
            .map(|i| {
                ContentId::for_blob(
                    format!("different-{i}").as_bytes(),
                    crate::cid::ContentFlags::default(),
                )
                .unwrap()
            })
            .collect();
        let old = bundle_bytes(&old_cids);
        let new = bundle_bytes(&new_cids);
        let update = encode_update(&old, &new).unwrap();
        // Complete replacement: delta = INSERT(3) = 3 + 96 = 99 bytes.
        // Full bundle = 96 bytes. Full is smaller.
        assert_eq!(update[0], UPDATE_FULL);
        assert_eq!(update.len(), 1 + new.len());
    }
}
