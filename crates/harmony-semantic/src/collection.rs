// SPDX-License-Identifier: Apache-2.0 OR MIT
//! Collection blob codec — groups up to 256 semantically similar content CIDs
//! into a navigable cluster (leaf node of the trie index).

use alloc::vec::Vec;

use crate::error::{SemanticError, SemanticResult};

/// HSC v1 magic bytes: "HSC" + version 1.
pub const COLLECTION_MAGIC: [u8; 4] = [0x48, 0x53, 0x43, 0x01];

/// Size of the fixed header: magic(4) + fingerprint(4) + count(4) + centroid(32).
pub const COLLECTION_HEADER_SIZE: usize = 44;

/// Size of a single collection entry: target_cid(32) + tier3(32).
pub const COLLECTION_ENTRY_SIZE: usize = 64;

/// Maximum number of entries in a collection blob.
pub const MAX_COLLECTION_ENTRIES: u32 = 256;

/// A single entry in a collection blob.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CollectionEntry {
    /// CID of the content this entry points to.
    pub target_cid: [u8; 32],
    /// Tier 3 (256-bit) binary vector for this content.
    pub tier3: [u8; 32],
}

/// A collection blob grouping up to 256 semantically similar content CIDs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CollectionBlob {
    /// Model fingerprint (SHA-256(model_id)[:4]).
    pub fingerprint: [u8; 4],
    /// Majority-vote centroid of the entry vectors.
    pub centroid: [u8; 32],
    /// The entries in this collection.
    pub entries: Vec<CollectionEntry>,
}

impl CollectionBlob {
    /// Serialize the collection blob to binary.
    ///
    /// Layout: magic[0..4] + fingerprint[4..8] + count(u32 BE)[8..12]
    /// + centroid[12..44] + N × (target_cid[0..32] + tier3[32..64]).
    ///
    /// Returns `CollectionOverflow` if entries exceed `MAX_COLLECTION_ENTRIES`.
    pub fn encode(&self) -> SemanticResult<Vec<u8>> {
        if self.entries.len() > MAX_COLLECTION_ENTRIES as usize {
            return Err(SemanticError::CollectionOverflow {
                count: self.entries.len() as u32,
            });
        }
        let entry_count = self.entries.len() as u32;
        let total = COLLECTION_HEADER_SIZE + (self.entries.len() * COLLECTION_ENTRY_SIZE);
        let mut buf = Vec::with_capacity(total);

        buf.extend_from_slice(&COLLECTION_MAGIC);
        buf.extend_from_slice(&self.fingerprint);
        buf.extend_from_slice(&entry_count.to_be_bytes());
        buf.extend_from_slice(&self.centroid);

        for entry in &self.entries {
            buf.extend_from_slice(&entry.target_cid);
            buf.extend_from_slice(&entry.tier3);
        }

        Ok(buf)
    }

    /// Deserialize a collection blob from bytes.
    pub fn decode(data: &[u8]) -> SemanticResult<Self> {
        if data.len() < COLLECTION_HEADER_SIZE {
            return Err(SemanticError::TruncatedHeader {
                expected: COLLECTION_HEADER_SIZE,
                actual: data.len(),
            });
        }

        let magic = &data[0..4];
        if magic != COLLECTION_MAGIC {
            return Err(SemanticError::InvalidMagic);
        }

        let mut fingerprint = [0u8; 4];
        fingerprint.copy_from_slice(&data[4..8]);

        let count = u32::from_be_bytes([data[8], data[9], data[10], data[11]]);
        if count > MAX_COLLECTION_ENTRIES {
            return Err(SemanticError::CollectionOverflow { count });
        }

        let mut centroid = [0u8; 32];
        centroid.copy_from_slice(&data[12..44]);

        let expected_len = COLLECTION_HEADER_SIZE + (count as usize * COLLECTION_ENTRY_SIZE);
        if data.len() < expected_len {
            return Err(SemanticError::TruncatedHeader {
                expected: expected_len,
                actual: data.len(),
            });
        }

        let mut entries = Vec::with_capacity(count as usize);
        for i in 0..count as usize {
            let base = COLLECTION_HEADER_SIZE + i * COLLECTION_ENTRY_SIZE;
            let mut target_cid = [0u8; 32];
            target_cid.copy_from_slice(&data[base..base + 32]);
            let mut tier3 = [0u8; 32];
            tier3.copy_from_slice(&data[base + 32..base + 64]);
            entries.push(CollectionEntry { target_cid, tier3 });
        }

        Ok(Self {
            fingerprint,
            centroid,
            entries,
        })
    }

    /// Compute the majority-vote centroid of the given entries.
    ///
    /// For each of the 256 bit positions: count how many entries have that bit
    /// set. If strictly more than half the entries have it set, set it in the
    /// centroid. Ties (even entry count, exactly half set) resolve to 0.
    pub fn compute_centroid(entries: &[CollectionEntry]) -> [u8; 32] {
        let mut centroid = [0u8; 32];
        let n = entries.len();
        if n == 0 {
            return centroid;
        }
        let threshold = n / 2;

        for (byte_idx, centroid_byte) in centroid.iter_mut().enumerate() {
            let mut result_byte = 0u8;
            for bit in 0..8 {
                let mask = 1u8 << (7 - bit);
                let count = entries
                    .iter()
                    .filter(|e| e.tier3[byte_idx] & mask != 0)
                    .count();
                if count > threshold {
                    result_byte |= mask;
                }
            }
            *centroid_byte = result_byte;
        }

        centroid
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use super::*;
    use crate::fingerprint::model_fingerprint;

    /// Helper: build a deterministic collection with `n` entries.
    fn test_collection(n: usize) -> CollectionBlob {
        let fp = model_fingerprint("test-collection-model");
        let mut entries = Vec::with_capacity(n);
        for i in 0..n {
            let mut target_cid = [0u8; 32];
            target_cid[0] = i as u8;
            target_cid[31] = (255 - i) as u8;
            let mut tier3 = [0u8; 32];
            tier3.fill(i as u8);
            entries.push(CollectionEntry { target_cid, tier3 });
        }
        let centroid = CollectionBlob::compute_centroid(&entries);
        CollectionBlob {
            fingerprint: fp,
            centroid,
            entries,
        }
    }

    #[test]
    fn collection_encode_decode_roundtrip() {
        let blob = test_collection(5);
        let encoded = blob.encode().expect("encode should succeed");
        let decoded = CollectionBlob::decode(&encoded).expect("decode should succeed");
        assert_eq!(decoded.fingerprint, blob.fingerprint);
        assert_eq!(decoded.centroid, blob.centroid);
        assert_eq!(decoded.entries.len(), blob.entries.len());
        for (a, b) in decoded.entries.iter().zip(blob.entries.iter()) {
            assert_eq!(a.target_cid, b.target_cid);
            assert_eq!(a.tier3, b.tier3);
        }
    }

    #[test]
    fn collection_decode_truncated_rejects() {
        let short = [0u8; 20];
        let err = CollectionBlob::decode(&short).unwrap_err();
        assert_eq!(
            err,
            SemanticError::TruncatedHeader {
                expected: COLLECTION_HEADER_SIZE,
                actual: 20,
            }
        );
    }

    #[test]
    fn collection_decode_bad_magic_rejects() {
        let mut buf = [0u8; COLLECTION_HEADER_SIZE];
        buf[0..4].copy_from_slice(&[0xFF, 0xFF, 0xFF, 0xFF]);
        let err = CollectionBlob::decode(&buf).unwrap_err();
        assert_eq!(err, SemanticError::InvalidMagic);
    }

    #[test]
    fn collection_overflow_rejects() {
        let mut buf = [0u8; COLLECTION_HEADER_SIZE];
        buf[0..4].copy_from_slice(&COLLECTION_MAGIC);
        // count = 257 in big-endian
        buf[8..12].copy_from_slice(&257u32.to_be_bytes());
        let err = CollectionBlob::decode(&buf).unwrap_err();
        assert_eq!(err, SemanticError::CollectionOverflow { count: 257 });
    }

    #[test]
    fn collection_empty_allowed() {
        let blob = test_collection(0);
        let encoded = blob.encode().expect("encode should succeed");
        assert_eq!(encoded.len(), COLLECTION_HEADER_SIZE);
        let decoded = CollectionBlob::decode(&encoded).expect("empty collection should decode");
        assert_eq!(decoded.entries.len(), 0);
    }

    #[test]
    fn centroid_computation() {
        // Three entries with known tier3 vectors.
        // Byte 0: 0xFF, 0xFF, 0x00 → all 8 bits: 2 out of 3 set → majority → 0xFF
        // Byte 1: 0x00, 0x00, 0xFF → all 8 bits: 1 out of 3 set → minority → 0x00
        // Byte 2: 0xAA, 0xAA, 0xAA → 0xAA (bits 7,5,3,1 set in all 3) → 0xAA
        let mut t3_a = [0u8; 32];
        let mut t3_b = [0u8; 32];
        let mut t3_c = [0u8; 32];

        t3_a[0] = 0xFF;
        t3_a[1] = 0x00;
        t3_a[2] = 0xAA;

        t3_b[0] = 0xFF;
        t3_b[1] = 0x00;
        t3_b[2] = 0xAA;

        t3_c[0] = 0x00;
        t3_c[1] = 0xFF;
        t3_c[2] = 0xAA;

        let entries = vec![
            CollectionEntry {
                target_cid: [1u8; 32],
                tier3: t3_a,
            },
            CollectionEntry {
                target_cid: [2u8; 32],
                tier3: t3_b,
            },
            CollectionEntry {
                target_cid: [3u8; 32],
                tier3: t3_c,
            },
        ];

        let centroid = CollectionBlob::compute_centroid(&entries);
        assert_eq!(centroid[0], 0xFF);
        assert_eq!(centroid[1], 0x00);
        assert_eq!(centroid[2], 0xAA);
        // Remaining bytes: all zero in all entries → 0 out of 3 → 0x00
        for byte in &centroid[3..] {
            assert_eq!(*byte, 0x00);
        }
    }

    #[test]
    fn collection_entry_size_is_64() {
        assert_eq!(COLLECTION_ENTRY_SIZE, 64);
    }
}
