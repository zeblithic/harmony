// SPDX-License-Identifier: Apache-2.0 OR MIT
//! Sidecar blob codec — encode/decode the 288-byte HSI fixed header.

use crate::error::{SemanticError, SemanticResult};
use crate::fingerprint::ModelFingerprint;

/// HSI v1 magic bytes: "HSI" + version 1.
pub const SIDECAR_V1_MAGIC: [u8; 4] = [0x48, 0x53, 0x49, 0x01];
/// HSI v2 magic bytes: "HSI" + version 2 (enriched sidecar with CBOR trailer).
pub const SIDECAR_V2_MAGIC: [u8; 4] = [0x48, 0x53, 0x49, 0x02];
/// Size of the fixed header (both v1 and v2).
pub const SIDECAR_HEADER_SIZE: usize = 288;

/// Decoded sidecar fixed header.
///
/// Contains the multi-tier binary embedding vectors for a single piece
/// of content. Tiers are nested prefixes of the same MRL vector — Tier 1
/// is the first 64 bits, Tier 2 is the first 128, and so on.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SidecarHeader {
    /// Model fingerprint (SHA-256(model_id)[:4]).
    pub fingerprint: ModelFingerprint,
    /// CID of the content this sidecar describes (32 bytes).
    pub target_cid: [u8; 32],
    /// Tier 1: 64-bit binary vector (8 bytes).
    pub tier1: [u8; 8],
    /// Tier 2: 128-bit binary vector (16 bytes).
    pub tier2: [u8; 16],
    /// Tier 3: 256-bit binary vector (32 bytes).
    pub tier3: [u8; 32],
    /// Tier 4: 512-bit binary vector (64 bytes).
    pub tier4: [u8; 64],
    /// Tier 5: 1024-bit binary vector (128 bytes).
    pub tier5: [u8; 128],
}

impl SidecarHeader {
    /// Encode the header as a 288-byte v1 sidecar blob.
    pub fn encode_v1(&self) -> [u8; SIDECAR_HEADER_SIZE] {
        let mut buf = [0u8; SIDECAR_HEADER_SIZE];
        buf[0..4].copy_from_slice(&SIDECAR_V1_MAGIC);
        buf[4..8].copy_from_slice(&self.fingerprint);
        buf[8..40].copy_from_slice(&self.target_cid);
        buf[40..48].copy_from_slice(&self.tier1);
        buf[48..64].copy_from_slice(&self.tier2);
        buf[64..96].copy_from_slice(&self.tier3);
        buf[96..160].copy_from_slice(&self.tier4);
        buf[160..288].copy_from_slice(&self.tier5);
        buf
    }

    /// Decode a sidecar header from bytes. Accepts both v1 and v2 magic.
    pub fn decode(data: &[u8]) -> SemanticResult<Self> {
        if data.len() < SIDECAR_HEADER_SIZE {
            return Err(SemanticError::TruncatedHeader {
                expected: SIDECAR_HEADER_SIZE,
                actual: data.len(),
            });
        }

        let magic = &data[0..4];
        if magic != SIDECAR_V1_MAGIC && magic != SIDECAR_V2_MAGIC {
            return Err(SemanticError::InvalidMagic);
        }

        let mut fingerprint = [0u8; 4];
        fingerprint.copy_from_slice(&data[4..8]);
        let mut target_cid = [0u8; 32];
        target_cid.copy_from_slice(&data[8..40]);
        let mut tier1 = [0u8; 8];
        tier1.copy_from_slice(&data[40..48]);
        let mut tier2 = [0u8; 16];
        tier2.copy_from_slice(&data[48..64]);
        let mut tier3 = [0u8; 32];
        tier3.copy_from_slice(&data[64..96]);
        let mut tier4 = [0u8; 64];
        tier4.copy_from_slice(&data[96..160]);
        let mut tier5 = [0u8; 128];
        tier5.copy_from_slice(&data[160..288]);

        Ok(Self {
            fingerprint,
            target_cid,
            tier1,
            tier2,
            tier3,
            tier4,
            tier5,
        })
    }

    /// Returns `true` if the given data starts with v2 magic.
    pub fn is_v2(data: &[u8]) -> bool {
        data.len() >= 4 && data[0..4] == SIDECAR_V2_MAGIC
    }

    /// Extract the tier data for a given embedding tier.
    pub fn tier_data(&self, tier: crate::tier::EmbeddingTier) -> &[u8] {
        use crate::tier::EmbeddingTier;
        match tier {
            EmbeddingTier::T1 => &self.tier1,
            EmbeddingTier::T2 => &self.tier2,
            EmbeddingTier::T3 => &self.tier3,
            EmbeddingTier::T4 => &self.tier4,
            EmbeddingTier::T5 => &self.tier5,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fingerprint::model_fingerprint;
    use crate::tier::EmbeddingTier;

    /// Build a test header with deterministic data.
    fn test_header() -> SidecarHeader {
        let fp = model_fingerprint("test-model-v1");
        let mut target_cid = [0u8; 32];
        for (i, b) in target_cid.iter_mut().enumerate() {
            *b = i as u8;
        }
        let mut tier1 = [0u8; 8];
        tier1.fill(0xAA);
        let mut tier2 = [0u8; 16];
        tier2.fill(0xBB);
        let mut tier3 = [0u8; 32];
        tier3.fill(0xCC);
        let mut tier4 = [0u8; 64];
        tier4.fill(0xDD);
        let mut tier5 = [0u8; 128];
        tier5.fill(0xEE);

        SidecarHeader {
            fingerprint: fp,
            target_cid,
            tier1,
            tier2,
            tier3,
            tier4,
            tier5,
        }
    }

    #[test]
    fn sidecar_v1_encode_decode_roundtrip() {
        let header = test_header();
        let encoded = header.encode_v1();
        assert_eq!(encoded.len(), SIDECAR_HEADER_SIZE);

        let decoded = SidecarHeader::decode(&encoded).expect("decode should succeed");
        assert_eq!(decoded, header);
    }

    #[test]
    fn sidecar_decode_truncated_rejects() {
        let short = [0u8; 100];
        let err = SidecarHeader::decode(&short).unwrap_err();
        assert_eq!(
            err,
            SemanticError::TruncatedHeader {
                expected: SIDECAR_HEADER_SIZE,
                actual: 100,
            }
        );
    }

    #[test]
    fn sidecar_decode_bad_magic_rejects() {
        let mut buf = [0u8; SIDECAR_HEADER_SIZE];
        buf[0..4].copy_from_slice(&[0xFF, 0xFF, 0xFF, 0xFF]);
        let err = SidecarHeader::decode(&buf).unwrap_err();
        assert_eq!(err, SemanticError::InvalidMagic);
    }

    #[test]
    fn sidecar_v2_magic_accepted() {
        let header = test_header();
        let mut encoded = header.encode_v1();
        // Overwrite magic with v2
        encoded[0..4].copy_from_slice(&SIDECAR_V2_MAGIC);

        let decoded = SidecarHeader::decode(&encoded).expect("v2 magic should be accepted");
        assert_eq!(decoded.fingerprint, header.fingerprint);
        assert_eq!(decoded.target_cid, header.target_cid);
        assert_eq!(decoded.tier1, header.tier1);
        assert_eq!(decoded.tier2, header.tier2);
        assert_eq!(decoded.tier3, header.tier3);
        assert_eq!(decoded.tier4, header.tier4);
        assert_eq!(decoded.tier5, header.tier5);
    }

    #[test]
    fn sidecar_tier_data_returns_correct_slice() {
        let header = test_header();
        assert_eq!(header.tier_data(EmbeddingTier::T1), &header.tier1);
        assert_eq!(header.tier_data(EmbeddingTier::T2), &header.tier2);
        assert_eq!(header.tier_data(EmbeddingTier::T3), &header.tier3);
        assert_eq!(header.tier_data(EmbeddingTier::T4), &header.tier4);
        assert_eq!(header.tier_data(EmbeddingTier::T5), &header.tier5);
    }
}
