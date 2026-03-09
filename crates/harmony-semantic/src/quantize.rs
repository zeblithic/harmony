// SPDX-License-Identifier: Apache-2.0 OR MIT
//! Binary quantization — float32 → binary vector conversion.

use alloc::vec::Vec;

use crate::error::{SemanticError, SemanticResult};
use crate::sidecar::SidecarHeader;

/// Quantize a float32 MRL vector to a binary vector.
///
/// Each dimension: if value > 0.0, output bit 1; else 0.
/// The output is packed into bytes, MSB first within each byte.
pub fn quantize_to_binary(float_vec: &[f32]) -> Vec<u8> {
    let byte_count = float_vec.len().div_ceil(8);
    let mut bytes = alloc::vec![0u8; byte_count];

    for (i, &val) in float_vec.iter().enumerate() {
        if val > 0.0 {
            bytes[i / 8] |= 1 << (7 - (i % 8));
        }
    }

    bytes
}

/// Pack a quantized binary vector into the five sidecar tiers.
///
/// The input must be at least 64 dimensions (8 bytes after quantization).
/// If shorter than 1024 dimensions, the remaining tiers are zero-filled.
pub fn pack_tiers(
    float_vec: &[f32],
    fingerprint: [u8; 4],
    target_cid: [u8; 32],
) -> SemanticResult<SidecarHeader> {
    if float_vec.is_empty() {
        return Err(SemanticError::DimensionMismatch {
            expected: 64,
            actual: 0,
        });
    }

    let binary = quantize_to_binary(float_vec);

    // Tier 1: first 8 bytes (64 bits). Minimum required.
    if binary.len() < 8 {
        return Err(SemanticError::DimensionMismatch {
            expected: 64,
            actual: float_vec.len(),
        });
    }

    let mut tier1 = [0u8; 8];
    tier1.copy_from_slice(&binary[..8]);

    let mut tier2 = [0u8; 16];
    let t2_len = binary.len().min(16);
    tier2[..t2_len].copy_from_slice(&binary[..t2_len]);

    let mut tier3 = [0u8; 32];
    let t3_len = binary.len().min(32);
    tier3[..t3_len].copy_from_slice(&binary[..t3_len]);

    let mut tier4 = [0u8; 64];
    let t4_len = binary.len().min(64);
    tier4[..t4_len].copy_from_slice(&binary[..t4_len]);

    let mut tier5 = [0u8; 128];
    let t5_len = binary.len().min(128);
    tier5[..t5_len].copy_from_slice(&binary[..t5_len]);

    Ok(SidecarHeader {
        fingerprint,
        target_cid,
        tier1,
        tier2,
        tier3,
        tier4,
        tier5,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quantize_positive_to_one() {
        let input = [1.0f32, -1.0, 0.5, -0.5];
        let binary = quantize_to_binary(&input);
        // 1 → bit 1, -1 → bit 0, 0.5 → bit 1, -0.5 → bit 0
        // MSB first: 0b1010_0000 = 0xA0
        assert_eq!(binary, &[0xA0]);
    }

    #[test]
    fn quantize_zero_to_zero() {
        // 0.0 is NOT > 0.0, so it maps to bit 0.
        let input = [0.0f32; 8];
        let binary = quantize_to_binary(&input);
        assert_eq!(binary, &[0x00]);
    }

    #[test]
    fn quantize_1024_dimensions() {
        // 1024 floats → 128 bytes.
        let input: Vec<f32> = (0..1024).map(|i| if i % 2 == 0 { 1.0 } else { -1.0 }).collect();
        let binary = quantize_to_binary(&input);
        assert_eq!(binary.len(), 128);
        // Every even bit is 1, odd is 0 → each byte is 0b10101010 = 0xAA.
        for &b in &binary {
            assert_eq!(b, 0xAA);
        }
    }

    #[test]
    fn pack_tiers_roundtrip() {
        // Create a 1024-dimension vector, pack it, then verify tiers.
        let input: Vec<f32> = (0..1024).map(|i| if i % 3 == 0 { 1.0 } else { -0.5 }).collect();
        let fp = [0xDE, 0xAD, 0xBE, 0xEF];
        let cid = [0x42u8; 32];

        let header = pack_tiers(&input, fp, cid).expect("pack should succeed");
        assert_eq!(header.fingerprint, fp);
        assert_eq!(header.target_cid, cid);

        // Encode as v1, decode back, verify identical.
        let encoded = header.encode_v1();
        let decoded = SidecarHeader::decode(&encoded).expect("decode should succeed");
        assert_eq!(decoded, header);
    }

    #[test]
    fn pack_tiers_rejects_empty() {
        let fp = [0u8; 4];
        let cid = [0u8; 32];
        let err = pack_tiers(&[], fp, cid).unwrap_err();
        assert_eq!(
            err,
            SemanticError::DimensionMismatch {
                expected: 64,
                actual: 0,
            }
        );
    }

    #[test]
    fn tiers_are_nested_prefixes() {
        // Build from a 1024-dim vector so all tiers are fully populated.
        let input: Vec<f32> = (0..1024)
            .map(|i| if (i * 7 + 3) % 11 > 5 { 1.0 } else { -1.0 })
            .collect();
        let fp = [0x01, 0x02, 0x03, 0x04];
        let cid = [0xAB; 32];

        let header = pack_tiers(&input, fp, cid).expect("pack should succeed");

        // tier1 == tier2[..8]
        assert_eq!(&header.tier1[..], &header.tier2[..8]);
        // tier2 == tier3[..16]
        assert_eq!(&header.tier2[..], &header.tier3[..16]);
        // tier3 == tier4[..32]
        assert_eq!(&header.tier3[..], &header.tier4[..32]);
        // tier4 == tier5[..64]
        assert_eq!(&header.tier4[..], &header.tier5[..64]);
    }
}
