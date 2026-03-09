// SPDX-License-Identifier: Apache-2.0 OR MIT
//! Embedding tier definitions — progressive resolution from 64 to 1024 bits.

/// The five embedding tiers, each a nested prefix of the MRL vector.
///
/// Tier N contains the first 2^(N+5) bits of the binary-quantized vector.
/// Higher tiers are strict supersets of lower tiers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum EmbeddingTier {
    /// 64-bit (8 bytes) — single GPR comparison. Network fanout tier.
    T1 = 1,
    /// 128-bit (16 bytes) — SSE register. Lightweight indexing.
    T2 = 2,
    /// 256-bit (32 bytes) — AVX2 register. Community search tier.
    T3 = 3,
    /// 512-bit (64 bytes) — AVX-512 register. High precision.
    T4 = 4,
    /// 1024-bit (128 bytes) — 2×AVX-512. Maximum precision.
    T5 = 5,
}

impl EmbeddingTier {
    /// Number of bits in this tier's binary vector.
    pub const fn bit_count(self) -> usize {
        match self {
            Self::T1 => 64,
            Self::T2 => 128,
            Self::T3 => 256,
            Self::T4 => 512,
            Self::T5 => 1024,
        }
    }

    /// Number of bytes in this tier's binary vector.
    pub const fn byte_count(self) -> usize {
        self.bit_count() / 8
    }

    /// Offset of this tier's data within the sidecar fixed header.
    /// Tier data starts at byte 40 (after magic + fingerprint + target CID).
    pub const fn sidecar_offset(self) -> usize {
        match self {
            Self::T1 => 40,
            Self::T2 => 48,
            Self::T3 => 64,
            Self::T4 => 96,
            Self::T5 => 160,
        }
    }
}
