use serde::{Deserialize, Serialize};

/// The cryptographic algorithm suite backing an identity.
///
/// Discriminant values match the UCAN wire format (first byte of token).
/// Serde encoding is pinned to the `#[repr(u8)]` discriminant via
/// `TryFrom<u8>` / `Into<u8>`, ensuring wire stability even if variants
/// are reordered or new ones are inserted.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(try_from = "u8", into = "u8")]
#[repr(u8)]
pub enum CryptoSuite {
    /// Ed25519 signing + X25519 encryption (Reticulum-compatible).
    /// Backward-compatibility layer — NOT post-quantum secure.
    Ed25519 = 0x00,
    /// ML-DSA-65 signing + ML-KEM-768 encryption (NIST FIPS 203/204).
    /// Harmony-native, post-quantum secure.
    MlDsa65 = 0x01,
}

impl From<CryptoSuite> for u8 {
    fn from(s: CryptoSuite) -> u8 {
        s as u8
    }
}

impl TryFrom<u8> for CryptoSuite {
    type Error = u8;

    fn try_from(b: u8) -> Result<Self, u8> {
        match b {
            0x00 => Ok(Self::Ed25519),
            0x01 => Ok(Self::MlDsa65),
            other => Err(other),
        }
    }
}

impl CryptoSuite {
    /// Returns `true` if this suite provides post-quantum security.
    pub fn is_post_quantum(self) -> bool {
        matches!(self, Self::MlDsa65)
    }

    /// Multicodec identifier for the signing algorithm.
    /// Ed25519 = `0x00ed`, ML-DSA-65 = `0x1211` (draft).
    pub fn signing_multicodec(self) -> u16 {
        match self {
            Self::Ed25519 => 0x00ed,
            Self::MlDsa65 => 0x1211,
        }
    }

    /// Multicodec identifier for the encryption/KEM algorithm.
    /// X25519 = `0x00ec`, ML-KEM-768 = `0x120c` (draft).
    pub fn encryption_multicodec(self) -> u16 {
        match self {
            Self::Ed25519 => 0x00ec,
            Self::MlDsa65 => 0x120c,
        }
    }

    /// Construct from a signing multicodec identifier.
    /// Returns `None` for unknown codes.
    pub fn from_signing_multicodec(code: u16) -> Option<Self> {
        match code {
            0x00ed => Some(Self::Ed25519),
            0x1211 => Some(Self::MlDsa65),
            _ => None,
        }
    }

    /// Construct from an encryption multicodec identifier.
    /// Returns `None` for unknown codes.
    pub fn from_encryption_multicodec(code: u16) -> Option<Self> {
        match code {
            0x00ec => Some(Self::Ed25519),
            0x120c => Some(Self::MlDsa65),
            _ => None,
        }
    }

    /// Construct from the UCAN wire discriminant byte (`0x00` or `0x01`).
    /// Returns `None` for unknown values.
    pub fn from_byte(byte: u8) -> Option<Self> {
        Self::try_from(byte).ok()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ed25519_is_not_post_quantum() {
        assert!(!CryptoSuite::Ed25519.is_post_quantum());
    }

    #[test]
    fn ml_dsa65_is_post_quantum() {
        assert!(CryptoSuite::MlDsa65.is_post_quantum());
    }

    #[test]
    fn signing_multicodec_values() {
        assert_eq!(CryptoSuite::Ed25519.signing_multicodec(), 0x00ed);
        assert_eq!(CryptoSuite::MlDsa65.signing_multicodec(), 0x1211);
    }

    #[test]
    fn encryption_multicodec_values() {
        assert_eq!(CryptoSuite::Ed25519.encryption_multicodec(), 0x00ec);
        assert_eq!(CryptoSuite::MlDsa65.encryption_multicodec(), 0x120c);
    }

    #[test]
    fn from_signing_multicodec_round_trip() {
        assert_eq!(
            CryptoSuite::from_signing_multicodec(0x00ed),
            Some(CryptoSuite::Ed25519)
        );
        assert_eq!(
            CryptoSuite::from_signing_multicodec(0x1211),
            Some(CryptoSuite::MlDsa65)
        );
        assert_eq!(CryptoSuite::from_signing_multicodec(0xFFFF), None);
    }

    #[test]
    fn from_encryption_multicodec_round_trip() {
        assert_eq!(
            CryptoSuite::from_encryption_multicodec(0x00ec),
            Some(CryptoSuite::Ed25519)
        );
        assert_eq!(
            CryptoSuite::from_encryption_multicodec(0x120c),
            Some(CryptoSuite::MlDsa65)
        );
        assert_eq!(CryptoSuite::from_encryption_multicodec(0x0000), None);
    }

    #[test]
    fn from_byte_round_trip() {
        assert_eq!(CryptoSuite::from_byte(0x00), Some(CryptoSuite::Ed25519));
        assert_eq!(CryptoSuite::from_byte(0x01), Some(CryptoSuite::MlDsa65));
        assert_eq!(CryptoSuite::from_byte(0x02), None);
        assert_eq!(CryptoSuite::from_byte(0xFF), None);
    }

    #[test]
    fn try_from_u8_round_trip() {
        assert_eq!(CryptoSuite::try_from(0x00), Ok(CryptoSuite::Ed25519));
        assert_eq!(CryptoSuite::try_from(0x01), Ok(CryptoSuite::MlDsa65));
        assert_eq!(CryptoSuite::try_from(0x02), Err(0x02));
    }

    #[test]
    fn wire_discriminant_values() {
        assert_eq!(CryptoSuite::Ed25519 as u8, 0x00);
        assert_eq!(CryptoSuite::MlDsa65 as u8, 0x01);
    }

    #[test]
    fn serde_round_trip_both_variants() {
        for suite in [CryptoSuite::Ed25519, CryptoSuite::MlDsa65] {
            let bytes = postcard::to_allocvec(&suite).unwrap();
            let decoded: CryptoSuite = postcard::from_bytes(&bytes).unwrap();
            assert_eq!(decoded, suite);
        }
    }

    #[test]
    fn serde_encodes_as_discriminant_byte() {
        // Verify serde encoding matches repr(u8) discriminant, not variant index
        let bytes = postcard::to_allocvec(&CryptoSuite::Ed25519).unwrap();
        assert_eq!(bytes, [0x00]);
        let bytes = postcard::to_allocvec(&CryptoSuite::MlDsa65).unwrap();
        assert_eq!(bytes, [0x01]);
    }
}
