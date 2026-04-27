//! Encrypted-file wire format: 13-byte header bound as AAD, salt, nonce,
//! ciphertext, Poly1305 tag. The header layout matches the structure of
//! ZEB-174's `identity.enc` but uses magic `HRMR` instead of `HRMI` so
//! the two file types can never be confused.
//!
//! Layout:
//! ```text
//! offset  size  field
//! 0       4     magic = b"HRMR"
//! 4       1     format_version = 0x01
//! 5       1     kdf_id = 0x01 (Argon2id)
//! 6       4     kdf_m_kib (u32 BE) = 65536
//! 10      2     kdf_t (u16 BE) = 3
//! 12      1     kdf_p (u8) = 1
//! 13      16    salt
//! 29      24    nonce (XChaCha20-Poly1305 needs 24)
//! 53      var   ciphertext
//! end-16  16    Poly1305 tag
//! ```

use crate::recovery::error::RecoveryError;

pub const MAGIC: &[u8; 4] = b"HRMR";
pub const FORMAT_VERSION: u8 = 0x01;
pub const KDF_ID_ARGON2ID: u8 = 0x01;
pub const KDF_M_KIB: u32 = 65536;
pub const KDF_T: u16 = 3;
pub const KDF_P: u8 = 1;
pub const KDF_OUT_LEN: usize = 32;

pub const HEADER_LEN: usize = 13;
pub const SALT_LEN: usize = 16;
pub const NONCE_LEN: usize = 24;
pub const TAG_LEN: usize = 16;

/// Wire-layer minimum: header + salt + nonce + 0-byte ciphertext + tag.
/// A valid file in practice is larger because the smallest CBOR-encoded
/// `RecoveryFileBody` exceeds zero bytes; this is the parse-time guard.
///
/// Both bounds are enforced by `encrypted_file::decrypt_inner` (Task 10),
/// not by `parse_header` — the parser is scoped to the 13-byte header
/// slice and does not see the full file length. Defining the constants
/// here keeps the v1 wire-format facts in one place.
pub const MIN_FILE_LEN: usize = HEADER_LEN + SALT_LEN + NONCE_LEN + TAG_LEN; // 69
pub const MAX_FILE_LEN: usize = 1024;

/// Build the 13-byte header bytes. Same content every time (no per-call
/// variability) — this is a constant-output helper that simply serializes
/// the locked KDF parameters.
pub fn serialize_header() -> [u8; HEADER_LEN] {
    let mut out = [0u8; HEADER_LEN];
    out[0..4].copy_from_slice(MAGIC);
    out[4] = FORMAT_VERSION;
    out[5] = KDF_ID_ARGON2ID;
    out[6..10].copy_from_slice(&KDF_M_KIB.to_be_bytes());
    out[10..12].copy_from_slice(&KDF_T.to_be_bytes());
    out[12] = KDF_P;
    out
}

/// Parse and validate a 13-byte header slice. STRICT EQUALITY on KDF
/// parameters: any deviation from the locked values returns
/// `UnsupportedKdfId` / `UnsupportedKdfParams` BEFORE the (potentially
/// attacker-controlled) bytes are passed to Argon2id. This prevents a
/// CPU/memory DoS via an adversarially-large `kdf_m_kib`.
pub fn parse_header(bytes: &[u8]) -> Result<(), RecoveryError> {
    if bytes.len() < HEADER_LEN {
        return Err(RecoveryError::TooSmall(bytes.len()));
    }
    if &bytes[0..4] != MAGIC {
        return Err(RecoveryError::UnrecognizedFormat);
    }
    if bytes[4] != FORMAT_VERSION {
        return Err(RecoveryError::UnsupportedVersion(bytes[4]));
    }
    if bytes[5] != KDF_ID_ARGON2ID {
        return Err(RecoveryError::UnsupportedKdfId(bytes[5]));
    }
    let m_kib = u32::from_be_bytes(bytes[6..10].try_into().unwrap());
    let t = u16::from_be_bytes(bytes[10..12].try_into().unwrap());
    let p = bytes[12];
    if m_kib != KDF_M_KIB || t != KDF_T || p != KDF_P {
        return Err(RecoveryError::UnsupportedKdfParams { id: bytes[5] });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn header_size_constants_add_up() {
        assert_eq!(HEADER_LEN, 13);
        assert_eq!(MIN_FILE_LEN, 13 + 16 + 24 + 16);
        assert_eq!(MIN_FILE_LEN, 69);
    }

    #[test]
    fn serialize_header_is_deterministic() {
        assert_eq!(serialize_header(), serialize_header());
    }

    #[test]
    fn serialize_header_layout_is_exact() {
        let h = serialize_header();
        assert_eq!(&h[0..4], b"HRMR");
        assert_eq!(h[4], 0x01);
        assert_eq!(h[5], 0x01);
        assert_eq!(u32::from_be_bytes(h[6..10].try_into().unwrap()), 65536);
        assert_eq!(u16::from_be_bytes(h[10..12].try_into().unwrap()), 3);
        assert_eq!(h[12], 1);
    }

    /// Golden vector: locks the v1 wire-format header byte-for-byte. Any
    /// future refactor that consistently flips both serialize and parse
    /// would silently pass `serialize_header_layout_is_exact` and
    /// `parse_header_round_trips` together — this test is the canary that
    /// catches that class of regression.
    #[test]
    fn serialize_header_matches_frozen_golden() {
        let expected: [u8; 13] = [
            0x48, 0x52, 0x4d, 0x52,   // "HRMR"
            0x01,                       // format_version
            0x01,                       // kdf_id (Argon2id)
            0x00, 0x01, 0x00, 0x00,   // kdf_m_kib = 65536 (BE u32)
            0x00, 0x03,                 // kdf_t = 3 (BE u16)
            0x01,                       // kdf_p
        ];
        assert_eq!(serialize_header(), expected);
    }

    #[test]
    fn parse_header_round_trips() {
        assert!(parse_header(&serialize_header()).is_ok());
    }

    #[test]
    fn parse_header_rejects_short() {
        let err = parse_header(&[0u8; 5]).unwrap_err();
        assert!(matches!(err, RecoveryError::TooSmall(5)));
    }

    #[test]
    fn parse_header_rejects_wrong_magic() {
        let mut h = serialize_header();
        h[0] = b'X';
        let err = parse_header(&h).unwrap_err();
        assert!(matches!(err, RecoveryError::UnrecognizedFormat));
    }

    #[test]
    fn parse_header_rejects_wrong_version() {
        let mut h = serialize_header();
        h[4] = 0x02;
        let err = parse_header(&h).unwrap_err();
        assert!(matches!(err, RecoveryError::UnsupportedVersion(0x02)));
    }

    #[test]
    fn parse_header_rejects_wrong_kdf_id() {
        let mut h = serialize_header();
        h[5] = 0x99;
        let err = parse_header(&h).unwrap_err();
        assert!(matches!(err, RecoveryError::UnsupportedKdfId(0x99)));
    }

    #[test]
    fn parse_header_rejects_wrong_m_kib() {
        let mut h = serialize_header();
        h[6..10].copy_from_slice(&32768u32.to_be_bytes());
        let err = parse_header(&h).unwrap_err();
        assert!(matches!(err, RecoveryError::UnsupportedKdfParams { id: 0x01 }));
    }

    #[test]
    fn parse_header_rejects_wrong_t() {
        let mut h = serialize_header();
        h[10..12].copy_from_slice(&7u16.to_be_bytes());
        let err = parse_header(&h).unwrap_err();
        assert!(matches!(err, RecoveryError::UnsupportedKdfParams { id: 0x01 }));
    }

    #[test]
    fn parse_header_rejects_wrong_p() {
        let mut h = serialize_header();
        h[12] = 4;
        let err = parse_header(&h).unwrap_err();
        assert!(matches!(err, RecoveryError::UnsupportedKdfParams { id: 0x01 }));
    }
}
