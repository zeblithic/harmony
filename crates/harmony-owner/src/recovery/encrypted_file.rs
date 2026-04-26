//! Argon2id + XChaCha20-Poly1305 encrypted-file encode/decode for the
//! 32-byte recovery seed plus optional metadata. Format spec lives in
//! `crate::recovery::wire`.

use serde::{Deserialize, Serialize};

pub const FORMAT_STRING: &str = "harmony-owner-recovery-v1";
pub const MAX_COMMENT_LEN: usize = 256;

/// CBOR-encoded plaintext payload inside the AEAD ciphertext. The `format`
/// string is defense-in-depth: even though Poly1305 already proves the
/// payload was produced by someone with the passphrase, validating the
/// format string after decryption protects against future format
/// bifurcations being silently accepted by older parsers.
///
/// `pub` (not `pub(crate)`) because `encrypt_with_params_for_test` accepts
/// `&RecoveryFileBody` as a parameter, and that helper is callable from
/// integration tests when the `test-fixtures` feature is enabled. The
/// struct is only visible at all when the `recovery` feature is on
/// (the entire module is gated).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RecoveryFileBody {
    pub format: String,
    #[serde(with = "serde_bytes")]
    pub seed: [u8; 32],
    pub mint_at: Option<u64>,
    pub comment: Option<String>,
}

#[cfg(test)]
mod body_tests {
    use super::*;

    #[test]
    fn cbor_round_trip_minimal() {
        let body = RecoveryFileBody {
            format: FORMAT_STRING.into(),
            seed: [42u8; 32],
            mint_at: None,
            comment: None,
        };
        let bytes = crate::cbor::to_canonical(&body).unwrap();
        let back: RecoveryFileBody = ciborium::de::from_reader(&bytes[..]).unwrap();
        assert_eq!(back, body);
    }

    #[test]
    fn cbor_round_trip_full() {
        let body = RecoveryFileBody {
            format: FORMAT_STRING.into(),
            seed: [42u8; 32],
            mint_at: Some(1_700_000_000),
            comment: Some("primary owner".into()),
        };
        let bytes = crate::cbor::to_canonical(&body).unwrap();
        let back: RecoveryFileBody = ciborium::de::from_reader(&bytes[..]).unwrap();
        assert_eq!(back, body);
    }
}
