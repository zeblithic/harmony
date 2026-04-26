//! Identity backup / restore — BIP39-24 mnemonic and encrypted-file
//! encodings of the master `RecoveryArtifact` seed.
//!
//! ## Threat models
//!
//! The two encodings serve distinct threat models and are independently
//! complete recovery artifacts:
//!
//! - **Mnemonic (24 BIP39 English words):** defense against complete data
//!   loss. User writes the words on paper; theft of the paper = theft of
//!   the identity. No passphrase wrap.
//! - **Encrypted file (Argon2id + XChaCha20-Poly1305):** defense against
//!   file leak. The file is portable (USB / cloud / email); without the
//!   passphrase it is useless. Both must be lost simultaneously to lose
//!   the identity.
//!
//! ## Security-critical invariants
//!
//! - The 13-byte encrypted-file header is bound as AEAD AAD: any
//!   tampering — including a downgrade attack on KDF parameters — is
//!   rejected by Poly1305 before the payload is decrypted.
//! - KDF parameters are checked for **strict equality** against the
//!   locked v1 values BEFORE Argon2id runs. Prevents a CPU/memory DoS
//!   via attacker-controlled `kdf_m_kib`.
//! - Seeds never appear in error strings or `Debug` output.
//! - Passphrases are passed as `SecretString` (auto-zeroizes on drop).
//! - Mnemonic strings are returned wrapped in `Zeroizing<String>` —
//!   knowing the 24 words is mathematically equivalent to knowing the
//!   seed.

pub mod error;
pub(crate) mod wire;
pub(crate) mod mnemonic;
pub(crate) mod encrypted_file;

pub use error::RecoveryError;

use crate::lifecycle::mint::RecoveryArtifact;
use secrecy::SecretString;
use zeroize::Zeroizing;

/// Encode-side input / decode-side output for the encrypted-file's
/// metadata. All fields are optional and non-secret on their own.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RecoveryMetadata {
    pub mint_at: Option<u64>,
    pub comment: Option<String>,
}

/// Result of decoding an encrypted recovery file. The artifact zeroizes
/// its seed on Drop (existing behavior from `lifecycle/mint.rs`).
pub struct RestoredArtifact {
    pub artifact: RecoveryArtifact,
    pub metadata: RecoveryMetadata,
}

impl RestoredArtifact {
    /// Discard the metadata and yield just the artifact.
    pub fn into_artifact(self) -> RecoveryArtifact {
        self.artifact
    }
}

impl RecoveryArtifact {
    /// Encode this artifact's seed as a 24-word BIP39 English mnemonic.
    /// Returns `Zeroizing<String>` because knowing the 24 words is
    /// mathematically equivalent to knowing the seed.
    pub fn to_mnemonic(&self) -> Zeroizing<String> {
        Zeroizing::new(mnemonic::to_mnemonic_inner(self.as_bytes()))
    }

    /// Decode a 24-word BIP39 English mnemonic into a `RecoveryArtifact`.
    /// Whitespace-tolerant, case-insensitive, ASCII-only.
    pub fn from_mnemonic(s: &str) -> Result<Self, RecoveryError> {
        let seed = mnemonic::from_mnemonic_inner(s)?;
        Ok(Self::from_seed(seed))
    }

    /// Encode this artifact + metadata as a passphrase-encrypted file.
    ///
    /// Output is a self-contained portable artifact: header || salt || nonce
    /// || ciphertext || tag, capped at 1024 bytes total. Each call generates
    /// a fresh random salt + nonce, so re-encoding the same artifact +
    /// passphrase produces different bytes (decrypt yields the same seed).
    ///
    /// Returns [`RecoveryError::CommentTooLong`] if `metadata.comment`
    /// exceeds 256 bytes. All other failure modes are infallible after
    /// argument validation (Argon2id and XChaCha20-Poly1305 cannot fail
    /// for in-bounds inputs).
    pub fn to_encrypted_file(
        &self,
        passphrase: &SecretString,
        metadata: &RecoveryMetadata,
    ) -> Result<Vec<u8>, RecoveryError> {
        // Pre-check comment length BEFORE cloning, so a hostile-large
        // comment is rejected with no wasted allocation. encrypt_inner
        // re-checks defensively (defense-in-depth + non-public-API
        // callers might bypass us), but that check is now redundant
        // for the public path.
        if let Some(c) = metadata.comment.as_ref() {
            if c.len() > encrypted_file::MAX_COMMENT_LEN {
                return Err(RecoveryError::CommentTooLong {
                    actual: c.len(),
                    max: encrypted_file::MAX_COMMENT_LEN,
                });
            }
        }
        encrypted_file::encrypt_inner(
            passphrase,
            self.as_bytes(),
            metadata.mint_at,
            metadata.comment.clone(),
        )
    }

    /// Decode a passphrase-encrypted recovery file.
    ///
    /// Errors map to specific failure modes:
    /// - [`RecoveryError::TooSmall`] / [`RecoveryError::TooLarge`]: file
    ///   length out of [69, 1024] byte range — rejected before Argon2id runs.
    /// - [`RecoveryError::UnrecognizedFormat`]: not a Harmony recovery file
    ///   (magic byte mismatch).
    /// - [`RecoveryError::UnsupportedVersion`]: format version known but
    ///   not implemented by this build.
    /// - [`RecoveryError::UnsupportedKdfId`] / [`RecoveryError::UnsupportedKdfParams`]:
    ///   non-standard KDF (defends against attacker-controlled DoS via
    ///   absurd Argon2 memory params — header is bound as AEAD AAD so
    ///   tampering is rejected here, BEFORE the expensive KDF runs).
    /// - [`RecoveryError::WrongPassphraseOrCorrupt`]: AEAD tag rejected.
    ///   The two are deliberately conflated: distinguishing them requires
    ///   speculative info the AEAD does not (and should not) reveal.
    /// - [`RecoveryError::PayloadDecodeFailed`] / [`RecoveryError::UnexpectedPayloadFormat`]:
    ///   AEAD passed but the inner payload is malformed or schema-mismatched.
    pub fn from_encrypted_file(
        bytes: &[u8],
        passphrase: &SecretString,
    ) -> Result<RestoredArtifact, RecoveryError> {
        let (seed, mint_at, comment) =
            encrypted_file::decrypt_inner(bytes, passphrase)?;
        Ok(RestoredArtifact {
            artifact: RecoveryArtifact::from_seed(seed),
            metadata: RecoveryMetadata { mint_at, comment },
        })
    }
}

// Test-fixtures public re-exports — exposed at the recovery module
// root so integration tests in `tests/` can reach them without
// navigating into private submodules.
#[cfg(feature = "test-fixtures")]
pub use encrypted_file::{encrypt_with_params_for_test, FORMAT_STRING, RecoveryFileBody};
#[cfg(feature = "test-fixtures")]
pub use wire::{NONCE_LEN, SALT_LEN};

#[cfg(test)]
mod equivalence_tests {
    use super::*;
    use subtle::ConstantTimeEq;

    /// Both encodings of the same seed must decode to artifacts that
    /// produce the same master `identity_hash` — this is the load-bearing
    /// correctness check that mnemonic and encrypted-file backups are
    /// truly equivalent.
    #[test]
    fn mnemonic_and_encrypted_file_yield_identical_master_pubkey() {
        let original = RecoveryArtifact::from_seed([42u8; 32]);
        let original_id = original.master_pubkey_bundle().identity_hash();

        // Round-trip via mnemonic.
        let m = original.to_mnemonic();
        let from_m = RecoveryArtifact::from_mnemonic(&m).unwrap();
        let id_via_m = from_m.master_pubkey_bundle().identity_hash();

        // Round-trip via encrypted file.
        let pass = SecretString::from("equiv-test".to_string());
        let bytes = original
            .to_encrypted_file(&pass, &RecoveryMetadata::default())
            .unwrap();
        let from_f = RecoveryArtifact::from_encrypted_file(&bytes, &pass)
            .unwrap()
            .into_artifact();
        let id_via_f = from_f.master_pubkey_bundle().identity_hash();

        // Constant-time equality on the identity hashes.
        assert!(bool::from(id_via_m.ct_eq(&original_id)));
        assert!(bool::from(id_via_f.ct_eq(&original_id)));
        assert!(bool::from(id_via_m.ct_eq(&id_via_f)));
    }

    #[test]
    fn restored_artifact_into_artifact_drops_metadata() {
        let original = RecoveryArtifact::from_seed([1u8; 32]);
        let pass = SecretString::from("io-test".to_string());
        let metadata = RecoveryMetadata {
            mint_at: Some(1_700_000_000),
            comment: Some("foo".into()),
        };
        let bytes = original.to_encrypted_file(&pass, &metadata).unwrap();
        let restored = RecoveryArtifact::from_encrypted_file(&bytes, &pass).unwrap();
        assert_eq!(restored.metadata, metadata);
        let _ = restored.into_artifact(); // type compiles, metadata gone
    }
}
