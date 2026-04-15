//! Signed-claim types and sans-I/O verification logic. See spec §4 and §5.1.

use serde::{Deserialize, Serialize};

/// Every byte range covered by a signature is the canonical-CBOR encoding
/// of the payload struct. Signers encode *once* and keep that byte slice
/// until the signature is produced; verifiers re-encode and compare.
pub fn canonical_cbor<T: Serialize>(value: &T) -> Result<Vec<u8>, ciborium::ser::Error<std::io::Error>> {
    let mut buf = Vec::new();
    ciborium::ser::into_writer(value, &mut buf)?;
    Ok(buf)
}

/// Serde helper for `[u8; N]` fields: encode as a CBOR byte string
/// (major type 2) instead of the default "array of integers" (major
/// type 4) that `#[derive(Serialize)]` would emit. Required for the
/// spec's canonical-CBOR wire format (RFC 8949 §4.2).
mod serde_byte_array {
    use serde::{Deserializer, Serializer};

    pub fn serialize<S, const N: usize>(bytes: &[u8; N], serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_bytes(bytes.as_slice())
    }

    pub fn deserialize<'de, D, const N: usize>(deserializer: D) -> Result<[u8; N], D::Error>
    where
        D: Deserializer<'de>,
    {
        struct Visitor<const N: usize>;
        impl<'de, const N: usize> serde::de::Visitor<'de> for Visitor<N> {
            type Value = [u8; N];
            fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(f, "a byte string of exactly {N} bytes")
            }
            fn visit_bytes<E: serde::de::Error>(self, v: &[u8]) -> Result<[u8; N], E> {
                v.try_into().map_err(|_| E::custom(format!("expected {N} bytes, got {}", v.len())))
            }
            fn visit_borrowed_bytes<E: serde::de::Error>(self, v: &'de [u8]) -> Result<[u8; N], E> {
                self.visit_bytes(v)
            }
            fn visit_byte_buf<E: serde::de::Error>(self, v: Vec<u8>) -> Result<[u8; N], E> {
                self.visit_bytes(&v)
            }
        }
        deserializer.deserialize_bytes(Visitor::<N>)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum MasterPubkey {
    Ed25519(#[serde(with = "serde_byte_array")] [u8; 32]),
    // MlDsa65(Box<[u8; 1952]>) — reserved; see spec §2.1, §4.1.
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SigningPubkey {
    Ed25519(#[serde(with = "serde_byte_array")] [u8; 32]),
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Signature {
    Ed25519(#[serde(with = "serde_byte_array")] [u8; 64]),
    // MlDsa65(Box<[u8; 3309]>), Hybrid(Box<HybridSignature>) — reserved.
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SignatureAlg {
    Ed25519,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DomainRecord {
    pub version: u8,
    pub master_pubkey: MasterPubkey,
    #[serde(with = "serde_byte_array")]
    pub domain_salt: [u8; 16],
    pub alg: SignatureAlg,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SigningKeyCert {
    pub version: u8,
    #[serde(with = "serde_byte_array")]
    pub signing_key_id: [u8; 8],
    pub signing_pubkey: SigningPubkey,
    pub valid_from: u64,
    pub valid_until: u64,
    pub domain: String,
    pub master_signature: Signature,
}

/// Signable view of `SigningKeyCert`: everything the master key signs.
/// The `master_signature` field is excluded; encoding produces the exact
/// byte range a verifier re-encodes and hands to `Ed25519::verify`.
#[derive(Debug, Serialize)]
pub struct SigningKeyCertSignable<'a> {
    pub version: u8,
    #[serde(with = "serde_byte_array")]
    pub signing_key_id: [u8; 8],
    pub signing_pubkey: &'a SigningPubkey,
    pub valid_from: u64,
    pub valid_until: u64,
    pub domain: &'a str,
}

impl SigningKeyCert {
    pub fn signable(&self) -> SigningKeyCertSignable<'_> {
        SigningKeyCertSignable {
            version: self.version,
            signing_key_id: self.signing_key_id,
            signing_pubkey: &self.signing_pubkey,
            valid_from: self.valid_from,
            valid_until: self.valid_until,
            domain: &self.domain,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ClaimPayload {
    pub version: u8,
    pub domain: String,
    #[serde(with = "serde_byte_array")]
    pub hashed_local_part: [u8; 32],
    pub email: String,
    #[serde(with = "serde_byte_array")]
    pub identity_hash: [u8; 16],
    pub issued_at: u64,
    pub expires_at: u64,
    pub serial: u64,
    #[serde(with = "serde_byte_array")]
    pub signing_key_id: [u8; 8],
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SignedClaim {
    pub payload: ClaimPayload,
    pub cert: SigningKeyCert,
    pub claim_signature: Signature,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RevocationList {
    pub version: u8,
    pub domain: String,
    pub issued_at: u64,
    pub revoked_certs: Vec<SigningKeyCert>,
    pub master_signature: Signature,
}

#[derive(Debug, Serialize)]
pub struct RevocationListSignable<'a> {
    pub version: u8,
    pub domain: &'a str,
    pub issued_at: u64,
    pub revoked_certs: &'a [SigningKeyCert],
}

impl RevocationList {
    pub fn signable(&self) -> RevocationListSignable<'_> {
        RevocationListSignable {
            version: self.version,
            domain: &self.domain,
            issued_at: self.issued_at,
            revoked_certs: &self.revoked_certs,
        }
    }
}

use sha2::{Digest, Sha256};

/// Canonical hashed_local_part per spec §4.3.
///
/// Local-part is NOT lowercased (RFC 5321 §2.3.11 defines local-parts
/// as case-sensitive). Domain casing is the caller's responsibility —
/// resolver always lowercases before handing it to anything in this
/// module.
pub fn hashed_local_part(local_part: &str, domain_salt: &[u8; 16]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(local_part.as_bytes());
    hasher.update([0x00]);
    hasher.update(domain_salt);
    hasher.finalize().into()
}

use ed25519_dalek::{Signature as EdSignature, Verifier, VerifyingKey};
use harmony_identity::IdentityHash;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifiedBinding {
    pub domain: String,
    pub email: String,
    pub identity_hash: IdentityHash,
    pub serial: u64,
    pub claim_expires_at: u64,
    pub signing_key_id: [u8; 8],
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum VerifyError {
    #[error("claim.domain, cert.domain, and queried_domain must all agree")]
    DomainMismatch,
    #[error("hashed_local_part does not match SHA-256(local_part || 0x00 || salt)")]
    HashedLocalPartMismatch,
    #[error("master signature over signing-key cert is invalid")]
    CertSignatureInvalid,
    #[error("claim signature under signing key is invalid")]
    ClaimSignatureInvalid,
    #[error("cert not yet valid (valid_from = {valid_from})")]
    CertNotYetValid { valid_from: u64 },
    #[error("cert expired (valid_until = {valid_until})")]
    CertExpired { valid_until: u64 },
    #[error("cert revoked at {revoked_at}")]
    CertRevoked { revoked_at: u64 },
    #[error("claim expired (expires_at = {expires_at})")]
    ClaimExpired { expires_at: u64 },
    #[error("unsupported version byte: {0}")]
    UnsupportedVersion(u8),
    #[error("unsupported signature algorithm")]
    UnsupportedAlgorithm,
    #[error("canonical CBOR encoding failed (unexpected)")]
    EncodingFailed,
}

/// View over the revocation cache passed into `verify`. The resolver
/// fills this from its revocation cache; tests use a trivial empty or
/// hand-built view.
#[derive(Debug, Default, Clone)]
pub struct RevocationView {
    /// Map of `signing_key_id -> (revoked_at = cert.valid_until)`. All
    /// certs in this view have `valid_until` in the past (by
    /// construction — see `RevocationList::revoked_certs` spec §4.4).
    pub revoked: std::collections::HashMap<[u8; 8], u64>,
}

impl RevocationView {
    pub fn empty() -> Self {
        Self::default()
    }
    pub fn insert(&mut self, signing_key_id: [u8; 8], revoked_at: u64) {
        self.revoked.insert(signing_key_id, revoked_at);
    }
}

impl SignedClaim {
    /// Spec §4.3 verification algorithm. `queried_domain` must be the
    /// domain the resolver intended to look up — prevents reflect
    /// attacks where a signed claim for `attacker.example` is served
    /// in response to a query for `q8.fyi`.
    pub fn verify(
        &self,
        domain_record: &DomainRecord,
        revocations: &RevocationView,
        now: u64,
        clock_skew_tolerance_secs: u64,
    ) -> Result<VerifiedBinding, VerifyError> {
        self.verify_against(
            &self.payload.domain,
            domain_record,
            revocations,
            now,
            clock_skew_tolerance_secs,
        )
    }

    /// Same as `verify`, but the caller supplies the queried domain
    /// explicitly. The resolver uses this because it knows which
    /// domain it actually queried.
    pub fn verify_against(
        &self,
        queried_domain: &str,
        domain_record: &DomainRecord,
        revocations: &RevocationView,
        now: u64,
        tolerance: u64,
    ) -> Result<VerifiedBinding, VerifyError> {
        // 1. Domain agreement.
        if self.payload.domain != queried_domain || self.cert.domain != queried_domain {
            return Err(VerifyError::DomainMismatch);
        }

        // 2. Version bytes.
        if self.payload.version != 1 {
            return Err(VerifyError::UnsupportedVersion(self.payload.version));
        }
        if self.cert.version != 1 {
            return Err(VerifyError::UnsupportedVersion(self.cert.version));
        }

        // 3. hashed_local_part consistency.
        let local_part = self
            .payload
            .email
            .split('@')
            .next()
            .unwrap_or("");
        let computed = hashed_local_part(local_part, &domain_record.domain_salt);
        if computed != self.payload.hashed_local_part {
            return Err(VerifyError::HashedLocalPartMismatch);
        }

        // 4. Master signature over the cert.
        let master_vk = match domain_record.master_pubkey {
            MasterPubkey::Ed25519(bytes) => VerifyingKey::from_bytes(&bytes)
                .map_err(|_| VerifyError::CertSignatureInvalid)?,
        };
        let cert_bytes = canonical_cbor(&self.cert.signable())
            .map_err(|_| VerifyError::EncodingFailed)?;
        let master_sig = match self.cert.master_signature {
            Signature::Ed25519(s) => EdSignature::from_bytes(&s),
        };
        master_vk
            .verify(&cert_bytes, &master_sig)
            .map_err(|_| VerifyError::CertSignatureInvalid)?;

        // 5. Cert validity window (with tolerance).
        if now.saturating_add(tolerance) < self.cert.valid_from {
            return Err(VerifyError::CertNotYetValid {
                valid_from: self.cert.valid_from,
            });
        }
        if now > self.cert.valid_until.saturating_add(tolerance) {
            return Err(VerifyError::CertExpired {
                valid_until: self.cert.valid_until,
            });
        }

        // 6. Revocation check with grandfathering.
        if let Some(&revoked_at) = revocations.revoked.get(&self.cert.signing_key_id) {
            if self.payload.issued_at > revoked_at {
                return Err(VerifyError::CertRevoked { revoked_at });
            }
            // Grandfathered (issued_at <= revoked_at): continue.
        }

        // 7. Claim signature under the signing key.
        let signing_vk = match self.cert.signing_pubkey {
            SigningPubkey::Ed25519(bytes) => VerifyingKey::from_bytes(&bytes)
                .map_err(|_| VerifyError::ClaimSignatureInvalid)?,
        };
        let payload_bytes = canonical_cbor(&self.payload)
            .map_err(|_| VerifyError::EncodingFailed)?;
        let claim_sig = match self.claim_signature {
            Signature::Ed25519(s) => EdSignature::from_bytes(&s),
        };
        signing_vk
            .verify(&payload_bytes, &claim_sig)
            .map_err(|_| VerifyError::ClaimSignatureInvalid)?;

        // 8. Claim expiry.
        if now > self.payload.expires_at.saturating_add(tolerance) {
            return Err(VerifyError::ClaimExpired {
                expires_at: self.payload.expires_at,
            });
        }

        Ok(VerifiedBinding {
            domain: self.payload.domain.clone(),
            email: self.payload.email.clone(),
            identity_hash: self.payload.identity_hash,
            serial: self.payload.serial,
            claim_expires_at: self.payload.expires_at,
            signing_key_id: self.cert.signing_key_id,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(feature = "test-support")]
    use crate::test_support::{ClaimBuilder, TestDomain};
    #[cfg(feature = "test-support")]
    use rand_core::OsRng;

    #[cfg(feature = "test-support")]
    const NOW: u64 = 2_000_000_000;
    #[cfg(feature = "test-support")]
    const TOLERANCE: u64 = 60;

    #[test]
    fn domain_record_roundtrips_via_canonical_cbor() {
        let rec = DomainRecord {
            version: 1,
            master_pubkey: MasterPubkey::Ed25519([7u8; 32]),
            domain_salt: [0x5au8; 16],
            alg: SignatureAlg::Ed25519,
        };
        let bytes = canonical_cbor(&rec).expect("encode");
        let decoded: DomainRecord = ciborium::de::from_reader(&bytes[..]).expect("decode");
        assert_eq!(decoded, rec);
    }

    #[test]
    fn signature_encodes_as_cbor_byte_string_not_array() {
        // A 64-byte signature must serialize as CBOR major type 2 (byte string),
        // not major type 4 (array). For N=64, the byte-string prefix is 0x58 0x40
        // (two bytes), while an array-of-64 would be 0x98 0x40 (two bytes) followed
        // by 64 one-byte encodings = 130 total payload bytes. Byte string: 2 + 64 = 66.
        let sig = Signature::Ed25519([0x42u8; 64]);
        let bytes = canonical_cbor(&sig).expect("encode");
        // CBOR enum encoding as "Ed25519" + bstr(64). The bstr header must appear.
        // The 0x58 is the major-type-2 header for a byte string of length 0x40 (64).
        assert!(
            bytes.windows(2).any(|w| w == [0x58, 0x40]),
            "expected CBOR byte-string header 0x58 0x40 in encoded signature, got: {:?}",
            bytes
        );
        // Roundtrip safety:
        let decoded: Signature = ciborium::de::from_reader(&bytes[..]).expect("decode");
        assert_eq!(decoded, sig);
    }

    #[test]
    fn hashed_local_part_matches_spec_formula() {
        // SHA-256("alice" || 0x00 || domain_salt)
        let salt = [0x11u8; 16];
        let h = hashed_local_part("alice", &salt);
        use sha2::{Digest, Sha256};
        let mut hasher = Sha256::new();
        hasher.update(b"alice");
        hasher.update([0x00]);
        hasher.update(salt);
        let expected: [u8; 32] = hasher.finalize().into();
        assert_eq!(h, expected);
    }

    #[test]
    fn hashed_local_part_is_case_preserving() {
        // Local-parts are case-sensitive per RFC 5321 §2.3.11.
        let salt = [0u8; 16];
        assert_ne!(
            hashed_local_part("Alice", &salt),
            hashed_local_part("alice", &salt),
        );
    }

    #[test]
    fn claim_payload_byte_fields_roundtrip() {
        let payload = ClaimPayload {
            version: 1,
            domain: "q8.fyi".into(),
            hashed_local_part: [0x11u8; 32],
            email: "alice@q8.fyi".into(),
            identity_hash: [0x22u8; 16],
            issued_at: 1_000,
            expires_at: 2_000,
            serial: 5,
            signing_key_id: [0x33u8; 8],
        };
        let bytes = canonical_cbor(&payload).expect("encode");
        let decoded: ClaimPayload = ciborium::de::from_reader(&bytes[..]).expect("decode");
        assert_eq!(decoded, payload);
    }

    #[cfg(feature = "test-support")]
    #[test]
    fn verify_accepts_well_formed_fresh_claim() {
        let mut rng = OsRng;
        let d = TestDomain::new(&mut rng, "q8.fyi");
        let sk = d.mint_signing_key(&mut rng, NOW - 1000, NOW + 90 * 86_400);
        let claim = ClaimBuilder::new(&d, &sk, NOW).build();

        let binding = claim
            .verify(&d.record(), &RevocationView::empty(), NOW, TOLERANCE)
            .expect("verify");

        assert_eq!(binding.domain, "q8.fyi");
        assert_eq!(binding.email, "alice@q8.fyi");
        assert_eq!(binding.identity_hash.as_slice(), &[0x11; 16]);
        assert_eq!(binding.serial, 1);
        assert_eq!(binding.signing_key_id, sk.cert.signing_key_id);
    }
}
