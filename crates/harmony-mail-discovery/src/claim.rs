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

/// `serde` helper for `[u8; 64]`: the workspace `serde_core` fork only
/// derives array impls up to `[T; 32]`, so we use a slice/Vec roundtrip
/// (the same pattern used in `harmony-roxy`'s `serde_sig` module).
mod serde_bytes64 {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    pub fn serialize<S: Serializer>(bytes: &[u8; 64], serializer: S) -> Result<S::Ok, S::Error> {
        bytes.as_slice().serialize(serializer)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(deserializer: D) -> Result<[u8; 64], D::Error> {
        let v: Vec<u8> = Deserialize::deserialize(deserializer)?;
        v.try_into()
            .map_err(|_| serde::de::Error::custom("expected exactly 64 bytes"))
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum MasterPubkey {
    Ed25519([u8; 32]),
    // MlDsa65(Box<[u8; 1952]>) — reserved; see spec §2.1, §4.1.
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SigningPubkey {
    Ed25519([u8; 32]),
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Signature {
    Ed25519(#[serde(with = "serde_bytes64")] [u8; 64]),
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
    pub domain_salt: [u8; 16],
    pub alg: SignatureAlg,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SigningKeyCert {
    pub version: u8,
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
    pub hashed_local_part: [u8; 32],
    pub email: String,
    pub identity_hash: [u8; 16],
    pub issued_at: u64,
    pub expires_at: u64,
    pub serial: u64,
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

#[cfg(test)]
mod tests {
    use super::*;

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
}
