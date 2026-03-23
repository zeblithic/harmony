//! SD-JWT disclosure verification and claim mapping (RFC 9901 §6.3).
//!
//! This module bridges the SD-JWT world (JSON disclosures with string salts)
//! to Harmony's binary credential layer (`harmony_credential::SaltedClaim`).

use alloc::string::String;
use alloc::vec::Vec;

use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use base64::Engine;
use sha2::{Digest, Sha256};

use crate::error::SdJwtError;
use crate::types::{Disclosure, SdJwt};
use harmony_credential::{Claim, SaltedClaim};
use harmony_crypto::hash::blake3_hash;

// ---------------------------------------------------------------------------
// verify_disclosures
// ---------------------------------------------------------------------------

/// Verify that every disclosure's SHA-256 hash appears in the signed `_sd`
/// digest list (RFC 9901 §6.3).
///
/// If `_sd_alg` is present and not `"sha-256"`, returns
/// [`SdJwtError::UnsupportedAlgorithm`].  If any disclosure's digest is
/// missing from the `_sd` list, returns [`SdJwtError::DisclosureHashMismatch`].
///
/// On success, returns a [`VerifiedDisclosures`] newtype that can only
/// be constructed by this function, ensuring `map_claims` cannot be
/// called on unverified disclosures.
pub fn verify_disclosures(sd_jwt: &SdJwt) -> Result<VerifiedDisclosures<'_>, SdJwtError> {
    // Check _sd_alg — only sha-256 (or absent, which defaults to sha-256) is
    // supported per RFC 9901 §5.1.2.
    if let Some(ref alg) = sd_jwt.payload.sd_alg {
        if alg != "sha-256" {
            return Err(SdJwtError::UnsupportedAlgorithm(alg.clone()));
        }
    }

    // Empty disclosures is valid (no selective claims).
    if sd_jwt.disclosures.is_empty() {
        return Ok(VerifiedDisclosures(Vec::new()));
    }

    // Build a BTreeSet for O((k + m) log m) lookup instead of O(k × m) linear scan.
    let sd_set: alloc::collections::BTreeSet<&String> =
        sd_jwt.payload.sd.iter().collect();
    let mut seen = alloc::collections::BTreeSet::new();

    for disclosure in &sd_jwt.disclosures {
        // Reject duplicate disclosures (holder must present each at most once).
        if !seen.insert(&disclosure.raw) {
            return Err(SdJwtError::DisclosureHashMismatch);
        }

        let digest = disclosure_digest(&disclosure.raw);
        if !sd_set.contains(&digest) {
            return Err(SdJwtError::DisclosureHashMismatch);
        }
    }

    Ok(VerifiedDisclosures(sd_jwt.disclosures.iter().collect()))
}

/// Disclosures that have been verified against the signed `_sd` list.
///
/// Can only be constructed by [`verify_disclosures`], ensuring that
/// `map_claims` cannot be called on unverified disclosures.
#[derive(Debug)]
pub struct VerifiedDisclosures<'a>(Vec<&'a Disclosure>);

impl<'a> VerifiedDisclosures<'a> {
    /// Access the verified disclosures.
    pub fn as_slice(&self) -> &[&'a Disclosure] {
        &self.0
    }

    /// Number of verified disclosures.
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Whether there are no verified disclosures.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

/// Compute `base64url_no_pad(SHA-256(ASCII(raw)))` for a disclosure.
fn disclosure_digest(raw: &str) -> String {
    let hash = Sha256::digest(raw.as_bytes());
    URL_SAFE_NO_PAD.encode(hash)
}

// ---------------------------------------------------------------------------
// map_claims
// ---------------------------------------------------------------------------

/// Map verified disclosures to Harmony `SaltedClaim` values.
///
/// Each disclosure is converted using:
/// - **type_id**: looked up from a static vocabulary, or derived via
///   `BLAKE3(name)[0..2] | 0x8000` for unknown claims. Array elements
///   (no name) get `0x0000`.
/// - **salt**: base64url-decoded from the disclosure salt string, truncated
///   to 16 bytes. Returns `Err` if decoded salt is shorter than 16 bytes
///   or not valid base64url.
/// - **value**: the `claim_value` bytes from the disclosure.
pub fn map_claims(disclosures: &VerifiedDisclosures<'_>) -> Result<Vec<SaltedClaim>, SdJwtError> {
    disclosures
        .as_slice()
        .iter()
        .map(|d| {
            let type_id = match &d.claim_name {
                Some(name) => vocabulary_type_id(name),
                None => 0x0000, // array element
            };
            Ok(SaltedClaim {
                claim: Claim {
                    type_id,
                    value: d.claim_value.clone(),
                },
                salt: decode_salt(&d.salt)?,
            })
        })
        .collect()
}

/// Look up a claim name in the static vocabulary.
///
/// Returns a well-known `type_id` for recognised SD-JWT claim names,
/// or a hash-derived ID with the high bit set for unknown names.
///
/// **Collision risk:** The hash-derived range uses 15 bits (32 768 values).
/// Collisions become probable around ~181 unique unknown claim names
/// (birthday paradox). Callers should not rely on hash-derived type_ids
/// for unique identification — use the original `claim_name` string
/// from the `Disclosure` when uniqueness matters.
fn vocabulary_type_id(name: &str) -> u16 {
    match name {
        "given_name" => 0x0100,
        "family_name" => 0x0101,
        "birthdate" => 0x0102,
        "age_over_18" => 0x0103,
        "nationality" => 0x0104,
        "email" => 0x0110,
        "phone_number" => 0x0111,
        "address" => 0x0112,
        _ => {
            let hash = blake3_hash(name.as_bytes());
            let raw = u16::from_be_bytes([hash[0], hash[1]]);
            raw | 0x8000
        }
    }
}

/// Decode the salt string from base64url into a fixed `[u8; 16]` buffer.
///
/// Returns `Err(Base64Error)` if the salt is not valid base64url.
/// Returns `Err(SaltTooShort)` if the decoded salt is shorter than
/// 16 bytes (RFC 9901 §5.2.1 requires at least 128 bits of entropy).
/// Truncates to 16 bytes if longer.
fn decode_salt(salt_str: &str) -> Result<[u8; 16], SdJwtError> {
    let bytes = URL_SAFE_NO_PAD
        .decode(salt_str)
        .map_err(|_| SdJwtError::Base64Error)?;

    if bytes.len() < 16 {
        return Err(SdJwtError::SaltTooShort);
    }

    let mut buf = [0u8; 16];
    buf.copy_from_slice(&bytes[..16]);
    Ok(buf)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{JwsHeader, JwtPayload};

    // -- test helpers -------------------------------------------------------

    /// Base64url-encode `json_str`, compute its SHA-256 digest, and return
    /// `(raw_base64url, digest_base64url)`.
    fn make_disclosure_with_digest(json_str: &str) -> (String, String) {
        let raw = URL_SAFE_NO_PAD.encode(json_str.as_bytes());
        let digest = disclosure_digest(&raw);
        (raw, digest)
    }

    /// Build a `Disclosure` from raw parts.
    fn disclosure_from_raw(
        raw: &str,
        salt: &str,
        name: Option<&str>,
        value_json: &str,
    ) -> Disclosure {
        Disclosure {
            raw: raw.to_string(),
            salt: salt.to_string(),
            claim_name: name.map(|n| n.to_string()),
            claim_value: value_json.as_bytes().to_vec(),
            value: serde_json::from_str(value_json).unwrap(),
        }
    }

    /// Construct a minimal `SdJwt` for testing disclosure verification.
    fn make_test_sdjwt(sd_digests: Vec<String>, disclosures: Vec<Disclosure>) -> SdJwt {
        SdJwt {
            header: JwsHeader {
                alg: "EdDSA".to_string(),
                typ: Some("sd+jwt".to_string()),
                kid: None,
            },
            payload: JwtPayload {
                iss: None,
                sub: None,
                iat: None,
                exp: None,
                nbf: None,
                sd: sd_digests,
                sd_alg: None,
                extra: Vec::new(),
            },
            signature: vec![0u8; 64],
            signing_input: String::new(),
            disclosures,
            key_binding_jwt: None,
        }
    }

    // -- verify_disclosures tests -------------------------------------------

    #[test]
    fn verify_matching_disclosures() {
        let (raw1, digest1) =
            make_disclosure_with_digest(r#"["salt1","given_name","Alice"]"#);
        let (raw2, digest2) =
            make_disclosure_with_digest(r#"["salt2","family_name","Smith"]"#);

        let d1 = disclosure_from_raw(&raw1, "salt1", Some("given_name"), r#""Alice""#);
        let d2 = disclosure_from_raw(&raw2, "salt2", Some("family_name"), r#""Smith""#);

        let sd_jwt = make_test_sdjwt(vec![digest1, digest2], vec![d1, d2]);
        let result = verify_disclosures(&sd_jwt);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 2);
    }

    #[test]
    fn verify_rejects_unmatched_disclosure() {
        let (raw1, digest1) =
            make_disclosure_with_digest(r#"["salt1","given_name","Alice"]"#);
        // Forge a disclosure whose raw value doesn't match any digest.
        let forged = disclosure_from_raw("forged_raw", "salt1", Some("given_name"), r#""Alice""#);

        let sd_jwt = make_test_sdjwt(vec![digest1], vec![
            disclosure_from_raw(&raw1, "salt1", Some("given_name"), r#""Alice""#),
            forged,
        ]);
        let result = verify_disclosures(&sd_jwt);
        assert!(matches!(result, Err(SdJwtError::DisclosureHashMismatch)));
    }

    #[test]
    fn verify_rejects_duplicate_disclosure() {
        let (raw, digest) = make_disclosure_with_digest(r#"["salt","given_name","Alice"]"#);
        let disc = disclosure_from_raw(&raw, "salt", Some("given_name"), "\"Alice\"");

        // Same disclosure presented twice
        let sd_jwt = make_test_sdjwt(vec![digest], vec![disc.clone(), disc]);
        assert!(matches!(
            verify_disclosures(&sd_jwt),
            Err(SdJwtError::DisclosureHashMismatch)
        ));
    }

    #[test]
    fn verify_empty_disclosures_ok() {
        let sd_jwt = make_test_sdjwt(vec![], vec![]);
        let result = verify_disclosures(&sd_jwt);
        assert!(result.is_ok());
        assert!(result.unwrap().is_empty());
    }

    #[test]
    fn verify_unsupported_sd_alg() {
        let mut sd_jwt = make_test_sdjwt(vec![], vec![]);
        sd_jwt.payload.sd_alg = Some("sha-384".to_string());
        let result = verify_disclosures(&sd_jwt);
        assert!(matches!(result, Err(SdJwtError::UnsupportedAlgorithm(_))));
    }

    #[test]
    fn verify_default_sha256_when_sd_alg_none() {
        // sd_alg = None should default to sha-256 and succeed.
        let (raw, digest) =
            make_disclosure_with_digest(r#"["s","name","val"]"#);
        let d = disclosure_from_raw(&raw, "s", Some("name"), r#""val""#);
        let sd_jwt = make_test_sdjwt(vec![digest], vec![d]);
        assert!(verify_disclosures(&sd_jwt).is_ok());
    }

    // -- map_claims tests ---------------------------------------------------

    #[test]
    fn map_known_vocabulary_claims() {
        let d = disclosure_from_raw("raw", "MDEyMzQ1Njc4OWFiY2RlZg", Some("given_name"), r#""Alice""#);
        let verified = VerifiedDisclosures(vec![&d]);
        let claims = map_claims(&verified).unwrap();
        assert_eq!(claims.len(), 1);
        assert_eq!(claims[0].claim.type_id, 0x0100);
        assert_eq!(claims[0].claim.value, br#""Alice""#);
    }

    #[test]
    fn map_unknown_claim_gets_hash_derived_id() {
        let d = disclosure_from_raw("raw", "MDEyMzQ1Njc4OWFiY2RlZg", Some("custom_field"), r#""value""#);
        let verified = VerifiedDisclosures(vec![&d]);
        let claims = map_claims(&verified).unwrap();
        assert_eq!(claims.len(), 1);
        // High bit must be set for unknown claims.
        assert_ne!(claims[0].claim.type_id & 0x8000, 0);
    }

    #[test]
    fn map_array_element_gets_zero_type_id() {
        // Array element disclosures have no claim_name.
        let d = disclosure_from_raw("raw", "MDEyMzQ1Njc4OWFiY2RlZg", None, r#""item""#);
        let verified = VerifiedDisclosures(vec![&d]);
        let claims = map_claims(&verified).unwrap();
        assert_eq!(claims[0].claim.type_id, 0x0000);
    }

    #[test]
    fn map_salt_base64_decoded() {
        // "MDEyMzQ1Njc4OWFiY2RlZg" is base64url for "0123456789abcdef" (16 bytes)
        let d = disclosure_from_raw("raw", "MDEyMzQ1Njc4OWFiY2RlZg", Some("email"), r#""a@b""#);
        let verified = VerifiedDisclosures(vec![&d]);
        let claims = map_claims(&verified).unwrap();
        assert_eq!(&claims[0].salt, b"0123456789abcdef");
    }

    #[test]
    fn map_salt_truncates_long_salt() {
        // 24 raw bytes → base64url encodes to 32 chars. After decode we get
        // 24 bytes back, but only first 16 should be kept.
        let long_bytes: Vec<u8> = (0u8..24).collect();
        let salt_str = URL_SAFE_NO_PAD.encode(&long_bytes);
        let d = disclosure_from_raw("raw", &salt_str, Some("email"), r#""x""#);
        let verified = VerifiedDisclosures(vec![&d]);
        let claims = map_claims(&verified).unwrap();
        let expected: [u8; 16] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
        assert_eq!(claims[0].salt, expected);
    }

    #[test]
    fn map_hash_derived_deterministic() {
        // Same unknown name must always produce the same type_id.
        let d1 = disclosure_from_raw("r1", "MDEyMzQ1Njc4OWFiY2RlZg", Some("foo_bar"), r#""a""#);
        let d2 = disclosure_from_raw("r2", "MDEyMzQ1Njc4OWFiY2RlZg", Some("foo_bar"), r#""b""#);
        let verified = VerifiedDisclosures(vec![&d1, &d2]);
        let claims = map_claims(&verified).unwrap();
        assert_eq!(claims[0].claim.type_id, claims[1].claim.type_id);
        assert_ne!(claims[0].claim.type_id & 0x8000, 0);
    }

    #[test]
    fn map_salt_short_rejects() {
        // 8 bytes → too short (minimum 16)
        let short = URL_SAFE_NO_PAD.encode(&[0u8; 8]);
        let d = disclosure_from_raw("raw", &short, Some("email"), r#""x""#);
        let verified = VerifiedDisclosures(vec![&d]);
        assert!(matches!(
            map_claims(&verified),
            Err(SdJwtError::SaltTooShort)
        ));
    }

    #[test]
    fn map_salt_invalid_base64_rejects() {
        let d = disclosure_from_raw("raw", "not!!base64url!!!", Some("email"), r#""x""#);
        let verified = VerifiedDisclosures(vec![&d]);
        assert!(matches!(
            map_claims(&verified),
            Err(SdJwtError::Base64Error)
        ));
    }
}
