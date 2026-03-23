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
/// On success, returns references to all verified disclosures.
pub fn verify_disclosures(sd_jwt: &SdJwt) -> Result<Vec<&Disclosure>, SdJwtError> {
    // Check _sd_alg — only sha-256 (or absent, which defaults to sha-256) is
    // supported per RFC 9901 §5.1.2.
    if let Some(ref alg) = sd_jwt.payload.sd_alg {
        if alg != "sha-256" {
            return Err(SdJwtError::UnsupportedAlgorithm(alg.clone()));
        }
    }

    // Empty disclosures is valid (no selective claims).
    if sd_jwt.disclosures.is_empty() {
        return Ok(Vec::new());
    }

    let sd_set = &sd_jwt.payload.sd;

    for disclosure in &sd_jwt.disclosures {
        let digest = disclosure_digest(&disclosure.raw);
        if !sd_set.contains(&digest) {
            return Err(SdJwtError::DisclosureHashMismatch);
        }
    }

    Ok(sd_jwt.disclosures.iter().collect())
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
/// - **salt**: base64url-decoded from the disclosure salt string, zero-padded
///   or truncated to 16 bytes.
/// - **value**: the `claim_value` bytes from the disclosure.
pub fn map_claims(_disclosures: &[&Disclosure]) -> Vec<SaltedClaim> {
    todo!("implement in task 3")
}

/// Look up a claim name in the static vocabulary.
///
/// Returns a well-known `type_id` for recognised SD-JWT claim names,
/// or a hash-derived ID with the high bit set for unknown names.
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
/// If the string is not valid base64url, falls back to raw UTF-8 bytes.
/// The result is zero-padded or truncated to exactly 16 bytes.
fn decode_salt(salt_str: &str) -> [u8; 16] {
    let bytes = URL_SAFE_NO_PAD
        .decode(salt_str)
        .unwrap_or_else(|_| salt_str.as_bytes().to_vec());

    let mut buf = [0u8; 16];
    let len = bytes.len().min(16);
    buf[..len].copy_from_slice(&bytes[..len]);
    buf
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
}
