//! Key Binding JWT (KB-JWT) verification per RFC 9901 section 11.6.
//!
//! A KB-JWT proves that the presenter holds the private key bound to the
//! SD-JWT credential. The verifier checks the KB-JWT signature, nonce,
//! audience, freshness (iat), and sd_hash binding.

use alloc::format;
use alloc::string::String;

use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use base64::Engine;
use harmony_identity::CryptoSuite;
use sha2::{Digest, Sha256};

use crate::error::SdJwtError;
use crate::types::SdJwt;

/// Verify a Key Binding JWT attached to an SD-JWT presentation.
///
/// # Arguments
///
/// * `sd_jwt` - The parsed SD-JWT (must have `key_binding_jwt` set)
/// * `holder_key` - The holder's public key bytes
/// * `holder_suite` - The cryptographic suite for the holder's key
/// * `expected_nonce` - The nonce the verifier expects in the KB-JWT
/// * `expected_aud` - The audience the verifier expects in the KB-JWT
/// * `now` - Current UNIX timestamp (seconds)
///
/// # Verification steps (RFC 9901 section 11.6)
///
/// 1. KB-JWT is present
/// 2. `typ` header is `kb+jwt` (or `application/kb+jwt`)
/// 3. `nonce` matches `expected_nonce`
/// 4. `aud` matches `expected_aud`
/// 5. `iat` is not in the future (60s skew) and not too old (300s max age)
/// 6. `_sd_alg` is `sha-256` or absent (only SHA-256 supported)
/// 7. `sd_hash` matches SHA-256 of the SD-JWT without the KB-JWT
/// 7. Signature over the KB-JWT signing input verifies against `holder_key`
pub fn verify_key_binding(
    sd_jwt: &SdJwt,
    holder_key: &[u8],
    holder_suite: CryptoSuite,
    expected_nonce: &str,
    expected_aud: &str,
    now: u64,
) -> Result<(), SdJwtError> {
    // 1. KB-JWT must be present.
    let kb_jwt = sd_jwt
        .key_binding_jwt
        .as_deref()
        .ok_or_else(|| SdJwtError::KeyBindingInvalid(String::from("missing key binding JWT")))?;

    // 2. Split KB-JWT into header.payload.signature (3 dot-separated parts).
    let parts: alloc::vec::Vec<&str> = kb_jwt.splitn(4, '.').collect();
    if parts.len() != 3 {
        return Err(SdJwtError::KeyBindingInvalid(String::from(
            "KB-JWT must have exactly 3 dot-separated parts",
        )));
    }
    let kb_header_b64 = parts[0];
    let kb_payload_b64 = parts[1];
    let kb_sig_b64 = parts[2];

    // 3. Decode and parse header JSON.
    let header_bytes = URL_SAFE_NO_PAD
        .decode(kb_header_b64)
        .map_err(|_| SdJwtError::Base64Error)?;
    let header_json: serde_json::Value =
        serde_json::from_slice(&header_bytes).map_err(|e| SdJwtError::JsonError(e.to_string()))?;

    // 4. Verify KB-JWT signature BEFORE any content checks.
    //    This prevents error-message oracles and avoids wasted
    //    computation on unauthenticated data.
    let kb_signing_input = format!("{}.{}", kb_header_b64, kb_payload_b64);
    let sig_bytes = URL_SAFE_NO_PAD
        .decode(kb_sig_b64)
        .map_err(|_| SdJwtError::Base64Error)?;
    harmony_identity::verify_signature(
        holder_suite,
        holder_key,
        kb_signing_input.as_bytes(),
        &sig_bytes,
    )
    .map_err(SdJwtError::SignatureInvalid)?;

    // --- Signature is valid. Now check authenticated header + claims. ---

    // 5. Verify typ is "kb+jwt" (case-insensitive, also accept "application/kb+jwt").
    let typ = header_json
        .get("typ")
        .and_then(|v| v.as_str())
        .ok_or_else(|| {
            SdJwtError::KeyBindingInvalid(String::from("KB-JWT typ header missing"))
        })?;
    if !typ.eq_ignore_ascii_case("kb+jwt") && !typ.eq_ignore_ascii_case("application/kb+jwt") {
        return Err(SdJwtError::KeyBindingInvalid(format!(
            "KB-JWT typ must be \"kb+jwt\", got \"{typ}\""
        )));
    }

    // 6. Decode and parse payload JSON.
    let payload_bytes = URL_SAFE_NO_PAD
        .decode(kb_payload_b64)
        .map_err(|_| SdJwtError::Base64Error)?;
    let payload_json: serde_json::Value = serde_json::from_slice(&payload_bytes)
        .map_err(|e| SdJwtError::JsonError(e.to_string()))?;

    // 7. Verify nonce.
    let nonce = payload_json
        .get("nonce")
        .and_then(|v| v.as_str())
        .ok_or_else(|| {
            SdJwtError::KeyBindingInvalid(String::from("KB-JWT nonce claim missing"))
        })?;
    if nonce != expected_nonce {
        return Err(SdJwtError::KeyBindingInvalid(format!(
            "KB-JWT nonce mismatch: expected \"{expected_nonce}\", got \"{nonce}\""
        )));
    }

    // 8. Verify aud (RFC 7519 §4.1.3: string or array of strings).
    let aud_val = payload_json
        .get("aud")
        .ok_or_else(|| SdJwtError::KeyBindingInvalid(String::from("KB-JWT aud claim missing")))?;
    let aud_match = if let Some(s) = aud_val.as_str() {
        s == expected_aud
    } else if let Some(arr) = aud_val.as_array() {
        arr.iter().any(|v| v.as_str() == Some(expected_aud))
    } else {
        return Err(SdJwtError::KeyBindingInvalid(String::from(
            "KB-JWT aud claim is not a string or array",
        )));
    };
    if !aud_match {
        return Err(SdJwtError::KeyBindingInvalid(format!(
            "KB-JWT aud does not contain expected audience \"{expected_aud}\""
        )));
    }

    // 9. Verify iat: not in the future AND not too old.
    const MAX_CLOCK_SKEW: u64 = 60;
    const MAX_AGE_SECS: u64 = 300; // 5 minutes
    let iat = payload_json
        .get("iat")
        .ok_or_else(|| SdJwtError::KeyBindingInvalid(String::from("KB-JWT iat claim missing")))?
        .as_u64()
        .ok_or_else(|| SdJwtError::KeyBindingInvalid(String::from(
            "KB-JWT iat claim is not a valid non-negative integer"
        )))?;
    if iat > now.saturating_add(MAX_CLOCK_SKEW) {
        return Err(SdJwtError::KeyBindingInvalid(format!(
            "KB-JWT iat is in the future: iat={iat}, now={now}"
        )));
    }
    if now > iat.saturating_add(MAX_AGE_SECS) {
        return Err(SdJwtError::KeyBindingInvalid(format!(
            "KB-JWT iat is too old: iat={iat}, now={now}, max_age={MAX_AGE_SECS}s"
        )));
    }

    // 10. Check _sd_alg — only sha-256 supported for sd_hash computation.
    if let Some(ref alg) = sd_jwt.payload.sd_alg {
        if alg != "sha-256" {
            return Err(SdJwtError::UnsupportedAlgorithm(alg.clone()));
        }
    }

    // 11. Reconstruct SD-JWT without KB-JWT and verify sd_hash.
    let sig_b64 = URL_SAFE_NO_PAD.encode(&sd_jwt.signature);
    let mut sd_jwt_without_kb = format!("{}.{}", sd_jwt.signing_input, sig_b64);
    for disc in &sd_jwt.disclosures {
        sd_jwt_without_kb.push('~');
        sd_jwt_without_kb.push_str(&disc.raw);
    }
    sd_jwt_without_kb.push('~');

    let computed_hash = Sha256::digest(sd_jwt_without_kb.as_bytes());
    let computed_sd_hash = URL_SAFE_NO_PAD.encode(computed_hash);

    let expected_sd_hash = payload_json
        .get("sd_hash")
        .and_then(|v| v.as_str())
        .ok_or_else(|| {
            SdJwtError::KeyBindingInvalid(String::from("KB-JWT sd_hash claim missing"))
        })?;
    if computed_sd_hash != expected_sd_hash {
        return Err(SdJwtError::KeyBindingInvalid(String::from(
            "KB-JWT sd_hash does not match the SD-JWT presentation",
        )));
    }

    Ok(())
}

#[cfg(all(test, feature = "std"))]
mod tests {
    use super::*;
    use crate::types::{Disclosure, JwsHeader, JwtPayload};
    use rand::rngs::OsRng;

    const B64: base64::engine::GeneralPurpose = URL_SAFE_NO_PAD;

    /// Build a complete SdJwt with a valid Key Binding JWT for testing.
    ///
    /// Returns `(sd_jwt, holder_pub_key_bytes)`.
    fn make_sd_jwt_with_kb(
        nonce: &str,
        aud: &str,
        iat: u64,
    ) -> (SdJwt, alloc::vec::Vec<u8>) {
        let issuer = harmony_identity::PrivateIdentity::generate(&mut OsRng);
        let holder = harmony_identity::PrivateIdentity::generate(&mut OsRng);
        let holder_pub = holder.public_identity().verifying_key.to_bytes().to_vec();

        // Build issuer JWS
        let header_json = r#"{"alg":"EdDSA","typ":"sd+jwt"}"#;
        let payload_json = r#"{"iss":"https://issuer.example","sub":"holder42"}"#;
        let header_b64 = B64.encode(header_json.as_bytes());
        let payload_b64 = B64.encode(payload_json.as_bytes());
        let signing_input = format!("{}.{}", header_b64, payload_b64);
        let signature = issuer.sign(signing_input.as_bytes());
        let sig_b64 = B64.encode(signature);

        // Build a disclosure
        let disc_json = r#"["salt123","given_name","Alice"]"#;
        let disc_raw = B64.encode(disc_json.as_bytes());
        let disclosure = Disclosure {
            raw: disc_raw.clone(),
            salt: String::from("salt123"),
            claim_name: Some(String::from("given_name")),
            claim_value: b"\"Alice\"".to_vec(),
            value: serde_json::json!("Alice"),
        };

        // Reconstruct SD-JWT without KB for sd_hash computation
        let sd_jwt_without_kb = format!("{}.{}~{}~", signing_input, sig_b64, disc_raw);
        let sd_hash = B64.encode(Sha256::digest(sd_jwt_without_kb.as_bytes()));

        // Build KB-JWT
        let kb_header = r#"{"alg":"EdDSA","typ":"kb+jwt"}"#;
        let kb_payload = format!(
            r#"{{"nonce":"{}","aud":"{}","iat":{},"sd_hash":"{}"}}"#,
            nonce, aud, iat, sd_hash
        );
        let kb_header_b64 = B64.encode(kb_header.as_bytes());
        let kb_payload_b64 = B64.encode(kb_payload.as_bytes());
        let kb_signing_input = format!("{}.{}", kb_header_b64, kb_payload_b64);
        let kb_sig = holder.sign(kb_signing_input.as_bytes());
        let kb_sig_b64 = B64.encode(kb_sig);
        let kb_jwt = format!("{}.{}", kb_signing_input, kb_sig_b64);

        // Parse the header/payload for the SdJwt struct
        let header = JwsHeader {
            alg: String::from("EdDSA"),
            typ: Some(String::from("sd+jwt")),
            kid: None,
        };
        let payload = JwtPayload {
            iss: Some(String::from("https://issuer.example")),
            sub: Some(String::from("holder42")),
            iat: None,
            exp: None,
            nbf: None,
            sd: alloc::vec::Vec::new(),
            sd_alg: None,
            extra: alloc::vec::Vec::new(),
        };

        let sd_jwt = SdJwt {
            header,
            payload,
            signature: signature.to_vec(),
            signing_input,
            disclosures: alloc::vec![disclosure],
            key_binding_jwt: Some(kb_jwt),
        };

        (sd_jwt, holder_pub)
    }

    #[test]
    fn valid_key_binding() {
        let now = 1_700_000_000u64;
        let (sd_jwt, holder_pub) = make_sd_jwt_with_kb("test-nonce", "https://verifier.example", now);
        let result = verify_key_binding(
            &sd_jwt,
            &holder_pub,
            CryptoSuite::Ed25519,
            "test-nonce",
            "https://verifier.example",
            now,
        );
        assert!(result.is_ok(), "expected Ok, got {:?}", result);
    }

    #[test]
    fn missing_kb_jwt() {
        let now = 1_700_000_000u64;
        let (mut sd_jwt, holder_pub) =
            make_sd_jwt_with_kb("nonce", "https://verifier.example", now);
        sd_jwt.key_binding_jwt = None;
        let result = verify_key_binding(
            &sd_jwt,
            &holder_pub,
            CryptoSuite::Ed25519,
            "nonce",
            "https://verifier.example",
            now,
        );
        assert!(matches!(result, Err(SdJwtError::KeyBindingInvalid(_))));
    }

    #[test]
    fn wrong_nonce() {
        let now = 1_700_000_000u64;
        let (sd_jwt, holder_pub) =
            make_sd_jwt_with_kb("correct-nonce", "https://verifier.example", now);
        let result = verify_key_binding(
            &sd_jwt,
            &holder_pub,
            CryptoSuite::Ed25519,
            "wrong-nonce",
            "https://verifier.example",
            now,
        );
        match result {
            Err(SdJwtError::KeyBindingInvalid(msg)) => {
                assert!(msg.contains("nonce"), "error should mention nonce: {msg}");
            }
            other => panic!("expected KeyBindingInvalid with nonce, got {:?}", other),
        }
    }

    #[test]
    fn wrong_aud() {
        let now = 1_700_000_000u64;
        let (sd_jwt, holder_pub) =
            make_sd_jwt_with_kb("nonce", "https://correct.example", now);
        let result = verify_key_binding(
            &sd_jwt,
            &holder_pub,
            CryptoSuite::Ed25519,
            "nonce",
            "https://wrong.example",
            now,
        );
        match result {
            Err(SdJwtError::KeyBindingInvalid(msg)) => {
                assert!(msg.contains("aud"), "error should mention aud: {msg}");
            }
            other => panic!("expected KeyBindingInvalid with aud, got {:?}", other),
        }
    }

    #[test]
    fn future_iat() {
        let now = 1_700_000_000u64;
        // iat is 120 seconds in the future (beyond 60-second tolerance)
        let future_iat = now + 120;
        let (sd_jwt, holder_pub) =
            make_sd_jwt_with_kb("nonce", "https://verifier.example", future_iat);
        let result = verify_key_binding(
            &sd_jwt,
            &holder_pub,
            CryptoSuite::Ed25519,
            "nonce",
            "https://verifier.example",
            now,
        );
        match result {
            Err(SdJwtError::KeyBindingInvalid(msg)) => {
                assert!(
                    msg.contains("future"),
                    "error should mention future: {msg}"
                );
            }
            other => panic!("expected KeyBindingInvalid with future, got {:?}", other),
        }
    }

    #[test]
    fn wrong_sd_hash() {
        let now = 1_700_000_000u64;
        let (mut sd_jwt, holder_pub) =
            make_sd_jwt_with_kb("nonce", "https://verifier.example", now);
        // Tamper with the signature to change the sd_hash computation
        if let Some(byte) = sd_jwt.signature.first_mut() {
            *byte ^= 0xff;
        }
        let result = verify_key_binding(
            &sd_jwt,
            &holder_pub,
            CryptoSuite::Ed25519,
            "nonce",
            "https://verifier.example",
            now,
        );
        match result {
            Err(SdJwtError::KeyBindingInvalid(msg)) => {
                assert!(
                    msg.contains("sd_hash"),
                    "error should mention sd_hash: {msg}"
                );
            }
            other => panic!("expected KeyBindingInvalid with sd_hash, got {:?}", other),
        }
    }

    #[test]
    fn wrong_holder_key() {
        let now = 1_700_000_000u64;
        let (sd_jwt, _holder_pub) =
            make_sd_jwt_with_kb("nonce", "https://verifier.example", now);
        // Use a different key
        let wrong_key = harmony_identity::PrivateIdentity::generate(&mut OsRng);
        let wrong_pub = wrong_key.public_identity().verifying_key.to_bytes();
        let result = verify_key_binding(
            &sd_jwt,
            &wrong_pub,
            CryptoSuite::Ed25519,
            "nonce",
            "https://verifier.example",
            now,
        );
        assert!(
            matches!(result, Err(SdJwtError::SignatureInvalid(_))),
            "expected SignatureInvalid, got {:?}",
            result
        );
    }

    #[test]
    fn wrong_typ() {
        let now = 1_700_000_000u64;
        let issuer = harmony_identity::PrivateIdentity::generate(&mut OsRng);
        let holder = harmony_identity::PrivateIdentity::generate(&mut OsRng);
        let holder_pub = holder.public_identity();

        // Build issuer JWS
        let header_b64 = B64.encode(r#"{"alg":"EdDSA","typ":"sd+jwt"}"#.as_bytes());
        let payload_b64 = B64.encode(r#"{"iss":"i","sub":"h"}"#.as_bytes());
        let signing_input = format!("{header_b64}.{payload_b64}");
        let issuer_sig = issuer.sign(signing_input.as_bytes());
        let sig_b64 = B64.encode(&issuer_sig);
        let sd_jwt_without_kb = format!("{signing_input}.{sig_b64}~");
        let sd_hash = B64.encode(Sha256::digest(sd_jwt_without_kb.as_bytes()));

        // Build KB-JWT with wrong typ but VALID signature
        let kb_header_b64 = B64.encode(r#"{"alg":"EdDSA","typ":"jwt"}"#.as_bytes());
        let kb_payload = serde_json::json!({
            "nonce": "n", "aud": "a", "iat": now, "sd_hash": sd_hash
        });
        let kb_payload_b64 = B64.encode(serde_json::to_vec(&kb_payload).unwrap());
        let kb_si = format!("{kb_header_b64}.{kb_payload_b64}");
        let kb_sig = holder.sign(kb_si.as_bytes());
        let kb_jwt = format!("{kb_si}.{}", B64.encode(&kb_sig));

        let sd_jwt = SdJwt {
            header: JwsHeader { alg: "EdDSA".into(), typ: Some("sd+jwt".into()), kid: None },
            payload: JwtPayload {
                iss: Some("i".into()), sub: Some("h".into()),
                iat: None, exp: None, nbf: None, sd: vec![], sd_alg: None,
                #[cfg(feature = "std")] extra: vec![],
            },
            signature: issuer_sig.to_vec(),
            signing_input,
            disclosures: vec![],
            key_binding_jwt: Some(kb_jwt),
        };

        let result = verify_key_binding(
            &sd_jwt,
            &holder_pub.verifying_key.to_bytes(),
            CryptoSuite::Ed25519,
            "n", "a", now,
        );
        assert!(matches!(
            result,
            Err(SdJwtError::KeyBindingInvalid(msg)) if msg.contains("typ")
        ));
    }

    #[test]
    fn aud_as_array_accepted() {
        let issuer = harmony_identity::PrivateIdentity::generate(&mut OsRng);
        let holder = harmony_identity::PrivateIdentity::generate(&mut OsRng);
        let holder_pub = holder.public_identity();
        let now = 1000u64;

        // Build SD-JWT with KB-JWT where aud is an array
        let header_b64 = B64.encode(r#"{"alg":"EdDSA","typ":"sd+jwt"}"#.as_bytes());
        let payload_b64 = B64.encode(r#"{"iss":"issuer","sub":"holder"}"#.as_bytes());
        let signing_input = format!("{header_b64}.{payload_b64}");
        let issuer_sig = issuer.sign(signing_input.as_bytes());
        let sig_b64 = B64.encode(&issuer_sig);
        let sd_jwt_without_kb = format!("{signing_input}.{sig_b64}~");
        let sd_hash = B64.encode(Sha256::digest(sd_jwt_without_kb.as_bytes()));

        // KB-JWT with aud as array
        let kb_header_b64 = B64.encode(r#"{"alg":"EdDSA","typ":"kb+jwt"}"#.as_bytes());
        let kb_payload = serde_json::json!({
            "nonce": "n",
            "aud": ["https://verifier.example"],
            "iat": now,
            "sd_hash": sd_hash
        });
        let kb_payload_b64 = B64.encode(serde_json::to_vec(&kb_payload).unwrap());
        let kb_si = format!("{kb_header_b64}.{kb_payload_b64}");
        let kb_sig = holder.sign(kb_si.as_bytes());
        let kb_jwt = format!("{kb_si}.{}", B64.encode(&kb_sig));

        let sd_jwt = SdJwt {
            header: JwsHeader { alg: "EdDSA".into(), typ: Some("sd+jwt".into()), kid: None },
            payload: JwtPayload {
                iss: Some("issuer".into()), sub: Some("holder".into()),
                iat: None, exp: None, nbf: None, sd: vec![], sd_alg: None,
                #[cfg(feature = "std")] extra: vec![],
            },
            signature: issuer_sig.to_vec(),
            signing_input,
            disclosures: vec![],
            key_binding_jwt: Some(kb_jwt),
        };

        assert!(verify_key_binding(
            &sd_jwt,
            &holder_pub.verifying_key.to_bytes(),
            CryptoSuite::Ed25519,
            "n",
            "https://verifier.example",
            now,
        ).is_ok());
    }

    #[test]
    fn too_old_iat() {
        // iat = 1000, now = 2000 — 1000 seconds old (> 300s max age)
        let (sd_jwt, holder_key) = make_sd_jwt_with_kb("n", "a", 1000);

        let result = verify_key_binding(
            &sd_jwt,
            &holder_key,
            CryptoSuite::Ed25519,
            "n",
            "a",
            2000,
        );
        assert!(matches!(
            result,
            Err(SdJwtError::KeyBindingInvalid(msg)) if msg.contains("too old")
        ));
    }
}
