use crate::error::SdJwtError;

/// Extract the raw base64url header and payload strings from a JWS compact serialization.
///
/// Returns `(signing_input, signature_b64)` where `signing_input` is the raw
/// `header.payload` string suitable for signature verification, and `signature_b64`
/// is the base64url-encoded signature.
///
/// This function does NOT require `serde_json` and works in `no_std` environments.
pub fn signing_input(compact: &str) -> Result<(&str, &str), SdJwtError> {
    if compact.is_empty() {
        return Err(SdJwtError::EmptyInput);
    }

    // The input may be an SD-JWT with disclosures appended after `~`.
    // The JWS is always the first segment before any `~`.
    // split always yields at least one element for non-empty input (checked above)
    let jws = compact.split('~').next().unwrap();

    // A JWS compact serialization has exactly 3 dot-separated parts.
    let mut dots = 0usize;
    let mut first_dot = None;
    let mut second_dot = None;
    for (i, b) in jws.bytes().enumerate() {
        if b == b'.' {
            dots += 1;
            match dots {
                1 => first_dot = Some(i),
                2 => second_dot = Some(i),
                _ => return Err(SdJwtError::MalformedCompact),
            }
        }
    }

    let first_dot = first_dot.ok_or(SdJwtError::MalformedCompact)?;
    let second_dot = second_dot.ok_or(SdJwtError::MalformedCompact)?;

    let signing_input_str = &jws[..second_dot];
    let signature_b64 = &jws[second_dot + 1..];

    // Validate that header and payload parts are non-empty.
    if first_dot == 0 || second_dot == first_dot + 1 {
        return Err(SdJwtError::MalformedCompact);
    }

    // Signature may be empty (unsecured JWS), but we require it for SD-JWT.
    if signature_b64.is_empty() {
        return Err(SdJwtError::MalformedCompact);
    }

    Ok((signing_input_str, signature_b64))
}

/// Parse an SD-JWT compact serialization into a fully decoded [`SdJwt`].
///
/// The input format is: `header.payload.signature~disclosure1~disclosure2~...~`
///
/// Per RFC 9901, the compact serialization consists of a JWS followed by zero or
/// more tilde-separated base64url-encoded disclosures, optionally terminated by a
/// trailing tilde.
///
/// # Parse algorithm
///
/// 1. Split entire input on `~` into segments
/// 2. Segment 0 = JWS, split on `.` into 3 parts (header, payload, signature)
/// 3. Store raw `header.payload` as `signing_input` for lossless verification
/// 4. Base64url decode each part, deserialize header/payload as JSON
/// 5. Extract `_sd`, `_sd_alg` from payload into dedicated fields
/// 6. Segments 1..N (non-empty) = disclosures, decoded as `[salt, name, value]`
///    (object property) or `[salt, value]` (array element) arrays
///
/// This function requires the `std` feature because it uses `serde_json` for
/// JSON deserialization.
///
/// # Security
///
/// This function verifies structure and decodes components, but does **not**
/// verify that each disclosure's hash appears in `payload.sd`.
/// Callers MUST hash each `Disclosure::raw` value (using `payload.sd_alg`,
/// defaulting to `sha-256`) and confirm the digest is present in
/// `payload.sd` before treating any disclosed claim as
/// issuer-attested. Without this check, an attacker could append forged
/// disclosures that were never signed by the issuer.
#[cfg(feature = "std")]
pub fn parse(compact: &str) -> Result<crate::types::SdJwt, SdJwtError> {
    use alloc::string::{String, ToString};
    use alloc::vec::Vec;
    use base64::engine::general_purpose::URL_SAFE_NO_PAD;
    use base64::Engine;

    if compact.is_empty() {
        return Err(SdJwtError::EmptyInput);
    }

    // Split on `~` to separate JWS from disclosures.
    let segments: Vec<&str> = compact.split('~').collect();

    // First segment is the JWS compact serialization.
    let jws = segments[0];

    // Split JWS into header, payload, signature.
    let jws_parts: Vec<&str> = jws.split('.').collect();
    if jws_parts.len() != 3 {
        return Err(SdJwtError::MalformedCompact);
    }

    let header_b64 = jws_parts[0];
    let payload_b64 = jws_parts[1];
    let signature_b64 = jws_parts[2];

    if header_b64.is_empty() || payload_b64.is_empty() || signature_b64.is_empty() {
        return Err(SdJwtError::MalformedCompact);
    }

    // Store the raw signing input for verification.
    let raw_signing_input = format!("{}.{}", header_b64, payload_b64);

    // Decode header.
    let header_bytes = URL_SAFE_NO_PAD
        .decode(header_b64)
        .map_err(|_| SdJwtError::Base64Error)?;
    let header_json: serde_json::Value =
        serde_json::from_slice(&header_bytes).map_err(|e| SdJwtError::JsonError(e.to_string()))?;

    // RFC 7515 §4: the JOSE Header MUST be a JSON object.
    if !header_json.is_object() {
        return Err(SdJwtError::MalformedCompact);
    }

    let alg = header_json
        .get("alg")
        .ok_or(SdJwtError::MissingAlgorithm)?
        .as_str()
        .ok_or(SdJwtError::MalformedCompact)?
        .to_string();
    let typ = header_json
        .get("typ")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());
    let kid = header_json
        .get("kid")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());

    let header = crate::types::JwsHeader { alg, typ, kid };

    // Decode payload.
    let payload_bytes = URL_SAFE_NO_PAD
        .decode(payload_b64)
        .map_err(|_| SdJwtError::Base64Error)?;
    let payload_json: serde_json::Value = serde_json::from_slice(&payload_bytes)
        .map_err(|e| SdJwtError::JsonError(e.to_string()))?;

    // RFC 7519 §4: the Claims Set MUST be a JSON object.
    if !payload_json.is_object() {
        return Err(SdJwtError::MalformedCompact);
    }

    let iss = payload_json
        .get("iss")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());
    let sub = payload_json
        .get("sub")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());
    let iat = payload_json.get("iat").and_then(|v| v.as_i64());
    let exp = payload_json.get("exp").and_then(|v| v.as_i64());
    let nbf = payload_json.get("nbf").and_then(|v| v.as_i64());

    // RFC 9901 §5.2: _sd MUST be an array if present.
    let sd: Vec<String> = match payload_json.get("_sd") {
        None => Vec::new(),
        Some(v) => v
            .as_array()
            .ok_or(SdJwtError::MalformedCompact)?
            .iter()
            .map(|v| {
                v.as_str()
                    .map(|s| s.to_string())
                    .ok_or(SdJwtError::MalformedCompact)
            })
            .collect::<Result<Vec<_>, _>>()?,
    };

    // RFC 9901 §5.1: _sd_alg MUST be a string when present.
    let sd_alg: Option<String> = match payload_json.get("_sd_alg") {
        None => None,
        Some(v) => Some(
            v.as_str()
                .ok_or(SdJwtError::MalformedCompact)?
                .to_string(),
        ),
    };

    // Collect extra claims (everything that isn't a known field).
    // payload_json is guaranteed to be an object by the is_object() guard above.
    let known_fields = ["iss", "sub", "iat", "exp", "nbf", "_sd", "_sd_alg"];
    let obj = payload_json.as_object().expect("guarded by is_object() above");
    let extra: Vec<(String, serde_json::Value)> = obj
        .iter()
        .filter(|(k, _)| !known_fields.contains(&k.as_str()))
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();

    let payload = crate::types::JwtPayload {
        iss,
        sub,
        iat,
        exp,
        nbf,
        sd,
        sd_alg,
        extra,
    };

    // Decode signature.
    let signature = URL_SAFE_NO_PAD
        .decode(signature_b64)
        .map_err(|_| SdJwtError::Base64Error)?;

    // Parse disclosures (segments 1..N, skipping empty trailing segments).
    // A KB-JWT (Key Binding JWT) may appear as the last non-empty segment —
    // it's a compact JWS (contains '.') and is NOT a disclosure. We detect
    // it by checking for '.' and capture it separately.
    let mut disclosures = Vec::new();
    let mut key_binding_jwt: Option<String> = None;
    for segment in &segments[1..] {
        if segment.is_empty() {
            continue;
        }

        // KB-JWT detection: disclosures are base64url strings (no dots),
        // while a KB-JWT is a compact JWS (header.payload.signature).
        // RFC 9901 §11.6: KB-JWT must be the final non-empty segment.
        // Reject any segment after the KB-JWT has been seen.
        if key_binding_jwt.is_some() {
            return Err(SdJwtError::MalformedCompact);
        }

        // KB-JWT detection: disclosures are base64url (no dots),
        // KB-JWT is a compact JWS (header.payload.signature).
        if segment.contains('.') {
            key_binding_jwt = Some((*segment).to_string());
            continue;
        }

        let disclosure_bytes = URL_SAFE_NO_PAD
            .decode(segment)
            .map_err(|_| SdJwtError::Base64Error)?;
        let disclosure_json: serde_json::Value = serde_json::from_slice(&disclosure_bytes)
            .map_err(|e| SdJwtError::JsonError(e.to_string()))?;

        let arr = disclosure_json
            .as_array()
            .ok_or(SdJwtError::InvalidDisclosure)?;

        let disclosure = match arr.len() {
            // [salt, value] — array element disclosure
            2 => {
                let salt = arr[0]
                    .as_str()
                    .ok_or(SdJwtError::InvalidDisclosure)?
                    .to_string();
                let value = arr[1].clone();
                let claim_value = serde_json::to_vec(&value)
                    .map_err(|e| SdJwtError::JsonError(e.to_string()))?;

                crate::types::Disclosure {
                    raw: (*segment).to_string(),
                    salt,
                    claim_name: None,
                    claim_value,
                    value,
                }
            }
            // [salt, name, value] — object property disclosure
            3 => {
                let salt = arr[0]
                    .as_str()
                    .ok_or(SdJwtError::InvalidDisclosure)?
                    .to_string();
                let name = arr[1]
                    .as_str()
                    .ok_or(SdJwtError::InvalidDisclosure)?
                    .to_string();
                let value = arr[2].clone();
                let claim_value = serde_json::to_vec(&value)
                    .map_err(|e| SdJwtError::JsonError(e.to_string()))?;

                crate::types::Disclosure {
                    raw: (*segment).to_string(),
                    salt,
                    claim_name: Some(name),
                    claim_value,
                    value,
                }
            }
            _ => return Err(SdJwtError::InvalidDisclosure),
        };

        disclosures.push(disclosure);
    }

    Ok(crate::types::SdJwt {
        header,
        payload,
        signature,
        signing_input: raw_signing_input,
        disclosures,
        key_binding_jwt,
    })
}

#[cfg(all(test, feature = "std"))]
mod tests {
    use super::*;
    use base64::engine::general_purpose::URL_SAFE_NO_PAD;
    use base64::Engine;

    /// Helper: build a minimal JWS compact serialization from JSON header, payload, and fake signature.
    fn build_jws(header: &serde_json::Value, payload: &serde_json::Value, sig: &[u8]) -> String {
        let h = URL_SAFE_NO_PAD.encode(serde_json::to_vec(header).unwrap());
        let p = URL_SAFE_NO_PAD.encode(serde_json::to_vec(payload).unwrap());
        let s = URL_SAFE_NO_PAD.encode(sig);
        format!("{h}.{p}.{s}")
    }

    /// Helper: build a base64url-encoded disclosure from a JSON array.
    fn build_disclosure(arr: &serde_json::Value) -> String {
        URL_SAFE_NO_PAD.encode(serde_json::to_vec(arr).unwrap())
    }

    #[test]
    fn parse_jws_no_disclosures() {
        let header = serde_json::json!({"alg": "EdDSA"});
        let payload = serde_json::json!({"iss": "https://example.com", "sub": "user123"});
        let compact = build_jws(&header, &payload, b"fakesig");

        let sd_jwt = parse(&compact).unwrap();
        assert_eq!(sd_jwt.header.alg, "EdDSA");
        assert_eq!(sd_jwt.payload.iss.as_deref(), Some("https://example.com"));
        assert_eq!(sd_jwt.payload.sub.as_deref(), Some("user123"));
        assert!(sd_jwt.disclosures.is_empty());
        assert_eq!(sd_jwt.signature, b"fakesig");
    }

    #[test]
    fn parse_with_disclosures() {
        let header = serde_json::json!({"alg": "EdDSA", "typ": "sd+jwt"});
        let payload = serde_json::json!({
            "iss": "https://issuer.example",
            "_sd": ["abc123"],
            "_sd_alg": "sha-256"
        });
        let disclosure = build_disclosure(&serde_json::json!(["salt1", "given_name", "John"]));
        let compact = format!("{}~{}~", build_jws(&header, &payload, b"sig"), disclosure);

        let sd_jwt = parse(&compact).unwrap();
        assert_eq!(sd_jwt.header.typ.as_deref(), Some("sd+jwt"));
        assert_eq!(sd_jwt.payload.sd, vec!["abc123"]);
        assert_eq!(sd_jwt.payload.sd_alg.as_deref(), Some("sha-256"));
        assert_eq!(sd_jwt.disclosures.len(), 1);

        let d = &sd_jwt.disclosures[0];
        assert_eq!(d.salt, "salt1");
        assert_eq!(d.claim_name.as_deref(), Some("given_name"));
        assert_eq!(d.value, serde_json::json!("John"));
    }

    #[test]
    fn parse_array_element_disclosure() {
        let header = serde_json::json!({"alg": "EdDSA"});
        let payload = serde_json::json!({"iss": "https://issuer.example"});
        // Array element disclosure: [salt, value] (no claim name)
        let disclosure = build_disclosure(&serde_json::json!(["saltydog", 42]));
        let compact = format!("{}~{}~", build_jws(&header, &payload, b"sig"), disclosure);

        let sd_jwt = parse(&compact).unwrap();
        assert_eq!(sd_jwt.disclosures.len(), 1);

        let d = &sd_jwt.disclosures[0];
        assert_eq!(d.salt, "saltydog");
        assert!(d.claim_name.is_none());
        assert_eq!(d.value, serde_json::json!(42));
    }

    #[test]
    fn header_field_extraction() {
        let header = serde_json::json!({"alg": "ES256", "typ": "sd+jwt", "kid": "key-1"});
        let payload = serde_json::json!({"iss": "test"});
        let compact = build_jws(&header, &payload, b"sig");

        let sd_jwt = parse(&compact).unwrap();
        assert_eq!(sd_jwt.header.alg, "ES256");
        assert_eq!(sd_jwt.header.typ.as_deref(), Some("sd+jwt"));
        assert_eq!(sd_jwt.header.kid.as_deref(), Some("key-1"));
    }

    #[test]
    fn payload_field_extraction() {
        let header = serde_json::json!({"alg": "EdDSA"});
        let payload = serde_json::json!({
            "iss": "https://issuer.example",
            "sub": "user42",
            "iat": 1700000000i64,
            "exp": 1700003600i64,
            "nbf": 1699999900i64,
            "_sd": ["digest1", "digest2"],
            "_sd_alg": "sha-256",
            "custom_claim": "custom_value"
        });
        let compact = build_jws(&header, &payload, b"sig");

        let sd_jwt = parse(&compact).unwrap();
        assert_eq!(sd_jwt.payload.iss.as_deref(), Some("https://issuer.example"));
        assert_eq!(sd_jwt.payload.sub.as_deref(), Some("user42"));
        assert_eq!(sd_jwt.payload.iat, Some(1700000000));
        assert_eq!(sd_jwt.payload.exp, Some(1700003600));
        assert_eq!(sd_jwt.payload.nbf, Some(1699999900));
        assert_eq!(sd_jwt.payload.sd, vec!["digest1", "digest2"]);
        assert_eq!(sd_jwt.payload.sd_alg.as_deref(), Some("sha-256"));

        // Extra claims
        assert!(sd_jwt
            .payload
            .extra
            .iter()
            .any(|(k, v)| k == "custom_claim" && v == "custom_value"));
    }

    #[test]
    fn preserves_signing_input() {
        let header = serde_json::json!({"alg": "EdDSA"});
        let payload = serde_json::json!({"iss": "test"});
        let jws = build_jws(&header, &payload, b"sig");

        // The signing input should be the first two dot-separated parts.
        let expected_signing_input: String = jws.rsplitn(2, '.').last().unwrap().to_string();

        let sd_jwt = parse(&jws).unwrap();
        assert_eq!(sd_jwt.signing_input, expected_signing_input);
    }

    #[test]
    fn error_empty_input() {
        let result = parse("");
        assert!(matches!(result, Err(SdJwtError::EmptyInput)));
    }

    #[test]
    fn error_missing_signature() {
        // Only two dot-separated parts (no signature).
        let h = URL_SAFE_NO_PAD.encode(b"{\"alg\":\"EdDSA\"}");
        let p = URL_SAFE_NO_PAD.encode(b"{\"iss\":\"test\"}");
        let compact = format!("{h}.{p}");

        let result = parse(&compact);
        assert!(matches!(result, Err(SdJwtError::MalformedCompact)));
    }

    #[test]
    fn error_invalid_base64() {
        // Invalid base64 in header.
        let compact = "!!!invalid!!!.eyJpc3MiOiJ0ZXN0In0.c2ln";
        let result = parse(compact);
        assert!(matches!(result, Err(SdJwtError::Base64Error)));
    }

    #[test]
    fn error_invalid_json() {
        // Valid base64 but not valid JSON.
        let h = URL_SAFE_NO_PAD.encode(b"not json");
        let p = URL_SAFE_NO_PAD.encode(b"{\"iss\":\"test\"}");
        let s = URL_SAFE_NO_PAD.encode(b"sig");
        let compact = format!("{h}.{p}.{s}");

        let result = parse(&compact);
        assert!(matches!(result, Err(SdJwtError::JsonError(_))));
    }

    #[test]
    fn error_missing_alg() {
        let header = serde_json::json!({"typ": "sd+jwt"});
        let payload = serde_json::json!({"iss": "test"});
        let compact = build_jws(&header, &payload, b"sig");

        let result = parse(&compact);
        assert!(matches!(result, Err(SdJwtError::MissingAlgorithm)));
    }

    #[test]
    fn error_invalid_disclosure() {
        let header = serde_json::json!({"alg": "EdDSA"});
        let payload = serde_json::json!({"iss": "test"});
        // A disclosure with only 1 element is invalid.
        let bad_disclosure = build_disclosure(&serde_json::json!(["only_salt"]));
        let compact = format!(
            "{}~{}~",
            build_jws(&header, &payload, b"sig"),
            bad_disclosure
        );

        let result = parse(&compact);
        assert!(matches!(result, Err(SdJwtError::InvalidDisclosure)));
    }

    #[test]
    fn parse_with_key_binding_jwt() {
        let header = serde_json::json!({"alg": "EdDSA"});
        let payload = serde_json::json!({"_sd": []});
        let disc = build_disclosure(&serde_json::json!(["salt1", "name", "value"]));
        // KB-JWT is a compact JWS (contains dots)
        let kb_jwt = "eyJhbGciOiJFZERTQSJ9.eyJub25jZSI6InRlc3QifQ.c2lnbmF0dXJl";
        let compact = format!(
            "{}~{}~{}~",
            build_jws(&header, &payload, b"sig"),
            disc,
            kb_jwt
        );

        let sd_jwt = parse(&compact).unwrap();
        assert_eq!(sd_jwt.disclosures.len(), 1);
        assert_eq!(sd_jwt.key_binding_jwt.as_deref(), Some(kb_jwt));
    }

    #[test]
    fn parse_rejects_multiple_key_binding_jwts() {
        let header = serde_json::json!({"alg": "EdDSA"});
        let payload = serde_json::json!({"_sd": []});
        let kb1 = "eyJhbGciOiJFZERTQSJ9.eyJub25jZSI6IjEifQ.c2ln";
        let kb2 = "eyJhbGciOiJFZERTQSJ9.eyJub25jZSI6IjIifQ.c2ln";
        let compact = format!(
            "{}~{}~{}~",
            build_jws(&header, &payload, b"sig"),
            kb1,
            kb2
        );
        assert!(matches!(parse(&compact), Err(SdJwtError::MalformedCompact)));
    }

    #[test]
    fn parse_rejects_disclosure_after_kb_jwt() {
        let header = serde_json::json!({"alg": "EdDSA"});
        let payload = serde_json::json!({"_sd": []});
        let kb_jwt = "eyJhbGciOiJFZERTQSJ9.eyJub25jZSI6InRlc3QifQ.c2ln";
        let disc = build_disclosure(&serde_json::json!(["salt1", "name", "value"]));
        // KB-JWT must be last — disclosure after it is invalid
        let compact = format!(
            "{}~{}~{}~",
            build_jws(&header, &payload, b"sig"),
            kb_jwt,
            disc
        );
        assert!(matches!(parse(&compact), Err(SdJwtError::MalformedCompact)));
    }

    #[test]
    fn parse_without_key_binding_jwt() {
        let header = serde_json::json!({"alg": "EdDSA"});
        let payload = serde_json::json!({"iss": "test"});
        let compact = build_jws(&header, &payload, b"sig");

        let sd_jwt = parse(&compact).unwrap();
        assert!(sd_jwt.key_binding_jwt.is_none());
    }

    // --- signing_input tests (no_std compatible) ---

    #[test]
    fn signing_input_extracts_correctly() {
        let header = serde_json::json!({"alg": "EdDSA"});
        let payload = serde_json::json!({"iss": "test"});
        let jws = build_jws(&header, &payload, b"sig");

        let (si, sig) = signing_input(&jws).unwrap();
        // signing_input is header.payload
        assert!(si.contains('.'));
        assert_eq!(si.matches('.').count(), 1);
        assert_eq!(sig, URL_SAFE_NO_PAD.encode(b"sig"));
    }

    #[test]
    fn signing_input_with_disclosures() {
        let header = serde_json::json!({"alg": "EdDSA"});
        let payload = serde_json::json!({"iss": "test"});
        let jws = build_jws(&header, &payload, b"sig");
        let compact = format!("{jws}~disclosure1~disclosure2~");

        let (si, sig) = signing_input(&compact).unwrap();
        // signing_input should only contain the header.payload portion.
        assert!(!si.contains('~'));
        assert_eq!(sig, URL_SAFE_NO_PAD.encode(b"sig"));
    }

    #[test]
    fn signing_input_error_empty() {
        let result = signing_input("");
        assert!(matches!(result, Err(SdJwtError::EmptyInput)));
    }

    #[test]
    fn signing_input_error_no_dots() {
        let result = signing_input("nodots");
        assert!(matches!(result, Err(SdJwtError::MalformedCompact)));
    }

    #[test]
    fn signing_input_error_missing_signature() {
        // header.payload but no signature part
        let result = signing_input("header.payload");
        assert!(matches!(result, Err(SdJwtError::MalformedCompact)));
    }

    #[test]
    fn multiple_disclosures_parsed() {
        let header = serde_json::json!({"alg": "EdDSA"});
        let payload = serde_json::json!({"iss": "issuer"});
        let d1 = build_disclosure(&serde_json::json!(["s1", "name", "Alice"]));
        let d2 = build_disclosure(&serde_json::json!(["s2", "age", 30]));
        let d3 = build_disclosure(&serde_json::json!(["s3", true])); // array element
        let compact = format!(
            "{}~{}~{}~{}~",
            build_jws(&header, &payload, b"sig"),
            d1,
            d2,
            d3
        );

        let sd_jwt = parse(&compact).unwrap();
        assert_eq!(sd_jwt.disclosures.len(), 3);

        assert_eq!(sd_jwt.disclosures[0].claim_name.as_deref(), Some("name"));
        assert_eq!(sd_jwt.disclosures[0].value, serde_json::json!("Alice"));

        assert_eq!(sd_jwt.disclosures[1].claim_name.as_deref(), Some("age"));
        assert_eq!(sd_jwt.disclosures[1].value, serde_json::json!(30));

        assert!(sd_jwt.disclosures[2].claim_name.is_none());
        assert_eq!(sd_jwt.disclosures[2].value, serde_json::json!(true));
    }
}
