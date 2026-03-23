//! DID (Decentralized Identifier) resolution for Harmony.
//!
//! Supports:
//! - `did:key` — multicodec-prefixed public keys (Ed25519, ML-DSA-65)
//! - `did:jwk` — JSON Web Key encoded DIDs (std only)

use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;

use crate::crypto_suite::CryptoSuite;

/// A resolved DID containing the cryptographic suite and raw public key bytes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedDid {
    /// The cryptographic suite identified by the DID.
    pub suite: CryptoSuite,
    /// The raw public key bytes.
    pub public_key: Vec<u8>,
}

/// Errors that can occur during DID resolution.
#[non_exhaustive]
#[derive(Debug)]
pub enum DidError {
    /// The DID method is not supported.
    UnsupportedMethod(String),
    /// The DID string is malformed.
    MalformedDid(String),
    /// Base58 or base64 decoding failed.
    DecodingError(String),
    /// The multicodec prefix is not recognized.
    UnknownMulticodec(u32),
}

impl core::fmt::Display for DidError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::UnsupportedMethod(method) => {
                write!(f, "unsupported DID method: {method}")
            }
            Self::MalformedDid(msg) => write!(f, "malformed DID: {msg}"),
            Self::DecodingError(msg) => write!(f, "decoding error: {msg}"),
            Self::UnknownMulticodec(code) => {
                write!(f, "unknown multicodec: 0x{code:04x}")
            }
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for DidError {}

/// Trait for DID resolution.
pub trait DidResolver {
    /// Resolve a DID string into its public key and crypto suite.
    fn resolve(&self, did: &str) -> Result<ResolvedDid, DidError>;
}

/// Default DID resolver supporting `did:key` and `did:jwk` (std only).
pub struct DefaultDidResolver;

impl DidResolver for DefaultDidResolver {
    fn resolve(&self, did: &str) -> Result<ResolvedDid, DidError> {
        resolve_did(did)
    }
}

/// Resolve a DID string into its public key and crypto suite.
///
/// Dispatches on method prefix:
/// - `did:key:` — multicodec-prefixed public key
/// - `did:jwk:` — JSON Web Key (std only)
/// - `did:web:` — not supported
pub fn resolve_did(did: &str) -> Result<ResolvedDid, DidError> {
    if let Some(encoded) = did.strip_prefix("did:key:") {
        resolve_did_key(encoded)
    } else if let Some(encoded) = did.strip_prefix("did:jwk:") {
        #[cfg(feature = "std")]
        {
            resolve_did_jwk(encoded)
        }
        #[cfg(not(feature = "std"))]
        {
            let _ = encoded;
            Err(DidError::UnsupportedMethod(String::from("jwk (requires std)")))
        }
    } else if did.starts_with("did:web:") {
        Err(DidError::UnsupportedMethod(String::from("web")))
    } else {
        // Extract method name for the error message
        let method = did
            .strip_prefix("did:")
            .and_then(|rest| rest.split(':').next())
            .unwrap_or(did);
        Err(DidError::UnsupportedMethod(String::from(method)))
    }
}

/// Resolve a `did:key` method-specific identifier.
///
/// Format: `z<base58btc(varint(multicodec) || raw_public_key)>`
pub fn resolve_did_key(encoded: &str) -> Result<ResolvedDid, DidError> {
    // Strip multibase prefix 'z' (base58btc)
    let b58_str = encoded
        .strip_prefix('z')
        .ok_or_else(|| DidError::MalformedDid(String::from("did:key value must start with 'z' multibase prefix")))?;

    // Base58 decode
    let bytes = bs58::decode(b58_str)
        .into_vec()
        .map_err(|e| DidError::DecodingError(format!("base58 decode failed: {e}")))?;

    if bytes.is_empty() {
        return Err(DidError::MalformedDid(String::from(
            "decoded did:key payload is empty",
        )));
    }

    // Read LEB128 varint multicodec prefix
    let (codec, varint_len) = decode_varint(&bytes)?;

    // Map multicodec to CryptoSuite (safe u32→u16 conversion)
    let suite = u16::try_from(codec)
        .ok()
        .and_then(CryptoSuite::from_signing_multicodec)
        .ok_or(DidError::UnknownMulticodec(codec))?;

    let key_bytes = &bytes[varint_len..];

    // Validate key length
    let expected_len = expected_key_length(suite);
    if key_bytes.len() != expected_len {
        return Err(DidError::MalformedDid(format!(
            "expected {expected_len} key bytes for {suite:?}, got {}",
            key_bytes.len()
        )));
    }

    Ok(ResolvedDid {
        suite,
        public_key: key_bytes.to_vec(),
    })
}

/// Resolve a `did:jwk` method-specific identifier (std only).
///
/// Format: `<base64url(JWK JSON)>`
#[cfg(feature = "std")]
pub fn resolve_did_jwk(encoded: &str) -> Result<ResolvedDid, DidError> {
    use base64::engine::general_purpose::URL_SAFE_NO_PAD;
    use base64::Engine;

    // base64url decode → JSON bytes
    let json_bytes = URL_SAFE_NO_PAD
        .decode(encoded)
        .map_err(|e| DidError::DecodingError(format!("base64url decode failed: {e}")))?;

    // Parse JSON
    let jwk: serde_json::Value = serde_json::from_slice(&json_bytes)
        .map_err(|e| DidError::MalformedDid(format!("invalid JWK JSON: {e}")))?;

    let kty = jwk
        .get("kty")
        .and_then(|v| v.as_str())
        .ok_or_else(|| DidError::MalformedDid(String::from("JWK missing 'kty' field")))?;

    let crv = jwk.get("crv").and_then(|v| v.as_str());

    match (kty, crv) {
        ("OKP", Some("Ed25519")) => {
            let x = jwk
                .get("x")
                .and_then(|v| v.as_str())
                .ok_or_else(|| DidError::MalformedDid(String::from("JWK missing 'x' field")))?;

            let key_bytes = URL_SAFE_NO_PAD
                .decode(x)
                .map_err(|e| DidError::DecodingError(format!("base64url decode of 'x' failed: {e}")))?;

            if key_bytes.len() != 32 {
                return Err(DidError::MalformedDid(format!(
                    "expected 32 key bytes for Ed25519, got {}",
                    key_bytes.len()
                )));
            }

            Ok(ResolvedDid {
                suite: CryptoSuite::Ed25519,
                public_key: key_bytes,
            })
        }
        _ => Err(DidError::MalformedDid(format!(
            "unsupported JWK key type: kty={kty}, crv={crv:?}"
        ))),
    }
}

/// Decode an unsigned LEB128 varint from the front of a byte slice.
///
/// Returns `(value, bytes_consumed)`.
fn decode_varint(bytes: &[u8]) -> Result<(u32, usize), DidError> {
    let mut result: u32 = 0;
    let mut shift: u32 = 0;

    for (i, &byte) in bytes.iter().enumerate() {
        if shift >= 35 {
            return Err(DidError::MalformedDid(String::from(
                "varint too large for u32",
            )));
        }

        let value = (byte & 0x7F) as u32;

        // On the 5th byte (shift=28), only 4 bits remain in u32.
        // Reject if bits 4-6 are set (would overflow).
        if shift == 28 && value > 0x0F {
            return Err(DidError::MalformedDid(String::from(
                "varint too large for u32",
            )));
        }

        result |= value << shift;
        shift += 7;

        if byte & 0x80 == 0 {
            return Ok((result, i + 1));
        }
    }

    Err(DidError::MalformedDid(String::from(
        "unterminated varint",
    )))
}

/// Expected raw public key length for a given crypto suite.
fn expected_key_length(suite: CryptoSuite) -> usize {
    match suite {
        CryptoSuite::Ed25519 => 32,
        CryptoSuite::MlDsa65 | CryptoSuite::MlDsa65Rotatable => 1952,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    /// Encode a u32 as unsigned LEB128.
    fn encode_varint(mut value: u32) -> Vec<u8> {
        let mut buf = Vec::new();
        loop {
            let mut byte = (value & 0x7F) as u8;
            value >>= 7;
            if value != 0 {
                byte |= 0x80;
            }
            buf.push(byte);
            if value == 0 {
                break;
            }
        }
        buf
    }

    #[test]
    fn did_key_ed25519_round_trip() {
        // Ed25519 multicodec = 0x00ed, LEB128 = [0xed, 0x01]
        let key = [0xABu8; 32];
        let mut payload = encode_varint(0x00ed);
        payload.extend_from_slice(&key);
        let encoded = format!("z{}", bs58::encode(&payload).into_string());
        let did = format!("did:key:{encoded}");

        let resolved = resolve_did(&did).unwrap();
        assert_eq!(resolved.suite, CryptoSuite::Ed25519);
        assert_eq!(resolved.public_key, key);
    }

    #[test]
    fn did_key_ml_dsa65_round_trip() {
        // ML-DSA-65 multicodec = 0x1211, LEB128 = [0x91, 0x24]
        let key = [0xCDu8; 1952];
        let mut payload = encode_varint(0x1211);
        payload.extend_from_slice(&key);
        let encoded = format!("z{}", bs58::encode(&payload).into_string());

        let resolved = resolve_did_key(&encoded).unwrap();
        assert_eq!(resolved.suite, CryptoSuite::MlDsa65);
        assert_eq!(resolved.public_key, key.to_vec());
    }

    #[test]
    fn did_key_unknown_multicodec() {
        let key = [0x00u8; 32];
        let mut payload = encode_varint(0xFFFF);
        payload.extend_from_slice(&key);
        let encoded = format!("z{}", bs58::encode(&payload).into_string());

        let err = resolve_did_key(&encoded).unwrap_err();
        assert!(matches!(err, DidError::UnknownMulticodec(0xFFFF)));
    }

    #[test]
    fn did_key_missing_z_prefix() {
        let err = resolve_did_key("not_starting_with_z").unwrap_err();
        assert!(matches!(err, DidError::MalformedDid(_)));
    }

    #[test]
    fn did_key_bad_base58() {
        // 'l' is not valid base58
        let err = resolve_did_key("zlllll").unwrap_err();
        assert!(matches!(err, DidError::DecodingError(_)));
    }

    #[test]
    fn did_key_wrong_key_length() {
        // Ed25519 expects 32 bytes, provide 16
        let key = [0xABu8; 16];
        let mut payload = encode_varint(0x00ed);
        payload.extend_from_slice(&key);
        let encoded = format!("z{}", bs58::encode(&payload).into_string());

        let err = resolve_did_key(&encoded).unwrap_err();
        assert!(matches!(err, DidError::MalformedDid(_)));
    }

    #[test]
    fn resolve_did_web_unsupported() {
        let err = resolve_did("did:web:example.com").unwrap_err();
        assert!(matches!(err, DidError::UnsupportedMethod(ref m) if m == "web"));
    }

    #[test]
    fn resolve_did_unknown_method() {
        let err = resolve_did("did:example:123").unwrap_err();
        assert!(matches!(err, DidError::UnsupportedMethod(ref m) if m == "example"));
    }

    #[test]
    fn resolve_did_dispatches_to_did_key() {
        let key = [0x42u8; 32];
        let mut payload = encode_varint(0x00ed);
        payload.extend_from_slice(&key);
        let encoded = format!("z{}", bs58::encode(&payload).into_string());
        let did = format!("did:key:{encoded}");

        let resolved = resolve_did(&did).unwrap();
        assert_eq!(resolved.suite, CryptoSuite::Ed25519);
        assert_eq!(resolved.public_key, key);
    }

    #[test]
    fn default_resolver_delegates() {
        let key = [0x99u8; 32];
        let mut payload = encode_varint(0x00ed);
        payload.extend_from_slice(&key);
        let encoded = format!("z{}", bs58::encode(&payload).into_string());
        let did = format!("did:key:{encoded}");

        let resolver = DefaultDidResolver;
        let resolved = resolver.resolve(&did).unwrap();
        assert_eq!(resolved.suite, CryptoSuite::Ed25519);
        assert_eq!(resolved.public_key, key);
    }

    #[test]
    fn varint_ed25519_encoding() {
        let bytes = encode_varint(0x00ed);
        assert_eq!(bytes, vec![0xed, 0x01]);
    }

    #[test]
    fn varint_ml_dsa65_encoding() {
        let bytes = encode_varint(0x1211);
        assert_eq!(bytes, vec![0x91, 0x24]);
    }
}

#[cfg(all(test, feature = "std"))]
mod jwk_tests {
    use super::*;
    use base64::engine::general_purpose::URL_SAFE_NO_PAD;
    use base64::Engine;

    fn make_ed25519_jwk(key: &[u8; 32]) -> String {
        let x = URL_SAFE_NO_PAD.encode(key);
        let jwk_json = format!(r#"{{"kty":"OKP","crv":"Ed25519","x":"{x}"}}"#);
        URL_SAFE_NO_PAD.encode(jwk_json.as_bytes())
    }

    #[test]
    fn did_jwk_ed25519() {
        let key = [0xABu8; 32];
        let encoded = make_ed25519_jwk(&key);

        let resolved = resolve_did_jwk(&encoded).unwrap();
        assert_eq!(resolved.suite, CryptoSuite::Ed25519);
        assert_eq!(resolved.public_key, key);
    }

    #[test]
    fn did_jwk_via_resolve_did() {
        let key = [0xCDu8; 32];
        let encoded = make_ed25519_jwk(&key);
        let did = format!("did:jwk:{encoded}");

        let resolved = resolve_did(&did).unwrap();
        assert_eq!(resolved.suite, CryptoSuite::Ed25519);
        assert_eq!(resolved.public_key, key);
    }

    #[test]
    fn did_jwk_malformed_json() {
        let encoded = URL_SAFE_NO_PAD.encode(b"not json at all {{{");
        let err = resolve_did_jwk(&encoded).unwrap_err();
        assert!(matches!(err, DidError::MalformedDid(_)));
    }

    #[test]
    fn did_jwk_missing_kty() {
        let jwk_json = r#"{"crv":"Ed25519","x":"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"}"#;
        let encoded = URL_SAFE_NO_PAD.encode(jwk_json.as_bytes());
        let err = resolve_did_jwk(&encoded).unwrap_err();
        assert!(matches!(err, DidError::MalformedDid(ref m) if m.contains("kty")));
    }

    #[test]
    fn did_jwk_unsupported_curve() {
        let jwk_json = r#"{"kty":"EC","crv":"P-256","x":"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"}"#;
        let encoded = URL_SAFE_NO_PAD.encode(jwk_json.as_bytes());
        let err = resolve_did_jwk(&encoded).unwrap_err();
        assert!(matches!(err, DidError::MalformedDid(_)));
    }
}
