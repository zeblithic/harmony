//! DID Document utilities: URL mapping and document parsing.
//!
//! This module provides:
//! - [`did_web_to_url`] — pure, no-I/O conversion from a `did:web` DID to its
//!   canonical HTTPS fetch URL (W3C did:web spec).
//! - [`parse_did_document`] — parses W3C DID Document JSON and extracts all
//!   supported verification methods (Ed25519, ML-DSA-65). Requires `std`.

use alloc::format;
use alloc::string::String;

use crate::did::DidError;

/// Convert a `did:web` DID string into the HTTPS URL where its DID Document
/// is published, following the W3C did:web specification.
///
/// Rules:
/// - `did:web:example.com` → `https://example.com/.well-known/did.json`
/// - `did:web:example.com:path:to:resource` → `https://example.com/path/to/resource/did.json`
/// - Percent-encoded characters in the domain (e.g. `%3A` for `:`) are decoded
///   before the URL is assembled.
///
/// Returns [`DidError::UnsupportedMethod`] if the DID does not start with `did:web:`,
/// and [`DidError::MalformedDid`] if the method-specific identifier is empty.
pub fn did_web_to_url(did: &str) -> Result<String, DidError> {
    let method_specific_id = did
        .strip_prefix("did:web:")
        .ok_or_else(|| DidError::UnsupportedMethod(String::from("web")))?;

    if method_specific_id.is_empty() {
        return Err(DidError::MalformedDid(String::from(
            "did:web method-specific identifier is empty",
        )));
    }

    // Split on ':' to separate domain from optional path segments.
    // The first segment is the domain (possibly percent-encoded), and
    // any remaining segments become path components.
    let mut parts = method_specific_id.split(':');
    let raw_domain = parts.next().expect("split always yields at least one element");

    // Decode percent-encoded characters in the domain (e.g. %3A → :)
    let domain = percent_decode(raw_domain)?;

    // Collect remaining segments as path components
    let path_segments: alloc::vec::Vec<&str> = parts.collect();

    let url = if path_segments.is_empty() {
        format!("https://{}/.well-known/did.json", domain)
    } else {
        format!("https://{}/{}/did.json", domain, path_segments.join("/"))
    };

    Ok(url)
}

/// Decode percent-encoded ASCII sequences (e.g. `%3A` → `:`) in a string.
///
/// Only single-byte ASCII code points (0x00–0x7F) are decoded. Non-ASCII
/// decoded bytes (≥ 0x80) are rejected — the W3C did:web spec uses punycode
/// for internationalized domain names, so multi-byte UTF-8 percent sequences
/// indicate a malformed DID. A bare `%` not followed by exactly two hex
/// digits is returned as-is (lenient decoding).
fn percent_decode(input: &str) -> Result<String, DidError> {
    let mut output = String::with_capacity(input.len());
    let bytes = input.as_bytes();
    let mut i = 0;

    while i < bytes.len() {
        if bytes[i] == b'%' && i + 2 < bytes.len() {
            let hi = bytes[i + 1];
            let lo = bytes[i + 2];
            if hi.is_ascii_hexdigit() && lo.is_ascii_hexdigit() {
                let decoded = hex_to_byte(hi, lo);
                if decoded > 0x7F {
                    return Err(DidError::MalformedDid(String::from(
                        "non-ASCII percent-encoded byte in did:web domain (use punycode)",
                    )));
                }
                output.push(decoded as char);
                i += 3;
                continue;
            }
        }
        output.push(bytes[i] as char);
        i += 1;
    }

    Ok(output)
}

/// Convert two ASCII hex digit bytes to a single byte value.
fn hex_to_byte(hi: u8, lo: u8) -> u8 {
    fn digit(b: u8) -> u8 {
        match b {
            b'0'..=b'9' => b - b'0',
            b'a'..=b'f' => b - b'a' + 10,
            b'A'..=b'F' => b - b'A' + 10,
            _ => 0,
        }
    }
    (digit(hi) << 4) | digit(lo)
}

/// Parse a W3C DID Document JSON and extract all supported verification methods.
///
/// Verification methods with unsupported key types are silently skipped.
/// Returns [`DidError::NoSupportedKeys`] if no methods use a supported suite
/// (Ed25519, ML-DSA-65).
///
/// Requires the `std` feature (uses `serde_json`).
#[cfg(feature = "std")]
pub fn parse_did_document(
    did: &str,
    json_bytes: &[u8],
) -> Result<crate::did::ResolvedDidDocument, DidError> {
    let doc: serde_json::Value = serde_json::from_slice(json_bytes)
        .map_err(|e| DidError::DecodingError(format!("invalid DID Document JSON: {e}")))?;

    let doc_id = doc
        .get("id")
        .and_then(|v| v.as_str())
        .ok_or_else(|| DidError::MalformedDid(String::from("DID Document missing id field")))?;

    if doc_id != did {
        return Err(DidError::MalformedDid(format!(
            "DID Document id mismatch: expected \"{did}\", got \"{doc_id}\""
        )));
    }

    let methods = doc
        .get("verificationMethod")
        .and_then(|v| v.as_array())
        .ok_or_else(|| {
            DidError::MalformedDid(String::from(
                "DID Document missing verificationMethod array",
            ))
        })?;

    let mut resolved = alloc::vec::Vec::new();
    for method in methods {
        if let Some(jwk) = method.get("publicKeyJwk") {
            if let Ok(r) = crate::did::parse_jwk_value(jwk) {
                resolved.push(r);
                continue;
            }
        }
        if let Some(mb) = method.get("publicKeyMultibase").and_then(|v| v.as_str()) {
            if let Ok(r) = crate::did::parse_multibase_key(mb) {
                resolved.push(r);
            }
        }
    }

    if resolved.is_empty() {
        return Err(DidError::NoSupportedKeys);
    }

    Ok(crate::did::ResolvedDidDocument {
        id: String::from(did),
        verification_methods: resolved,
    })
}

/// Trait for fetching HTTP resources. Injected by the caller to preserve sans-I/O.
#[cfg(feature = "std")]
pub trait WebDidFetcher {
    fn fetch(&self, url: &str) -> Result<alloc::vec::Vec<u8>, DidError>;
}

/// A [`crate::did::DidResolver`] that supports `did:key`, `did:jwk`, and `did:web`.
#[cfg(feature = "std")]
pub struct WebDidResolver<F: WebDidFetcher> {
    fetcher: F,
}

#[cfg(feature = "std")]
impl<F: WebDidFetcher> WebDidResolver<F> {
    pub fn new(fetcher: F) -> Self {
        Self { fetcher }
    }
}

#[cfg(feature = "std")]
impl<F: WebDidFetcher> crate::did::DidResolver for WebDidResolver<F> {
    fn resolve(&self, did: &str) -> Result<crate::did::ResolvedDid, DidError> {
        if did.starts_with("did:web:") {
            let doc = self.resolve_document(did)?;
            doc.verification_methods
                .into_iter()
                .next()
                .ok_or(DidError::NoSupportedKeys)
        } else {
            crate::did::resolve_did(did)
        }
    }

    fn resolve_document(&self, did: &str) -> Result<crate::did::ResolvedDidDocument, DidError> {
        if did.starts_with("did:web:") {
            let url = did_web_to_url(did)?;
            let bytes = self.fetcher.fetch(&url)?;
            parse_did_document(did, &bytes)
        } else {
            let resolved = crate::did::resolve_did(did)?;
            Ok(crate::did::ResolvedDidDocument {
                id: alloc::string::String::from(did),
                verification_methods: alloc::vec![resolved],
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn did_web_root_domain() {
        let url = did_web_to_url("did:web:example.com").unwrap();
        assert_eq!(url, "https://example.com/.well-known/did.json");
    }

    #[test]
    fn did_web_path_based() {
        let url = did_web_to_url("did:web:example.com:issuers:1").unwrap();
        assert_eq!(url, "https://example.com/issuers/1/did.json");
    }

    #[test]
    fn did_web_percent_encoded_port() {
        let url = did_web_to_url("did:web:example.com%3A8443").unwrap();
        assert_eq!(url, "https://example.com:8443/.well-known/did.json");
    }

    #[test]
    fn did_web_percent_encoded_port_with_path() {
        let url = did_web_to_url("did:web:example.com%3A8443:users:alice").unwrap();
        assert_eq!(url, "https://example.com:8443/users/alice/did.json");
    }

    #[test]
    fn did_web_not_did_web_prefix() {
        assert!(did_web_to_url("did:key:z6Mk...").is_err());
    }

    #[test]
    fn did_web_empty_identifier() {
        assert!(did_web_to_url("did:web:").is_err());
    }

    #[test]
    fn did_web_non_ascii_percent_rejected() {
        // %C3%A9 is UTF-8 for 'é' — did:web domains must use punycode
        let err = did_web_to_url("did:web:caf%C3%A9.example").unwrap_err();
        assert!(matches!(err, DidError::MalformedDid(ref msg) if msg.contains("punycode")));
    }

    #[cfg(feature = "std")]
    mod parse_tests {
        use super::super::*;
        use base64::Engine as _;
        use crate::crypto_suite::CryptoSuite;

        fn ed25519_jwk_document(did: &str, key_b64: &str) -> String {
            format!(
                r#"{{
                "@context": "https://www.w3.org/ns/did/v1",
                "id": "{did}",
                "verificationMethod": [{{
                    "id": "{did}#key-1",
                    "type": "JsonWebKey2020",
                    "controller": "{did}",
                    "publicKeyJwk": {{
                        "kty": "OKP",
                        "crv": "Ed25519",
                        "x": "{key_b64}"
                    }}
                }}]
            }}"#
            )
        }

        #[test]
        fn parse_ed25519_jwk_document() {
            let key_bytes = [42u8; 32];
            let key_b64 =
                base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(&key_bytes);
            let did = "did:web:example.com";
            let json = ed25519_jwk_document(did, &key_b64);
            let doc = parse_did_document(did, json.as_bytes()).unwrap();
            assert_eq!(doc.id, did);
            assert_eq!(doc.verification_methods.len(), 1);
            assert_eq!(doc.verification_methods[0].suite, CryptoSuite::Ed25519);
            assert_eq!(doc.verification_methods[0].public_key, key_bytes.to_vec());
        }

        #[test]
        fn parse_multibase_document() {
            // Ed25519 multicodec 0x00ed = LEB128 [0xed, 0x01]
            let key = [0xABu8; 32];
            let mut payload = alloc::vec![0xed, 0x01];
            payload.extend_from_slice(&key);
            let multibase = format!("z{}", bs58::encode(&payload).into_string());
            let did = "did:web:example.com";
            let json = format!(
                r#"{{
                "@context": "https://www.w3.org/ns/did/v1",
                "id": "{did}",
                "verificationMethod": [{{
                    "id": "{did}#key-1",
                    "type": "Multikey",
                    "controller": "{did}",
                    "publicKeyMultibase": "{multibase}"
                }}]
            }}"#
            );
            let doc = parse_did_document(did, json.as_bytes()).unwrap();
            assert_eq!(doc.verification_methods.len(), 1);
            assert_eq!(doc.verification_methods[0].suite, CryptoSuite::Ed25519);
        }

        #[test]
        fn parse_skips_unsupported_keys() {
            let did = "did:web:example.com";
            let json = format!(
                r#"{{
                "@context": "https://www.w3.org/ns/did/v1",
                "id": "{did}",
                "verificationMethod": [{{
                    "id": "{did}#key-1",
                    "type": "JsonWebKey2020",
                    "controller": "{did}",
                    "publicKeyJwk": {{
                        "kty": "EC",
                        "crv": "P-256",
                        "x": "AAAA",
                        "y": "BBBB"
                    }}
                }}, {{
                    "id": "{did}#key-2",
                    "type": "JsonWebKey2020",
                    "controller": "{did}",
                    "publicKeyJwk": {{
                        "kty": "OKP",
                        "crv": "Ed25519",
                        "x": "{}"
                    }}
                }}]
            }}"#,
                base64::engine::general_purpose::URL_SAFE_NO_PAD.encode([99u8; 32])
            );
            let doc = parse_did_document(did, json.as_bytes()).unwrap();
            assert_eq!(doc.verification_methods.len(), 1);
            assert_eq!(doc.verification_methods[0].suite, CryptoSuite::Ed25519);
        }

        #[test]
        fn parse_id_mismatch_rejected() {
            let json = r#"{
                "@context": "https://www.w3.org/ns/did/v1",
                "id": "did:web:evil.com",
                "verificationMethod": []
            }"#;
            let err =
                parse_did_document("did:web:example.com", json.as_bytes()).unwrap_err();
            assert!(
                matches!(err, crate::did::DidError::MalformedDid(ref msg) if msg.contains("mismatch"))
            );
        }

        #[test]
        fn parse_no_supported_keys_error() {
            let did = "did:web:example.com";
            let json = format!(
                r#"{{
                "@context": "https://www.w3.org/ns/did/v1",
                "id": "{did}",
                "verificationMethod": [{{
                    "id": "{did}#key-1",
                    "type": "JsonWebKey2020",
                    "controller": "{did}",
                    "publicKeyJwk": {{ "kty": "EC", "crv": "P-256", "x": "AA", "y": "BB" }}
                }}]
            }}"#
            );
            let err = parse_did_document(did, json.as_bytes()).unwrap_err();
            assert!(matches!(err, crate::did::DidError::NoSupportedKeys));
        }

        #[test]
        fn parse_missing_verification_method() {
            let did = "did:web:example.com";
            let json = format!(
                r#"{{
                "@context": "https://www.w3.org/ns/did/v1",
                "id": "{did}"
            }}"#
            );
            assert!(parse_did_document(did, json.as_bytes()).is_err());
        }
    }

    #[cfg(feature = "std")]
    mod resolver_tests {
        use super::super::*;
        use base64::Engine as _;
        use crate::did::{DidError, DidResolver};
        use crate::crypto_suite::CryptoSuite;

        struct MockFetcher {
            response: Result<alloc::vec::Vec<u8>, DidError>,
        }

        impl WebDidFetcher for MockFetcher {
            fn fetch(&self, _url: &str) -> Result<alloc::vec::Vec<u8>, DidError> {
                match &self.response {
                    Ok(bytes) => Ok(bytes.clone()),
                    Err(e) => Err(DidError::DecodingError(alloc::format!("{e:?}"))),
                }
            }
        }

        #[test]
        fn web_resolver_resolves_did_web() {
            let key_bytes = [42u8; 32];
            let key_b64 = base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(&key_bytes);
            let did = "did:web:example.com";
            let json = alloc::format!(r#"{{
                "@context": "https://www.w3.org/ns/did/v1",
                "id": "{did}",
                "verificationMethod": [{{
                    "id": "{did}#key-1",
                    "type": "JsonWebKey2020",
                    "controller": "{did}",
                    "publicKeyJwk": {{ "kty": "OKP", "crv": "Ed25519", "x": "{key_b64}" }}
                }}]
            }}"#);
            let fetcher = MockFetcher { response: Ok(json.into_bytes()) };
            let resolver = WebDidResolver::new(fetcher);
            let resolved = resolver.resolve(did).unwrap();
            assert_eq!(resolved.suite, CryptoSuite::Ed25519);
            assert_eq!(resolved.public_key, key_bytes.to_vec());
        }

        #[test]
        fn web_resolver_resolve_document_returns_all_keys() {
            let key1 = [1u8; 32];
            let key2 = [2u8; 32];
            let b64_1 = base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(&key1);
            let b64_2 = base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(&key2);
            let did = "did:web:example.com";
            let json = alloc::format!(r#"{{
                "@context": "https://www.w3.org/ns/did/v1",
                "id": "{did}",
                "verificationMethod": [
                    {{ "id": "{did}#k1", "type": "JsonWebKey2020", "controller": "{did}",
                       "publicKeyJwk": {{ "kty": "OKP", "crv": "Ed25519", "x": "{b64_1}" }} }},
                    {{ "id": "{did}#k2", "type": "JsonWebKey2020", "controller": "{did}",
                       "publicKeyJwk": {{ "kty": "OKP", "crv": "Ed25519", "x": "{b64_2}" }} }}
                ]
            }}"#);
            let resolver = WebDidResolver::new(MockFetcher { response: Ok(json.into_bytes()) });
            let doc = resolver.resolve_document(did).unwrap();
            assert_eq!(doc.verification_methods.len(), 2);
        }

        #[test]
        fn web_resolver_falls_back_for_did_key() {
            // Ed25519 multicodec 0x00ed = LEB128 [0xed, 0x01]
            let key = [0xABu8; 32];
            let mut payload = alloc::vec![0xed, 0x01];
            payload.extend_from_slice(&key);
            let encoded = alloc::format!("z{}", bs58::encode(&payload).into_string());
            let did = alloc::format!("did:key:{encoded}");
            let fetcher = MockFetcher {
                response: Err(DidError::DecodingError("should not be called".into())),
            };
            let resolver = WebDidResolver::new(fetcher);
            let resolved = resolver.resolve(&did).unwrap();
            assert_eq!(resolved.suite, CryptoSuite::Ed25519);
        }

        #[test]
        fn web_resolver_fetch_failure() {
            let fetcher = MockFetcher {
                response: Err(DidError::DecodingError("network error".into())),
            };
            let resolver = WebDidResolver::new(fetcher);
            assert!(resolver.resolve("did:web:unreachable.example").is_err());
        }
    }
}
