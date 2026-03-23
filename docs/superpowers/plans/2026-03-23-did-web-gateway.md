# did:web Gateway Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enable constrained mesh nodes to resolve `did:web` identifiers by delegating HTTPS fetches to a harmony-node gateway that returns parsed keys over Zenoh.

**Architecture:** Pure DID Document parsing lives in harmony-identity (sans-I/O). harmony-node declares a Zenoh queryable on `harmony/identity/web/**`, fetches DID Documents via async reqwest, caches with TTL, and replies with postcard-serialized `ResolvedDidDocument`.

**Tech Stack:** Rust, serde/postcard, serde_json, reqwest (rustls-tls), zenoh queryables, harmony-identity DID resolution

**Spec:** `docs/superpowers/specs/2026-03-23-did-web-gateway-design.md`

---

### Task 1: Add serde derives, ResolvedDidDocument, and DidResolver extension

Add postcard/serde serialization support to DID resolution types and extend the DidResolver trait with a document-level method.

**Files:**
- Modify: `crates/harmony-identity/src/did.rs:14-58` — add derives to ResolvedDid, add ResolvedDidDocument, add NoSupportedKeys error, update Display, update DidResolver trait
- Modify: `crates/harmony-identity/src/lib.rs:14` — export ResolvedDidDocument

**Context:**
- `serde` is **already** a dependency in harmony-identity (`serde = { workspace = true, ... }`)
- `CryptoSuite` **already** has `Serialize, Deserialize` derives (crypto_suite.rs:9)
- `ResolvedDid` is at `did.rs:14-20` — does NOT yet have Serialize/Deserialize
- `DidError` is at `did.rs:23-34`, already `#[non_exhaustive]`
- `Display` impl is at `did.rs:36-48` — exhaustive match, must add new variant
- `DidResolver` trait is at `did.rs:55-58` with single `resolve` method
- `lib.rs:14` exports: `resolve_did, DefaultDidResolver, DidError, DidResolver, ResolvedDid`

- [ ] **Step 1: Add Serialize/Deserialize to ResolvedDid, add ResolvedDidDocument and NoSupportedKeys**

In `did.rs`:

Add `use serde::{Serialize, Deserialize};` at the top.

Add `Serialize, Deserialize` to `ResolvedDid`'s derive list.

Add after `ResolvedDid`:
```rust
/// A resolved DID Document containing all supported verification methods.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResolvedDidDocument {
    /// The DID this document belongs to.
    pub id: String,
    /// All verification methods with supported cryptographic suites.
    pub verification_methods: Vec<ResolvedDid>,
}
```

Add `NoSupportedKeys` variant to `DidError`:
```rust
/// The DID Document contains no verification methods with a supported crypto suite.
NoSupportedKeys,
```

Add the matching arm to the `Display` impl (did.rs:36-48):
```rust
Self::NoSupportedKeys => write!(f, "no supported verification methods found"),
```

- [ ] **Step 2: Add default `resolve_document` to DidResolver trait**

Replace the existing `DidResolver` trait (did.rs:55-58) with:
```rust
pub trait DidResolver {
    fn resolve(&self, did: &str) -> Result<ResolvedDid, DidError>;

    fn resolve_document(&self, did: &str) -> Result<ResolvedDidDocument, DidError> {
        let resolved = self.resolve(did)?;
        Ok(ResolvedDidDocument {
            id: did.to_string(),
            verification_methods: alloc::vec![resolved],
        })
    }
}
```

- [ ] **Step 3: Update lib.rs exports**

Add `ResolvedDidDocument` to the pub use line at `lib.rs:14`:
```rust
pub use did::{resolve_did, DefaultDidResolver, DidError, DidResolver, ResolvedDid, ResolvedDidDocument};
```

- [ ] **Step 4: Write test for default resolve_document**

In `did.rs` tests section (after existing tests ~line 468), add:
```rust
#[test]
fn default_resolve_document_wraps_single_key() {
    // Follow same pattern as did_key_ed25519_round_trip (line 273)
    let key = [0xABu8; 32];
    let mut payload = encode_varint(0x00ed);
    payload.extend_from_slice(&key);
    let encoded = format!("z{}", bs58::encode(&payload).into_string());
    let did = format!("did:key:{encoded}");

    let resolver = DefaultDidResolver;
    let doc = resolver.resolve_document(&did).unwrap();
    let single = resolver.resolve(&did).unwrap();
    assert_eq!(doc.id, did);
    assert_eq!(doc.verification_methods.len(), 1);
    assert_eq!(doc.verification_methods[0], single);
}
```

- [ ] **Step 5: Run tests**

```bash
cargo test -p harmony-identity --features std
```
Expected: all existing tests pass + new `default_resolve_document_wraps_single_key` passes.

- [ ] **Step 6: Commit**

```bash
git add crates/harmony-identity/
git commit -m "feat(identity): add ResolvedDidDocument, serde derives, DidResolver::resolve_document"
```

---

### Task 2: Extract key parsing helpers from resolve_did_key and resolve_did_jwk

Refactor to expose `parse_multibase_key` and `parse_jwk_value` as public functions, reusable by DID Document parsing. No behavior changes — purely mechanical extraction.

**Files:**
- Modify: `crates/harmony-identity/src/did.rs:109-203` — extract inner logic into public functions

**Context:**
- `resolve_did_key(encoded: &str)` is at lines 109-150. Its entire body becomes `parse_multibase_key`.
- `resolve_did_jwk(encoded: &str)` is at lines 156-203 (std-only). The JWK JSON value parsing becomes `parse_jwk_value`; `resolve_did_jwk` handles base64url decoding then delegates.

- [ ] **Step 1: Extract parse_multibase_key**

Add before `resolve_did_key`:
```rust
/// Parse a multibase-encoded public key string (z-prefixed base58btc with multicodec varint).
///
/// Used by both `resolve_did_key` and DID Document `publicKeyMultibase` parsing.
pub fn parse_multibase_key(encoded: &str) -> Result<ResolvedDid, DidError> {
    // Move the ENTIRE body of resolve_did_key here unchanged
}
```

Then replace `resolve_did_key` body with:
```rust
pub fn resolve_did_key(encoded: &str) -> Result<ResolvedDid, DidError> {
    parse_multibase_key(encoded)
}
```

- [ ] **Step 2: Extract parse_jwk_value**

Add before `resolve_did_jwk` (inside `#[cfg(feature = "std")]`):
```rust
/// Parse a JWK JSON value into a ResolvedDid.
///
/// Supports Ed25519 (kty="OKP", crv="Ed25519"). Used by both `resolve_did_jwk`
/// and DID Document `publicKeyJwk` parsing.
#[cfg(feature = "std")]
pub fn parse_jwk_value(jwk: &serde_json::Value) -> Result<ResolvedDid, DidError> {
    // Extract the kty/crv/x parsing logic from resolve_did_jwk
    // (everything after the base64url decode + JSON parse)
}
```

Then simplify `resolve_did_jwk`:
```rust
#[cfg(feature = "std")]
pub fn resolve_did_jwk(encoded: &str) -> Result<ResolvedDid, DidError> {
    let json_bytes = URL_SAFE_NO_PAD
        .decode(encoded)
        .map_err(|_| DidError::DecodingError(String::from("invalid base64url in did:jwk")))?;
    let jwk: serde_json::Value = serde_json::from_slice(&json_bytes)
        .map_err(|e| DidError::DecodingError(format!("invalid JWK JSON: {e}")))?;
    parse_jwk_value(&jwk)
}
```

- [ ] **Step 3: Add parse_multibase_key and parse_jwk_value to lib.rs exports**

```rust
pub use did::{..., parse_multibase_key, parse_jwk_value};
```

Note: `parse_jwk_value` is std-only. The export may need a `#[cfg(feature = "std")]` guard, or it can be exported unconditionally since the function itself is gated.

- [ ] **Step 4: Run tests to verify no regressions**

```bash
cargo test -p harmony-identity --features std
```
Expected: all tests pass (behavior unchanged).

- [ ] **Step 5: Commit**

```bash
git add crates/harmony-identity/
git commit -m "refactor(identity): extract parse_multibase_key and parse_jwk_value helpers"
```

---

### Task 3: DID-to-URL mapping

Implement `did_web_to_url` in a new `did_document` module. Pure function, no I/O, no std required.

**Files:**
- Create: `crates/harmony-identity/src/did_document.rs`
- Modify: `crates/harmony-identity/src/lib.rs` — declare module

**Context:**
- W3C did:web spec URL mapping rules:
  - `did:web:example.com` → `https://example.com/.well-known/did.json`
  - `did:web:example.com:issuers:1` → `https://example.com/issuers/1/did.json`
  - `did:web:example.com%3A8443` → `https://example.com:8443/.well-known/did.json`
  - Colons after method-specific-id are path separators
  - Percent-encoded characters are decoded in the domain

- [ ] **Step 1: Create did_document.rs with module declaration**

Add to `lib.rs`:
```rust
pub mod did_document;
```

Create `crates/harmony-identity/src/did_document.rs`:
```rust
//! DID Document parsing and did:web URL resolution.
//!
//! Pure logic — no I/O. The caller provides fetched bytes; this module
//! parses them into [`ResolvedDidDocument`].

use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;

use crate::did::DidError;
```

- [ ] **Step 2: Write failing tests for did_web_to_url**

At the bottom of `did_document.rs`:
```rust
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
}
```

- [ ] **Step 3: Run tests to verify they fail**

```bash
cargo test -p harmony-identity --features std did_document
```
Expected: FAIL — `did_web_to_url` not found.

- [ ] **Step 4: Implement did_web_to_url**

In `did_document.rs`, above the tests module:
```rust
/// Convert a `did:web` DID to its HTTPS URL per W3C did:web spec.
///
/// # Examples
///
/// - `did:web:example.com` → `https://example.com/.well-known/did.json`
/// - `did:web:example.com:issuers:1` → `https://example.com/issuers/1/did.json`
/// - `did:web:example.com%3A8443` → `https://example.com:8443/.well-known/did.json`
pub fn did_web_to_url(did: &str) -> Result<String, DidError> {
    let method_specific = did
        .strip_prefix("did:web:")
        .ok_or_else(|| DidError::MalformedDid(String::from("not a did:web DID")))?;

    if method_specific.is_empty() {
        return Err(DidError::MalformedDid(String::from(
            "empty did:web identifier",
        )));
    }

    let parts: Vec<&str> = method_specific.split(':').collect();
    let domain = percent_decode(parts[0]);

    if parts.len() == 1 {
        Ok(format!("https://{}/.well-known/did.json", domain))
    } else {
        let path = parts[1..].join("/");
        Ok(format!("https://{}/{}/did.json", domain, path))
    }
}

/// Simple percent-decoding for DID domain components.
fn percent_decode(s: &str) -> String {
    let mut result = String::with_capacity(s.len());
    let bytes = s.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' && i + 2 < bytes.len() {
            if let Ok(byte) = u8::from_str_radix(
                core::str::from_utf8(&bytes[i + 1..i + 3]).unwrap_or(""),
                16,
            ) {
                result.push(byte as char);
                i += 3;
                continue;
            }
        }
        result.push(bytes[i] as char);
        i += 1;
    }
    result
}
```

- [ ] **Step 5: Run tests**

```bash
cargo test -p harmony-identity --features std did_document
```
Expected: all 6 `did_web_to_url` tests pass.

- [ ] **Step 6: Commit**

```bash
git add crates/harmony-identity/src/did_document.rs crates/harmony-identity/src/lib.rs
git commit -m "feat(identity): add did_web_to_url DID-to-URL mapping"
```

---

### Task 4: DID Document parsing

Implement `parse_did_document` to extract verification methods from a W3C DID Document JSON. Uses `parse_multibase_key` and `parse_jwk_value` from Task 2. Gated behind `#[cfg(feature = "std")]`.

**Files:**
- Modify: `crates/harmony-identity/src/did_document.rs` — add parse_did_document function and tests

**Context:**
- DID Document JSON has `id` (string), `verificationMethod` (array of objects)
- Each method may have `publicKeyJwk` (JWK object) or `publicKeyMultibase` (string)
- Unsupported key types are silently skipped
- `id` must match the DID being resolved
- At least one supported method must be present (else `NoSupportedKeys`)

- [ ] **Step 1: Write failing tests**

Add to the existing tests module in `did_document.rs`:
```rust
#[cfg(feature = "std")]
mod parse_tests {
    use super::super::*;
    use crate::crypto_suite::CryptoSuite;

    fn ed25519_jwk_document(did: &str, key_b64: &str) -> String {
        format!(r#"{{
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
        }}"#)
    }

    #[test]
    fn parse_ed25519_jwk_document() {
        let key_bytes = [42u8; 32];
        let key_b64 = base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(&key_bytes);
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
        let mut payload = vec![0xed, 0x01];
        payload.extend_from_slice(&key);
        let multibase = format!("z{}", bs58::encode(&payload).into_string());
        let did = "did:web:example.com";
        let json = format!(r#"{{
            "@context": "https://www.w3.org/ns/did/v1",
            "id": "{did}",
            "verificationMethod": [{{
                "id": "{did}#key-1",
                "type": "Multikey",
                "controller": "{did}",
                "publicKeyMultibase": "{multibase}"
            }}]
        }}"#);
        let doc = parse_did_document(did, json.as_bytes()).unwrap();
        assert_eq!(doc.verification_methods.len(), 1);
        assert_eq!(doc.verification_methods[0].suite, CryptoSuite::Ed25519);
    }

    #[test]
    fn parse_skips_unsupported_keys() {
        let did = "did:web:example.com";
        let json = format!(r#"{{
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
        }}"#, base64::engine::general_purpose::URL_SAFE_NO_PAD.encode([99u8; 32]));
        let doc = parse_did_document(did, json.as_bytes()).unwrap();
        // P-256 skipped, Ed25519 kept
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
        let err = parse_did_document("did:web:example.com", json.as_bytes()).unwrap_err();
        assert!(matches!(err, crate::did::DidError::MalformedDid(msg) if msg.contains("mismatch")));
    }

    #[test]
    fn parse_no_supported_keys_error() {
        let did = "did:web:example.com";
        let json = format!(r#"{{
            "@context": "https://www.w3.org/ns/did/v1",
            "id": "{did}",
            "verificationMethod": [{{
                "id": "{did}#key-1",
                "type": "JsonWebKey2020",
                "controller": "{did}",
                "publicKeyJwk": {{ "kty": "EC", "crv": "P-256", "x": "AA", "y": "BB" }}
            }}]
        }}"#);
        let err = parse_did_document(did, json.as_bytes()).unwrap_err();
        assert!(matches!(err, crate::did::DidError::NoSupportedKeys));
    }

    #[test]
    fn parse_missing_verification_method() {
        let did = "did:web:example.com";
        let json = format!(r#"{{
            "@context": "https://www.w3.org/ns/did/v1",
            "id": "{did}"
        }}"#);
        assert!(parse_did_document(did, json.as_bytes()).is_err());
    }
}
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cargo test -p harmony-identity --features std parse_tests
```
Expected: FAIL — `parse_did_document` not found.

- [ ] **Step 3: Implement parse_did_document**

Add to `did_document.rs` (inside a `#[cfg(feature = "std")]` block):
```rust
#[cfg(feature = "std")]
use crate::did::ResolvedDidDocument;

/// Parse a W3C DID Document JSON and extract all supported verification methods.
///
/// Verification methods with unsupported key types are silently skipped.
/// Returns `NoSupportedKeys` if no methods use a supported suite (Ed25519, ML-DSA-65).
///
/// # Validation
///
/// - The document's `id` must match the expected `did`.
/// - At least one supported verification method must be present.
#[cfg(feature = "std")]
pub fn parse_did_document(did: &str, json_bytes: &[u8]) -> Result<ResolvedDidDocument, DidError> {
    let doc: serde_json::Value = serde_json::from_slice(json_bytes)
        .map_err(|e| DidError::DecodingError(format!("invalid DID Document JSON: {e}")))?;

    // Validate id matches the DID we asked for
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
            DidError::MalformedDid(String::from("DID Document missing verificationMethod array"))
        })?;

    let mut resolved = Vec::new();
    for method in methods {
        // Try publicKeyJwk first
        if let Some(jwk) = method.get("publicKeyJwk") {
            if let Ok(r) = crate::did::parse_jwk_value(jwk) {
                resolved.push(r);
                continue;
            }
        }
        // Try publicKeyMultibase
        if let Some(mb) = method.get("publicKeyMultibase").and_then(|v| v.as_str()) {
            if let Ok(r) = crate::did::parse_multibase_key(mb) {
                resolved.push(r);
            }
        }
    }

    if resolved.is_empty() {
        return Err(DidError::NoSupportedKeys);
    }

    Ok(ResolvedDidDocument {
        id: String::from(did),
        verification_methods: resolved,
    })
}
```

- [ ] **Step 4: Run tests**

```bash
cargo test -p harmony-identity --features std
```
Expected: all tests pass including new parse_tests.

- [ ] **Step 5: Commit**

```bash
git add crates/harmony-identity/src/did_document.rs
git commit -m "feat(identity): add DID Document parsing with JWK and multibase support"
```

---

### Task 5: WebDidFetcher trait and WebDidResolver

Add the sans-I/O fetcher trait and a resolver that delegates did:web to it while falling back to existing did:key/did:jwk for other methods.

**Files:**
- Modify: `crates/harmony-identity/src/did_document.rs` — add trait, resolver, tests
- Modify: `crates/harmony-identity/src/lib.rs` — export new types

**Context:**
- `WebDidFetcher` is the I/O boundary — callers inject HTTP
- `WebDidResolver<F>` implements `DidResolver`, dispatching did:web through the fetcher
- For non-web DIDs, delegates to `resolve_did()` (existing free function at did.rs:75)
- Tests use a mock fetcher returning fixture JSON
- **Note:** The harmony-node gateway (Task 7) uses async reqwest directly with `did_web_to_url` + `parse_did_document`, bypassing the sync `WebDidFetcher` trait. `WebDidFetcher`/`WebDidResolver` serve library consumers and unit testing — not the async gateway.

- [ ] **Step 1: Write failing tests**

Add to `did_document.rs` tests (inside `#[cfg(feature = "std")]`):
```rust
#[cfg(feature = "std")]
mod resolver_tests {
    use super::super::*;
    use crate::did::{DidError, DidResolver, ResolvedDid};
    use crate::crypto_suite::CryptoSuite;

    struct MockFetcher {
        response: Result<Vec<u8>, DidError>,
    }

    impl WebDidFetcher for MockFetcher {
        fn fetch(&self, _url: &str) -> Result<Vec<u8>, DidError> {
            match &self.response {
                Ok(bytes) => Ok(bytes.clone()),
                Err(e) => Err(DidError::DecodingError(format!("{e:?}"))),
            }
        }
    }

    #[test]
    fn web_resolver_resolves_did_web() {
        let key_bytes = [42u8; 32];
        let key_b64 = base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(&key_bytes);
        let did = "did:web:example.com";
        let json = format!(r#"{{
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
        let json = format!(r#"{{
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
        // WebDidResolver should handle did:key without using the fetcher
        // Ed25519 multicodec 0x00ed = LEB128 [0xed, 0x01]
        let key = [0xABu8; 32];
        let mut payload = vec![0xed, 0x01];
        payload.extend_from_slice(&key);
        let encoded = format!("z{}", bs58::encode(&payload).into_string());
        let did = format!("did:key:{encoded}");
        // Fetcher that would fail if called
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cargo test -p harmony-identity --features std resolver_tests
```
Expected: FAIL — `WebDidFetcher` and `WebDidResolver` not found.

- [ ] **Step 3: Implement WebDidFetcher and WebDidResolver**

Add to `did_document.rs` (inside `#[cfg(feature = "std")]` block):
```rust
/// Trait for fetching HTTP resources. Injected by the caller to preserve sans-I/O.
///
/// Implementors provide the actual HTTP client (e.g., reqwest). The library
/// parses the response bytes.
#[cfg(feature = "std")]
pub trait WebDidFetcher {
    fn fetch(&self, url: &str) -> Result<Vec<u8>, DidError>;
}

/// A [`DidResolver`] that supports `did:key`, `did:jwk`, and `did:web`.
///
/// `did:web` resolution delegates HTTP fetching to the injected [`WebDidFetcher`].
/// Other DID methods fall through to [`resolve_did`](crate::did::resolve_did).
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
                id: String::from(did),
                verification_methods: alloc::vec![resolved],
            })
        }
    }
}
```

- [ ] **Step 4: Export new types from lib.rs**

Add to `lib.rs`:
```rust
#[cfg(feature = "std")]
pub use did_document::{WebDidFetcher, WebDidResolver};
pub use did_document::did_web_to_url;
```

- [ ] **Step 5: Run tests**

```bash
cargo test -p harmony-identity --features std
```
Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add crates/harmony-identity/
git commit -m "feat(identity): add WebDidFetcher trait and WebDidResolver for did:web"
```

---

### Task 6: Zenoh namespace key expressions

Add `web` sub-module to the identity namespace for the gateway's Zenoh key expressions.

**Files:**
- Modify: `crates/harmony-zenoh/src/namespace.rs:387-424` — add web module inside identity

**Context:**
- Existing `identity` module is at namespace.rs:387-424 with PREFIX, ALL_ANNOUNCES, etc.
- Gateway listens on `harmony/identity/web/**`
- Key format: `harmony/identity/web/{domain}/{path...}`

- [ ] **Step 1: Add web module**

Inside the `identity` module in namespace.rs (after the existing `alive_key` function), add:
```rust
    /// Zenoh key expressions for the did:web gateway.
    pub mod web {
        use alloc::format;
        use alloc::string::String;

        /// Wildcard matching all did:web gateway queries.
        pub const ALL: &str = "harmony/identity/web/**";

        /// Build a key expression for a specific did:web DID.
        ///
        /// Domain and path segments map directly from the DID:
        /// `did:web:example.com:issuers:1` → `harmony/identity/web/example.com/issuers/1`
        pub fn key(domain: &str, path: &str) -> String {
            if path.is_empty() {
                format!("{}/web/{}", super::PREFIX, domain)
            } else {
                format!("{}/web/{}/{}", super::PREFIX, domain, path)
            }
        }
    }
```

- [ ] **Step 2: Write test**

Add to the namespace tests (find existing test module):
```rust
#[test]
fn identity_web_key_root() {
    assert_eq!(
        identity::web::key("example.com", ""),
        "harmony/identity/web/example.com"
    );
}

#[test]
fn identity_web_key_with_path() {
    assert_eq!(
        identity::web::key("example.com", "issuers/1"),
        "harmony/identity/web/example.com/issuers/1"
    );
}
```

- [ ] **Step 3: Run tests**

```bash
cargo test -p harmony-zenoh
```
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add crates/harmony-zenoh/src/namespace.rs
git commit -m "feat(zenoh): add identity::web namespace for did:web gateway"
```

---

### Task 7: Gateway service, config, and event loop wiring

Implement the gateway as an async task in harmony-node. Declares a Zenoh queryable, handles queries with TTL caching, fetches via async reqwest.

**Files:**
- Modify: `crates/harmony-node/Cargo.toml` — add reqwest, postcard dependencies
- Modify: `crates/harmony-node/src/config.rs:56-74` — add `did_web_cache_ttl` field
- Create: `crates/harmony-node/src/did_web_gateway.rs` — gateway module
- Modify: `crates/harmony-node/src/main.rs` or `src/lib.rs` — declare module
- Modify: `crates/harmony-node/src/event_loop.rs` — spawn gateway at startup

**Context:**
- harmony-node is a **binary crate** (no lib target). Module declared in `main.rs`.
- Event loop startup is in `event_loop.rs`. Zenoh session is created there.
- `RuntimeAction::SendReply` is NOT yet wired (line 921-923 is a debug stub). The gateway handles replies directly in its own spawned task, bypassing the runtime.
- Config is loaded from TOML via `config.rs:56-74`.
- reqwest should use `rustls-tls` feature (no OpenSSL dependency).

- [ ] **Step 1: Add dependencies to harmony-node Cargo.toml**

Add to `[dependencies]`:
```toml
reqwest = { version = "0.12", default-features = false, features = ["rustls-tls"] }
postcard = { version = "1", features = ["alloc"] }
```

Check if `postcard` is already a dependency (it may be via harmony-discovery or similar). If so, skip adding it.

- [ ] **Step 2: Add config field**

In `config.rs`, add to `ConfigFile` struct:
```rust
pub did_web_cache_ttl: Option<u64>,
```

This field is optional — defaults to 300 seconds (5 minutes) if not specified.

- [ ] **Step 3: Create did_web_gateway.rs**

Declare the module in `main.rs`:
```rust
mod did_web_gateway;
```

Create `crates/harmony-node/src/did_web_gateway.rs`:
```rust
//! did:web gateway service.
//!
//! Declares a Zenoh queryable on `harmony/identity/web/**` and resolves
//! did:web DIDs by fetching DID Documents over HTTPS. Results are cached
//! with a configurable TTL.

use std::collections::HashMap;
use std::time::{Duration, Instant};

use harmony_identity::did_document::{did_web_to_url, parse_did_document};
use harmony_identity::ResolvedDidDocument;

/// Cached DID Document with expiry.
struct CacheEntry {
    document: ResolvedDidDocument,
    expires_at: Instant,
}

/// Run the did:web gateway loop.
///
/// This function runs indefinitely, processing Zenoh queries on
/// `harmony/identity/web/**`. Each query maps to a did:web DID,
/// which is resolved via HTTPS fetch + DID Document parsing.
///
/// Results are cached in memory with the given TTL.
pub async fn run(
    queryable: zenoh::query::Queryable<()>,
    cache_ttl_secs: u64,
) {
    let client = reqwest::Client::new();
    let ttl = Duration::from_secs(cache_ttl_secs);
    let mut cache: HashMap<String, CacheEntry> = HashMap::new();

    tracing::info!("did:web gateway started (TTL={cache_ttl_secs}s)");

    while let Ok(query) = queryable.recv_async().await {
        let key_expr = query.key_expr().to_string();

        let did = match key_expr_to_did(&key_expr) {
            Some(d) => d,
            None => {
                tracing::warn!(%key_expr, "malformed did:web gateway query");
                continue;
            }
        };

        // Check cache
        let now = Instant::now();
        if let Some(entry) = cache.get(&did) {
            if entry.expires_at > now {
                tracing::debug!(%did, "did:web cache hit");
                reply_with_document(&query, &entry.document).await;
                continue;
            }
        }

        // Fetch via HTTPS
        let url = match did_web_to_url(&did) {
            Ok(u) => u,
            Err(e) => {
                tracing::warn!(%did, err = ?e, "invalid did:web DID");
                reply_empty(&query).await;
                continue;
            }
        };

        if !url.starts_with("https://") {
            tracing::warn!(%did, %url, "did:web requires HTTPS");
            reply_empty(&query).await;
            continue;
        }

        match client.get(&url).send().await {
            Ok(resp) => {
                if !resp.status().is_success() {
                    tracing::warn!(%did, status = %resp.status(), "did:web fetch failed");
                    reply_empty(&query).await;
                    continue;
                }
                match resp.bytes().await {
                    Ok(bytes) => match parse_did_document(&did, &bytes) {
                        Ok(doc) => {
                            tracing::info!(%did, keys = doc.verification_methods.len(), "did:web resolved");
                            reply_with_document(&query, &doc).await;
                            cache.insert(
                                did,
                                CacheEntry {
                                    document: doc,
                                    expires_at: now + ttl,
                                },
                            );
                        }
                        Err(e) => {
                            tracing::warn!(%did, err = ?e, "did:web document parse failed");
                            reply_empty(&query).await;
                        }
                    },
                    Err(e) => {
                        tracing::warn!(%did, err = %e, "did:web response read failed");
                        reply_empty(&query).await;
                    }
                }
            }
            Err(e) => {
                tracing::warn!(%did, err = %e, "did:web HTTPS fetch failed");
                reply_empty(&query).await;
            }
        }
    }

    tracing::info!("did:web gateway stopped");
}

/// Parse a Zenoh key expression into a did:web DID.
///
/// `harmony/identity/web/example.com/issuers/1` → `did:web:example.com:issuers:1`
fn key_expr_to_did(key_expr: &str) -> Option<String> {
    let rest = key_expr.strip_prefix("harmony/identity/web/")?;
    if rest.is_empty() {
        return None;
    }
    Some(format!("did:web:{}", rest.replace('/', ":")))
}

/// Reply with a postcard-serialized ResolvedDidDocument.
async fn reply_with_document(
    query: &zenoh::query::Query,
    doc: &ResolvedDidDocument,
) {
    match postcard::to_allocvec(doc) {
        Ok(payload) => {
            if let Err(e) = query
                .reply(query.key_expr(), payload)
                .await
            {
                tracing::warn!(err = ?e, "did:web gateway reply failed");
            }
        }
        Err(e) => {
            tracing::error!(err = ?e, "postcard serialization failed");
        }
    }
}

/// Reply with empty payload (signal "not found").
async fn reply_empty(query: &zenoh::query::Query) {
    let _ = query.reply(query.key_expr(), Vec::<u8>::new()).await;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn key_expr_to_did_root_domain() {
        assert_eq!(
            key_expr_to_did("harmony/identity/web/example.com"),
            Some("did:web:example.com".to_string())
        );
    }

    #[test]
    fn key_expr_to_did_with_path() {
        assert_eq!(
            key_expr_to_did("harmony/identity/web/example.com/issuers/1"),
            Some("did:web:example.com:issuers:1".to_string())
        );
    }

    #[test]
    fn key_expr_to_did_empty_rest() {
        assert_eq!(key_expr_to_did("harmony/identity/web/"), None);
    }

    #[test]
    fn key_expr_to_did_wrong_prefix() {
        assert_eq!(key_expr_to_did("harmony/identity/announce/abc"), None);
    }
}
```

**Important:** The exact Zenoh query/reply API may differ from what's shown above. Check the zenoh version in `Cargo.lock` and adapt the `query.reply()` calls. The pattern in the existing event_loop.rs (`RuntimeAction::SendReply` at line 921) shows the expected interface shape. Look at zenoh docs or existing code for the correct method signature.

- [ ] **Step 4: Thread config through main.rs to event_loop.rs**

The event_loop `run()` function (event_loop.rs:186) takes individual parameters, NOT the ConfigFile struct. The config is consumed in `main.rs` around lines 560-611.

**In main.rs** — resolve the TTL and pass it as a new parameter. Find the call to `crate::event_loop::run(...)` around line 602. Add a new parameter:
```rust
let did_web_cache_ttl = config_file.did_web_cache_ttl.unwrap_or(300);
```
Then pass `did_web_cache_ttl` as a new argument to `event_loop::run()`.

**In event_loop.rs** — add `did_web_cache_ttl: u64` to the `run()` function signature (line 186-195).

- [ ] **Step 5: Wire up gateway queryable in event_loop.rs**

Find where queryables are declared at startup (around lines 656-733 in `event_loop.rs`). After the existing queryable declarations, add:

```rust
// did:web gateway queryable
match session.declare_queryable(harmony_zenoh::namespace::identity::web::ALL).await {
    Ok(qbl) => {
        tokio::spawn(crate::did_web_gateway::run(qbl, did_web_cache_ttl));
    }
    Err(e) => {
        tracing::error!(err = %e, "failed to declare did:web gateway queryable");
    }
}
```

- [ ] **Step 6: Run tests**

```bash
cargo test -p harmony-node
```
Expected: `key_expr_to_did` tests pass. Full compilation succeeds.

Then run the full workspace to check no regressions:
```bash
cargo test --workspace
```

- [ ] **Step 7: Run clippy**

```bash
cargo clippy --workspace
```
Fix any warnings.

- [ ] **Step 8: Commit**

```bash
git add crates/harmony-node/
git commit -m "feat(node): add did:web gateway with TTL cache and Zenoh queryable"
```
