# DID Resolution Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Resolve `did:key` and `did:jwk` identifiers to raw public key bytes + `CryptoSuite`, with a `DidResolver` trait for extensibility.

**Architecture:** New `did.rs` module in `harmony-identity`. did:key uses base58btc + LEB128 varint multicodec (inverse of `identity_to_did_key`). did:jwk uses base64url + JSON parsing (behind `std`). `DidResolver` trait enables did:web via harmony-fyr later.

**Tech Stack:** Rust, bs58 (base58btc), base64, serde_json (optional/std), harmony-identity (CryptoSuite)

---

### File Structure

| File | Responsibility | Change |
|------|---------------|--------|
| `Cargo.toml` (workspace root) | Workspace deps | Add `bs58` to workspace deps |
| `crates/harmony-identity/Cargo.toml` | Crate manifest | Add `bs58`, `base64`, `serde_json` deps |
| `crates/harmony-identity/src/did.rs` | DID resolution | Create |
| `crates/harmony-identity/src/lib.rs` | Module + re-exports | Add `did` module |

---

### Task 1: Add dependencies and scaffold module

**Files:**
- Modify: `Cargo.toml` (workspace root)
- Modify: `crates/harmony-identity/Cargo.toml`
- Create: `crates/harmony-identity/src/did.rs`
- Modify: `crates/harmony-identity/src/lib.rs`

- [ ] **Step 1: Add bs58 to workspace deps**

In root `Cargo.toml` under `[workspace.dependencies]`, add:
```toml
bs58 = { version = "0.5", default-features = false, features = ["alloc"] }
```

- [ ] **Step 2: Add deps to harmony-identity**

In `crates/harmony-identity/Cargo.toml`, add to `[dependencies]`:
```toml
bs58 = { workspace = true }
base64 = { workspace = true, features = ["alloc"] }
serde_json = { workspace = true, optional = true }
```

Add `serde_json` to the `std` feature:
```toml
std = [
    # ... existing entries ...
    "dep:serde_json",
    "base64/std",
]
```

- [ ] **Step 3: Create did.rs with types**

```rust
//! DID resolution: parse `did:key` and `did:jwk` to raw key bytes.

use alloc::string::{String, ToString};
use alloc::vec::Vec;

use crate::crypto_suite::CryptoSuite;

/// Resolved key material from a DID.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedDid {
    /// The cryptographic suite for this key.
    pub suite: CryptoSuite,
    /// Raw public key bytes.
    pub public_key: Vec<u8>,
}

/// Errors from DID resolution.
#[non_exhaustive]
#[derive(Debug)]
pub enum DidError {
    /// DID method not supported (e.g., did:web without a gateway).
    UnsupportedMethod(String),
    /// DID string is structurally malformed.
    MalformedDid(String),
    /// Base58 or base64url decoding failed.
    DecodingError,
    /// Multicodec prefix not recognized.
    UnknownMulticodec(u16),
}

impl core::fmt::Display for DidError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::UnsupportedMethod(m) => write!(f, "unsupported DID method: {m}"),
            Self::MalformedDid(msg) => write!(f, "malformed DID: {msg}"),
            Self::DecodingError => write!(f, "base encoding/decoding failed"),
            Self::UnknownMulticodec(code) => write!(f, "unknown multicodec: 0x{code:04x}"),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for DidError {}

/// Trait for extensible DID resolution (e.g., did:web via gateway).
pub trait DidResolver {
    fn resolve(&self, did: &str) -> Result<ResolvedDid, DidError>;
}

/// Default resolver using built-in did:key and did:jwk.
pub struct DefaultDidResolver;

impl DidResolver for DefaultDidResolver {
    fn resolve(&self, did: &str) -> Result<ResolvedDid, DidError> {
        resolve_did(did)
    }
}

/// Resolve a DID string to key material.
///
/// Supports `did:key` natively (no_std). `did:jwk` requires the `std` feature.
/// Returns `UnsupportedMethod` for `did:web` and unknown methods.
pub fn resolve_did(did: &str) -> Result<ResolvedDid, DidError> {
    todo!()
}
```

- [ ] **Step 4: Wire up in lib.rs**

Add to `crates/harmony-identity/src/lib.rs`:
```rust
pub mod did;

pub use did::{resolve_did, DefaultDidResolver, DidError, DidResolver, ResolvedDid};
```

- [ ] **Step 5: Verify compilation**

Run: `cargo check -p harmony-identity 2>&1 | tail -5`
Expected: Compiles (todo! is unused in non-test)

- [ ] **Step 6: Commit**

```bash
git add Cargo.toml Cargo.lock crates/harmony-identity/
git commit -m "feat: scaffold DID resolution module with types and trait"
```

---

### Task 2: Implement did:key resolution

**Files:**
- Modify: `crates/harmony-identity/src/did.rs`

- [ ] **Step 1: Implement resolve_did and resolve_did_key**

Replace the `todo!()` in `resolve_did` and add `resolve_did_key`:

```rust
pub fn resolve_did(did: &str) -> Result<ResolvedDid, DidError> {
    if let Some(rest) = did.strip_prefix("did:key:") {
        resolve_did_key(rest)
    } else if did.starts_with("did:jwk:") {
        #[cfg(feature = "std")]
        {
            let rest = &did["did:jwk:".len()..];
            resolve_did_jwk(rest)
        }
        #[cfg(not(feature = "std"))]
        Err(DidError::UnsupportedMethod("jwk (requires std)".into()))
    } else if did.starts_with("did:web:") {
        Err(DidError::UnsupportedMethod("web".into()))
    } else {
        let method = did
            .strip_prefix("did:")
            .and_then(|s| s.split(':').next())
            .unwrap_or("unknown")
            .to_string();
        Err(DidError::UnsupportedMethod(method))
    }
}

/// Resolve a did:key identifier to key material.
///
/// Decodes the multibase (base58btc) payload, reads the LEB128
/// multicodec prefix, and maps to CryptoSuite. This is the inverse
/// of `harmony_credential::jsonld::identity_to_did_key`.
///
/// Note: `0x1211` decodes as `MlDsa65` (not `MlDsa65Rotatable`) —
/// the round-trip is intentionally lossy.
pub fn resolve_did_key(encoded: &str) -> Result<ResolvedDid, DidError> {
    // Multibase prefix 'z' = base58btc
    let encoded = encoded
        .strip_prefix('z')
        .ok_or_else(|| DidError::MalformedDid("did:key must use multibase 'z' (base58btc)".into()))?;

    let bytes = bs58::decode(encoded)
        .into_vec()
        .map_err(|_| DidError::DecodingError)?;

    if bytes.is_empty() {
        return Err(DidError::MalformedDid("empty key payload".into()));
    }

    // Decode LEB128 varint multicodec prefix
    let (multicodec, key_start) = decode_varint(&bytes)?;

    let suite = CryptoSuite::from_signing_multicodec(multicodec as u16)
        .ok_or(DidError::UnknownMulticodec(multicodec as u16))?;

    let public_key = bytes[key_start..].to_vec();

    // Validate key length
    let expected_len = match suite {
        CryptoSuite::Ed25519 => 32,
        CryptoSuite::MlDsa65 | CryptoSuite::MlDsa65Rotatable => 1952,
    };
    if public_key.len() != expected_len {
        return Err(DidError::MalformedDid(
            alloc::format!("expected {} key bytes, got {}", expected_len, public_key.len()),
        ));
    }

    Ok(ResolvedDid { suite, public_key })
}

/// Decode an unsigned LEB128 varint, returning (value, bytes_consumed).
fn decode_varint(bytes: &[u8]) -> Result<(u32, usize), DidError> {
    let mut value: u32 = 0;
    let mut shift: u32 = 0;
    for (i, &byte) in bytes.iter().enumerate() {
        value |= ((byte & 0x7F) as u32) << shift;
        if byte & 0x80 == 0 {
            return Ok((value, i + 1));
        }
        shift += 7;
        if shift >= 32 {
            return Err(DidError::MalformedDid("varint overflow".into()));
        }
    }
    Err(DidError::MalformedDid("truncated varint".into()))
}
```

- [ ] **Step 2: Add tests**

Add `#[cfg(test)] mod tests` at the bottom of `did.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn did_key_ed25519_round_trip() {
        // Encode a known Ed25519 key as did:key, then decode it back
        let key = [0x42u8; 32];
        let suite = CryptoSuite::Ed25519;
        let multicodec = suite.signing_multicodec(); // 0x00ed

        // LEB128 encode
        let mut varint = Vec::new();
        let mut v = multicodec as u32;
        loop {
            let mut byte = (v & 0x7F) as u8;
            v >>= 7;
            if v != 0 { byte |= 0x80; }
            varint.push(byte);
            if v == 0 { break; }
        }

        let mut payload = Vec::new();
        payload.extend_from_slice(&varint);
        payload.extend_from_slice(&key);

        let encoded = alloc::format!("z{}", bs58::encode(&payload).into_string());
        let resolved = resolve_did_key(&encoded).unwrap();

        assert_eq!(resolved.suite, CryptoSuite::Ed25519);
        assert_eq!(resolved.public_key, key);
    }

    #[test]
    fn did_key_ml_dsa65_round_trip() {
        let key = [0x55u8; 1952];
        let suite = CryptoSuite::MlDsa65;
        let multicodec = suite.signing_multicodec(); // 0x1211

        let mut varint = Vec::new();
        let mut v = multicodec as u32;
        loop {
            let mut byte = (v & 0x7F) as u8;
            v >>= 7;
            if v != 0 { byte |= 0x80; }
            varint.push(byte);
            if v == 0 { break; }
        }

        let mut payload = Vec::new();
        payload.extend_from_slice(&varint);
        payload.extend_from_slice(&key);

        let encoded = alloc::format!("z{}", bs58::encode(&payload).into_string());
        let resolved = resolve_did_key(&encoded).unwrap();

        assert_eq!(resolved.suite, CryptoSuite::MlDsa65);
        assert_eq!(resolved.public_key, key);
    }

    #[test]
    fn did_key_unknown_multicodec() {
        // Varint 0xFFFF = [0xFF, 0xFF, 0x03]
        let payload = [0xFF, 0xFF, 0x03, 0x00];
        let encoded = alloc::format!("z{}", bs58::encode(&payload).into_string());
        assert!(matches!(
            resolve_did_key(&encoded),
            Err(DidError::UnknownMulticodec(_))
        ));
    }

    #[test]
    fn did_key_missing_z_prefix() {
        assert!(matches!(
            resolve_did_key("notZprefixed"),
            Err(DidError::MalformedDid(_))
        ));
    }

    #[test]
    fn did_key_bad_base58() {
        assert!(matches!(
            resolve_did_key("z!!!invalid!!!"),
            Err(DidError::DecodingError)
        ));
    }

    #[test]
    fn did_key_wrong_key_length() {
        // Ed25519 multicodec (0x00ed) but only 16 bytes of key
        let payload = [0xed, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                       0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                       0x00, 0x00];
        let encoded = alloc::format!("z{}", bs58::encode(&payload).into_string());
        assert!(matches!(
            resolve_did_key(&encoded),
            Err(DidError::MalformedDid(_))
        ));
    }

    #[test]
    fn resolve_did_web_unsupported() {
        assert!(matches!(
            resolve_did("did:web:example.com"),
            Err(DidError::UnsupportedMethod(m)) if m == "web"
        ));
    }

    #[test]
    fn resolve_did_unknown_method() {
        assert!(matches!(
            resolve_did("did:ethr:0x1234"),
            Err(DidError::UnsupportedMethod(m)) if m == "ethr"
        ));
    }

    #[test]
    fn resolve_did_dispatches_to_did_key() {
        let key = [0x42u8; 32];
        let mut varint = Vec::new();
        let mut v = 0x00edu32;
        loop {
            let mut byte = (v & 0x7F) as u8;
            v >>= 7;
            if v != 0 { byte |= 0x80; }
            varint.push(byte);
            if v == 0 { break; }
        }
        let mut payload = Vec::new();
        payload.extend_from_slice(&varint);
        payload.extend_from_slice(&key);
        let did = alloc::format!("did:key:z{}", bs58::encode(&payload).into_string());

        let resolved = resolve_did(&did).unwrap();
        assert_eq!(resolved.suite, CryptoSuite::Ed25519);
        assert_eq!(resolved.public_key, key);
    }

    #[test]
    fn default_resolver_delegates() {
        let resolver = DefaultDidResolver;
        assert!(matches!(
            resolver.resolve("did:web:example.com"),
            Err(DidError::UnsupportedMethod(_))
        ));
    }
}
```

- [ ] **Step 3: Run tests**

Run: `cargo test -p harmony-identity did 2>&1 | tail -15`
Expected: ALL tests pass

- [ ] **Step 4: Commit**

```bash
git add crates/harmony-identity/src/did.rs
git commit -m "feat: did:key resolution with LEB128 varint multicodec decoding"
```

---

### Task 3: Implement did:jwk resolution (std only)

**Files:**
- Modify: `crates/harmony-identity/src/did.rs`

- [ ] **Step 1: Add resolve_did_jwk**

Add behind `#[cfg(feature = "std")]`:

```rust
/// Resolve a did:jwk identifier to key material.
///
/// Decodes the base64url-encoded JWK and extracts the public key.
/// Requires `std` feature for JSON parsing.
#[cfg(feature = "std")]
pub fn resolve_did_jwk(encoded: &str) -> Result<ResolvedDid, DidError> {
    use base64::Engine;

    let json_bytes = base64::engine::general_purpose::URL_SAFE_NO_PAD
        .decode(encoded)
        .map_err(|_| DidError::DecodingError)?;

    let jwk: serde_json::Value = serde_json::from_slice(&json_bytes)
        .map_err(|e| DidError::MalformedDid(alloc::format!("invalid JWK JSON: {e}")))?;

    let kty = jwk.get("kty")
        .and_then(|v| v.as_str())
        .ok_or_else(|| DidError::MalformedDid("missing kty".into()))?;
    let crv = jwk.get("crv")
        .and_then(|v| v.as_str());

    match (kty, crv) {
        ("OKP", Some("Ed25519")) => {
            let x = jwk.get("x")
                .and_then(|v| v.as_str())
                .ok_or_else(|| DidError::MalformedDid("missing x field".into()))?;
            let key_bytes = base64::engine::general_purpose::URL_SAFE_NO_PAD
                .decode(x)
                .map_err(|_| DidError::DecodingError)?;
            if key_bytes.len() != 32 {
                return Err(DidError::MalformedDid(
                    alloc::format!("Ed25519 key must be 32 bytes, got {}", key_bytes.len()),
                ));
            }
            Ok(ResolvedDid {
                suite: CryptoSuite::Ed25519,
                public_key: key_bytes,
            })
        }
        _ => Err(DidError::UnsupportedMethod(
            alloc::format!("unsupported JWK kty={kty} crv={}", crv.unwrap_or("none")),
        )),
    }
}
```

- [ ] **Step 2: Add did:jwk tests**

Add to the `tests` module (these tests require std since they call resolve_did_jwk):

```rust
    #[cfg(feature = "std")]
    mod jwk_tests {
        use super::*;
        use base64::Engine;

        const B64: base64::engine::GeneralPurpose =
            base64::engine::general_purpose::URL_SAFE_NO_PAD;

        #[test]
        fn did_jwk_ed25519() {
            let key = [0x42u8; 32];
            let x = B64.encode(&key);
            let jwk = serde_json::json!({
                "kty": "OKP",
                "crv": "Ed25519",
                "x": x
            });
            let encoded = B64.encode(serde_json::to_vec(&jwk).unwrap());

            let resolved = resolve_did_jwk(&encoded).unwrap();
            assert_eq!(resolved.suite, CryptoSuite::Ed25519);
            assert_eq!(resolved.public_key, key);
        }

        #[test]
        fn did_jwk_via_resolve_did() {
            let key = [0x42u8; 32];
            let x = B64.encode(&key);
            let jwk = serde_json::json!({
                "kty": "OKP",
                "crv": "Ed25519",
                "x": x
            });
            let encoded = B64.encode(serde_json::to_vec(&jwk).unwrap());
            let did = alloc::format!("did:jwk:{encoded}");

            let resolved = resolve_did(&did).unwrap();
            assert_eq!(resolved.suite, CryptoSuite::Ed25519);
        }

        #[test]
        fn did_jwk_malformed_json() {
            let encoded = B64.encode(b"not json");
            assert!(matches!(
                resolve_did_jwk(&encoded),
                Err(DidError::MalformedDid(_))
            ));
        }

        #[test]
        fn did_jwk_missing_kty() {
            let jwk = serde_json::json!({"crv": "Ed25519", "x": "AAAA"});
            let encoded = B64.encode(serde_json::to_vec(&jwk).unwrap());
            assert!(matches!(
                resolve_did_jwk(&encoded),
                Err(DidError::MalformedDid(_))
            ));
        }

        #[test]
        fn did_jwk_unsupported_curve() {
            let jwk = serde_json::json!({"kty": "EC", "crv": "P-256", "x": "AA", "y": "BB"});
            let encoded = B64.encode(serde_json::to_vec(&jwk).unwrap());
            assert!(matches!(
                resolve_did_jwk(&encoded),
                Err(DidError::UnsupportedMethod(_))
            ));
        }
    }
```

- [ ] **Step 3: Run all tests**

Run: `cargo test -p harmony-identity did 2>&1 | tail -15`
Expected: ALL tests pass (did:key + did:jwk)

Run: `cargo check -p harmony-identity --no-default-features 2>&1 | tail -3`
Expected: Compiles without std (did:jwk gated)

- [ ] **Step 4: Commit**

```bash
git add crates/harmony-identity/src/did.rs
git commit -m "feat: did:jwk resolution with JWK Ed25519 key extraction

Closes harmony-95u"
```
