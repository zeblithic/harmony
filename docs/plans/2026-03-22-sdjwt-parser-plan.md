# SD-JWT Parser Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** New `harmony-sdjwt` crate that parses SD-JWT compact serialization (RFC 9901) and verifies JWS signatures via `harmony-identity`.

**Architecture:** Split compact serialization on `~`, decode JWS `header.payload.signature` via base64url, extract disclosures as JSON arrays. Preserve raw signing input for lossless signature verification. Gate `serde_json` behind `std` feature.

**Tech Stack:** Rust, serde, serde_json (optional/std), base64 (no_std), harmony-identity (verify_signature)

---

### File Structure

| File | Responsibility | Change |
|------|---------------|--------|
| `Cargo.toml` (workspace root) | Workspace members + deps | Add `harmony-sdjwt` to members, add `serde_json` to workspace deps |
| `crates/harmony-sdjwt/Cargo.toml` | Crate manifest | Create |
| `crates/harmony-sdjwt/src/lib.rs` | Crate root, re-exports | Create |
| `crates/harmony-sdjwt/src/error.rs` | `SdJwtError` enum | Create |
| `crates/harmony-sdjwt/src/types.rs` | `SdJwt`, `JwsHeader`, `JwtPayload`, `Disclosure` structs | Create |
| `crates/harmony-sdjwt/src/parse.rs` | `parse` + `signing_input` functions | Create |
| `crates/harmony-sdjwt/src/verify.rs` | `verify` function | Create |

---

### Task 1: Scaffold crate and error types

**Files:**
- Modify: `Cargo.toml` (workspace root)
- Create: `crates/harmony-sdjwt/Cargo.toml`
- Create: `crates/harmony-sdjwt/src/lib.rs`
- Create: `crates/harmony-sdjwt/src/error.rs`

- [ ] **Step 1: Add crate to workspace and create Cargo.toml**

Add `"crates/harmony-sdjwt"` to `[workspace] members` in root `Cargo.toml`.

Add `serde_json` to `[workspace.dependencies]` if not already present:
```toml
serde_json = { version = "1", default-features = false }
```

Create `crates/harmony-sdjwt/Cargo.toml`:

```toml
[package]
name = "harmony-sdjwt"
description = "SD-JWT (RFC 9901) parser and JWS signature verification"
version.workspace = true
edition.workspace = true
license.workspace = true
repository.workspace = true
rust-version.workspace = true

[dependencies]
harmony-identity = { workspace = true }
serde = { workspace = true, default-features = false, features = ["derive", "alloc"] }
serde_json = { workspace = true, optional = true }
base64 = { version = "0.22", default-features = false, features = ["alloc"] }

[features]
default = ["std"]
std = ["dep:serde_json", "harmony-identity/std", "serde/std", "base64/std"]

[dev-dependencies]
serde_json = { workspace = true }
rand = { workspace = true }
```

- [ ] **Step 2: Create error.rs**

Create `crates/harmony-sdjwt/src/error.rs`:

```rust
use alloc::string::String;

/// Errors from SD-JWT parsing and verification.
#[derive(Debug)]
pub enum SdJwtError {
    /// Input string is empty.
    EmptyInput,
    /// JWS compact serialization missing required segments (header.payload.signature).
    MalformedCompact,
    /// Base64url decoding failed.
    Base64Error,
    /// JSON parsing failed.
    JsonError(String),
    /// JWS header missing required "alg" field.
    MissingAlgorithm,
    /// JWS algorithm not supported for verification.
    UnsupportedAlgorithm(String),
    /// Disclosure is not a valid JSON array of 2 or 3 elements.
    InvalidDisclosure,
    /// JWS signature verification failed.
    SignatureInvalid(harmony_identity::IdentityError),
}

impl core::fmt::Display for SdJwtError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::EmptyInput => write!(f, "empty input"),
            Self::MalformedCompact => write!(f, "malformed JWS compact serialization"),
            Self::Base64Error => write!(f, "base64url decoding failed"),
            Self::JsonError(msg) => write!(f, "JSON parse error: {msg}"),
            Self::MissingAlgorithm => write!(f, "JWS header missing 'alg' field"),
            Self::UnsupportedAlgorithm(alg) => write!(f, "unsupported JWS algorithm: {alg}"),
            Self::InvalidDisclosure => write!(f, "disclosure is not a valid [salt, name?, value] array"),
            Self::SignatureInvalid(e) => write!(f, "signature verification failed: {e}"),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for SdJwtError {}
```

- [ ] **Step 3: Create lib.rs**

Create `crates/harmony-sdjwt/src/lib.rs`:

```rust
#![cfg_attr(not(feature = "std"), no_std)]
extern crate alloc;

pub mod error;
pub mod parse;
pub mod types;
pub mod verify;

pub use error::SdJwtError;
pub use parse::{parse, signing_input};
pub use types::{Disclosure, JwsHeader, JwtPayload, SdJwt};
pub use verify::verify;
```

- [ ] **Step 4: Verify compilation**

Run: `cargo check -p harmony-sdjwt 2>&1 | tail -5`
Expected: Errors about missing modules (parse, types, verify) — that's fine for now.

- [ ] **Step 5: Commit**

```bash
git add Cargo.toml Cargo.lock crates/harmony-sdjwt/
git commit -m "feat: scaffold harmony-sdjwt crate with error types"
```

---

### Task 2: Define types and implement parse

**Files:**
- Create: `crates/harmony-sdjwt/src/types.rs`
- Create: `crates/harmony-sdjwt/src/parse.rs`

- [ ] **Step 1: Create types.rs**

```rust
use alloc::string::String;
use alloc::vec::Vec;

/// Decoded JWS header.
#[derive(Debug, Clone)]
pub struct JwsHeader {
    /// Signature algorithm (e.g., "EdDSA", "ES256").
    pub alg: String,
    /// Token type (e.g., "sd-jwt").
    pub typ: Option<String>,
    /// Key identifier (e.g., "did:key:z6Mk..." or "did:web:example.com#key-1").
    pub kid: Option<String>,
}

/// JWT payload with SD-JWT extensions.
#[derive(Debug, Clone)]
pub struct JwtPayload {
    /// Issuer.
    pub iss: Option<String>,
    /// Subject.
    pub sub: Option<String>,
    /// Issued-at (Unix epoch seconds).
    pub iat: Option<u64>,
    /// Expiration (Unix epoch seconds).
    pub exp: Option<u64>,
    /// Not-before (Unix epoch seconds).
    pub nbf: Option<u64>,
    /// Top-level `_sd` array: base64url-encoded digests of selectively
    /// disclosable claims. Nested `_sd` arrays in sub-objects are
    /// captured in `extra`.
    pub sd_digests: Vec<String>,
    /// Hash algorithm for disclosure digests (defaults to "sha-256").
    pub sd_alg: Option<String>,
    /// All other claims as key-value pairs (requires `std` feature
    /// for `serde_json::Value`).
    #[cfg(feature = "std")]
    pub extra: Vec<(String, serde_json::Value)>,
}

/// A decoded SD-JWT disclosure.
#[derive(Debug, Clone)]
pub struct Disclosure {
    /// Original base64url-encoded string (needed for digest verification).
    pub raw: String,
    /// Random salt.
    pub salt: String,
    /// Claim name (None for array element disclosures).
    pub claim_name: Option<String>,
    /// Claim value as raw JSON bytes.
    pub claim_value: Vec<u8>,
    /// Claim value as serde_json::Value (requires `std` feature).
    #[cfg(feature = "std")]
    pub value: serde_json::Value,
}

/// A parsed SD-JWT with all decoded components.
#[derive(Debug, Clone)]
pub struct SdJwt {
    /// Decoded JWS header.
    pub header: JwsHeader,
    /// Decoded JWT payload.
    pub payload: JwtPayload,
    /// Raw signature bytes.
    pub signature: Vec<u8>,
    /// Decoded disclosures.
    pub disclosures: Vec<Disclosure>,
    /// Raw `base64url(header).base64url(payload)` string preserved
    /// for lossless signature verification.
    pub signing_input: String,
}
```

- [ ] **Step 2: Create parse.rs with tests**

```rust
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use base64::Engine;

use crate::error::SdJwtError;
use crate::types::{Disclosure, JwsHeader, JwtPayload, SdJwt};

const B64: base64::engine::GeneralPurpose = base64::engine::general_purpose::URL_SAFE_NO_PAD;

/// Extract the JWS signing input (`header.payload`) from compact
/// serialization without decoding. Returns (header_b64, payload_b64).
pub fn signing_input(compact: &str) -> Result<(&str, &str), SdJwtError> {
    if compact.is_empty() {
        return Err(SdJwtError::EmptyInput);
    }
    // Take everything before the first ~ (the JWS part)
    let jws = compact.split('~').next().unwrap();
    let mut parts = jws.splitn(3, '.');
    let header = parts.next().ok_or(SdJwtError::MalformedCompact)?;
    let payload = parts.next().ok_or(SdJwtError::MalformedCompact)?;
    // Verify signature segment exists
    let _ = parts.next().ok_or(SdJwtError::MalformedCompact)?;
    Ok((header, payload))
}

/// Parse an SD-JWT compact serialization string.
///
/// Splits on `~` to separate JWS from disclosures. Decodes header,
/// payload, and signature via base64url. Preserves the raw signing
/// input for lossless signature verification.
#[cfg(feature = "std")]
pub fn parse(compact: &str) -> Result<SdJwt, SdJwtError> {
    if compact.is_empty() {
        return Err(SdJwtError::EmptyInput);
    }

    let segments: Vec<&str> = compact.split('~').collect();

    // First segment is the JWS (header.payload.signature)
    let jws = segments[0];
    let jws_parts: Vec<&str> = jws.splitn(3, '.').collect();
    if jws_parts.len() != 3 {
        return Err(SdJwtError::MalformedCompact);
    }

    let header_b64 = jws_parts[0];
    let payload_b64 = jws_parts[1];
    let sig_b64 = jws_parts[2];

    // Preserve raw signing input
    let signing_input = alloc::format!("{}.{}", header_b64, payload_b64);

    // Decode header
    let header_bytes = B64.decode(header_b64).map_err(|_| SdJwtError::Base64Error)?;
    let header_json: serde_json::Value =
        serde_json::from_slice(&header_bytes).map_err(|e| SdJwtError::JsonError(e.to_string()))?;

    let alg = header_json
        .get("alg")
        .and_then(|v| v.as_str())
        .ok_or(SdJwtError::MissingAlgorithm)?
        .to_string();
    let typ = header_json.get("typ").and_then(|v| v.as_str()).map(String::from);
    let kid = header_json.get("kid").and_then(|v| v.as_str()).map(String::from);

    let header = JwsHeader { alg, typ, kid };

    // Decode payload
    let payload_bytes = B64.decode(payload_b64).map_err(|_| SdJwtError::Base64Error)?;
    let payload_json: serde_json::Value =
        serde_json::from_slice(&payload_bytes).map_err(|e| SdJwtError::JsonError(e.to_string()))?;

    let iss = payload_json.get("iss").and_then(|v| v.as_str()).map(String::from);
    let sub = payload_json.get("sub").and_then(|v| v.as_str()).map(String::from);
    let iat = payload_json.get("iat").and_then(|v| v.as_u64());
    let exp = payload_json.get("exp").and_then(|v| v.as_u64());
    let nbf = payload_json.get("nbf").and_then(|v| v.as_u64());

    let sd_digests = payload_json
        .get("_sd")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str().map(String::from))
                .collect()
        })
        .unwrap_or_default();

    let sd_alg = payload_json.get("_sd_alg").and_then(|v| v.as_str()).map(String::from);

    // Collect extra claims (everything except known JWT/SD-JWT fields)
    let known_keys = ["iss", "sub", "iat", "exp", "nbf", "_sd", "_sd_alg"];
    let extra: Vec<(String, serde_json::Value)> = payload_json
        .as_object()
        .map(|obj| {
            obj.iter()
                .filter(|(k, _)| !known_keys.contains(&k.as_str()))
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect()
        })
        .unwrap_or_default();

    let payload = JwtPayload {
        iss, sub, iat, exp, nbf, sd_digests, sd_alg, extra,
    };

    // Decode signature
    let signature = B64.decode(sig_b64).map_err(|_| SdJwtError::Base64Error)?;

    // Decode disclosures (segments 1..N, skip empty strings from trailing ~)
    let mut disclosures = Vec::new();
    for &segment in &segments[1..] {
        if segment.is_empty() {
            continue;
        }
        let disc_bytes = B64.decode(segment).map_err(|_| SdJwtError::Base64Error)?;
        let disc_json: serde_json::Value =
            serde_json::from_slice(&disc_bytes).map_err(|e| SdJwtError::JsonError(e.to_string()))?;

        let arr = disc_json.as_array().ok_or(SdJwtError::InvalidDisclosure)?;

        let (salt, claim_name, claim_value_json) = match arr.len() {
            // [salt, value] — array element disclosure
            2 => {
                let salt = arr[0].as_str().ok_or(SdJwtError::InvalidDisclosure)?;
                (salt.to_string(), None, &arr[1])
            }
            // [salt, name, value] — object property disclosure
            3 => {
                let salt = arr[0].as_str().ok_or(SdJwtError::InvalidDisclosure)?;
                let name = arr[1].as_str().ok_or(SdJwtError::InvalidDisclosure)?;
                (salt.to_string(), Some(name.to_string()), &arr[2])
            }
            _ => return Err(SdJwtError::InvalidDisclosure),
        };

        let claim_value = serde_json::to_vec(claim_value_json)
            .map_err(|e| SdJwtError::JsonError(e.to_string()))?;

        disclosures.push(Disclosure {
            raw: segment.to_string(),
            salt,
            claim_name,
            claim_value,
            value: claim_value_json.clone(),
        });
    }

    Ok(SdJwt {
        header,
        payload,
        signature,
        disclosures,
        signing_input,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: base64url-encode a JSON string.
    fn b64(json: &str) -> String {
        B64.encode(json.as_bytes())
    }

    /// Helper: build a minimal JWS compact serialization.
    fn make_jws(header: &str, payload: &str) -> String {
        let sig = B64.encode(b"fake-signature");
        alloc::format!("{}.{}.{}", b64(header), b64(payload), sig)
    }

    #[test]
    fn parse_jws_without_disclosures() {
        let compact = make_jws(r#"{"alg":"EdDSA"}"#, r#"{"iss":"did:key:z123"}"#);
        let sd_jwt = parse(&compact).unwrap();
        assert_eq!(sd_jwt.header.alg, "EdDSA");
        assert_eq!(sd_jwt.payload.iss.as_deref(), Some("did:key:z123"));
        assert!(sd_jwt.disclosures.is_empty());
    }

    #[test]
    fn parse_with_disclosures() {
        let jws = make_jws(r#"{"alg":"EdDSA"}"#, r#"{"_sd":["abc"]}"#);
        let disc1 = b64(r#"["salt1","given_name","Alice"]"#);
        let disc2 = b64(r#"["salt2","family_name","Smith"]"#);
        let compact = alloc::format!("{}~{}~{}~", jws, disc1, disc2);

        let sd_jwt = parse(&compact).unwrap();
        assert_eq!(sd_jwt.disclosures.len(), 2);
        assert_eq!(sd_jwt.disclosures[0].salt, "salt1");
        assert_eq!(sd_jwt.disclosures[0].claim_name.as_deref(), Some("given_name"));
        assert_eq!(sd_jwt.disclosures[1].claim_name.as_deref(), Some("family_name"));
    }

    #[test]
    fn parse_array_element_disclosure() {
        let jws = make_jws(r#"{"alg":"EdDSA"}"#, r#"{"_sd":[]}"#);
        let disc = b64(r#"["salt1","value_only"]"#);
        let compact = alloc::format!("{}~{}", jws, disc);

        let sd_jwt = parse(&compact).unwrap();
        assert_eq!(sd_jwt.disclosures.len(), 1);
        assert!(sd_jwt.disclosures[0].claim_name.is_none());
    }

    #[test]
    fn parse_extracts_header_fields() {
        let compact = make_jws(
            r#"{"alg":"EdDSA","typ":"sd-jwt","kid":"did:key:z6Mk#key-1"}"#,
            r#"{}"#,
        );
        let sd_jwt = parse(&compact).unwrap();
        assert_eq!(sd_jwt.header.alg, "EdDSA");
        assert_eq!(sd_jwt.header.typ.as_deref(), Some("sd-jwt"));
        assert_eq!(sd_jwt.header.kid.as_deref(), Some("did:key:z6Mk#key-1"));
    }

    #[test]
    fn parse_extracts_payload_fields() {
        let compact = make_jws(
            r#"{"alg":"EdDSA"}"#,
            r#"{"iss":"alice","sub":"bob","iat":1000,"exp":5000,"nbf":900,"_sd":["d1","d2"],"_sd_alg":"sha-256","custom":"value"}"#,
        );
        let sd_jwt = parse(&compact).unwrap();
        assert_eq!(sd_jwt.payload.iss.as_deref(), Some("alice"));
        assert_eq!(sd_jwt.payload.sub.as_deref(), Some("bob"));
        assert_eq!(sd_jwt.payload.iat, Some(1000));
        assert_eq!(sd_jwt.payload.exp, Some(5000));
        assert_eq!(sd_jwt.payload.nbf, Some(900));
        assert_eq!(sd_jwt.payload.sd_digests, vec!["d1", "d2"]);
        assert_eq!(sd_jwt.payload.sd_alg.as_deref(), Some("sha-256"));
        assert_eq!(sd_jwt.payload.extra.len(), 1);
        assert_eq!(sd_jwt.payload.extra[0].0, "custom");
    }

    #[test]
    fn parse_preserves_signing_input() {
        let header_b64 = b64(r#"{"alg":"EdDSA"}"#);
        let payload_b64 = b64(r#"{"iss":"alice"}"#);
        let sig_b64 = B64.encode(b"sig");
        let compact = alloc::format!("{}.{}.{}", header_b64, payload_b64, sig_b64);

        let sd_jwt = parse(&compact).unwrap();
        assert_eq!(sd_jwt.signing_input, alloc::format!("{}.{}", header_b64, payload_b64));
    }

    #[test]
    fn parse_empty_input_rejected() {
        assert!(matches!(parse(""), Err(SdJwtError::EmptyInput)));
    }

    #[test]
    fn parse_missing_signature_rejected() {
        let compact = alloc::format!("{}.{}", b64(r#"{"alg":"EdDSA"}"#), b64(r#"{}"#));
        assert!(matches!(parse(&compact), Err(SdJwtError::MalformedCompact)));
    }

    #[test]
    fn parse_invalid_base64_rejected() {
        assert!(matches!(parse("!!!.@@@.###"), Err(SdJwtError::Base64Error)));
    }

    #[test]
    fn parse_invalid_header_json_rejected() {
        let bad_header = B64.encode(b"not json");
        let payload = b64(r#"{}"#);
        let sig = B64.encode(b"sig");
        let compact = alloc::format!("{}.{}.{}", bad_header, payload, sig);
        assert!(matches!(parse(&compact), Err(SdJwtError::JsonError(_))));
    }

    #[test]
    fn parse_missing_alg_rejected() {
        let compact = make_jws(r#"{"typ":"jwt"}"#, r#"{}"#);
        assert!(matches!(parse(&compact), Err(SdJwtError::MissingAlgorithm)));
    }

    #[test]
    fn parse_invalid_disclosure_rejected() {
        let jws = make_jws(r#"{"alg":"EdDSA"}"#, r#"{}"#);
        let bad_disc = b64(r#""not an array""#);
        let compact = alloc::format!("{}~{}", jws, bad_disc);
        assert!(matches!(parse(&compact), Err(SdJwtError::InvalidDisclosure)));
    }

    #[test]
    fn signing_input_extracts_correctly() {
        let header_b64 = b64(r#"{"alg":"EdDSA"}"#);
        let payload_b64 = b64(r#"{}"#);
        let sig_b64 = B64.encode(b"sig");
        let compact = alloc::format!("{}.{}.{}~disc~", header_b64, payload_b64, sig_b64);

        let (h, p) = signing_input(&compact).unwrap();
        assert_eq!(h, header_b64);
        assert_eq!(p, payload_b64);
    }
}
```

- [ ] **Step 3: Verify tests pass**

Run: `cargo test -p harmony-sdjwt --features std 2>&1 | tail -15`
Expected: ALL parse tests pass

- [ ] **Step 4: Commit**

```bash
git add crates/harmony-sdjwt/src/types.rs crates/harmony-sdjwt/src/parse.rs
git commit -m "feat: SD-JWT compact serialization parser with tests"
```

---

### Task 3: Implement JWS signature verification

**Files:**
- Create: `crates/harmony-sdjwt/src/verify.rs`

- [ ] **Step 1: Write verify.rs with tests**

```rust
use harmony_identity::{CryptoSuite, IdentityError};

use crate::error::SdJwtError;
use crate::types::SdJwt;

/// Map JWS algorithm name to Harmony CryptoSuite.
fn alg_to_suite(alg: &str) -> Result<CryptoSuite, SdJwtError> {
    match alg {
        "EdDSA" => Ok(CryptoSuite::Ed25519),
        "MLDSA65" => Ok(CryptoSuite::MlDsa65),
        other => Err(SdJwtError::UnsupportedAlgorithm(
            alloc::string::String::from(other),
        )),
    }
}

/// Verify the JWS signature of a parsed SD-JWT.
///
/// Uses the preserved `signing_input` (raw `header.payload` base64url
/// string) as the message — no lossy re-encoding. The caller is
/// responsible for resolving the correct public key via DID resolution.
///
/// `MLDSA65` maps to `CryptoSuite::MlDsa65` (not `MlDsa65Rotatable`).
/// Rotation awareness is out of scope for this crate.
pub fn verify(
    sd_jwt: &SdJwt,
    suite: CryptoSuite,
    public_key: &[u8],
) -> Result<(), SdJwtError> {
    harmony_identity::verify_signature(
        suite,
        public_key,
        sd_jwt.signing_input.as_bytes(),
        &sd_jwt.signature,
    )
    .map_err(SdJwtError::SignatureInvalid)
}

/// Verify using the algorithm from the JWS header.
///
/// Convenience wrapper that maps the `alg` header to a `CryptoSuite`.
pub fn verify_from_header(
    sd_jwt: &SdJwt,
    public_key: &[u8],
) -> Result<(), SdJwtError> {
    let suite = alg_to_suite(&sd_jwt.header.alg)?;
    verify(sd_jwt, suite, public_key)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parse::parse;
    use base64::Engine;
    use rand::rngs::OsRng;

    const B64: base64::engine::GeneralPurpose =
        base64::engine::general_purpose::URL_SAFE_NO_PAD;

    fn b64(json: &str) -> alloc::string::String {
        B64.encode(json.as_bytes())
    }

    /// Build a real signed SD-JWT using Ed25519.
    fn make_signed_sdjwt(
        private: &harmony_identity::PrivateIdentity,
        payload_json: &str,
    ) -> alloc::string::String {
        let header_b64 = b64(r#"{"alg":"EdDSA","typ":"sd-jwt"}"#);
        let payload_b64 = b64(payload_json);
        let signing_input = alloc::format!("{}.{}", header_b64, payload_b64);
        let signature = private.sign(signing_input.as_bytes());
        let sig_b64 = B64.encode(&signature);
        alloc::format!("{}.{}", signing_input, sig_b64)
    }

    #[test]
    fn verify_valid_ed25519_signature() {
        let private = harmony_identity::PrivateIdentity::generate(&mut OsRng);
        let identity = private.public_identity();
        let compact = make_signed_sdjwt(&private, r#"{"iss":"alice","sub":"bob"}"#);

        let sd_jwt = parse(&compact).unwrap();
        assert!(verify(
            &sd_jwt,
            CryptoSuite::Ed25519,
            &identity.verifying_key.to_bytes(),
        )
        .is_ok());
    }

    #[test]
    fn verify_rejects_wrong_key() {
        let private = harmony_identity::PrivateIdentity::generate(&mut OsRng);
        let other_private = harmony_identity::PrivateIdentity::generate(&mut OsRng);
        let other_identity = other_private.public_identity();
        let compact = make_signed_sdjwt(&private, r#"{"iss":"alice"}"#);

        let sd_jwt = parse(&compact).unwrap();
        assert!(verify(
            &sd_jwt,
            CryptoSuite::Ed25519,
            &other_identity.verifying_key.to_bytes(),
        )
        .is_err());
    }

    #[test]
    fn verify_from_header_maps_eddsa() {
        let private = harmony_identity::PrivateIdentity::generate(&mut OsRng);
        let identity = private.public_identity();
        let compact = make_signed_sdjwt(&private, r#"{"iss":"alice"}"#);

        let sd_jwt = parse(&compact).unwrap();
        assert!(verify_from_header(&sd_jwt, &identity.verifying_key.to_bytes()).is_ok());
    }

    #[test]
    fn verify_unsupported_algorithm() {
        assert!(matches!(
            alg_to_suite("RS256"),
            Err(SdJwtError::UnsupportedAlgorithm(_))
        ));
    }
}
```

- [ ] **Step 2: Update lib.rs re-exports**

Add to the existing `pub use` block in `lib.rs`:
```rust
pub use verify::{verify, verify_from_header};
```

- [ ] **Step 3: Run all tests**

Run: `cargo test -p harmony-sdjwt --features std 2>&1 | tail -15`
Expected: ALL tests pass (parse + verify)

Run: `cargo check -p harmony-sdjwt --no-default-features 2>&1 | tail -5`
Expected: Compiles without std (parse function gated behind std feature)

- [ ] **Step 4: Commit**

```bash
git add crates/harmony-sdjwt/src/verify.rs crates/harmony-sdjwt/src/lib.rs
git commit -m "feat: JWS signature verification for SD-JWT

Ed25519 and ML-DSA-65 verification via harmony-identity.
Uses preserved signing_input for lossless verification.

Closes harmony-yfk"
```
