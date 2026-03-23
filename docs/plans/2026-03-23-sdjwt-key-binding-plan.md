# SD-JWT Key Binding Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Verify the KB-JWT in SD-JWT presentations to confirm holder binding (RFC 9901 §11.6).

**Architecture:** New `key_binding.rs` module in `harmony-sdjwt` behind `std` feature. Parses the KB-JWT JWS, validates typ/nonce/aud/iat/sd_hash, and verifies the holder's signature. Reconstructs sd_hash from `SdJwt` fields.

**Tech Stack:** Rust, sha2 (SHA-256), base64, serde_json, harmony-identity (verify_signature, CryptoSuite)

---

### File Structure

| File | Responsibility | Change |
|------|---------------|--------|
| `crates/harmony-sdjwt/Cargo.toml` | Crate manifest | Make `sha2` available under `std` feature |
| `crates/harmony-sdjwt/src/key_binding.rs` | KB-JWT verification | Create |
| `crates/harmony-sdjwt/src/error.rs` | Error enum | Add `KeyBindingInvalid` variant |
| `crates/harmony-sdjwt/src/lib.rs` | Module + re-exports | Add `key_binding` module |

---

### Task 1: Add error variant and scaffold module

**Files:**
- Modify: `crates/harmony-sdjwt/Cargo.toml`
- Modify: `crates/harmony-sdjwt/src/error.rs`
- Modify: `crates/harmony-sdjwt/src/lib.rs`
- Create: `crates/harmony-sdjwt/src/key_binding.rs`

- [ ] **Step 1: Make sha2 available under std feature**

In `crates/harmony-sdjwt/Cargo.toml`, add `"dep:sha2"` to the `std` feature array:
```toml
std = ["dep:serde_json", "serde_json?/std", "harmony-identity/std", "base64/std", "dep:sha2"]
```

This makes sha2 available for both `std` and `credential` features (credential already has it).

- [ ] **Step 2: Add KeyBindingInvalid error variant**

In `error.rs`, add:
```rust
    /// Key Binding JWT verification failed.
    #[cfg(feature = "std")]
    KeyBindingInvalid(String),
```

Add Display arm:
```rust
    #[cfg(feature = "std")]
    Self::KeyBindingInvalid(msg) => write!(f, "key binding verification failed: {msg}"),
```

- [ ] **Step 3: Create key_binding.rs stub**

```rust
//! SD-JWT Key Binding (KB-JWT) verification (RFC 9901 §11.6).

use alloc::string::String;
use harmony_identity::CryptoSuite;

use crate::error::SdJwtError;
use crate::types::SdJwt;

/// Verify the Key Binding JWT in an SD-JWT presentation.
///
/// Confirms the holder possesses the private key by verifying the
/// KB-JWT signature, and validates nonce, audience, timestamp, and
/// sd_hash binding.
///
/// # Note
///
/// This function does NOT extract the holder's key from the issuer's
/// `cnf` claim — the caller must provide the raw key bytes and suite.
pub fn verify_key_binding(
    _sd_jwt: &SdJwt,
    _holder_key: &[u8],
    _holder_suite: CryptoSuite,
    _expected_nonce: &str,
    _expected_aud: &str,
    _now: u64,
) -> Result<(), SdJwtError> {
    todo!()
}
```

- [ ] **Step 4: Wire up in lib.rs**

Add:
```rust
#[cfg(feature = "std")]
pub mod key_binding;

#[cfg(feature = "std")]
pub use key_binding::verify_key_binding;
```

- [ ] **Step 5: Verify compilation**

Run: `cargo check -p harmony-sdjwt --features std 2>&1 | tail -5`

- [ ] **Step 6: Commit**

```bash
git add crates/harmony-sdjwt/
git commit -m "feat: scaffold key_binding module with KeyBindingInvalid error"
```

---

### Task 2: Implement verify_key_binding

**Files:**
- Modify: `crates/harmony-sdjwt/src/key_binding.rs`

- [ ] **Step 1: Implement the full function with tests**

Replace `key_binding.rs` with full implementation:

```rust
//! SD-JWT Key Binding (KB-JWT) verification (RFC 9901 §11.6).

use alloc::format;
use alloc::string::{String, ToString};
use base64::Engine;
use sha2::{Digest, Sha256};

use harmony_identity::CryptoSuite;

use crate::error::SdJwtError;
use crate::types::SdJwt;

const B64: base64::engine::GeneralPurpose =
    base64::engine::general_purpose::URL_SAFE_NO_PAD;

/// Maximum clock skew allowed for `iat` validation (seconds).
const MAX_CLOCK_SKEW: u64 = 60;

/// Verify the Key Binding JWT in an SD-JWT presentation.
///
/// Confirms the holder possesses the private key by verifying the
/// KB-JWT signature, and validates nonce, audience, timestamp, and
/// sd_hash binding.
///
/// # Note
///
/// This function does NOT extract the holder's key from the issuer's
/// `cnf` claim — the caller must provide the raw key bytes and suite.
/// It also does NOT validate time-based issuer claims (exp/nbf) —
/// that remains the caller's responsibility.
pub fn verify_key_binding(
    sd_jwt: &SdJwt,
    holder_key: &[u8],
    holder_suite: CryptoSuite,
    expected_nonce: &str,
    expected_aud: &str,
    now: u64,
) -> Result<(), SdJwtError> {
    // 1. Check KB-JWT is present
    let kb_jwt_str = sd_jwt.key_binding_jwt.as_deref()
        .ok_or_else(|| SdJwtError::KeyBindingInvalid(
            "no key binding JWT present".into()
        ))?;

    // 2. Split KB-JWT into header.payload.signature
    let parts: Vec<&str> = kb_jwt_str.splitn(3, '.').collect();
    if parts.len() != 3 {
        return Err(SdJwtError::KeyBindingInvalid(
            "KB-JWT must have 3 dot-separated segments".into()
        ));
    }
    let (kb_header_b64, kb_payload_b64, kb_sig_b64) = (parts[0], parts[1], parts[2]);

    // 3. Parse header
    let header_bytes = B64.decode(kb_header_b64)
        .map_err(|_| SdJwtError::KeyBindingInvalid("KB-JWT header base64 decode failed".into()))?;
    let header_json: serde_json::Value = serde_json::from_slice(&header_bytes)
        .map_err(|e| SdJwtError::KeyBindingInvalid(format!("KB-JWT header JSON: {e}")))?;

    // 4. Verify typ == "kb+jwt" (case-insensitive, accept application/ prefix)
    let typ = header_json.get("typ")
        .and_then(|v| v.as_str())
        .ok_or_else(|| SdJwtError::KeyBindingInvalid("KB-JWT missing typ".into()))?;
    if !typ.eq_ignore_ascii_case("kb+jwt")
        && !typ.eq_ignore_ascii_case("application/kb+jwt")
    {
        return Err(SdJwtError::KeyBindingInvalid(
            format!("KB-JWT typ must be kb+jwt, got {typ}")
        ));
    }

    // 5. Parse payload
    let payload_bytes = B64.decode(kb_payload_b64)
        .map_err(|_| SdJwtError::KeyBindingInvalid("KB-JWT payload base64 decode failed".into()))?;
    let payload_json: serde_json::Value = serde_json::from_slice(&payload_bytes)
        .map_err(|e| SdJwtError::KeyBindingInvalid(format!("KB-JWT payload JSON: {e}")))?;

    // 6. Verify nonce
    let nonce = payload_json.get("nonce")
        .and_then(|v| v.as_str())
        .ok_or_else(|| SdJwtError::KeyBindingInvalid("KB-JWT missing nonce".into()))?;
    if nonce != expected_nonce {
        return Err(SdJwtError::KeyBindingInvalid(
            format!("nonce mismatch: expected {expected_nonce}, got {nonce}")
        ));
    }

    // 7. Verify aud
    let aud = payload_json.get("aud")
        .and_then(|v| v.as_str())
        .ok_or_else(|| SdJwtError::KeyBindingInvalid("KB-JWT missing aud".into()))?;
    if aud != expected_aud {
        return Err(SdJwtError::KeyBindingInvalid(
            format!("aud mismatch: expected {expected_aud}, got {aud}")
        ));
    }

    // 8. Verify iat not in the future
    let iat = payload_json.get("iat")
        .and_then(|v| v.as_u64())
        .ok_or_else(|| SdJwtError::KeyBindingInvalid("KB-JWT missing or invalid iat".into()))?;
    if iat > now + MAX_CLOCK_SKEW {
        return Err(SdJwtError::KeyBindingInvalid(
            format!("KB-JWT iat is in the future: {iat} > {now}")
        ));
    }

    // 9. Verify sd_hash
    let expected_sd_hash = payload_json.get("sd_hash")
        .and_then(|v| v.as_str())
        .ok_or_else(|| SdJwtError::KeyBindingInvalid("KB-JWT missing sd_hash".into()))?;

    let sd_jwt_without_kb = reconstruct_sd_jwt_without_kb(sd_jwt);
    let computed_hash = Sha256::digest(sd_jwt_without_kb.as_bytes());
    let computed_sd_hash = B64.encode(computed_hash);

    if computed_sd_hash != expected_sd_hash {
        return Err(SdJwtError::KeyBindingInvalid(
            "sd_hash mismatch: KB-JWT does not match this presentation".into()
        ));
    }

    // 10. Verify KB-JWT signature
    let kb_signing_input = format!("{kb_header_b64}.{kb_payload_b64}");
    let kb_signature = B64.decode(kb_sig_b64)
        .map_err(|_| SdJwtError::KeyBindingInvalid("KB-JWT signature base64 decode failed".into()))?;

    harmony_identity::verify_signature(
        holder_suite,
        holder_key,
        kb_signing_input.as_bytes(),
        &kb_signature,
    )
    .map_err(SdJwtError::SignatureInvalid)
}

/// Reconstruct the SD-JWT compact serialization without the KB-JWT.
///
/// Format: `{signing_input}.{base64url(signature)}~{disc1.raw}~...~{discN.raw}~`
fn reconstruct_sd_jwt_without_kb(sd_jwt: &SdJwt) -> String {
    let sig_b64 = B64.encode(&sd_jwt.signature);
    let mut result = format!("{}.{}", sd_jwt.signing_input, sig_b64);

    for disc in &sd_jwt.disclosures {
        result.push('~');
        result.push_str(&disc.raw);
    }
    result.push('~');

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Disclosure, JwsHeader, JwtPayload};
    use rand::rngs::OsRng;

    /// Build a minimal SdJwt for KB testing.
    fn make_sd_jwt_with_kb(
        issuer_private: &harmony_identity::PrivateIdentity,
        holder_private: &harmony_identity::PrivateIdentity,
        nonce: &str,
        aud: &str,
        iat: u64,
    ) -> SdJwt {
        // Build issuer JWS
        let header_b64 = B64.encode(r#"{"alg":"EdDSA","typ":"sd+jwt"}"#.as_bytes());
        let payload_b64 = B64.encode(r#"{"iss":"issuer","sub":"holder"}"#.as_bytes());
        let signing_input = format!("{header_b64}.{payload_b64}");
        let issuer_sig = issuer_private.sign(signing_input.as_bytes());

        // Build the SD-JWT without KB for sd_hash
        let sig_b64 = B64.encode(&issuer_sig);
        let sd_jwt_without_kb = format!("{signing_input}.{sig_b64}~");

        // Compute sd_hash
        let sd_hash = B64.encode(Sha256::digest(sd_jwt_without_kb.as_bytes()));

        // Build KB-JWT
        let kb_header_b64 = B64.encode(r#"{"alg":"EdDSA","typ":"kb+jwt"}"#.as_bytes());
        let kb_payload = serde_json::json!({
            "nonce": nonce,
            "aud": aud,
            "iat": iat,
            "sd_hash": sd_hash
        });
        let kb_payload_b64 = B64.encode(serde_json::to_vec(&kb_payload).unwrap());
        let kb_signing_input = format!("{kb_header_b64}.{kb_payload_b64}");
        let kb_sig = holder_private.sign(kb_signing_input.as_bytes());
        let kb_sig_b64 = B64.encode(&kb_sig);
        let kb_jwt = format!("{kb_signing_input}.{kb_sig_b64}");

        SdJwt {
            header: JwsHeader {
                alg: "EdDSA".into(),
                typ: Some("sd+jwt".into()),
                kid: None,
            },
            payload: JwtPayload {
                iss: Some("issuer".into()),
                sub: Some("holder".into()),
                iat: None,
                exp: None,
                nbf: None,
                sd: vec![],
                sd_alg: None,
                #[cfg(feature = "std")]
                extra: vec![],
            },
            signature: issuer_sig.to_vec(),
            signing_input,
            disclosures: vec![],
            key_binding_jwt: Some(kb_jwt),
        }
    }

    #[test]
    fn valid_key_binding() {
        let issuer = harmony_identity::PrivateIdentity::generate(&mut OsRng);
        let holder = harmony_identity::PrivateIdentity::generate(&mut OsRng);
        let holder_pub = holder.public_identity();

        let sd_jwt = make_sd_jwt_with_kb(&issuer, &holder, "test-nonce", "verifier.example", 1000);

        assert!(verify_key_binding(
            &sd_jwt,
            &holder_pub.verifying_key.to_bytes(),
            CryptoSuite::Ed25519,
            "test-nonce",
            "verifier.example",
            1000,
        ).is_ok());
    }

    #[test]
    fn missing_kb_jwt() {
        let issuer = harmony_identity::PrivateIdentity::generate(&mut OsRng);
        let holder = harmony_identity::PrivateIdentity::generate(&mut OsRng);
        let mut sd_jwt = make_sd_jwt_with_kb(&issuer, &holder, "n", "a", 1000);
        sd_jwt.key_binding_jwt = None;

        assert!(matches!(
            verify_key_binding(&sd_jwt, &[0; 32], CryptoSuite::Ed25519, "n", "a", 1000),
            Err(SdJwtError::KeyBindingInvalid(_))
        ));
    }

    #[test]
    fn wrong_nonce() {
        let issuer = harmony_identity::PrivateIdentity::generate(&mut OsRng);
        let holder = harmony_identity::PrivateIdentity::generate(&mut OsRng);
        let holder_pub = holder.public_identity();
        let sd_jwt = make_sd_jwt_with_kb(&issuer, &holder, "correct-nonce", "aud", 1000);

        assert!(matches!(
            verify_key_binding(
                &sd_jwt,
                &holder_pub.verifying_key.to_bytes(),
                CryptoSuite::Ed25519,
                "wrong-nonce",
                "aud",
                1000,
            ),
            Err(SdJwtError::KeyBindingInvalid(msg)) if msg.contains("nonce")
        ));
    }

    #[test]
    fn wrong_aud() {
        let issuer = harmony_identity::PrivateIdentity::generate(&mut OsRng);
        let holder = harmony_identity::PrivateIdentity::generate(&mut OsRng);
        let holder_pub = holder.public_identity();
        let sd_jwt = make_sd_jwt_with_kb(&issuer, &holder, "n", "correct-aud", 1000);

        assert!(matches!(
            verify_key_binding(
                &sd_jwt,
                &holder_pub.verifying_key.to_bytes(),
                CryptoSuite::Ed25519,
                "n",
                "wrong-aud",
                1000,
            ),
            Err(SdJwtError::KeyBindingInvalid(msg)) if msg.contains("aud")
        ));
    }

    #[test]
    fn future_iat() {
        let issuer = harmony_identity::PrivateIdentity::generate(&mut OsRng);
        let holder = harmony_identity::PrivateIdentity::generate(&mut OsRng);
        let holder_pub = holder.public_identity();
        // iat = 5000, now = 1000 — far in the future
        let sd_jwt = make_sd_jwt_with_kb(&issuer, &holder, "n", "a", 5000);

        assert!(matches!(
            verify_key_binding(
                &sd_jwt,
                &holder_pub.verifying_key.to_bytes(),
                CryptoSuite::Ed25519,
                "n",
                "a",
                1000,
            ),
            Err(SdJwtError::KeyBindingInvalid(msg)) if msg.contains("future")
        ));
    }

    #[test]
    fn wrong_sd_hash() {
        let issuer = harmony_identity::PrivateIdentity::generate(&mut OsRng);
        let holder = harmony_identity::PrivateIdentity::generate(&mut OsRng);
        let holder_pub = holder.public_identity();
        let mut sd_jwt = make_sd_jwt_with_kb(&issuer, &holder, "n", "a", 1000);

        // Tamper with signature to change sd_hash
        sd_jwt.signature = vec![0xFF; 64];

        assert!(matches!(
            verify_key_binding(
                &sd_jwt,
                &holder_pub.verifying_key.to_bytes(),
                CryptoSuite::Ed25519,
                "n",
                "a",
                1000,
            ),
            Err(SdJwtError::KeyBindingInvalid(msg)) if msg.contains("sd_hash")
        ));
    }

    #[test]
    fn wrong_holder_key() {
        let issuer = harmony_identity::PrivateIdentity::generate(&mut OsRng);
        let holder = harmony_identity::PrivateIdentity::generate(&mut OsRng);
        let wrong_key = harmony_identity::PrivateIdentity::generate(&mut OsRng);
        let sd_jwt = make_sd_jwt_with_kb(&issuer, &holder, "n", "a", 1000);

        assert!(matches!(
            verify_key_binding(
                &sd_jwt,
                &wrong_key.public_identity().verifying_key.to_bytes(),
                CryptoSuite::Ed25519,
                "n",
                "a",
                1000,
            ),
            Err(SdJwtError::SignatureInvalid(_))
        ));
    }

    #[test]
    fn wrong_typ() {
        let issuer = harmony_identity::PrivateIdentity::generate(&mut OsRng);
        let holder = harmony_identity::PrivateIdentity::generate(&mut OsRng);
        let holder_pub = holder.public_identity();

        // Build a KB-JWT with wrong typ
        let header_b64 = B64.encode(r#"{"alg":"EdDSA","typ":"sd+jwt"}"#.as_bytes());
        let payload_b64 = B64.encode(r#"{"nonce":"n","aud":"a","iat":1000,"sd_hash":"x"}"#.as_bytes());
        let si = format!("{header_b64}.{payload_b64}");
        let sig = holder.sign(si.as_bytes());
        let kb_jwt = format!("{si}.{}", B64.encode(&sig));

        let mut sd_jwt = make_sd_jwt_with_kb(&issuer, &holder, "n", "a", 1000);
        sd_jwt.key_binding_jwt = Some(kb_jwt);

        assert!(matches!(
            verify_key_binding(
                &sd_jwt,
                &holder_pub.verifying_key.to_bytes(),
                CryptoSuite::Ed25519,
                "n",
                "a",
                1000,
            ),
            Err(SdJwtError::KeyBindingInvalid(msg)) if msg.contains("typ")
        ));
    }
}
```

- [ ] **Step 2: Run tests**

Run: `cargo test -p harmony-sdjwt --features std key_binding 2>&1 | tail -15`
Expected: ALL 8 tests pass

- [ ] **Step 3: Run full test suite**

Run: `cargo test -p harmony-sdjwt --features std 2>&1 | tail -5`
Expected: ALL tests pass (existing + new)

- [ ] **Step 4: Commit**

```bash
git add crates/harmony-sdjwt/src/key_binding.rs
git commit -m "feat: verify_key_binding — KB-JWT holder verification per RFC 9901 §11.6

Validates typ, nonce, aud, iat, sd_hash, and holder signature.
Reconstructs sd_hash from SdJwt fields for presentation binding.

Closes harmony-lth"
```
