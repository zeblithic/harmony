# SD-JWT Disclosure Verification and Claim Mapping Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Verify SD-JWT disclosure hashes against signed `_sd` digests, then map verified disclosures to Harmony's native `SaltedClaim` type.

**Architecture:** New `claims.rs` module in `harmony-sdjwt` behind a `credential` feature flag. Two functions: `verify_disclosures` (SHA-256 hash check per RFC 9901 §6.3) and `map_claims` (static vocabulary dictionary + hash-derived fallback for `type_id`).

**Tech Stack:** Rust, sha2 (SHA-256), base64 (base64url encoding), harmony-credential (SaltedClaim), harmony-crypto (BLAKE3 for type_id derivation)

---

### File Structure

| File | Responsibility | Change |
|------|---------------|--------|
| `crates/harmony-sdjwt/Cargo.toml` | Crate manifest | Add `credential` feature, `sha2` + `harmony-credential` deps |
| `crates/harmony-sdjwt/src/claims.rs` | Disclosure verification + claim mapping | Create |
| `crates/harmony-sdjwt/src/error.rs` | Error enum | Add `DisclosureHashMismatch` variant |
| `crates/harmony-sdjwt/src/lib.rs` | Module declarations | Add `claims` module behind `credential` feature |

---

### Task 1: Add `credential` feature and `DisclosureHashMismatch` error

**Files:**
- Modify: `crates/harmony-sdjwt/Cargo.toml`
- Modify: `crates/harmony-sdjwt/src/error.rs`
- Modify: `crates/harmony-sdjwt/src/lib.rs`

- [ ] **Step 1: Update Cargo.toml**

Add dependencies and feature:

```toml
[dependencies]
# ... existing deps ...
sha2 = { workspace = true, optional = true }
harmony-credential = { workspace = true, optional = true }
harmony-crypto = { workspace = true, optional = true }

[features]
# ... existing features ...
credential = ["dep:sha2", "dep:harmony-credential", "dep:harmony-crypto", "std"]
```

- [ ] **Step 2: Add `DisclosureHashMismatch` to error.rs**

Add variant to `SdJwtError`:

```rust
    /// A disclosure's hash does not appear in the signed `_sd` list.
    DisclosureHashMismatch,
```

Add Display arm:

```rust
    Self::DisclosureHashMismatch => write!(f, "disclosure hash not found in signed _sd list"),
```

- [ ] **Step 3: Add claims module to lib.rs**

```rust
#[cfg(feature = "credential")]
pub mod claims;

#[cfg(feature = "credential")]
pub use claims::{verify_disclosures, map_claims};
```

- [ ] **Step 4: Create stub claims.rs**

```rust
//! SD-JWT disclosure hash verification and claim mapping.

use crate::error::SdJwtError;
use crate::types::{Disclosure, SdJwt};

/// Verify that each disclosure's hash appears in the signed `_sd` list.
pub fn verify_disclosures<'a>(_sd_jwt: &'a SdJwt) -> Result<Vec<&'a Disclosure>, SdJwtError> {
    todo!()
}

/// Map verified disclosures to Harmony's native SaltedClaim type.
pub fn map_claims(_disclosures: &[&Disclosure]) -> Vec<harmony_credential::SaltedClaim> {
    todo!()
}
```

- [ ] **Step 5: Verify compilation**

Run: `cargo check -p harmony-sdjwt --features credential 2>&1 | tail -5`
Expected: Compiles (stubs are unused in non-test code)

- [ ] **Step 6: Commit**

```bash
git add crates/harmony-sdjwt/
git commit -m "feat: scaffold claims module with credential feature flag"
```

---

### Task 2: Implement `verify_disclosures`

**Files:**
- Modify: `crates/harmony-sdjwt/src/claims.rs`

- [ ] **Step 1: Write tests and implementation**

Replace `claims.rs` with full implementation and tests:

```rust
//! SD-JWT disclosure hash verification and claim mapping.
//!
//! Requires the `credential` feature flag.

use alloc::string::String;
use alloc::vec::Vec;
use base64::Engine;
use sha2::{Digest, Sha256};

use crate::error::SdJwtError;
use crate::types::{Disclosure, SdJwt};

const B64: base64::engine::GeneralPurpose =
    base64::engine::general_purpose::URL_SAFE_NO_PAD;

/// Verify that each disclosure's hash appears in the signed `_sd` list.
///
/// Per RFC 9901 §6.3: compute `base64url(SHA-256(ASCII(disclosure.raw)))`
/// and check the result is in `sd_jwt.payload.sd`. Returns the verified
/// disclosures on success.
///
/// # Errors
///
/// Returns `DisclosureHashMismatch` if ANY disclosure does not match
/// a signed `_sd` entry. Returns `UnsupportedAlgorithm` if `sd_alg`
/// is present and not `"sha-256"`.
pub fn verify_disclosures<'a>(sd_jwt: &'a SdJwt) -> Result<Vec<&'a Disclosure>, SdJwtError> {
    // Only sha-256 is supported (the RFC default).
    if let Some(ref alg) = sd_jwt.payload.sd_alg {
        if alg != "sha-256" {
            return Err(SdJwtError::UnsupportedAlgorithm(alg.clone()));
        }
    }

    let mut verified = Vec::with_capacity(sd_jwt.disclosures.len());

    for disclosure in &sd_jwt.disclosures {
        // RFC 9901 §4.2.3: hash the ASCII bytes of the base64url-encoded
        // disclosure string (the segment between ~ separators).
        let hash = Sha256::digest(disclosure.raw.as_bytes());
        let digest = B64.encode(hash);

        if !sd_jwt.payload.sd.contains(&digest) {
            return Err(SdJwtError::DisclosureHashMismatch);
        }

        verified.push(disclosure);
    }

    Ok(verified)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a disclosure raw string and its expected _sd digest.
    fn make_disclosure_with_digest(json_array: &str) -> (String, String) {
        let raw = B64.encode(json_array.as_bytes());
        let hash = Sha256::digest(raw.as_bytes());
        let digest = B64.encode(hash);
        (raw, digest)
    }

    /// Build a minimal SdJwt for testing disclosure verification.
    fn make_test_sdjwt(
        sd_digests: Vec<String>,
        disclosures: Vec<Disclosure>,
    ) -> SdJwt {
        use crate::types::{JwsHeader, JwtPayload};

        SdJwt {
            header: JwsHeader {
                alg: "EdDSA".into(),
                typ: Some("sd+jwt".into()),
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
                #[cfg(feature = "std")]
                extra: Vec::new(),
            },
            signature: Vec::new(),
            signing_input: String::new(),
            disclosures,
            key_binding_jwt: None,
        }
    }

    /// Build a Disclosure from a raw base64url string.
    fn disclosure_from_raw(raw: &str, salt: &str, name: Option<&str>, value_json: &str) -> Disclosure {
        Disclosure {
            raw: raw.to_string(),
            salt: salt.to_string(),
            claim_name: name.map(|n| n.to_string()),
            claim_value: value_json.as_bytes().to_vec(),
            #[cfg(feature = "std")]
            value: serde_json::from_str(value_json).unwrap(),
        }
    }

    #[test]
    fn verify_matching_disclosures() {
        let (raw1, digest1) = make_disclosure_with_digest(r#"["salt1","given_name","Alice"]"#);
        let (raw2, digest2) = make_disclosure_with_digest(r#"["salt2","family_name","Smith"]"#);

        let disc1 = disclosure_from_raw(&raw1, "salt1", Some("given_name"), "\"Alice\"");
        let disc2 = disclosure_from_raw(&raw2, "salt2", Some("family_name"), "\"Smith\"");

        let sd_jwt = make_test_sdjwt(
            vec![digest1, digest2],
            vec![disc1, disc2],
        );

        let verified = verify_disclosures(&sd_jwt).unwrap();
        assert_eq!(verified.len(), 2);
    }

    #[test]
    fn verify_rejects_unmatched_disclosure() {
        let (raw1, digest1) = make_disclosure_with_digest(r#"["salt1","given_name","Alice"]"#);
        let disc1 = disclosure_from_raw(&raw1, "salt1", Some("given_name"), "\"Alice\"");

        // Forged disclosure — not in _sd
        let forged = disclosure_from_raw("forged_base64", "badsalt", Some("evil"), "\"bad\"");

        let sd_jwt = make_test_sdjwt(
            vec![digest1],
            vec![disc1, forged],
        );

        assert!(matches!(
            verify_disclosures(&sd_jwt),
            Err(SdJwtError::DisclosureHashMismatch)
        ));
    }

    #[test]
    fn verify_empty_disclosures_ok() {
        let sd_jwt = make_test_sdjwt(vec!["some_digest".into()], vec![]);
        let verified = verify_disclosures(&sd_jwt).unwrap();
        assert!(verified.is_empty());
    }

    #[test]
    fn verify_unsupported_sd_alg() {
        let mut sd_jwt = make_test_sdjwt(vec![], vec![]);
        sd_jwt.payload.sd_alg = Some("sha-384".into());
        assert!(matches!(
            verify_disclosures(&sd_jwt),
            Err(SdJwtError::UnsupportedAlgorithm(_))
        ));
    }

    #[test]
    fn verify_default_sha256_when_sd_alg_none() {
        let (raw, digest) = make_disclosure_with_digest(r#"["s","n","v"]"#);
        let disc = disclosure_from_raw(&raw, "s", Some("n"), "\"v\"");
        let mut sd_jwt = make_test_sdjwt(vec![digest], vec![disc]);
        sd_jwt.payload.sd_alg = None; // defaults to sha-256
        assert!(verify_disclosures(&sd_jwt).is_ok());
    }
}
```

- [ ] **Step 2: Run tests**

Run: `cargo test -p harmony-sdjwt --features credential verify_disclosures 2>&1`
Expected: ALL 5 tests pass

- [ ] **Step 3: Commit**

```bash
git add crates/harmony-sdjwt/src/claims.rs
git commit -m "feat: verify_disclosures — SHA-256 hash check per RFC 9901 §6.3"
```

---

### Task 3: Implement `map_claims`

**Files:**
- Modify: `crates/harmony-sdjwt/src/claims.rs`

- [ ] **Step 1: Add map_claims implementation and tests**

Add below `verify_disclosures` (before `#[cfg(test)]`):

```rust
/// Map verified disclosures to Harmony's native `SaltedClaim` type.
///
/// Uses a static vocabulary dictionary for known EUDI PID / OpenID4VP
/// claim names (range `0x0100–0x01FF`). Unknown claim names get a
/// hash-derived `type_id` in range `0x8000–0xFFFF`. Array element
/// disclosures (no claim name) get `type_id = 0x0000`.
pub fn map_claims(disclosures: &[&Disclosure]) -> Vec<harmony_credential::SaltedClaim> {
    disclosures.iter().map(|d| {
        let type_id = match d.claim_name.as_deref() {
            None => 0x0000, // array element
            Some(name) => vocabulary_type_id(name),
        };

        let salt = decode_salt(&d.salt);

        harmony_credential::SaltedClaim {
            claim: harmony_credential::Claim {
                type_id,
                value: d.claim_value.clone(),
            },
            salt,
        }
    }).collect()
}

/// Look up a claim name in the static vocabulary, falling back to
/// hash-derived type_id.
fn vocabulary_type_id(name: &str) -> u16 {
    match name {
        // EUDI PID
        "given_name" => 0x0100,
        "family_name" => 0x0101,
        "birthdate" => 0x0102,
        "age_over_18" => 0x0103,
        "nationality" => 0x0104,
        // OpenID4VP
        "email" => 0x0110,
        "phone_number" => 0x0111,
        "address" => 0x0112,
        // Unknown: hash-derive with high bit set
        _ => {
            let hash = harmony_crypto::hash::blake3_hash(name.as_bytes());
            let raw = u16::from_be_bytes([hash[0], hash[1]]);
            raw | 0x8000 // ensure high bit is set
        }
    }
}

/// Decode an SD-JWT salt string to a fixed-size Harmony salt.
/// Base64url-decodes the string; zero-pads if < 16 bytes,
/// truncates if > 16 bytes.
fn decode_salt(salt_str: &str) -> [u8; 16] {
    let mut out = [0u8; 16];
    match B64.decode(salt_str) {
        Ok(bytes) => {
            let len = bytes.len().min(16);
            out[..len].copy_from_slice(&bytes[..len]);
        }
        Err(_) => {
            // Fallback: use UTF-8 bytes if not valid base64url
            let bytes = salt_str.as_bytes();
            let len = bytes.len().min(16);
            out[..len].copy_from_slice(&bytes[..len]);
        }
    }
    out
}
```

Add tests to the `tests` module:

```rust
    // --- map_claims tests ---

    #[test]
    fn map_known_vocabulary_claims() {
        let (raw, _) = make_disclosure_with_digest(r#"["salt","given_name","Alice"]"#);
        let disc = disclosure_from_raw(&raw, "c2FsdA", Some("given_name"), "\"Alice\"");
        let disclosures: Vec<&Disclosure> = vec![&disc];

        let claims = map_claims(&disclosures);
        assert_eq!(claims.len(), 1);
        assert_eq!(claims[0].claim.type_id, 0x0100);
    }

    #[test]
    fn map_unknown_claim_gets_hash_derived_id() {
        let (raw, _) = make_disclosure_with_digest(r#"["salt","favorite_color","blue"]"#);
        let disc = disclosure_from_raw(&raw, "c2FsdA", Some("favorite_color"), "\"blue\"");
        let disclosures: Vec<&Disclosure> = vec![&disc];

        let claims = map_claims(&disclosures);
        assert_eq!(claims.len(), 1);
        // High bit must be set for hash-derived type_ids
        assert!(claims[0].claim.type_id & 0x8000 != 0);
    }

    #[test]
    fn map_array_element_gets_zero_type_id() {
        let (raw, _) = make_disclosure_with_digest(r#"["salt","value"]"#);
        let disc = disclosure_from_raw(&raw, "c2FsdA", None, "\"value\"");
        let disclosures: Vec<&Disclosure> = vec![&disc];

        let claims = map_claims(&disclosures);
        assert_eq!(claims[0].claim.type_id, 0x0000);
    }

    #[test]
    fn map_salt_base64_decoded() {
        // "c2FsdA" is base64url for "salt" (4 bytes)
        let (raw, _) = make_disclosure_with_digest(r#"["c2FsdA","name","val"]"#);
        let disc = disclosure_from_raw(&raw, "c2FsdA", Some("name"), "\"val\"");
        let disclosures: Vec<&Disclosure> = vec![&disc];

        let claims = map_claims(&disclosures);
        // "salt" = [115, 97, 108, 116] + 12 zero bytes
        assert_eq!(claims[0].salt[0], 115); // 's'
        assert_eq!(claims[0].salt[1], 97);  // 'a'
        assert_eq!(claims[0].salt[2], 108); // 'l'
        assert_eq!(claims[0].salt[3], 116); // 't'
        assert_eq!(claims[0].salt[4..], [0u8; 12]);
    }

    #[test]
    fn map_salt_truncates_long_salt() {
        // 24 bytes base64url-encoded as 32 chars
        let long_salt = B64.encode(&[0xAA; 24]);
        let (raw, _) = make_disclosure_with_digest(r#"["long","n","v"]"#);
        let disc = disclosure_from_raw(&raw, &long_salt, Some("n"), "\"v\"");
        let disclosures: Vec<&Disclosure> = vec![&disc];

        let claims = map_claims(&disclosures);
        // First 16 bytes should be 0xAA
        assert_eq!(claims[0].salt, [0xAA; 16]);
    }

    #[test]
    fn map_hash_derived_deterministic() {
        let (raw, _) = make_disclosure_with_digest(r#"["s","custom","v"]"#);
        let d1 = disclosure_from_raw(&raw, "c2FsdA", Some("custom"), "\"v\"");
        let d2 = disclosure_from_raw(&raw, "c2FsdA", Some("custom"), "\"v\"");
        let c1 = map_claims(&[&d1]);
        let c2 = map_claims(&[&d2]);
        assert_eq!(c1[0].claim.type_id, c2[0].claim.type_id);
    }
```

- [ ] **Step 2: Run all tests**

Run: `cargo test -p harmony-sdjwt --features credential 2>&1 | tail -10`
Expected: ALL tests pass (verify_disclosures + map_claims)

- [ ] **Step 3: Commit**

```bash
git add crates/harmony-sdjwt/src/claims.rs
git commit -m "feat: map_claims — static vocabulary + hash-derived type_id mapping

EUDI PID and OpenID4VP claim names mapped to 0x0100-0x01FF range.
Unknown claims get BLAKE3-derived type_id with 0x8000 high bit.
Salt base64url-decoded and zero-padded/truncated to [u8; 16].

Closes harmony-b3m"
```
