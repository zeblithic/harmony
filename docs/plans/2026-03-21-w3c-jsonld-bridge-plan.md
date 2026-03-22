# W3C JSON-LD Bridge Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Export Harmony credentials and presentations to W3C Verifiable Credentials Data Model 2.0 JSON-LD format.

**Architecture:** New `jsonld.rs` module in `harmony-credential` behind a `jsonld` feature flag. Three public functions: `credential_to_jsonld`, `presentation_to_jsonld`, `identity_to_did_key`. Uses `serde_json` for JSON, `base64` for encoding, and unsigned-varint + base58btc for did:key.

**Tech Stack:** Rust, serde_json, base64 (base64url), bs58 (base58btc for did:key multibase encoding)

---

### File Structure

| File | Responsibility | Change |
|------|---------------|--------|
| `crates/harmony-credential/Cargo.toml` | Crate manifest | Add optional deps: serde_json, base64, bs58; add `jsonld` feature |
| `crates/harmony-credential/src/jsonld.rs` | JSON-LD export functions | Create |
| `crates/harmony-credential/src/lib.rs` | Crate root | Add `mod jsonld` behind feature gate, add re-exports |

Note: `bs58` is used for base58btc encoding (did:key multibase prefix `z`). It's smaller and simpler than the full `multibase` crate — we only need one encoding.

---

### Task 1: Add `jsonld` feature and dependencies

**Files:**
- Modify: `crates/harmony-credential/Cargo.toml`

- [ ] **Step 1: Add optional dependencies and feature flag**

Add to `[dependencies]` section:

```toml
serde_json = { version = "1", optional = true }
base64 = { version = "0.22", optional = true }
bs58 = { version = "0.5", optional = true }
```

Add to `[features]` section:

```toml
jsonld = ["dep:serde_json", "dep:base64", "dep:bs58", "std"]
```

- [ ] **Step 2: Verify the feature compiles**

Run: `cargo check -p harmony-credential --features jsonld 2>&1 | tail -5`
Expected: Compiles with no errors (no code uses the deps yet)

Run: `cargo check -p harmony-credential 2>&1 | tail -5`
Expected: Still compiles without the feature (deps are optional)

- [ ] **Step 3: Commit**

```bash
git add crates/harmony-credential/Cargo.toml Cargo.lock
git commit -m "feat: add jsonld feature flag with serde_json, base64, bs58 deps"
```

---

### Task 2: Implement `identity_to_did_key`

**Files:**
- Create: `crates/harmony-credential/src/jsonld.rs`
- Modify: `crates/harmony-credential/src/lib.rs`

- [ ] **Step 1: Write failing tests for did:key encoding**

Create `crates/harmony-credential/src/jsonld.rs` with tests first:

```rust
//! W3C JSON-LD export for Harmony verifiable credentials.
//!
//! Converts `Credential` and `Presentation` to W3C Verifiable
//! Credentials Data Model 2.0 JSON-LD format. Export only —
//! import is not supported (see harmony-6wt).

use alloc::string::String;
use harmony_identity::{CryptoSuite, IdentityRef};

/// Encode an IdentityRef + public key bytes as a `did:key` string.
///
/// Uses the multicodec prefix from `CryptoSuite::signing_multicodec()`
/// and base58btc (multibase prefix `z`) encoding.
///
/// # Key formats
/// - Ed25519: 32-byte verifying key
/// - ML-DSA-65: 1952-byte signing public key
pub fn identity_to_did_key(identity: &IdentityRef, public_key: &[u8]) -> String {
    todo!()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn did_key_ed25519_format() {
        let id = IdentityRef::new([0xAA; 16], CryptoSuite::Ed25519);
        let key = [0x42u8; 32]; // 32-byte Ed25519 key
        let did = identity_to_did_key(&id, &key);

        assert!(did.starts_with("did:key:z"), "must start with did:key:z (base58btc multibase)");
        // Decode and verify multicodec prefix
        let encoded = &did["did:key:z".len()..];
        let decoded = bs58::decode(encoded).into_vec().unwrap();
        // Ed25519 multicodec = 0x00ed, unsigned varint = [0xed, 0x01]
        assert_eq!(decoded[0], 0xed);
        assert_eq!(decoded[1], 0x01);
        // Followed by the 32-byte key
        assert_eq!(&decoded[2..], &key);
    }

    #[test]
    fn did_key_ml_dsa65_format() {
        let id = IdentityRef::new([0xBB; 16], CryptoSuite::MlDsa65);
        let key = [0x55u8; 1952]; // 1952-byte ML-DSA-65 key
        let did = identity_to_did_key(&id, &key);

        assert!(did.starts_with("did:key:z"));
        let encoded = &did["did:key:z".len()..];
        let decoded = bs58::decode(encoded).into_vec().unwrap();
        // ML-DSA-65 multicodec = 0x1211, unsigned varint = [0x91, 0x24]
        assert_eq!(decoded[0], 0x91);
        assert_eq!(decoded[1], 0x24);
        assert_eq!(&decoded[2..], &key);
    }

    #[test]
    fn did_key_deterministic() {
        let id = IdentityRef::new([0xAA; 16], CryptoSuite::Ed25519);
        let key = [0x42u8; 32];
        assert_eq!(identity_to_did_key(&id, &key), identity_to_did_key(&id, &key));
    }
}
```

- [ ] **Step 2: Wire up the module in lib.rs**

Add to `crates/harmony-credential/src/lib.rs`:

```rust
#[cfg(feature = "jsonld")]
pub mod jsonld;
```

And add re-exports:

```rust
#[cfg(feature = "jsonld")]
pub use jsonld::{credential_to_jsonld, identity_to_did_key, presentation_to_jsonld};
```

Note: `credential_to_jsonld` and `presentation_to_jsonld` don't exist yet — the re-export line will cause a compile error until Task 3. For now, only re-export `identity_to_did_key`:

```rust
#[cfg(feature = "jsonld")]
pub use jsonld::identity_to_did_key;
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `cargo test -p harmony-credential --features jsonld identity_to_did_key 2>&1`
Expected: FAIL — `todo!()` panics

- [ ] **Step 4: Implement `identity_to_did_key`**

Replace `todo!()` with:

```rust
pub fn identity_to_did_key(identity: &IdentityRef, public_key: &[u8]) -> String {
    let multicodec = identity.suite.signing_multicodec();

    // Encode multicodec as unsigned varint (LEB128)
    let mut varint = alloc::vec::Vec::with_capacity(3);
    let mut value = multicodec as u32;
    loop {
        let mut byte = (value & 0x7F) as u8;
        value >>= 7;
        if value != 0 {
            byte |= 0x80;
        }
        varint.push(byte);
        if value == 0 {
            break;
        }
    }

    // Concatenate varint prefix + raw public key
    let mut payload = alloc::vec::Vec::with_capacity(varint.len() + public_key.len());
    payload.extend_from_slice(&varint);
    payload.extend_from_slice(public_key);

    // Base58btc encode with multibase prefix 'z'
    let encoded = bs58::encode(&payload).into_string();
    alloc::format!("did:key:z{encoded}")
}
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cargo test -p harmony-credential --features jsonld identity_to_did_key 2>&1`
Expected: ALL 3 tests pass

- [ ] **Step 6: Commit**

```bash
git add crates/harmony-credential/src/jsonld.rs crates/harmony-credential/src/lib.rs
git commit -m "feat: add identity_to_did_key for W3C did:key encoding"
```

---

### Task 3: Implement `credential_to_jsonld`

**Files:**
- Modify: `crates/harmony-credential/src/jsonld.rs`
- Modify: `crates/harmony-credential/src/lib.rs`

- [ ] **Step 1: Write failing tests for credential export**

Add to the `tests` module in `jsonld.rs`:

```rust
    use crate::credential::CredentialBuilder;
    use crate::error::CredentialError;
    use harmony_identity::IdentityRef;

    fn test_issuer_ref() -> IdentityRef {
        IdentityRef::new([0xAA; 16], CryptoSuite::Ed25519)
    }

    fn test_subject_ref() -> IdentityRef {
        IdentityRef::new([0xBB; 16], CryptoSuite::Ed25519)
    }

    fn build_test_credential() -> crate::Credential {
        let mut builder = CredentialBuilder::new(
            test_issuer_ref(), test_subject_ref(), 1000, 5000, [0x01; 16],
        );
        builder.add_claim(1, alloc::vec![0xAA, 0xBB], [0x11; 16]);
        builder.add_claim(2, alloc::vec![0xCC], [0x22; 16]);
        builder.status_list_index(42);
        let payload = builder.signable_payload();
        let (cred, _) = builder.build(alloc::vec![0xDE, 0xAD, 0xBE, 0xEF]);
        cred
    }

    #[test]
    fn credential_export_has_required_fields() {
        let cred = build_test_credential();
        let issuer_key = [0x42u8; 32];
        let subject_key = [0x43u8; 32];

        let json = super::credential_to_jsonld(&cred, &issuer_key, &subject_key).unwrap();

        // Required W3C VC fields
        assert!(json["@context"].is_array());
        assert!(json["type"].is_array());
        assert!(json["issuer"].is_string());
        assert!(json["credentialSubject"].is_object());
        assert!(json["proof"].is_object());

        // Context includes VC v2 and data integrity
        let contexts: Vec<&str> = json["@context"].as_array().unwrap()
            .iter().map(|v| v.as_str().unwrap()).collect();
        assert!(contexts.contains(&"https://www.w3.org/ns/credentials/v2"));
        assert!(contexts.contains(&"https://w3id.org/security/data-integrity/v2"));

        // Type
        let types: Vec<&str> = json["type"].as_array().unwrap()
            .iter().map(|v| v.as_str().unwrap()).collect();
        assert!(types.contains(&"VerifiableCredential"));
    }

    #[test]
    fn credential_export_issuer_is_did_key() {
        let cred = build_test_credential();
        let issuer_key = [0x42u8; 32];
        let subject_key = [0x43u8; 32];

        let json = super::credential_to_jsonld(&cred, &issuer_key, &subject_key).unwrap();
        assert!(json["issuer"].as_str().unwrap().starts_with("did:key:z"));
        assert!(json["credentialSubject"]["id"].as_str().unwrap().starts_with("did:key:z"));
    }

    #[test]
    fn credential_export_timestamps_iso8601() {
        let cred = build_test_credential();
        let json = super::credential_to_jsonld(&cred, &[0x42; 32], &[0x43; 32]).unwrap();

        let valid_from = json["validFrom"].as_str().unwrap();
        let valid_until = json["validUntil"].as_str().unwrap();
        // Unix epoch 1000 = 1970-01-01T00:16:40Z
        assert!(valid_from.ends_with("Z"), "must be UTC");
        assert!(valid_until.ends_with("Z"), "must be UTC");
    }

    #[test]
    fn credential_export_status_list_present() {
        let cred = build_test_credential(); // has status_list_index = Some(42)
        let json = super::credential_to_jsonld(&cred, &[0x42; 32], &[0x43; 32]).unwrap();

        assert!(json["credentialStatus"].is_object());
        assert_eq!(json["credentialStatus"]["statusListIndex"].as_str().unwrap(), "42");
    }

    #[test]
    fn credential_export_status_list_absent_when_none() {
        let builder = CredentialBuilder::new(
            test_issuer_ref(), test_subject_ref(), 1000, 5000, [0x01; 16],
        );
        let payload = builder.signable_payload();
        let (cred, _) = builder.build(alloc::vec![0xDE, 0xAD]);

        let json = super::credential_to_jsonld(&cred, &[0x42; 32], &[0x43; 32]).unwrap();
        assert!(json.get("credentialStatus").is_none() || json["credentialStatus"].is_null());
    }

    #[test]
    fn credential_export_proof_fields() {
        let cred = build_test_credential();
        let json = super::credential_to_jsonld(&cred, &[0x42; 32], &[0x43; 32]).unwrap();

        let proof = &json["proof"];
        assert_eq!(proof["type"].as_str().unwrap(), "DataIntegrityProof");
        assert_eq!(proof["cryptosuite"].as_str().unwrap(), "eddsa-2022");
        assert!(proof["proofValue"].is_string());
        assert!(proof["nonce"].is_string());
    }

    #[test]
    fn credential_export_delegation_proof() {
        let mut builder = CredentialBuilder::new(
            test_issuer_ref(), test_subject_ref(), 1000, 5000, [0x01; 16],
        );
        builder.proof([0xFF; 32]);
        let payload = builder.signable_payload();
        let (cred, _) = builder.build(alloc::vec![0xDE, 0xAD]);

        let json = super::credential_to_jsonld(&cred, &[0x42; 32], &[0x43; 32]).unwrap();

        // Delegation proof is in a separate field, not in the W3C proof object
        assert!(json["delegationProof"].is_string());
        // W3C proof still present for the signature
        assert!(json["proof"].is_object());
    }

    #[test]
    fn credential_export_claims_as_digests() {
        let cred = build_test_credential(); // 2 claims
        let json = super::credential_to_jsonld(&cred, &[0x42; 32], &[0x43; 32]).unwrap();

        let claims = json["credentialSubject"]["claims"].as_array().unwrap();
        assert_eq!(claims.len(), 2);
        // Each claim has a digest field (base64url-encoded)
        for claim in claims {
            assert!(claim["digest"].is_string());
        }
    }
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test -p harmony-credential --features jsonld credential_export 2>&1`
Expected: FAIL — `credential_to_jsonld` doesn't exist

- [ ] **Step 3: Implement `credential_to_jsonld`**

Add to `jsonld.rs` (above the tests module):

```rust
use base64::Engine;
use crate::credential::Credential;
use crate::disclosure::Presentation;
use crate::error::CredentialError;
use alloc::string::ToString;
use alloc::vec::Vec;
use serde_json::{json, Value};

/// Export a Credential to W3C Verifiable Credentials Data Model 2.0 JSON-LD.
///
/// `issuer_key` and `subject_key` are the raw public key bytes for
/// did:key encoding. Key length must match the credential's CryptoSuite
/// (32 bytes for Ed25519, 1952 bytes for ML-DSA-65).
pub fn credential_to_jsonld(
    credential: &Credential,
    issuer_key: &[u8],
    subject_key: &[u8],
) -> Result<Value, CredentialError> {
    let issuer_did = identity_to_did_key(&credential.issuer, issuer_key);
    let subject_did = identity_to_did_key(&credential.subject, subject_key);

    let cryptosuite = match credential.issuer.suite {
        CryptoSuite::Ed25519 => "eddsa-2022",
        CryptoSuite::MlDsa65 | CryptoSuite::MlDsa65Rotatable => "mldsa65-2025",
    };

    // Claim digests as base64url
    let claims: Vec<Value> = credential
        .claim_digests
        .iter()
        .map(|d| {
            json!({ "digest": base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(d) })
        })
        .collect();

    let mut vc = json!({
        "@context": [
            "https://www.w3.org/ns/credentials/v2",
            "https://w3id.org/security/data-integrity/v2"
        ],
        "type": ["VerifiableCredential"],
        "issuer": issuer_did,
        "validFrom": epoch_to_iso8601(credential.not_before),
        "validUntil": epoch_to_iso8601(credential.expires_at),
        "credentialSubject": {
            "id": subject_did,
            "claims": claims
        },
        "proof": {
            "type": "DataIntegrityProof",
            "cryptosuite": cryptosuite,
            "proofValue": base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(&credential.signature),
            "nonce": hex_encode(&credential.nonce)
        }
    });

    // Optional: credential status
    if let Some(idx) = credential.status_list_index {
        vc["credentialStatus"] = json!({
            "type": "BitstringStatusListEntry",
            "statusListIndex": idx.to_string(),
            "statusListCredential": alloc::format!(
                "harmony:status-list:{}",
                hex_encode(&credential.issuer.hash)
            )
        });
    }

    // Optional: delegation chain reference
    if let Some(ref proof_hash) = credential.proof {
        vc["delegationProof"] = json!(
            base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(proof_hash)
        );
    }

    Ok(vc)
}

/// Convert Unix epoch seconds to ISO 8601 UTC string.
fn epoch_to_iso8601(epoch: u64) -> String {
    let secs = epoch;
    let days = secs / 86400;
    let time_secs = secs % 86400;
    let hours = time_secs / 3600;
    let minutes = (time_secs % 3600) / 60;
    let seconds = time_secs % 60;

    // Simple date calculation from days since epoch
    let (year, month, day) = days_to_ymd(days);
    alloc::format!(
        "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}Z",
        year, month, day, hours, minutes, seconds
    )
}

/// Convert days since Unix epoch to (year, month, day).
fn days_to_ymd(days: u64) -> (u64, u64, u64) {
    // Algorithm from Howard Hinnant's date algorithms
    let z = days + 719468;
    let era = z / 146097;
    let doe = z - era * 146097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    (y, m, d)
}

/// Hex-encode a byte slice (lowercase).
fn hex_encode(bytes: &[u8]) -> String {
    bytes.iter().map(|b| alloc::format!("{:02x}", b)).collect()
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test -p harmony-credential --features jsonld 2>&1`
Expected: ALL tests pass (did:key + credential export)

- [ ] **Step 5: Commit**

```bash
git add crates/harmony-credential/src/jsonld.rs
git commit -m "feat: add credential_to_jsonld for W3C VC JSON-LD export"
```

---

### Task 4: Implement `presentation_to_jsonld` and finalize re-exports

**Files:**
- Modify: `crates/harmony-credential/src/jsonld.rs`
- Modify: `crates/harmony-credential/src/lib.rs`

- [ ] **Step 1: Write failing tests for presentation export**

Add to the `tests` module in `jsonld.rs`:

```rust
    #[test]
    fn presentation_export_disclosed_claims() {
        let mut builder = CredentialBuilder::new(
            test_issuer_ref(), test_subject_ref(), 1000, 5000, [0x01; 16],
        );
        builder.add_claim(1, alloc::vec![0xAA, 0xBB], [0x11; 16]);
        builder.add_claim(2, alloc::vec![0xCC], [0x22; 16]);
        let payload = builder.signable_payload();
        let (cred, salted_claims) = builder.build(alloc::vec![0xDE, 0xAD]);

        // Disclose only the first claim
        let presentation = Presentation {
            credential: cred,
            disclosed_claims: vec![salted_claims[0].clone()],
        };

        let json = super::presentation_to_jsonld(
            &presentation, &[0x42; 32], &[0x43; 32],
        ).unwrap();

        // VP envelope
        assert!(json["@context"].is_array());
        let types: Vec<&str> = json["type"].as_array().unwrap()
            .iter().map(|v| v.as_str().unwrap()).collect();
        assert!(types.contains(&"VerifiablePresentation"));

        // Inner VC
        let vc = &json["verifiableCredential"][0];
        let claims = vc["credentialSubject"]["claims"].as_array().unwrap();
        assert_eq!(claims.len(), 2); // 2 total claims

        // First claim is disclosed (has typeId + value + digest)
        let disclosed = claims.iter().find(|c| c.get("typeId").is_some()).unwrap();
        assert_eq!(disclosed["typeId"].as_u64().unwrap(), 1);
        assert!(disclosed["value"].is_string()); // base64url-encoded
        assert!(disclosed["digest"].is_string());

        // Second claim is undisclosed (digest only)
        let undisclosed_count = claims.iter().filter(|c| c.get("typeId").is_none()).count();
        assert_eq!(undisclosed_count, 1);
    }

    #[test]
    fn presentation_export_ml_dsa65() {
        let issuer = IdentityRef::new([0xAA; 16], CryptoSuite::MlDsa65);
        let subject = IdentityRef::new([0xBB; 16], CryptoSuite::MlDsa65);

        let builder = CredentialBuilder::new(issuer, subject, 1000, 5000, [0x01; 16]);
        let payload = builder.signable_payload();
        let (cred, _) = builder.build(alloc::vec![0xDE; 3309]); // ML-DSA-65 sig length

        let json = super::credential_to_jsonld(
            &cred, &[0x55; 1952], &[0x66; 1952],
        ).unwrap();

        assert_eq!(json["proof"]["cryptosuite"].as_str().unwrap(), "mldsa65-2025");
        assert!(json["issuer"].as_str().unwrap().starts_with("did:key:z"));
    }
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test -p harmony-credential --features jsonld presentation_export 2>&1`
Expected: FAIL — `presentation_to_jsonld` doesn't exist

- [ ] **Step 3: Implement `presentation_to_jsonld`**

Add to `jsonld.rs`:

```rust
/// Export a Presentation to W3C Verifiable Presentation JSON-LD.
///
/// Disclosed claims include `typeId`, `value` (base64url), and `digest`.
/// Undisclosed claims include only `digest`.
pub fn presentation_to_jsonld(
    presentation: &Presentation,
    issuer_key: &[u8],
    subject_key: &[u8],
) -> Result<Value, CredentialError> {
    let credential = &presentation.credential;
    let issuer_did = identity_to_did_key(&credential.issuer, issuer_key);
    let subject_did = identity_to_did_key(&credential.subject, subject_key);

    let cryptosuite = match credential.issuer.suite {
        CryptoSuite::Ed25519 => "eddsa-2022",
        CryptoSuite::MlDsa65 | CryptoSuite::MlDsa65Rotatable => "mldsa65-2025",
    };

    // Build claims array: disclosed claims get typeId + value + digest,
    // undisclosed claims get digest only.
    let claims: Vec<Value> = credential
        .claim_digests
        .iter()
        .map(|digest| {
            let digest_b64 = base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(digest);

            // Check if this digest has a disclosed claim
            if let Some(sc) = presentation.disclosed_claims.iter().find(|sc| &sc.digest() == digest) {
                json!({
                    "typeId": sc.claim.type_id,
                    "value": base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(&sc.claim.value),
                    "digest": digest_b64
                })
            } else {
                json!({ "digest": digest_b64 })
            }
        })
        .collect();

    let mut vc = json!({
        "@context": [
            "https://www.w3.org/ns/credentials/v2",
            "https://w3id.org/security/data-integrity/v2"
        ],
        "type": ["VerifiableCredential"],
        "issuer": issuer_did,
        "validFrom": epoch_to_iso8601(credential.not_before),
        "validUntil": epoch_to_iso8601(credential.expires_at),
        "credentialSubject": {
            "id": subject_did,
            "claims": claims
        },
        "proof": {
            "type": "DataIntegrityProof",
            "cryptosuite": cryptosuite,
            "proofValue": base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(&credential.signature),
            "nonce": hex_encode(&credential.nonce)
        }
    });

    if let Some(idx) = credential.status_list_index {
        vc["credentialStatus"] = json!({
            "type": "BitstringStatusListEntry",
            "statusListIndex": idx.to_string(),
            "statusListCredential": alloc::format!(
                "harmony:status-list:{}", hex_encode(&credential.issuer.hash)
            )
        });
    }

    if let Some(ref proof_hash) = credential.proof {
        vc["delegationProof"] = json!(
            base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(proof_hash)
        );
    }

    // Wrap in VP envelope
    Ok(json!({
        "@context": ["https://www.w3.org/ns/credentials/v2"],
        "type": ["VerifiablePresentation"],
        "verifiableCredential": [vc]
    }))
}
```

- [ ] **Step 4: Update lib.rs re-exports**

Replace the jsonld re-export line:
```rust
#[cfg(feature = "jsonld")]
pub use jsonld::identity_to_did_key;
```
with:
```rust
#[cfg(feature = "jsonld")]
pub use jsonld::{credential_to_jsonld, identity_to_did_key, presentation_to_jsonld};
```

- [ ] **Step 5: Run all tests**

Run: `cargo test -p harmony-credential --features jsonld 2>&1`
Expected: ALL tests pass (did:key + credential + presentation)

Run: `cargo test -p harmony-credential 2>&1`
Expected: ALL tests pass without the jsonld feature (no regression)

- [ ] **Step 6: Commit**

```bash
git add crates/harmony-credential/src/jsonld.rs crates/harmony-credential/src/lib.rs
git commit -m "feat: add presentation_to_jsonld for W3C VP JSON-LD export

Complete W3C JSON-LD bridge: credential_to_jsonld,
presentation_to_jsonld, identity_to_did_key. Behind jsonld
feature flag. Disclosed claims include typeId + value + digest;
undisclosed claims include digest only.

Closes harmony-8ws"
```
