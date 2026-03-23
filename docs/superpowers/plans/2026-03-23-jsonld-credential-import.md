# W3C JSON-LD Credential Import Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
> **Note:** The `- [ ]` checkboxes below are execution tracking markers for the agentic worker, not persistent TODO items. Task tracking uses `bd` (beads) — see bead `harmony-6wt`.

**Goal:** Import W3C JSON-LD Verifiable Credentials into Harmony's compact binary credential system, verifying both Harmony custom proofs and standard JCS Data Integrity proofs.

**Architecture:** New `import.rs` module in `harmony-credential` (feature-gated behind `jsonld`). Single entry point `import_jsonld_vc()` parses the VC envelope, resolves issuer/subject DIDs, dispatches proof verification by cryptosuite (Harmony postcard path or JCS canonicalization path), extracts claims (Harmony format or external vocabulary mapping), and produces a `Credential` + `Vec<SaltedClaim>`.

**Tech Stack:** Rust, `harmony-credential` (Credential, SaltedClaim, jsonld feature), `harmony-jcs` (canonicalize), `harmony-identity` (DidResolver, ResolvedDid), `harmony-crypto` (SHA-256, BLAKE3, Ed25519/ML-DSA verify), `serde_json`, `base64`, `bs58`

**Spec:** `docs/superpowers/specs/2026-03-23-jsonld-credential-import-design.md`

---

## File Structure

```
crates/harmony-credential/
├── Cargo.toml     — Add harmony-jcs, harmony-identity deps to jsonld feature
├── src/
│   ├── lib.rs     — Export import module under jsonld feature
│   ├── import.rs  — NEW: import_jsonld_vc(), types, helpers
│   └── jsonld.rs  — Existing export (read-only reference for round-trip tests)
```

---

### Task 1: Create import module scaffold and types

**Files:**
- Create: `crates/harmony-credential/src/import.rs`
- Modify: `crates/harmony-credential/src/lib.rs`
- Modify: `crates/harmony-credential/Cargo.toml`

- [ ] **Step 1: Add dependencies to Cargo.toml**

In `Cargo.toml`, add `harmony-jcs` and `hex` to the `[dependencies]` section:

```toml
harmony-jcs = { workspace = true, optional = true }
hex = { workspace = true, optional = true }
```

`harmony-identity` is already a non-optional dependency. Update the `jsonld` feature to include the new optional deps:

```toml
jsonld = ["dep:serde_json", "dep:base64", "dep:bs58", "dep:harmony-jcs", "dep:hex", "std"]
```

- [ ] **Step 2: Create import.rs with types and stub**

```rust
//! W3C JSON-LD import for Harmony verifiable credentials.
//!
//! Parses a W3C VC Data Model 2.0 JSON-LD document, verifies the
//! Data Integrity proof (Harmony custom or JCS-based), extracts claims,
//! and produces a Harmony [`Credential`] + [`Vec<SaltedClaim>`].

use alloc::string::String;
use alloc::vec::Vec;
use serde_json::Value;

use harmony_identity::did::{DidError, DidResolver, ResolvedDid};
use harmony_identity::{CryptoSuite, IdentityRef};

use crate::claim::{Claim, SaltedClaim};
use crate::credential::Credential;

/// Sentinel salt for imported claims that lack native Harmony salts.
/// All-zeros signals "imported, not selectively disclosable."
pub const IMPORT_SENTINEL_SALT: [u8; 16] = [0u8; 16];

/// Result of importing a W3C JSON-LD Verifiable Credential.
#[derive(Debug, Clone)]
pub struct ImportedCredential {
    /// The Harmony binary credential with verified signature.
    pub credential: Credential,
    /// Extracted claims (with sentinel salt for external VCs).
    pub claims: Vec<SaltedClaim>,
    /// Which proof verification path was used.
    pub proof_type: ImportedProofType,
}

/// How the imported credential's proof was verified.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImportedProofType {
    /// Harmony custom cryptosuite (postcard payload signature).
    HarmonyCustom,
    /// Standard W3C JCS Data Integrity proof.
    JcsDataIntegrity,
}

/// Errors from credential import operations.
#[derive(Debug, Clone)]
pub enum ImportError {
    /// JSON structure doesn't match W3C VC format.
    MalformedVc(String),
    /// DID resolution failed.
    DidResolution(DidError),
    /// Proof cryptosuite not supported.
    UnsupportedCryptosuite(String),
    /// Data Integrity proof verification failed.
    ProofInvalid,
    /// Timestamp parsing failed.
    InvalidTimestamp(String),
    /// Claim extraction failed.
    ClaimError(String),
    /// Base encoding (base58, base64url) failed.
    EncodingError(String),
}

impl From<DidError> for ImportError {
    fn from(e: DidError) -> Self {
        ImportError::DidResolution(e)
    }
}

impl core::fmt::Display for ImportError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::MalformedVc(msg) => write!(f, "malformed VC: {msg}"),
            Self::DidResolution(e) => write!(f, "DID resolution: {e}"),
            Self::UnsupportedCryptosuite(s) => write!(f, "unsupported cryptosuite: {s}"),
            Self::ProofInvalid => write!(f, "proof verification failed"),
            Self::InvalidTimestamp(msg) => write!(f, "invalid timestamp: {msg}"),
            Self::ClaimError(msg) => write!(f, "claim error: {msg}"),
            Self::EncodingError(msg) => write!(f, "encoding error: {msg}"),
        }
    }
}

/// Import a W3C Verifiable Credential from JSON-LD format.
///
/// Parses the VC envelope, resolves issuer/subject DIDs, verifies the
/// Data Integrity proof, extracts claims, and produces a Harmony credential.
pub fn import_jsonld_vc(
    _vc_json: &Value,
    _resolver: &impl DidResolver,
) -> Result<ImportedCredential, ImportError> {
    // TODO: implement in subsequent tasks
    Err(ImportError::MalformedVc(String::from("not implemented")))
}
```

- [ ] **Step 3: Export from lib.rs**

Add after the existing `jsonld` exports:

```rust
#[cfg(feature = "jsonld")]
pub mod import;

#[cfg(feature = "jsonld")]
pub use import::{import_jsonld_vc, ImportedCredential, ImportedProofType, ImportError};
```

- [ ] **Step 4: Run tests**

Run: `cargo test -p harmony-credential --features jsonld`

- [ ] **Step 5: Commit**

```bash
git commit -m "feat(credential): import module scaffold with types

ImportedCredential, ImportedProofType, ImportError types.
Stub import_jsonld_vc() behind jsonld feature.
harmony-jcs added to jsonld feature dependencies."
```

---

### Task 2: Implement VC envelope parsing and DID resolution

**Files:**
- Modify: `crates/harmony-credential/src/import.rs`

- [ ] **Step 1: Write tests for envelope parsing**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use harmony_identity::did::DefaultDidResolver;
    use serde_json::json;

    #[test]
    fn import_malformed_vc_rejected() {
        let resolver = DefaultDidResolver;
        // Missing required fields
        let vc = json!({"foo": "bar"});
        let result = import_jsonld_vc(&vc, &resolver);
        assert!(matches!(result, Err(ImportError::MalformedVc(_))));
    }

    #[test]
    fn import_missing_proof_rejected() {
        let resolver = DefaultDidResolver;
        let vc = json!({
            "issuer": "did:key:z6MkhaXgBZDvotDkL5257faiztiGiC2QtKLGpbnnEGta2doK",
            "credentialSubject": {"id": "did:key:z6MkhaXgBZDvotDkL5257faiztiGiC2QtKLGpbnnEGta2doK"},
            "validFrom": "2024-01-01T00:00:00Z",
        });
        let result = import_jsonld_vc(&vc, &resolver);
        assert!(matches!(result, Err(ImportError::MalformedVc(_))));
    }

    #[test]
    fn import_unsupported_cryptosuite() {
        let resolver = DefaultDidResolver;
        let vc = json!({
            "issuer": "did:key:z6MkhaXgBZDvotDkL5257faiztiGiC2QtKLGpbnnEGta2doK",
            "credentialSubject": {"id": "did:key:z6MkhaXgBZDvotDkL5257faiztiGiC2QtKLGpbnnEGta2doK"},
            "validFrom": "2024-01-01T00:00:00Z",
            "proof": {
                "type": "DataIntegrityProof",
                "cryptosuite": "unknown-suite-2099",
                "proofValue": "zFake",
                "verificationMethod": "did:key:z6MkhaXgBZDvotDkL5257faiztiGiC2QtKLGpbnnEGta2doK#key-1",
                "created": "2024-01-01T00:00:00Z",
                "proofPurpose": "assertionMethod"
            }
        });
        let result = import_jsonld_vc(&vc, &resolver);
        assert!(matches!(result, Err(ImportError::UnsupportedCryptosuite(_))));
    }
}
```

- [ ] **Step 2: Implement envelope parsing helpers**

```rust
/// Derive an identity hash from a signing public key for imported credentials.
/// Uses SHA-256(sign_key)[:16] — distinct from native Harmony addresses which
/// hash both encryption and signing keys.
fn derive_import_identity_hash(public_key: &[u8]) -> [u8; 16] {
    harmony_crypto::hash::truncated_hash(public_key)
}

/// Parse ISO 8601 timestamp to Unix epoch seconds.
/// Supports basic format: "YYYY-MM-DDTHH:MM:SSZ"
fn parse_iso8601(s: &str) -> Result<u64, ImportError> {
    // Minimal ISO 8601 parser for the common W3C VC timestamp format.
    // Full ISO 8601 parsing is complex; we handle the subset that
    // W3C VC Data Model 2.0 uses.
    let s = s.trim();
    if s.len() < 20 || !s.ends_with('Z') {
        return Err(ImportError::InvalidTimestamp(String::from(s)));
    }
    // "YYYY-MM-DDTHH:MM:SSZ" → parse parts
    let year: u64 = s[0..4].parse().map_err(|_| ImportError::InvalidTimestamp(String::from(s)))?;
    let month: u64 = s[5..7].parse().map_err(|_| ImportError::InvalidTimestamp(String::from(s)))?;
    let day: u64 = s[8..10].parse().map_err(|_| ImportError::InvalidTimestamp(String::from(s)))?;
    let hour: u64 = s[11..13].parse().map_err(|_| ImportError::InvalidTimestamp(String::from(s)))?;
    let min: u64 = s[14..16].parse().map_err(|_| ImportError::InvalidTimestamp(String::from(s)))?;
    let sec: u64 = s[17..19].parse().map_err(|_| ImportError::InvalidTimestamp(String::from(s)))?;

    // Approximate Unix timestamp (ignoring leap seconds, good enough for credential time bounds)
    let days = days_since_epoch(year, month, day);
    Ok(days * 86400 + hour * 3600 + min * 60 + sec)
}

/// Days from Unix epoch (1970-01-01) to the given date.
fn days_since_epoch(year: u64, month: u64, day: u64) -> u64 {
    // Simplified algorithm — accurate for dates 1970-2099
    let y = if month <= 2 { year - 1 } else { year };
    let m = if month <= 2 { month + 9 } else { month - 3 };
    let era = y / 400;
    let yoe = y - era * 400;
    let doy = (153 * m + 2) / 5 + day - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    let days = era * 146097 + doe;
    days.saturating_sub(719468) // Offset from 0000-03-01 to 1970-01-01
}

/// Parsed VC envelope fields needed for import.
struct VcEnvelope<'a> {
    issuer_did: &'a str,
    subject_did: Option<&'a str>,
    proof: &'a Value,
    cryptosuite: &'a str,
    proof_value_raw: &'a str,
    verification_method: &'a str,
    not_before: u64,
    expires_at: u64,
    issued_at: u64,
    nonce: [u8; 16],
    credential_subject: &'a Value,
}

/// Parse the VC JSON into a VcEnvelope with all required fields extracted.
fn parse_vc_envelope(vc: &Value) -> Result<VcEnvelope<'_>, ImportError> {
    // Extract issuer (string or object with "id")
    let issuer_did = match vc.get("issuer") {
        Some(Value::String(s)) => s.as_str(),
        Some(obj) => obj.get("id")
            .and_then(Value::as_str)
            .ok_or_else(|| ImportError::MalformedVc(String::from("issuer missing id")))?,
        None => return Err(ImportError::MalformedVc(String::from("missing issuer"))),
    };

    let credential_subject = vc.get("credentialSubject")
        .ok_or_else(|| ImportError::MalformedVc(String::from("missing credentialSubject")))?;

    let subject_did = credential_subject.get("id").and_then(Value::as_str);

    let proof = vc.get("proof")
        .ok_or_else(|| ImportError::MalformedVc(String::from("missing proof")))?;

    let cryptosuite = proof.get("cryptosuite")
        .and_then(Value::as_str)
        .ok_or_else(|| ImportError::MalformedVc(String::from("missing proof.cryptosuite")))?;

    let proof_value_raw = proof.get("proofValue")
        .and_then(Value::as_str)
        .ok_or_else(|| ImportError::MalformedVc(String::from("missing proof.proofValue")))?;

    let verification_method = proof.get("verificationMethod")
        .and_then(Value::as_str)
        .ok_or_else(|| ImportError::MalformedVc(String::from("missing proof.verificationMethod")))?;

    let not_before = match vc.get("validFrom").and_then(Value::as_str) {
        Some(ts) => parse_iso8601(ts)?,
        None => 0,
    };

    let expires_at = match vc.get("validUntil").and_then(Value::as_str) {
        Some(ts) => parse_iso8601(ts)?,
        None => 0, // 0 = no expiry
    };

    let issued_at = match proof.get("created").and_then(Value::as_str) {
        Some(ts) => parse_iso8601(ts)?,
        None => 0,
    };

    let nonce = match proof.get("nonce").and_then(Value::as_str) {
        Some(hex_str) => {
            let bytes = hex::decode(hex_str)
                .map_err(|_| ImportError::EncodingError(String::from("bad nonce hex")))?;
            if bytes.len() >= 16 {
                let mut arr = [0u8; 16];
                arr.copy_from_slice(&bytes[..16]);
                arr
            } else {
                [0u8; 16]
            }
        }
        None => [0u8; 16],
    };

    Ok(VcEnvelope {
        issuer_did,
        subject_did,
        proof,
        cryptosuite,
        proof_value_raw,
        verification_method,
        not_before,
        expires_at,
        issued_at,
        nonce,
        credential_subject,
    })
}
```

Add `hex` as a dependency if not already available. Check — it's likely already pulled in via `harmony-crypto` or similar. If not, add it.

- [ ] **Step 3: Wire envelope parsing into import_jsonld_vc stub**

Replace the stub body with:

```rust
pub fn import_jsonld_vc(
    vc_json: &Value,
    resolver: &impl DidResolver,
) -> Result<ImportedCredential, ImportError> {
    let env = parse_vc_envelope(vc_json)?;

    // Resolve issuer DID
    let issuer_resolved = resolver.resolve(env.issuer_did)?;
    let issuer_hash = derive_import_identity_hash(&issuer_resolved.public_key);
    let issuer_ref = IdentityRef { hash: issuer_hash, suite: issuer_resolved.suite };

    // Resolve subject DID (defaults to issuer if absent)
    let (subject_ref, _subject_key) = if let Some(sub_did) = env.subject_did {
        let sub_resolved = resolver.resolve(sub_did)?;
        let sub_hash = derive_import_identity_hash(&sub_resolved.public_key);
        (IdentityRef { hash: sub_hash, suite: sub_resolved.suite }, sub_resolved.public_key)
    } else {
        (issuer_ref, issuer_resolved.public_key.clone())
    };

    // Dispatch on cryptosuite
    match env.cryptosuite {
        "harmony-eddsa-2022" | "harmony-mldsa65-2025" => {
            Err(ImportError::MalformedVc(String::from("Harmony proof path not yet implemented")))
        }
        "eddsa-jcs-2022" | "mldsa65-jcs-2024" | "di-mldsa-jcs-2025" => {
            Err(ImportError::MalformedVc(String::from("JCS proof path not yet implemented")))
        }
        other => Err(ImportError::UnsupportedCryptosuite(String::from(other))),
    }
}
```

- [ ] **Step 4: Run tests**

Run: `cargo test -p harmony-credential --features jsonld import_`

- [ ] **Step 5: Commit**

```bash
git commit -m "feat(credential/import): VC envelope parsing and DID resolution

parse_vc_envelope() extracts issuer, subject, proof fields.
derive_import_identity_hash() uses SHA-256(sign_key)[:16].
parse_iso8601() for W3C timestamp format. Cryptosuite dispatch stub."
```

---

### Task 3: Implement JCS proof verification

**Files:**
- Modify: `crates/harmony-credential/src/import.rs`

- [ ] **Step 1: Write test for JCS proof verification**

This test constructs a VC, signs it with JCS Data Integrity, and verifies import works.

```rust
    #[test]
    fn import_jcs_eddsa_proof() {
        use harmony_identity::PrivateIdentity;
        use rand::rngs::OsRng;

        let identity = PrivateIdentity::generate(&mut OsRng);
        let pub_id = identity.public_identity();
        let pubkey_bytes = pub_id.verifying_key.as_bytes();

        // Build a did:key for the issuer
        let id_ref = IdentityRef {
            hash: derive_import_identity_hash(&pubkey_bytes),
            suite: CryptoSuite::Ed25519,
        };
        let issuer_did = crate::jsonld::identity_to_did_key(&id_ref, &pubkey_bytes).unwrap();

        // Build a minimal VC document
        let mut vc = json!({
            "@context": [
                "https://www.w3.org/ns/credentials/v2",
                "https://w3id.org/security/data-integrity/v2"
            ],
            "type": ["VerifiableCredential"],
            "issuer": &issuer_did,
            "credentialSubject": {
                "id": &issuer_did,
                "givenName": "Alice",
                "familyName": "Smith"
            },
            "validFrom": "2024-01-01T00:00:00Z",
            "validUntil": "2025-01-01T00:00:00Z"
        });

        // Build proof options (without proofValue)
        let proof_options = json!({
            "type": "DataIntegrityProof",
            "cryptosuite": "eddsa-jcs-2022",
            "created": "2024-01-01T00:00:00Z",
            "verificationMethod": format!("{}#key-1", issuer_did),
            "proofPurpose": "assertionMethod"
        });

        // JCS Data Integrity: hash(canonicalize(proof_options)) || hash(canonicalize(doc))
        let proof_hash = harmony_crypto::hash::full_hash(
            &harmony_jcs::canonicalize(&proof_options)
        );
        let doc_hash = harmony_crypto::hash::full_hash(
            &harmony_jcs::canonicalize(&vc)
        );
        let mut verify_bytes = Vec::with_capacity(64);
        verify_bytes.extend_from_slice(&proof_hash);
        verify_bytes.extend_from_slice(&doc_hash);

        // Sign
        let signature = identity.sign(&verify_bytes);
        let proof_value = alloc::format!("z{}", bs58::encode(&signature).into_string());

        // Attach proof to VC
        let mut proof = proof_options.clone();
        proof.as_object_mut().unwrap().insert("proofValue".into(), Value::String(proof_value));
        vc.as_object_mut().unwrap().insert("proof".into(), proof);

        // Import
        let resolver = DefaultDidResolver;
        let result = import_jsonld_vc(&vc, &resolver);
        assert!(result.is_ok(), "JCS import failed: {:?}", result.err());

        let imported = result.unwrap();
        assert_eq!(imported.proof_type, ImportedProofType::JcsDataIntegrity);
        assert_eq!(imported.claims.len(), 2); // givenName + familyName
        assert_eq!(imported.credential.issuer.hash, id_ref.hash);
    }
```

Note: Check the exact Ed25519 identity API. The identity may be named `Ed25519Identity` or similar. Search for `Ed25519` in `harmony-identity/src/`. Adapt the test to match the actual API — the key operations are: generate keypair, get public key bytes, sign bytes.

- [ ] **Step 2: Implement verify_jcs_proof**

```rust
/// Decode a multibase base58btc proofValue (strip leading 'z', base58 decode).
fn decode_proof_value(raw: &str) -> Result<Vec<u8>, ImportError> {
    let encoded = raw.strip_prefix('z')
        .ok_or_else(|| ImportError::EncodingError(String::from("proofValue must start with 'z'")))?;
    bs58::decode(encoded)
        .into_vec()
        .map_err(|e| ImportError::EncodingError(alloc::format!("base58: {e}")))
}

/// Verify a JCS Data Integrity proof (eddsa-jcs-2022 / mldsa65-jcs-2024).
///
/// Algorithm (W3C Data Integrity Cryptosuites v1.0 §4.3):
/// 1. Remove proof from document → unsigned_doc
/// 2. Remove proofValue from proof → proof_options
/// 3. proof_hash = SHA-256(canonicalize(proof_options))
/// 4. doc_hash = SHA-256(canonicalize(unsigned_doc))
/// 5. verify_bytes = proof_hash || doc_hash
/// 6. Verify signature over verify_bytes
fn verify_jcs_proof(
    vc_json: &Value,
    signature_bytes: &[u8],
    issuer_key: &[u8],
    suite: CryptoSuite,
) -> Result<(), ImportError> {
    // 1. Clone doc, remove proof
    let mut unsigned_doc = vc_json.clone();
    unsigned_doc.as_object_mut()
        .ok_or_else(|| ImportError::MalformedVc(String::from("VC is not an object")))?
        .remove("proof");

    // 2. Clone proof, remove proofValue
    let mut proof_options = vc_json.get("proof")
        .ok_or_else(|| ImportError::MalformedVc(String::from("missing proof")))?
        .clone();
    proof_options.as_object_mut()
        .ok_or_else(|| ImportError::MalformedVc(String::from("proof is not an object")))?
        .remove("proofValue");

    // 3-4. Canonicalize and hash
    let proof_hash = harmony_crypto::hash::full_hash(
        &harmony_jcs::canonicalize(&proof_options)
    );
    let doc_hash = harmony_crypto::hash::full_hash(
        &harmony_jcs::canonicalize(&unsigned_doc)
    );

    // 5. Concatenate hashes
    let mut verify_bytes = Vec::with_capacity(64);
    verify_bytes.extend_from_slice(&proof_hash);
    verify_bytes.extend_from_slice(&doc_hash);

    // 6. Verify signature
    harmony_identity::verify_signature(suite, issuer_key, &verify_bytes, signature_bytes)
        .map_err(|_| ImportError::ProofInvalid)
}
```

Note: Check the exact `verify_ed25519` function name and `MlDsaPublicKey::verify` signature. The Ed25519 verify function is in `harmony-identity` — search for `pub fn verify`. Adapt to match the actual API.

- [ ] **Step 3: Wire JCS path into import_jsonld_vc**

Replace the JCS stub in the match arm:

```rust
        "eddsa-jcs-2022" | "mldsa65-jcs-2024" | "di-mldsa-jcs-2025" => {
            let signature_bytes = decode_proof_value(env.proof_value_raw)?;
            verify_jcs_proof(vc_json, &signature_bytes, &issuer_resolved.public_key, issuer_resolved.suite)?;

            let claims = extract_claims(env.credential_subject)?;
            let claim_digests: Vec<[u8; 32]> = claims.iter().map(|sc| sc.digest()).collect();

            let credential = Credential {
                issuer: issuer_ref,
                subject: subject_ref,
                claim_digests,
                status_list_index: None,
                not_before: env.not_before,
                expires_at: env.expires_at,
                issued_at: env.issued_at,
                nonce: env.nonce,
                proof: None,
                signature: signature_bytes,
            };

            Ok(ImportedCredential {
                credential,
                claims,
                proof_type: ImportedProofType::JcsDataIntegrity,
            })
        }
```

Note: `Credential` fields are all `pub`, so direct struct construction works. `signable_bytes()` is `pub(crate)`, accessible from `import.rs` since it's in the same crate.

- [ ] **Step 4: Implement extract_claims (external format)**

```rust
/// Extract claims from credentialSubject, mapping to Harmony SaltedClaim format.
fn extract_claims(subject: &Value) -> Result<Vec<SaltedClaim>, ImportError> {
    let obj = subject.as_object()
        .ok_or_else(|| ImportError::ClaimError(String::from("credentialSubject not an object")))?;

    let mut claims = Vec::new();

    // Check for Harmony format (has "claims" array)
    if let Some(Value::Array(harmony_claims)) = obj.get("claims") {
        return extract_harmony_claims(harmony_claims);
    }

    // External format: map each field through vocabulary
    for (key, value) in obj {
        // Skip standard JSON-LD / VC fields
        if matches!(key.as_str(), "id" | "type" | "@context") {
            continue;
        }

        let type_id = vocabulary_type_id(key);
        // Serialize the JSON value compactly as the claim value
        let value_bytes = serde_json::to_vec(value)
            .map_err(|e| ImportError::ClaimError(alloc::format!("json serialize: {e}")))?;

        claims.push(SaltedClaim {
            claim: Claim {
                type_id,
                value: value_bytes,
            },
            salt: IMPORT_SENTINEL_SALT,
        });
    }

    Ok(claims)
}

/// Extract claims from Harmony's structured claims array format.
fn extract_harmony_claims(claims_array: &[Value]) -> Result<Vec<SaltedClaim>, ImportError> {
    let mut claims = Vec::new();
    let engine = base64::engine::general_purpose::URL_SAFE_NO_PAD;

    for item in claims_array {
        // Disclosed claims have typeId + value; undisclosed have only digest
        if let (Some(type_id), Some(value_b64)) = (
            item.get("typeId").and_then(Value::as_u64),
            item.get("value").and_then(Value::as_str),
        ) {
            use base64::Engine;
            let value = engine.decode(value_b64)
                .map_err(|e| ImportError::EncodingError(alloc::format!("base64: {e}")))?;

            // Check for explicit salt; fall back to sentinel
            let salt = if let Some(salt_hex) = item.get("salt").and_then(Value::as_str) {
                let salt_bytes = hex::decode(salt_hex)
                    .map_err(|_| ImportError::EncodingError(String::from("bad salt hex")))?;
                if salt_bytes.len() >= 16 {
                    let mut arr = [0u8; 16];
                    arr.copy_from_slice(&salt_bytes[..16]);
                    arr
                } else {
                    IMPORT_SENTINEL_SALT
                }
            } else {
                IMPORT_SENTINEL_SALT
            };

            claims.push(SaltedClaim {
                claim: Claim {
                    type_id: type_id as u16,
                    value,
                },
                salt,
            });
        }
        // Skip digest-only (undisclosed) claims — we only import disclosed claims
    }

    Ok(claims)
}

/// Map a JSON-LD claim name to a Harmony type ID.
/// Known names get fixed IDs; unknown get BLAKE3-derived IDs with high bit set.
fn vocabulary_type_id(name: &str) -> u16 {
    match name {
        "given_name" | "givenName" => 0x0100,
        "family_name" | "familyName" => 0x0101,
        "birthdate" | "birthDate" => 0x0102,
        "age_over_18" | "ageOver18" => 0x0103,
        "nationality" => 0x0104,
        "email" => 0x0110,
        "phone_number" | "phoneNumber" => 0x0111,
        "address" => 0x0112,
        _ => {
            let hash = harmony_crypto::hash::blake3_hash(name.as_bytes());
            u16::from_le_bytes([hash[0], hash[1]]) | 0x8000
        }
    }
}
```

Note: The vocabulary includes both snake_case (SD-JWT convention) and camelCase (JSON-LD convention) for the same claims. This handles both.

- [ ] **Step 5: Run tests**

Run: `RUST_MIN_STACK=8388608 cargo test -p harmony-credential --features jsonld import_`

- [ ] **Step 6: Commit**

```bash
git commit -m "feat(credential/import): JCS proof verification and claim extraction

verify_jcs_proof() implements W3C Data Integrity §4.3: canonicalize
proof options and document separately, SHA-256 hash each, verify
signature over concatenated hashes. extract_claims() handles both
Harmony format and external vocabulary mapping with sentinel salt."
```

---

### Task 4: Implement Harmony custom proof verification and round-trip test

**Files:**
- Modify: `crates/harmony-credential/src/import.rs`

- [ ] **Step 1: Write round-trip test**

```rust
    #[test]
    fn import_harmony_eddsa_roundtrip() {
        use harmony_identity::PrivateIdentity;
        use rand::rngs::OsRng;

        let identity = PrivateIdentity::generate(&mut OsRng);
        let pub_id = identity.public_identity();
        let pubkey = pub_id.verifying_key.as_bytes();

        // Build a credential using the standard builder
        let issuer_ref = IdentityRef::from(pub_id);
        let mut builder = crate::CredentialBuilder::new(
            issuer_ref, issuer_ref, 1000, 2000, [0u8; 16],
        );
        builder.add_claim(0x0100, b"Alice".to_vec(), [1u8; 16]);
        let payload = builder.signable_payload();
        let signature = identity.sign(&payload);
        let (credential, _claims) = builder.build(signature.to_vec());

        // Export to JSON-LD
        let vc_json = crate::jsonld::credential_to_jsonld(&credential, &pubkey, &pubkey).unwrap();

        // Import back
        let resolver = DefaultDidResolver;
        let result = import_jsonld_vc(&vc_json, &resolver);
        assert!(result.is_ok(), "round-trip failed: {:?}", result.err());

        let imported = result.unwrap();
        assert_eq!(imported.proof_type, ImportedProofType::HarmonyCustom);
        assert_eq!(imported.credential.issuer.suite, CryptoSuite::Ed25519);
    }
```

Note: Adapt to actual API — `PrivateIdentity` may not exist by that name. Search for Ed25519 identity generation in `harmony-identity/src/`. The key operations: generate, get public bytes, sign bytes, get IdentityRef.

- [ ] **Step 2: Implement verify_harmony_proof**

```rust
/// Verify a Harmony custom proof (postcard payload signature).
fn verify_harmony_proof(
    env: &VcEnvelope<'_>,
    signature_bytes: &[u8],
    issuer_ref: &IdentityRef,
    subject_ref: &IdentityRef,
    issuer_key: &[u8],
    claim_digests: &[[u8; 32]],
) -> Result<(), ImportError> {
    // Reconstruct the credential with the extracted fields to get signable_bytes()
    let credential = Credential {
        issuer: *issuer_ref,
        subject: *subject_ref,
        claim_digests: claim_digests.to_vec(),
        status_list_index: None,
        not_before: env.not_before,
        expires_at: env.expires_at,
        issued_at: env.issued_at,
        nonce: env.nonce,
        proof: None,
        signature: signature_bytes.to_vec(),
    };

    let payload = credential.signable_bytes();

    harmony_identity::verify_signature(issuer_ref.suite, issuer_key, &payload, signature_bytes)
        .map_err(|_| ImportError::ProofInvalid)
}
```

- [ ] **Step 3: Wire Harmony path into import_jsonld_vc**

Replace the Harmony stub in the match arm:

```rust
        "harmony-eddsa-2022" | "harmony-mldsa65-2025" => {
            let signature_bytes = decode_proof_value(env.proof_value_raw)?;

            // For Harmony custom proofs, extract claim digests from the exported format
            let claim_digests = extract_harmony_digests(env.credential_subject)?;
            let claims = extract_claims(env.credential_subject)?;

            verify_harmony_proof(
                &env, &signature_bytes, &issuer_ref, &subject_ref,
                &issuer_resolved.public_key, &claim_digests,
            )?;

            let credential = Credential {
                issuer: issuer_ref,
                subject: subject_ref,
                claim_digests,
                status_list_index: None,
                not_before: env.not_before,
                expires_at: env.expires_at,
                issued_at: env.issued_at,
                nonce: env.nonce,
                proof: None,
                signature: signature_bytes,
            };

            Ok(ImportedCredential {
                credential,
                claims,
                proof_type: ImportedProofType::HarmonyCustom,
            })
        }
```

Add helper to extract digests from Harmony format:

```rust
/// Extract claim digests from Harmony's exported claims array.
fn extract_harmony_digests(subject: &Value) -> Result<Vec<[u8; 32]>, ImportError> {
    use base64::Engine;
    let engine = base64::engine::general_purpose::URL_SAFE_NO_PAD;

    let claims = subject.get("claims")
        .and_then(Value::as_array)
        .ok_or_else(|| ImportError::ClaimError(String::from("missing claims array")))?;

    let mut digests = Vec::new();
    for item in claims {
        let digest_b64 = item.get("digest")
            .and_then(Value::as_str)
            .ok_or_else(|| ImportError::ClaimError(String::from("claim missing digest")))?;
        let bytes = engine.decode(digest_b64)
            .map_err(|e| ImportError::EncodingError(alloc::format!("base64: {e}")))?;
        if bytes.len() != 32 {
            return Err(ImportError::ClaimError(String::from("digest not 32 bytes")));
        }
        let mut arr = [0u8; 32];
        arr.copy_from_slice(&bytes);
        digests.push(arr);
    }
    Ok(digests)
}
```

- [ ] **Step 4: Run tests**

Run: `RUST_MIN_STACK=8388608 cargo test -p harmony-credential --features jsonld import_`

- [ ] **Step 5: Commit**

```bash
git commit -m "feat(credential/import): Harmony custom proof verification

verify_harmony_proof() reconstructs postcard SignablePayload via
Credential::signable_bytes(), verifies Ed25519/ML-DSA signature.
Round-trip test: export → import for Harmony VCs."
```

---

### Task 5: Additional tests and edge cases

**Files:**
- Modify: `crates/harmony-credential/src/import.rs`

- [ ] **Step 1: Add remaining tests**

```rust
    #[test]
    fn import_external_claims_vocabulary() {
        // Test that known claim names map to correct type IDs
        assert_eq!(vocabulary_type_id("givenName"), 0x0100);
        assert_eq!(vocabulary_type_id("given_name"), 0x0100);
        assert_eq!(vocabulary_type_id("familyName"), 0x0101);
        assert_eq!(vocabulary_type_id("email"), 0x0110);
    }

    #[test]
    fn import_external_claims_unknown() {
        // Unknown claims get hash-derived IDs with high bit set
        let id = vocabulary_type_id("customField");
        assert!(id & 0x8000 != 0, "unknown claims should have high bit set");
    }

    #[test]
    fn import_sentinel_salt() {
        assert_eq!(IMPORT_SENTINEL_SALT, [0u8; 16]);
    }

    #[test]
    fn import_timestamps_parsed() {
        let ts = parse_iso8601("2024-01-15T12:30:45Z").unwrap();
        // 2024-01-15 12:30:45 UTC
        // Approximate check: should be in the right ballpark (1705312245 ± some)
        assert!(ts > 1700000000, "timestamp should be after 2023");
        assert!(ts < 1800000000, "timestamp should be before 2027");
    }

    #[test]
    fn import_invalid_proof_rejected() {
        // Construct a VC with valid structure but tampered proofValue
        let resolver = DefaultDidResolver;
        let vc = json!({
            "issuer": "did:key:z6MkhaXgBZDvotDkL5257faiztiGiC2QtKLGpbnnEGta2doK",
            "credentialSubject": {
                "id": "did:key:z6MkhaXgBZDvotDkL5257faiztiGiC2QtKLGpbnnEGta2doK",
                "givenName": "Alice"
            },
            "validFrom": "2024-01-01T00:00:00Z",
            "proof": {
                "type": "DataIntegrityProof",
                "cryptosuite": "eddsa-jcs-2022",
                "proofValue": "zInvalidSignatureData",
                "verificationMethod": "did:key:z6MkhaXgBZDvotDkL5257faiztiGiC2QtKLGpbnnEGta2doK#key-1",
                "created": "2024-01-01T00:00:00Z",
                "proofPurpose": "assertionMethod"
            }
        });
        let result = import_jsonld_vc(&vc, &resolver);
        assert!(matches!(result, Err(ImportError::ProofInvalid)));
    }

    #[test]
    fn import_missing_subject_defaults_to_issuer() {
        // credentialSubject without "id" → subject = issuer
        // (This is tested implicitly in the JCS test, but let's verify explicitly)
        let resolver = DefaultDidResolver;
        let vc = json!({
            "issuer": "did:key:z6MkhaXgBZDvotDkL5257faiztiGiC2QtKLGpbnnEGta2doK",
            "credentialSubject": {
                "givenName": "Alice"
            },
            "validFrom": "2024-01-01T00:00:00Z",
            "proof": {
                "type": "DataIntegrityProof",
                "cryptosuite": "eddsa-jcs-2022",
                "proofValue": "zFake",
                "verificationMethod": "did:key:z6MkhaXgBZDvotDkL5257faiztiGiC2QtKLGpbnnEGta2doK#key-1",
                "created": "2024-01-01T00:00:00Z",
                "proofPurpose": "assertionMethod"
            }
        });
        // Will fail on proof verification (fake signature), but should get past parsing
        let result = import_jsonld_vc(&vc, &resolver);
        assert!(matches!(result, Err(ImportError::ProofInvalid)));
        // The fact that it reached ProofInvalid (not MalformedVc) means parsing + DID resolution succeeded
    }

    #[test]
    fn import_harmony_mldsa65_roundtrip() {
        use harmony_identity::PqPrivateIdentity;
        use rand::rngs::OsRng;

        let identity = PqPrivateIdentity::generate(&mut OsRng);
        let pub_id = identity.public_identity();
        let pubkey = pub_id.verifying_key.as_bytes();

        let issuer_ref = IdentityRef::from(pub_id);
        let mut builder = crate::CredentialBuilder::new(
            issuer_ref, issuer_ref, 1000, 2000, [0u8; 16],
        );
        builder.add_claim(0x0100, b"Alice".to_vec(), [1u8; 16]);
        let payload = builder.signable_payload();
        let signature = identity.sign(&payload).unwrap();
        let (credential, _claims) = builder.build(signature.as_bytes().to_vec());

        let vc_json = crate::jsonld::credential_to_jsonld(&credential, &pubkey, &pubkey).unwrap();

        let resolver = DefaultDidResolver;
        let result = import_jsonld_vc(&vc_json, &resolver);
        assert!(result.is_ok(), "ML-DSA round-trip failed: {:?}", result.err());
        assert_eq!(result.unwrap().proof_type, ImportedProofType::HarmonyCustom);
    }

    // Note: import_jcs_mldsa65_proof test requires constructing a JCS-signed VC
    // with ML-DSA-65. This is expensive (PQ keygen needs 8MB stack). Add it
    // following the same pattern as import_jcs_eddsa_proof but with
    // PqPrivateIdentity::generate() and RUST_MIN_STACK=8388608.

    #[test]
    fn import_identity_hash_derivation() {
        let key = [42u8; 32]; // arbitrary key bytes
        let hash = derive_import_identity_hash(&key);
        // SHA-256(key)[:16]
        let full = harmony_crypto::hash::full_hash(&key);
        assert_eq!(hash, full[..16]);
    }
```

- [ ] **Step 2: Run all tests**

Run: `RUST_MIN_STACK=8388608 cargo test -p harmony-credential --features jsonld`

- [ ] **Step 3: Commit**

```bash
git commit -m "feat(credential/import): additional tests and edge cases

Vocabulary mapping, sentinel salt, timestamp parsing, invalid proof
rejection, missing subject default, identity hash derivation."
```

---

### Task 6: Cleanup and verification

- [ ] **Step 1: Run clippy**

Run: `cargo clippy -p harmony-credential --features jsonld`

- [ ] **Step 2: Run workspace tests**

Run: `RUST_MIN_STACK=8388608 cargo test --workspace`

- [ ] **Step 3: Run fmt**

Run: `cargo fmt -p harmony-credential -- --check`

- [ ] **Step 4: Commit (if fixes needed)**

```bash
git commit -m "chore: clippy/fmt fixes for JSON-LD credential import"
```

---

## Summary

| Task | Description | Key Output |
|------|-------------|------------|
| 1 | Module scaffold + types | `ImportedCredential`, `ImportedProofType`, `ImportError`, stub |
| 2 | Envelope parsing + DID resolution | `parse_vc_envelope()`, `derive_import_identity_hash()`, `parse_iso8601()` |
| 3 | JCS proof verification + claim extraction | `verify_jcs_proof()`, `extract_claims()`, `vocabulary_type_id()` |
| 4 | Harmony proof verification + round-trip | `verify_harmony_proof()`, `extract_harmony_digests()` |
| 5 | Additional tests | Vocabulary, sentinel salt, timestamps, error paths |
| 6 | Cleanup | Clippy clean, workspace tests pass |

**End-to-end flow after this bead:**
```rust
let vc_json: serde_json::Value = serde_json::from_str(json_str)?;
let resolver = harmony_identity::did::DefaultDidResolver;
let imported = harmony_credential::import_jsonld_vc(&vc_json, &resolver)?;
// imported.credential — Harmony binary credential with verified signature
// imported.claims — extracted claims (sentinel salt for external VCs)
// imported.proof_type — HarmonyCustom or JcsDataIntegrity
```
