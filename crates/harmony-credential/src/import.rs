//! W3C JSON-LD import for Harmony verifiable credentials.
//!
//! Parses a W3C VC Data Model 2.0 JSON-LD document, verifies the
//! Data Integrity proof (Harmony custom or JCS-based), extracts claims,
//! and produces a Harmony [`Credential`] + [`Vec<SaltedClaim>`].

use alloc::string::String;
use alloc::vec::Vec;
use serde_json::Value;

use harmony_identity::did::{DidError, DidResolver};

use crate::claim::SaltedClaim;
use crate::credential::Credential;

/// Sentinel salt for imported claims that lack native Harmony salts.
/// All-zeros signals "imported, not selectively disclosable."
pub const IMPORT_SENTINEL_SALT: [u8; 16] = [0u8; 16];

/// Result of importing a W3C JSON-LD Verifiable Credential.
#[derive(Debug, Clone)]
pub struct ImportedCredential {
    pub credential: Credential,
    pub claims: Vec<SaltedClaim>,
    pub proof_type: ImportedProofType,
}

/// How the imported credential's proof was verified.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImportedProofType {
    HarmonyCustom,
    JcsDataIntegrity,
}

/// Errors from credential import operations.
#[derive(Debug)]
pub enum ImportError {
    MalformedVc(String),
    DidResolution(DidError),
    UnsupportedCryptosuite(String),
    ProofInvalid,
    InvalidTimestamp(String),
    ClaimError(String),
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

// ─── Parsed VC envelope ─────────────────────────────────────────────────────

/// All fields extracted from a W3C VC envelope, ready for proof dispatch.
// Fields will be consumed by proof-verification tasks (Tasks 3 & 4).
#[allow(dead_code)]
#[derive(Debug)]
pub(crate) struct VcEnvelope {
    pub issuer_did: String,
    pub subject_did: Option<String>,
    pub not_before: u64,
    pub expires_at: u64,
    pub issued_at: u64,
    /// Cryptosuite name from `proof.cryptosuite`.
    pub cryptosuite: String,
    /// Raw proof value bytes (multibase-decoded).
    pub proof_value: Vec<u8>,
    /// Verification method DID string.
    pub verification_method: String,
    /// Nonce bytes (hex-decoded; all-zeros if absent).
    pub nonce: [u8; 16],
    /// The full proof object, for JCS verification.
    pub proof_obj: Value,
    /// The full VC JSON, minus the proof, for JCS canonicalization.
    pub vc_without_proof: Value,
    /// Status list index (from credentialStatus.statusListIndex), if present.
    pub status_list_index: Option<u32>,
    /// Delegation proof hash (from delegationProof, base64url), if present.
    pub delegation_proof: Option<[u8; 32]>,
}

// ─── Public helpers (also exposed for tests) ────────────────────────────────

/// Derive the identity hash for an imported public key.
///
/// Distinct from native Harmony address derivation (which hashes both
/// enc+sign keys together).  Import uses `SHA-256(sign_key)[:16]` so
/// that a `did:key` → `IdentityRef` mapping is deterministic and
/// round-trippable without the enc key.
pub fn derive_import_identity_hash(public_key: &[u8]) -> [u8; 16] {
    harmony_crypto::hash::truncated_hash(public_key)
}

/// Parse a W3C VC ISO 8601 timestamp (`"YYYY-MM-DDTHH:MM:SSZ"`) to Unix epoch
/// seconds.
///
/// Only the `Z` (UTC) suffix is accepted.  Sub-second precision and timezone
/// offsets are not part of the W3C VC timestamp subset and are rejected.
pub fn parse_iso8601(s: &str) -> Result<u64, ImportError> {
    // Expected format: YYYY-MM-DDTHH:MM:SSZ  (20 chars)
    let s = s
        .strip_suffix('Z')
        .ok_or_else(|| ImportError::InvalidTimestamp(String::from("must end with 'Z'")))?;

    let err = || ImportError::InvalidTimestamp(alloc::format!("invalid ISO 8601: '{s}Z'"));

    // Split on 'T'
    let (date_part, time_part) = s.split_once('T').ok_or_else(err)?;

    // Parse date: YYYY-MM-DD
    let date_parts: Vec<&str> = date_part.split('-').collect();
    if date_parts.len() != 3 {
        return Err(err());
    }
    let year: u64 = date_parts[0].parse().map_err(|_| err())?;
    let month: u64 = date_parts[1].parse().map_err(|_| err())?;
    let day: u64 = date_parts[2].parse().map_err(|_| err())?;

    // Validate ranges
    if !(1..=12).contains(&month) || year < 1970 {
        return Err(err());
    }

    // Validate day against actual month length (including leap year)
    let max_day = match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 => {
            let is_leap = (year % 4 == 0 && year % 100 != 0) || (year % 400 == 0);
            if is_leap { 29 } else { 28 }
        }
        _ => return Err(err()),
    };
    if day < 1 || day > max_day {
        return Err(err());
    }

    // Parse time: HH:MM:SS
    let time_parts: Vec<&str> = time_part.split(':').collect();
    if time_parts.len() != 3 {
        return Err(err());
    }
    let hours: u64 = time_parts[0].parse().map_err(|_| err())?;
    let minutes: u64 = time_parts[1].parse().map_err(|_| err())?;
    let seconds: u64 = time_parts[2].parse().map_err(|_| err())?;

    if hours > 23 || minutes > 59 || seconds > 59 {
        return Err(err());
    }

    // Convert calendar date to days-since-epoch (1970-01-01).
    // Algorithm: civil_from_days inverse (Hatcher/Gregorian).
    let days = ymd_to_days(year, month, day)?;

    let epoch_secs = days * 86400 + hours * 3600 + minutes * 60 + seconds;
    Ok(epoch_secs)
}

/// Convert a calendar date (year, month 1-12, day 1-31) to days since
/// 1970-01-01.  Returns `Err` if the date is before the epoch.
///
/// This is the exact inverse of the `days_to_ymd` algorithm used in
/// `jsonld.rs`.  Uses the Gregorian calendar civil-to-days formula.
fn ymd_to_days(year: u64, month: u64, day: u64) -> Result<u64, ImportError> {
    // Shift so March is month 0 to simplify leap-year math.
    let (y, m) = if month <= 2 {
        (year - 1, month + 9)
    } else {
        (year, month - 3)
    };

    let era = y / 400;
    let yoe = y - era * 400; // year of era [0, 399]
    let doy = (153 * m + 2) / 5 + day - 1; // day of year from March 1 [0, 365]
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy; // day of era [0, 146096]

    // days since the proleptic Gregorian calendar epoch (0000-03-01),
    // *without* the 719468 bias.  That is: era * 146097 + doe.
    // To get days since 1970-01-01 we subtract 719468 (the number of
    // days from 0000-03-01 to 1970-01-01).
    let days_since_civil_epoch = era * 146097 + doe;

    if days_since_civil_epoch < 719468 {
        return Err(ImportError::InvalidTimestamp(String::from(
            "date before Unix epoch (1970-01-01)",
        )));
    }
    Ok(days_since_civil_epoch - 719468)
}

// ─── Harmony custom proof helpers ────────────────────────────────────────────

/// Extract the 32-byte claim digests from a Harmony-format `credentialSubject`.
///
/// Each element of `credentialSubject.claims` must have a `digest` field
/// encoded as base64url (no padding).
fn extract_harmony_digests(subject: &Value) -> Result<Vec<[u8; 32]>, ImportError> {
    use base64::Engine;
    let engine = base64::engine::general_purpose::URL_SAFE_NO_PAD;

    let claims = subject
        .get("claims")
        .and_then(Value::as_array)
        .ok_or_else(|| ImportError::ClaimError(String::from("missing claims array")))?;

    let mut digests = Vec::new();
    for item in claims {
        let digest_b64 = item
            .get("digest")
            .and_then(Value::as_str)
            .ok_or_else(|| ImportError::ClaimError(String::from("claim missing digest")))?;
        let bytes = engine
            .decode(digest_b64)
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

/// Reconstruct the postcard signable payload and verify the Harmony proof.
///
/// Harmony proofs sign the postcard-serialized `SignablePayload` (via
/// [`Credential::signable_bytes()`]), NOT a JCS/RDFC canonicalization.
fn verify_harmony_proof(
    env: &VcEnvelope,
    issuer_ref: &harmony_identity::IdentityRef,
    subject_ref: &harmony_identity::IdentityRef,
    issuer_key: &[u8],
    claim_digests: &[[u8; 32]],
) -> Result<(), ImportError> {
    // Reconstruct a temporary Credential so we can call signable_bytes().
    // The `signature` field is not part of the signable payload — any
    // value works here; we use the proof_value for clarity.
    let credential = Credential {
        issuer: *issuer_ref,
        subject: *subject_ref,
        claim_digests: claim_digests.to_vec(),
        status_list_index: env.status_list_index,
        not_before: env.not_before,
        expires_at: env.expires_at,
        issued_at: env.issued_at,
        nonce: env.nonce,
        proof: env.delegation_proof,
        signature: env.proof_value.clone(),
    };

    let payload = credential.signable_bytes();
    harmony_identity::verify_signature(issuer_ref.suite, issuer_key, &payload, &env.proof_value)
        .map_err(|_| ImportError::ProofInvalid)
}

// ─── JCS proof verification ──────────────────────────────────────────────────

/// Verify a W3C Data Integrity JCS proof.
///
/// Algorithm (§4.3):
/// 1. Remove `proofValue` from `proof_obj` → proof_options
/// 2. `proof_hash = SHA-256(JCS(proof_options))`
/// 3. `doc_hash   = SHA-256(JCS(vc_without_proof))`
/// 4. `verify_bytes = proof_hash || doc_hash`
/// 5. Verify signature over `verify_bytes` with issuer key
fn verify_jcs_proof(
    env: &VcEnvelope,
    public_key: &[u8],
    suite: harmony_identity::CryptoSuite,
) -> Result<(), ImportError> {
    // Build proof options: proof object without proofValue
    let mut proof_options = env.proof_obj.clone();
    if let Some(obj) = proof_options.as_object_mut() {
        obj.remove("proofValue");
    }

    let proof_hash = harmony_crypto::hash::full_hash(&harmony_jcs::canonicalize(&proof_options));
    let doc_hash =
        harmony_crypto::hash::full_hash(&harmony_jcs::canonicalize(&env.vc_without_proof));

    let mut verify_bytes = Vec::with_capacity(64);
    verify_bytes.extend_from_slice(&proof_hash);
    verify_bytes.extend_from_slice(&doc_hash);

    harmony_identity::verify_signature(suite, public_key, &verify_bytes, &env.proof_value)
        .map_err(|_| ImportError::ProofInvalid)
}

// ─── Claim extraction ─────────────────────────────────────────────────────────

/// Map a W3C VC claim name to a Harmony vocabulary type ID.
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

/// Extract `SaltedClaim`s from a `credentialSubject` JSON object.
///
/// Handles two formats:
/// - **Harmony format**: `credentialSubject.claims` array with `digest`/`typeId`/`value`/`salt`
/// - **External format**: arbitrary key-value pairs (skip `id`, `type`, `@context`)
fn extract_claims(subject: &Value) -> Result<Vec<SaltedClaim>, ImportError> {
    let subj_obj = subject.as_object().ok_or_else(|| {
        ImportError::MalformedVc(String::from("credentialSubject must be an object"))
    })?;

    // Harmony format: has a `claims` array
    if let Some(Value::Array(claims_arr)) = subj_obj.get("claims") {
        return extract_harmony_claims(claims_arr);
    }

    // External format: arbitrary fields
    extract_external_claims(subj_obj)
}

/// Extract claims from the Harmony `claims` array format.
fn extract_harmony_claims(claims_arr: &[Value]) -> Result<Vec<SaltedClaim>, ImportError> {
    use base64::Engine;

    let mut out = Vec::with_capacity(claims_arr.len());
    for item in claims_arr {
        let obj = item.as_object().ok_or_else(|| {
            ImportError::MalformedVc(String::from("claim array item must be object"))
        })?;

        // digest is required
        let digest_b64 = obj
            .get("digest")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ImportError::MalformedVc(String::from("claim item missing 'digest'")))?;

        let _digest_bytes = base64::engine::general_purpose::URL_SAFE_NO_PAD
            .decode(digest_b64)
            .map_err(|e| ImportError::EncodingError(alloc::format!("claim digest decode: {e}")))?;

        // Salt: from `salt` field (base64url) or sentinel
        let salt: [u8; 16] = if let Some(salt_str) = obj.get("salt").and_then(|v| v.as_str()) {
            let salt_bytes = base64::engine::general_purpose::URL_SAFE_NO_PAD
                .decode(salt_str)
                .map_err(|e| {
                    ImportError::EncodingError(alloc::format!("claim salt decode: {e}"))
                })?;
            if salt_bytes.len() != 16 {
                return Err(ImportError::MalformedVc(alloc::format!(
                    "claim salt must be 16 bytes, got {}",
                    salt_bytes.len()
                )));
            }
            let mut arr = [0u8; 16];
            arr.copy_from_slice(&salt_bytes);
            arr
        } else {
            IMPORT_SENTINEL_SALT
        };

        // typeId + value (optional)
        let type_id = obj
            .get("typeId")
            .and_then(|v| v.as_u64())
            .and_then(|n| u16::try_from(n).ok())
            .unwrap_or(0x8000);

        let value: Vec<u8> = if let Some(val_str) = obj.get("value").and_then(|v| v.as_str()) {
            base64::engine::general_purpose::URL_SAFE_NO_PAD
                .decode(val_str)
                .map_err(|e| {
                    ImportError::EncodingError(alloc::format!("claim value decode: {e}"))
                })?
        } else {
            // No value: store the digest bytes as the value (opaque)
            _digest_bytes
        };

        out.push(SaltedClaim {
            claim: crate::claim::Claim { type_id, value },
            salt,
        });
    }
    Ok(out)
}

/// Extract claims from arbitrary JSON-LD `credentialSubject` key-value pairs.
fn extract_external_claims(
    subj_obj: &serde_json::Map<String, Value>,
) -> Result<Vec<SaltedClaim>, ImportError> {
    let mut out = Vec::new();
    let skip = ["id", "type", "@context"];

    for (key, val) in subj_obj {
        if skip.contains(&key.as_str()) {
            continue;
        }
        let type_id = vocabulary_type_id(key);
        // For strings, store raw bytes (no JSON quotes) to match native claims.
        // For other types (numbers, booleans, objects, arrays), use JSON encoding.
        let value = match val {
            Value::String(s) => s.as_bytes().to_vec(),
            _ => val.to_string().into_bytes(),
        };
        out.push(SaltedClaim {
            claim: crate::claim::Claim { type_id, value },
            salt: IMPORT_SENTINEL_SALT,
        });
    }
    Ok(out)
}

// ─── VC envelope parser ──────────────────────────────────────────────────────

/// Extract all required fields from a W3C VC 2.0 JSON-LD document.
pub(crate) fn parse_vc_envelope(vc: &Value) -> Result<VcEnvelope, ImportError> {
    let vc_obj = vc
        .as_object()
        .ok_or_else(|| ImportError::MalformedVc(String::from("VC must be a JSON object")))?;

    // ── issuer ────────────────────────────────────────────────────────────────
    let issuer_did = match vc_obj.get("issuer") {
        Some(Value::String(s)) => s.clone(),
        Some(Value::Object(obj)) => obj
            .get("id")
            .and_then(|v| v.as_str())
            .map(String::from)
            .ok_or_else(|| {
                ImportError::MalformedVc(String::from("issuer object missing 'id' field"))
            })?,
        _ => {
            return Err(ImportError::MalformedVc(String::from(
                "missing or invalid 'issuer' field",
            )))
        }
    };

    // ── credentialSubject ─────────────────────────────────────────────────────
    let subject_obj = vc_obj
        .get("credentialSubject")
        .and_then(|v| v.as_object())
        .ok_or_else(|| {
            ImportError::MalformedVc(String::from("missing or invalid 'credentialSubject' field"))
        })?;

    let subject_did = subject_obj
        .get("id")
        .and_then(|v| v.as_str())
        .map(String::from);

    // ── timestamps ────────────────────────────────────────────────────────────
    let not_before = match vc_obj.get("validFrom") {
        Some(Value::String(s)) => parse_iso8601(s)?,
        Some(_) => {
            return Err(ImportError::MalformedVc(String::from(
                "'validFrom' must be a string",
            )))
        }
        None => {
            return Err(ImportError::MalformedVc(String::from(
                "missing required 'validFrom' field",
            )))
        }
    };

    let expires_at = match vc_obj.get("validUntil") {
        Some(Value::String(s)) => parse_iso8601(s)?,
        Some(_) => {
            return Err(ImportError::MalformedVc(String::from(
                "'validUntil' must be a string",
            )))
        }
        // Optional: default to not_before + 10 years if absent
        None => not_before + 315_576_000,
    };

    // ── proof ─────────────────────────────────────────────────────────────────
    let proof_val = vc_obj
        .get("proof")
        .ok_or_else(|| ImportError::MalformedVc(String::from("missing 'proof' field")))?;

    let proof_obj_map = proof_val
        .as_object()
        .ok_or_else(|| ImportError::MalformedVc(String::from("'proof' must be a JSON object")))?;

    let cryptosuite = proof_obj_map
        .get("cryptosuite")
        .and_then(|v| v.as_str())
        .ok_or_else(|| ImportError::MalformedVc(String::from("proof missing 'cryptosuite' field")))?
        .to_owned();

    let proof_value_str = proof_obj_map
        .get("proofValue")
        .and_then(|v| v.as_str())
        .ok_or_else(|| {
            ImportError::MalformedVc(String::from("proof missing 'proofValue' field"))
        })?;

    // proofValue is multibase base58btc (prefix 'z') per W3C Data Integrity spec
    let proof_value_b58 = proof_value_str.strip_prefix('z').ok_or_else(|| {
        ImportError::MalformedVc(String::from(
            "proofValue must use multibase base58btc prefix 'z'",
        ))
    })?;
    let proof_value = bs58::decode(proof_value_b58)
        .into_vec()
        .map_err(|e| ImportError::EncodingError(alloc::format!("proofValue base58 decode: {e}")))?;

    let verification_method = proof_obj_map
        .get("verificationMethod")
        .and_then(|v| v.as_str())
        .ok_or_else(|| {
            ImportError::MalformedVc(String::from("proof missing 'verificationMethod' field"))
        })?
        .to_owned();

    let created_str = proof_obj_map
        .get("created")
        .and_then(|v| v.as_str())
        .ok_or_else(|| ImportError::MalformedVc(String::from("proof missing 'created' field")))?;
    let issued_at = parse_iso8601(created_str)?;

    // Nonce: hex-encoded in Harmony export; all-zeros if absent
    let nonce = match proof_obj_map.get("nonce").and_then(|v| v.as_str()) {
        Some(hex_str) => {
            let bytes = hex::decode(hex_str)
                .map_err(|e| ImportError::EncodingError(alloc::format!("nonce hex decode: {e}")))?;
            if bytes.len() != 16 {
                return Err(ImportError::MalformedVc(alloc::format!(
                    "nonce must be 16 bytes, got {}",
                    bytes.len()
                )));
            }
            let mut arr = [0u8; 16];
            arr.copy_from_slice(&bytes);
            arr
        }
        None => [0u8; 16],
    };

    // Build VC-without-proof for JCS canonicalization (future use)
    let mut vc_without_proof = vc.clone();
    if let Some(obj) = vc_without_proof.as_object_mut() {
        obj.remove("proof");
    }

    // Extract optional status list index (from credentialStatus.statusListIndex)
    let status_list_index = vc
        .get("credentialStatus")
        .and_then(|cs| cs.get("statusListIndex"))
        .and_then(|v| v.as_str())
        .and_then(|s| s.parse::<u32>().ok());

    // Extract optional delegation proof hash (from delegationProof, base64url)
    let delegation_proof = vc
        .get("delegationProof")
        .and_then(Value::as_str)
        .and_then(|b64| {
            use base64::Engine;
            let bytes = base64::engine::general_purpose::URL_SAFE_NO_PAD
                .decode(b64)
                .ok()?;
            if bytes.len() == 32 {
                let mut arr = [0u8; 32];
                arr.copy_from_slice(&bytes);
                Some(arr)
            } else {
                None
            }
        });

    Ok(VcEnvelope {
        issuer_did,
        subject_did,
        not_before,
        expires_at,
        issued_at,
        cryptosuite,
        proof_value,
        verification_method,
        nonce,
        proof_obj: proof_val.clone(),
        vc_without_proof,
        status_list_index,
        delegation_proof,
    })
}

// ─── Main entry point ────────────────────────────────────────────────────────

/// Import a W3C Verifiable Credential from JSON-LD format.
///
/// Parses the VC envelope, resolves the issuer and subject DIDs, then
/// dispatches on the cryptosuite.  Proof verification is performed by the
/// suite-specific handlers (implemented in later tasks).
pub fn import_jsonld_vc(
    vc_json: &Value,
    resolver: &impl DidResolver,
) -> Result<ImportedCredential, ImportError> {
    let env = parse_vc_envelope(vc_json)?;

    // Resolve the signing key from verificationMethod (may differ from issuer DID
    // in general, though for did:key they resolve to the same key). Per W3C Data
    // Integrity, verificationMethod identifies the actual signing key.
    let issuer_resolved = resolver.resolve(&env.verification_method)?;
    // Use the issuer DID for the identity hash (not verificationMethod)
    let issuer_did_resolved = resolver.resolve(&env.issuer_did)?;
    let issuer_hash = derive_import_identity_hash(&issuer_did_resolved.public_key);
    let issuer_ref = harmony_identity::IdentityRef {
        hash: issuer_hash,
        suite: issuer_did_resolved.suite,
    };

    // Resolve subject DID (defaults to issuer if absent).
    let subject_ref = if let Some(ref sub_did) = env.subject_did {
        let sub_resolved = resolver.resolve(sub_did)?;
        let sub_hash = derive_import_identity_hash(&sub_resolved.public_key);
        harmony_identity::IdentityRef {
            hash: sub_hash,
            suite: sub_resolved.suite,
        }
    } else {
        issuer_ref
    };

    // Dispatch on cryptosuite
    match env.cryptosuite.as_str() {
        "harmony-eddsa-2022" | "harmony-mldsa65-2025" => {
            // The credentialSubject is present in the full vc_json (and also in
            // vc_without_proof, since credentialSubject is not the proof field).
            let credential_subject = vc_json.get("credentialSubject").ok_or_else(|| {
                ImportError::MalformedVc(String::from("missing credentialSubject in VC"))
            })?;

            let claim_digests = extract_harmony_digests(credential_subject)?;
            let claims = extract_claims(credential_subject)?;

            verify_harmony_proof(
                &env,
                &issuer_ref,
                &subject_ref,
                &issuer_resolved.public_key,
                &claim_digests,
            )?;

            let credential = Credential {
                issuer: issuer_ref,
                subject: subject_ref,
                claim_digests,
                status_list_index: env.status_list_index,
                not_before: env.not_before,
                expires_at: env.expires_at,
                issued_at: env.issued_at,
                nonce: env.nonce,
                proof: env.delegation_proof,
                signature: env.proof_value.clone(),
            };

            Ok(ImportedCredential {
                credential,
                claims,
                proof_type: ImportedProofType::HarmonyCustom,
            })
        }
        "eddsa-jcs-2022" | "mldsa65-jcs-2024" | "di-mldsa-jcs-2025" => {
            // Verify JCS Data Integrity proof
            verify_jcs_proof(&env, &issuer_resolved.public_key, issuer_resolved.suite)?;

            // Extract claims from credentialSubject
            let credential_subject =
                env.vc_without_proof
                    .get("credentialSubject")
                    .ok_or_else(|| {
                        ImportError::MalformedVc(String::from("missing credentialSubject in VC"))
                    })?;
            let claims = extract_claims(credential_subject)?;
            let claim_digests = claims.iter().map(|sc| sc.digest()).collect();

            let credential = Credential {
                issuer: issuer_ref,
                subject: subject_ref,
                claim_digests,
                status_list_index: env.status_list_index,
                not_before: env.not_before,
                expires_at: env.expires_at,
                issued_at: env.issued_at,
                nonce: env.nonce,
                proof: env.delegation_proof,
                signature: env.proof_value.clone(),
            };

            Ok(ImportedCredential {
                credential,
                claims,
                proof_type: ImportedProofType::JcsDataIntegrity,
            })
        }
        other => Err(ImportError::UnsupportedCryptosuite(String::from(other))),
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use harmony_identity::did::DefaultDidResolver;

    #[test]
    fn import_malformed_vc_rejected() {
        let resolver = DefaultDidResolver;
        let vc = serde_json::json!({"foo": "bar"});
        let result = import_jsonld_vc(&vc, &resolver);
        assert!(matches!(result, Err(ImportError::MalformedVc(_))));
    }

    #[test]
    fn import_missing_proof_rejected() {
        let resolver = DefaultDidResolver;
        let vc = serde_json::json!({
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
        let vc = serde_json::json!({
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
        assert!(matches!(
            result,
            Err(ImportError::UnsupportedCryptosuite(_))
        ));
    }

    #[test]
    fn import_timestamps_parsed() {
        let ts = parse_iso8601("2024-01-15T12:30:45Z").unwrap();
        assert!(ts > 1_700_000_000, "timestamp should be after 2023");
        assert!(ts < 1_800_000_000, "timestamp should be before 2027");
    }

    // ── Additional unit tests for parse_iso8601 ──────────────────────────────

    #[test]
    fn parse_iso8601_epoch() {
        // 1970-01-01T00:00:00Z = Unix epoch = 0
        assert_eq!(parse_iso8601("1970-01-01T00:00:00Z").unwrap(), 0);
    }

    #[test]
    fn parse_iso8601_known_date() {
        // 2024-01-01T00:00:00Z
        // Days from 1970-01-01 to 2024-01-01 = 54 years
        // Use the known value: 1704067200
        let ts = parse_iso8601("2024-01-01T00:00:00Z").unwrap();
        assert_eq!(ts, 1_704_067_200);
    }

    #[test]
    fn parse_iso8601_rejects_no_z_suffix() {
        assert!(matches!(
            parse_iso8601("2024-01-01T00:00:00"),
            Err(ImportError::InvalidTimestamp(_))
        ));
    }

    #[test]
    fn parse_iso8601_rejects_before_epoch() {
        assert!(matches!(
            parse_iso8601("1969-12-31T23:59:59Z"),
            Err(ImportError::InvalidTimestamp(_))
        ));
    }

    #[test]
    fn parse_iso8601_roundtrip_with_export() {
        // Test that parse_iso8601 is the inverse of the jsonld::epoch_to_iso8601
        // for several known timestamps.
        let known_pairs: &[(&str, u64)] = &[
            ("1970-01-01T00:00:00Z", 0),
            ("2000-01-01T00:00:00Z", 946_684_800),
            ("2024-06-15T08:45:30Z", 1_718_441_130),
        ];
        for (s, expected) in known_pairs {
            let parsed = parse_iso8601(s).unwrap();
            assert_eq!(parsed, *expected, "mismatch for {s}");
        }
    }

    // ── VcEnvelope parser tests ───────────────────────────────────────────────

    #[test]
    fn parse_vc_envelope_issuer_as_string() {
        // proofValue must use multibase base58btc prefix 'z' with valid base58 payload.
        let vc = serde_json::json!({
            "issuer": "did:key:z6MkTest",
            "credentialSubject": {},
            "validFrom": "2024-01-01T00:00:00Z",
            "proof": {
                "cryptosuite": "eddsa-jcs-2022",
                "proofValue": "z3yMkTest",
                "verificationMethod": "did:key:z6MkTest#key-1",
                "created": "2024-01-01T00:00:00Z"
            }
        });
        let env = parse_vc_envelope(&vc).unwrap();
        assert_eq!(env.issuer_did, "did:key:z6MkTest");
        assert!(env.subject_did.is_none());
        assert_eq!(env.cryptosuite, "eddsa-jcs-2022");
    }

    #[test]
    fn parse_vc_envelope_issuer_as_object() {
        let vc = serde_json::json!({
            "issuer": {"id": "did:key:z6MkObj", "name": "Test Issuer"},
            "credentialSubject": {"id": "did:key:z6MkSubj"},
            "validFrom": "2024-01-01T00:00:00Z",
            "proof": {
                "cryptosuite": "eddsa-jcs-2022",
                "proofValue": "z3yMkTest",
                "verificationMethod": "did:key:z6MkObj#key-1",
                "created": "2024-01-01T00:00:00Z"
            }
        });
        let env = parse_vc_envelope(&vc).unwrap();
        assert_eq!(env.issuer_did, "did:key:z6MkObj");
        assert_eq!(env.subject_did.as_deref(), Some("did:key:z6MkSubj"));
    }

    #[test]
    fn parse_vc_envelope_nonce_absent_is_zeros() {
        let vc = serde_json::json!({
            "issuer": "did:key:z6MkTest",
            "credentialSubject": {},
            "validFrom": "2024-01-01T00:00:00Z",
            "proof": {
                "cryptosuite": "eddsa-jcs-2022",
                "proofValue": "z3yMkTest",
                "verificationMethod": "did:key:z6MkTest#key-1",
                "created": "2024-01-01T00:00:00Z"
            }
        });
        let env = parse_vc_envelope(&vc).unwrap();
        assert_eq!(env.nonce, [0u8; 16]);
    }

    #[test]
    fn parse_vc_envelope_nonce_hex_decoded() {
        let vc = serde_json::json!({
            "issuer": "did:key:z6MkTest",
            "credentialSubject": {},
            "validFrom": "2024-01-01T00:00:00Z",
            "proof": {
                "cryptosuite": "eddsa-jcs-2022",
                "proofValue": "z3yMkTest",
                "verificationMethod": "did:key:z6MkTest#key-1",
                "created": "2024-01-01T00:00:00Z",
                "nonce": "0102030405060708090a0b0c0d0e0f10"
            }
        });
        let env = parse_vc_envelope(&vc).unwrap();
        assert_eq!(
            env.nonce,
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
        );
    }

    #[test]
    fn parse_vc_envelope_missing_issuer_rejected() {
        let vc = serde_json::json!({
            "credentialSubject": {},
            "validFrom": "2024-01-01T00:00:00Z",
            "proof": {
                "cryptosuite": "eddsa-jcs-2022",
                "proofValue": "z3yMkTest",
                "verificationMethod": "did:key:z6MkTest#key-1",
                "created": "2024-01-01T00:00:00Z"
            }
        });
        assert!(matches!(
            parse_vc_envelope(&vc),
            Err(ImportError::MalformedVc(_))
        ));
    }

    #[test]
    fn parse_vc_envelope_missing_valid_from_rejected() {
        let vc = serde_json::json!({
            "issuer": "did:key:z6MkTest",
            "credentialSubject": {},
            "proof": {
                "cryptosuite": "eddsa-jcs-2022",
                "proofValue": "z3yMkTest",
                "verificationMethod": "did:key:z6MkTest#key-1",
                "created": "2024-01-01T00:00:00Z"
            }
        });
        assert!(matches!(
            parse_vc_envelope(&vc),
            Err(ImportError::MalformedVc(_))
        ));
    }

    #[test]
    fn parse_vc_envelope_proof_value_not_multibase_rejected() {
        let vc = serde_json::json!({
            "issuer": "did:key:z6MkTest",
            "credentialSubject": {},
            "validFrom": "2024-01-01T00:00:00Z",
            "proof": {
                "cryptosuite": "eddsa-jcs-2022",
                "proofValue": "notmultibase",
                "verificationMethod": "did:key:z6MkTest#key-1",
                "created": "2024-01-01T00:00:00Z"
            }
        });
        assert!(matches!(
            parse_vc_envelope(&vc),
            Err(ImportError::MalformedVc(_))
        ));
    }

    #[test]
    fn derive_import_identity_hash_is_deterministic() {
        let key = [0x42u8; 32];
        let h1 = derive_import_identity_hash(&key);
        let h2 = derive_import_identity_hash(&key);
        assert_eq!(h1, h2);
    }

    #[test]
    fn derive_import_identity_hash_differs_from_empty() {
        let h_non_empty = derive_import_identity_hash(&[0x42u8; 32]);
        let h_empty = derive_import_identity_hash(&[]);
        assert_ne!(h_non_empty, h_empty);
    }

    // ── Harmony-suite VC with invalid proof (bad signature) ──────────────────

    #[test]
    fn import_harmony_eddsa_bad_proof_rejected() {
        // A Harmony-suite VC that has no `claims` array fails at digest extraction.
        let resolver = DefaultDidResolver;
        let vc = serde_json::json!({
            "issuer": "did:key:z6MkhaXgBZDvotDkL5257faiztiGiC2QtKLGpbnnEGta2doK",
            "credentialSubject": {"id": "did:key:z6MkhaXgBZDvotDkL5257faiztiGiC2QtKLGpbnnEGta2doK"},
            "validFrom": "2024-01-01T00:00:00Z",
            "proof": {
                "type": "DataIntegrityProof",
                "cryptosuite": "harmony-eddsa-2022",
                "proofValue": format!("z{}", bs58::encode([0u8; 64]).into_string()),
                "verificationMethod": "did:key:z6MkhaXgBZDvotDkL5257faiztiGiC2QtKLGpbnnEGta2doK#key-1",
                "created": "2024-01-01T00:00:00Z",
                "proofPurpose": "assertionMethod"
            }
        });
        let result = import_jsonld_vc(&vc, &resolver);
        // Missing `claims` array → ClaimError
        assert!(matches!(result, Err(ImportError::ClaimError(_))));
    }

    #[test]
    fn import_jcs_bad_proof_rejected() {
        // Valid DID key but invalid (all-zero) proof bytes → ProofInvalid
        let resolver = DefaultDidResolver;
        let vc = serde_json::json!({
            "issuer": "did:key:z6MkhaXgBZDvotDkL5257faiztiGiC2QtKLGpbnnEGta2doK",
            "credentialSubject": {"id": "did:key:z6MkhaXgBZDvotDkL5257faiztiGiC2QtKLGpbnnEGta2doK"},
            "validFrom": "2024-01-01T00:00:00Z",
            "proof": {
                "type": "DataIntegrityProof",
                "cryptosuite": "eddsa-jcs-2022",
                // base58-encode 64 zero bytes = "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111"
                "proofValue": format!("z{}", bs58::encode([0u8; 64]).into_string()),
                "verificationMethod": "did:key:z6MkhaXgBZDvotDkL5257faiztiGiC2QtKLGpbnnEGta2doK#key-1",
                "created": "2024-01-01T00:00:00Z",
                "proofPurpose": "assertionMethod"
            }
        });
        let result = import_jsonld_vc(&vc, &resolver);
        assert!(matches!(result, Err(ImportError::ProofInvalid)));
    }

    #[test]
    fn import_jcs_eddsa_proof() {
        use harmony_identity::PrivateIdentity;
        use rand::rngs::OsRng;

        let identity = PrivateIdentity::generate(&mut OsRng);
        let pub_id = identity.public_identity();
        let pubkey_bytes = pub_id.verifying_key.as_bytes();

        let id_hash = derive_import_identity_hash(pubkey_bytes);
        let id_ref = harmony_identity::IdentityRef {
            hash: id_hash,
            suite: harmony_identity::CryptoSuite::Ed25519,
        };
        let issuer_did = crate::jsonld::identity_to_did_key(&id_ref, pubkey_bytes).unwrap();

        // Build VC without proof
        let mut vc = serde_json::json!({
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

        // Build proof options (no proofValue)
        let proof_options = serde_json::json!({
            "type": "DataIntegrityProof",
            "cryptosuite": "eddsa-jcs-2022",
            "created": "2024-01-01T00:00:00Z",
            "verificationMethod": format!("{}#key-1", issuer_did),
            "proofPurpose": "assertionMethod"
        });

        // JCS Data Integrity signing
        let proof_hash =
            harmony_crypto::hash::full_hash(&harmony_jcs::canonicalize(&proof_options));
        let doc_hash = harmony_crypto::hash::full_hash(&harmony_jcs::canonicalize(&vc));
        let mut verify_bytes = Vec::with_capacity(64);
        verify_bytes.extend_from_slice(&proof_hash);
        verify_bytes.extend_from_slice(&doc_hash);

        let signature = identity.sign(&verify_bytes);
        let proof_value = format!("z{}", bs58::encode(&signature).into_string());

        // Attach proof
        let mut proof = proof_options.clone();
        proof
            .as_object_mut()
            .unwrap()
            .insert("proofValue".into(), serde_json::Value::String(proof_value));
        vc.as_object_mut().unwrap().insert("proof".into(), proof);

        // Import
        let resolver = DefaultDidResolver;
        let result = import_jsonld_vc(&vc, &resolver);
        assert!(result.is_ok(), "JCS import failed: {:?}", result.err());

        let imported = result.unwrap();
        assert_eq!(imported.proof_type, ImportedProofType::JcsDataIntegrity);
        assert_eq!(imported.claims.len(), 2); // givenName + familyName
    }

    // ── Harmony export → import round-trip ───────────────────────────────────

    #[test]
    fn import_harmony_eddsa_roundtrip() {
        use harmony_identity::PrivateIdentity;
        use rand::rngs::OsRng;

        let identity = PrivateIdentity::generate(&mut OsRng);
        let pub_id = identity.public_identity();
        let pubkey = pub_id.verifying_key.as_bytes();

        // The import path derives identity hash as SHA-256(Ed25519_pub)[:16].
        // To make the round-trip verifiable, build the credential with the same
        // hash derivation so that the signable payload matches on import.
        let import_hash = derive_import_identity_hash(pubkey);
        let issuer_ref = harmony_identity::IdentityRef {
            hash: import_hash,
            suite: harmony_identity::CryptoSuite::Ed25519,
        };

        // Build credential with a claim
        let mut builder =
            crate::CredentialBuilder::new(issuer_ref, issuer_ref, 1000, 2000, [0u8; 16]);
        builder.add_claim(0x0100, b"Alice".to_vec(), [1u8; 16]);

        let payload = builder.signable_payload();
        let signature = identity.sign(&payload);
        let (credential, _claims) = builder.build(signature.to_vec());

        // Export to JSON-LD using the Ed25519 verifying key for both issuer and subject.
        let vc_json = crate::jsonld::credential_to_jsonld(&credential, pubkey, pubkey).unwrap();

        // Import back — resolver decodes the did:key and returns the Ed25519 verifying key.
        let resolver = DefaultDidResolver;
        let result = import_jsonld_vc(&vc_json, &resolver);
        assert!(result.is_ok(), "round-trip failed: {:?}", result.err());

        let imported = result.unwrap();
        assert_eq!(imported.proof_type, ImportedProofType::HarmonyCustom);

        assert_eq!(
            imported.credential.issuer.suite,
            harmony_identity::CryptoSuite::Ed25519
        );
        assert_eq!(
            imported.credential.subject.suite,
            harmony_identity::CryptoSuite::Ed25519
        );

        // Claim digests must survive the round-trip
        assert_eq!(imported.credential.claim_digests.len(), 1);
    }

    // ── Vocabulary mapping ───────────────────────────────────────────────────

    #[test]
    fn import_external_claims_vocabulary() {
        assert_eq!(vocabulary_type_id("givenName"), 0x0100);
        assert_eq!(vocabulary_type_id("given_name"), 0x0100);
        assert_eq!(vocabulary_type_id("familyName"), 0x0101);
        assert_eq!(vocabulary_type_id("email"), 0x0110);
    }

    #[test]
    fn import_external_claims_unknown() {
        let id = vocabulary_type_id("customField");
        assert!(id & 0x8000 != 0, "unknown claims should have high bit set");
    }

    #[test]
    fn import_sentinel_salt() {
        assert_eq!(IMPORT_SENTINEL_SALT, [0u8; 16]);
    }

    #[test]
    fn import_identity_hash_derivation() {
        let key = [42u8; 32];
        let hash = derive_import_identity_hash(&key);
        let full = harmony_crypto::hash::full_hash(&key);
        assert_eq!(hash, full[..16]);
    }

    #[test]
    fn import_invalid_proof_rejected() {
        // Use a valid-base58btc-encoded but cryptographically invalid 64-byte signature.
        let resolver = DefaultDidResolver;
        let vc = serde_json::json!({
            "issuer": "did:key:z6MkhaXgBZDvotDkL5257faiztiGiC2QtKLGpbnnEGta2doK",
            "credentialSubject": {
                "id": "did:key:z6MkhaXgBZDvotDkL5257faiztiGiC2QtKLGpbnnEGta2doK",
                "givenName": "Alice"
            },
            "validFrom": "2024-01-01T00:00:00Z",
            "proof": {
                "type": "DataIntegrityProof",
                "cryptosuite": "eddsa-jcs-2022",
                "proofValue": format!("z{}", bs58::encode([0u8; 64]).into_string()),
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
        // Use a valid-base58btc-encoded but cryptographically invalid 64-byte signature.
        let resolver = DefaultDidResolver;
        let vc = serde_json::json!({
            "issuer": "did:key:z6MkhaXgBZDvotDkL5257faiztiGiC2QtKLGpbnnEGta2doK",
            "credentialSubject": {
                "givenName": "Alice"
            },
            "validFrom": "2024-01-01T00:00:00Z",
            "proof": {
                "type": "DataIntegrityProof",
                "cryptosuite": "eddsa-jcs-2022",
                "proofValue": format!("z{}", bs58::encode([0u8; 64]).into_string()),
                "verificationMethod": "did:key:z6MkhaXgBZDvotDkL5257faiztiGiC2QtKLGpbnnEGta2doK#key-1",
                "created": "2024-01-01T00:00:00Z",
                "proofPurpose": "assertionMethod"
            }
        });
        // Will fail on proof verification (all-zero sig), but should reach that point
        // (not fail on missing subject). ProofInvalid means parsing succeeded.
        let result = import_jsonld_vc(&vc, &resolver);
        assert!(matches!(result, Err(ImportError::ProofInvalid)));
    }
}
