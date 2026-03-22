//! W3C JSON-LD export for Harmony verifiable credentials.

use alloc::string::String;
use alloc::vec::Vec;
use base64::Engine;
use harmony_identity::{CryptoSuite, IdentityRef};
use serde_json::{json, Value};

use crate::credential::Credential;
use crate::disclosure::Presentation;
use crate::error::CredentialError;

/// Harmony-specific JSON-LD context URL for extension fields
/// (`delegationProof`, claim types, status list URIs).
const HARMONY_CONTEXT: &str = "https://harmony.id/ns/credentials/v1";

/// Encode an IdentityRef + public key bytes as a `did:key` string.
///
/// Returns `Err(SignatureInvalid)` if the key length doesn't match the suite
/// (32 bytes for Ed25519, 1952 bytes for ML-DSA-65).
pub fn identity_to_did_key(
    identity: &IdentityRef,
    public_key: &[u8],
) -> Result<String, CredentialError> {
    let expected_len = match identity.suite {
        CryptoSuite::Ed25519 => 32,
        CryptoSuite::MlDsa65 | CryptoSuite::MlDsa65Rotatable => 1952,
    };
    if public_key.len() != expected_len {
        return Err(CredentialError::SignatureInvalid);
    }
    let multicodec = identity.suite.signing_multicodec();

    // Encode multicodec as unsigned varint (LEB128)
    let mut varint = Vec::with_capacity(3);
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

    let mut payload = Vec::with_capacity(varint.len() + public_key.len());
    payload.extend_from_slice(&varint);
    payload.extend_from_slice(public_key);

    let encoded = bs58::encode(&payload).into_string();
    Ok(alloc::format!("did:key:z{encoded}"))
}

/// Export a Credential to W3C VC Data Model 2.0 JSON-LD.
///
/// `issuer_key` and `subject_key` are the raw public key bytes for
/// did:key encoding. Key length must match the credential's CryptoSuite
/// (32 bytes for Ed25519, 1952 bytes for ML-DSA-65).
pub fn credential_to_jsonld(
    credential: &Credential,
    issuer_key: &[u8],
    subject_key: &[u8],
) -> Result<Value, CredentialError> {
    let issuer_did = identity_to_did_key(&credential.issuer, issuer_key)?;
    let subject_did = identity_to_did_key(&credential.subject, subject_key)?;

    let claims: Vec<Value> = credential
        .claim_digests
        .iter()
        .map(|d| {
            json!({ "digest": base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(d) })
        })
        .collect();

    Ok(build_vc_json(credential, &issuer_did, &subject_did, claims))
}

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
    let issuer_did = identity_to_did_key(&credential.issuer, issuer_key)?;
    let subject_did = identity_to_did_key(&credential.subject, subject_key)?;

    // Disclosed claims get typeId + value + digest; undisclosed get digest only.
    let claims: Vec<Value> = credential
        .claim_digests
        .iter()
        .map(|digest| {
            let digest_b64 = base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(digest);
            if let Some(sc) = presentation
                .disclosed_claims
                .iter()
                .find(|sc| &sc.digest() == digest)
            {
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

    let vc = build_vc_json(credential, &issuer_did, &subject_did, claims);

    Ok(json!({
        "@context": ["https://www.w3.org/ns/credentials/v2"],
        "type": ["VerifiablePresentation"],
        "holder": subject_did,
        "verifiableCredential": [vc]
    }))
}

/// Build the inner W3C VC JSON object (shared by credential and presentation export).
fn build_vc_json(
    credential: &Credential,
    issuer_did: &str,
    subject_did: &str,
    claims: Vec<Value>,
) -> Value {
    // Harmony-specific cryptosuite names: the proofValue covers the
    // postcard-serialized SignablePayload, NOT the RDFC-2022 canonicalized
    // JSON-LD document. Standard eddsa-2022/mldsa65-2025 verifiers would
    // fail because they expect RDFC-2022 canonicalization. Using distinct
    // names makes this transparent — a Harmony verifier knows to check
    // the postcard payload, while external tools see an unrecognized suite
    // and can fall back to treating the credential as unverified rather
    // than silently failing verification.
    let cryptosuite = match credential.issuer.suite {
        CryptoSuite::Ed25519 => "harmony-eddsa-2022",
        CryptoSuite::MlDsa65 | CryptoSuite::MlDsa65Rotatable => "harmony-mldsa65-2025",
    };

    // proofValue uses multibase base58btc encoding (prefix 'z') per
    // W3C Data Integrity EdDSA Cryptosuites 1.0 spec.
    let proof_value = alloc::format!("z{}", bs58::encode(&credential.signature).into_string());

    let mut vc = json!({
        "@context": [
            "https://www.w3.org/ns/credentials/v2",
            "https://w3id.org/security/data-integrity/v2",
            HARMONY_CONTEXT
        ],
        "type": ["VerifiableCredential"],
        "id": alloc::format!("urn:harmony:{}", hex_encode(&credential.content_hash())),
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
            "created": epoch_to_iso8601(credential.issued_at),
            "verificationMethod": alloc::format!("{}#key-1", issuer_did),
            "proofPurpose": "assertionMethod",
            "proofValue": proof_value,
            "nonce": hex_encode(&credential.nonce)
        }
    });

    if let Some(idx) = credential.status_list_index {
        let status_list_url = alloc::format!(
            "harmony:status-list:{}",
            hex_encode(&credential.issuer.hash)
        );
        vc["credentialStatus"] = json!({
            "id": alloc::format!("{}#{}", status_list_url, idx),
            "type": "BitstringStatusListEntry",
            "statusPurpose": "revocation",
            "statusListIndex": idx.to_string(),
            "statusListCredential": status_list_url
        });
    }

    if let Some(ref proof_hash) = credential.proof {
        vc["delegationProof"] = json!(
            base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(proof_hash)
        );
    }

    vc
}

fn epoch_to_iso8601(epoch: u64) -> String {
    let secs = epoch;
    let days = secs / 86400;
    let time_secs = secs % 86400;
    let hours = time_secs / 3600;
    let minutes = (time_secs % 3600) / 60;
    let seconds = time_secs % 60;
    let (year, month, day) = days_to_ymd(days);
    alloc::format!(
        "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}Z",
        year, month, day, hours, minutes, seconds
    )
}

fn days_to_ymd(days: u64) -> (u64, u64, u64) {
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

fn hex_encode(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        use core::fmt::Write;
        let _ = write!(s, "{:02x}", b);
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::credential::CredentialBuilder;

    fn test_issuer_ref() -> IdentityRef {
        IdentityRef::new([0xAA; 16], CryptoSuite::Ed25519)
    }

    fn test_subject_ref() -> IdentityRef {
        IdentityRef::new([0xBB; 16], CryptoSuite::Ed25519)
    }

    fn build_test_credential() -> crate::Credential {
        let mut builder = CredentialBuilder::new(
            test_issuer_ref(),
            test_subject_ref(),
            1000,
            5000,
            [0x01; 16],
        );
        builder.add_claim(1, alloc::vec![0xAA, 0xBB], [0x11; 16]);
        builder.add_claim(2, alloc::vec![0xCC], [0x22; 16]);
        builder.status_list_index(42);
        let _payload = builder.signable_payload();
        let (cred, _) = builder.build(alloc::vec![0xDE, 0xAD, 0xBE, 0xEF]);
        cred
    }

    #[test]
    fn did_key_ed25519_format() {
        let id = IdentityRef::new([0xAA; 16], CryptoSuite::Ed25519);
        let key = [0x42u8; 32];
        let did = identity_to_did_key(&id, &key).unwrap();
        assert!(did.starts_with("did:key:z"));
        let encoded = &did["did:key:z".len()..];
        let decoded = bs58::decode(encoded).into_vec().unwrap();
        assert_eq!(decoded[0], 0xed);
        assert_eq!(decoded[1], 0x01);
        assert_eq!(&decoded[2..], &key);
    }

    #[test]
    fn did_key_ml_dsa65_format() {
        let id = IdentityRef::new([0xBB; 16], CryptoSuite::MlDsa65);
        let key = [0x55u8; 1952];
        let did = identity_to_did_key(&id, &key).unwrap();
        assert!(did.starts_with("did:key:z"));
        let encoded = &did["did:key:z".len()..];
        let decoded = bs58::decode(encoded).into_vec().unwrap();
        assert_eq!(decoded[0], 0x91);
        assert_eq!(decoded[1], 0x24);
        assert_eq!(&decoded[2..], &key);
    }

    #[test]
    fn did_key_deterministic() {
        let id = IdentityRef::new([0xAA; 16], CryptoSuite::Ed25519);
        let key = [0x42u8; 32];
        assert_eq!(
            identity_to_did_key(&id, &key).unwrap(),
            identity_to_did_key(&id, &key).unwrap()
        );
    }

    #[test]
    fn did_key_wrong_length_rejected() {
        let id = IdentityRef::new([0xAA; 16], CryptoSuite::Ed25519);
        assert!(identity_to_did_key(&id, &[0x42; 16]).is_err()); // too short
        assert!(identity_to_did_key(&id, &[0x42; 64]).is_err()); // too long

        let id_pq = IdentityRef::new([0xBB; 16], CryptoSuite::MlDsa65);
        assert!(identity_to_did_key(&id_pq, &[0x55; 32]).is_err()); // Ed25519 size for PQ suite
    }

    #[test]
    fn credential_export_has_required_fields() {
        let cred = build_test_credential();
        let json = credential_to_jsonld(&cred, &[0x42; 32], &[0x43; 32]).unwrap();
        assert!(json["@context"].is_array());
        assert!(json["type"].is_array());
        assert!(json["issuer"].is_string());
        assert!(json["credentialSubject"].is_object());
        assert!(json["proof"].is_object());
        let contexts: Vec<&str> = json["@context"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap())
            .collect();
        assert!(contexts.contains(&"https://www.w3.org/ns/credentials/v2"));
        assert!(contexts.contains(&"https://w3id.org/security/data-integrity/v2"));
        assert!(contexts.contains(&HARMONY_CONTEXT));
    }

    #[test]
    fn credential_export_issuer_is_did_key() {
        let cred = build_test_credential();
        let json = credential_to_jsonld(&cred, &[0x42; 32], &[0x43; 32]).unwrap();
        assert!(json["issuer"].as_str().unwrap().starts_with("did:key:z"));
        assert!(json["credentialSubject"]["id"]
            .as_str()
            .unwrap()
            .starts_with("did:key:z"));
    }

    #[test]
    fn credential_export_timestamps_iso8601() {
        let cred = build_test_credential();
        let json = credential_to_jsonld(&cred, &[0x42; 32], &[0x43; 32]).unwrap();
        assert!(json["validFrom"].as_str().unwrap().ends_with("Z"));
        assert!(json["validUntil"].as_str().unwrap().ends_with("Z"));
    }

    #[test]
    fn credential_export_status_list_present() {
        let cred = build_test_credential();
        let json = credential_to_jsonld(&cred, &[0x42; 32], &[0x43; 32]).unwrap();
        assert!(json["credentialStatus"].is_object());
        assert_eq!(
            json["credentialStatus"]["statusListIndex"]
                .as_str()
                .unwrap(),
            "42"
        );
        assert_eq!(
            json["credentialStatus"]["statusPurpose"]
                .as_str()
                .unwrap(),
            "revocation"
        );
        // W3C required: id must be a URL, distinct from statusListCredential
        let status_id = json["credentialStatus"]["id"].as_str().unwrap();
        assert!(status_id.contains("#42"));
    }

    #[test]
    fn credential_export_status_list_absent_when_none() {
        let builder = CredentialBuilder::new(
            test_issuer_ref(),
            test_subject_ref(),
            1000,
            5000,
            [0x01; 16],
        );
        let _payload = builder.signable_payload();
        let (cred, _) = builder.build(alloc::vec![0xDE, 0xAD]);
        let json = credential_to_jsonld(&cred, &[0x42; 32], &[0x43; 32]).unwrap();
        assert!(json.get("credentialStatus").is_none() || json["credentialStatus"].is_null());
    }

    #[test]
    fn credential_export_proof_required_fields() {
        let cred = build_test_credential();
        let json = credential_to_jsonld(&cred, &[0x42; 32], &[0x43; 32]).unwrap();
        let proof = &json["proof"];
        assert_eq!(proof["type"].as_str().unwrap(), "DataIntegrityProof");
        assert_eq!(proof["cryptosuite"].as_str().unwrap(), "harmony-eddsa-2022");
        assert_eq!(proof["proofPurpose"].as_str().unwrap(), "assertionMethod");
        // proofValue must be multibase base58btc (prefix 'z')
        let pv = proof["proofValue"].as_str().unwrap();
        assert!(pv.starts_with('z'), "proofValue must use multibase base58btc prefix 'z'");
        assert!(proof["nonce"].is_string());
        // W3C required: created and verificationMethod
        assert!(proof["created"].as_str().unwrap().ends_with("Z"));
        let vm = proof["verificationMethod"].as_str().unwrap();
        assert!(vm.starts_with("did:key:z"));
        assert!(vm.ends_with("#key-1"));
    }

    #[test]
    fn credential_export_has_id() {
        let cred = build_test_credential();
        let json = credential_to_jsonld(&cred, &[0x42; 32], &[0x43; 32]).unwrap();
        let id = json["id"].as_str().unwrap();
        assert!(id.starts_with("urn:harmony:"));
    }

    #[test]
    fn credential_export_delegation_proof() {
        let mut builder = CredentialBuilder::new(
            test_issuer_ref(),
            test_subject_ref(),
            1000,
            5000,
            [0x01; 16],
        );
        builder.proof([0xFF; 32]);
        let _payload = builder.signable_payload();
        let (cred, _) = builder.build(alloc::vec![0xDE, 0xAD]);
        let json = credential_to_jsonld(&cred, &[0x42; 32], &[0x43; 32]).unwrap();
        assert!(json["delegationProof"].is_string());
        assert!(json["proof"].is_object());
    }

    #[test]
    fn credential_export_claims_as_digests() {
        let cred = build_test_credential();
        let json = credential_to_jsonld(&cred, &[0x42; 32], &[0x43; 32]).unwrap();
        let claims = json["credentialSubject"]["claims"].as_array().unwrap();
        assert_eq!(claims.len(), 2);
        for claim in claims {
            assert!(claim["digest"].is_string());
        }
    }

    #[test]
    fn presentation_export_disclosed_claims() {
        use crate::disclosure::Presentation;
        let mut builder = CredentialBuilder::new(
            test_issuer_ref(),
            test_subject_ref(),
            1000,
            5000,
            [0x01; 16],
        );
        builder.add_claim(1, alloc::vec![0xAA, 0xBB], [0x11; 16]);
        builder.add_claim(2, alloc::vec![0xCC], [0x22; 16]);
        let _payload = builder.signable_payload();
        let (cred, salted_claims) = builder.build(alloc::vec![0xDE, 0xAD]);

        let presentation = Presentation {
            credential: cred,
            disclosed_claims: vec![salted_claims[0].clone()],
        };

        let json =
            presentation_to_jsonld(&presentation, &[0x42; 32], &[0x43; 32]).unwrap();

        // VP envelope
        assert!(json["@context"].is_array());
        let types: Vec<&str> = json["type"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap())
            .collect();
        assert!(types.contains(&"VerifiablePresentation"));
        // holder binds presentation to the subject
        assert!(json["holder"].as_str().unwrap().starts_with("did:key:z"));

        // Inner VC
        let vc = &json["verifiableCredential"][0];
        let claims = vc["credentialSubject"]["claims"].as_array().unwrap();
        assert_eq!(claims.len(), 2);

        // One disclosed (has typeId), one undisclosed (digest only)
        let disclosed = claims.iter().find(|c| c.get("typeId").is_some()).unwrap();
        assert_eq!(disclosed["typeId"].as_u64().unwrap(), 1);
        assert!(disclosed["value"].is_string());
        assert!(disclosed["digest"].is_string());

        let undisclosed_count = claims
            .iter()
            .filter(|c| c.get("typeId").is_none())
            .count();
        assert_eq!(undisclosed_count, 1);
    }

    #[test]
    fn credential_export_ml_dsa65() {
        let issuer = IdentityRef::new([0xAA; 16], CryptoSuite::MlDsa65);
        let subject = IdentityRef::new([0xBB; 16], CryptoSuite::MlDsa65);

        let builder = CredentialBuilder::new(issuer, subject, 1000, 5000, [0x01; 16]);
        let _payload = builder.signable_payload();
        let (cred, _) = builder.build(alloc::vec![0xDE; 3309]);

        let json = credential_to_jsonld(&cred, &[0x55; 1952], &[0x66; 1952]).unwrap();

        assert_eq!(
            json["proof"]["cryptosuite"].as_str().unwrap(),
            "harmony-mldsa65-2025"
        );
        assert!(json["issuer"].as_str().unwrap().starts_with("did:key:z"));
    }

    #[test]
    fn credential_export_ml_dsa65_rotatable_lossy() {
        // MlDsa65Rotatable shares multicodec 0x1211 with MlDsa65.
        // Export intentionally loses the "rotatable" distinction —
        // the did:key and cryptosuite are identical to MlDsa65.
        // This matches crypto_suite.rs::multicodec_round_trip_lossy_for_rotatable.
        let issuer = IdentityRef::new([0xAA; 16], CryptoSuite::MlDsa65Rotatable);
        let subject = IdentityRef::new([0xBB; 16], CryptoSuite::MlDsa65Rotatable);

        let builder = CredentialBuilder::new(issuer, subject, 1000, 5000, [0x01; 16]);
        let _payload = builder.signable_payload();
        let (cred, _) = builder.build(alloc::vec![0xDE; 3309]);

        let json = credential_to_jsonld(&cred, &[0x55; 1952], &[0x66; 1952]).unwrap();

        // Same cryptosuite as static MlDsa65 — rotatable semantics are lost
        assert_eq!(
            json["proof"]["cryptosuite"].as_str().unwrap(),
            "harmony-mldsa65-2025"
        );
        // did:key uses same multicodec (0x1211) — indistinguishable
        let did = json["issuer"].as_str().unwrap();
        assert!(did.starts_with("did:key:z"));
    }
}
