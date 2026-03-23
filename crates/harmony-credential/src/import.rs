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

/// Import a W3C Verifiable Credential from JSON-LD format.
pub fn import_jsonld_vc(
    _vc_json: &Value,
    _resolver: &impl DidResolver,
) -> Result<ImportedCredential, ImportError> {
    Err(ImportError::MalformedVc(String::from("not implemented")))
}
