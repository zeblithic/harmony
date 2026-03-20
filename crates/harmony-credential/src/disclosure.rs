use alloc::vec::Vec;
use serde::{Deserialize, Serialize};

use crate::claim::SaltedClaim;
use crate::credential::Credential;
use crate::error::CredentialError;

/// A credential presentation with selectively disclosed claims.
///
/// The holder sends this to a verifier, revealing only the claims
/// they choose. The verifier checks that each disclosed claim's
/// digest appears in the signed credential.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Presentation {
    pub credential: Credential,
    pub disclosed_claims: Vec<SaltedClaim>,
}

impl Presentation {
    /// Verify that all disclosed claims match digests in the credential.
    ///
    /// Checks:
    /// 1. Each disclosed claim's digest appears in `credential.claim_digests`
    /// 2. No two disclosed claims map to the same digest (no duplicates)
    pub fn verify_disclosures(&self) -> Result<(), CredentialError> {
        let mut matched_indices = Vec::new();

        for disclosed in &self.disclosed_claims {
            let digest = disclosed.digest();
            let pos = self
                .credential
                .claim_digests
                .iter()
                .position(|d| *d == digest)
                .ok_or(CredentialError::DisclosureMismatch)?;

            if matched_indices.contains(&pos) {
                return Err(CredentialError::DuplicateDisclosure);
            }
            matched_indices.push(pos);
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::claim::{Claim, SaltedClaim};
    use crate::credential::CredentialBuilder;
    use harmony_identity::{CryptoSuite, IdentityRef};

    fn build_test_credential() -> (Credential, Vec<SaltedClaim>) {
        let issuer = IdentityRef::new([0xAA; 16], CryptoSuite::Ed25519);
        let subject = IdentityRef::new([0xBB; 16], CryptoSuite::Ed25519);
        let mut builder = CredentialBuilder::new(issuer, subject, 1000, 2000, [0x01; 16]);
        builder.add_claim(1, alloc::vec![0xAA], [0x11; 16]);
        builder.add_claim(2, alloc::vec![0xBB], [0x22; 16]);
        builder.add_claim(3, alloc::vec![0xCC], [0x33; 16]);
        let payload = builder.signable_payload();
        builder.build(payload)
    }

    #[test]
    fn verify_all_disclosures() {
        let (cred, claims) = build_test_credential();
        let presentation = Presentation {
            credential: cred,
            disclosed_claims: claims,
        };
        assert!(presentation.verify_disclosures().is_ok());
    }

    #[test]
    fn verify_subset_disclosures() {
        let (cred, claims) = build_test_credential();
        let presentation = Presentation {
            credential: cred,
            disclosed_claims: vec![claims[0].clone(), claims[2].clone()],
        };
        assert!(presentation.verify_disclosures().is_ok());
    }

    #[test]
    fn verify_empty_disclosures() {
        let (cred, _claims) = build_test_credential();
        let presentation = Presentation {
            credential: cred,
            disclosed_claims: Vec::new(),
        };
        assert!(presentation.verify_disclosures().is_ok());
    }

    #[test]
    fn rejects_tampered_claim() {
        let (cred, mut claims) = build_test_credential();
        claims[0].claim.value = alloc::vec![0xFF]; // tamper
        let presentation = Presentation {
            credential: cred,
            disclosed_claims: vec![claims[0].clone()],
        };
        assert_eq!(
            presentation.verify_disclosures().unwrap_err(),
            CredentialError::DisclosureMismatch
        );
    }

    #[test]
    fn rejects_unknown_claim() {
        let (cred, _claims) = build_test_credential();
        let unknown = SaltedClaim {
            claim: Claim {
                type_id: 99,
                value: alloc::vec![0xDD],
            },
            salt: [0x99; 16],
        };
        let presentation = Presentation {
            credential: cred,
            disclosed_claims: vec![unknown],
        };
        assert_eq!(
            presentation.verify_disclosures().unwrap_err(),
            CredentialError::DisclosureMismatch
        );
    }

    #[test]
    fn rejects_duplicate_disclosures() {
        let (cred, claims) = build_test_credential();
        let presentation = Presentation {
            credential: cred,
            disclosed_claims: vec![claims[0].clone(), claims[0].clone()],
        };
        assert_eq!(
            presentation.verify_disclosures().unwrap_err(),
            CredentialError::DuplicateDisclosure
        );
    }

    #[test]
    fn serde_round_trip() {
        let (cred, claims) = build_test_credential();
        let presentation = Presentation {
            credential: cred,
            disclosed_claims: vec![claims[1].clone()],
        };
        let bytes = postcard::to_allocvec(&presentation).unwrap();
        let decoded: Presentation = postcard::from_bytes(&bytes).unwrap();
        assert_eq!(decoded.credential.issuer, presentation.credential.issuer);
        assert_eq!(decoded.disclosed_claims.len(), 1);
        assert!(decoded.verify_disclosures().is_ok());
    }
}
