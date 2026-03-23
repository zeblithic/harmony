//! SD-JWT disclosure verification and claim mapping (RFC 9901 §6.3).
//!
//! This module bridges the SD-JWT world (JSON disclosures with string salts)
//! to Harmony's binary credential layer (`harmony_credential::SaltedClaim`).

use crate::error::SdJwtError;
use crate::types::{Disclosure, SdJwt};
use harmony_credential::SaltedClaim;

/// Verify that every disclosure's hash appears in the signed `_sd` list.
///
/// Returns the verified disclosures on success.
pub fn verify_disclosures(_sd_jwt: &SdJwt) -> Result<Vec<&Disclosure>, SdJwtError> {
    todo!("implement in task 2")
}

/// Map verified disclosures to Harmony `SaltedClaim` values.
pub fn map_claims(_disclosures: &[&Disclosure]) -> Vec<SaltedClaim> {
    todo!("implement in task 3")
}
