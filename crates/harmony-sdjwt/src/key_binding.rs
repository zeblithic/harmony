//! Key Binding JWT (KB-JWT) verification per RFC 9901 section 11.6.
//!
//! A KB-JWT proves that the presenter holds the private key bound to the
//! SD-JWT credential. The verifier checks the KB-JWT signature, nonce,
//! audience, freshness (iat), and sd_hash binding.

use harmony_identity::CryptoSuite;

use crate::error::SdJwtError;
use crate::types::SdJwt;

/// Verify a Key Binding JWT attached to an SD-JWT presentation.
///
/// # Arguments
///
/// * `sd_jwt` - The parsed SD-JWT (must have `key_binding_jwt` set)
/// * `holder_key` - The holder's public key bytes
/// * `holder_suite` - The cryptographic suite for the holder's key
/// * `expected_nonce` - The nonce the verifier expects in the KB-JWT
/// * `expected_aud` - The audience the verifier expects in the KB-JWT
/// * `now` - Current UNIX timestamp (seconds)
pub fn verify_key_binding(
    _sd_jwt: &SdJwt,
    _holder_key: &[u8],
    _holder_suite: CryptoSuite,
    _expected_nonce: &str,
    _expected_aud: &str,
    _now: u64,
) -> Result<(), SdJwtError> {
    todo!("KB-JWT verification not yet implemented")
}
