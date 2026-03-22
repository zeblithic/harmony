use crate::error::SdJwtError;
use crate::types::SdJwt;
use harmony_identity::CryptoSuite;

pub fn verify(_sd_jwt: &SdJwt, _suite: CryptoSuite, _public_key: &[u8]) -> Result<(), SdJwtError> {
    todo!()
}

pub fn verify_from_header(
    _sd_jwt: &SdJwt,
    _public_key: &[u8],
) -> Result<(), SdJwtError> {
    todo!()
}
