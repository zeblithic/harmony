use harmony_identity::CryptoSuite;

use crate::error::SdJwtError;
use crate::types::SdJwt;

/// Map JWS algorithm name to Harmony CryptoSuite.
fn alg_to_suite(alg: &str) -> Result<CryptoSuite, SdJwtError> {
    match alg {
        "EdDSA" => Ok(CryptoSuite::Ed25519),
        // IETF draft-ietf-jose-post-quantum-algs uses "ML-DSA-65" (with hyphens)
        "ML-DSA-65" => Ok(CryptoSuite::MlDsa65),
        other => Err(SdJwtError::UnsupportedAlgorithm(
            alloc::string::String::from(other),
        )),
    }
}

/// Verify the JWS signature of a parsed SD-JWT.
///
/// Uses `sd_jwt.signing_input` as the message for lossless verification.
/// The caller resolves the correct public key via DID resolution.
/// `ML-DSA-65` maps to `CryptoSuite::MlDsa65` (not Rotatable).
///
/// # Note
///
/// This function only verifies the cryptographic signature over the JWS
/// signing input. It does **not** validate time-based claims (`exp`, `nbf`,
/// `iat`). Callers MUST separately check that the token is not expired and
/// is past its not-before time before accepting the disclosed claims.
pub fn verify(
    sd_jwt: &SdJwt,
    suite: CryptoSuite,
    public_key: &[u8],
) -> Result<(), SdJwtError> {
    harmony_identity::verify_signature(
        suite,
        public_key,
        sd_jwt.signing_input.as_bytes(),
        &sd_jwt.signature,
    )
    .map_err(SdJwtError::SignatureInvalid)
}

/// Verify using the algorithm and type from the JWS header.
///
/// Checks that `typ` is `"sd+jwt"` per RFC 9901 §3.3 before verifying
/// the signature. This prevents cross-format token confusion where a
/// plain JWT with a valid signature passes as an SD-JWT.
///
/// # Note
///
/// This function only verifies the cryptographic signature and the `typ`
/// header. It does **not** validate time-based claims (`exp`, `nbf`,
/// `iat`). Callers MUST separately check that the token is not expired
/// and is past its not-before time before accepting the disclosed claims.
pub fn verify_from_header(
    sd_jwt: &SdJwt,
    public_key: &[u8],
) -> Result<(), SdJwtError> {
    // RFC 9901 §3.3: typ MUST be "sd+jwt" for issuer-signed SD-JWTs.
    // RFC 7515 §4.1.9: typ comparisons SHOULD be case-insensitive.
    // Accept both short form ("sd+jwt") and full media type ("application/sd+jwt").
    match sd_jwt.header.typ.as_deref() {
        Some(t) if t.eq_ignore_ascii_case("sd+jwt") || t.eq_ignore_ascii_case("application/sd+jwt") => {}
        _ => return Err(SdJwtError::WrongTokenType),
    }
    let suite = alg_to_suite(&sd_jwt.header.alg)?;
    verify(sd_jwt, suite, public_key)
}

#[cfg(all(test, feature = "std"))]
mod tests {
    use super::*;
    use base64::Engine;
    use rand::rngs::OsRng;

    const B64: base64::engine::GeneralPurpose =
        base64::engine::general_purpose::URL_SAFE_NO_PAD;

    fn b64(json: &str) -> alloc::string::String {
        B64.encode(json.as_bytes())
    }

    fn make_signed_sdjwt(
        private: &harmony_identity::PrivateIdentity,
        payload_json: &str,
    ) -> alloc::string::String {
        let header_b64 = b64(r#"{"alg":"EdDSA","typ":"sd+jwt"}"#);
        let payload_b64 = b64(payload_json);
        let signing_input = alloc::format!("{}.{}", header_b64, payload_b64);
        let signature = private.sign(signing_input.as_bytes());
        let sig_b64 = B64.encode(&signature);
        alloc::format!("{}.{}", signing_input, sig_b64)
    }

    #[test]
    fn verify_valid_ed25519() {
        let private = harmony_identity::PrivateIdentity::generate(&mut OsRng);
        let identity = private.public_identity();
        let compact = make_signed_sdjwt(&private, r#"{"iss":"alice"}"#);

        let sd_jwt = crate::parse::parse(&compact).unwrap();
        assert!(verify(&sd_jwt, CryptoSuite::Ed25519, &identity.verifying_key.to_bytes()).is_ok());
    }

    #[test]
    fn verify_rejects_wrong_key() {
        let private = harmony_identity::PrivateIdentity::generate(&mut OsRng);
        let other = harmony_identity::PrivateIdentity::generate(&mut OsRng);
        let compact = make_signed_sdjwt(&private, r#"{"iss":"alice"}"#);

        let sd_jwt = crate::parse::parse(&compact).unwrap();
        assert!(verify(&sd_jwt, CryptoSuite::Ed25519, &other.public_identity().verifying_key.to_bytes()).is_err());
    }

    #[test]
    fn verify_from_header_works() {
        let private = harmony_identity::PrivateIdentity::generate(&mut OsRng);
        let identity = private.public_identity();
        let compact = make_signed_sdjwt(&private, r#"{"iss":"alice"}"#);

        let sd_jwt = crate::parse::parse(&compact).unwrap();
        assert!(verify_from_header(&sd_jwt, &identity.verifying_key.to_bytes()).is_ok());
    }

    #[test]
    fn unsupported_algorithm() {
        assert!(matches!(alg_to_suite("RS256"), Err(SdJwtError::UnsupportedAlgorithm(_))));
    }

    #[test]
    fn verify_from_header_rejects_missing_typ() {
        let private = harmony_identity::PrivateIdentity::generate(&mut OsRng);
        let identity = private.public_identity();
        // Build a JWS with no typ field — valid signature but wrong token type
        let header_b64 = B64.encode(r#"{"alg":"EdDSA"}"#.as_bytes());
        let payload_b64 = B64.encode(r#"{"iss":"alice"}"#.as_bytes());
        let signing_input = alloc::format!("{}.{}", header_b64, payload_b64);
        let sig_b64 = B64.encode(&private.sign(signing_input.as_bytes()));
        let compact = alloc::format!("{}.{}", signing_input, sig_b64);

        let sd_jwt = crate::parse::parse(&compact).unwrap();
        assert!(matches!(
            verify_from_header(&sd_jwt, &identity.verifying_key.to_bytes()),
            Err(SdJwtError::WrongTokenType)
        ));
    }
}
