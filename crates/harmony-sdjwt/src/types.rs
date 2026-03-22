use alloc::string::String;
use alloc::vec::Vec;

/// Parsed JWS header (JOSE header).
#[derive(Debug, Clone)]
pub struct JwsHeader {
    /// Signing algorithm, e.g. "EdDSA", "ES256".
    pub alg: String,
    /// Token type, e.g. "sd+jwt".
    pub typ: Option<String>,
    /// Key identifier.
    pub kid: Option<String>,
}

/// Parsed JWT payload claims.
#[derive(Debug, Clone)]
pub struct JwtPayload {
    /// Issuer (`iss` claim).
    pub iss: Option<String>,
    /// Subject (`sub` claim).
    pub sub: Option<String>,
    /// Issued-at timestamp (`iat` claim).
    pub iat: Option<i64>,
    /// Expiration timestamp (`exp` claim).
    pub exp: Option<i64>,
    /// Not-before timestamp (`nbf` claim).
    pub nbf: Option<i64>,
    /// Selective disclosure digests (`_sd` claim).
    pub sd: Vec<String>,
    /// Hash algorithm used for disclosures (`_sd_alg` claim).
    pub sd_alg: Option<String>,
    /// All remaining claims as key-value pairs (requires `std` for JSON parsing).
    #[cfg(feature = "std")]
    pub extra: Vec<(String, serde_json::Value)>,
}

/// A decoded selective disclosure.
#[derive(Debug, Clone)]
pub struct Disclosure {
    /// Raw base64url-encoded disclosure string (for hashing).
    pub raw: String,
    /// Random salt.
    pub salt: String,
    /// Claim name (absent for array element disclosures).
    pub claim_name: Option<String>,
    /// Raw JSON bytes of the claim value (always available, no_std safe).
    pub claim_value: Vec<u8>,
    /// Parsed claim value (requires `std` for JSON parsing).
    #[cfg(feature = "std")]
    pub value: serde_json::Value,
}

/// A fully parsed SD-JWT.
#[derive(Debug, Clone)]
pub struct SdJwt {
    /// The JWS header.
    pub header: JwsHeader,
    /// The JWT payload.
    pub payload: JwtPayload,
    /// Raw signature bytes.
    pub signature: Vec<u8>,
    /// The raw `header.payload` signing input (base64url encoded, for verification).
    pub signing_input: String,
    /// Decoded selective disclosures.
    pub disclosures: Vec<Disclosure>,
}
