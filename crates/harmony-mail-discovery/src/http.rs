//! HTTPS fetch + parse for `.well-known/harmony-users` and
//! `.well-known/harmony-revocations` (spec §4.5, §5.4).

use async_trait::async_trait;
use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use base64::Engine;

use crate::claim::{RevocationList, SignedClaim};

/// Thin abstraction so we can inject fakes without spinning up a real
/// HTTP server. Production impl is `ReqwestHttpClient`.
#[async_trait]
pub trait HttpClient: Send + Sync + 'static {
    /// Fetch `url` and return the response.
    ///
    /// Implementors MUST enforce a body-size cap and surface oversize
    /// responses as [`HttpError::BodyTooLarge`] — the `fetch_*` helpers
    /// in this module pass `body` straight to the CBOR parser and do
    /// not guard against unbounded input. Implementors MUST also refuse
    /// redirects (return [`HttpError::RedirectRefused`]) to prevent
    /// downgrade attacks on the `.well-known` endpoints.
    async fn get(&self, url: &str) -> Result<HttpResponse, HttpError>;
}

#[derive(Debug, Clone)]
pub struct HttpResponse {
    pub status: u16,
    pub body: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum HttpError {
    #[error("connect: {0}")]
    Connect(String),
    #[error("timeout")]
    Timeout,
    #[error("tls: {0}")]
    Tls(String),
    #[error("body too large (> cap)")]
    BodyTooLarge,
    #[error("redirect refused")]
    RedirectRefused,
    #[error("other: {0}")]
    Other(String),
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum HttpFetchError {
    #[error("transport: {0}")]
    Transport(#[from] HttpError),
    #[error("server returned {0}")]
    Server(u16),
    #[error("malformed CBOR: {0}")]
    MalformedCbor(String),
}

#[derive(Debug)]
#[allow(clippy::large_enum_variant)]
pub enum ClaimFetchResult {
    Found(SignedClaim),
    NotFound,
}

#[derive(Debug)]
pub enum RevocationFetchResult {
    Found(RevocationList),
    Empty, // 404 -> authoritative empty
}

pub fn claim_url(domain: &str, hashed_local_part: &[u8; 32]) -> String {
    let h = URL_SAFE_NO_PAD.encode(hashed_local_part);
    format!(
        "https://{}/.well-known/harmony-users?h={}",
        domain.to_ascii_lowercase(),
        h
    )
}

pub fn revocation_url(domain: &str) -> String {
    format!(
        "https://{}/.well-known/harmony-revocations",
        domain.to_ascii_lowercase()
    )
}

pub async fn fetch_claim(
    http: &dyn HttpClient,
    domain: &str,
    hashed_local_part: &[u8; 32],
) -> Result<ClaimFetchResult, HttpFetchError> {
    let resp = http.get(&claim_url(domain, hashed_local_part)).await?;
    match resp.status {
        200 => {
            let claim: SignedClaim = ciborium::de::from_reader(&resp.body[..])
                .map_err(|e| HttpFetchError::MalformedCbor(e.to_string()))?;
            Ok(ClaimFetchResult::Found(claim))
        }
        404 => Ok(ClaimFetchResult::NotFound),
        code => Err(HttpFetchError::Server(code)),
    }
}

pub async fn fetch_revocation_list(
    http: &dyn HttpClient,
    domain: &str,
) -> Result<RevocationFetchResult, HttpFetchError> {
    let resp = http.get(&revocation_url(domain)).await?;
    match resp.status {
        200 => {
            let list: RevocationList = ciborium::de::from_reader(&resp.body[..])
                .map_err(|e| HttpFetchError::MalformedCbor(e.to_string()))?;
            Ok(RevocationFetchResult::Found(list))
        }
        404 => Ok(RevocationFetchResult::Empty),
        code => Err(HttpFetchError::Server(code)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn claim_url_uses_base64url_no_pad() {
        let h = [0xffu8; 32];
        let url = claim_url("q8.fyi", &h);
        assert!(url.starts_with("https://q8.fyi/.well-known/harmony-users?h="));
        let hash_value = url.split("?h=").nth(1).unwrap_or("");
        assert!(
            !hash_value.contains('='),
            "must use base64url no-pad: {url}"
        );
    }

    #[test]
    fn revocation_url_is_well_known_path() {
        assert_eq!(
            revocation_url("Q8.fyi"),
            "https://q8.fyi/.well-known/harmony-revocations"
        );
    }
}
