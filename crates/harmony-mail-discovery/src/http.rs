//! HTTPS fetch + parse for `.well-known/harmony-users` and
//! `.well-known/harmony-revocations` (spec §4.5, §5.4).

use std::time::Duration;

use async_trait::async_trait;
use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use base64::Engine;
use reqwest::redirect::Policy;

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

pub struct ReqwestHttpClient {
    inner: reqwest::Client,
    body_cap_bytes: usize,
}

impl ReqwestHttpClient {
    pub fn new(
        connect_timeout: Duration,
        total_timeout: Duration,
        body_cap_bytes: usize,
    ) -> Result<Self, HttpError> {
        let inner = reqwest::Client::builder()
            .redirect(Policy::none())
            .connect_timeout(connect_timeout)
            .timeout(total_timeout)
            .https_only(true)
            .build()
            .map_err(|e| HttpError::Other(e.to_string()))?;
        Ok(Self {
            inner,
            body_cap_bytes,
        })
    }
}

#[async_trait]
impl HttpClient for ReqwestHttpClient {
    async fn get(&self, url: &str) -> Result<HttpResponse, HttpError> {
        let resp = self.inner.get(url).send().await.map_err(|e| {
            if e.is_timeout() {
                HttpError::Timeout
            } else if e.is_connect() {
                HttpError::Connect(e.to_string())
            } else {
                HttpError::Other(e.to_string())
            }
        })?;
        let status = resp.status();
        if status.is_redirection() {
            return Err(HttpError::RedirectRefused);
        }
        let status = status.as_u16();
        // Bounded body read — prevents oversize responses from blowing
        // memory. reqwest doesn't expose chunked reads cleanly, so we
        // check Content-Length as a first line of defense, then cap on
        // the collected bytes as a second.
        if let Some(len) = resp.content_length() {
            if len > self.body_cap_bytes as u64 {
                return Err(HttpError::BodyTooLarge);
            }
        }
        let bytes = resp
            .bytes()
            .await
            .map_err(|e| HttpError::Other(e.to_string()))?;
        if bytes.len() > self.body_cap_bytes {
            return Err(HttpError::BodyTooLarge);
        }
        Ok(HttpResponse {
            status,
            body: bytes.to_vec(),
        })
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

    #[test]
    fn reqwest_client_builds_with_sensible_params() {
        ReqwestHttpClient::new(Duration::from_secs(3), Duration::from_secs(5), 1024 * 1024)
            .expect("builder should accept normal timeouts + 1 MiB cap");
    }
}
