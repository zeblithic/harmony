//! DKIM/SPF/DMARC integration via the `mail-auth` crate.
//!
//! Provides:
//! - [`AuthResults`]: bridges mail-auth output to our spam scoring engine
//! - [`DkimSigner`]: DKIM signing for outbound RFC 5322 messages
//! - Helper functions for creating the `MessageAuthenticator` and running
//!   verification pipelines
//!
//! DNS-dependent verification (SPF, DKIM check, DMARC) is async and requires
//! a live `MessageAuthenticator`. The server module creates the authenticator
//! once and passes it to these functions per-connection.

use mail_auth::common::headers::HeaderWriter;

use crate::spam::{DkimResult, DmarcResult, SpfResult};

/// Results from authenticating an inbound message.
///
/// Fed into [`SpamSignals`](crate::spam::SpamSignals) for scoring.
#[derive(Debug, Clone)]
pub struct AuthResults {
    pub spf: SpfResult,
    pub dkim: DkimResult,
    pub dmarc: DmarcResult,
}

impl Default for AuthResults {
    fn default() -> Self {
        Self {
            spf: SpfResult::None,
            dkim: DkimResult::Missing,
            dmarc: DmarcResult::None,
        }
    }
}

/// Create a `MessageAuthenticator` from system DNS configuration.
///
/// The authenticator is reusable across connections. Create it once at
/// server startup and pass it to verification functions.
pub fn create_authenticator() -> Result<mail_auth::MessageAuthenticator, AuthError> {
    mail_auth::MessageAuthenticator::new_system_conf()
        .map_err(|e| AuthError::Resolver(format!("{e}")))
}

/// Verify DKIM signatures on a raw RFC 5322 message.
pub async fn verify_dkim(
    authenticator: &mail_auth::MessageAuthenticator,
    message: &[u8],
) -> DkimResult {
    let authenticated = match mail_auth::AuthenticatedMessage::parse(message) {
        Some(msg) => msg,
        // Unparseable message is an auth failure, not "missing DKIM"
        None => return DkimResult::Fail,
    };

    let results = authenticator.verify_dkim(&authenticated).await;
    for output in results.iter() {
        if output.result() == &mail_auth::DkimResult::Pass {
            return DkimResult::Pass;
        }
    }

    if results.is_empty() {
        DkimResult::Missing
    } else {
        DkimResult::Fail
    }
}

/// Map a `mail_auth::SpfOutput` to our `SpfResult`.
pub fn map_spf_result(output: &mail_auth::SpfOutput) -> SpfResult {
    match output.result() {
        mail_auth::SpfResult::Pass => SpfResult::Pass,
        mail_auth::SpfResult::Fail => SpfResult::Fail,
        mail_auth::SpfResult::SoftFail => SpfResult::SoftFail,
        _ => SpfResult::None,
    }
}

/// Map a `mail_auth::DmarcOutput` to our `DmarcResult`.
pub fn map_dmarc_result(output: &mail_auth::DmarcOutput) -> DmarcResult {
    // Check if either DKIM or SPF passed DMARC alignment
    let dkim_aligned = output.dkim_result() == &mail_auth::DmarcResult::Pass;
    let spf_aligned = output.spf_result() == &mail_auth::DmarcResult::Pass;

    if dkim_aligned || spf_aligned {
        DmarcResult::Pass
    } else {
        match output.policy() {
            // No DMARC record → can't fail what doesn't exist
            mail_auth::dmarc::Policy::Unspecified => DmarcResult::None,
            // Domain has a DMARC record but alignment failed
            _ => DmarcResult::Fail,
        }
    }
}

/// DKIM signing configuration for outbound messages.
pub struct DkimSigner {
    pub selector: String,
    pub domain: String,
    key_der: Vec<u8>,
}

impl DkimSigner {
    /// Create a new DKIM signer from a DER-encoded PKCS#8 Ed25519 private key file.
    pub fn from_key_file(
        selector: &str,
        domain: &str,
        key_path: &std::path::Path,
    ) -> Result<Self, AuthError> {
        let key_der = std::fs::read(key_path)
            .map_err(|e| AuthError::SigningKey(format!("reading {}: {e}", key_path.display())))?;
        // Validate the key parses before storing — fail fast on bad keys
        let _ = mail_auth::common::crypto::Ed25519Key::from_pkcs8_der(&key_der)
            .map_err(|e| AuthError::SigningKey(format!("invalid key in {}: {e}", key_path.display())))?;
        Ok(Self {
            selector: selector.to_string(),
            domain: domain.to_string(),
            key_der,
        })
    }

    /// Sign an RFC 5322 message, returning the message with DKIM-Signature header prepended.
    pub fn sign(&self, message: &[u8]) -> Result<Vec<u8>, AuthError> {
        let pk = mail_auth::common::crypto::Ed25519Key::from_pkcs8_der(&self.key_der)
            .map_err(|e| AuthError::SigningKey(format!("{e}")))?;

        let signature = mail_auth::dkim::DkimSigner::from_key(pk)
            .domain(&self.domain)
            .selector(&self.selector)
            .headers(["From", "To", "Subject", "Date", "Message-ID", "MIME-Version", "Content-Type"])
            .sign(message)
            .map_err(|e| AuthError::Signing(format!("{e}")))?;

        let header = signature.to_header();
        let mut signed = Vec::with_capacity(header.len() + message.len());
        signed.extend_from_slice(header.as_bytes());
        signed.extend_from_slice(message);
        Ok(signed)
    }
}

/// Authentication errors.
#[derive(Debug, thiserror::Error)]
pub enum AuthError {
    #[error("failed to create DNS resolver: {0}")]
    Resolver(String),

    #[error("failed to load signing key: {0}")]
    SigningKey(String),

    #[error("DKIM signing failed: {0}")]
    Signing(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn auth_results_default() {
        let results = AuthResults::default();
        assert_eq!(results.spf, SpfResult::None);
        assert_eq!(results.dkim, DkimResult::Missing);
        assert_eq!(results.dmarc, DmarcResult::None);
    }

    #[test]
    fn auth_results_feeds_spam_scorer_reject() {
        use crate::spam::{self, SpamAction, SpamSignals};

        let signals = SpamSignals {
            dnsbl_listed: false,
            fcrdns_pass: true,
            spf_result: SpfResult::Fail,
            dkim_result: DkimResult::Fail,
            dmarc_result: DmarcResult::Fail,
            has_executable_attachment: false,
            url_count: 0,
            empty_subject: false,
            known_harmony_sender: false,
            gateway_trust: None,
            first_contact: false,
        };

        let verdict = spam::score(&signals, 5);
        // SPF fail (+3) + DKIM fail (+3) + DMARC fail (+3) = 9 >= 5
        assert!(verdict.score >= 5, "score should be >= 5, got {}", verdict.score);
        assert_eq!(verdict.action, SpamAction::Reject);
    }

    #[test]
    fn auth_results_feeds_spam_scorer_deliver() {
        use crate::spam::{self, SpamAction, SpamSignals};

        let signals = SpamSignals {
            dnsbl_listed: false,
            fcrdns_pass: true,
            spf_result: SpfResult::Pass,
            dkim_result: DkimResult::Pass,
            dmarc_result: DmarcResult::Pass,
            has_executable_attachment: false,
            url_count: 0,
            empty_subject: false,
            known_harmony_sender: true,
            gateway_trust: Some(0.9),
            first_contact: false,
        };

        let verdict = spam::score(&signals, 5);
        assert!(verdict.score <= 0, "score should be <= 0, got {}", verdict.score);
        assert_eq!(verdict.action, SpamAction::Deliver);
    }

    #[tokio::test]
    async fn verify_dkim_missing_on_unsigned() {
        let authenticator = match create_authenticator() {
            Ok(a) => a,
            Err(e) => {
                eprintln!("SKIP verify_dkim_missing_on_unsigned: DNS resolver unavailable ({e})");
                return;
            }
        };
        let unsigned = b"From: test@example.com\r\nTo: recv@test.com\r\n\r\nHello\r\n";
        let result = verify_dkim(&authenticator, unsigned).await;
        assert_eq!(result, DkimResult::Missing);
    }
}
