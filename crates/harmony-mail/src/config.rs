use serde::Deserialize;
use std::path::Path;

#[derive(Debug, Deserialize)]
pub struct Config {
    pub domain: DomainConfig,
    pub gateway: GatewayConfig,
    pub tls: TlsConfig,
    pub dkim: DkimConfig,
    pub spam: SpamConfig,
    pub outbound: OutboundConfig,
    pub harmony: HarmonyConfig,
}

#[derive(Debug, Deserialize)]
pub struct DomainConfig {
    pub name: String,
    pub mx_host: String,
}

#[derive(Debug, Deserialize)]
pub struct GatewayConfig {
    pub identity_key: String,
    #[serde(default = "default_listen_smtp")]
    pub listen_smtp: String,
    #[serde(default = "default_listen_submission")]
    pub listen_submission: String,
    #[serde(default = "default_listen_submission_starttls")]
    pub listen_submission_starttls: String,
}

fn default_listen_smtp() -> String {
    "0.0.0.0:25".to_string()
}
fn default_listen_submission() -> String {
    "0.0.0.0:465".to_string()
}
fn default_listen_submission_starttls() -> String {
    "0.0.0.0:587".to_string()
}

/// TLS provisioning mode.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum TlsMode {
    /// Automatic certificate provisioning via ACME (Let's Encrypt).
    #[default]
    Acme,
    /// Manual certificate files (cert + key paths required).
    Manual,
}

#[derive(Debug, Deserialize)]
pub struct TlsConfig {
    #[serde(default)]
    pub mode: TlsMode,
    pub acme_email: Option<String>,
    #[serde(default = "default_acme_challenge")]
    pub acme_challenge: String,
    pub cert: Option<String>,
    pub key: Option<String>,
}

impl TlsConfig {
    /// Validate TLS configuration:
    /// - Manual mode: cert and key paths are required.
    /// - Acme mode: acme_email is required for certificate registration.
    pub fn validate(&self) -> Result<(), String> {
        match self.mode {
            TlsMode::Manual => {
                if self.cert.is_none() {
                    return Err("tls.cert is required when mode = \"manual\"".to_string());
                }
                if self.key.is_none() {
                    return Err("tls.key is required when mode = \"manual\"".to_string());
                }
            }
            TlsMode::Acme => {
                if self.acme_email.is_none() {
                    return Err("tls.acme_email is required when mode = \"acme\"".to_string());
                }
            }
        }
        Ok(())
    }
}
fn default_acme_challenge() -> String {
    "dns-01".to_string()
}

#[derive(Debug, Deserialize)]
pub struct DkimConfig {
    #[serde(default = "default_dkim_selector")]
    pub selector: String,
    #[serde(default = "default_dkim_algorithm")]
    pub algorithm: String,
    pub key: String,
}

fn default_dkim_selector() -> String {
    "harmony".to_string()
}
fn default_dkim_algorithm() -> String {
    "ed25519".to_string()
}

#[derive(Debug, Deserialize)]
pub struct SpamConfig {
    #[serde(default)]
    pub dnsbl: Vec<String>,
    #[serde(default = "default_reject_threshold")]
    pub reject_threshold: i32,
    /// Maximum concurrent connections allowed from a single IP address.
    ///
    /// Enforced by the I/O layer at TCP accept time, not through [`super::spam::score`].
    #[serde(default = "default_max_connections_per_ip")]
    pub max_connections_per_ip: usize,
    #[serde(default = "default_max_message_size")]
    pub max_message_size: String,
}

fn default_reject_threshold() -> i32 {
    5
}
fn default_max_connections_per_ip() -> usize {
    5
}
fn default_max_message_size() -> String {
    "25MB".to_string()
}

#[derive(Debug, Deserialize)]
pub struct OutboundConfig {
    pub queue_path: String,
    #[serde(default = "default_max_retries")]
    pub max_retries: usize,
    #[serde(default = "default_retry_schedule")]
    pub retry_schedule: Vec<u64>,
}

fn default_max_retries() -> usize {
    10
}
fn default_retry_schedule() -> Vec<u64> {
    vec![
        300, 900, 1800, 3600, 7200, 14400, 28800, 86400, 172800, 259200,
    ]
}

#[derive(Debug, Deserialize)]
pub struct HarmonyConfig {
    pub node_config: String,
    #[serde(default = "default_trust_topic")]
    pub trust_topic: String,
    #[serde(default = "default_announce_interval")]
    pub announce_interval: u64,
}

fn default_trust_topic() -> String {
    "harmony/mail/trust/v1".to_string()
}
fn default_announce_interval() -> u64 {
    3600
}

impl Config {
    pub fn from_toml(s: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let config: Self = toml::from_str(s)?;
        config.tls.validate()?;
        Ok(config)
    }

    pub fn from_file(path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let contents = std::fs::read_to_string(path)?;
        Self::from_toml(&contents)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const FULL_CONFIG: &str = r#"
[domain]
name = "example.com"
mx_host = "mail.example.com"

[gateway]
identity_key = "/etc/harmony-mail/identity.key"
listen_smtp = "0.0.0.0:25"
listen_submission = "0.0.0.0:465"
listen_submission_starttls = "0.0.0.0:587"

[tls]
mode = "acme"
acme_email = "admin@example.com"
acme_challenge = "dns-01"

[dkim]
selector = "harmony"
algorithm = "ed25519"
key = "/etc/harmony-mail/dkim.key"

[spam]
dnsbl = ["zen.spamhaus.org", "bl.spamcop.net"]
reject_threshold = 5
max_connections_per_ip = 5
max_message_size = "25MB"

[outbound]
queue_path = "/var/spool/harmony-mail"
max_retries = 10
retry_schedule = [300, 900, 1800, 3600, 7200, 14400, 28800, 86400, 172800, 259200]

[harmony]
node_config = "/etc/harmony/node.toml"
trust_topic = "harmony/mail/trust/v1"
announce_interval = 3600
"#;

    #[test]
    fn parse_full_config() {
        let config = Config::from_toml(FULL_CONFIG).expect("should parse full config");

        // Domain
        assert_eq!(config.domain.name, "example.com");
        assert_eq!(config.domain.mx_host, "mail.example.com");

        // Gateway
        assert_eq!(
            config.gateway.identity_key,
            "/etc/harmony-mail/identity.key"
        );
        assert_eq!(config.gateway.listen_smtp, "0.0.0.0:25");
        assert_eq!(config.gateway.listen_submission, "0.0.0.0:465");
        assert_eq!(config.gateway.listen_submission_starttls, "0.0.0.0:587");

        // TLS
        assert_eq!(config.tls.mode, TlsMode::Acme);
        assert_eq!(config.tls.acme_email.as_deref(), Some("admin@example.com"));
        assert_eq!(config.tls.acme_challenge, "dns-01");
        assert!(config.tls.cert.is_none());
        assert!(config.tls.key.is_none());

        // DKIM
        assert_eq!(config.dkim.selector, "harmony");
        assert_eq!(config.dkim.algorithm, "ed25519");
        assert_eq!(config.dkim.key, "/etc/harmony-mail/dkim.key");

        // Spam
        assert_eq!(config.spam.dnsbl.len(), 2);
        assert_eq!(config.spam.dnsbl[0], "zen.spamhaus.org");
        assert_eq!(config.spam.dnsbl[1], "bl.spamcop.net");
        assert_eq!(config.spam.reject_threshold, 5);
        assert_eq!(config.spam.max_connections_per_ip, 5);
        assert_eq!(config.spam.max_message_size, "25MB");

        // Outbound
        assert_eq!(config.outbound.queue_path, "/var/spool/harmony-mail");
        assert_eq!(config.outbound.max_retries, 10);
        assert_eq!(config.outbound.retry_schedule.len(), 10);
        assert_eq!(config.outbound.retry_schedule[0], 300);

        // Harmony
        assert_eq!(config.harmony.node_config, "/etc/harmony/node.toml");
        assert_eq!(config.harmony.trust_topic, "harmony/mail/trust/v1");
        assert_eq!(config.harmony.announce_interval, 3600);
    }

    #[test]
    fn parse_minimal_config() {
        let minimal = r#"
[domain]
name = "minimal.example"
mx_host = "mx.minimal.example"

[gateway]
identity_key = "/tmp/test.key"

[tls]
acme_email = "admin@minimal.example"

[dkim]
key = "/tmp/dkim.key"

[spam]

[outbound]
queue_path = "/tmp/queue"

[harmony]
node_config = "/tmp/node.toml"
"#;

        let config = Config::from_toml(minimal).expect("should parse minimal config");

        // Required fields present
        assert_eq!(config.domain.name, "minimal.example");
        assert_eq!(config.gateway.identity_key, "/tmp/test.key");
        assert_eq!(config.dkim.key, "/tmp/dkim.key");
        assert_eq!(config.outbound.queue_path, "/tmp/queue");
        assert_eq!(config.harmony.node_config, "/tmp/node.toml");

        // Defaults applied
        assert_eq!(config.gateway.listen_smtp, "0.0.0.0:25");
        assert_eq!(config.gateway.listen_submission, "0.0.0.0:465");
        assert_eq!(config.gateway.listen_submission_starttls, "0.0.0.0:587");
        assert_eq!(config.tls.mode, TlsMode::Acme);
        assert_eq!(config.tls.acme_challenge, "dns-01");
        assert_eq!(
            config.tls.acme_email.as_deref(),
            Some("admin@minimal.example")
        );
        assert_eq!(config.dkim.selector, "harmony");
        assert_eq!(config.dkim.algorithm, "ed25519");
        assert!(config.spam.dnsbl.is_empty());
        assert_eq!(config.spam.reject_threshold, 5);
        assert_eq!(config.spam.max_connections_per_ip, 5);
        assert_eq!(config.spam.max_message_size, "25MB");
        assert_eq!(config.outbound.max_retries, 10);
        assert_eq!(config.outbound.retry_schedule.len(), 10);
        assert_eq!(config.harmony.trust_topic, "harmony/mail/trust/v1");
        assert_eq!(config.harmony.announce_interval, 3600);
    }

    #[test]
    fn tls_manual_mode_requires_cert_and_key() {
        let manual_tls = r#"
[domain]
name = "example.com"
mx_host = "mail.example.com"

[gateway]
identity_key = "/tmp/test.key"

[tls]
mode = "manual"

[dkim]
key = "/tmp/dkim.key"

[spam]

[outbound]
queue_path = "/tmp/queue"

[harmony]
node_config = "/tmp/node.toml"
"#;

        let result = Config::from_toml(manual_tls);
        assert!(
            result.is_err(),
            "manual mode without cert/key should fail validation"
        );
    }

    #[test]
    fn tls_invalid_mode_rejected() {
        let bad_mode = r#"
[domain]
name = "example.com"
mx_host = "mail.example.com"

[gateway]
identity_key = "/tmp/test.key"

[tls]
mode = "typo"

[dkim]
key = "/tmp/dkim.key"

[spam]

[outbound]
queue_path = "/tmp/queue"

[harmony]
node_config = "/tmp/node.toml"
"#;

        let result = Config::from_toml(bad_mode);
        assert!(
            result.is_err(),
            "invalid TLS mode should fail deserialization"
        );
    }

    #[test]
    fn tls_acme_mode_requires_email() {
        let acme_no_email = r#"
[domain]
name = "example.com"
mx_host = "mail.example.com"

[gateway]
identity_key = "/tmp/test.key"

[tls]
mode = "acme"

[dkim]
key = "/tmp/dkim.key"

[spam]

[outbound]
queue_path = "/tmp/queue"

[harmony]
node_config = "/tmp/node.toml"
"#;

        let result = Config::from_toml(acme_no_email);
        assert!(
            result.is_err(),
            "acme mode without acme_email should fail validation"
        );
    }

    #[test]
    fn parse_invalid_config() {
        // Missing required [domain] section entirely
        let invalid = r#"
[gateway]
identity_key = "/tmp/test.key"
"#;

        let result = Config::from_toml(invalid);
        assert!(
            result.is_err(),
            "should fail when required field is missing"
        );
    }
}
