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

#[derive(Debug, Deserialize)]
pub struct TlsConfig {
    #[serde(default = "default_tls_mode")]
    pub mode: String, // "acme" or "manual"
    pub acme_email: Option<String>,
    #[serde(default = "default_acme_challenge")]
    pub acme_challenge: String,
    pub cert: Option<String>,
    pub key: Option<String>,
}

fn default_tls_mode() -> String {
    "acme".to_string()
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
    pub fn from_toml(s: &str) -> Result<Self, toml::de::Error> {
        toml::from_str(s)
    }

    pub fn from_file(path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let contents = std::fs::read_to_string(path)?;
        let config = Self::from_toml(&contents)?;
        Ok(config)
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
        assert_eq!(config.tls.mode, "acme");
        assert_eq!(
            config.tls.acme_email.as_deref(),
            Some("admin@example.com")
        );
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
        assert_eq!(config.tls.mode, "acme");
        assert_eq!(config.tls.acme_challenge, "dns-01");
        assert!(config.tls.acme_email.is_none());
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
    fn parse_invalid_config() {
        // Missing required [domain] section entirely
        let invalid = r#"
[gateway]
identity_key = "/tmp/test.key"
"#;

        let result = Config::from_toml(invalid);
        assert!(result.is_err(), "should fail when required field is missing");
    }
}
