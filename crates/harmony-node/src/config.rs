use serde::Deserialize;
use std::path::{Path, PathBuf};

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub enum ConfigError {
    Io {
        path: PathBuf,
        source: std::io::Error,
    },
    Parse {
        path: PathBuf,
        message: String,
    },
}

impl std::fmt::Display for ConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ConfigError::Io { path, source } => {
                write!(f, "failed to read config file {}: {}", path.display(), source)
            }
            ConfigError::Parse { path, message } => {
                write!(f, "failed to parse config file {}: {}", path.display(), message)
            }
        }
    }
}

impl std::error::Error for ConfigError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            ConfigError::Io { source, .. } => Some(source),
            ConfigError::Parse { .. } => None,
        }
    }
}

// ---------------------------------------------------------------------------
// Serde structs
// ---------------------------------------------------------------------------

#[derive(Deserialize, Default, Debug)]
#[serde(deny_unknown_fields)]
pub struct ConfigFile {
    pub listen_address: Option<String>,
    pub identity_file: Option<PathBuf>,
    pub cache_capacity: Option<usize>,
    pub compute_budget: Option<u64>,
    pub filter_broadcast_ticks: Option<u32>,
    pub filter_mutation_threshold: Option<u32>,
    pub encrypted_durable_persist: Option<bool>,
    pub encrypted_durable_announce: Option<bool>,
    pub no_public_ephemeral_announce: Option<bool>,
    pub no_mdns: Option<bool>,
    pub mdns_stale_timeout: Option<u64>,
    pub relay_url: Option<String>,
    pub logging: Option<LoggingConfig>,
    pub peers: Option<Vec<PeerEntry>>,
    pub tunnels: Option<Vec<TunnelEntry>>,
}

#[derive(Deserialize, Debug)]
#[serde(deny_unknown_fields)]
pub struct LoggingConfig {
    pub level: Option<String>,
}

#[derive(Deserialize, Debug)]
#[serde(deny_unknown_fields)]
pub struct PeerEntry {
    pub address: String,
}

#[derive(Deserialize, Debug, Clone)]
#[serde(deny_unknown_fields)]
pub struct TunnelEntry {
    pub node_id: String,
    pub name: Option<String>,
}

// ---------------------------------------------------------------------------
// Load
// ---------------------------------------------------------------------------

/// Load a `ConfigFile` from `path`.
///
/// Returns `Ok(ConfigFile::default())` when the file does not exist, so callers
/// can treat a missing config identically to an all-defaults config.
pub fn load(path: &Path) -> Result<ConfigFile, ConfigError> {
    let content = match std::fs::read_to_string(path) {
        Ok(s) => s,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(ConfigFile::default()),
        Err(e) => {
            return Err(ConfigError::Io {
                path: path.to_path_buf(),
                source: e,
            })
        }
    };
    toml::from_str(&content).map_err(|e| ConfigError::Parse {
        path: path.to_path_buf(),
        message: e.to_string(),
    })
}

// ---------------------------------------------------------------------------
// Path resolution
// ---------------------------------------------------------------------------

/// Resolve the config file path.
///
/// Returns `(path, explicit)` where `explicit` is `true` when the caller
/// supplied the path via CLI and `false` when the default was used.
pub fn resolve_config_path(cli_override: Option<&Path>) -> Result<(PathBuf, bool), String> {
    if let Some(p) = cli_override {
        return Ok((p.to_path_buf(), true));
    }
    let home = std::env::var("HOME").map_err(|_| {
        "Cannot determine config file path: $HOME not set. Use --config.".to_string()
    })?;
    Ok((PathBuf::from(home).join(".harmony").join("node.toml"), false))
}

// ---------------------------------------------------------------------------
// Merge helpers
// ---------------------------------------------------------------------------

/// Three-way merge: CLI (highest) > config file > hardcoded default.
pub fn resolve<T>(cli: Option<T>, file: Option<T>, default: T) -> T {
    cli.or(file).unwrap_or(default)
}

/// Merge a boolean CLI flag with a config file value.
/// CLI booleans are `bool` not `Option<bool>` — `true` means "explicitly passed".
/// Only works for flags where the CLI default is `false`.
pub fn resolve_bool(cli: bool, file: Option<bool>, default: bool) -> bool {
    if cli { true } else { file.unwrap_or(default) }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write as _;

    fn write_temp(content: &str) -> (tempfile::NamedTempFile, PathBuf) {
        let mut f = tempfile::NamedTempFile::new().unwrap();
        f.write_all(content.as_bytes()).unwrap();
        let path = f.path().to_path_buf();
        (f, path)
    }

    #[test]
    fn load_missing_file_returns_default() {
        let path = PathBuf::from("/tmp/harmony-config-does-not-exist-xyzzy.toml");
        let cfg = load(&path).expect("missing file should return Ok(default)");
        assert!(cfg.listen_address.is_none());
        assert!(cfg.peers.is_none());
        assert!(cfg.tunnels.is_none());
    }

    #[test]
    fn load_valid_minimal_file() {
        let (_f, path) = write_temp(r#"listen_address = "127.0.0.1:4242""#);
        let cfg = load(&path).expect("should parse minimal config");
        assert_eq!(cfg.listen_address.as_deref(), Some("127.0.0.1:4242"));
        assert!(cfg.cache_capacity.is_none());
    }

    #[test]
    fn load_valid_full_file() {
        let toml = r#"
listen_address = "0.0.0.0:4242"
cache_capacity = 2048
compute_budget = 50000
filter_broadcast_ticks = 60
filter_mutation_threshold = 200
encrypted_durable_persist = true
encrypted_durable_announce = true
no_public_ephemeral_announce = false
no_mdns = false
mdns_stale_timeout = 120
relay_url = "https://relay.example.com"

[logging]
level = "debug"

[[peers]]
address = "192.168.1.10:4242"

[[peers]]
address = "192.168.1.11:4242"

[[tunnels]]
node_id = "aabbccddeeff"
name = "my-peer"

[[tunnels]]
node_id = "112233445566"
"#;
        let (_f, path) = write_temp(toml);
        let cfg = load(&path).expect("should parse full config");

        assert_eq!(cfg.listen_address.as_deref(), Some("0.0.0.0:4242"));
        assert_eq!(cfg.cache_capacity, Some(2048));
        assert_eq!(cfg.compute_budget, Some(50000));
        assert_eq!(cfg.filter_broadcast_ticks, Some(60));
        assert_eq!(cfg.filter_mutation_threshold, Some(200));
        assert_eq!(cfg.encrypted_durable_persist, Some(true));
        assert_eq!(cfg.encrypted_durable_announce, Some(true));
        assert_eq!(cfg.no_public_ephemeral_announce, Some(false));
        assert_eq!(cfg.no_mdns, Some(false));
        assert_eq!(cfg.mdns_stale_timeout, Some(120));
        assert_eq!(cfg.relay_url.as_deref(), Some("https://relay.example.com"));

        let logging = cfg.logging.as_ref().expect("logging section present");
        assert_eq!(logging.level.as_deref(), Some("debug"));

        let peers = cfg.peers.as_ref().expect("peers present");
        assert_eq!(peers.len(), 2);
        assert_eq!(peers[0].address, "192.168.1.10:4242");
        assert_eq!(peers[1].address, "192.168.1.11:4242");

        let tunnels = cfg.tunnels.as_ref().expect("tunnels present");
        assert_eq!(tunnels.len(), 2);
        assert_eq!(tunnels[0].node_id, "aabbccddeeff");
        assert_eq!(tunnels[0].name.as_deref(), Some("my-peer"));
        assert_eq!(tunnels[1].node_id, "112233445566");
        assert!(tunnels[1].name.is_none());
    }

    #[test]
    fn load_malformed_toml_errors() {
        let (_f, path) = write_temp("listen_address = [unclosed");
        let err = load(&path).expect_err("malformed TOML should fail");
        assert!(matches!(err, ConfigError::Parse { .. }));
        let msg = err.to_string();
        assert!(msg.contains("failed to parse config file"), "unexpected: {msg}");
    }

    #[test]
    fn load_unknown_key_errors() {
        let (_f, path) = write_temp("typo_key = true\n");
        let err = load(&path).expect_err("unknown key should fail due to deny_unknown_fields");
        assert!(matches!(err, ConfigError::Parse { .. }));
    }

    #[test]
    fn resolve_config_path_explicit() {
        let explicit = Path::new("/etc/harmony/node.toml");
        let (path, is_explicit) =
            resolve_config_path(Some(explicit)).expect("explicit override must succeed");
        assert_eq!(path, explicit);
        assert!(is_explicit);
    }

    #[test]
    fn resolve_config_path_default_uses_home() {
        // Ensure $HOME is set (it always is in a normal test environment).
        let home = std::env::var("HOME").expect("$HOME must be set for this test");
        let (path, is_explicit) =
            resolve_config_path(None).expect("default path resolution must succeed");
        assert!(!is_explicit);
        let expected = PathBuf::from(home).join(".harmony").join("node.toml");
        assert_eq!(path, expected);
    }

    // ── Merge helper tests ──────────────────────────────────────────────

    #[test]
    fn resolve_cli_wins() {
        assert_eq!(resolve(Some(42), Some(99), 0), 42);
    }

    #[test]
    fn resolve_file_wins_when_cli_none() {
        assert_eq!(resolve(None, Some(99), 0), 99);
    }

    #[test]
    fn resolve_default_when_both_none() {
        assert_eq!(resolve::<i32>(None, None, 7), 7);
    }

    #[test]
    fn resolve_bool_cli_true_wins() {
        assert_eq!(resolve_bool(true, Some(false), false), true);
    }

    #[test]
    fn resolve_bool_cli_false_defers_to_file() {
        assert_eq!(resolve_bool(false, Some(true), false), true);
    }

    #[test]
    fn resolve_bool_all_false_returns_default() {
        assert_eq!(resolve_bool(false, None, false), false);
    }
}
