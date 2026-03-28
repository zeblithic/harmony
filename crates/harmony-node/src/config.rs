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
                write!(
                    f,
                    "failed to read config file {}: {}",
                    path.display(),
                    source
                )
            }
            ConfigError::Parse { path, message } => {
                write!(
                    f,
                    "failed to parse config file {}: {}",
                    path.display(),
                    message
                )
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
    /// Directory for persistent CAS book storage. When set, durable books
    /// are written to disk and reloaded on restart.
    pub data_dir: Option<PathBuf>,
    /// Disk quota for CAS book storage (e.g. "10 GiB"). Requires `data_dir`.
    /// If absent, disk usage is unbounded.
    pub disk_quota: Option<String>,
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
    pub did_web_cache_ttl: Option<u64>,
    pub rawlink_interface: Option<String>,
    pub archivist: Option<ArchivistConfig>,
    /// Hex-encoded 32-byte CID of the GGUF model file in CAS.
    pub inference_model_gguf_cid: Option<String>,
    /// Hex-encoded 32-byte CID of the tokenizer.json file in CAS.
    pub inference_model_tokenizer_cid: Option<String>,
    /// Hex-encoded 32-byte CID of the Engram manifest in CAS.
    pub engram_manifest_cid: Option<String>,
}

#[derive(Deserialize, Debug)]
#[serde(deny_unknown_fields)]
pub struct ArchivistConfig {
    pub bucket: String,
    pub prefix: String,
    pub region: Option<String>,
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
// Byte-size parser
// ---------------------------------------------------------------------------

/// Parse a human-readable byte size string like "10 GiB" or "500 MB".
///
/// Supported suffixes (case-insensitive): B, KB, MB, GB, TB, KiB, MiB, GiB, TiB.
/// A bare number without a suffix is rejected to avoid ambiguity.
pub fn parse_byte_size(s: &str) -> Result<u64, String> {
    let s = s.trim();
    if s.is_empty() {
        return Err("empty byte size string".into());
    }
    let num_end = s.find(|c: char| !c.is_ascii_digit()).unwrap_or(s.len());
    if num_end == 0 {
        return Err(format!("no numeric value in '{s}'"));
    }
    let num: u64 = s[..num_end]
        .parse()
        .map_err(|e| format!("invalid number in '{s}': {e}"))?;
    let suffix = s[num_end..].trim().to_ascii_lowercase();
    let multiplier: u64 = match suffix.as_str() {
        "b" => 1,
        "kb" => 1_000,
        "mb" => 1_000_000,
        "gb" => 1_000_000_000,
        "kib" => 1_024,
        "mib" => 1_024 * 1_024,
        "gib" => 1_024 * 1_024 * 1_024,
        "tb" => 1_000_000_000_000,
        "tib" => 1_024 * 1_024 * 1_024 * 1_024,
        "" => {
            return Err(format!(
                "bare number '{s}' requires a unit suffix (B, KB, MB, GB, TB, KiB, MiB, GiB, TiB)"
            ))
        }
        other => return Err(format!("unknown unit suffix '{other}' in '{s}'")),
    };
    num.checked_mul(multiplier)
        .ok_or_else(|| format!("byte size overflow: {num} * {multiplier}"))
}

// ---------------------------------------------------------------------------
// Load
// ---------------------------------------------------------------------------

/// Load a `ConfigFile` from `path`.
///
/// When `explicit` is false (default path), returns `Ok(ConfigFile::default())`
/// if the file does not exist. When `explicit` is true (`--config` was given),
/// a missing file is an error — the user explicitly asked for this file.
pub fn load(path: &Path, explicit: bool) -> Result<ConfigFile, ConfigError> {
    let content = match std::fs::read_to_string(path) {
        Ok(s) => s,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound && !explicit => {
            return Ok(ConfigFile::default());
        }
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
    // If $HOME is not set (containers, CI), use a path that won't exist.
    // load() with explicit=false treats missing files as all-defaults.
    let Ok(home) = std::env::var("HOME") else {
        return Ok((PathBuf::from("/nonexistent/.harmony/node.toml"), false));
    };
    Ok((
        PathBuf::from(home).join(".harmony").join("node.toml"),
        false,
    ))
}

// ---------------------------------------------------------------------------
// Merge helpers
// ---------------------------------------------------------------------------

/// Three-way merge: CLI (highest) > config file > hardcoded default.
pub fn resolve<T>(cli: Option<T>, file: Option<T>, default: T) -> T {
    cli.or(file).unwrap_or(default)
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
        let cfg = load(&path, false).expect("missing file should return Ok(default)");
        assert!(cfg.listen_address.is_none());
        assert!(cfg.peers.is_none());
        assert!(cfg.tunnels.is_none());
    }

    #[test]
    fn load_missing_explicit_file_errors() {
        let path = PathBuf::from("/tmp/harmony-config-does-not-exist-xyzzy.toml");
        let err = load(&path, true).expect_err("explicit missing file should error");
        assert!(matches!(err, ConfigError::Io { .. }));
    }

    #[test]
    fn load_valid_minimal_file() {
        let (_f, path) = write_temp(r#"listen_address = "127.0.0.1:4242""#);
        let cfg = load(&path, false).expect("should parse minimal config");
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
node_id = "aabbccddeeff00112233445566778899aabbccddeeff00112233445566778899"
name = "my-peer"

[[tunnels]]
node_id = "112233445566778899aabbccddeeff00112233445566778899aabbccddeeff00"
"#;
        let (_f, path) = write_temp(toml);
        let cfg = load(&path, false).expect("should parse full config");

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
        assert_eq!(
            tunnels[0].node_id,
            "aabbccddeeff00112233445566778899aabbccddeeff00112233445566778899"
        );
        assert_eq!(tunnels[0].name.as_deref(), Some("my-peer"));
        assert_eq!(
            tunnels[1].node_id,
            "112233445566778899aabbccddeeff00112233445566778899aabbccddeeff00"
        );
        assert!(tunnels[1].name.is_none());
    }

    #[test]
    fn load_malformed_toml_errors() {
        let (_f, path) = write_temp("listen_address = [unclosed");
        let err = load(&path, false).expect_err("malformed TOML should fail");
        assert!(matches!(err, ConfigError::Parse { .. }));
        let msg = err.to_string();
        assert!(
            msg.contains("failed to parse config file"),
            "unexpected: {msg}"
        );
    }

    #[test]
    fn load_unknown_key_errors() {
        let (_f, path) = write_temp("typo_key = true\n");
        let err =
            load(&path, false).expect_err("unknown key should fail due to deny_unknown_fields");
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

    // ── parse_byte_size tests ───────────────────────────────────────────

    #[test]
    fn parse_byte_size_binary_units() {
        assert_eq!(parse_byte_size("1 KiB").unwrap(), 1024);
        assert_eq!(parse_byte_size("2 MiB").unwrap(), 2 * 1024 * 1024);
        assert_eq!(parse_byte_size("10 GiB").unwrap(), 10 * 1024 * 1024 * 1024);
        assert_eq!(
            parse_byte_size("2 TiB").unwrap(),
            2 * 1024 * 1024 * 1024 * 1024
        );
    }

    #[test]
    fn parse_byte_size_decimal_units() {
        assert_eq!(parse_byte_size("500 MB").unwrap(), 500 * 1_000_000);
        assert_eq!(parse_byte_size("1 GB").unwrap(), 1_000_000_000);
        assert_eq!(parse_byte_size("100 KB").unwrap(), 100_000);
        assert_eq!(parse_byte_size("1 TB").unwrap(), 1_000_000_000_000);
    }

    #[test]
    fn parse_byte_size_bytes_suffix() {
        assert_eq!(parse_byte_size("1024 B").unwrap(), 1024);
    }

    #[test]
    fn parse_byte_size_case_insensitive() {
        assert_eq!(parse_byte_size("1 gib").unwrap(), 1024 * 1024 * 1024);
        assert_eq!(parse_byte_size("1 GIB").unwrap(), 1024 * 1024 * 1024);
        assert_eq!(parse_byte_size("500 mb").unwrap(), 500 * 1_000_000);
    }

    #[test]
    fn parse_byte_size_no_space() {
        assert_eq!(parse_byte_size("10GiB").unwrap(), 10 * 1024 * 1024 * 1024);
    }

    #[test]
    fn parse_byte_size_bare_number_rejected() {
        assert!(parse_byte_size("1234").is_err());
    }

    #[test]
    fn parse_byte_size_invalid_suffix_rejected() {
        assert!(parse_byte_size("10 PB").is_err());
        assert!(parse_byte_size("abc").is_err());
        assert!(parse_byte_size("").is_err());
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
}
