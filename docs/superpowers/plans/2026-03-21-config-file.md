# TOML Config File Support Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Load harmony-node configuration from a TOML file with CLI override precedence, plus new config-only sections for bootstrap peers, tunnel peers, and logging.

**Architecture:** A new `config.rs` module defines serde-deserializable structs matching the TOML format. All `ConfigFile` fields are `Option<T>`. CLI flags with `default_value_t` change to `Option<T>` so the three-way merge works: `cli.or(file).unwrap_or(hardcoded_default)`. Tracing uses a `reload::Handle` so the log level from the config file can be applied after subscriber init.

**Tech Stack:** Rust, serde 1 (derive), toml 0.8, tracing-subscriber (reload), clap 4

**Spec:** `docs/superpowers/specs/2026-03-21-config-file-design.md`

---

## File Structure

| File | Responsibility |
|------|---------------|
| `crates/harmony-node/Cargo.toml` | Add `serde`, `toml` deps |
| `crates/harmony-node/src/config.rs` | **New**: ConfigFile serde struct, ConfigError, load(), resolve_config_path(), merge helpers |
| `crates/harmony-node/src/main.rs` | `mod config;`, `--config` flag, CLI fields → `Option<T>`, load+merge+validate, pass new params |
| `crates/harmony-node/src/event_loop.rs` | `bootstrap_peers` param, `ConfigTunnelPeers` reconnect logic in timer tick |

---

### Task 1: ConfigFile serde struct and load/parse (TDD)

**Files:**
- Modify: `crates/harmony-node/Cargo.toml`
- Create: `crates/harmony-node/src/config.rs`
- Modify: `crates/harmony-node/src/main.rs:1` (add `mod config;`)

Pure data types + file I/O. No integration with main.rs or event_loop.rs yet.

- [ ] **Step 1: Add serde and toml dependencies**

In `crates/harmony-node/Cargo.toml`, add to `[dependencies]`:

```toml
serde = { workspace = true, features = ["std", "derive"] }
toml.workspace = true
```

Also update the workspace `Cargo.toml` (root) to add the `reload` feature to
`tracing-subscriber`. Find the line `tracing-subscriber = { version = "0.3", features = ["env-filter"] }`
and change it to:

```toml
tracing-subscriber = { version = "0.3", features = ["env-filter", "reload"] }
```

This is needed for Task 3's `tracing_subscriber::reload::Layer`.

- [ ] **Step 2: Add `mod config;` to main.rs**

Add after the existing `mod compute;` line:

```rust
mod config;
```

- [ ] **Step 3: Create config.rs with serde structs, error type, and test stubs**

Create `crates/harmony-node/src/config.rs` with:

```rust
use serde::Deserialize;
use std::path::{Path, PathBuf};

// ── Error type ──────────────────────────────────────────────────────────────

#[derive(Debug)]
pub enum ConfigError {
    Io { path: PathBuf, source: std::io::Error },
    Parse { path: PathBuf, message: String },
}

impl std::fmt::Display for ConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io { path, source } => write!(f, "config file {}: {source}", path.display()),
            Self::Parse { path, message } => write!(f, "config file {}: {message}", path.display()),
        }
    }
}

impl std::error::Error for ConfigError {}

// ── Config structs ──────────────────────────────────────────────────────────

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

// ── Loading ─────────────────────────────────────────────────────────────────

/// Load a config file from `path`. Returns `Ok(ConfigFile::default())` if the
/// file does not exist. Returns `Err` if the file exists but is malformed.
pub fn load(path: &Path) -> Result<ConfigFile, ConfigError> {
    todo!()
}

/// Resolve the config file path. Returns `(path, explicit)` where `explicit`
/// is true when `--config` was provided. If explicit and file doesn't exist,
/// returns `Err`. If implicit and file doesn't exist, returns `Ok` (caller
/// proceeds with defaults).
pub fn resolve_config_path(cli_override: Option<&Path>) -> Result<(PathBuf, bool), String> {
    todo!()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn load_missing_file_returns_default() {
        let result = load(Path::new("/tmp/nonexistent-harmony-test.toml"));
        assert!(result.is_ok());
        let cfg = result.unwrap();
        assert!(cfg.listen_address.is_none());
        assert!(cfg.peers.is_none());
        assert!(cfg.tunnels.is_none());
    }

    #[test]
    fn load_valid_minimal_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.toml");
        std::fs::write(&path, "listen_address = \"127.0.0.1:9999\"\n").unwrap();
        let cfg = load(&path).unwrap();
        assert_eq!(cfg.listen_address.as_deref(), Some("127.0.0.1:9999"));
    }

    #[test]
    fn load_valid_full_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.toml");
        std::fs::write(&path, r#"
listen_address = "0.0.0.0:4242"
cache_capacity = 512
compute_budget = 50000
filter_broadcast_ticks = 60
filter_mutation_threshold = 200
encrypted_durable_persist = true
encrypted_durable_announce = true
no_public_ephemeral_announce = false
no_mdns = false
mdns_stale_timeout = 120
relay_url = "https://i.q8.fyi"

[logging]
level = "debug"

[[peers]]
address = "192.168.1.10:4242"

[[peers]]
address = "10.0.0.5:4242"

[[tunnels]]
node_id = "aabbccdd"
name = "gateway"
"#).unwrap();
        let cfg = load(&path).unwrap();
        assert_eq!(cfg.cache_capacity, Some(512));
        assert_eq!(cfg.peers.as_ref().unwrap().len(), 2);
        assert_eq!(cfg.tunnels.as_ref().unwrap().len(), 1);
        assert_eq!(cfg.tunnels.as_ref().unwrap()[0].name.as_deref(), Some("gateway"));
        assert_eq!(cfg.logging.as_ref().unwrap().level.as_deref(), Some("debug"));
    }

    #[test]
    fn load_malformed_toml_errors() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("bad.toml");
        std::fs::write(&path, "not valid toml {{{\n").unwrap();
        let result = load(&path);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, ConfigError::Parse { .. }));
    }

    #[test]
    fn load_unknown_key_errors() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("bad.toml");
        std::fs::write(&path, "cache_capactiy = 512\n").unwrap(); // typo
        let result = load(&path);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), ConfigError::Parse { .. }));
    }

    #[test]
    fn resolve_config_path_explicit() {
        let (path, explicit) = resolve_config_path(Some(Path::new("/etc/harmony/node.toml"))).unwrap();
        assert_eq!(path, PathBuf::from("/etc/harmony/node.toml"));
        assert!(explicit);
    }

    #[test]
    fn resolve_config_path_default_uses_home() {
        // This test depends on $HOME being set (normal on all dev machines)
        if std::env::var("HOME").is_ok() {
            let (path, explicit) = resolve_config_path(None).unwrap();
            assert!(!explicit);
            assert!(path.to_string_lossy().ends_with(".harmony/node.toml"));
        }
    }
}
```

- [ ] **Step 4: Run tests — verify they fail with todo!()**

Run: `cargo test -p harmony-node -- config::tests`
Expected: todo!() panics

- [ ] **Step 5: Implement load() and resolve_config_path()**

```rust
pub fn load(path: &Path) -> Result<ConfigFile, ConfigError> {
    let content = match std::fs::read_to_string(path) {
        Ok(s) => s,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            return Ok(ConfigFile::default());
        }
        Err(e) => return Err(ConfigError::Io { path: path.to_path_buf(), source: e }),
    };
    toml::from_str(&content).map_err(|e| ConfigError::Parse {
        path: path.to_path_buf(),
        message: e.to_string(),
    })
}

pub fn resolve_config_path(cli_override: Option<&Path>) -> Result<(PathBuf, bool), String> {
    if let Some(p) = cli_override {
        return Ok((p.to_path_buf(), true));
    }
    let home = std::env::var("HOME")
        .map_err(|_| "Cannot determine config file path: $HOME not set. Use --config.".to_string())?;
    Ok((PathBuf::from(home).join(".harmony").join("node.toml"), false))
}
```

- [ ] **Step 6: Run tests — verify they pass**

Run: `cargo test -p harmony-node -- config::tests -v`
Expected: all 7 tests pass

- [ ] **Step 7: Commit**

```bash
git add crates/harmony-node/Cargo.toml crates/harmony-node/src/config.rs crates/harmony-node/src/main.rs
git commit -m "feat(harmony-node): add ConfigFile serde struct and TOML loader

ConfigFile with deny_unknown_fields, ConfigError type, load() that
returns default for missing files and errors for malformed TOML.
7 unit tests.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Merge helpers and CLI field type changes

**Files:**
- Modify: `crates/harmony-node/src/config.rs` (add merge function)
- Modify: `crates/harmony-node/src/main.rs` (change CLI field types to Option<T>, add --config flag)

Change CLI flags from `T` with `default_value_t` to `Option<T>`, add `--config` flag, implement three-way merge.

- [ ] **Step 1: Add merge helper and tests to config.rs**

Add to `config.rs` (above `#[cfg(test)]`):

```rust
// ── Merge helpers ───────────────────────────────────────────────────────────

/// Three-way merge: CLI (highest priority) > config file > hardcoded default.
pub fn resolve<T>(cli: Option<T>, file: Option<T>, default: T) -> T {
    cli.or(file).unwrap_or(default)
}

/// Merge a boolean CLI flag with a config file value.
/// CLI booleans are `bool` not `Option<bool>` — `true` means "explicitly passed",
/// `false` means "not passed" (use file or default).
pub fn resolve_bool(cli: bool, file: Option<bool>, default: bool) -> bool {
    if cli { true } else { file.unwrap_or(default) }
}
```

Add these tests to the test module:

```rust
    #[test]
    fn resolve_cli_wins() {
        assert_eq!(resolve(Some(42), Some(99), 0), 42);
    }

    #[test]
    fn resolve_file_wins_when_cli_none() {
        assert_eq!(resolve::<u32>(None, Some(99), 0), 99);
    }

    #[test]
    fn resolve_default_when_both_none() {
        assert_eq!(resolve::<u32>(None, None, 7), 7);
    }

    #[test]
    fn resolve_bool_cli_true_wins() {
        assert!(resolve_bool(true, Some(false), false));
    }

    #[test]
    fn resolve_bool_cli_false_defers_to_file() {
        assert!(resolve_bool(false, Some(true), false));
    }

    #[test]
    fn resolve_bool_all_false_returns_default() {
        assert!(!resolve_bool(false, None, false));
    }
```

- [ ] **Step 2: Change CLI fields to Option<T> and add --config flag in main.rs**

In the `Run` variant of `Commands`, change these fields:

```rust
    Run {
        /// Path to config file (default: ~/.harmony/node.toml)
        #[arg(long, value_name = "PATH")]
        config: Option<std::path::PathBuf>,
        /// W-TinyLFU cache capacity (number of items, default: 1024)
        #[arg(long)]
        cache_capacity: Option<usize>,
        /// WASM compute fuel budget per tick (default: 100000)
        #[arg(long)]
        compute_budget: Option<u64>,
        /// Accept encrypted durable (10) content for storage
        #[arg(long)]
        encrypted_durable_persist: bool,
        /// Announce encrypted durable (10) content on Zenoh
        #[arg(long)]
        encrypted_durable_announce: bool,
        /// Disable announcing public ephemeral (01) content on Zenoh
        #[arg(long)]
        no_public_ephemeral_announce: bool,
        /// Bloom filter broadcast interval in ticks (default: 30)
        #[arg(long)]
        filter_broadcast_ticks: Option<u32>,
        /// Bloom filter broadcast mutation threshold (default: 100)
        #[arg(long)]
        filter_mutation_threshold: Option<u32>,
        /// Path to the identity key file
        #[arg(long, value_name = "PATH")]
        identity_file: Option<std::path::PathBuf>,
        /// UDP listen address for Reticulum mesh packets (default: 0.0.0.0:4242)
        #[arg(long)]
        listen_address: Option<String>,
        /// Disable mDNS peer discovery (broadcast-only mode)
        #[arg(long)]
        no_mdns: bool,
        /// Seconds before evicting a silent mDNS peer (default: 60)
        #[arg(long)]
        mdns_stale_timeout: Option<u64>,
        /// iroh relay URL for NAT-traversal tunnels (enables tunnel accept)
        #[arg(long, value_name = "URL")]
        relay_url: Option<String>,
        /// iroh NodeId of a peer to connect to (outbound tunnel) [deprecated: use config file [[tunnels]]]
        #[arg(long, value_name = "NODE_ID")]
        tunnel_peer: Option<String>,
    },
```

- [ ] **Step 3: Update the Run handler to load config and merge**

In the `Commands::Run { ... }` match arm, add `config` to the destructure. Replace the validation + config-building section with:

```rust
        Commands::Run {
            config,
            cache_capacity,
            compute_budget,
            encrypted_durable_persist,
            encrypted_durable_announce,
            no_public_ephemeral_announce,
            filter_broadcast_ticks,
            filter_mutation_threshold,
            identity_file,
            listen_address,
            no_mdns,
            mdns_stale_timeout,
            relay_url,
            tunnel_peer,
        } => {
            use crate::config::{resolve, resolve_bool};
            use crate::runtime::{NodeConfig, NodeRuntime};
            use harmony_compute::InstructionBudget;
            use harmony_content::book::MemoryBookStore;
            use harmony_content::storage_tier::{
                ContentPolicy, FilterBroadcastConfig, StorageBudget,
            };

            // ── Load config file ────────────────────────────────────────
            let (config_path, explicit) = crate::config::resolve_config_path(config.as_deref())?;
            // Check existence before loading: explicit --config path must exist.
            if explicit && !config_path.exists() {
                return Err(format!("config file not found: {}", config_path.display()).into());
            }
            let config_file = crate::config::load(&config_path).map_err(|e| {
                format!("{e}")
            })?;

            // ── Merge CLI > file > defaults ─────────────────────────────
            let cache_capacity = resolve(cache_capacity, config_file.cache_capacity, 1024);
            let compute_budget = resolve(compute_budget, config_file.compute_budget, 100_000);
            let filter_broadcast_ticks = resolve(filter_broadcast_ticks, config_file.filter_broadcast_ticks, 30);
            let filter_mutation_threshold = resolve(filter_mutation_threshold, config_file.filter_mutation_threshold, 100);
            let mdns_stale_timeout = resolve(mdns_stale_timeout, config_file.mdns_stale_timeout, 60);
            let listen_address = resolve(listen_address, config_file.listen_address, "0.0.0.0:4242".to_string());
            let identity_file = identity_file.or(config_file.identity_file);
            let relay_url = relay_url.or(config_file.relay_url);
            let no_mdns = resolve_bool(no_mdns, config_file.no_mdns, false);
            let encrypted_durable_persist = resolve_bool(encrypted_durable_persist, config_file.encrypted_durable_persist, false);
            let encrypted_durable_announce = resolve_bool(encrypted_durable_announce, config_file.encrypted_durable_announce, false);
            let no_public_ephemeral_announce = resolve_bool(no_public_ephemeral_announce, config_file.no_public_ephemeral_announce, false);

            // ── Parse config-only sections ───────────────────────────────
            let bootstrap_peers: Vec<std::net::SocketAddr> = config_file.peers
                .unwrap_or_default()
                .iter()
                .map(|p| p.address.parse::<std::net::SocketAddr>())
                .collect::<Result<_, _>>()
                .map_err(|e| format!("invalid peer address in config file: {e}"))?;

            let tunnel_entries: Vec<crate::config::TunnelEntry> = {
                let mut entries = config_file.tunnels.unwrap_or_default();
                // Merge deprecated --tunnel-peer into the list
                if let Some(ref peer) = tunnel_peer {
                    entries.push(crate::config::TunnelEntry {
                        node_id: peer.clone(),
                        name: None,
                    });
                }
                // Validate tunnel node_ids are valid hex
                for entry in &entries {
                    hex::decode(&entry.node_id)
                        .map_err(|e| format!("invalid tunnel node_id '{}': {e}", entry.node_id))?;
                }
                entries
            };

            // ── Existing validation (unchanged) ─────────────────────────
            // (cache_capacity, filter_broadcast_ticks, encrypted_durable, mdns_stale_timeout, listen_addr validation)
            // ... rest of the existing validation and startup code, using merged values ...
```

Note: the existing validation code (`cache_capacity == 0`, etc.) stays exactly as-is, just operating on the merged values instead of raw CLI values.

- [ ] **Step 4: Update existing CLI tests for Option<T> changes**

Tests that check default values (e.g., `assert_eq!(cache_capacity, 1024)`) need to change to `assert!(cache_capacity.is_none())` since defaults are now applied during merge, not at parse time. Update each affected test.

- [ ] **Step 5: Add CLI test for --config flag**

```rust
    #[test]
    fn cli_parses_config_flag() {
        let cli = Cli::try_parse_from(["harmony", "run", "--config", "/tmp/test.toml"]).unwrap();
        if let Commands::Run { config, .. } = cli.command {
            assert_eq!(config, Some(std::path::PathBuf::from("/tmp/test.toml")));
        } else {
            panic!("expected Run command");
        }
    }

    #[test]
    fn cli_config_default_none() {
        let cli = Cli::try_parse_from(["harmony", "run"]).unwrap();
        if let Commands::Run { config, .. } = cli.command {
            assert!(config.is_none());
        } else {
            panic!("expected Run command");
        }
    }
```

- [ ] **Step 6: Run all tests**

Run: `cargo test -p harmony-node -v`
Expected: all tests pass

- [ ] **Step 7: Commit**

```bash
git add crates/harmony-node/src/config.rs crates/harmony-node/src/main.rs
git commit -m "feat(harmony-node): config file loading and CLI merge

CLI fields changed to Option<T> for three-way merge (CLI > file > default).
--config flag, resolve/resolve_bool helpers, config-only [peers] and
[[tunnels]] sections parsed and validated. --tunnel-peer merged with
[[tunnels]] for backward compat.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Tracing reload for config-driven log level

**Files:**
- Modify: `crates/harmony-node/src/main.rs` (tracing init changes)

Move tracing init to support reload, apply `[logging].level` from config file.

- [ ] **Step 1: Update tracing initialization to use reload handle**

In `main()`, replace the current tracing init block with:

```rust
    // Initialize tracing with a default filter and a reload handle.
    // The reload handle lets us reconfigure the filter after loading
    // the config file (which may specify [logging].level).
    let env_filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|e| {
            if std::env::var("RUST_LOG").is_ok() {
                eprintln!("Warning: invalid RUST_LOG directive ({e}), defaulting to info");
            }
            tracing_subscriber::EnvFilter::new("info")
        });
    let (filter, reload_handle) = tracing_subscriber::reload::Layer::new(env_filter);
    use tracing_subscriber::layer::SubscriberExt;
    use tracing_subscriber::util::SubscriberInitExt;
    tracing_subscriber::registry()
        .with(filter)
        .with(
            tracing_subscriber::fmt::layer()
                .with_target(false)
                .with_ansi(false)
                .without_time()
                .with_writer(std::io::stderr),
        )
        .init();
```

- [ ] **Step 2: Apply config file log level after loading**

In the `Commands::Run` handler, after loading the config file and before the merge section, add:

```rust
            // Apply config file log level if RUST_LOG is not set.
            if std::env::var("RUST_LOG").is_err() {
                if let Some(ref logging) = config_file.logging {
                    if let Some(ref level) = logging.level {
                        match level.parse::<tracing_subscriber::EnvFilter>() {
                            Ok(new_filter) => {
                                let _ = reload_handle.reload(new_filter);
                                tracing::debug!(level = %level, "applied log level from config file");
                            }
                            Err(e) => {
                                tracing::warn!(level = %level, err = %e, "invalid log level in config file — keeping default");
                            }
                        }
                    }
                }
            }
```

The `reload_handle` must be passed from `main()` into the `run()` function. Define
a type alias at the top of `main.rs`:

```rust
type LogReloadHandle = tracing_subscriber::reload::Handle<
    tracing_subscriber::EnvFilter,
    tracing_subscriber::Registry,
>;
```

Then update `run()` signature:

```rust
async fn run(cli: Cli, reload_handle: LogReloadHandle) -> Result<(), Box<dyn std::error::Error>> {
```

And update `main()` to call `run(cli, reload_handle).await`.

- [ ] **Step 3: Run tests**

Run: `cargo test -p harmony-node -v`
Expected: all tests pass (tracing init change should be transparent to tests)

- [ ] **Step 4: Commit**

```bash
git add crates/harmony-node/src/main.rs
git commit -m "feat(harmony-node): config-driven log level via tracing reload

Tracing subscriber uses reload::Handle so [logging].level from config
file can be applied after loading. RUST_LOG env var takes precedence.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: Bootstrap peers in event loop

**Files:**
- Modify: `crates/harmony-node/src/event_loop.rs` (add bootstrap_peers param)
- Modify: `crates/harmony-node/src/main.rs` (pass bootstrap_peers)

Simple: add a `Vec<SocketAddr>` param, send a unicast probe to each at startup.

- [ ] **Step 1: Add bootstrap_peers parameter to event_loop::run()**

Update the signature:

```rust
pub async fn run(
    mut runtime: NodeRuntime<MemoryBookStore>,
    startup_actions: Vec<RuntimeAction>,
    listen_addr: SocketAddr,
    mdns_addr: Option<[u8; 16]>,
    mdns_stale_timeout: Duration,
    tunnel_config: Option<TunnelConfig>,
    bootstrap_peers: Vec<SocketAddr>,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
```

- [ ] **Step 2: Send unicast probe to each bootstrap peer after startup actions**

After the startup actions loop and before the select loop, add:

```rust
    // ── Bootstrap peer probes (one-shot unicast) ────────────────────────────
    // Add each bootstrap peer to the PeerTable so they receive unicast traffic
    // alongside broadcast. This is the simplest approach: no special packet,
    // just treat them as known peers. The peer will hear our broadcasts and
    // Reticulum announces on the first timer tick. Adding to PeerTable also
    // means they get passive liveness tracking for free.
    for peer_addr in &bootstrap_peers {
        tracing::info!(peer = %peer_addr, "adding bootstrap peer");
        // Use a zeroed reticulum_addr — we don't know it yet.
        // The peer's actual addr will be learned from their Reticulum announce.
        // The zeroed addr won't match our our_addr (unless our_addr is also zeroed,
        // which only happens with --no-mdns, and even then [0;16] is the PeerTable's
        // our_addr so the self-filter catches it). Use [0x01; 16] as a placeholder.
        peer_table.add_peer(*peer_addr, [0x01; 16], 0);
    }
```

- [ ] **Step 3: Update the event_loop::run() call in main.rs**

Pass `bootstrap_peers` as the last argument:

```rust
            crate::event_loop::run(
                rt,
                startup_actions,
                listen_addr,
                if no_mdns { None } else { Some(our_addr_bytes) },
                std::time::Duration::from_secs(mdns_stale_timeout),
                tunnel_config,
                bootstrap_peers,
            )
```

- [ ] **Step 4: Run tests**

Run: `cargo test -p harmony-node -v`
Expected: all tests pass

- [ ] **Step 5: Commit**

```bash
git add crates/harmony-node/src/event_loop.rs crates/harmony-node/src/main.rs
git commit -m "feat(harmony-node): bootstrap peer probes from config [peers]

Send one-shot unicast probe to each bootstrap peer at startup.
Seeds peer discovery for cross-subnet nodes that mDNS can't reach.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: Tunnel peer reconnect from config [[tunnels]]

**Files:**
- Modify: `crates/harmony-node/src/event_loop.rs` (ConfigTunnelPeers, reconnect logic)
- Modify: `crates/harmony-node/src/main.rs` (pass tunnel_entries)

Persistent tunnel peers with exponential backoff reconnection.

- [ ] **Step 1: Add ConfigTunnelPeers struct to event_loop.rs**

```rust
/// Tracks tunnel peers from the config file for persistent reconnection.
/// Fields are populated now but actual reconnection is deferred to Bead harmony-h6k.
#[allow(dead_code)]
struct ConfigTunnelPeers {
    peers: Vec<ConfigTunnelPeer>,
}

#[allow(dead_code)]
struct ConfigTunnelPeer {
    node_id: String,
    name: Option<String>,
    interface_name: String,
    next_retry: Option<tokio::time::Instant>,
    backoff: std::time::Duration,
    connected: bool,
}

const INITIAL_BACKOFF: std::time::Duration = std::time::Duration::from_secs(1);
const MAX_BACKOFF: std::time::Duration = std::time::Duration::from_secs(60);
```

Implement:
- `ConfigTunnelPeers::new(entries: Vec<TunnelEntry>) -> Self` — creates peers with initial state
- `ConfigTunnelPeers::mark_connected(interface_name: &str)` — reset backoff on success
- `ConfigTunnelPeers::mark_disconnected(interface_name: &str)` — schedule retry with backoff
- `ConfigTunnelPeers::due_reconnects() -> Vec<&ConfigTunnelPeer>` — return peers past their retry time

- [ ] **Step 2: Add tunnel_entries parameter to event_loop::run()**

```rust
pub async fn run(
    ...,
    bootstrap_peers: Vec<SocketAddr>,
    tunnel_entries: Vec<crate::config::TunnelEntry>,
) -> ...
```

- [ ] **Step 3: Initialize ConfigTunnelPeers and initiate connections**

After iroh endpoint setup, create `ConfigTunnelPeers` and initiate initial connections:

```rust
    let mut config_tunnels = ConfigTunnelPeers::new(tunnel_entries);
    // Initial connection attempts happen on the first timer tick
    // (not inline here, to keep startup fast and let the event loop handle errors).
```

- [ ] **Step 4: Add reconnect check to timer tick (Arm 2)**

In the timer tick handler, after mDNS stale eviction and after `runtime.tick()` actions are dispatched, add:

```rust
                // Check for tunnel reconnects
                for peer in config_tunnels.due_reconnects() {
                    tracing::info!(
                        node_id = %peer.node_id,
                        name = ?peer.name,
                        "reconnecting config tunnel peer"
                    );
                    // TODO: Initiate outbound tunnel connection
                    // This requires the contact store (Bead #3) to look up
                    // the remote peer's PqIdentity for ML-KEM handshake.
                    // For now, log the intent. The actual InitiateTunnel
                    // wiring lands with Bead harmony-h6k.
                }
```

- [ ] **Step 5: Wire TunnelClosed to ConfigTunnelPeers**

In the tunnel bridge event handler (TunnelClosed arm), after removing from tunnel_senders, check if this was a config tunnel and schedule reconnect:

```rust
                    TunnelBridgeEvent::TunnelClosed { interface_name, reason, connection_id } => {
                        // ... existing handling ...
                        config_tunnels.mark_disconnected(&interface_name);
                    }
```

- [ ] **Step 6: Update main.rs to pass tunnel_entries**

Pass `tunnel_entries` to event_loop::run().

- [ ] **Step 7: Run tests**

Run: `cargo test -p harmony-node -v`
Expected: all tests pass

- [ ] **Step 8: Commit**

```bash
git add crates/harmony-node/src/event_loop.rs crates/harmony-node/src/main.rs
git commit -m "feat(harmony-node): persistent tunnel peers from config [[tunnels]]

ConfigTunnelPeers tracks config-sourced tunnels with exponential backoff
reconnection (1s-60s). Reconnect checks run on the timer tick.
Actual outbound connection wiring deferred to Bead harmony-h6k.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 6: Full verification

**Files:** None (verification only)

- [ ] **Step 1: Run full workspace tests**

Run: `cargo test --workspace`
Expected: all tests pass

- [ ] **Step 2: Run clippy**

Run: `cargo clippy --workspace`
Expected: no new warnings from our changes

- [ ] **Step 3: Run fmt check**

Run: `cargo fmt --all -- --check`
Expected: our files are formatted correctly

- [ ] **Step 4: Manual smoke test with config file**

Create a temp config file and verify it loads:

```bash
mkdir -p /tmp/harmony-test
cat > /tmp/harmony-test/node.toml << 'EOF'
cache_capacity = 512
listen_address = "127.0.0.1:4242"

[logging]
level = "harmony_node=debug"

[[peers]]
address = "192.168.1.10:4242"
EOF

# This should start (briefly) with the config values applied
timeout 2 cargo run -p harmony-node -- run --config /tmp/harmony-test/node.toml 2>&1 || true
# Check output shows "cache_capacity" = 512 and debug-level logs
```
