# TOML Config File Support for harmony-node

**Bead:** harmony-r4o
**Date:** 2026-03-21
**Status:** Draft

## Problem

harmony-node is configured entirely via CLI flags. This works on OpenWRT (where the
init script translates UCI to CLI flags) but is cumbersome for non-OpenWRT deployments.
Some configuration doesn't map well to CLI flags: lists of bootstrap peers, named tunnel
entries, per-interface settings. A config file provides a more ergonomic and extensible
configuration surface.

## Solution

Load configuration from a TOML file at `$HOME/.harmony/node.toml`. All fields are
optional — missing values use defaults. CLI flags override file values. A `--config`
flag overrides the default path. The file is opt-in: if the default path doesn't exist,
proceed silently with defaults. If an explicit `--config` path doesn't exist, error.

New config-only sections (`[peers]`, `[tunnels]`, `[logging]`) provide functionality
not available via CLI flags.

## Config File Format

```toml
# $HOME/.harmony/node.toml

# ── Node settings (all optional, defaults shown) ─────────────

listen_address = "0.0.0.0:4242"
identity_file = "/home/user/.harmony/identity.key"  # absolute path required (no ~ expansion)
cache_capacity = 1024
compute_budget = 100000
filter_broadcast_ticks = 30
filter_mutation_threshold = 100

# Content policy
encrypted_durable_persist = false
encrypted_durable_announce = false
no_public_ephemeral_announce = false

# mDNS peer discovery
no_mdns = false
mdns_stale_timeout = 60

# Tunnel relay (global — all tunnels use this relay for NAT traversal)
relay_url = "https://i.q8.fyi"

# ── Logging ───────────────────────────────────────────────────

[logging]
level = "info"                    # RUST_LOG equivalent

# ── Bootstrap peers (UDP, startup-only) ───────────────────────

[[peers]]
address = "192.168.1.10:4242"

[[peers]]
address = "10.0.0.5:4242"

# ── Tunnel peers (iroh, persistent with backoff) ──────────────

[[tunnels]]
node_id = "abc123..."            # iroh NodeId hex
name = "gateway"                  # optional human label

[[tunnels]]
node_id = "def456..."
name = "backup-relay"
```

### Format Decisions

- Top-level keys match CLI flag names (underscores, not hyphens)
- `[[peers]]` and `[[tunnels]]` use TOML array-of-tables for multiple entries
- `[logging]` replaces `RUST_LOG` env var for config-file users
- `[interfaces]` reserved for future use (not implemented in v1)
- Unknown keys are rejected (`deny_unknown_fields`) to catch typos

## Architecture

### New file: `config.rs`

**Serde structs:**

```rust
#[derive(Deserialize, Default)]
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

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LoggingConfig {
    pub level: Option<String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PeerEntry {
    pub address: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TunnelEntry {
    pub node_id: String,
    pub name: Option<String>,
}
```

**Error type:**

```rust
pub enum ConfigError {
    Io { path: PathBuf, source: std::io::Error },
    Parse { path: PathBuf, message: String },
}
```

`ConfigError` implements `Display` with the path and error details so the user sees
which file and what went wrong. `Io` covers read failures; `Parse` covers both
malformed TOML and `deny_unknown_fields` rejections (both come from `toml::de::Error`).

**Loading:**

```rust
pub fn load(path: &Path) -> Result<ConfigFile, ConfigError>
```

- File doesn't exist → `Ok(ConfigFile::default())`
- File exists, valid TOML → `Ok(parsed)`
- File exists, malformed → `Err(ConfigError::Parse { path, message })`
- Unknown keys → `Err(ConfigError::Parse { ... })` (via `deny_unknown_fields`)

**Path resolution:**

```rust
pub fn resolve_config_path(cli_override: Option<&Path>) -> Result<(PathBuf, bool), String>
```

Returns `(path, explicit)` where `explicit` is true when `--config` was provided.
If explicit and the file doesn't exist, the caller errors. If implicit (default path)
and the file doesn't exist, the caller proceeds with defaults.

### Merge logic

```rust
fn resolve<T>(cli: Option<T>, file: Option<T>, default: T) -> T {
    cli.or(file).unwrap_or(default)
}
```

For boolean CLI flags (`--no-mdns`, `--encrypted-durable-persist`, etc.): clap
represents these as `bool` (not `Option<bool>`). `false` means "not passed" and
`true` means "passed." The merge treats `false` as "not set" — the file value
wins only when the CLI flag was not passed. This works because all boolean flags
default to `false` in clap, and `true` is the only meaningful CLI override.

### CLI changes (`main.rs`)

New flag on the `Run` variant (not the top-level `Cli` struct — config only
applies to `harmony run`, not `harmony identity`):

```rust
/// Path to config file (default: ~/.harmony/node.toml)
#[arg(long, value_name = "PATH")]
config: Option<PathBuf>,
```

**CLI field type changes:** To support three-way merge (CLI > file > default), fields
that currently use `default_value_t` must change to `Option<T>`. Otherwise clap always
provides a value and the file can never win. The affected fields:

| Field | Before | After |
|-------|--------|-------|
| `cache_capacity` | `usize` with `default_value_t = 1024` | `Option<usize>` |
| `compute_budget` | `u64` with `default_value_t = 100_000` | `Option<u64>` |
| `filter_broadcast_ticks` | `u32` with `default_value_t = 30` | `Option<u32>` |
| `filter_mutation_threshold` | `u32` with `default_value_t = 100` | `Option<u32>` |
| `mdns_stale_timeout` | `u64` with `default_value_t = 60` | `Option<u64>` |
| `listen_address` | `String` with `default_value = "0.0.0.0:4242"` | `Option<String>` |

Defaults move to the merge step: `resolve(cli, file, hardcoded_default)`.

**Existing `--tunnel-peer` flag:** Deprecated in favor of `[[tunnels]]` in the config
file. For backward compatibility, if `--tunnel-peer` is provided, it is treated as a
single-entry tunnel list (merged with any `[[tunnels]]` from the config file). A future
release can remove the flag entirely.

**Startup sequence:**

1. Initialize tracing with default `info` filter (so early errors are formatted)
2. Resolve config file path (`--config` or `$HOME/.harmony/node.toml`)
3. Load config file (silent default if missing at default path, error if missing at
   explicit path, error if malformed)
4. If `[logging].level` is set and `RUST_LOG` is not in env, reconfigure the tracing
   filter using `tracing_subscriber::reload` handle
5. Merge CLI flags with file values (CLI wins)
6. Parse and validate `[peers]` addresses (`String` → `SocketAddr`) — error if any
   address is unparseable. Parse and validate `[tunnels]` node IDs — error if invalid
   hex. This happens in `main.rs` before passing to the event loop.
7. Run existing validation pipeline on merged values
8. Load identity, build NodeConfig, start event loop

**Logging integration:** Tracing initializes at startup with a default `info` filter
and a `reload::Handle`. After config loading, if `[logging].level` is set and
`RUST_LOG` is not in env, the handle reconfigures the filter without reinitializing
the subscriber. This ensures early errors (config parse failures, etc.) are always
formatted through tracing, while still supporting config-driven log levels.

### Event loop changes

**Bootstrap UDP peers** (`Vec<SocketAddr>` from `[peers]`):

After startup actions execute, send a single unicast probe to each bootstrap peer
address. This seeds Reticulum's path table so the peer knows we exist. No retry, no
state tracking — mDNS and Reticulum announces handle ongoing discovery.

**Tunnel peers** (`Vec<TunnelEntry>` from `[tunnels]`):

Each entry becomes an outbound tunnel connection. The existing tunnel infrastructure
handles connection lifecycle. New addition: a `ConfigTunnelPeers` struct that tracks
which tunnels came from the config file. On `TunnelClosed` for a config-sourced
tunnel, schedule a reconnect with exponential backoff (1s → 2s → 4s → ... capped
at 60s). Non-config tunnels (accepted inbound) don't get reconnected.

Reconnect scheduling: runs in the timer tick handler (arm 2), after `runtime.tick()`
returns actions and after mDNS stale eviction. On each 250ms tick, iterate
`ConfigTunnelPeers` and initiate reconnect for any tunnel past its `next_retry`.
The struct stores `next_retry: Instant` and `backoff: Duration` per tunnel. No new
select! arm is needed — the timer tick is the natural place for periodic housekeeping.

### What is NOT in scope

- No hot-reload. Restart the node to apply config changes.
- No `[interfaces]` implementation. Reserved in the format, not parsed.
- Bootstrap peers don't go into PeerTable. They're one-shot UDP sends.
- No change to OpenWRT UCI config. The init script continues to translate UCI to
  CLI flags. The config file is for non-OpenWRT users.

## Error Handling

| Scenario | Behavior |
|----------|----------|
| Default path missing | Silent, proceed with defaults |
| Explicit `--config` path missing | Hard error, node won't start |
| Malformed TOML | Hard error with parse location |
| Unknown keys | Hard error (deny_unknown_fields) |
| Invalid values (e.g., `cache_capacity = -1`) | Caught by existing validation after merge |
| Unparseable peer address | Error at startup |
| Invalid tunnel node_id hex | Error at startup |

## Testing

**Unit tests for `config.rs`:**
- `load` with valid file, missing file, malformed TOML, unknown keys
- Merge logic: CLI overrides file, file overrides default, missing file = all defaults
- Section parsing: `[peers]`, `[tunnels]`, `[logging]`
- Path resolution: explicit vs default, missing explicit errors

**CLI tests:**
- `--config` flag parsing
- `--config /nonexistent.toml` produces error

**Integration test:**
- Write temp TOML file, verify merged values via NodeConfig

## File Changes

| File | Change |
|------|--------|
| `crates/harmony-node/Cargo.toml` | Add `serde = { workspace = true, features = ["derive"] }`, `toml.workspace = true` |
| `crates/harmony-node/src/config.rs` | New: ConfigFile struct, load, resolve_config_path, merge helpers |
| `crates/harmony-node/src/main.rs` | `--config` flag, load+merge before validation, pass bootstrap_peers and tunnel_entries to event_loop |
| `crates/harmony-node/src/event_loop.rs` | `bootstrap_peers: Vec<SocketAddr>` param, `ConfigTunnelPeers` reconnect logic |
