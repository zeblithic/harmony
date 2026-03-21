mod compute;
mod config;
mod discovery;
mod event_loop;
mod identity_file;
mod runtime;
// Zenoh-over-tunnel, initiator, and close paths are forward-looking (Bead #3).
#[allow(dead_code)]
mod tunnel_bridge;
#[allow(dead_code)]
mod tunnel_task;

use clap::{Parser, Subcommand};

type LogReloadHandle = tracing_subscriber::reload::Handle<
    tracing_subscriber::EnvFilter,
    tracing_subscriber::Registry,
>;

#[derive(Parser)]
#[command(name = "harmony", about = "Harmony decentralized network tools")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Identity management commands
    Identity {
        #[command(subcommand)]
        action: IdentityAction,
    },
    /// Start the Harmony node runtime
    Run {
        /// Path to config file (default: ~/.harmony/node.toml)
        #[arg(long, value_name = "PATH")]
        config: Option<std::path::PathBuf>,
        /// W-TinyLFU cache capacity (number of items)
        #[arg(long)]
        cache_capacity: Option<usize>,
        /// WASM compute fuel budget per tick
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
        /// Bloom filter broadcast interval in ticks
        #[arg(long)]
        filter_broadcast_ticks: Option<u32>,
        /// Bloom filter broadcast mutation threshold
        #[arg(long)]
        filter_mutation_threshold: Option<u32>,
        /// Path to the identity key file
        #[arg(long, value_name = "PATH")]
        identity_file: Option<std::path::PathBuf>,
        /// UDP listen address for Reticulum mesh packets
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
        /// iroh NodeId of a peer to connect to (outbound tunnel)
        /// TODO: Outbound initiator connections need the remote peer's PqIdentity
        /// for the ML-KEM encapsulation, which comes from the contact store (Bead #3).
        #[arg(long, value_name = "NODE_ID")]
        tunnel_peer: Option<String>,
    },
}

#[derive(Subcommand)]
enum IdentityAction {
    /// Generate a new identity keypair
    New,
    /// Display public info from a private key
    Show {
        /// Hex-encoded private key (128 hex chars)
        private_key: String,
    },
    /// Sign a message with a private key
    Sign {
        /// Hex-encoded private key (128 hex chars)
        private_key: String,
        /// Message to sign (UTF-8 string)
        message: String,
    },
    /// Verify a signature against a public key
    Verify {
        /// Hex-encoded public key (128 hex chars)
        public_key: String,
        /// Original message (UTF-8 string)
        message: String,
        /// Hex-encoded signature (128 hex chars)
        signature: String,
    },
}

// NodeRuntime is !Send — all tasks run on a single thread. Making the
// flavor explicit prevents silent behavior change if rt-multi-thread
// is ever added to the tokio feature set.
#[tokio::main(flavor = "current_thread")]
async fn main() {
    // Initialize structured logging. Output goes to stderr (procd captures
    // it for syslog on OpenWRT). Filter via RUST_LOG env var, default info.
    // Uses reload::Handle so [logging].level from config file can reconfigure
    // the filter after loading.
    let env_filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|e| {
            // Only warn if RUST_LOG is set but malformed — missing is the normal case.
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
    // Tip: use RUST_LOG=harmony_node=debug for harmony-only debug output.
    // Plain RUST_LOG=debug includes Zenoh's verbose internal traces.

    let cli = Cli::parse();
    if let Err(e) = run(cli, reload_handle).await {
        // Use eprintln for the top-level error — tracing may not flush to
        // a piped stderr before process::exit, and integration tests check
        // this output for specific error messages.
        eprintln!("Error: {e}");
        std::process::exit(1);
    }
}

fn decode_hex_key(
    hex_str: &str,
    expected_len: usize,
    label: &str,
) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let bytes = hex::decode(hex_str).map_err(|e| format!("invalid {label} hex: {e}"))?;
    if bytes.len() != expected_len {
        return Err(format!(
            "{label}: expected {expected_len} bytes, got {}",
            bytes.len()
        )
        .into());
    }
    Ok(bytes)
}

async fn run(cli: Cli, reload_handle: LogReloadHandle) -> Result<(), Box<dyn std::error::Error>> {
    match cli.command {
        Commands::Identity { action } => match action {
            IdentityAction::New => {
                let id = harmony_identity::PrivateIdentity::generate(&mut rand::rngs::OsRng);
                let pub_id = id.public_identity();
                println!("Address:     {}", hex::encode(pub_id.address_hash));
                println!("Public key:  {}", hex::encode(pub_id.to_public_bytes()));
                println!("Private key: {}", hex::encode(id.to_private_bytes()));
                Ok(())
            }
            IdentityAction::Show { private_key } => {
                let bytes = decode_hex_key(&private_key, 64, "private key")?;
                let id = harmony_identity::PrivateIdentity::from_private_bytes(&bytes)?;
                let pub_id = id.public_identity();
                println!("Address:     {}", hex::encode(pub_id.address_hash));
                println!("Public key:  {}", hex::encode(pub_id.to_public_bytes()));
                Ok(())
            }
            IdentityAction::Sign {
                private_key,
                message,
            } => {
                let bytes = decode_hex_key(&private_key, 64, "private key")?;
                let id = harmony_identity::PrivateIdentity::from_private_bytes(&bytes)?;
                let signature = id.sign(message.as_bytes());
                println!("Signature: {}", hex::encode(signature));
                Ok(())
            }
            IdentityAction::Verify {
                public_key,
                message,
                signature,
            } => {
                let pub_bytes = decode_hex_key(&public_key, 64, "public key")?;
                let sig_bytes = decode_hex_key(&signature, 64, "signature")?;
                let identity = harmony_identity::Identity::from_public_bytes(&pub_bytes)?;
                let sig_array: [u8; 64] = sig_bytes
                    .try_into()
                    .map_err(|_| "signature: expected exactly 64 bytes")?;
                match identity.verify(message.as_bytes(), &sig_array) {
                    Ok(()) => {
                        println!("Valid");
                        Ok(())
                    }
                    Err(_) => {
                        eprintln!("Error: invalid signature");
                        std::process::exit(1);
                    }
                }
            }
        },
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
            use crate::runtime::{NodeConfig, NodeRuntime};
            use harmony_compute::InstructionBudget;
            use harmony_content::book::MemoryBookStore;
            use harmony_content::storage_tier::{
                ContentPolicy, FilterBroadcastConfig, StorageBudget,
            };

            // ── Load config file ────────────────────────────────────────
            let (config_path, explicit) = crate::config::resolve_config_path(config.as_deref())?;
            if explicit && !config_path.exists() {
                return Err(format!("config file not found: {}", config_path.display()).into());
            }
            let config_file = crate::config::load(&config_path).map_err(|e| format!("{e}"))?;

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

            // ── Merge CLI > file > defaults ─────────────────────────────
            use crate::config::{resolve, resolve_bool};
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
                if let Some(ref peer) = tunnel_peer {
                    entries.push(crate::config::TunnelEntry {
                        node_id: peer.clone(),
                        name: None,
                    });
                }
                for entry in &entries {
                    hex::decode(&entry.node_id)
                        .map_err(|e| format!("invalid tunnel node_id '{}': {e}", entry.node_id))?;
                }
                entries
            };

            // Suppress unused-variable warnings until Tasks 4/5 consume these.
            let _ = &bootstrap_peers;
            let _ = &tunnel_entries;

            if cache_capacity == 0 {
                return Err("--cache-capacity must be > 0".into());
            }

            // Bloom filter bit count = expected_items * 14.378 at fp_rate=0.001.
            // Cap at 200M to stay well within u32::MAX bits (~2.88 billion).
            const MAX_CACHE_CAPACITY: usize = 200_000_000;
            if cache_capacity > MAX_CACHE_CAPACITY {
                return Err(format!(
                    "--cache-capacity {} exceeds maximum {} for Bloom filter sizing",
                    cache_capacity, MAX_CACHE_CAPACITY,
                )
                .into());
            }

            if filter_broadcast_ticks < 2 {
                return Err(
                    "--filter-broadcast-ticks must be >= 2: with interval=1 the timer \
                     fires every tick (counter reaches 1 immediately after increment)"
                        .into(),
                );
            }

            if encrypted_durable_announce && !encrypted_durable_persist {
                return Err(
                    "--encrypted-durable-announce requires --encrypted-durable-persist: \
                     content rejected by class admission will never reach the announce check"
                        .into(),
                );
            }

            if !no_mdns && mdns_stale_timeout == 0 {
                return Err(
                    "--mdns-stale-timeout must be > 0: a zero timeout evicts every peer \
                     on every timer tick, silently disabling unicast delivery"
                        .into(),
                );
            }

            let listen_addr: std::net::SocketAddr = listen_address
                .parse()
                .map_err(|e| format!("Invalid --listen-address: {e}"))?;

            // Load or generate node identity (after validation so bad flags exit fast).
            let id_path = crate::identity_file::resolve_path(identity_file.as_deref())?;
            let identity = crate::identity_file::load_or_generate(&id_path)?;
            let our_addr_bytes: [u8; 16] = identity.ed25519.public_identity().address_hash;
            let node_addr = hex::encode(our_addr_bytes);
            tracing::info!(address = %node_addr, path = %id_path.display(), "identity loaded");

            // Destructure to control per-field drop timing.
            let crate::identity_file::NodeIdentity { pq, ed25519 } = identity;

            // Build tunnel config if --relay-url was provided.
            // The PQ identity is wrapped in Arc because tunnel tasks need
            // references and PqPrivateIdentity is not Clone.
            let tunnel_config = if relay_url.is_some() || tunnel_peer.is_some() {
                if tunnel_peer.is_some() {
                    tracing::warn!("--tunnel-peer: outbound connections not yet wired (needs contact store, Bead #3)");
                }
                Some(crate::event_loop::TunnelConfig {
                    relay_url,
                    local_identity: std::sync::Arc::new(pq),
                })
            } else {
                drop(pq); // zeroize-on-drop
                None
            };
            // Ed25519 key material is no longer needed; zeroize-on-drop fires now.
            drop(ed25519);

            let content_policy = ContentPolicy {
                encrypted_durable_persist,
                encrypted_durable_announce,
                public_ephemeral_announce: !no_public_ephemeral_announce,
            };

            let config = NodeConfig {
                storage_budget: StorageBudget {
                    cache_capacity,
                    max_pinned_bytes: 100_000_000,
                },
                compute_budget: InstructionBudget {
                    fuel: compute_budget,
                },
                schedule: Default::default(),
                content_policy,
                filter_broadcast_config: FilterBroadcastConfig {
                    mutation_threshold: filter_mutation_threshold,
                    max_interval_ticks: filter_broadcast_ticks,
                    expected_items: cache_capacity as u32,
                    fp_rate: 0.001,
                },
                node_addr,
            };
            let (rt, startup_actions) = NodeRuntime::new(config, MemoryBookStore::new());

            tracing::info!(cache_capacity, compute_budget, %listen_addr, "harmony node starting");

            crate::event_loop::run(
                rt,
                startup_actions,
                listen_addr,
                if no_mdns { None } else { Some(our_addr_bytes) },
                std::time::Duration::from_secs(mdns_stale_timeout),
                tunnel_config,
            ).await
                .map_err(|e| -> Box<dyn std::error::Error> { e.to_string().into() })?;
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    /// Create a dummy `LogReloadHandle` for tests that call `run()`.
    fn dummy_reload_handle() -> LogReloadHandle {
        let filter = tracing_subscriber::EnvFilter::new("info");
        let (_layer, handle) = tracing_subscriber::reload::Layer::new(filter);
        handle
    }

    #[test]
    fn cli_parses_run_command() {
        let cli = Cli::try_parse_from(["harmony", "run"]).unwrap();
        assert!(matches!(cli.command, Commands::Run { .. }));
    }

    #[test]
    fn cli_parses_run_with_compute_budget() {
        let cli = Cli::try_parse_from(["harmony", "run", "--compute-budget", "50000"]).unwrap();
        if let Commands::Run { compute_budget, .. } = cli.command {
            assert_eq!(compute_budget, Some(50000));
        } else {
            panic!("expected Run command");
        }
    }

    #[test]
    fn cli_parses_run_with_cache_capacity() {
        let cli = Cli::try_parse_from(["harmony", "run", "--cache-capacity", "2048"]).unwrap();
        if let Commands::Run { cache_capacity, .. } = cli.command {
            assert_eq!(cache_capacity, Some(2048));
        } else {
            panic!("expected Run command");
        }
    }

    #[test]
    fn cli_parses_run_with_policy_flags() {
        let cli = Cli::try_parse_from([
            "harmony",
            "run",
            "--encrypted-durable-persist",
            "--encrypted-durable-announce",
            "--no-public-ephemeral-announce",
        ])
        .unwrap();
        if let Commands::Run {
            encrypted_durable_persist,
            encrypted_durable_announce,
            no_public_ephemeral_announce,
            ..
        } = cli.command
        {
            assert!(encrypted_durable_persist);
            assert!(encrypted_durable_announce);
            assert!(no_public_ephemeral_announce);
        } else {
            panic!("expected Run command");
        }
    }

    #[test]
    fn cli_policy_defaults() {
        let cli = Cli::try_parse_from(["harmony", "run"]).unwrap();
        if let Commands::Run {
            encrypted_durable_persist,
            encrypted_durable_announce,
            no_public_ephemeral_announce,
            ..
        } = cli.command
        {
            assert!(!encrypted_durable_persist);
            assert!(!encrypted_durable_announce);
            assert!(!no_public_ephemeral_announce);
        } else {
            panic!("expected Run command");
        }
    }

    #[tokio::test]
    async fn cli_rejects_announce_without_persist() {
        let cli = Cli::try_parse_from(["harmony", "run", "--encrypted-durable-announce"]).unwrap();
        let result = run(cli, dummy_reload_handle()).await;
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("--encrypted-durable-announce requires --encrypted-durable-persist"),
            "unexpected error: {msg}"
        );
    }

    #[tokio::test]
    async fn cli_rejects_zero_cache_capacity() {
        let cli = Cli::try_parse_from(["harmony", "run", "--cache-capacity", "0"]).unwrap();
        let result = run(cli, dummy_reload_handle()).await;
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("--cache-capacity must be > 0"),
            "unexpected error: {msg}"
        );
    }

    #[tokio::test]
    async fn cli_rejects_oversized_cache_capacity() {
        // 200_000_001 exceeds MAX_CACHE_CAPACITY (200M).
        let cli = Cli::try_parse_from(["harmony", "run", "--cache-capacity", "200000001"]).unwrap();
        let result = run(cli, dummy_reload_handle()).await;
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("exceeds maximum"), "unexpected error: {msg}");
    }

    #[test]
    fn cli_parses_run_with_filter_config() {
        let cli = Cli::try_parse_from([
            "harmony",
            "run",
            "--filter-broadcast-ticks",
            "60",
            "--filter-mutation-threshold",
            "200",
        ])
        .unwrap();
        if let Commands::Run {
            filter_broadcast_ticks,
            filter_mutation_threshold,
            ..
        } = cli.command
        {
            assert_eq!(filter_broadcast_ticks, Some(60));
            assert_eq!(filter_mutation_threshold, Some(200));
        } else {
            panic!("expected Run command");
        }
    }

    #[tokio::test]
    async fn cli_rejects_filter_broadcast_ticks_below_two() {
        let cli = Cli::try_parse_from(["harmony", "run", "--filter-broadcast-ticks", "1"]).unwrap();
        let result = run(cli, dummy_reload_handle()).await;
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("--filter-broadcast-ticks must be >= 2"),
            "unexpected error: {msg}"
        );
    }

    #[test]
    fn cli_parses_listen_address() {
        let cli = Cli::try_parse_from([
            "harmony",
            "run",
            "--listen-address",
            "127.0.0.1:9999",
        ])
        .unwrap();
        if let Commands::Run { listen_address, .. } = cli.command {
            assert_eq!(listen_address.as_deref(), Some("127.0.0.1:9999"));
        } else {
            panic!("expected Run command");
        }
    }

    #[test]
    fn cli_listen_address_default() {
        let cli = Cli::try_parse_from(["harmony", "run"]).unwrap();
        if let Commands::Run { listen_address, .. } = cli.command {
            assert!(listen_address.is_none());
        } else {
            panic!("expected Run command");
        }
    }

    #[tokio::test]
    async fn cli_rejects_invalid_listen_address() {
        let cli =
            Cli::try_parse_from(["harmony", "run", "--listen-address", "not-an-addr"]).unwrap();
        let result = run(cli, dummy_reload_handle()).await;
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("Invalid --listen-address"),
            "unexpected error: {msg}"
        );
    }

    #[test]
    fn cli_parses_no_mdns_flag() {
        let cli = Cli::try_parse_from(["harmony", "run", "--no-mdns"]).unwrap();
        if let Commands::Run { no_mdns, .. } = cli.command {
            assert!(no_mdns);
        } else {
            panic!("expected Run command");
        }
    }

    #[test]
    fn cli_no_mdns_default_false() {
        let cli = Cli::try_parse_from(["harmony", "run"]).unwrap();
        if let Commands::Run { no_mdns, .. } = cli.command {
            assert!(!no_mdns);
        } else {
            panic!("expected Run command");
        }
    }

    #[test]
    fn cli_parses_mdns_stale_timeout() {
        let cli = Cli::try_parse_from(["harmony", "run", "--mdns-stale-timeout", "120"]).unwrap();
        if let Commands::Run { mdns_stale_timeout, .. } = cli.command {
            assert_eq!(mdns_stale_timeout, Some(120));
        } else {
            panic!("expected Run command");
        }
    }

    #[test]
    fn cli_mdns_stale_timeout_default() {
        let cli = Cli::try_parse_from(["harmony", "run"]).unwrap();
        if let Commands::Run { mdns_stale_timeout, .. } = cli.command {
            assert!(mdns_stale_timeout.is_none());
        } else {
            panic!("expected Run command");
        }
    }

    #[tokio::test]
    async fn cli_rejects_zero_mdns_stale_timeout() {
        let cli =
            Cli::try_parse_from(["harmony", "run", "--mdns-stale-timeout", "0"]).unwrap();
        let result = run(cli, dummy_reload_handle()).await;
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("--mdns-stale-timeout must be > 0"),
            "unexpected error: {msg}"
        );
    }

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
}
