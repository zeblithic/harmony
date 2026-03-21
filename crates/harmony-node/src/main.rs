mod compute;
mod event_loop;
mod identity_file;
mod runtime;

use clap::{Parser, Subcommand};

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
        /// W-TinyLFU cache capacity (number of items)
        #[arg(long, default_value_t = 1024)]
        cache_capacity: usize,
        /// WASM compute fuel budget per tick
        #[arg(long, default_value_t = 100_000)]
        compute_budget: u64,
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
        #[arg(long, default_value_t = 30)]
        filter_broadcast_ticks: u32,
        /// Bloom filter broadcast mutation threshold
        #[arg(long, default_value_t = 100)]
        filter_mutation_threshold: u32,
        /// Path to the identity key file
        #[arg(long, value_name = "PATH")]
        identity_file: Option<std::path::PathBuf>,
        /// UDP listen address for Reticulum mesh packets
        #[arg(long, default_value = "0.0.0.0:4242")]
        listen_address: String,
        /// Add a tunnel peer: <identity_hash_hex>:<node_id_hex>[@relay_url]
        ///
        /// Can be specified multiple times. Each peer is added as a contact
        /// with tunnel addressing and peering enabled at Normal priority.
        #[arg(long, value_name = "PEER_SPEC")]
        add_tunnel_peer: Vec<String>,
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
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|e| {
                    // Only warn if RUST_LOG is set but malformed — missing is the normal case.
                    if std::env::var("RUST_LOG").is_ok() {
                        eprintln!("Warning: invalid RUST_LOG directive ({e}), defaulting to info");
                    }
                    tracing_subscriber::EnvFilter::new("info")
                }),
        )
        .with_target(false)
        .with_ansi(false)
        .without_time()
        .with_writer(std::io::stderr)
        .init();
    // Tip: use RUST_LOG=harmony_node=debug for harmony-only debug output.
    // Plain RUST_LOG=debug includes Zenoh's verbose internal traces.

    let cli = Cli::parse();
    if let Err(e) = run(cli).await {
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

async fn run(cli: Cli) -> Result<(), Box<dyn std::error::Error>> {
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
            cache_capacity,
            compute_budget,
            encrypted_durable_persist,
            encrypted_durable_announce,
            no_public_ephemeral_announce,
            filter_broadcast_ticks,
            filter_mutation_threshold,
            identity_file,
            listen_address,
            add_tunnel_peer,
        } => {
            use crate::runtime::{NodeConfig, NodeRuntime};
            use harmony_compute::InstructionBudget;
            use harmony_content::book::MemoryBookStore;
            use harmony_content::storage_tier::{
                ContentPolicy, FilterBroadcastConfig, StorageBudget,
            };

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

            let listen_addr: std::net::SocketAddr = listen_address
                .parse()
                .map_err(|e| format!("Invalid --listen-address: {e}"))?;

            // Load or generate node identity (after validation so bad flags exit fast).
            let id_path = crate::identity_file::resolve_path(identity_file.as_deref())?;
            let identity = crate::identity_file::load_or_generate(&id_path)?;
            let node_addr = hex::encode(identity.ed25519.public_identity().address_hash);
            tracing::info!(address = %node_addr, path = %id_path.display(), "identity loaded");
            drop(identity); // key material no longer needed; zeroize-on-drop fires now

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
            let (mut rt, startup_actions) = NodeRuntime::new(config, MemoryBookStore::new());

            // Register tunnel peers from CLI.
            for spec in &add_tunnel_peer {
                let TunnelPeerSpec { identity_hash, node_id, relay_url } = parse_tunnel_peer_spec(spec)?;
                let contact = harmony_contacts::Contact {
                    identity_hash,
                    display_name: None,
                    peering: harmony_contacts::PeeringPolicy {
                        enabled: true,
                        priority: harmony_contacts::PeeringPriority::Normal,
                    },
                    added_at: std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .map(|d| d.as_secs())
                        .unwrap_or(0),
                    last_seen: None,
                    notes: None,
                    addresses: vec![harmony_contacts::ContactAddress::Tunnel {
                        node_id,
                        relay_url,
                        direct_addrs: vec![],
                    }],
                };
                rt.contact_store_mut()
                    .add(contact)
                    .map_err(|e| format!("failed to add tunnel peer: {e}"))?;
                rt.push_event(crate::runtime::RuntimeEvent::ContactChanged {
                    identity_hash,
                });
                tracing::info!(
                    identity = %hex::encode(identity_hash),
                    node_id = %hex::encode(&node_id[..8]),
                    "tunnel peer registered"
                );
            }

            tracing::info!(cache_capacity, compute_budget, %listen_addr, "harmony node starting");

            crate::event_loop::run(rt, startup_actions, listen_addr).await
                .map_err(|e| -> Box<dyn std::error::Error> { e.to_string().into() })?;
            Ok(())
        }
    }
}

/// Parsed tunnel peer specification.
#[derive(Debug)]
struct TunnelPeerSpec {
    identity_hash: [u8; 16],
    node_id: [u8; 32],
    relay_url: Option<String>,
}

/// Parse a tunnel peer spec: `<identity_hash_hex>:<node_id_hex>[@relay_url]`
fn parse_tunnel_peer_spec(
    spec: &str,
) -> Result<TunnelPeerSpec, Box<dyn std::error::Error>> {
    let (id_and_node, relay_url) = match spec.split_once('@') {
        Some((left, url)) if !url.is_empty() => (left, Some(url.to_string())),
        Some((left, _)) => (left, None), // trailing '@' with no URL → treat as no relay
        None => (spec, None),
    };
    let (id_hex, node_hex) = id_and_node.split_once(':').ok_or_else(|| {
        format!(
            "invalid --add-tunnel-peer format: expected <identity_hash>:<node_id>[@relay_url], got '{spec}'"
        )
    })?;
    let id_bytes = hex::decode(id_hex)
        .map_err(|e| format!("invalid identity_hash hex in --add-tunnel-peer: {e}"))?;
    let node_bytes = hex::decode(node_hex)
        .map_err(|e| format!("invalid node_id hex in --add-tunnel-peer: {e}"))?;
    let identity_hash: [u8; 16] = id_bytes.try_into().map_err(|v: Vec<u8>| {
        format!(
            "identity_hash must be 16 bytes (32 hex chars), got {} bytes",
            v.len()
        )
    })?;
    let node_id: [u8; 32] = node_bytes.try_into().map_err(|v: Vec<u8>| {
        format!(
            "node_id must be 32 bytes (64 hex chars), got {} bytes",
            v.len()
        )
    })?;
    Ok(TunnelPeerSpec {
        identity_hash,
        node_id,
        relay_url,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    #[test]
    fn cli_parses_run_command() {
        let cli = Cli::try_parse_from(["harmony", "run"]).unwrap();
        assert!(matches!(cli.command, Commands::Run { .. }));
    }

    #[test]
    fn cli_parses_run_with_compute_budget() {
        let cli = Cli::try_parse_from(["harmony", "run", "--compute-budget", "50000"]).unwrap();
        if let Commands::Run { compute_budget, .. } = cli.command {
            assert_eq!(compute_budget, 50000);
        } else {
            panic!("expected Run command");
        }
    }

    #[test]
    fn cli_parses_run_with_cache_capacity() {
        let cli = Cli::try_parse_from(["harmony", "run", "--cache-capacity", "2048"]).unwrap();
        if let Commands::Run { cache_capacity, .. } = cli.command {
            assert_eq!(cache_capacity, 2048);
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
        let result = run(cli).await;
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
        let result = run(cli).await;
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
        let result = run(cli).await;
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
            assert_eq!(filter_broadcast_ticks, 60);
            assert_eq!(filter_mutation_threshold, 200);
        } else {
            panic!("expected Run command");
        }
    }

    #[tokio::test]
    async fn cli_rejects_filter_broadcast_ticks_below_two() {
        let cli = Cli::try_parse_from(["harmony", "run", "--filter-broadcast-ticks", "1"]).unwrap();
        let result = run(cli).await;
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
            assert_eq!(listen_address, "127.0.0.1:9999");
        } else {
            panic!("expected Run command");
        }
    }

    #[test]
    fn cli_listen_address_default() {
        let cli = Cli::try_parse_from(["harmony", "run"]).unwrap();
        if let Commands::Run { listen_address, .. } = cli.command {
            assert_eq!(listen_address, "0.0.0.0:4242");
        } else {
            panic!("expected Run command");
        }
    }

    #[tokio::test]
    async fn cli_rejects_invalid_listen_address() {
        let cli =
            Cli::try_parse_from(["harmony", "run", "--listen-address", "not-an-addr"]).unwrap();
        let result = run(cli).await;
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("Invalid --listen-address"),
            "unexpected error: {msg}"
        );
    }

    #[test]
    fn parse_tunnel_peer_spec_full() {
        let id_hex = "aa".repeat(16); // 32 hex chars = 16 bytes
        let node_hex = "bb".repeat(32); // 64 hex chars = 32 bytes
        let spec = format!("{id_hex}:{node_hex}@https://relay.example.com");
        let parsed = parse_tunnel_peer_spec(&spec).unwrap();
        assert_eq!(parsed.identity_hash, [0xAA; 16]);
        assert_eq!(parsed.node_id, [0xBB; 32]);
        assert_eq!(parsed.relay_url, Some("https://relay.example.com".to_string()));
    }

    #[test]
    fn parse_tunnel_peer_spec_no_relay() {
        let id_hex = "cc".repeat(16);
        let node_hex = "dd".repeat(32);
        let spec = format!("{id_hex}:{node_hex}");
        let parsed = parse_tunnel_peer_spec(&spec).unwrap();
        assert_eq!(parsed.identity_hash, [0xCC; 16]);
        assert_eq!(parsed.node_id, [0xDD; 32]);
        assert_eq!(parsed.relay_url, None);
    }

    #[test]
    fn parse_tunnel_peer_spec_bad_format() {
        let result = parse_tunnel_peer_spec("no-colon-here");
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("invalid --add-tunnel-peer format"), "{msg}");
    }

    #[test]
    fn parse_tunnel_peer_spec_bad_hex() {
        let result = parse_tunnel_peer_spec("gggg:hhhh");
        assert!(result.is_err());
    }

    #[test]
    fn parse_tunnel_peer_spec_wrong_id_length() {
        let node_hex = "bb".repeat(32);
        let spec = format!("aabb:{node_hex}");
        let result = parse_tunnel_peer_spec(&spec);
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("identity_hash must be 16 bytes"), "{msg}");
    }

    #[test]
    fn parse_tunnel_peer_spec_wrong_node_length() {
        let id_hex = "aa".repeat(16);
        let spec = format!("{id_hex}:bbcc");
        let result = parse_tunnel_peer_spec(&spec);
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("node_id must be 32 bytes"), "{msg}");
    }

    #[test]
    fn cli_parses_add_tunnel_peer() {
        let id_hex = "aa".repeat(16);
        let node_hex = "bb".repeat(32);
        let spec = format!("{id_hex}:{node_hex}@https://relay.example.com");
        let cli = Cli::try_parse_from([
            "harmony",
            "run",
            "--add-tunnel-peer",
            &spec,
        ])
        .unwrap();
        if let Commands::Run { add_tunnel_peer, .. } = cli.command {
            assert_eq!(add_tunnel_peer.len(), 1);
            assert_eq!(add_tunnel_peer[0], spec);
        } else {
            panic!("expected Run command");
        }
    }

    #[test]
    fn cli_parses_multiple_tunnel_peers() {
        let spec1 = format!("{}:{}",  "aa".repeat(16), "bb".repeat(32));
        let spec2 = format!("{}:{}", "cc".repeat(16), "dd".repeat(32));
        let cli = Cli::try_parse_from([
            "harmony",
            "run",
            "--add-tunnel-peer",
            &spec1,
            "--add-tunnel-peer",
            &spec2,
        ])
        .unwrap();
        if let Commands::Run { add_tunnel_peer, .. } = cli.command {
            assert_eq!(add_tunnel_peer.len(), 2);
        } else {
            panic!("expected Run command");
        }
    }
}
