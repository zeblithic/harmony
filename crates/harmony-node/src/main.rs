mod compute;
mod config;
mod did_web_gateway;
mod discovery;
#[allow(dead_code)]
pub(crate) mod disk_io;
mod event_loop;
mod identity_file;
#[allow(dead_code)]
mod inference;
mod runtime;
// Zenoh-over-tunnel, initiator, and close paths are forward-looking (Bead #3).
#[allow(dead_code)]
mod tunnel_bridge;
#[allow(dead_code)]
mod tunnel_task;

use clap::{Parser, Subcommand};

type LogReloadHandle =
    tracing_subscriber::reload::Handle<tracing_subscriber::EnvFilter, tracing_subscriber::Registry>;

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
    /// Memo attestation commands
    Memo {
        #[command(subcommand)]
        action: MemoAction,
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
        #[arg(long, num_args = 0..=1, default_missing_value = "true")]
        encrypted_durable_persist: Option<bool>,
        /// Announce encrypted durable (10) content on Zenoh
        #[arg(long, num_args = 0..=1, default_missing_value = "true")]
        encrypted_durable_announce: Option<bool>,
        /// Disable announcing public ephemeral (01) content on Zenoh
        #[arg(long, num_args = 0..=1, default_missing_value = "true")]
        no_public_ephemeral_announce: Option<bool>,
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
        #[arg(long, num_args = 0..=1, default_missing_value = "true")]
        no_mdns: Option<bool>,
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
        /// Add a tunnel peer: <identity_hash_hex>:<node_id_hex>[@relay_url]
        ///
        /// Can be specified multiple times. Each peer is added as a contact
        /// with tunnel addressing and peering enabled at Normal priority.
        #[arg(long, value_name = "PEER_SPEC")]
        add_tunnel_peer: Vec<String>,
    },
    /// Compute ContentId for a file or stdin
    Cid {
        /// Path to file (reads stdin if omitted)
        #[arg(long, value_name = "PATH")]
        file: Option<std::path::PathBuf>,
        /// Print CID metadata (type, size, chunks) to stderr
        #[arg(long)]
        verbose: bool,
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

#[derive(Subcommand)]
enum MemoAction {
    /// Sign a memo attesting that input CID produces output CID
    Sign {
        /// Input CID (hex, 64 chars)
        #[arg(long)]
        input: String,
        /// Output CID (hex, 64 chars)
        #[arg(long)]
        output: String,
        /// Path to identity key file
        #[arg(long, value_name = "PATH")]
        identity_file: Option<std::path::PathBuf>,
        /// Expiry in seconds from now (default: 365 days)
        #[arg(long, default_value_t = 31_536_000)]
        expires_in: u64,
    },
    /// List known memos for an input CID
    List {
        /// Input CID (hex, 64 chars)
        #[arg(long)]
        input: String,
    },
    /// Show memo verification status for an input CID
    Verify {
        /// Input CID (hex, 64 chars)
        #[arg(long)]
        input: String,
    },
}

// Zenoh requires a multi-thread Tokio runtime. We use worker_threads = 1
// to give Zenoh a multi-thread scheduler while keeping only one OS thread
// for worker tasks. NodeRuntime's !Send types are safe here because they
// live exclusively in the block_on future (main thread), never in a
// tokio::spawn task — the compiler enforces this via Send bounds on spawn.
#[tokio::main(flavor = "multi_thread", worker_threads = 1)]
async fn main() {
    // Initialize structured logging. Output goes to stderr (procd captures
    // it for syslog on OpenWRT). Filter via RUST_LOG env var, default info.
    // Uses reload::Handle so [logging].level from config file can reconfigure
    // the filter after loading.
    let env_filter = tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|e| {
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
        Commands::Memo { action } => match action {
            MemoAction::Sign {
                input,
                output,
                identity_file,
                expires_in,
            } => {
                let input_bytes: [u8; 32] = decode_hex_key(&input, 32, "input CID")?
                    .try_into()
                    .map_err(|_| "input CID: internal length mismatch")?;
                let output_bytes: [u8; 32] = decode_hex_key(&output, 32, "output CID")?
                    .try_into()
                    .map_err(|_| "output CID: internal length mismatch")?;

                let input_cid = harmony_content::ContentId::from_bytes(input_bytes);
                let output_cid = harmony_content::ContentId::from_bytes(output_bytes);

                let id_path = crate::identity_file::resolve_path(identity_file.as_deref())?;
                let identity = crate::identity_file::load_or_generate(&id_path)?;

                let now = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_secs())
                    .map_err(|_| "system clock is before Unix epoch")?;
                let expires_at = now
                    .checked_add(expires_in)
                    .ok_or("--expires-in value causes timestamp overflow")?;

                let memo = harmony_memo::create::create_memo(
                    input_cid,
                    output_cid,
                    &identity.pq,
                    &mut rand::rngs::OsRng,
                    now,
                    expires_at,
                )
                .map_err(|e| format!("failed to create memo: {e}"))?;

                let memo_bytes =
                    harmony_memo::serialize(&memo).map_err(|e| format!("serialize: {e}"))?;

                let signer_hex = hex::encode(identity.pq.public_identity().address_hash);
                // Use canonical lowercase hex from parsed bytes, not raw user input.
                // Upper/lower case differences would produce different Zenoh keys.
                let input_hex = hex::encode(input_cid.to_bytes());
                let output_hex = hex::encode(output_cid.to_bytes());
                let key_expr =
                    harmony_zenoh::namespace::memo::sign_key(&input_hex, &output_hex, &signer_hex);

                println!("Key:    {key_expr}");
                println!("Memo:   {}", hex::encode(&memo_bytes));
                println!("Signer: {signer_hex}");

                // Zeroize identity on drop.
                drop(identity);
                Ok(())
            }
            MemoAction::List { input: _ } => {
                eprintln!("MemoStore not yet connected to runtime");
                Ok(())
            }
            MemoAction::Verify { input: _ } => {
                eprintln!("MemoStore not yet connected to runtime");
                Ok(())
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
            add_tunnel_peer,
        } => {
            use crate::runtime::{NodeConfig, NodeRuntime};
            use harmony_compute::InstructionBudget;
            use harmony_content::book::MemoryBookStore;
            use harmony_content::storage_tier::{
                ContentPolicy, FilterBroadcastConfig, StorageBudget,
            };

            // ── Load config file ────────────────────────────────────────
            let (config_path, explicit) = crate::config::resolve_config_path(config.as_deref())?;
            let config_file =
                crate::config::load(&config_path, explicit).map_err(|e| format!("{e}"))?;

            // Apply config file log level if RUST_LOG is not set.
            if std::env::var("RUST_LOG").is_err() {
                if let Some(ref logging) = config_file.logging {
                    if let Some(ref level) = logging.level {
                        match tracing_subscriber::EnvFilter::try_new(level) {
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
            use crate::config::resolve;
            let cache_capacity = resolve(cache_capacity, config_file.cache_capacity, 1024);
            let compute_budget = resolve(compute_budget, config_file.compute_budget, 100_000);
            let filter_broadcast_ticks = resolve(
                filter_broadcast_ticks,
                config_file.filter_broadcast_ticks,
                30,
            );
            let filter_mutation_threshold = resolve(
                filter_mutation_threshold,
                config_file.filter_mutation_threshold,
                100,
            );
            let mdns_stale_timeout =
                resolve(mdns_stale_timeout, config_file.mdns_stale_timeout, 60);
            let listen_address = resolve(
                listen_address,
                config_file.listen_address,
                "0.0.0.0:4242".to_string(),
            );
            let identity_file = identity_file.or(config_file.identity_file);
            let relay_url = relay_url.or(config_file.relay_url);
            let no_mdns = resolve(no_mdns, config_file.no_mdns, false);
            let encrypted_durable_persist = resolve(
                encrypted_durable_persist,
                config_file.encrypted_durable_persist,
                false,
            );
            let encrypted_durable_announce = resolve(
                encrypted_durable_announce,
                config_file.encrypted_durable_announce,
                false,
            );
            let no_public_ephemeral_announce = resolve(
                no_public_ephemeral_announce,
                config_file.no_public_ephemeral_announce,
                false,
            );
            let did_web_cache_ttl = config_file.did_web_cache_ttl.unwrap_or(300);
            let rawlink_interface = config_file.rawlink_interface.clone();
            let archivist_config = config_file.archivist;

            // ── Parse config-only sections ───────────────────────────────
            let bootstrap_peers: Vec<std::net::SocketAddr> = config_file
                .peers
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
                    let decoded = hex::decode(&entry.node_id)
                        .map_err(|e| format!("invalid tunnel node_id '{}': {e}", entry.node_id))?;
                    if decoded.len() != 32 {
                        return Err(format!(
                            "invalid tunnel node_id '{}': expected 32 bytes (64 hex chars), got {}",
                            entry.node_id,
                            decoded.len()
                        )
                        .into());
                    }
                }
                entries
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

            // Extract PQ identity hash and DSA public key before pq is moved into TunnelConfig.
            // Used by the discover queryable to verify self-issued Discovery tokens.
            let pq_pub = pq.public_identity();
            let local_pq_identity_hash = pq_pub.address_hash;
            let local_dsa_pubkey = pq_pub.verifying_key.as_bytes();

            // Build tunnel config if --relay-url was provided.
            // The PQ identity is wrapped in Arc because tunnel tasks need
            // references and PqPrivateIdentity is not Clone.
            let tunnel_config = if relay_url.is_some()
                || tunnel_peer.is_some()
                || !tunnel_entries.is_empty()
            {
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
            // Extract Ed25519 private bytes for Reticulum announcing destination,
            // then drop the key (zeroize-on-drop fires).
            let reticulum_identity_bytes = Some(ed25519.to_private_bytes());
            drop(ed25519);

            let content_policy = ContentPolicy {
                encrypted_durable_persist,
                encrypted_durable_announce,
                public_ephemeral_announce: !no_public_ephemeral_announce,
            };

            // Scan disk for persisted CIDs+sizes (spawn_blocking to avoid blocking tokio).
            let disk_entries = match &config_file.data_dir {
                Some(dir) => {
                    let dir = dir.clone();
                    tokio::task::spawn_blocking(move || {
                        let entries = crate::disk_io::scan_books(&dir);
                        tracing::info!(
                            count = entries.len(),
                            total_bytes = entries.iter().map(|(_, s)| s).sum::<u64>(),
                            path = %dir.display(),
                            "loaded book entries from disk"
                        );
                        entries
                    })
                    .await
                    .unwrap_or_default()
                }
                None => Vec::new(),
            };

            let disk_quota = match &config_file.disk_quota {
                Some(s) => {
                    let bytes = crate::config::parse_byte_size(s)
                        .map_err(|e| format!("invalid disk_quota '{s}': {e}"))?;
                    if config_file.data_dir.is_none() {
                        tracing::warn!(
                            raw = %s,
                            "disk_quota is set but data_dir is not configured — quota will be ignored"
                        );
                    } else {
                        tracing::info!(quota_bytes = bytes, raw = %s, "disk quota configured");
                    }
                    Some(bytes)
                }
                None => {
                    if config_file.data_dir.is_some() {
                        tracing::info!(
                            "disk persistence enabled without quota — disk usage is unbounded"
                        );
                    }
                    None
                }
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
                local_identity_hash: our_addr_bytes,
                local_pq_identity_hash,
                local_dsa_pubkey,
                reticulum_identity_bytes,
                inference_gguf_cid: config_file
                    .inference_model_gguf_cid
                    .as_deref()
                    .and_then(|s| {
                        hex::decode(s)
                            .ok()
                            .and_then(|v| <[u8; 32]>::try_from(v).ok())
                            .or_else(|| {
                                tracing::warn!("inference_model_gguf_cid is not a valid 32-byte hex string; inference disabled");
                                None
                            })
                    }),
                inference_tokenizer_cid: config_file
                    .inference_model_tokenizer_cid
                    .as_deref()
                    .and_then(|s| {
                        hex::decode(s)
                            .ok()
                            .and_then(|v| <[u8; 32]>::try_from(v).ok())
                            .or_else(|| {
                                tracing::warn!("inference_model_tokenizer_cid is not a valid 32-byte hex string; inference disabled");
                                None
                            })
                    }),
                engram_manifest_cid: config_file
                    .engram_manifest_cid
                    .as_deref()
                    .and_then(|s| {
                        hex::decode(s)
                            .ok()
                            .and_then(|v| <[u8; 32]>::try_from(v).ok())
                            .or_else(|| {
                                tracing::warn!("engram_manifest_cid is not a valid 32-byte hex string; Engram disabled");
                                None
                            })
                    }),
                disk_enabled: config_file.data_dir.is_some(),
                disk_entries,
                disk_quota,
                s3_enabled: {
                    #[cfg(feature = "archivist")]
                    { archivist_config.is_some() }
                    #[cfg(not(feature = "archivist"))]
                    { false }
                },
            };
            let (mut rt, startup_actions) = NodeRuntime::new(config, MemoryBookStore::new());

            // Register tunnel peers from CLI.
            for spec in &add_tunnel_peer {
                let TunnelPeerSpec {
                    identity_hash,
                    node_id,
                    relay_url,
                } = parse_tunnel_peer_spec(spec)?;
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
                    replication: None,
                };
                rt.contact_store_mut()
                    .add(contact)
                    .map_err(|e| format!("failed to add tunnel peer: {e}"))?;
                rt.push_event(crate::runtime::RuntimeEvent::ContactChanged { identity_hash });
                tracing::info!(
                    identity = %hex::encode(identity_hash),
                    node_id = %hex::encode(&node_id[..8]),
                    "tunnel peer registered"
                );
            }

            tracing::info!(cache_capacity, compute_budget, %listen_addr, "harmony node starting");

            crate::event_loop::run(
                rt,
                startup_actions,
                listen_addr,
                if no_mdns { None } else { Some(our_addr_bytes) },
                std::time::Duration::from_secs(mdns_stale_timeout),
                tunnel_config,
                bootstrap_peers,
                tunnel_entries,
                did_web_cache_ttl,
                rawlink_interface,
                archivist_config,
                config_file.data_dir.clone(),
            )
            .await
            .map_err(|e| -> Box<dyn std::error::Error> { e.to_string().into() })?;
            Ok(())
        }
        Commands::Cid { file, verbose } => {
            use harmony_content::book::MemoryBookStore;
            use harmony_content::chunker::{chunk_all, ChunkerConfig};
            use harmony_content::cid::{CidType, ContentFlags, ContentId, MAX_PAYLOAD_SIZE};
            use harmony_content::dag;

            let data = match file {
                Some(path) => std::fs::read(&path)
                    .map_err(|e| format!("Failed to read {}: {e}", path.display()))?,
                None => {
                    use std::io::Read;
                    let mut buf = Vec::new();
                    std::io::stdin()
                        .read_to_end(&mut buf)
                        .map_err(|e| format!("Failed to read stdin: {e}"))?;
                    buf
                }
            };

            if data.is_empty() {
                return Err("Empty input — cannot compute CID".into());
            }

            let cid = if data.len() <= MAX_PAYLOAD_SIZE {
                ContentId::for_book(&data, ContentFlags::default())
                    .map_err(|e| format!("CID computation failed: {e}"))?
            } else {
                // Large files: full DAG ingest (FastCDC chunking → Merkle DAG).
                // The MemoryBookStore is ephemeral — chunks are not persisted
                // to a CAS. The CID is deterministic and can be used as a
                // content fingerprint; reassembly requires a separate ingest
                // into a persistent store.
                let mut store = MemoryBookStore::new();
                dag::ingest(&data, &ChunkerConfig::DEFAULT, &mut store)
                    .map_err(|e| format!("DAG ingest failed: {e}"))?
            };

            println!("{}", hex::encode(cid.to_bytes()));

            if verbose {
                let input_size = data.len();
                match cid.cid_type() {
                    CidType::Book => {
                        eprintln!("Type:   Book (single chunk)");
                        eprintln!("Size:   {} bytes", input_size);
                        eprintln!("Chunks: 1");
                    }
                    CidType::Bundle(depth) => {
                        let chunks = chunk_all(&data, &ChunkerConfig::DEFAULT)
                            .map(|r| r.len())
                            .unwrap_or(0);
                        eprintln!("Type:   Bundle (Merkle DAG, depth {})", depth);
                        eprintln!("Size:   {} bytes ({:.1} MB)", input_size, input_size as f64 / (1024.0 * 1024.0));
                        eprintln!("Chunks: {}", chunks);
                    }
                    other => {
                        eprintln!("Type:   {:?}", other);
                        eprintln!("Size:   {} bytes", input_size);
                    }
                }
                eprintln!("Flags:  {:?}", cid.flags());
            }

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
fn parse_tunnel_peer_spec(spec: &str) -> Result<TunnelPeerSpec, Box<dyn std::error::Error>> {
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
            assert_eq!(encrypted_durable_persist, Some(true));
            assert_eq!(encrypted_durable_announce, Some(true));
            assert_eq!(no_public_ephemeral_announce, Some(true));
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
            assert_eq!(encrypted_durable_persist, None);
            assert_eq!(encrypted_durable_announce, None);
            assert_eq!(no_public_ephemeral_announce, None);
        } else {
            panic!("expected Run command");
        }
    }

    #[tokio::test]
    async fn cli_rejects_announce_without_persist() {
        let cli = Cli::try_parse_from([
            "harmony",
            "run",
            "--config",
            "/dev/null",
            "--encrypted-durable-announce",
        ])
        .unwrap();
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
        let cli = Cli::try_parse_from([
            "harmony",
            "run",
            "--config",
            "/dev/null",
            "--cache-capacity",
            "0",
        ])
        .unwrap();
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
        let cli = Cli::try_parse_from([
            "harmony",
            "run",
            "--config",
            "/dev/null",
            "--cache-capacity",
            "200000001",
        ])
        .unwrap();
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
        let cli = Cli::try_parse_from([
            "harmony",
            "run",
            "--config",
            "/dev/null",
            "--filter-broadcast-ticks",
            "1",
        ])
        .unwrap();
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
        let cli =
            Cli::try_parse_from(["harmony", "run", "--listen-address", "127.0.0.1:9999"]).unwrap();
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
        let cli = Cli::try_parse_from([
            "harmony",
            "run",
            "--config",
            "/dev/null",
            "--listen-address",
            "not-an-addr",
        ])
        .unwrap();
        let result = run(cli, dummy_reload_handle()).await;
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
        assert_eq!(
            parsed.relay_url,
            Some("https://relay.example.com".to_string())
        );
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
        let cli = Cli::try_parse_from(["harmony", "run", "--add-tunnel-peer", &spec]).unwrap();
        if let Commands::Run {
            add_tunnel_peer, ..
        } = cli.command
        {
            assert_eq!(add_tunnel_peer.len(), 1);
            assert_eq!(add_tunnel_peer[0], spec);
        } else {
            panic!("expected Run command");
        }
    }

    #[test]
    fn cli_parses_no_mdns_flag() {
        let cli = Cli::try_parse_from(["harmony", "run", "--no-mdns"]).unwrap();
        if let Commands::Run { no_mdns, .. } = cli.command {
            assert_eq!(no_mdns, Some(true));
        } else {
            panic!("expected Run command");
        }
    }

    #[test]
    fn cli_parses_multiple_tunnel_peers() {
        let spec1 = format!("{}:{}", "aa".repeat(16), "bb".repeat(32));
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
        if let Commands::Run {
            add_tunnel_peer, ..
        } = cli.command
        {
            assert_eq!(add_tunnel_peer.len(), 2);
        } else {
            panic!("expected Run command");
        }
    }

    #[test]
    fn cli_no_mdns_default_false() {
        let cli = Cli::try_parse_from(["harmony", "run"]).unwrap();
        if let Commands::Run { no_mdns, .. } = cli.command {
            assert_eq!(no_mdns, None);
        } else {
            panic!("expected Run command");
        }
    }

    #[test]
    fn cli_parses_mdns_stale_timeout() {
        let cli = Cli::try_parse_from(["harmony", "run", "--mdns-stale-timeout", "120"]).unwrap();
        if let Commands::Run {
            mdns_stale_timeout, ..
        } = cli.command
        {
            assert_eq!(mdns_stale_timeout, Some(120));
        } else {
            panic!("expected Run command");
        }
    }

    #[test]
    fn cli_mdns_stale_timeout_default() {
        let cli = Cli::try_parse_from(["harmony", "run"]).unwrap();
        if let Commands::Run {
            mdns_stale_timeout, ..
        } = cli.command
        {
            assert!(mdns_stale_timeout.is_none());
        } else {
            panic!("expected Run command");
        }
    }

    #[tokio::test]
    async fn cli_rejects_zero_mdns_stale_timeout() {
        let cli = Cli::try_parse_from([
            "harmony",
            "run",
            "--config",
            "/dev/null",
            "--mdns-stale-timeout",
            "0",
        ])
        .unwrap();
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

    #[test]
    fn cli_parses_memo_sign() {
        let input_hex = "aa".repeat(32);
        let output_hex = "bb".repeat(32);
        let cli = Cli::try_parse_from([
            "harmony",
            "memo",
            "sign",
            "--input",
            &input_hex,
            "--output",
            &output_hex,
        ])
        .unwrap();
        assert!(matches!(cli.command, Commands::Memo { .. }));
    }

    #[test]
    fn cli_parses_memo_list() {
        let cli =
            Cli::try_parse_from(["harmony", "memo", "list", "--input", &"cc".repeat(32)]).unwrap();
        assert!(matches!(cli.command, Commands::Memo { .. }));
    }

    #[test]
    fn cli_parses_memo_verify() {
        let cli = Cli::try_parse_from(["harmony", "memo", "verify", "--input", &"dd".repeat(32)])
            .unwrap();
        assert!(matches!(cli.command, Commands::Memo { .. }));
    }

    #[test]
    fn cli_parses_cid_with_file() {
        let cli = Cli::try_parse_from(["harmony", "cid", "--file", "/tmp/test.bin"]).unwrap();
        if let Commands::Cid { file, .. } = cli.command {
            assert_eq!(file, Some(std::path::PathBuf::from("/tmp/test.bin")));
        } else {
            panic!("expected Cid command");
        }
    }

    #[test]
    fn cli_parses_cid_stdin_mode() {
        let cli = Cli::try_parse_from(["harmony", "cid"]).unwrap();
        if let Commands::Cid { file, .. } = cli.command {
            assert!(file.is_none());
        } else {
            panic!("expected Cid command");
        }
    }
}
