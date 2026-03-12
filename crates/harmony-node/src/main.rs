mod compute;
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

fn main() {
    let cli = Cli::parse();
    if let Err(e) = run(cli) {
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

fn run(cli: Cli) -> Result<(), Box<dyn std::error::Error>> {
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
                        eprintln!("Invalid");
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
        } => {
            use crate::runtime::{NodeConfig, NodeRuntime, RuntimeAction};
            use harmony_compute::InstructionBudget;
            use harmony_content::blob::MemoryBlobStore;
            use harmony_content::storage_tier::{ContentPolicy, FilterBroadcastConfig, StorageBudget};

            if encrypted_durable_announce && !encrypted_durable_persist {
                return Err(
                    "--encrypted-durable-announce requires --encrypted-durable-persist: \
                     content rejected by class admission will never reach the announce check"
                        .into(),
                );
            }

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
                filter_broadcast_config: FilterBroadcastConfig::default(),
            };
            let (rt, startup_actions) = NodeRuntime::new(config, MemoryBlobStore::new());

            println!("Harmony node runtime initialized");
            println!("  Cache capacity:   {cache_capacity} items");
            println!("  Compute budget:   {compute_budget} fuel/tick");
            println!("  Router queue:     {} pending", rt.router_queue_len());
            println!("  Storage queue:    {} pending", rt.storage_queue_len());
            println!("  Compute queue:    {} tracked", rt.compute_queue_len());
            println!("\nStartup actions:");
            for action in &startup_actions {
                match action {
                    RuntimeAction::DeclareQueryable { key_expr } => {
                        println!("  queryable: {key_expr}");
                    }
                    RuntimeAction::Subscribe { key_expr } => {
                        println!("  subscribe: {key_expr}");
                    }
                    _ => {}
                }
            }
            println!(
                "\n{} queryables, {} subscriptions declared",
                startup_actions
                    .iter()
                    .filter(|a| matches!(a, RuntimeAction::DeclareQueryable { .. }))
                    .count(),
                startup_actions
                    .iter()
                    .filter(|a| matches!(a, RuntimeAction::Subscribe { .. }))
                    .count(),
            );
            println!("\nNode ready. (Event loop requires async runtime — not yet wired.)");
            Ok(())
        }
    }
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

    #[test]
    fn cli_rejects_announce_without_persist() {
        let cli = Cli::try_parse_from([
            "harmony",
            "run",
            "--encrypted-durable-announce",
        ])
        .unwrap();
        let result = run(cli);
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("--encrypted-durable-announce requires --encrypted-durable-persist"),
            "unexpected error: {msg}"
        );
    }
}
