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
        Commands::Run { cache_capacity } => {
            use crate::runtime::{NodeConfig, NodeRuntime, RuntimeAction};
            use harmony_content::blob::MemoryBlobStore;
            use harmony_content::storage_tier::StorageBudget;

            let config = NodeConfig {
                storage_budget: StorageBudget {
                    cache_capacity,
                    max_pinned_bytes: 100_000_000,
                },
            };
            let (rt, startup_actions) = NodeRuntime::new(config, MemoryBlobStore::new());

            println!("Harmony node runtime initialized");
            println!("  Cache capacity: {cache_capacity} items");
            println!("  Router queue:   {} pending", rt.router_queue_len());
            println!("  Storage queue:  {} pending", rt.storage_queue_len());
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
    fn cli_parses_run_with_cache_capacity() {
        let cli = Cli::try_parse_from(["harmony", "run", "--cache-capacity", "2048"]).unwrap();
        if let Commands::Run { cache_capacity, .. } = cli.command {
            assert_eq!(cache_capacity, 2048);
        } else {
            panic!("expected Run command");
        }
    }
}
