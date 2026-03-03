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
                println!("Address:    {}", hex::encode(pub_id.address_hash));
                println!("Public key: {}", hex::encode(pub_id.to_public_bytes()));
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
                let sig_array: [u8; 64] = sig_bytes.try_into().unwrap();
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
    }
}
