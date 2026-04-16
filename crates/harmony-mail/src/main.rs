use std::path::Path;

use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "harmony-mail", about = "Harmony Mail Gateway")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Initialize gateway (generate keys, create config, print DNS records)
    Init {
        #[arg(long)]
        domain: String,
        #[arg(long)]
        admin_email: String,
    },
    /// Start the mail gateway
    Run {
        #[arg(long, default_value = "/etc/harmony-mail/config.toml")]
        config: String,
    },
    /// User management
    User {
        #[command(subcommand)]
        action: UserAction,
    },
    /// Alias management
    Alias {
        #[command(subcommand)]
        action: AliasAction,
    },
    /// TLS certificate management
    Tls {
        #[arg(long)]
        acme: bool,
        #[arg(long, name = "dns-challenge")]
        dns_challenge: bool,
    },
    /// Domain warming mode
    Warm {
        #[arg(long)]
        target: u32,
        #[arg(long)]
        days: u32,
    },
}

#[derive(Subcommand)]
enum UserAction {
    Add {
        /// IMAP login username
        #[arg(long)]
        name: String,
        /// Hex-encoded 16-byte harmony address (32 hex chars)
        #[arg(long)]
        identity: String,
        /// Path to config file (for store_path)
        #[arg(long, default_value = "/etc/harmony-mail/config.toml")]
        config: String,
    },
}

#[derive(Subcommand)]
enum AliasAction {
    Add {
        /// Alias name
        alias: String,
        /// Target (identity hex or name)
        target: String,
    },
}

async fn build_remote_delivery(
    config: &harmony_mail::config::Config,
) -> Result<harmony_mail::RemoteDeliveryContext, Box<dyn std::error::Error>> {
    use std::sync::Arc;
    use std::time::Duration;

    let identity_bytes = std::fs::read(&config.gateway.identity_key)?;
    let gateway_identity = Arc::new(
        harmony_identity::PrivateIdentity::from_private_bytes(&identity_bytes)?
    );

    let zenoh_session = Arc::new(
        zenoh::open(zenoh::Config::default())
            .await
            .map_err(|e| -> Box<dyn std::error::Error> { Box::new(std::io::Error::other(e.to_string())) })?,
    );

    let recipient_resolver: Arc<dyn harmony_mail::remote_delivery::RecipientResolver> = Arc::new(
        harmony_mail::remote_delivery::ZenohRecipientResolver::new(
            Arc::clone(&zenoh_session),
            Duration::from_secs(5),
        ),
    );

    let dns: Arc<dyn harmony_mail_discovery::dns::DnsClient> = Arc::new(
        harmony_mail_discovery::dns::HickoryDnsClient::from_system(Duration::from_secs(5)),
    );
    let http: Arc<dyn harmony_mail_discovery::http::HttpClient> = Arc::new(
        harmony_mail_discovery::http::ReqwestHttpClient::new(
            Duration::from_secs(5),
            Duration::from_secs(10),
            1_000_000,
        )?,
    );
    let time: Arc<dyn harmony_mail_discovery::cache::TimeSource> =
        Arc::new(harmony_mail_discovery::cache::SystemTimeSource);

    let email_resolver = Arc::new(
        harmony_mail_discovery::resolver::DefaultEmailResolver::new(
            dns, http, time,
            harmony_mail_discovery::resolver::ResolverConfig::default(),
        ),
    );
    email_resolver.spawn_background_refresh().await;

    Ok(harmony_mail::RemoteDeliveryContext {
        gateway_identity,
        recipient_resolver,
        email_resolver,
    })
}

#[tokio::main(flavor = "multi_thread", worker_threads = 1)]
async fn main() {
    tracing_subscriber::fmt::init();

    let cli = Cli::parse();
    match cli.command {
        Commands::Init {
            domain,
            admin_email,
        } => {
            println!("Initializing gateway for {domain} (admin: {admin_email})");
            println!("Not yet implemented");
        }
        Commands::Run {
            config: config_path,
        } => {
            let config = harmony_mail::config::Config::from_file(Path::new(&config_path))
                .unwrap_or_else(|e| {
                    eprintln!("Failed to load config from {config_path}: {e}");
                    std::process::exit(1);
                });

            let remote_delivery = match build_remote_delivery(&config).await {
                Ok(ctx) => {
                    tracing::info!("Remote delivery enabled (gateway + discovery)");
                    Some(ctx)
                }
                Err(e) => {
                    tracing::warn!("Remote delivery disabled: {e}");
                    None
                }
            };

            if let Err(e) = harmony_mail::server::run(config, remote_delivery).await {
                eprintln!("Server error: {e}");
                std::process::exit(1);
            }
        }
        Commands::User { action } => match action {
            UserAction::Add {
                name,
                identity,
                config: config_path,
            } => {
                // Validate identity hex
                let addr_bytes = match hex::decode(&identity) {
                    Ok(bytes) if bytes.len() == harmony_mail::message::ADDRESS_HASH_LEN => {
                        let mut arr = [0u8; harmony_mail::message::ADDRESS_HASH_LEN];
                        arr.copy_from_slice(&bytes);
                        arr
                    }
                    Ok(bytes) => {
                        eprintln!(
                            "Error: identity must be {} hex chars ({} bytes), got {} bytes",
                            harmony_mail::message::ADDRESS_HASH_LEN * 2,
                            harmony_mail::message::ADDRESS_HASH_LEN,
                            bytes.len()
                        );
                        std::process::exit(1);
                    }
                    Err(e) => {
                        eprintln!("Error: invalid hex in --identity: {e}");
                        std::process::exit(1);
                    }
                };

                // Prompt for password
                let password = rpassword::prompt_password_stderr("Password: ").unwrap_or_else(|e| {
                    eprintln!("Error reading password: {e}");
                    std::process::exit(1);
                });
                let confirm = rpassword::prompt_password_stderr("Confirm password: ").unwrap_or_else(|e| {
                    eprintln!("Error reading password: {e}");
                    std::process::exit(1);
                });
                if password != confirm {
                    eprintln!("Error: passwords do not match");
                    std::process::exit(1);
                }
                if password.is_empty() {
                    eprintln!("Error: password cannot be empty");
                    std::process::exit(1);
                }

                // Load config and open store
                let cfg = harmony_mail::config::Config::from_file(Path::new(&config_path))
                    .unwrap_or_else(|e| {
                        eprintln!("Failed to load config from {config_path}: {e}");
                        std::process::exit(1);
                    });
                let store = harmony_mail::imap_store::ImapStore::open(Path::new(&cfg.imap.store_path))
                    .unwrap_or_else(|e| {
                        eprintln!("Failed to open IMAP store: {e}");
                        std::process::exit(1);
                    });

                match store.create_user(&name, &password, &addr_bytes) {
                    Ok(()) => println!("User '{name}' created successfully"),
                    Err(harmony_mail::imap_store::StoreError::UserExists(_)) => {
                        eprintln!("Error: user '{name}' already exists");
                        std::process::exit(1);
                    }
                    Err(e) => {
                        eprintln!("Error creating user: {e}");
                        std::process::exit(1);
                    }
                }
            }
        },
        Commands::Alias { action } => match action {
            AliasAction::Add { alias, target } => {
                println!("Adding alias {alias} -> {target}");
                println!("Not yet implemented");
            }
        },
        Commands::Tls {
            acme,
            dns_challenge,
        } => {
            println!("TLS setup (acme: {acme}, dns_challenge: {dns_challenge})");
            println!("Not yet implemented");
        }
        Commands::Warm { target, days } => {
            println!("Starting domain warming: target {target}/day over {days} days");
            println!("Not yet implemented");
        }
    }
}
