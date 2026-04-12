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
        #[arg(long)]
        name: String,
        #[arg(long)]
        namespace: String,
        #[arg(long)]
        identity: String,
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
            if let Err(e) = harmony_mail::server::run(config).await {
                eprintln!("Server error: {e}");
                std::process::exit(1);
            }
        }
        Commands::User { action } => match action {
            UserAction::Add {
                name,
                namespace,
                identity,
            } => {
                println!("Adding user {name}_{namespace} with identity {identity}");
                println!("Not yet implemented");
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
