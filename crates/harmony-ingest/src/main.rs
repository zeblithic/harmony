mod config;

use clap::{Parser, Subcommand};
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "harmony-ingest", about = "Ingest data into Harmony CAS")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Ingest an Engram embedding table from a safetensors file.
    Engram {
        /// Path to TOML config file.
        #[arg(long)]
        config: PathBuf,

        /// Path to safetensors file.
        #[arg(long)]
        input: PathBuf,

        /// S3 bucket name.
        #[arg(long)]
        bucket: Option<String>,

        /// S3 key prefix (default: "harmony/").
        #[arg(long, default_value = "harmony/")]
        prefix: String,

        /// AWS region (default: from environment).
        #[arg(long)]
        region: Option<String>,

        /// Local directory for book cache.
        #[arg(long)]
        local_dir: Option<PathBuf>,

        /// Resume from shard index N (reads journal).
        #[arg(long)]
        resume_from: Option<u64>,
    },
}

#[tokio::main(flavor = "current_thread")]
async fn main() {
    tracing_subscriber::fmt::init();
    let cli = Cli::parse();

    match cli.command {
        Commands::Engram {
            config,
            input,
            bucket,
            prefix,
            region,
            local_dir,
            resume_from,
        } => {
            if bucket.is_none() && local_dir.is_none() {
                eprintln!("error: at least one of --bucket or --local-dir is required");
                std::process::exit(1);
            }
            let _config = match config::EngramConfig::load(&config) {
                Ok(c) => c,
                Err(e) => {
                    eprintln!("error: {e}");
                    std::process::exit(1);
                }
            };
            tracing::info!("config loaded, input={}", input.display());
            // Stages 2-6 implemented in later tasks.
            let _ = (prefix, region, resume_from, bucket, local_dir);
        }
    }
}
