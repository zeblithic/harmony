mod config;
mod journal;
mod manifest;
mod shard;
mod storage;

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

async fn run_engram(
    config_path: PathBuf,
    input: PathBuf,
    bucket: Option<String>,
    prefix: String,
    region: Option<String>,
    local_dir: Option<PathBuf>,
    resume_from: Option<u64>,
) -> Result<(), String> {
    // Stage 1: Load config.
    let cfg = config::EngramConfig::load(&config_path)?;
    tracing::info!(version = %cfg.version, tensor = %cfg.tensor, "config loaded");

    // Stage 2: Open safetensors.
    let file_bytes =
        std::fs::read(&input).map_err(|e| format!("failed to read {}: {e}", input.display()))?;
    let tensors = safetensors::SafeTensors::deserialize(&file_bytes)
        .map_err(|e| format!("invalid safetensors: {e}"))?;
    let tensor = tensors
        .tensor(&cfg.tensor)
        .map_err(|e| format!("tensor '{}' not found: {e}", cfg.tensor))?;

    let shape = tensor.shape();
    if shape.len() != 2 {
        return Err(format!("expected 2D tensor, got {}D", shape.len()));
    }
    let total_entries = shape[0] as u64;
    let embedding_dim = shape[1];

    // Determine source dtype for conversion.
    let src_dtype = match tensor.dtype() {
        safetensors::Dtype::F16 => shard::SourceDtype::F16,
        safetensors::Dtype::F32 => {
            tracing::info!("f32 tensor — will convert to f16 during sharding");
            shard::SourceDtype::F32
        }
        safetensors::Dtype::BF16 => {
            tracing::info!("bf16 tensor — will convert to f16 during sharding");
            shard::SourceDtype::BF16
        }
        other => return Err(format!("unsupported tensor dtype: {other:?}")),
    };

    let tensor_bytes = tensor.data();
    let n_shards = shard::num_shards(total_entries, cfg.shard_size);
    tracing::info!(total_entries, embedding_dim, n_shards, "tensor validated");

    // Set up S3 if configured.
    let s3 = if let Some(ref bucket_name) = bucket {
        Some(
            harmony_s3::S3Library::new(bucket_name.clone(), prefix.clone(), region.clone())
                .await
                .map_err(|e| format!("S3 init: {e}"))?,
        )
    } else {
        None
    };

    // Stage 3: Shard and upload.
    let journal_path = input.with_extension("journal");
    let mut journal = journal::CidJournal::open(&journal_path)?;

    // Handle resume.
    let start_shard = resume_from.unwrap_or(0);
    let mut shard_cids: Vec<[u8; 32]> = if start_shard > 0 {
        let existing = journal::CidJournal::read_all(&journal_path)?;
        if existing.len() as u64 != start_shard {
            return Err(format!(
                "journal has {} entries but --resume-from={start_shard}",
                existing.len()
            ));
        }
        tracing::info!(start_shard, "resuming from journal");
        existing
    } else {
        Vec::with_capacity(n_shards as usize)
    };

    for i in start_shard..n_shards {
        let s = shard::slice_shard(tensor_bytes, i, cfg.shard_size, embedding_dim, src_dtype);

        if let Some(ref s3_lib) = s3 {
            storage::upload_s3_book(s3_lib, &s.cid, s.data.clone()).await?;
        }
        if let Some(ref dir) = local_dir {
            storage::write_local_book(dir, &s.cid, &s.data)?;
        }

        journal.append(&s.cid)?;
        shard_cids.push(s.cid.to_bytes());

        if (i + 1) % 10_000 == 0 || i + 1 == n_shards {
            tracing::info!(shard = i + 1, total = n_shards, "progress");
        }
    }

    // Stage 4: Build manifest.
    let header = manifest::make_header(
        cfg.version.clone(),
        embedding_dim as u32,
        cfg.num_heads(),
        cfg.hash_seeds.clone(),
        total_entries,
        cfg.shard_size,
        n_shards,
    );
    let (root_cid, dag_store) = manifest::build_manifest(&header, &shard_cids)?;
    tracing::info!(root_cid = %hex::encode(root_cid.to_bytes()), "manifest built");

    // Stage 5: Upload manifest DAG books.
    let mut dag_book_count = 0u64;
    for (cid, data) in dag_store.into_books() {
        if let Some(ref s3_lib) = s3 {
            storage::upload_s3_book(s3_lib, &cid, data.clone()).await?;
        }
        if let Some(ref dir) = local_dir {
            storage::write_local_book(dir, &cid, &data)?;
        }
        dag_book_count += 1;
    }
    tracing::info!(dag_book_count, "manifest DAG uploaded");

    // Stage 6: Output.
    println!("Engram table ingested successfully.");
    println!("  Version:       {}", cfg.version);
    println!("  Total entries: {total_entries}");
    println!("  Embedding dim: {embedding_dim}");
    println!("  Num shards:    {n_shards}");
    println!("  Manifest CID:  {}", hex::encode(root_cid.to_bytes()));

    Ok(())
}

#[tokio::main(flavor = "current_thread")]
async fn main() {
    tracing_subscriber::fmt::init();
    let cli = Cli::parse();

    let result = match cli.command {
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
            run_engram(
                config,
                input,
                bucket,
                prefix,
                region,
                local_dir,
                resume_from,
            )
            .await
        }
    };

    if let Err(e) = result {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}
