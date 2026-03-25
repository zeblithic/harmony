# Engram Checkpoint Ingestion Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a CLI tool (`harmony-ingest engram`) that reads DeepSeek Engram tables from safetensors files and ingests them as sharded CAS books into S3 and/or a local directory.

**Architecture:** Binary crate with no library target. Reads TOML config + safetensors file, slices the embedding table into 64KB shards, computes CIDs, uploads to S3/local, builds a Merkle DAG manifest, and outputs the root CID. A binary journal file enables crash-safe resume.

**Tech Stack:** Rust, safetensors (mmap tensor access), harmony-engram (ManifestHeader), harmony-content (ContentId, dag::ingest, MemoryBookStore), harmony-s3 (S3Library), clap (CLI), tokio (async), half (f16)

**Spec:** `docs/superpowers/specs/2026-03-24-engram-ingest-design.md`

---

## File Structure

### New Files

| File | Responsibility |
|------|---------------|
| `crates/harmony-ingest/Cargo.toml` | Binary crate manifest |
| `crates/harmony-ingest/src/main.rs` | CLI entry point, clap parsing, orchestration |
| `crates/harmony-ingest/src/config.rs` | TOML config parsing + validation |
| `crates/harmony-ingest/src/shard.rs` | Tensor slicing into shards, CID computation, last-shard padding |
| `crates/harmony-ingest/src/journal.rs` | Binary CID journal (append, read, truncation recovery) |
| `crates/harmony-ingest/src/storage.rs` | S3 upload (with retry) + local dir write (two-level prefix) |
| `crates/harmony-ingest/src/manifest.rs` | Build ManifestHeader, assemble manifest bytes, DAG ingestion |

### Modified Files

| File | Change |
|------|--------|
| `Cargo.toml` (workspace root) | Add `harmony-ingest` member + `safetensors` workspace dep |
| `crates/harmony-content/src/book.rs` | Add `into_books()` method to `MemoryBookStore` |

---

### Task 1: Crate Scaffold + Config Parsing

**Files:**
- Create: `crates/harmony-ingest/Cargo.toml`
- Create: `crates/harmony-ingest/src/main.rs`
- Create: `crates/harmony-ingest/src/config.rs`
- Modify: `Cargo.toml` (workspace root)

- [ ] **Step 1: Add workspace dependencies and member**

In workspace root `Cargo.toml`:
- Add `"crates/harmony-ingest"` to `members` list (alphabetically between `harmony-identity` and `harmony-jain`)
- Add to `[workspace.dependencies]` utilities section:
  ```toml
  safetensors = "0.4"
  ```
No need to add `harmony-ingest` to `[workspace.dependencies]` — it's a pure binary crate with no lib target, so nothing depends on it.

- [ ] **Step 2: Create crate Cargo.toml**

```toml
[package]
name = "harmony-ingest"
version.workspace = true
edition.workspace = true
license.workspace = true
rust-version.workspace = true
repository.workspace = true
description = "CLI tool for ingesting data into Harmony CAS"

[[bin]]
name = "harmony-ingest"
path = "src/main.rs"

[dependencies]
clap = { workspace = true }
half = { workspace = true }
harmony-content = { workspace = true, features = ["std"] }
harmony-engram = { workspace = true, features = ["std"] }
harmony-s3 = { workspace = true }
hex.workspace = true
safetensors = { workspace = true }
serde = { workspace = true, features = ["derive"] }
tokio = { workspace = true }
toml = { workspace = true }
tracing.workspace = true
tracing-subscriber = { workspace = true }
```

- [ ] **Step 3: Create config.rs with tests**

```rust
//! TOML config file parsing and validation for Engram ingestion.

use serde::Deserialize;
use std::path::Path;

/// Parsed Engram ingestion config.
#[derive(Debug, Clone, Deserialize)]
pub struct EngramConfig {
    /// Table version identifier (e.g. "v1").
    pub version: String,
    /// Embeddings per shard (typically 200).
    pub shard_size: u32,
    /// Per-head xxhash64 seeds.
    pub hash_seeds: Vec<u64>,
    /// Name of the tensor to extract from the safetensors file.
    pub tensor: String,
}

impl EngramConfig {
    /// Load and validate from a TOML file.
    pub fn load(path: &Path) -> Result<Self, String> {
        let text = std::fs::read_to_string(path)
            .map_err(|e| format!("failed to read config: {e}"))?;
        let config: Self =
            toml::from_str(&text).map_err(|e| format!("invalid config TOML: {e}"))?;
        config.validate()?;
        Ok(config)
    }

    /// Parse and validate from a TOML string (for testing).
    pub fn parse(text: &str) -> Result<Self, String> {
        let config: Self =
            toml::from_str(text).map_err(|e| format!("invalid config TOML: {e}"))?;
        config.validate()?;
        Ok(config)
    }

    fn validate(&self) -> Result<(), String> {
        if self.shard_size == 0 {
            return Err("shard_size must be > 0".into());
        }
        if self.hash_seeds.is_empty() {
            return Err("hash_seeds must not be empty".into());
        }
        if self.tensor.is_empty() {
            return Err("tensor name must not be empty".into());
        }
        Ok(())
    }

    /// Number of hash heads (implied by hash_seeds length).
    pub fn num_heads(&self) -> u32 {
        self.hash_seeds.len() as u32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn valid_config() {
        let toml = r#"
            version = "v1"
            shard_size = 200
            hash_seeds = [111, 222, 333, 444]
            tensor = "model.engram_table.weight"
        "#;
        let config = EngramConfig::parse(toml).unwrap();
        assert_eq!(config.version, "v1");
        assert_eq!(config.shard_size, 200);
        assert_eq!(config.hash_seeds.len(), 4);
        assert_eq!(config.num_heads(), 4);
        assert_eq!(config.tensor, "model.engram_table.weight");
    }

    #[test]
    fn zero_shard_size() {
        let toml = r#"
            version = "v1"
            shard_size = 0
            hash_seeds = [111]
            tensor = "t"
        "#;
        let err = EngramConfig::parse(toml).unwrap_err();
        assert!(err.contains("shard_size"));
    }

    #[test]
    fn empty_hash_seeds() {
        let toml = r#"
            version = "v1"
            shard_size = 200
            hash_seeds = []
            tensor = "t"
        "#;
        let err = EngramConfig::parse(toml).unwrap_err();
        assert!(err.contains("hash_seeds"));
    }

    #[test]
    fn missing_field() {
        let toml = r#"
            version = "v1"
            shard_size = 200
        "#;
        let err = EngramConfig::parse(toml).unwrap_err();
        assert!(err.contains("invalid config"));
    }
}
```

- [ ] **Step 4: Create minimal main.rs**

```rust
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

#[tokio::main]
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
```

- [ ] **Step 5: Verify it compiles and config tests pass**

Run: `cargo test -p harmony-ingest && cargo clippy -p harmony-ingest`
Expected: 4 tests pass, clippy clean

- [ ] **Step 6: Commit**

```bash
git add crates/harmony-ingest/ Cargo.toml
git commit -m "feat(ingest): scaffold harmony-ingest crate with config parsing

New binary crate for offline ingestion of data into Harmony CAS.
TOML config parsing with validation for Engram table parameters."
```

---

### Task 2: Shard Slicing + CID Computation

**Files:**
- Create: `crates/harmony-ingest/src/shard.rs`

- [ ] **Step 1: Write failing tests for shard slicing**

```rust
//! Tensor slicing into fixed-size shards with CID computation.
//!
//! Handles f32/bf16 → f16 conversion for non-f16 checkpoints.

use half::f16;
use harmony_content::cid::{ContentFlags, ContentId};

/// A single shard ready for storage.
#[derive(Debug)]
pub struct Shard {
    /// The shard's content identifier.
    pub cid: ContentId,
    /// Raw shard bytes (contiguous f16 vectors).
    pub data: Vec<u8>,
}

/// Source tensor dtype for conversion.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SourceDtype {
    F16,
    F32,
    BF16,
}

/// Compute the number of shards needed for the given table.
pub fn num_shards(total_entries: u64, shard_size: u32) -> u64 {
    (total_entries + shard_size as u64 - 1) / shard_size as u64
}

/// Convert a slice of source-dtype bytes to f16 bytes.
///
/// For F16 input, returns a copy. For F32/BF16, converts each element.
pub fn convert_to_f16(src: &[u8], dtype: SourceDtype) -> Vec<u8> {
    match dtype {
        SourceDtype::F16 => src.to_vec(),
        SourceDtype::F32 => {
            let mut out = Vec::with_capacity(src.len() / 2);
            for chunk in src.chunks_exact(4) {
                let val = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                out.extend_from_slice(&f16::from_f32(val).to_le_bytes());
            }
            out
        }
        SourceDtype::BF16 => {
            let mut out = Vec::with_capacity(src.len());
            for chunk in src.chunks_exact(2) {
                // bf16 → f32: place bf16 bytes in upper 16 bits of f32.
                let val = f32::from_bits((chunk[1] as u32) << 24 | (chunk[0] as u32) << 16);
                out.extend_from_slice(&f16::from_f32(val).to_le_bytes());
            }
            out
        }
    }
}

/// Bytes per element for the source dtype.
pub fn source_dtype_bytes(dtype: SourceDtype) -> usize {
    match dtype {
        SourceDtype::F16 | SourceDtype::BF16 => 2,
        SourceDtype::F32 => 4,
    }
}

/// Slice a single shard from the tensor's raw bytes, converting to f16.
///
/// If the shard is the last one and has fewer than `shard_size` entries,
/// the output is zero-padded to the full shard size.
pub fn slice_shard(
    tensor_bytes: &[u8],
    shard_index: u64,
    shard_size: u32,
    embedding_dim: usize,
    dtype: SourceDtype,
) -> Shard {
    let src_vector_bytes = embedding_dim * source_dtype_bytes(dtype);
    let src_shard_bytes = shard_size as usize * src_vector_bytes;
    let start = shard_index as usize * src_shard_bytes;
    let available = tensor_bytes.len().saturating_sub(start);
    let copy_len = available.min(src_shard_bytes);

    let src_slice = &tensor_bytes[start..start + copy_len];
    let f16_data = convert_to_f16(src_slice, dtype);

    // Target shard size in f16 bytes.
    let f16_vector_bytes = embedding_dim * 2;
    let f16_shard_bytes = shard_size as usize * f16_vector_bytes;

    let mut data = vec![0u8; f16_shard_bytes];
    let copy = f16_data.len().min(f16_shard_bytes);
    data[..copy].copy_from_slice(&f16_data[..copy]);

    let cid = ContentId::for_book(&data, ContentFlags::default())
        .expect("shard size is well under 1MB limit");

    Shard { cid, data }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn num_shards_exact_division() {
        assert_eq!(num_shards(12, 3), 4);
        assert_eq!(num_shards(200, 200), 1);
    }

    #[test]
    fn num_shards_remainder() {
        assert_eq!(num_shards(7, 3), 3); // 3 + 3 + 1
        assert_eq!(num_shards(1, 200), 1);
    }

    #[test]
    fn slice_shard_basic() {
        // 4 entries, dim=2, f16 (2 bytes) → 4 bytes per vector, 16 bytes total
        // shard_size=2 → 2 shards of 8 bytes each
        let tensor_bytes: Vec<u8> = (0..16).collect();
        let shard = slice_shard(&tensor_bytes, 0, 2, 2, SourceDtype::F16);
        assert_eq!(shard.data, &tensor_bytes[0..8]);
        assert_eq!(shard.data.len(), 8);
    }

    #[test]
    fn slice_shard_second() {
        let tensor_bytes: Vec<u8> = (0..16).collect();
        let shard = slice_shard(&tensor_bytes, 1, 2, 2, SourceDtype::F16);
        assert_eq!(shard.data, &tensor_bytes[8..16]);
    }

    #[test]
    fn slice_shard_last_padded() {
        // 5 entries, shard_size=3 → shard 1 has 2 entries + 1 zero-padded
        let vector_bytes = 2 * 2; // dim=2, f16=2 bytes
        let tensor_bytes: Vec<u8> = (0..(5 * vector_bytes) as u8).collect();
        let shard = slice_shard(&tensor_bytes, 1, 3, 2, SourceDtype::F16);
        // Should be 3 * 4 = 12 bytes (f16), last 4 are zeros
        assert_eq!(shard.data.len(), 12);
        assert_eq!(&shard.data[0..8], &tensor_bytes[12..20]);
        assert_eq!(&shard.data[8..12], &[0, 0, 0, 0]);
    }

    #[test]
    fn slice_shard_deterministic_cid() {
        let tensor_bytes: Vec<u8> = (0..16).collect();
        let s1 = slice_shard(&tensor_bytes, 0, 2, 2, SourceDtype::F16);
        let s2 = slice_shard(&tensor_bytes, 0, 2, 2, SourceDtype::F16);
        assert_eq!(s1.cid, s2.cid);
    }

    #[test]
    fn different_shards_different_cids() {
        let tensor_bytes: Vec<u8> = (0..16).collect();
        let s1 = slice_shard(&tensor_bytes, 0, 2, 2, SourceDtype::F16);
        let s2 = slice_shard(&tensor_bytes, 1, 2, 2, SourceDtype::F16);
        assert_ne!(s1.cid, s2.cid);
    }

    #[test]
    fn convert_f32_to_f16() {
        // Two f32 values: 1.0 and 2.0
        let mut src = Vec::new();
        src.extend_from_slice(&1.0f32.to_le_bytes());
        src.extend_from_slice(&2.0f32.to_le_bytes());
        let f16_bytes = convert_to_f16(&src, SourceDtype::F32);
        assert_eq!(f16_bytes.len(), 4); // 2 × 2 bytes
        let v0 = f16::from_le_bytes([f16_bytes[0], f16_bytes[1]]);
        let v1 = f16::from_le_bytes([f16_bytes[2], f16_bytes[3]]);
        assert_eq!(v0.to_f32(), 1.0);
        assert_eq!(v1.to_f32(), 2.0);
    }

    #[test]
    fn slice_shard_f32_converts_to_f16() {
        // 2 entries, dim=2, f32 input → shard should be f16
        let mut tensor_bytes = Vec::new();
        for val in &[1.0f32, 2.0, 3.0, 4.0] {
            tensor_bytes.extend_from_slice(&val.to_le_bytes());
        }
        // 4 f32 values = 16 bytes, shard_size=2, dim=2
        let shard = slice_shard(&tensor_bytes, 0, 2, 2, SourceDtype::F32);
        // Output should be f16: 2 vectors × 2 dims × 2 bytes = 8 bytes
        assert_eq!(shard.data.len(), 8);
        let v0 = f16::from_le_bytes([shard.data[0], shard.data[1]]);
        assert_eq!(v0.to_f32(), 1.0);
    }
}
```

- [ ] **Step 2: Run tests to verify they pass**

Run: `cargo test -p harmony-ingest shard::tests -- --no-capture`
Expected: all 6 tests PASS (the code is written inline with tests)

- [ ] **Step 3: Add `mod shard;` to main.rs**

- [ ] **Step 4: Commit**

```bash
git add crates/harmony-ingest/src/shard.rs crates/harmony-ingest/src/main.rs
git commit -m "feat(ingest): shard slicing with CID computation and f16 conversion

slice_shard() extracts fixed-size embedding chunks from raw tensor
bytes with zero-padding for the last shard. convert_to_f16() handles
f32/bf16 → f16 conversion for non-f16 checkpoints."
```

---

### Task 3: CID Journal (Resume Support)

**Files:**
- Create: `crates/harmony-ingest/src/journal.rs`

- [ ] **Step 1: Write journal implementation with tests**

```rust
//! Binary CID journal for crash-safe resume.
//!
//! Each entry is a raw 32-byte ContentId, appended after each shard upload.
//! On resume, the journal is read to recover previously collected CIDs.

use harmony_content::cid::ContentId;
use std::fs::{File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

/// Append-only binary journal of 32-byte CID entries.
pub struct CidJournal {
    path: PathBuf,
    file: File,
}

impl CidJournal {
    /// Create or open a journal file.  Truncates any partial last entry.
    pub fn open(path: &Path) -> Result<Self, String> {
        // If file exists, truncate to a multiple of 32 bytes.
        if path.exists() {
            let len = std::fs::metadata(path)
                .map_err(|e| format!("journal metadata: {e}"))?
                .len();
            let aligned = len - (len % 32);
            if aligned < len {
                let file = OpenOptions::new()
                    .write(true)
                    .open(path)
                    .map_err(|e| format!("journal open for truncate: {e}"))?;
                file.set_len(aligned)
                    .map_err(|e| format!("journal truncate: {e}"))?;
            }
        }

        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .map_err(|e| format!("journal open: {e}"))?;

        Ok(Self {
            path: path.to_path_buf(),
            file,
        })
    }

    /// Append a CID to the journal.
    pub fn append(&mut self, cid: &ContentId) -> Result<(), String> {
        self.file
            .write_all(&cid.to_bytes())
            .map_err(|e| format!("journal write: {e}"))?;
        self.file
            .flush()
            .map_err(|e| format!("journal flush: {e}"))?;
        Ok(())
    }

    /// Read all CIDs from the journal.
    pub fn read_all(path: &Path) -> Result<Vec<[u8; 32]>, String> {
        if !path.exists() {
            return Ok(Vec::new());
        }
        let mut data = Vec::new();
        File::open(path)
            .map_err(|e| format!("journal read: {e}"))?
            .read_to_end(&mut data)
            .map_err(|e| format!("journal read: {e}"))?;

        // Truncate to aligned length (ignore partial last entry).
        let aligned = data.len() - (data.len() % 32);
        let count = aligned / 32;
        let mut cids = Vec::with_capacity(count);
        for chunk in data[..aligned].chunks_exact(32) {
            let mut cid = [0u8; 32];
            cid.copy_from_slice(chunk);
            cids.push(cid);
        }
        Ok(cids)
    }

    /// Number of entries written so far (based on file size).
    pub fn entry_count(&self) -> Result<u64, String> {
        let len = std::fs::metadata(&self.path)
            .map_err(|e| format!("journal metadata: {e}"))?
            .len();
        Ok(len / 32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use harmony_content::cid::ContentFlags;

    fn make_cid(data: &[u8]) -> ContentId {
        ContentId::for_book(data, ContentFlags::default()).unwrap()
    }

    #[test]
    fn write_and_read_back() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.journal");

        let cid1 = make_cid(b"shard one");
        let cid2 = make_cid(b"shard two");

        {
            let mut journal = CidJournal::open(&path).unwrap();
            journal.append(&cid1).unwrap();
            journal.append(&cid2).unwrap();
            assert_eq!(journal.entry_count().unwrap(), 2);
        }

        let cids = CidJournal::read_all(&path).unwrap();
        assert_eq!(cids.len(), 2);
        assert_eq!(cids[0], cid1.to_bytes());
        assert_eq!(cids[1], cid2.to_bytes());
    }

    #[test]
    fn truncates_partial_entry() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.journal");

        // Write 1 full entry + 10 garbage bytes.
        let cid = make_cid(b"complete");
        {
            let mut f = File::create(&path).unwrap();
            f.write_all(&cid.to_bytes()).unwrap();
            f.write_all(&[0xFF; 10]).unwrap(); // partial entry
        }

        // Open should truncate the partial entry.
        let _journal = CidJournal::open(&path).unwrap();

        let cids = CidJournal::read_all(&path).unwrap();
        assert_eq!(cids.len(), 1);
        assert_eq!(cids[0], cid.to_bytes());
    }

    #[test]
    fn empty_journal() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("nonexistent.journal");
        let cids = CidJournal::read_all(&path).unwrap();
        assert!(cids.is_empty());
    }

    #[test]
    fn reopen_and_append() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.journal");

        let cid1 = make_cid(b"first");
        let cid2 = make_cid(b"second");

        {
            let mut journal = CidJournal::open(&path).unwrap();
            journal.append(&cid1).unwrap();
        }
        {
            let mut journal = CidJournal::open(&path).unwrap();
            journal.append(&cid2).unwrap();
        }

        let cids = CidJournal::read_all(&path).unwrap();
        assert_eq!(cids.len(), 2);
    }
}
```

Note: add `tempfile = "3"` to `[dev-dependencies]` in `crates/harmony-ingest/Cargo.toml`.

- [ ] **Step 2: Run tests**

Run: `cargo test -p harmony-ingest journal::tests -- --no-capture`
Expected: all 4 tests PASS

- [ ] **Step 3: Add `mod journal;` to main.rs**

- [ ] **Step 4: Commit**

```bash
git add crates/harmony-ingest/src/journal.rs crates/harmony-ingest/src/main.rs crates/harmony-ingest/Cargo.toml
git commit -m "feat(ingest): binary CID journal for crash-safe resume

Append-only 32-byte CID entries with partial-write truncation recovery.
Enables resuming ingestion from the last successfully uploaded shard."
```

---

### Task 4: Storage Layer (S3 + Local Dir)

**Files:**
- Create: `crates/harmony-ingest/src/storage.rs`

- [ ] **Step 1: Write storage implementation with tests**

```rust
//! Storage backends for shard and manifest book uploads.
//!
//! Supports S3 (via harmony-s3) and/or local directory with two-level
//! hex prefix layout for filesystem scalability.

use harmony_content::cid::ContentId;
use std::path::{Path, PathBuf};

/// Write a book to the local directory with two-level hex prefix.
///
/// Layout: `{base}/book/{hex[0..2]}/{hex_cid}`
pub fn write_local_book(base: &Path, cid: &ContentId, data: &[u8]) -> Result<(), String> {
    let hex_cid = hex::encode(cid.to_bytes());
    let prefix_dir = base.join("book").join(&hex_cid[..2]);
    std::fs::create_dir_all(&prefix_dir)
        .map_err(|e| format!("create dir {}: {e}", prefix_dir.display()))?;
    let file_path = prefix_dir.join(&hex_cid);
    std::fs::write(&file_path, data)
        .map_err(|e| format!("write {}: {e}", file_path.display()))?;
    Ok(())
}

/// Compute the expected local path for a book.
pub fn local_book_path(base: &Path, cid: &ContentId) -> PathBuf {
    let hex_cid = hex::encode(cid.to_bytes());
    base.join("book").join(&hex_cid[..2]).join(&hex_cid)
}

/// Upload a book to S3 with retry (exponential backoff).
///
/// Retries up to 3 times with 1s, 2s, 4s delays.
pub async fn upload_s3_book(
    s3: &harmony_s3::S3Library,
    cid: &ContentId,
    data: Vec<u8>,
) -> Result<(), String> {
    let cid_bytes = cid.to_bytes();
    let delays = [1, 2, 4];
    let mut last_err = String::new();

    for (attempt, delay_secs) in delays.iter().enumerate() {
        match s3.put_book(&cid_bytes, data.clone()).await {
            Ok(()) => return Ok(()),
            Err(e) => {
                last_err = format!("{e}");
                tracing::warn!(
                    attempt = attempt + 1,
                    delay_secs,
                    err = %e,
                    "S3 upload failed, retrying"
                );
                tokio::time::sleep(std::time::Duration::from_secs(*delay_secs)).await;
            }
        }
    }

    Err(format!("S3 upload failed after 3 attempts: {last_err}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use harmony_content::cid::ContentFlags;

    fn make_cid(data: &[u8]) -> ContentId {
        ContentId::for_book(data, ContentFlags::default()).unwrap()
    }

    #[test]
    fn local_book_path_two_level_prefix() {
        let cid = make_cid(b"test data");
        let path = local_book_path(Path::new("/mnt/usb"), &cid);
        let hex_cid = hex::encode(cid.to_bytes());
        let expected = format!("/mnt/usb/book/{}/{}", &hex_cid[..2], hex_cid);
        assert_eq!(path.to_str().unwrap(), expected);
    }

    #[test]
    fn write_and_read_local_book() {
        let dir = tempfile::tempdir().unwrap();
        let data = b"shard bytes here";
        let cid = make_cid(data);

        write_local_book(dir.path(), &cid, data).unwrap();

        let path = local_book_path(dir.path(), &cid);
        let read_back = std::fs::read(&path).unwrap();
        assert_eq!(read_back, data);
    }

    #[test]
    fn write_same_book_twice_is_idempotent() {
        let dir = tempfile::tempdir().unwrap();
        let data = b"same content";
        let cid = make_cid(data);

        write_local_book(dir.path(), &cid, data).unwrap();
        write_local_book(dir.path(), &cid, data).unwrap(); // no error

        let path = local_book_path(dir.path(), &cid);
        let read_back = std::fs::read(&path).unwrap();
        assert_eq!(read_back, data);
    }
}
```

- [ ] **Step 2: Run tests**

Run: `cargo test -p harmony-ingest storage::tests -- --no-capture`
Expected: all 3 tests PASS

- [ ] **Step 3: Add `mod storage;` to main.rs**

- [ ] **Step 4: Commit**

```bash
git add crates/harmony-ingest/src/storage.rs crates/harmony-ingest/src/main.rs
git commit -m "feat(ingest): S3 upload with retry and local dir storage

write_local_book() uses two-level hex prefix (256 subdirectories).
upload_s3_book() retries with exponential backoff (1s/2s/4s)."
```

---

### Task 5: Manifest Building + MemoryBookStore.into_books()

**Files:**
- Create: `crates/harmony-ingest/src/manifest.rs`
- Modify: `crates/harmony-content/src/book.rs`

- [ ] **Step 1: Add `into_books()` to MemoryBookStore**

In `crates/harmony-content/src/book.rs`, add to `impl MemoryBookStore`:

```rust
    /// Consume the store and return all (CID, data) pairs.
    pub fn into_books(self) -> impl Iterator<Item = (ContentId, Vec<u8>)> {
        self.data.into_iter()
    }
```

- [ ] **Step 2: Write manifest module with tests**

```rust
//! Manifest building: ManifestHeader + shard CID list → Merkle DAG.

use harmony_content::book::MemoryBookStore;
use harmony_content::chunker::ChunkerConfig;
use harmony_content::cid::ContentId;
use harmony_content::dag;
use harmony_engram::ManifestHeader;

/// Build the manifest DAG from header + shard CIDs.
///
/// Returns (root_cid, store_containing_all_dag_books).
pub fn build_manifest(
    header: &ManifestHeader,
    shard_cids: &[[u8; 32]],
) -> Result<(ContentId, MemoryBookStore), String> {
    // Serialize header.
    let header_bytes = header
        .to_bytes()
        .map_err(|e| format!("manifest header serialize: {e}"))?;

    // Concatenate: [header_bytes | shard_cid_bytes]
    let cid_bytes_len = shard_cids.len() * 32;
    let mut combined = Vec::with_capacity(header_bytes.len() + cid_bytes_len);
    combined.extend_from_slice(&header_bytes);
    for cid in shard_cids {
        combined.extend_from_slice(cid);
    }

    // Ingest into a fresh MemoryBookStore.
    let mut store = MemoryBookStore::new();
    let root_cid = dag::ingest(&combined, &ChunkerConfig::DEFAULT, &mut store)
        .map_err(|e| format!("dag ingest: {e}"))?;

    Ok((root_cid, store))
}

/// Construct a ManifestHeader from config + tensor metadata.
pub fn make_header(
    version: String,
    embedding_dim: u32,
    num_heads: u32,
    hash_seeds: Vec<u64>,
    total_entries: u64,
    shard_size: u32,
    num_shards: u64,
) -> ManifestHeader {
    ManifestHeader {
        version,
        embedding_dim,
        dtype_bytes: 2, // f16
        num_heads,
        hash_seeds,
        total_entries,
        shard_size,
        num_shards,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use harmony_content::dag;

    #[test]
    fn build_and_recover_manifest() {
        let header = make_header(
            "v1".into(),
            4,    // embedding_dim
            2,    // num_heads
            vec![42, 99],
            6,    // total_entries
            3,    // shard_size
            2,    // num_shards
        );

        // 2 dummy shard CIDs.
        let shard_cids: Vec<[u8; 32]> = vec![[0xAA; 32], [0xBB; 32]];

        let (root_cid, store) = build_manifest(&header, &shard_cids).unwrap();

        // Reassemble the manifest from the DAG.
        let reassembled = dag::reassemble(&root_cid, &store).unwrap();

        // First part is the postcard-encoded header.
        let header_bytes = header.to_bytes().unwrap();
        let recovered_header =
            ManifestHeader::from_bytes(&reassembled[..header_bytes.len()]).unwrap();
        assert_eq!(recovered_header, header);

        // Remaining bytes are shard CIDs.
        let cid_bytes = &reassembled[header_bytes.len()..];
        assert_eq!(cid_bytes.len(), 64); // 2 × 32
        assert_eq!(&cid_bytes[0..32], &[0xAA; 32]);
        assert_eq!(&cid_bytes[32..64], &[0xBB; 32]);
    }

    #[test]
    fn into_books_returns_all_dag_nodes() {
        let header = make_header("v1".into(), 4, 1, vec![0], 3, 3, 1);
        let shard_cids = vec![[0x11; 32]];
        let (_root_cid, store) = build_manifest(&header, &shard_cids).unwrap();

        // Store should contain at least 1 book (the manifest data).
        let books: Vec<_> = store.into_books().collect();
        assert!(!books.is_empty());
        // Each book has a CID and data.
        for (cid, data) in &books {
            assert!(!data.is_empty());
            assert_eq!(cid.to_bytes().len(), 32);
        }
    }
}
```

- [ ] **Step 3: Run tests**

Run: `cargo test -p harmony-ingest manifest::tests -- --no-capture`
Expected: 2 tests PASS

Run: `cargo test -p harmony-content` to verify `into_books` doesn't break anything.

- [ ] **Step 4: Add `mod manifest;` to main.rs**

- [ ] **Step 5: Commit**

```bash
git add crates/harmony-content/src/book.rs crates/harmony-ingest/src/manifest.rs crates/harmony-ingest/src/main.rs
git commit -m "feat(ingest): manifest DAG building from header + shard CIDs

build_manifest() serializes ManifestHeader + shard CID list, ingests
via dag::ingest into a MemoryBookStore, returns root CID + store.

Adds into_books() to MemoryBookStore for DAG book enumeration."
```

---

### Task 6: End-to-End Pipeline Wiring

**Files:**
- Modify: `crates/harmony-ingest/src/main.rs`

This task wires all modules together in `main.rs` — the full engram ingestion pipeline.

- [ ] **Step 1: Implement the `run_engram` async function**

Add to `main.rs`:

```rust
mod config;
mod journal;
mod manifest;
mod shard;
mod storage;

use clap::{Parser, Subcommand};
use harmony_content::cid::ContentId;
use std::path::PathBuf;

// ... (Cli and Commands structs from Task 1, unchanged) ...

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
    let file_bytes = std::fs::read(&input)
        .map_err(|e| format!("failed to read {}: {e}", input.display()))?;
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
    tracing::info!(
        total_entries,
        embedding_dim,
        n_shards,
        "tensor validated"
    );

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
        let s = shard::slice_shard(
            tensor_bytes,
            i,
            cfg.shard_size,
            embedding_dim,
            src_dtype,
        );

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

#[tokio::main]
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
            run_engram(config, input, bucket, prefix, region, local_dir, resume_from).await
        }
    };

    if let Err(e) = result {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}
```

- [ ] **Step 2: Write integration test**

Create `crates/harmony-ingest/tests/engram_ingest.rs`:

```rust
//! End-to-end integration test with a tiny safetensors file.

use half::f16;
use harmony_content::book::{BookStore, MemoryBookStore};
use harmony_content::chunker::ChunkerConfig;
use harmony_content::cid::{ContentFlags, ContentId};
use harmony_content::dag;
use harmony_engram::ManifestHeader;
use std::collections::HashMap;
use std::io::Write;

/// Create a minimal safetensors file with a single f16 tensor.
fn create_test_safetensors(name: &str, rows: usize, cols: usize) -> Vec<u8> {
    // Build raw f16 data: value = (row * cols + col) as f16.
    let mut f16_bytes = Vec::with_capacity(rows * cols * 2);
    for r in 0..rows {
        for c in 0..cols {
            let val = f16::from_f32((r * cols + c) as f32);
            f16_bytes.extend_from_slice(&val.to_le_bytes());
        }
    }

    // safetensors format: 8-byte header_size (LE u64), JSON header, tensor data.
    let tensor_info = serde_json::json!({
        name: {
            "dtype": "F16",
            "shape": [rows, cols],
            "data_offsets": [0, f16_bytes.len()]
        }
    });
    let header_json = serde_json::to_string(&tensor_info).unwrap();
    let header_len = header_json.len() as u64;

    let mut buf = Vec::new();
    buf.extend_from_slice(&header_len.to_le_bytes());
    buf.extend_from_slice(header_json.as_bytes());
    buf.extend_from_slice(&f16_bytes);
    buf
}

#[test]
fn end_to_end_local_dir() {
    let dir = tempfile::tempdir().unwrap();
    let local_dir = dir.path().join("cache");
    let input_path = dir.path().join("test.safetensors");

    // 6 entries, dim=4, shard_size=3 → 2 shards
    let safetensors_bytes = create_test_safetensors("engram.weight", 6, 4);
    std::fs::write(&input_path, &safetensors_bytes).unwrap();

    // Write config.
    let config_path = dir.path().join("config.toml");
    std::fs::write(
        &config_path,
        r#"
        version = "v1"
        shard_size = 3
        hash_seeds = [42, 99]
        tensor = "engram.weight"
        "#,
    )
    .unwrap();

    // Parse the safetensors to get raw tensor bytes for verification.
    let tensors = safetensors::SafeTensors::deserialize(&safetensors_bytes).unwrap();
    let tensor = tensors.tensor("engram.weight").unwrap();
    let tensor_bytes = tensor.data();
    let vector_bytes = 4 * 2; // dim=4, f16=2 bytes
    let shard_bytes = 3 * vector_bytes; // shard_size=3

    // Manually compute expected shards.
    let shard0_data = &tensor_bytes[0..shard_bytes];
    let shard1_data = &tensor_bytes[shard_bytes..2 * shard_bytes];
    let cid0 = ContentId::for_book(shard0_data, ContentFlags::default()).unwrap();
    let cid1 = ContentId::for_book(shard1_data, ContentFlags::default()).unwrap();

    // Run the CLI (local-dir only, no S3).
    let status = std::process::Command::new(env!("CARGO_BIN_EXE_harmony-ingest"))
        .args([
            "engram",
            "--config",
            config_path.to_str().unwrap(),
            "--input",
            input_path.to_str().unwrap(),
            "--local-dir",
            local_dir.to_str().unwrap(),
        ])
        .status()
        .unwrap();
    assert!(status.success(), "harmony-ingest failed");

    // Verify shard files exist with correct content.
    let hex0 = hex::encode(cid0.to_bytes());
    let path0 = local_dir.join("book").join(&hex0[..2]).join(&hex0);
    assert!(path0.exists(), "shard 0 not found at {}", path0.display());
    assert_eq!(std::fs::read(&path0).unwrap(), shard0_data);

    let hex1 = hex::encode(cid1.to_bytes());
    let path1 = local_dir.join("book").join(&hex1[..2]).join(&hex1);
    assert!(path1.exists(), "shard 1 not found at {}", path1.display());
    assert_eq!(std::fs::read(&path1).unwrap(), shard1_data);

    // Verify journal has 2 entries.
    let journal_path = input_path.with_extension("journal");
    let journal_data = std::fs::read(&journal_path).unwrap();
    assert_eq!(journal_data.len(), 64); // 2 × 32 bytes
    assert_eq!(&journal_data[0..32], &cid0.to_bytes());
    assert_eq!(&journal_data[32..64], &cid1.to_bytes());

    // Verify manifest DAG books exist in local dir.
    // At least 1 book (the manifest itself) should be present beyond the 2 shards.
    let mut book_count = 0;
    for entry in walkdir::WalkDir::new(local_dir.join("book"))
        .into_iter()
        .filter_map(|e| e.ok())
    {
        if entry.file_type().is_file() {
            book_count += 1;
        }
    }
    // 2 shards + at least 1 manifest book
    assert!(
        book_count >= 3,
        "expected >= 3 books, found {book_count}"
    );
}
```

Note: add `serde_json = "1"` and `walkdir = "2"` to `[dev-dependencies]` in Cargo.toml.

- [ ] **Step 3: Verify everything compiles and tests pass**

Run: `cargo test -p harmony-ingest -- --no-capture`
Expected: all unit tests + integration test pass

Run: `cargo clippy -p harmony-ingest`
Expected: clean

- [ ] **Step 4: Run cargo fmt**

Run: `cargo fmt -p harmony-ingest -p harmony-content`

- [ ] **Step 5: Commit**

```bash
git add crates/harmony-ingest/ crates/harmony-content/src/book.rs
git commit -m "feat(ingest): wire end-to-end Engram ingestion pipeline

Stages 1-6: config → safetensors → shard → upload → manifest DAG → output.
Integration test with tiny safetensors file verifies full round-trip."
```
