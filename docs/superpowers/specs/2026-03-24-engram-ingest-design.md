# Engram Checkpoint Ingestion Pipeline

## Goal

Import DeepSeek Engram embedding tables from safetensors files into Harmony CAS books, uploading to S3 Great Library and/or a local directory cache. A single flat embedding table is sharded into 64KB CAS books, a Merkle DAG manifest is built, and the root CID becomes the table's permanent identifier.

## Architecture

**`harmony-ingest`** — new binary crate in `crates/harmony-ingest/`. Offline batch CLI tool for ingesting data into Harmony CAS. First subcommand: `engram`.

```
harmony-ingest engram \
  --config engram-deepseek-v1.toml \
  --input model.safetensors \
  --bucket harmony-data \
  --prefix harmony/ \
  --local-dir /mnt/usb/harmony-cache/
```

Not a library crate. No sans-I/O pattern. This is a CLI tool that orchestrates I/O — reads files, uploads to S3, writes to local disk.

### Dependencies

| Crate | Purpose |
|-------|---------|
| `safetensors` | Read tensor files (mmap, zero-copy) |
| `harmony-engram` | `ManifestHeader` serialization |
| `harmony-content` | `BookStore`, `ContentId`, `BundleBuilder`, `dag::ingest` |
| `harmony-s3` | `S3Library` for remote upload |
| `half` | f16 type and conversion |
| `clap` | CLI argument parsing |
| `tokio` | Async runtime for S3 operations |
| `tracing` / `tracing-subscriber` | Structured logging |
| `toml` / `serde` | Config file parsing |
| `hex` | CID hex encoding for filenames |

## Config File Format

```toml
version = "v1"
shard_size = 200
hash_seeds = [111, 222, 333, 444]
tensor = "model.engram_table.weight"
```

| Field | Type | Description |
|-------|------|-------------|
| `version` | String | Table version identifier, used in Zenoh key expressions |
| `shard_size` | u32 | Embeddings per shard (typically 200) |
| `hash_seeds` | Vec\<u64\> | Per-head xxhash64 seeds. Length = num_heads |
| `tensor` | String | Name of the tensor to extract from the safetensors file |

The number of heads is implied by `hash_seeds.len()`. The tool derives `embedding_dim`, `total_entries`, `num_shards`, and `dtype_bytes` from the tensor's shape and dtype.

### Validation

- `shard_size > 0`
- `hash_seeds` non-empty
- Named tensor exists in the safetensors file
- Tensor shape is 2D: `[total_entries, embedding_dim]`
- Tensor dtype is f16 (or f32/bf16 with automatic f16 conversion)
- `total_entries` is divisible by `shard_size` (or the last shard is padded with zeros)

## Ingestion Pipeline

### Stage 1: Load Config

Parse the TOML config file. Validate all fields. Derive `num_heads = hash_seeds.len()`.

### Stage 2: Open Safetensors

Open the safetensors file via mmap (zero-copy access). Extract the named tensor. Validate shape is `[total_entries, embedding_dim]`. If dtype is f32 or bf16, convert to f16 in a streaming fashion (one shard's worth at a time to avoid doubling memory usage). If dtype is f16, use the raw bytes directly.

The Engram table is a single shared flat table. Multiple hash heads index into different positions of the same table at query time — there are no per-head tables.

### Stage 3: Shard and Upload

For each shard index `i` in `0..num_shards`:

1. Slice rows `[i * shard_size .. (i+1) * shard_size]` from the tensor → contiguous f16 bytes (typically 64KB)
2. Compute `ContentId::for_book(&shard_bytes, ContentFlags::default())`
3. If S3 configured: `s3.put_book(&cid_bytes, shard_bytes.to_vec()).await`
4. If local dir configured: write to `{local-dir}/book/{hex[0..2]}/{hex_cid}` (two-level prefix for filesystem scalability)
5. Append the 32-byte CID to the journal file
6. Log progress every 10,000 shards

**Last shard handling:** If `total_entries % shard_size != 0`, the last shard has fewer than `shard_size` vectors. Pad with zero bytes to fill the shard to the expected size. This ensures all shards are uniform and byte offsets computed by `compute_lookup` are always valid.

### Stage 4: Build Manifest

1. Construct `ManifestHeader` from config + tensor metadata:
   ```
   ManifestHeader {
       version, embedding_dim, dtype_bytes: 2, num_heads,
       hash_seeds, total_entries, shard_size, num_shards,
   }
   ```
2. Serialize header via `ManifestHeader::to_bytes()` (postcard)
3. Concatenate all shard CIDs as raw bytes: `num_shards × 32 bytes`
4. Combine: `[header_bytes | shard_cid_bytes]`
5. Ingest combined bytes into CAS via `dag::ingest()` → root ContentId

The `dag::ingest` function handles chunking the 1.6GB manifest into ~1MB books, building the Merkle DAG, and returning the root CID. All intermediate books are stored in the provided `BookStore`.

### Stage 5: Upload Manifest DAG

Iterate all books in the BookStore produced by `dag::ingest()`. Upload each to S3 and/or write to local dir. These are the manifest DAG chunks — typically ~1600 books for a 50M shard table.

### Stage 6: Output

Print the manifest root CID (64 hex chars). This single CID identifies the entire Engram table version. Nodes configure this CID to load the table at startup.

```
Engram table ingested successfully.
  Version:       v1
  Total entries: 10,000,000,000
  Embedding dim: 160
  Num shards:    50,000,000
  Manifest CID:  ab01cd23...ef89
```

## Local Directory Layout

Two-level hex prefix for filesystem scalability (256 subdirectories, ~195K files each for 50M shards):

```
{local-dir}/
  book/
    00/
      00a1b2c3...64chars.bin
      00d4e5f6...64chars.bin
    01/
      01...
    ...
    ff/
      ff...
```

Each file is raw book bytes (no header). The filename is the full 64-char lowercase hex ContentId. This matches S3's `{prefix}book/{hex_cid}` key format (minus the prefix path).

## Error Handling and Resumability

### Retry

S3 upload failures are retried with exponential backoff: 3 attempts at 1s, 2s, 4s intervals. After 3 failures on the same shard, the tool exits with the failing shard index.

### CID Journal

As each shard is uploaded, its 32-byte CID is appended to a binary journal file (`{input}.journal`). The journal is append-only and crash-safe — a partial write of the last entry is detected and truncated on resume.

On resume (`--resume-from <shard_index>`), the tool:
1. Reads the journal to recover CIDs for shards `0..shard_index`
2. Validates journal has exactly `shard_index` entries (32 bytes each)
3. Continues sharding and uploading from `shard_index` onward

### Idempotency

S3 PUT is idempotent for CAS content — re-uploading the same CID overwrites with identical bytes. Local file writes are idempotent for the same reason. A crash mid-shard means that shard is re-uploaded on resume. No corruption risk.

## CLI Interface

```
harmony-ingest engram [OPTIONS]

Required:
  --config <PATH>       Path to TOML config file
  --input <PATH>        Path to safetensors file

Storage (at least one required):
  --bucket <NAME>       S3 bucket name
  --prefix <PREFIX>     S3 key prefix (default: "harmony/")
  --region <REGION>     AWS region (default: from environment)
  --local-dir <PATH>    Local directory for book cache

Resume:
  --resume-from <N>     Resume from shard index N (reads journal)
```

## Testing Strategy

### Unit tests

- Config parsing: valid TOML → correct struct, missing fields → error, invalid shard_size → error
- Shard slicing: known tensor bytes → correct shard boundaries
- Last shard padding: non-divisible total_entries → zero-padded last shard
- CID journal: write N entries → read back N entries, truncated last entry detected and handled
- Local dir layout: CID → correct two-level prefix path

### Integration test

End-to-end with a tiny in-memory safetensors file:
- 6 entries, embedding_dim=4, shard_size=3 → 2 shards
- Local dir only (no S3)
- Verify: 2 shard files with correct bytes, manifest DAG written, ManifestHeader round-trips correctly from the DAG, journal has 2 entries matching shard CIDs

### No S3 integration tests

S3 upload is a thin wrapper over `harmony-s3::S3Library::put_book()`, already tested in that crate. The ingestion tool tests with local dir and trusts the S3 path.

## Scope Exclusions

- **Downloading checkpoints** — the tool assumes the safetensors file is already on disk
- **Model-specific tensor discovery** — the config explicitly names the tensor
- **Ingestion of non-Engram data** — future subcommands (e.g., `harmony-ingest gguf`) are out of scope
- **Serving ingested tables** — that's harmony-node's job, using the Engram client from harmony-engram
- **Key projection matrices** — the per-head `W_K` matrices are model weights, not part of the Engram table. They live in the model checkpoint, not in CAS.
