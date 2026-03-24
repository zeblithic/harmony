# Engram-as-CAS: NDN-Backed Distributed Engram Tables

## Goal

Map DeepSeek's Engram conditional memory tables onto Harmony's content-addressed storage, enabling edge devices to perform O(1) factual lookups against a 100-billion-entry embedding table distributed across the mesh — without storing the table locally.

## Architecture

Three components:

- **Engram table format** — Sharded embedding tables stored as Harmony CAS books. A Vine DAG manifest maps shard indices to CIDs. Each shard holds ~200 embedding vectors (~64KB).
- **Query protocol** — Zenoh queryable namespace `harmony/engram/{version}/`. Client computes multi-head hashes locally, issues parallel shard fetches, extracts vectors by byte offset.
- **harmony-engram crate** — Sans-I/O client performing tokenizer compression, multi-head hashing, shard resolution, and vector aggregation. Emits lookup requests; caller handles network I/O.

## Engram Table Storage Format

### Manifest

The manifest is a structured document containing table metadata and the complete shard-to-CID mapping:

```
version: "v1"
embedding_dim: 160
dtype: "float16"  (2 bytes per component → 320 bytes per vector)
num_heads: 4
hash_function: "xxhash64"
hash_seeds: [seed_0, seed_1, seed_2, seed_3]
total_entries: 10_000_000_000
shard_size: 200  (embeddings per shard)
num_shards: 50_000_000
shards: [CID_0, CID_1, ..., CID_49999999]
```

The manifest itself is serialized via postcard and stored as a Harmony Vine DAG (Merkle tree of books), since the shard CID list (50M × 32 bytes = 1.6GB) exceeds the 1MB book limit. The manifest's **root CID** is the single identifier for the entire Engram table version.

### Shard Books

Each shard is a contiguous block of embedding vectors with no per-shard header:

```
[vector_0: 320 bytes][vector_1: 320 bytes]...[vector_199: 320 bytes]
```

Shard size: `shard_size × embedding_dim × dtype_bytes` = 200 × 160 × 2 = 64,000 bytes (62.5KB). Well within the 1MB book limit.

### Addressing

Given a hash value `h`:
- Shard index: `h / shard_size`
- Byte offset within shard: `(h % shard_size) * embedding_dim * dtype_bytes`

The shard's CID is looked up from the manifest by shard index. The vector is a fixed-size byte slice at the computed offset.

## Query Protocol

### Zenoh Key Expressions

```
harmony/engram/{version}/manifest        — root CID of the manifest DAG
harmony/engram/{version}/shard/{index}   — individual shard queryable
```

Nodes hosting Engram shards declare queryables on `harmony/engram/{version}/shard/**`. Any node holding a cached shard can respond.

### Lookup Flow

1. **Tokenizer compression** — normalize input tokens (lowercase, Unicode canonicalization). Pure local computation.

2. **Multi-head hashing** — compute `num_heads` independent hash values using xxhash64 with per-head seeds. Each hash produces a table index. Pure local computation.

3. **Shard resolution** — for each hash value, compute `shard_index = hash / shard_size`. Look up the shard's CID from the locally-cached manifest.

4. **Parallel shard fetch** — issue up to `num_heads` Zenoh queries in parallel for needed shards. Locally cached shards resolve instantly. Uncached shards route through the mesh — any node holding the shard responds. Fetched shards are cached locally (content-addressed, so cache is always valid).

5. **Vector extraction** — for each shard, extract the embedding at `(hash % shard_size) * embedding_bytes`. No deserialization — direct byte slice.

6. **Aggregation** — sum the `num_heads` embedding vectors into a single combined embedding.

7. **Context-aware gating** — the model's forward pass computes a scalar gate. This is model computation, not part of the Engram protocol.

### Caching Properties

N-gram access follows Zipfian distribution — a small subset of common phrases and facts is accessed exponentially more often. Hot shards naturally cache at edge nodes. Cold shards (rare trivia) fall back through the mesh to S3 Great Library. After warmup, most lookups resolve from local cache with zero network latency.

## harmony-engram Crate

Sans-I/O client. No network dependencies. Testable without Zenoh.

### Core Types

```rust
/// Configuration for an Engram table (from the manifest).
pub struct EngramConfig {
    pub version: String,
    pub embedding_dim: usize,
    pub dtype_bytes: usize,
    pub num_heads: u32,
    pub shard_size: u32,
    pub num_shards: u64,
    pub hash_seeds: Vec<u64>,
}

/// Result of hashing an N-gram — which shards to fetch and where to read.
pub struct EngramLookup {
    pub shard_indices: Vec<u64>,
    pub entry_offsets: Vec<usize>,  // byte offsets within each shard
}

/// Client for performing Engram lookups.
pub struct EngramClient {
    config: EngramConfig,
    manifest_cids: Vec<[u8; 32]>,  // shard index → CID
}
```

### API

```rust
impl EngramClient {
    /// Load from a parsed manifest.
    pub fn from_manifest(config: EngramConfig, shard_cids: Vec<[u8; 32]>) -> Self;

    /// Compute which shards and offsets are needed for an N-gram.
    /// Pure computation — no I/O.
    pub fn lookup(&self, ngram_tokens: &[u32]) -> EngramLookup;

    /// Get the CID for a shard index.
    pub fn shard_cid(&self, shard_index: u64) -> Option<&[u8; 32]>;

    /// Extract and aggregate embedding vectors from fetched shard bytes.
    /// Returns the combined embedding (sum of all heads).
    pub fn resolve(&self, lookup: &EngramLookup, shard_data: &[&[u8]]) -> Vec<u8>;
}
```

The caller's responsibility: take the `EngramLookup`, fetch each shard by CID from the network, pass the bytes to `resolve()`. This keeps the crate sans-I/O.

### Tokenizer Compression

Internal to `lookup()`. Normalizes token IDs by:
- Lowercasing (mapping uppercase variants to lowercase canonical forms)
- Unicode normalization (collapsing equivalent representations)
- Vocabulary compression (~23% reduction per DeepSeek's findings)

The compressed token sequence is then windowed into N-grams for hashing.

### Multi-Head Hashing

Internal to `lookup()`. For each N-gram window:
- Compute `xxhash64(ngram_bytes, seed_i)` for each head `i`
- Map each hash to a table index: `hash_value % total_entries`
- Compute shard index and entry offset from the table index

Deterministic across platforms — xxhash64 is a well-specified algorithm with identical results on all architectures.

## Versioning

- Version prefix in Zenoh key expressions: `harmony/engram/v1/`, `harmony/engram/v2/`
- Multiple versions coexist on the network
- Each version has its own manifest with its own shard CIDs
- Edge nodes request from the version matching their loaded model
- Old versions naturally age out of cache and tier to S3 Glacier via Intelligent-Tiering

## Integration with Harmony Node

The event loop integration follows the same pattern as the S3 archivist and rawlink bridge:

- **Config:** `engram: Option<EngramConfig>` with version, manifest CID
- **Startup:** Fetch the manifest DAG, build the `EngramClient`
- **Runtime:** Model inference code calls `client.lookup()`, the event loop fetches shards via Zenoh, `client.resolve()` returns vectors
- **Caching:** Fetched shards are stored in the local ContentStore (W-TinyLFU cache) as normal CAS books

No changes to the Engram query path are needed in the Reticulum or content layers — shards are standard books fetched through standard Zenoh content queries.

## Testing Strategy

### Unit tests (harmony-engram)

- Hash determinism: same N-gram + seeds → same shard indices across runs
- Shard resolution: boundary cases (first/last entry, shard boundaries)
- Vector extraction: known shard bytes → correct embedding slice
- Multi-head aggregation: 4 vectors → expected sum
- Manifest parsing: serialized manifest → correct config + shard CIDs

### Integration tests (harmony-node, mock)

- Lookup round-trip with MockBookStore (small table, 5 shards)
- Cache hit on repeated lookup
- Missing shard handled gracefully (skip that head)

### No model tests

Gating is model-specific. The crate's contract ends at raw embedding vectors.

## Scope Exclusions

- **PyTorch checkpoint ingestion** — separate bead (harmony-8wa)
- **Context-aware gating** — model computation, not protocol
- **Training/fine-tuning** — out of scope entirely
- **Tokenizer implementation** — first pass uses configurable token normalization rules; full DeepSeek tokenizer integration is future work
- **Manifest creation tooling** — part of the ingestion pipeline (harmony-8wa)
