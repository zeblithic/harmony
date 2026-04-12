# harmony-oluo Redesign: USearch-Backed Semantic Search (ZEB-105, Phase 3)

## Problem

harmony-oluo currently uses a flat `HashMap<CID, IndexEntry>` with brute-force Hamming distance scanning for semantic search. This is O(N) per query. The trie structure intended to replace it was never implemented (stubbed out, dead code). With CompoundIndex now available from harmony-search, oluo can use HNSW-backed ANN search for orders-of-magnitude faster queries at scale.

## Solution

Replace oluo's flat HashMap + brute-force scan with `CompoundIndex` from harmony-search. Keep the sans-I/O event/action pattern, privacy gating, metadata flow, and search scopes. Metadata moves to a side-table; geometry lives in the index. Drop all trie-related code.

## OluoEngine internal state

```rust
struct OluoEngine {
    index: CompoundIndex,                              // vector geometry
    metadata: HashMap<u64, (SidecarMetadata, [u8; 32])>, // key → (metadata, target_cid)
    key_counter: u64,                                  // monotonic key assignment
    threshold: u32,                                    // Hamming distance threshold (compat)
}
```

Key mapping: CompoundIndex uses u64 keys. Each ingest assigns `key_counter` (monotonic), metadata table maps that u64 back to CID + metadata.

## Event/action changes

### Events (minimal API change)

- `Ingest { header, metadata, decision, now_ms }` — unchanged signature. Internally: assign key, unpack tier3 to f32, `index.add()`, metadata insert. If `should_compact()`, compact and emit bytes.
- `Search { query_id, query }` — unchanged signature. Internally: unpack query embedding to f32, `index.search()`, look up metadata for results.
- `EvictExpired { now_ms }` — iterate metadata table, remove expired entries from index + metadata.
- **New:** `CompactComplete { path: String }` — caller signals CAS storage done, engine calls `index.load_base(path)`.
- **Removed:** `SyncReceived` (trie, never implemented).

### Actions

- `SearchResults { query_id, results }` — unchanged.
- `IndexUpdated` — simplified, no trie root.
- **New:** `CompactRequest { bytes: Vec<u8> }` — engine provides compacted bytes for caller to CAS-store.
- **Removed:** `FetchTrieNode`, `FetchSidecar`, `PublishTrieRoot`, `PersistBlob` (all trie-related).

## Ingest flow

1. Check `IngestGate` decision (unchanged)
2. Check privacy tier — `EncryptedEphemeral` blocked (unchanged)
3. Assign key = `self.key_counter`; increment
4. Extract `header.tier3` ([u8; 32] = 256 bits)
5. Unpack to `[f32; 256]` (0.0/1.0 per bit, MSB-first matching harmony-semantic)
6. `index.add(key, &f32_vector)`
7. `metadata.insert(key, (metadata, header.target_cid))`
8. If `index.should_compact()`: `compact()` and emit `CompactRequest { bytes }`
9. Emit `IndexUpdated`

## Search flow

1. Unpack `query.embedding` ([u8; 32]) to `[f32; 256]` (MSB-first)
2. `results = index.search(&f32_vector, query.max_results)`
3. For each `Match { key, distance }`: look up `metadata[key]` → `(sidecar_metadata, target_cid)`, normalize score = `distance / 256.0`
4. Emit `SearchResults { query_id, results }`

## CompoundIndex configuration

```rust
VectorIndexConfig {
    dimensions: 256,           // 256-bit tier3 → 256 f32 dimensions
    metric: Metric::Hamming,   // Hamming distance on 0.0/1.0 floats
    quantization: Quantization::F32,  // f32 storage (not B1, see harmony-search docs)
    capacity: 10_000,          // initial capacity
    connectivity: 16,          // HNSW M parameter
    expansion_add: 128,        // ef_construction
    expansion_search: 64,      // ef at query time
}
```

Compact threshold: configurable, default 1000 (compact every ~1000 ingests).

## Files removed

- `trie.rs` (211 lines) — `TrieNode`, `get_bit()`, encode/decode
- `search.rs` (127 lines) — `scan_collection()`, `SearchHit`
- `ranking.rs` (51 lines) — `normalize_score()`
- `zenoh_keys.rs` (8 lines) — trie sync constants

## Files modified

- `engine.rs` — replace internals with CompoundIndex + metadata table
- `error.rs` — add search error variant
- `lib.rs` — update module declarations, remove trie/search/ranking re-exports

## Files unchanged

- `filter.rs` — `RetrievalFilter` trait
- `scope.rs` — `SearchQuery`, `SearchScope`
- `ingest.rs` — `IngestGate`, `IngestDecision`

## New dependency

`harmony-search` added to `harmony-oluo/Cargo.toml`.

## Net effect

~400 lines removed, ~150 lines added/modified. Oluo becomes simpler, faster, and ready for CAS integration.

## Success criteria

- All existing test behaviors preserved (ingest, search ranking, privacy filtering, eviction)
- Search uses HNSW instead of brute-force
- Compaction produces bytes for CAS storage
- CompactComplete loads memory-mapped base
- Trie code completely removed
- harmony-node event loop integration unchanged (minimal API surface change)
