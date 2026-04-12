# harmony-db Phase 2 Design — Native Prolly Tree Engine

## Goal

Replace harmony-db's Phase 1 internals (sorted in-memory pages with JSON
persistence) with a native Prolly Tree engine. This gives the database
history-independent tree topology, O(d) diffing proportional to changes rather
than total dataset size, and structural sharing via CAS — identical subsets
produce identical CIDs and are stored once.

The public API is unchanged. Phase 2 is a drop-in internal replacement.

## Background

Phase 1 (ZEB-97, PR #215) proved out the harmony-db API with sorted `Vec<Entry>`
tables and JSON persistence. It works correctly but has O(n) characteristics —
the full index is in memory, every mutation rewrites the entire index.json, and
diffing loads both complete page blobs.

The Gemini research document on CAS database primitives (April 2026) evaluated
Prolly Trees, Merkle Search Trees, CRDT-backed Merkle DAGs, and IPLD ADLs.
Prolly Trees emerged as the optimal long-term primitive for harmony-db due to:
- History independence via CDF-based content-defined chunking
- O(d) diffing (proportional to diff size, not dataset size)
- No hash-mining DoS vulnerability (unlike MSTs)
- Better range scan performance than MSTs
- Key-only boundary hashing prevents cascading on value updates

## Scope

**In scope:**
- CDF chunker with BLAKE3 key-only hashing
- LeafNode and BranchNode types with postcard serialization
- ProllyTree struct with insert, get, range, remove, update_meta
- Recursive O(d) tree diff
- Commit, rebuild_from, open_from_cas using Prolly Tree roots
- Minimal index.json for root tracking (replaces full-index JSON)
- Comprehensive tests including history independence
- Existing public API preserved — all Phase 1 db.rs tests pass

**Out of scope:**
- Zenoh sync / DAG block exchange (separate follow-up)
- Zero-copy binary node encoding (postcard first, optimize later)
- Secondary indexes (future work)
- Phase 1 → Phase 2 data migration tooling (separate follow-up)
- Anchor Nodes (academic optimization, unnecessary at our scale)

## Architecture

```text
HarmonyDb (unchanged public API)
├── prolly/
│   ├── mod.rs        # ProllyTree struct — insert, get, range, remove
│   ├── chunker.rs    # CDF boundary decision (BLAKE3 key-only hash)
│   ├── node.rs       # LeafNode, BranchNode — postcard serialized
│   └── diff.rs       # Recursive O(d) tree diff
├── db.rs             # Delegates to ProllyTree instead of Table
├── persist.rs        # Shrunk: save_roots/load_roots + blob I/O
├── types.rs          # Entry, EntryMeta, Diff, TableDiff — unchanged
├── error.rs          # DbError — unchanged
└── lib.rs            # Public re-exports — unchanged
```

**Deleted:** `table.rs` (replaced by ProllyTree), `commit.rs` (rewritten).

## CDF Chunker

### Algorithm

For each entry being appended to a chunk under construction, compute the
conditional probability that this entry should trigger a boundary:

1. If `current_chunk_size < MIN_SIZE` (512B): P = 0 (never split)
2. If `current_chunk_size + entry_size >= MAX_SIZE` (16KB): P = 1 (force split)
3. Otherwise: `P = (F(s+a) - F(s)) / (1 - F(s))` where F is the CDF of
   Normal(mean=4096, stddev=1024), s = current size, a = entry size

Then: hash the key (ONLY the key) with BLAKE3, extract first 4 bytes as u32,
normalize to [0, 1). If random variable <= P, trigger a boundary.

### Configuration

```rust
pub(crate) struct ChunkerConfig {
    pub target_size: usize,   // 4096
    pub min_size: usize,      // 512
    pub max_size: usize,      // 16384
    pub std_dev: f64,         // 1024.0
}
```

### Key-only hashing guarantee

The boundary decision hashes ONLY the key bytes. This ensures:
- **Value updates**: same key = same hash = same boundary = no cascading. Only
  the modified leaf + path to root gets rewritten.
- **Metadata updates**: same key = same boundary. Vertical cascade only.
- **Insert/delete**: new or removed key may shift boundaries, but CDF probability
  mass prevents cascading beyond the immediate neighborhood.

### CDF computation

For v1, direct computation via logistic approximation of the normal CDF. The hot
path is dominated by the BLAKE3 hash anyway. Precomputed lookup tables can be
added later if profiling shows the CDF computation matters.

## Node Types

### LeafNode

Contains sorted key-value entries — the actual database data.

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct LeafNode {
    pub entries: Vec<LeafEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub(crate) struct LeafEntry {
    pub key: Vec<u8>,
    pub value_cid: ContentId,
    pub timestamp: u64,
    pub metadata: EntryMeta,
}
```

LeafEntry mirrors the public `Entry` type. ProllyTree converts between them at
the API boundary.

### BranchNode

Contains routing entries — child CID + boundary key (highest key in subtree).

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct BranchNode {
    pub children: Vec<BranchEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct BranchEntry {
    pub boundary_key: Vec<u8>,
    pub child_cid: ContentId,
}
```

### Serialization

A `Node` enum wraps both types for serialization:

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) enum Node {
    Leaf(LeafNode),
    Branch(BranchNode),
}
```

Serialized via postcard (already a workspace dependency, no-std compatible, fast
binary encoding). After serialization, `ContentId::for_book(bytes)` produces the
CID. Node bytes stored in `commits/{cid_hex}.bin` (same directory as Phase 1
commit blobs).

### Fanout estimates (at ~200 byte entries, 4KB target chunks)

- Leaf: ~20 entries per node
- Branch: ~64 entries per node (32B key + 32B CID = 64B per entry)
- Tree depth: 3 levels = ~80K entries, 4 levels = ~5M entries

## ProllyTree Operations

### Struct

```rust
pub(crate) struct ProllyTree {
    root: Option<ContentId>,
    config: ChunkerConfig,
}
```

The tree is stateless beyond the root CID. All data lives in CAS blobs.
Operations read nodes from disk, mutate, write new nodes, and return a new root
CID. The old root remains valid and accessible (immutable CAS).

### Get — O(log n)

Walk root → branch (binary search on boundary keys) → leaf (binary search on
keys). ~3-4 CAS reads for a million-entry database.

### Range — O(log n + k)

Find the leaf containing `start`, iterate leaves forward via the parent branch
structure until passing `end`. Returns collected entries.

### Insert — O(log n)

1. Walk to the target leaf
2. Insert entry in sorted position
3. Run CDF chunker over the modified leaf's entry sequence
4. No split: serialize new leaf, write to CAS, get new CID
5. Split: produce two leaves, write both, get two CIDs
6. Propagate new CID(s) up to parent branch — parent may also rechunk
7. Cascade to root, return new root CID

Average cost: ~1 node rewrite per tree level = ~4-5 nodes for typical depth.

### Remove — O(log n)

Same descent, excise the entry, rechunk the leaf. If leaf shrinks below min size,
merge with sibling and rechunk. Cascade up.

### Update metadata — O(log n), no rechunking

Key stays the same → boundary stays the same → only leaf content changes. Walk to
leaf, update metadata, write new leaf, cascade CID changes up to root. No
rechunking needed.

### Mutation cascade pattern

Every mutation follows: descend → mutate → rechunk → ascend → return new root.
Immutable: every operation produces a new root CID. Old state remains accessible.

## Diff — Recursive O(d)

### Algorithm

Given two tree roots:

1. Root CIDs equal → empty diff. O(1).
2. Load both root nodes from CAS.
3. Both are leaves → two-pointer merge on sorted entries (same as Phase 1).
4. Both are branches → two-pointer walk over boundary keys:
   - Key + child CID match → **skip entire subtree** (the O(d) win)
   - Keys align, CIDs differ → **recurse** into both children
   - Key only in old → subtree removed, traverse and emit all as `removed`
   - Key only in new → subtree added, traverse and emit all as `added`
5. Depth mismatch (leaf vs branch) → flatten both to entry lists, two-pointer
   merge.

### Change detection

Same as Phase 1: entries are "changed" when same key but different `value_cid` or
different `metadata`. Timestamp is excluded from change detection (it records
insertion time, not semantic state).

### Integration with HarmonyDb::diff()

The public `diff(old, new, store)` method loads commit manifests for both CIDs,
then for each table calls `ProllyTree::diff(root_a, root_b)`. The commit
manifest format stays the same — a BTreeMap of table names to CIDs. In Phase 1
those were page CIDs; now they're tree root CIDs. Same structure.

### Performance example

100K entries, 5000 leaf nodes, 3-level tree. 10 entries changed:
- Phase 1: load 2 full pages (~200KB each), O(n) merge = O(100K)
- Prolly Tree: ~3 branch reads + ~10 leaf reads ≈ 50KB, O(d) = O(10)

## Persistence

### Minimal root tracking

`persist.rs` keeps blob I/O (`write_blob`, `read_blob`, `write_blob_raw`,
`ensure_dirs`) unchanged. The index tracking shrinks to:

```json
{
  "version": 2,
  "head": "optional_commit_cid_hex",
  "table_roots": {
    "inbox": "tree_root_cid_hex",
    "sent": "tree_root_cid_hex"
  }
}
```

~200 bytes regardless of database size (vs Phase 1's index.json which stored
every entry). Saved after every mutation via atomic tmp+rename.

### Commit

Simpler than Phase 1:

1. Build root manifest: `{ version, parent, tables: { name → tree_root_cid } }`
2. Serialize, hash → commit CID, write to `commits/`
3. Optionally push to BookStore: walk each tree, push any nodes the store
   doesn't have, push manifest last (atomic visibility)

No need to "serialize tables to page blobs" because tables ARE already CAS blobs.
Commit just snapshots which root CIDs correspond to which table names.

### Rebuild

1. Load commit manifest (local or BookStore, with CID verification)
2. Extract table name → tree root CID mappings
3. For each table: tree is immediately walkable from root CID. Optionally prefetch
   all nodes from BookStore to local cache.
4. Save roots to minimal index file

No in-memory index reconstruction needed — the tree IS the index.

## Files Changed

| File | Action | Notes |
|------|--------|-------|
| `src/prolly/mod.rs` | **New** | ProllyTree struct, insert/get/range/remove/update_meta |
| `src/prolly/chunker.rs` | **New** | CDF boundary decision, ChunkerConfig |
| `src/prolly/node.rs` | **New** | LeafNode, BranchNode, Node enum, CAS read/write |
| `src/prolly/diff.rs` | **New** | Recursive O(d) tree diff |
| `src/db.rs` | **Modified** | Switch from Table+commit to ProllyTree |
| `src/persist.rs` | **Modified** | Replace save_index/load_index with save_roots/load_roots |
| `src/commit.rs` | **Deleted** | Replaced by prolly/diff.rs + simplified commit in db.rs |
| `src/table.rs` | **Deleted** | Replaced by prolly/mod.rs |
| `src/types.rs` | Unchanged | Entry, EntryMeta, Diff, TableDiff |
| `src/error.rs` | Unchanged | DbError |
| `src/lib.rs` | **Modified** | Replace `mod table; mod commit;` with `mod prolly;` |
| `Cargo.toml` | **Modified** | Add `postcard` dependency |
| `examples/mail_workflow.rs` | Unchanged | Uses only public API |

## Dependencies

Add to harmony-db's Cargo.toml:
- `postcard = { workspace = true }` — node serialization (already in workspace)
- `blake3 = { workspace = true }` — direct key hashing for CDF chunker (already
  in workspace; harmony-content uses it internally but doesn't re-export the
  hasher)

`ContentId::for_book()` is still used for computing node CIDs after serialization.
The direct `blake3` dep is specifically for the chunker's key-only boundary hash.

## Testing Strategy

### Chunker tests

| Test | Validates |
|------|-----------|
| `boundary_below_min_never_splits` | No split when chunk < 512B |
| `boundary_above_max_always_splits` | Forced split at 16KB |
| `boundary_deterministic` | Same key → same decision at same chunk size |
| `boundary_key_only` | Different values, same key → same decision |
| `chunk_size_distribution` | 1000 random keys produce chunks centered ~4KB |

### Node tests

| Test | Validates |
|------|-----------|
| `leaf_serialize_round_trip` | postcard encode → decode preserves entries |
| `branch_serialize_round_trip` | postcard encode → decode preserves children |
| `node_cas_round_trip` | Serialize → for_book → store → load → deserialize |

### Tree operation tests

| Test | Validates |
|------|-----------|
| `insert_and_get` | Insert KV, get returns it |
| `insert_sorted_order` | Range over all returns sorted |
| `insert_upsert` | Same key replaces entry |
| `remove_entry` | Remove returns entry, get returns None |
| `update_meta_no_rechunk` | Metadata update doesn't change tree structure |
| `range_query` | Range returns correct subset |
| `large_insert_builds_tree` | 1000 entries, depth > 1, all retrievable |
| `history_independence` | Same entries in different order → same root CID |
| `value_update_stability` | Value update only changes path to root |

### Diff tests

| Test | Validates |
|------|-----------|
| `diff_identical_roots` | Same root → empty diff |
| `diff_additions` | New entries detected |
| `diff_removals` | Removed entries detected |
| `diff_changes` | Value/metadata changes detected |
| `diff_subtree_skip` | Unchanged subtrees not loaded |
| `diff_large_dataset_small_change` | 1000 entries, change 1, ~4 nodes read |

### Regression suite

All existing `db.rs` integration tests (50 tests from Phase 1) must pass
unchanged with the new engine. The `mail_workflow` example must pass unchanged.

## Future: Phase 3 Considerations

- **Zero-copy node encoding**: Replace postcard with trailing-offset binary
  format if profiling shows node deserialization is a bottleneck
- **Zenoh DAG sync**: Broadcast root CID, peers request missing blocks,
  Prolly Tree maps to Zenoh's era-based alignment protocol
- **Secondary indexes**: Separate Prolly Trees mapping index values to primary
  keys
- **Anchor Nodes**: Academic optimization for strict O(2H) mutation bounds if
  cascading becomes a problem at scale
