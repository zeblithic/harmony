# harmony-oluo Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace harmony-oluo's flat HashMap + brute-force Hamming scan with CompoundIndex from harmony-search, keeping the sans-I/O event/action pattern.

**Architecture:** OluoEngine's internal state changes from `HashMap<CID, IndexEntry>` to `CompoundIndex` (for vector geometry) + `HashMap<u64, EntryMetadata>` (for metadata side-table). Events/actions get minimal API changes: trie variants removed, compact variants added. ~400 lines removed, ~150 added.

**Tech Stack:** Rust, harmony-search (CompoundIndex, VectorIndex), harmony-semantic (SidecarHeader, SidecarMetadata)

**Spec:** `docs/superpowers/specs/2026-04-12-oluo-redesign-design.md`

---

### Task 1: Remove dead code and update dependencies

**Files:**
- Delete: `crates/harmony-oluo/src/trie.rs`
- Delete: `crates/harmony-oluo/src/search.rs`
- Delete: `crates/harmony-oluo/src/ranking.rs`
- Delete: `crates/harmony-oluo/src/zenoh_keys.rs`
- Modify: `crates/harmony-oluo/src/lib.rs`
- Modify: `crates/harmony-oluo/Cargo.toml`

Remove the four files that are being replaced (trie, brute-force search, ranking, zenoh keys). Update lib.rs to remove the module declarations and re-exports. Add `harmony-search` dependency. Switch from `no_std` to `std` (no consumers use `no_std` currently).

- [ ] **Step 1: Delete the four files**

```bash
rm crates/harmony-oluo/src/trie.rs
rm crates/harmony-oluo/src/search.rs
rm crates/harmony-oluo/src/ranking.rs
rm crates/harmony-oluo/src/zenoh_keys.rs
```

- [ ] **Step 2: Update lib.rs**

Replace the contents of `crates/harmony-oluo/src/lib.rs` with:

```rust
// SPDX-License-Identifier: Apache-2.0 OR MIT
//! Oluo — Harmony's semantic search engine.
//!
//! Uses USearch HNSW (via harmony-search) for approximate nearest-neighbor
//! search on binary embeddings with a sans-I/O state machine design.

pub mod engine;
pub mod error;
pub mod filter;
pub mod ingest;
pub mod scope;

pub use engine::{OluoAction, OluoEngine, OluoEvent};
pub use error::{OluoError, OluoResult};
pub use filter::{FilteredSearchResult, RawSearchResult, RetrievalContext, RetrievalFilter};
pub use ingest::{IngestDecision, IngestGate};
pub use scope::{SearchQuery, SearchScope};
```

- [ ] **Step 3: Update Cargo.toml**

Replace `crates/harmony-oluo/Cargo.toml` with:

```toml
[package]
name = "harmony-oluo"
version.workspace = true
edition.workspace = true
license.workspace = true
repository.workspace = true
rust-version.workspace = true
description = "Oluo — Harmony's semantic search engine (USearch HNSW, sans-I/O)"

[features]
default = ["std"]
no-neon = ["harmony-crypto/no-neon"]
std = [
    "harmony-semantic/std",
    "harmony-crypto/std",
]

[dependencies]
harmony-semantic = { workspace = true, default-features = false }
harmony-crypto = { workspace = true, default-features = false }
harmony-search = { path = "../harmony-search" }
hashbrown = { workspace = true }

[dev-dependencies]
```

- [ ] **Step 4: Verify it compiles (with expected errors from engine.rs)**

Run: `cargo check -p harmony-oluo 2>&1 | head -20`
Expected: Errors from engine.rs referencing removed modules (hamming_distance, etc.) — that's Task 2.

- [ ] **Step 5: Commit**

```bash
git add -A crates/harmony-oluo/
git commit -m "refactor: remove trie, brute-force search, ranking from harmony-oluo"
```

---

### Task 2: Rewrite OluoEngine with CompoundIndex

**Files:**
- Modify: `crates/harmony-oluo/src/engine.rs`

This is the core change. Replace OluoEngine's internals with CompoundIndex + metadata side-table. Update events/actions. Rewrite handle_ingest and handle_search. All existing test behaviors must be preserved.

- [ ] **Step 1: Rewrite engine.rs**

Replace the entire contents of `crates/harmony-oluo/src/engine.rs` with the new implementation. The key changes are:

1. **Imports:** Replace `hamming_distance` with `harmony_search::{CompoundIndex, VectorIndexConfig, Metric, Quantization, Match}`
2. **OluoEvent:** Remove `SyncReceived`, add `CompactComplete { path: String }`
3. **OluoAction:** Remove `FetchTrieNode`, `FetchSidecar`, `PublishTrieRoot`, `PersistBlob`. Add `CompactRequest { bytes: Vec<u8> }`. Change `IndexUpdated` to have no fields.
4. **OluoEngine struct:** Replace `entries: HashMap<CID, IndexEntry>` with `index: CompoundIndex`, `metadata: HashMap<u64, EntryMetadata>`, `key_counter: u64`
5. **EntryMetadata:** New struct holding `target_cid`, `metadata`, `expires_at`, `ingested_at_ms`
6. **handle_ingest:** Assign key from counter, unpack tier3 bits to f32 MSB-first, `index.add()`, metadata insert, check `should_compact()`
7. **handle_search:** Unpack query embedding to f32, `index.search()`, look up metadata, normalize scores
8. **handle_evict:** Iterate metadata for expired entries, `index.remove()` each, clean metadata
9. **handle_compact_complete:** `index.load_base(path)`
10. **Bit unpacking helper:** `fn unpack_tier3_to_f32(tier3: &[u8; 32]) -> [f32; 256]` — MSB-first, matching harmony-semantic convention

The new engine must:
- Create a `CompoundIndex` with `VectorIndexConfig { dimensions: 256, metric: Metric::Hamming, quantization: Quantization::F32, capacity: 10_000, connectivity: 16, expansion_add: 128, expansion_search: 64 }` and `compact_threshold: 1000`
- Use `with_compact_threshold(threshold)` constructor for custom thresholds
- Track `key_counter: u64` for monotonic key assignment
- Store metadata in `hashbrown::HashMap<u64, EntryMetadata>` where `EntryMetadata` has `target_cid: [u8; 32]`, `metadata: SidecarMetadata`, `expires_at: Option<u64>`, `ingested_at_ms: u64`

**Important:** The existing tests use exact score comparisons (0.0, 0.5, 1.0). USearch's Hamming distance on unpacked f32 bits should produce the same ranking order but the raw distance values will be different (USearch computes distance differently than the pure `hamming_distance` function). Adjust score normalization in the tests or use ordering-based assertions.

- [ ] **Step 2: Rewrite the tests**

Preserve all existing test behaviors but adapt to the new API:
- `engine_ingest_stores_entry` — ingest and verify `entry_count()` increments
- `engine_ingest_reject_blocks` — reject decision blocks ingest
- `engine_ingest_privacy_tier_3_blocked` — EncryptedEphemeral blocked
- `engine_search_non_t3_returns_empty` — non-T3 queries return empty (keep this behavior)
- `engine_search_empty_returns_empty` — empty index returns empty results
- `engine_search_returns_nearest` — ranking order preserved (verify A before B before C, use ordering-based assertions rather than exact score values since USearch Hamming distance may differ from pure hamming_distance)
- `engine_search_includes_logically_expired_without_eviction` — expired entries visible until eviction
- `engine_evict_removes_expired` — TTL eviction with boundary cases
- **New:** `engine_compact_request` — verify CompactRequest emitted when threshold reached
- **New:** `engine_compact_complete` — verify load_base after compact

- [ ] **Step 3: Verify all tests pass**

Run: `cargo test -p harmony-oluo`
Expected: All pass

- [ ] **Step 4: Commit**

```bash
git add crates/harmony-oluo/src/engine.rs
git commit -m "feat: rewrite OluoEngine with CompoundIndex + metadata side-table"
```

---

### Task 3: Update harmony-node integration (if needed)

**Files:**
- Check: `crates/harmony-node/src/event_loop.rs`

Check if harmony-node references the removed event/action variants. If so, update. If not (likely, since trie was never wired), this task is a no-op.

- [ ] **Step 1: Check for references to removed variants**

```bash
grep -rn "SyncReceived\|FetchTrieNode\|FetchSidecar\|PublishTrieRoot\|PersistBlob\|normalize_score\|scan_collection\|SearchHit\|TrieNode\|get_bit" crates/harmony-node/src/
```

If no matches: skip to commit. If matches: update the event loop to remove/replace them.

- [ ] **Step 2: Check for references in other crates**

```bash
grep -rn "harmony_oluo" crates/ --include="*.rs" | grep -v "harmony-oluo/src"
```

Update any external references.

- [ ] **Step 3: Full workspace check**

Run: `cargo check -p harmony-oluo -p harmony-node 2>&1 | tail -20`
Expected: Clean compilation

- [ ] **Step 4: Commit (if changes were needed)**

```bash
git add crates/harmony-node/
git commit -m "fix: update harmony-node for oluo API changes"
```

---

### Task 4: Push and create PR

**Files:** None (git operations only)

- [ ] **Step 1: Run full test suite**

Run: `cargo test -p harmony-oluo -p harmony-search`
Expected: All pass

- [ ] **Step 2: Push and create PR**

```bash
git push -u origin HEAD
gh pr create --title "feat: harmony-oluo redesign — USearch HNSW replaces brute-force scan (ZEB-105 Phase 3)" \
  --body "$(cat <<'EOF'
## Summary

- Replace flat HashMap + brute-force Hamming scan with CompoundIndex (HNSW)
- Metadata moves to side-table, geometry in the index
- Drop all trie code (never implemented), ranking, brute-force search
- Add CompactRequest/CompactComplete for CAS integration
- Keep sans-I/O event/action pattern, privacy gating, search scopes
- ~400 lines removed, ~150 added

### What changes

| Before | After |
|--------|-------|
| O(N) brute-force Hamming scan | O(log N) HNSW approximate search |
| HashMap<CID, IndexEntry> | CompoundIndex + HashMap<u64, Metadata> |
| Unfinished trie structure | Removed entirely |
| No compaction | CompactRequest/CompactComplete for CAS |

## Test plan

- [ ] All existing test behaviors preserved
- [ ] Ingest/search/eviction work with CompoundIndex
- [ ] CompactRequest emitted at threshold
- [ ] CompactComplete loads base

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```
