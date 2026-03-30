# MemoStore Disk Persistence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add disk persistence and LFU eviction to MemoStore so memos survive node restarts and bounded storage is managed automatically.

**Architecture:** Flat files following the existing `disk_io.rs` pattern (write-to-temp-then-rename, hex prefix sharding). LFU counters tracked in-memory and periodically flushed to a single binary file. Event loop gains startup loading, persist-on-insert, and periodic LFU flush.

**Tech Stack:** Rust, postcard serialization (existing), tokio::task::spawn_blocking (existing pattern), tempfile (existing dev-dep)

**Spec:** `docs/superpowers/specs/2026-03-30-memostore-disk-persistence-design.md`

---

## File Structure

**Create:**
- `crates/harmony-node/src/memo_io.rs` — Disk I/O for memos (mirrors disk_io.rs)

**Modify:**
- `crates/harmony-memo/src/store.rs` — Add LFU counters, evict_lfu(), LFU serialization
- `crates/harmony-node/src/main.rs` — Add `mod memo_io;` declaration
- `crates/harmony-node/src/event_loop.rs` — Startup loading, persist-on-insert, periodic LFU flush

---

### Task 1: memo_io.rs — Disk I/O Module

**Files:**
- Create: `crates/harmony-node/src/memo_io.rs`
- Modify: `crates/harmony-node/src/main.rs` (add `mod memo_io;`)

- [ ] **Step 1: Write the failing test for write/read roundtrip**

Create `crates/harmony-node/src/memo_io.rs` with just the test module:

```rust
//! Synchronous disk I/O helpers for memo persistence.
//!
//! Mirrors `disk_io.rs` for CAS books. Called inside `tokio::task::spawn_blocking`.
//!
//! File layout: `{data_dir}/memo/{input_hex[8..10]}/{input_hex}_{output_hex}_{signer_hex}`

use harmony_content::cid::ContentId;
use harmony_memo::Memo;
use std::{
    fs,
    io::{self, Write as _},
    path::{Path, PathBuf},
};

#[cfg(test)]
mod tests {
    use super::*;
    use harmony_identity::pq_identity::PqPrivateIdentity;
    use harmony_memo::create::create_memo;

    fn make_cid(byte: u8) -> ContentId {
        ContentId::from_bytes([byte; 32])
    }

    fn make_test_memo(input_byte: u8, output_byte: u8) -> Memo {
        let identity = PqPrivateIdentity::generate(&mut rand::rngs::OsRng);
        create_memo(
            make_cid(input_byte),
            make_cid(output_byte),
            &identity,
            &mut rand::rngs::OsRng,
            1000,
            9999,
        )
        .expect("create_memo")
    }

    #[test]
    fn write_and_read_roundtrip() {
        let dir = tempfile::TempDir::new().unwrap();
        let memo = make_test_memo(0x01, 0x02);

        write_memo(dir.path(), &memo).expect("write should succeed");
        let path = memo_path(dir.path(), &memo);
        let read_back = read_memo(&path).expect("read should succeed");

        assert_eq!(read_back.input, memo.input);
        assert_eq!(read_back.output, memo.output);
        assert_eq!(
            read_back.credential.issuer.hash,
            memo.credential.issuer.hash
        );
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p harmony-node memo_io::tests::write_and_read_roundtrip`
Expected: FAIL — functions `write_memo`, `memo_path`, `read_memo` don't exist

- [ ] **Step 3: Implement memo_path, write_memo, read_memo**

Add above the test module in `memo_io.rs`:

```rust
/// Derive the signer hash (16 bytes = 32 hex chars) from a memo's credential issuer.
fn signer_hex(memo: &Memo) -> String {
    hex::encode(memo.credential.issuer.hash)
}

/// Return the filesystem path for a memo under `data_dir`.
///
/// Layout: `{data_dir}/memo/{input_hex[8..10]}/{input_hex}_{output_hex}_{signer_hex}`
pub fn memo_path(data_dir: &Path, memo: &Memo) -> PathBuf {
    let input_hex = hex::encode(memo.input.to_bytes());
    let output_hex = hex::encode(memo.output.to_bytes());
    let signer = signer_hex(memo);
    let prefix = &input_hex[8..10];
    let filename = format!("{}_{}_{}",  input_hex, output_hex, signer);
    data_dir.join("memo").join(prefix).join(filename)
}

/// Write a memo to disk using write-to-temp-then-rename for crash safety.
pub fn write_memo(data_dir: &Path, memo: &Memo) -> Result<u64, io::Error> {
    let bytes = harmony_memo::serialize(memo)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, format!("{e}")))?;
    let path = memo_path(data_dir, memo);
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let tmp_path = path.with_extension("tmp");
    {
        let mut file = fs::File::create(&tmp_path)?;
        file.write_all(&bytes)?;
        file.sync_all()?;
    }
    let size = bytes.len() as u64;
    fs::rename(&tmp_path, &path)?;
    Ok(size)
}

/// Read and deserialize a memo from a file path.
pub fn read_memo(path: &Path) -> Result<Memo, io::Error> {
    let bytes = fs::read(path)?;
    harmony_memo::deserialize(&bytes)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, format!("{e}")))
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p harmony-node memo_io::tests::write_and_read_roundtrip`
Expected: PASS

- [ ] **Step 5: Add delete_memo and scan_memos**

Add after `read_memo`:

```rust
/// Delete a memo file from disk.
pub fn delete_memo(data_dir: &Path, memo: &Memo) -> Result<(), io::Error> {
    fs::remove_file(memo_path(data_dir, memo))
}

/// Walk `{data_dir}/memo/` and return all valid memos with their on-disk sizes.
///
/// Invalid or corrupt files are skipped with warnings.
pub fn scan_memos(data_dir: &Path) -> Vec<(Memo, u64)> {
    let memo_root = data_dir.join("memo");
    let mut memos = Vec::new();

    let prefix_dir = match fs::read_dir(&memo_root) {
        Ok(rd) => rd,
        Err(e) => {
            if e.kind() != io::ErrorKind::NotFound {
                tracing::warn!("scan_memos: cannot read {}: {}", memo_root.display(), e);
            }
            return memos;
        }
    };

    for prefix_entry in prefix_dir {
        let prefix_entry = match prefix_entry {
            Ok(e) => e,
            Err(e) => {
                tracing::warn!("scan_memos: directory entry error: {}", e);
                continue;
            }
        };
        let prefix_path = prefix_entry.path();
        if !prefix_path.is_dir() {
            continue;
        }

        let entries = match fs::read_dir(&prefix_path) {
            Ok(rd) => rd,
            Err(e) => {
                tracing::warn!("scan_memos: cannot read {}: {}", prefix_path.display(), e);
                continue;
            }
        };

        for entry in entries {
            let entry = match entry {
                Ok(e) => e,
                Err(e) => {
                    tracing::warn!("scan_memos: entry error: {}", e);
                    continue;
                }
            };

            let path = entry.path();
            // Skip temp files from interrupted writes
            if path.extension().map(|e| e == "tmp").unwrap_or(false) {
                tracing::debug!("scan_memos: skipping temp file: {}", path.display());
                continue;
            }

            match read_memo(&path) {
                Ok(memo) => {
                    let size = entry.metadata().map(|m| m.len()).unwrap_or(0);
                    memos.push((memo, size));
                }
                Err(e) => {
                    tracing::warn!("scan_memos: skipping {}: {}", path.display(), e);
                }
            }
        }
    }

    memos
}
```

- [ ] **Step 6: Write remaining tests**

Add to the test module:

```rust
    #[test]
    fn scan_discovers_written_memos() {
        let dir = tempfile::TempDir::new().unwrap();
        let m1 = make_test_memo(0x01, 0x02);
        let m2 = make_test_memo(0x03, 0x04);
        let m3 = make_test_memo(0x05, 0x06);

        write_memo(dir.path(), &m1).unwrap();
        write_memo(dir.path(), &m2).unwrap();
        write_memo(dir.path(), &m3).unwrap();

        let found = scan_memos(dir.path());
        assert_eq!(found.len(), 3, "should discover all 3 memos");
    }

    #[test]
    fn delete_removes_file() {
        let dir = tempfile::TempDir::new().unwrap();
        let memo = make_test_memo(0x01, 0x02);

        write_memo(dir.path(), &memo).unwrap();
        assert!(memo_path(dir.path(), &memo).exists());

        delete_memo(dir.path(), &memo).unwrap();
        assert!(!memo_path(dir.path(), &memo).exists());
    }

    #[test]
    fn scan_skips_invalid_files() {
        let dir = tempfile::TempDir::new().unwrap();
        let memo = make_test_memo(0x01, 0x02);
        write_memo(dir.path(), &memo).unwrap();

        // Inject a garbage file in the memo prefix dir
        let prefix_dir = memo_path(dir.path(), &memo).parent().unwrap().to_path_buf();
        fs::write(prefix_dir.join("garbage.dat"), b"not a memo").unwrap();

        let found = scan_memos(dir.path());
        assert_eq!(found.len(), 1, "should find only the valid memo");
    }

    #[test]
    fn scan_empty_returns_empty() {
        let dir = tempfile::TempDir::new().unwrap();
        let found = scan_memos(dir.path());
        assert!(found.is_empty());
    }

    #[test]
    fn write_returns_byte_size() {
        let dir = tempfile::TempDir::new().unwrap();
        let memo = make_test_memo(0x01, 0x02);
        let size = write_memo(dir.path(), &memo).unwrap();
        assert!(size > 0, "serialized memo should have non-zero size");

        // Verify matches actual file size
        let path = memo_path(dir.path(), &memo);
        let file_size = fs::metadata(&path).unwrap().len();
        assert_eq!(size, file_size);
    }
```

- [ ] **Step 7: Add `mod memo_io;` to main.rs**

In `crates/harmony-node/src/main.rs`, add alongside the existing `mod disk_io;`:

```rust
pub(crate) mod memo_io;
```

- [ ] **Step 8: Run all memo_io tests**

Run: `cargo test -p harmony-node memo_io`
Expected: 5 tests PASS

- [ ] **Step 9: Commit**

```bash
git add crates/harmony-node/src/memo_io.rs crates/harmony-node/src/main.rs
git commit -m "feat(node): add memo_io disk persistence module (mirrors disk_io)"
```

---

### Task 2: MemoStore LFU Extension

**Files:**
- Modify: `crates/harmony-memo/src/store.rs`

- [ ] **Step 1: Write failing test for LFU counter increment**

Add to the existing `tests` module in `store.rs`:

```rust
    #[test]
    fn lfu_counts_increment_on_query() {
        let identity = PqPrivateIdentity::generate(&mut rand::rngs::OsRng);
        let mut store = MemoStore::new();
        let memo = make_memo(&identity, 0x01, 0x02);
        let input = memo.input;

        store.insert(memo);
        assert_eq!(store.lfu_count(&input, &make_cid(0x02), &identity.address_hash()), 0);

        store.get_by_input(&input);
        assert_eq!(store.lfu_count(&input, &make_cid(0x02), &identity.address_hash()), 1);

        store.get_by_input(&input);
        assert_eq!(store.lfu_count(&input, &make_cid(0x02), &identity.address_hash()), 2);
    }
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p harmony-memo lfu_counts_increment`
Expected: FAIL — `lfu_count` method doesn't exist

- [ ] **Step 3: Add LFU tracking to MemoStore**

In `store.rs`, add a new field and modify the struct:

```rust
/// Dedup key for a memo: (input, output, signer_hash).
type MemoKey = (ContentId, ContentId, [u8; 16]);

pub struct MemoStore {
    by_input: HashMap<ContentId, Vec<Memo>>,
    /// Total memo count across all inputs. Maintained on insert for O(1) len/is_empty.
    total: usize,
    /// LFU access counters keyed by (input, output, signer_hash).
    lfu_counts: HashMap<MemoKey, u32>,
}
```

Update `new()`:

```rust
    pub fn new() -> Self {
        Self {
            by_input: HashMap::new(),
            total: 0,
            lfu_counts: HashMap::new(),
        }
    }
```

Update `insert()` — initialize LFU counter to 0 on successful insert:

```rust
        // After entry.push(memo) and self.total += 1:
        self.lfu_counts.insert((memo.input, memo.output, signer_hash), 0);
```

Note: `memo` is moved into `entry.push(memo)` before this line. Capture the key fields before the push:

```rust
    pub fn insert(&mut self, memo: Memo) -> bool {
        let signer_hash = memo.credential.issuer.hash;
        let input = memo.input;
        let output = memo.output;
        let entry = self.by_input.entry(input).or_insert_with(Vec::new);

        let already_present = entry.iter().any(|existing| {
            existing.output == output && existing.credential.issuer.hash == signer_hash
        });

        if already_present {
            return false;
        }

        entry.push(memo);
        self.total += 1;
        self.lfu_counts.insert((input, output, signer_hash), 0);
        true
    }
```

Change `get_by_input` from `&self` to `&mut self` to increment counters:

```rust
    pub fn get_by_input(&mut self, input: &ContentId) -> &[Memo] {
        match self.by_input.get(input) {
            Some(v) => {
                // Increment LFU counters for all returned memos.
                for memo in v {
                    let key = (memo.input, memo.output, memo.credential.issuer.hash);
                    if let Some(count) = self.lfu_counts.get_mut(&key) {
                        *count = count.saturating_add(1);
                    }
                }
                v.as_slice()
            }
            None => &[],
        }
    }
```

Similarly update `get_by_input_and_signer` to `&mut self` and increment:

```rust
    pub fn get_by_input_and_signer(
        &mut self,
        input: &ContentId,
        signer: &[u8; 16],
    ) -> Vec<&Memo> {
        match self.by_input.get(input) {
            Some(memos) => {
                let results: Vec<&Memo> = memos
                    .iter()
                    .filter(|m| &m.credential.issuer.hash == signer)
                    .collect();
                // Increment only the returned memos.
                for memo in &results {
                    let key = (memo.input, memo.output, memo.credential.issuer.hash);
                    if let Some(count) = self.lfu_counts.get_mut(&key) {
                        *count = count.saturating_add(1);
                    }
                }
                results
            }
            None => Vec::new(),
        }
    }
```

Add `lfu_count` query method:

```rust
    /// Return the LFU access count for a specific memo.
    pub fn lfu_count(&self, input: &ContentId, output: &ContentId, signer: &[u8; 16]) -> u32 {
        self.lfu_counts
            .get(&(*input, *output, *signer))
            .copied()
            .unwrap_or(0)
    }
```

- [ ] **Step 4: Fix existing tests that call get_by_input on `&self`**

The existing tests use `store.get_by_input(...)` on a non-mut binding. Change `let store` to `let mut store` in tests that call `get_by_input` or `get_by_input_and_signer`:
- `insert_and_query_by_input`: `let mut store`
- `dedup_same_signer_same_input_output`: `let mut store` (already mut)
- `different_signers_same_input_output_coexist`: `let mut store` (already mut)
- `different_outputs_same_input_coexist`: `let mut store` (already mut)
- `outputs_for_input_groups_correctly`: `let mut store` (already mut) — note: `outputs_for_input` stays `&self` since it's a grouping view, not an access
- `get_by_input_and_signer`: `let mut store` (already mut)

The `insert_and_query_by_input` test binds `store` as non-mut then calls `get_by_input`. Add `mut`.

- [ ] **Step 5: Run tests**

Run: `cargo test -p harmony-memo`
Expected: ALL PASS including the new `lfu_counts_increment_on_query`

- [ ] **Step 6: Write eviction test and implement evict_lfu**

Add test:

```rust
    #[test]
    fn evict_lfu_returns_lowest() {
        let identity = PqPrivateIdentity::generate(&mut rand::rngs::OsRng);
        let mut store = MemoStore::new();

        let m1 = make_memo(&identity, 0x01, 0x02);
        let m2 = make_memo(&identity, 0x03, 0x04);
        let m3 = make_memo(&identity, 0x05, 0x06);
        let input1 = m1.input;
        let input3 = m3.input;

        store.insert(m1);
        store.insert(m2);
        store.insert(m3);

        // Query m1 ten times, m3 once, m2 never
        for _ in 0..10 {
            store.get_by_input(&input1);
        }
        store.get_by_input(&input3);

        // Evict should return m2 (count 0, lowest)
        let evicted = store.evict_lfu().expect("should evict one memo");
        assert_eq!(evicted.input, make_cid(0x03));
        assert_eq!(evicted.output, make_cid(0x04));
        assert_eq!(store.len(), 2);
    }

    #[test]
    fn evict_lfu_empty_returns_none() {
        let mut store = MemoStore::new();
        assert!(store.evict_lfu().is_none());
    }
```

Implement `evict_lfu`:

```rust
    /// Evict the least-frequently-used memo from the store.
    ///
    /// Returns the evicted memo (for offloading to an archivist), or `None` if empty.
    pub fn evict_lfu(&mut self) -> Option<Memo> {
        // Find the key with the minimum access count.
        let min_key = self
            .lfu_counts
            .iter()
            .min_by_key(|(_, count)| **count)
            .map(|(key, _)| *key)?;

        let (input, output, signer_hash) = min_key;

        // Remove from by_input.
        if let Some(memos) = self.by_input.get_mut(&input) {
            if let Some(idx) = memos.iter().position(|m| {
                m.output == output && m.credential.issuer.hash == signer_hash
            }) {
                let evicted = memos.remove(idx);
                self.total -= 1;
                self.lfu_counts.remove(&min_key);

                // Clean up empty input entry.
                if memos.is_empty() {
                    self.by_input.remove(&input);
                }

                return Some(evicted);
            }
        }

        // Shouldn't reach here, but clean up the orphan key.
        self.lfu_counts.remove(&min_key);
        None
    }
```

- [ ] **Step 7: Run all tests**

Run: `cargo test -p harmony-memo`
Expected: ALL PASS

- [ ] **Step 8: Add LFU serialization/deserialization**

Add methods for persisting LFU counts:

```rust
    /// Serialize LFU counts to bytes (postcard format).
    pub fn serialize_lfu_counts(&self) -> Result<Vec<u8>, postcard::Error> {
        let entries: Vec<((ContentId, ContentId, [u8; 16]), u32)> =
            self.lfu_counts.iter().map(|(k, v)| (*k, *v)).collect();
        postcard::to_allocvec(&entries)
    }

    /// Load LFU counts from bytes. Counts for unknown memos are silently discarded.
    pub fn load_lfu_counts(&mut self, data: &[u8]) -> Result<usize, postcard::Error> {
        let entries: Vec<((ContentId, ContentId, [u8; 16]), u32)> =
            postcard::from_bytes(data)?;
        let mut loaded = 0;
        for (key, count) in entries {
            if self.lfu_counts.contains_key(&key) {
                self.lfu_counts.insert(key, count);
                loaded += 1;
            }
        }
        Ok(loaded)
    }
```

Add test:

```rust
    #[test]
    fn lfu_counts_serialize_roundtrip() {
        let identity = PqPrivateIdentity::generate(&mut rand::rngs::OsRng);
        let mut store = MemoStore::new();

        let memo = make_memo(&identity, 0x01, 0x02);
        let input = memo.input;
        store.insert(memo);

        // Bump the counter
        store.get_by_input(&input);
        store.get_by_input(&input);

        let bytes = store.serialize_lfu_counts().unwrap();

        // Create a new store with the same memo, load the counts
        let mut store2 = MemoStore::new();
        let memo2 = make_memo(&identity, 0x01, 0x02);
        store2.insert(memo2);
        let loaded = store2.load_lfu_counts(&bytes).unwrap();
        assert_eq!(loaded, 1);
        assert_eq!(
            store2.lfu_count(&input, &make_cid(0x02), &identity.address_hash()),
            2
        );
    }
```

- [ ] **Step 9: Run all tests**

Run: `cargo test -p harmony-memo`
Expected: ALL PASS

- [ ] **Step 10: Commit**

```bash
git add crates/harmony-memo/src/store.rs
git commit -m "feat(memo): add LFU tracking, eviction, and count serialization to MemoStore"
```

---

### Task 3: Event Loop Integration

**Files:**
- Modify: `crates/harmony-node/src/event_loop.rs`

This task integrates memo disk I/O into the existing event loop. Because the event loop is large (~2000+ lines) and the memo runtime integration is stubbed, the changes are additive — we add loading at startup, persisting on insert, and periodic LFU flushing alongside the existing book I/O patterns.

- [ ] **Step 1: Add memo startup loading**

In `event_loop.rs`, locate the section after `scan_books` (around line 548-570) where existing disk books are loaded. Add memo loading in the same block, following the same pattern:

```rust
    // ── Load persisted memos ───────────────────────────────────────────
    let mut memo_store = harmony_memo::store::MemoStore::new();
    let mut memo_disk_bytes: u64 = 0;

    if let Some(ref dir) = data_dir {
        let dir_clone = dir.clone();
        let memo_entries = tokio::task::spawn_blocking(move || {
            crate::memo_io::scan_memos(&dir_clone)
        })
        .await
        .unwrap_or_else(|e| {
            tracing::warn!("memo scan task panicked: {}", e);
            Vec::new()
        });

        for (memo, size) in &memo_entries {
            if memo_store.insert(memo.clone()) {
                memo_disk_bytes += size;
            }
        }
        tracing::info!("Loaded {} memos from disk ({} bytes)", memo_store.len(), memo_disk_bytes);

        // Load LFU counts if available.
        let lfu_path = dir.join("memo_lfu.bin");
        if lfu_path.exists() {
            match std::fs::read(&lfu_path) {
                Ok(bytes) => match memo_store.load_lfu_counts(&bytes) {
                    Ok(n) => tracing::info!("Loaded {} LFU counters from disk", n),
                    Err(e) => tracing::warn!("Failed to parse memo_lfu.bin: {} — starting with zero counts", e),
                },
                Err(e) => tracing::warn!("Failed to read memo_lfu.bin: {} — starting with zero counts", e),
            }
        }
    }
```

- [ ] **Step 2: Add memo persist-on-insert helper**

Add a helper function at the bottom of the file (near the existing `handle_storage_tier_action` function):

```rust
/// Persist a memo to disk and track its size. Returns the written size.
fn persist_memo(
    data_dir: &Option<std::path::PathBuf>,
    memo: &harmony_memo::Memo,
) {
    if let Some(ref dir) = data_dir {
        let dir = dir.clone();
        let memo = memo.clone();
        tokio::task::spawn_blocking(move || {
            if let Err(e) = crate::memo_io::write_memo(&dir, &memo) {
                tracing::warn!("Failed to persist memo: {}", e);
            }
        });
    }
}
```

This function is called wherever a memo is inserted into the store (currently from Zenoh subscription handlers — which are stubbed but will be wired later). For now, place it and document that callers should invoke it after `memo_store.insert()`.

- [ ] **Step 3: Add periodic LFU flush to the tick timer**

Locate the periodic tick handler in the event loop (the `interval.tick()` branch). Add LFU flushing:

```rust
    // Inside the tick handler, after existing periodic work:
    // ── Flush memo LFU counters every 5 minutes ─────────────────────
    static MEMO_LFU_FLUSH_TICKS: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
    let tick_count = MEMO_LFU_FLUSH_TICKS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    // Assuming tick interval is ~10s, 30 ticks ≈ 5 minutes.
    if tick_count % 30 == 0 {
        if let Some(ref dir) = data_dir {
            let lfu_path = dir.join("memo_lfu.bin");
            match memo_store.serialize_lfu_counts() {
                Ok(bytes) => {
                    let path = lfu_path.clone();
                    tokio::task::spawn_blocking(move || {
                        if let Err(e) = std::fs::write(&path, &bytes) {
                            tracing::warn!("Failed to flush memo_lfu.bin: {}", e);
                        }
                    });
                }
                Err(e) => tracing::warn!("Failed to serialize LFU counts: {}", e),
            }
        }
    }
```

Note: Check the actual tick interval in the event loop. If it's not 10s, adjust the modulo accordingly to achieve ~5 minute flushes.

- [ ] **Step 4: Verify compilation**

Run: `cargo check -p harmony-node`
Expected: Compiles (may have warnings about unused `memo_store` — that's expected until Zenoh wiring happens)

- [ ] **Step 5: Commit**

```bash
git add crates/harmony-node/src/event_loop.rs
git commit -m "feat(node): integrate memo disk persistence into event loop startup and tick"
```

---

### Task 4: Full Test Suite Verification

**Files:** None — verification only

- [ ] **Step 1: Run harmony-memo tests**

Run: `cargo test -p harmony-memo`
Expected: ALL PASS (existing + new LFU tests)

- [ ] **Step 2: Run harmony-node tests**

Run: `cargo test -p harmony-node`
Expected: ALL PASS (existing + new memo_io tests)

- [ ] **Step 3: Run workspace tests**

Run: `cargo test --workspace`
Expected: ALL PASS

- [ ] **Step 4: Run clippy**

Run: `cargo clippy --workspace`
Expected: No errors (warnings about unused variables are acceptable since memo runtime wiring is a separate bead)

- [ ] **Step 5: Commit any fixes**

If any tests or clippy issues arose, fix and commit:
```bash
git commit -m "fix: address test/clippy issues from memo persistence"
```
