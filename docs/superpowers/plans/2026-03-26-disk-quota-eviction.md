# Disk Space Management and Eviction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add configurable disk quota with LRU eviction for the CAS `data_dir` persistence layer.

**Architecture:** `StorageTier` gains an in-memory LRU list and byte-size tracking alongside the existing `disk_index`. After every `PersistToDisk` emission, an eviction check runs and emits `RemoveFromDisk` actions for the oldest unpinned entries. The event loop handles deletions via `spawn_blocking`. A new `disk_quota` config field controls the byte limit.

**Tech Stack:** Rust, `no_std`-compatible core (`harmony-content`), Tokio async runtime (`harmony-node`)

**Spec:** `docs/superpowers/specs/2026-03-26-disk-quota-eviction-design.md`

---

## File Structure

| File | Responsibility | Action |
|------|---------------|--------|
| `crates/harmony-node/src/disk_io.rs` | Filesystem I/O helpers | Modify: `scan_books` return type, add `delete_book` |
| `crates/harmony-node/src/config.rs` | Config parsing | Modify: add `disk_quota` field, byte-size parser |
| `crates/harmony-content/src/storage_tier.rs` | Sans-I/O storage state machine | Modify: `disk_index` type, new fields, eviction logic, LRU |
| `crates/harmony-node/src/runtime.rs` | Runtime action/event dispatch | Modify: `RuntimeAction` enum, `NodeConfig`, dispatch |
| `crates/harmony-node/src/event_loop.rs` | Async event loop | Modify: `RemoveFromDisk` handler |
| `crates/harmony-node/src/main.rs` | Startup wiring | Modify: `scan_books` result threading, `disk_quota` config |

---

### Task 1: disk_io — `delete_book` and `scan_books` with sizes

**Files:**
- Modify: `crates/harmony-node/src/disk_io.rs`

**Context:** `disk_io.rs` has three functions: `book_path` (line 21), `write_book` (line 45), `read_book` (line 67), and `scan_books` (line 79). `scan_books` currently returns `Vec<ContentId>`. Tests use `tempfile::tempdir()`.

- [ ] **Step 1: Write failing test for `delete_book`**

In the existing `#[cfg(test)] mod tests` block at the bottom of `disk_io.rs`, add:

```rust
#[test]
fn delete_book_removes_file() {
    let dir = tempfile::tempdir().unwrap();
    let cid = ContentId::from_bytes([0xAB; 32]);
    write_book(dir.path(), &cid, b"hello").unwrap();
    assert!(book_path(dir.path(), &cid).exists());

    delete_book(dir.path(), &cid).unwrap();
    assert!(!book_path(dir.path(), &cid).exists());
}

#[test]
fn delete_book_missing_file_returns_not_found() {
    let dir = tempfile::tempdir().unwrap();
    let cid = ContentId::from_bytes([0xCD; 32]);
    let err = delete_book(dir.path(), &cid).unwrap_err();
    assert_eq!(err.kind(), std::io::ErrorKind::NotFound);
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p harmony-node delete_book -- --nocapture`
Expected: FAIL — `delete_book` not found

- [ ] **Step 3: Implement `delete_book`**

Add after `read_book` (around line 71):

```rust
/// Remove a book file from disk. Returns `NotFound` if the file does not exist.
pub fn delete_book(data_dir: &std::path::Path, cid: &ContentId) -> Result<(), std::io::Error> {
    std::fs::remove_file(book_path(data_dir, cid))
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p harmony-node delete_book -- --nocapture`
Expected: PASS

- [ ] **Step 5: Write failing test for `scan_books` returning sizes**

Update the existing `scan_books` test (if one exists) or add:

```rust
#[test]
fn scan_books_returns_cid_and_size() {
    let dir = tempfile::tempdir().unwrap();
    let cid1 = ContentId::from_bytes([0x01; 32]);
    let cid2 = ContentId::from_bytes([0x02; 32]);
    write_book(dir.path(), &cid1, &[0u8; 100]).unwrap();
    write_book(dir.path(), &cid2, &[0u8; 200]).unwrap();

    let mut entries = scan_books(dir.path());
    entries.sort_by_key(|(_, size)| *size);

    assert_eq!(entries.len(), 2);
    assert_eq!(entries[0], (cid1, 100));
    assert_eq!(entries[1], (cid2, 200));
}
```

- [ ] **Step 6: Run test to verify it fails**

Run: `cargo test -p harmony-node scan_books_returns -- --nocapture`
Expected: FAIL — type mismatch (returns `Vec<ContentId>` not `Vec<(ContentId, u64)>`)

- [ ] **Step 7: Update `scan_books` to return `Vec<(ContentId, u64)>`**

Change the return type and add `fs::metadata().len()` to each entry. The current implementation walks directories and parses CID hex filenames. For each valid CID file, read `metadata().len()` and include it in the tuple. If metadata fails, skip the file with a warning (same pattern as invalid filenames).

Update the function signature (line 79):
```rust
pub fn scan_books(data_dir: &std::path::Path) -> Vec<(ContentId, u64)>
```

Where CIDs are currently pushed: change `results.push(cid)` to include size:
```rust
match entry.metadata() {
    Ok(meta) => results.push((cid, meta.len())),
    Err(e) => {
        tracing::warn!(path = %entry.path().display(), error = %e, "skipping book: metadata read failed");
    }
}
```

**Important:** This will break callers in `main.rs`. That's expected — Task 7 fixes them.

- [ ] **Step 8: Fix existing `scan_books` tests and `main.rs` caller**

Changing the `scan_books` return type from `Vec<ContentId>` to `Vec<(ContentId, u64)>` breaks existing tests and `main.rs`. Fix them:

**Existing tests in `disk_io.rs`** (~3 tests):
- `scan_discovers_written_books`: Update sort and assertions to use tuples. Sort by `|(c, _)| c.to_bytes()`, compare `found[0].0 == cid_a` etc., or compare full tuples with expected sizes.
- `scan_skips_invalid_filenames`: Change `assert_eq!(found[0], cid)` to `assert_eq!(found[0].0, cid)`.
- `scan_empty_directory_returns_empty`: No change needed (`.is_empty()` works on any vec).

**`main.rs` caller** (line ~553): Add a temporary adapter to unblock compilation:

```rust
let cids = crate::disk_io::scan_books(&dir);
// Temporary: discard sizes until Task 7 threads them through
cids.into_iter().map(|(cid, _)| cid).collect::<Vec<_>>()
```

This will be properly replaced in Task 7.

- [ ] **Step 9: Run tests to verify they pass**

Run: `cargo test -p harmony-node -- disk_io --nocapture`
Expected: PASS

- [ ] **Step 10: Commit**

```bash
git add crates/harmony-node/src/disk_io.rs crates/harmony-node/src/main.rs
git commit -m "feat(node): add delete_book and return sizes from scan_books"
```

---

### Task 2: Config — byte-size parser and `disk_quota` field

**Files:**
- Modify: `crates/harmony-node/src/config.rs`

**Context:** `ConfigFile` (line 56) is a `#[derive(Deserialize)]` struct parsed from TOML. `NodeConfig` (line 41 of `runtime.rs`) is built from it in `main.rs`. No existing byte-size parsing exists.

- [ ] **Step 1: Write failing tests for byte-size parser**

Add a new test module at the bottom of `config.rs` (or in the existing test module):

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_byte_size_binary_units() {
        assert_eq!(parse_byte_size("1 KiB").unwrap(), 1024);
        assert_eq!(parse_byte_size("2 MiB").unwrap(), 2 * 1024 * 1024);
        assert_eq!(parse_byte_size("10 GiB").unwrap(), 10 * 1024 * 1024 * 1024);
    }

    #[test]
    fn parse_byte_size_decimal_units() {
        assert_eq!(parse_byte_size("500 MB").unwrap(), 500 * 1_000_000);
        assert_eq!(parse_byte_size("1 GB").unwrap(), 1_000_000_000);
        assert_eq!(parse_byte_size("100 KB").unwrap(), 100_000);
    }

    #[test]
    fn parse_byte_size_bytes_suffix() {
        assert_eq!(parse_byte_size("1024 B").unwrap(), 1024);
    }

    #[test]
    fn parse_byte_size_case_insensitive() {
        assert_eq!(parse_byte_size("1 gib").unwrap(), 1024 * 1024 * 1024);
        assert_eq!(parse_byte_size("1 GIB").unwrap(), 1024 * 1024 * 1024);
        assert_eq!(parse_byte_size("500 mb").unwrap(), 500 * 1_000_000);
    }

    #[test]
    fn parse_byte_size_no_space() {
        assert_eq!(parse_byte_size("10GiB").unwrap(), 10 * 1024 * 1024 * 1024);
    }

    #[test]
    fn parse_byte_size_bare_number_rejected() {
        assert!(parse_byte_size("1234").is_err());
    }

    #[test]
    fn parse_byte_size_invalid_suffix_rejected() {
        assert!(parse_byte_size("10 TB").is_err());
        assert!(parse_byte_size("abc").is_err());
        assert!(parse_byte_size("").is_err());
    }
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test -p harmony-node parse_byte_size -- --nocapture`
Expected: FAIL — `parse_byte_size` not found

- [ ] **Step 3: Implement `parse_byte_size`**

Add above the test module:

```rust
/// Parse a human-readable byte size string like "10 GiB" or "500 MB".
///
/// Supported suffixes (case-insensitive): B, KB, MB, GB, KiB, MiB, GiB.
/// A bare number without a suffix is rejected to avoid ambiguity.
pub fn parse_byte_size(s: &str) -> Result<u64, String> {
    let s = s.trim();
    if s.is_empty() {
        return Err("empty byte size string".into());
    }

    // Find the boundary between digits and suffix
    let num_end = s
        .find(|c: char| !c.is_ascii_digit())
        .unwrap_or(s.len());

    if num_end == 0 {
        return Err(format!("no numeric value in '{s}'"));
    }

    let num: u64 = s[..num_end]
        .parse()
        .map_err(|e| format!("invalid number in '{s}': {e}"))?;

    let suffix = s[num_end..].trim().to_ascii_lowercase();

    let multiplier: u64 = match suffix.as_str() {
        "b" => 1,
        "kb" => 1_000,
        "mb" => 1_000_000,
        "gb" => 1_000_000_000,
        "kib" => 1_024,
        "mib" => 1_024 * 1_024,
        "gib" => 1_024 * 1_024 * 1_024,
        "" => return Err(format!("bare number '{s}' requires a unit suffix (B, KB, MB, GB, KiB, MiB, GiB)")),
        other => return Err(format!("unknown unit suffix '{other}' in '{s}'")),
    };

    num.checked_mul(multiplier)
        .ok_or_else(|| format!("byte size overflow: {num} * {multiplier}"))
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test -p harmony-node parse_byte_size -- --nocapture`
Expected: PASS

- [ ] **Step 5: Add `disk_quota` field to `ConfigFile`**

In `ConfigFile` struct (around line 65, after `data_dir`):

```rust
/// Disk quota for CAS book storage (e.g. "10 GiB"). Requires `data_dir`.
/// If absent, disk usage is unbounded.
pub disk_quota: Option<String>,
```

This is a serde-deserialized `Option<String>`. The parsing to `u64` happens in `main.rs` at startup.

- [ ] **Step 6: Run full crate tests**

Run: `cargo test -p harmony-node -- --nocapture`
Expected: PASS (new field with Option defaults to None in deserialization)

- [ ] **Step 7: Commit**

```bash
git add crates/harmony-node/src/config.rs
git commit -m "feat(node): add byte-size parser and disk_quota config field"
```

---

### Task 3: StorageTier state — `disk_index` to `HashMap`, new fields, `enable_disk` signature

**Files:**
- Modify: `crates/harmony-content/src/storage_tier.rs`

**Context:** `StorageTier` struct is at line 192. `disk_index` is `HashSet<ContentId>` at line 205. `enable_disk` is at line 287. The cache field is `cache: ContentStore<B>` at line 193, and `ContentStore::is_pinned()` already exists (cache.rs:139).

- [ ] **Step 1: Write failing tests for new state**

Add to the existing `#[cfg(test)] mod tests` block:

Use the existing test constructor pattern (returns a tuple):

```rust
#[test]
fn enable_disk_with_sizes_tracks_bytes() {
    let budget = StorageBudget { cache_capacity: 10, max_pinned_bytes: 0 };
    let (mut tier, _) = StorageTier::new(
        MemoryBookStore::new(),
        budget,
        ContentPolicy::default(),
        FilterBroadcastConfig::default(),
    );

    let cid1 = ContentId::from_bytes([0x01; 32]);
    let cid2 = ContentId::from_bytes([0x02; 32]);

    tier.enable_disk(vec![(cid1, 100), (cid2, 200)]);

    assert_eq!(tier.disk_used_bytes(), 300);
    assert!(tier.disk_contains(&cid1));
    assert!(tier.disk_contains(&cid2));
}

#[test]
fn set_disk_quota() {
    let budget = StorageBudget { cache_capacity: 10, max_pinned_bytes: 0 };
    let (mut tier, _) = StorageTier::new(
        MemoryBookStore::new(),
        budget,
        ContentPolicy::default(),
        FilterBroadcastConfig::default(),
    );
    tier.set_disk_quota(1024);
    assert_eq!(tier.disk_quota(), Some(1024));
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test -p harmony-content enable_disk_with_sizes -- --nocapture`
Expected: FAIL — signature mismatch / methods not found

- [ ] **Step 3: Update `StorageTier` fields and methods**

Change the struct fields (around line 205):

```rust
// Replace:
disk_index: HashSet<ContentId>,

// With:
disk_index: HashMap<ContentId, u64>,
disk_lru: VecDeque<ContentId>,
disk_used_bytes: u64,
disk_quota: Option<u64>,
```

Add necessary imports at the top of the file:
```rust
use alloc::collections::VecDeque;
// HashMap should already be imported; verify
```

Update `new()` to initialize new fields:
```rust
disk_index: HashMap::new(),
disk_lru: VecDeque::new(),
disk_used_bytes: 0,
disk_quota: None,
```

Update `enable_disk` (line 287):
```rust
pub fn enable_disk(&mut self, entries: impl IntoIterator<Item = (ContentId, u64)>) {
    self.disk_enabled = true;
    for (cid, size) in entries {
        self.disk_index.insert(cid, size);
        self.disk_lru.push_back(cid);
        self.disk_used_bytes += size;
    }
}
```

Add new methods:
```rust
pub fn set_disk_quota(&mut self, quota_bytes: u64) {
    self.disk_quota = Some(quota_bytes);
}

/// Current disk usage in bytes.
pub fn disk_used_bytes(&self) -> u64 {
    self.disk_used_bytes
}

/// Configured disk quota, if any.
pub fn disk_quota(&self) -> Option<u64> {
    self.disk_quota
}

/// Whether a CID is tracked in the disk index.
pub fn disk_contains(&self, cid: &ContentId) -> bool {
    self.disk_index.contains_key(cid)
}
```

**Fix all existing code that uses `disk_index`:**
- `disk_index.contains(&cid)` → `disk_index.contains_key(&cid)` (HashSet → HashMap)
- `disk_index.insert(cid)` → `disk_index.insert(cid, size)` (need size at each insert site — see below)
- `disk_index.remove(&cid)` stays the same on HashMap

**Fix all existing `enable_disk` callers in tests** (signature changed from `impl IntoIterator<Item = ContentId>` to `impl IntoIterator<Item = (ContentId, u64)>`):
- `tier.enable_disk(std::iter::empty())` → `tier.enable_disk(std::iter::empty::<(ContentId, u64)>())`
- `tier.enable_disk(std::iter::once(cid))` → `tier.enable_disk(std::iter::once((cid, 100)))`
- `tier.enable_disk(vec![cid_a, cid_b])` → `tier.enable_disk(vec![(cid_a, 100), (cid_b, 100)])`

There are ~5 existing test call sites that need updating. Search for `enable_disk` in the test module and fix each one.

The `PersistToDisk` emission sites (lines ~520 and ~569) currently do `disk_index.insert(cid)`. These need to become `disk_index.insert(cid, data.len() as u64)` and also update `disk_used_bytes` and `disk_lru`. This will be completed in Task 4 (eviction logic) since the eviction check runs right after.

For now, to make compilation pass, update the insert calls minimally:
```rust
self.disk_index.insert(cid, persist_bytes.len() as u64);
```

The `DiskReadFailed` handler (line ~379) does `disk_index.remove(&cid)`. Update to also adjust `disk_used_bytes`:
```rust
if let Some(size) = self.disk_index.remove(&cid) {
    self.disk_used_bytes = self.disk_used_bytes.saturating_sub(size);
    self.disk_lru.retain(|c| c != &cid);
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test -p harmony-content -- --nocapture`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add crates/harmony-content/src/storage_tier.rs
git commit -m "feat(content): change disk_index to HashMap with size tracking and LRU state"
```

---

### Task 4: Eviction logic

**Files:**
- Modify: `crates/harmony-content/src/storage_tier.rs`

**Context:** After Task 3, `StorageTier` has `disk_index: HashMap<ContentId, u64>`, `disk_lru: VecDeque<ContentId>`, `disk_used_bytes: u64`, `disk_quota: Option<u64>`. The `PersistToDisk` emission sites need to call eviction after updating bookkeeping.

- [ ] **Step 1: Write failing tests for eviction**

```rust
#[test]
fn eviction_removes_oldest_when_over_quota() {
    let budget = StorageBudget { cache_capacity: 10, max_pinned_bytes: 0 };
    let (mut tier, _) = StorageTier::new(
        MemoryBookStore::new(), budget,
        ContentPolicy::default(), FilterBroadcastConfig::default(),
    );
    tier.enable_disk(Vec::new());
    tier.set_disk_quota(250); // 250 bytes quota

    // Simulate persisting 3 × 100-byte books (total 300 > 250)
    let cid_a = ContentId::from_bytes([0x0A; 32]);
    let cid_b = ContentId::from_bytes([0x0B; 32]);
    let cid_c = ContentId::from_bytes([0x0C; 32]);

    let actions_a = tier.test_persist_to_disk(cid_a, 100);
    assert!(!actions_a.iter().any(|a| matches!(a, StorageTierAction::RemoveFromDisk { .. })));

    let actions_b = tier.test_persist_to_disk(cid_b, 100);
    assert!(!actions_b.iter().any(|a| matches!(a, StorageTierAction::RemoveFromDisk { .. })));

    // Third book pushes over quota — should evict cid_a (oldest)
    let actions_c = tier.test_persist_to_disk(cid_c, 100);
    assert!(actions_c.iter().any(|a| matches!(a, StorageTierAction::RemoveFromDisk { cid } if *cid == cid_a)));
    assert!(!tier.disk_contains(&cid_a));
    assert!(tier.disk_contains(&cid_b));
    assert!(tier.disk_contains(&cid_c));
    assert_eq!(tier.disk_used_bytes(), 200);
}

#[test]
fn eviction_skips_pinned_books() {
    let budget = StorageBudget { cache_capacity: 10, max_pinned_bytes: 0 };
    let (mut tier, _) = StorageTier::new(
        MemoryBookStore::new(), budget,
        ContentPolicy::default(), FilterBroadcastConfig::default(),
    );
    tier.enable_disk(Vec::new());
    tier.set_disk_quota(250);

    let cid_a = ContentId::from_bytes([0x0A; 32]);
    let cid_b = ContentId::from_bytes([0x0B; 32]);
    let cid_c = ContentId::from_bytes([0x0C; 32]);

    tier.test_persist_to_disk(cid_a, 100);
    tier.test_persist_to_disk(cid_b, 100);
    tier.cache_pin(cid_a); // pin the oldest

    // Over quota — should skip cid_a (pinned) and evict cid_b
    let actions = tier.test_persist_to_disk(cid_c, 100);
    assert!(actions.iter().any(|a| matches!(a, StorageTierAction::RemoveFromDisk { cid } if *cid == cid_b)));
    assert!(tier.disk_contains(&cid_a)); // pinned, still present
    assert!(!tier.disk_contains(&cid_b)); // evicted
    assert!(tier.disk_contains(&cid_c));
}

#[test]
fn no_eviction_without_quota() {
    let budget = StorageBudget { cache_capacity: 10, max_pinned_bytes: 0 };
    let (mut tier, _) = StorageTier::new(
        MemoryBookStore::new(), budget,
        ContentPolicy::default(), FilterBroadcastConfig::default(),
    );
    tier.enable_disk(Vec::new());
    // No set_disk_quota call — unlimited

    let cid = ContentId::from_bytes([0x01; 32]);
    let actions = tier.test_persist_to_disk(cid, 1_000_000);
    assert!(!actions.iter().any(|a| matches!(a, StorageTierAction::RemoveFromDisk { .. })));
}

#[test]
fn eviction_safety_valve_all_pinned() {
    let budget = StorageBudget { cache_capacity: 100, max_pinned_bytes: 0 };
    let (mut tier, _) = StorageTier::new(
        MemoryBookStore::new(), budget,
        ContentPolicy::default(), FilterBroadcastConfig::default(),
    );
    tier.enable_disk(Vec::new());
    tier.set_disk_quota(150);

    let cid_a = ContentId::from_bytes([0x0A; 32]);
    let cid_b = ContentId::from_bytes([0x0B; 32]);
    tier.test_persist_to_disk(cid_a, 100);
    tier.cache_pin(cid_a);

    // cid_b pushes over quota, but cid_a is pinned and cid_b is just-persisted
    // Safety valve should prevent infinite loop
    let actions = tier.test_persist_to_disk(cid_b, 100);
    assert!(!actions.iter().any(|a| matches!(a, StorageTierAction::RemoveFromDisk { .. })));
    // Both still present — quota exceeded but nothing evictable
    assert!(tier.disk_contains(&cid_a));
    assert!(tier.disk_contains(&cid_b));
}

#[test]
fn dedup_on_re_persist() {
    let budget = StorageBudget { cache_capacity: 10, max_pinned_bytes: 0 };
    let (mut tier, _) = StorageTier::new(
        MemoryBookStore::new(), budget,
        ContentPolicy::default(), FilterBroadcastConfig::default(),
    );
    tier.enable_disk(Vec::new());
    tier.set_disk_quota(500);

    let cid = ContentId::from_bytes([0x01; 32]);
    tier.test_persist_to_disk(cid, 100);
    tier.test_persist_to_disk(cid, 100); // same CID again
    assert_eq!(tier.disk_used_bytes(), 100); // counted once
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test -p harmony-content eviction -- --nocapture`
Expected: FAIL — `test_persist_to_disk`, `cache_pin` methods not found

- [ ] **Step 3: Implement eviction logic**

Add a `#[cfg(test)]`-gated test helper:

```rust
#[cfg(test)]
impl<B: BookStore> StorageTier<B> {
    /// Test helper: simulate a PersistToDisk event and run eviction.
    /// Returns any actions generated (PersistToDisk + RemoveFromDisk).
    pub fn test_persist_to_disk(&mut self, cid: ContentId, size: u64) -> Vec<StorageTierAction> {
        let mut actions = Vec::new();
        self.record_disk_persist(cid, size, &mut actions);
        actions
    }

    /// Test helper: delegate to cache pin.
    pub fn cache_pin(&mut self, cid: ContentId) -> bool {
        self.cache.pin(cid)
    }
}
```

Add the core eviction method (private, not test-only):

```rust
/// Record a book persisted to disk and run eviction if over quota.
fn record_disk_persist(
    &mut self,
    cid: ContentId,
    size: u64,
    actions: &mut Vec<StorageTierAction>,
) {
    // Dedup: if already on disk, skip
    if self.disk_index.contains_key(&cid) {
        return;
    }

    // Update bookkeeping
    self.disk_index.insert(cid, size);
    self.disk_lru.push_back(cid);
    self.disk_used_bytes += size;

    // Check quota
    let quota = match self.disk_quota {
        Some(q) => q,
        None => return,
    };

    if self.disk_used_bytes <= quota {
        return;
    }

    // Evict LRU entries until under quota
    let mut skipped: usize = 0;
    while self.disk_used_bytes > quota {
        if skipped >= self.disk_lru.len() {
            // Safety valve: all entries are unevictable.
            // Cannot log here (harmony-content is no_std / no tracing).
            // The runtime can detect this by comparing disk_used_bytes > quota
            // after processing actions if logging is desired.
            break;
        }

        let candidate = match self.disk_lru.pop_front() {
            Some(c) => c,
            None => break,
        };

        // Never evict the just-persisted book
        if candidate == cid {
            self.disk_lru.push_back(candidate);
            skipped += 1;
            continue;
        }

        // Skip pinned books
        if self.cache.is_pinned(&candidate) {
            self.disk_lru.push_back(candidate);
            skipped += 1;
            continue;
        }

        // Evict
        if let Some(entry_size) = self.disk_index.remove(&candidate) {
            self.disk_used_bytes = self.disk_used_bytes.saturating_sub(entry_size);
        }
        actions.push(StorageTierAction::RemoveFromDisk { cid: candidate });
        skipped = 0; // Reset after successful eviction
    }
}
```

**Wire into PersistToDisk emission sites.** At both `handle_transit` (~line 520) and `handle_publish` (~line 569), after the existing `PersistToDisk` push, replace the manual `disk_index.insert(cid)` with a call to `record_disk_persist`:

```rust
// Replace the existing disk_index.insert + PersistToDisk push with:
actions.push(StorageTierAction::PersistToDisk {
    cid,
    data: persist_bytes.clone(),
});
self.record_disk_persist(cid, persist_bytes.len() as u64, &mut actions);
```

Note: `record_disk_persist` handles the `disk_index` insert internally, so remove the standalone `disk_index.insert` that was there before.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test -p harmony-content -- --nocapture`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add crates/harmony-content/src/storage_tier.rs
git commit -m "feat(content): add LRU disk eviction with pin exemption and safety valve"
```

---

### Task 5: LRU maintenance — `DiskReadComplete` refresh

**Files:**
- Modify: `crates/harmony-content/src/storage_tier.rs`

**Context:** After Task 4, the LRU is updated on persist and eviction. We still need to refresh ordering when a book is loaded from disk (DiskReadComplete), so frequently-accessed books don't get evicted.

- [ ] **Step 1: Write failing test**

```rust
#[test]
fn disk_read_complete_refreshes_lru_ordering() {
    let budget = StorageBudget { cache_capacity: 10, max_pinned_bytes: 0 };
    let (mut tier, _) = StorageTier::new(
        MemoryBookStore::new(), budget,
        ContentPolicy::default(), FilterBroadcastConfig::default(),
    );
    tier.enable_disk(Vec::new());
    tier.set_disk_quota(250);

    let cid_a = ContentId::from_bytes([0x0A; 32]);
    let cid_b = ContentId::from_bytes([0x0B; 32]);
    let cid_c = ContentId::from_bytes([0x0C; 32]);

    tier.test_persist_to_disk(cid_a, 100);
    tier.test_persist_to_disk(cid_b, 100);

    // Simulate a disk read of cid_a — moves it to back of LRU
    tier.touch_disk_lru(&cid_a);

    // Now cid_b is oldest. Adding cid_c should evict cid_b, not cid_a.
    let actions = tier.test_persist_to_disk(cid_c, 100);
    assert!(actions.iter().any(|a| matches!(a, StorageTierAction::RemoveFromDisk { cid } if *cid == cid_b)));
    assert!(tier.disk_contains(&cid_a)); // refreshed, not evicted
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p harmony-content disk_read_complete_refreshes -- --nocapture`
Expected: FAIL — `touch_disk_lru` not found

- [ ] **Step 3: Implement LRU touch**

Add a `pub(crate)` method to `StorageTier` (accessible from tests and from within the crate):

```rust
/// Move a CID to the back of the disk LRU (most recently used).
pub(crate) fn touch_disk_lru(&mut self, cid: &ContentId) {
    self.disk_lru.retain(|c| c != cid);
    self.disk_lru.push_back(*cid);
}
```

**Wire into `DiskReadComplete` handler** (around line 324-373). After the successful re-cache (around line 361), add:

```rust
self.touch_disk_lru(&cid);
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test -p harmony-content -- --nocapture`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add crates/harmony-content/src/storage_tier.rs
git commit -m "feat(content): refresh disk LRU ordering on DiskReadComplete"
```

---

### Task 6: Runtime + event loop wiring

**Files:**
- Modify: `crates/harmony-node/src/runtime.rs` (RuntimeAction enum, NodeConfig, dispatch)
- Modify: `crates/harmony-node/src/event_loop.rs` (RemoveFromDisk handler)

**Context:** `RuntimeAction` enum is at line 269 of runtime.rs. The `RemoveFromDisk` placeholder is at line ~1898. `NodeConfig` has `disk_cids: Vec<ContentId>` at line 77. The event loop handles `PersistToDisk` at line 1509 and `DiskLookup` at line 1523.

- [ ] **Step 1: Add `RemoveFromDisk` to `RuntimeAction` enum**

In `runtime.rs`, in the `RuntimeAction` enum (after `DiskLookup` at line ~329):

```rust
/// Delete a book from disk (evicted by quota enforcement).
RemoveFromDisk { cid: ContentId },
```

- [ ] **Step 2: Update `NodeConfig`**

Change `disk_cids` field (line ~77):
```rust
// Replace:
pub disk_cids: Vec<ContentId>,

// With:
pub disk_entries: Vec<(ContentId, u64)>,
```

Add new field:
```rust
pub disk_quota: Option<u64>,
```

- [ ] **Step 3: Update `NodeRuntime::new()` to wire quota and entries**

In `NodeRuntime::new()` (around line 903-905), update the `enable_disk` call:

```rust
if config.disk_enabled {
    rt.storage.enable_disk(config.disk_entries);
    if let Some(quota) = config.disk_quota {
        rt.storage.set_disk_quota(quota);
    }
}
```

- [ ] **Step 4: Wire `RemoveFromDisk` dispatch in runtime**

Replace the placeholder (line ~1898):
```rust
// Replace:
StorageTierAction::RemoveFromDisk { .. } => {
    // Deferred to disk eviction bead (harmony-mti6).
}

// With:
StorageTierAction::RemoveFromDisk { cid } => {
    out.push(RuntimeAction::RemoveFromDisk { cid });
}
```

- [ ] **Step 5: Add `RemoveFromDisk` handler in event loop**

In `event_loop.rs`, in the action dispatch match (after `DiskLookup` handler, around line 1551):

```rust
RuntimeAction::RemoveFromDisk { cid } => {
    if let Some(ref dir) = data_dir {
        let dir = dir.clone();
        tokio::task::spawn_blocking(move || {
            if let Err(e) = crate::disk_io::delete_book(&dir, &cid) {
                tracing::warn!(?cid, error = %e, "failed to delete book from disk");
            }
        });
    }
}
```

- [ ] **Step 6: Fix compilation**

Run: `cargo check -p harmony-node`

Fix any remaining references to `disk_cids` → `disk_entries` (there will be at least one in `main.rs` from Task 1's temporary fix). This ensures the crate compiles even if `main.rs` still needs full wiring (Task 7).

- [ ] **Step 7: Run tests**

Run: `cargo test -p harmony-node -- --nocapture`
Expected: PASS

- [ ] **Step 8: Commit**

```bash
git add crates/harmony-node/src/runtime.rs crates/harmony-node/src/event_loop.rs crates/harmony-node/src/main.rs
git commit -m "feat(node): wire RemoveFromDisk through runtime and event loop"
```

---

### Task 7: Startup integration — `main.rs` threading

**Files:**
- Modify: `crates/harmony-node/src/main.rs`

**Context:** `main.rs` calls `scan_books` at line ~548, passes results as `disk_cids` to `NodeConfig` at line ~609, and passes `data_dir` to the event loop at line ~665. After Tasks 1-6, `scan_books` returns `Vec<(ContentId, u64)>` and `NodeConfig` expects `disk_entries: Vec<(ContentId, u64)>` and `disk_quota: Option<u64>`.

- [ ] **Step 1: Update `scan_books` result handling**

At line ~548, update the variable name and type:

```rust
// Replace disk_cids with disk_entries
let disk_entries = match &config_file.data_dir {
    Some(dir) => {
        let dir = dir.clone();
        tokio::task::spawn_blocking(move || {
            let entries = crate::disk_io::scan_books(&dir);
            tracing::info!(
                count = entries.len(),
                total_bytes = entries.iter().map(|(_, s)| s).sum::<u64>(),
                path = %dir.display(),
                "loaded book entries from disk"
            );
            entries
        })
        .await
        .unwrap_or_default()
    }
    None => Vec::new(),
};
```

- [ ] **Step 2: Parse `disk_quota` from config**

After the `disk_entries` block, add:

```rust
let disk_quota = match &config_file.disk_quota {
    Some(s) => {
        let bytes = crate::config::parse_byte_size(s)
            .map_err(|e| anyhow::anyhow!("invalid disk_quota '{}': {}", s, e))?;
        tracing::info!(quota_bytes = bytes, raw = %s, "disk quota configured");
        Some(bytes)
    }
    None => {
        if config_file.data_dir.is_some() {
            tracing::info!("disk persistence enabled without quota — disk usage is unbounded");
        }
        None
    }
};
```

- [ ] **Step 3: Update `NodeConfig` construction**

At line ~609, replace:
```rust
disk_cids: disk_cids,
```

With:
```rust
disk_entries,
disk_quota,
```

- [ ] **Step 4: Verify full compilation and tests**

Run: `cargo clippy --workspace --all-targets -- -D warnings && cargo test --workspace`
Expected: PASS — everything compiles and all tests pass

- [ ] **Step 5: Run nightly rustfmt**

Run: `rustup run nightly cargo fmt --all -- --check`

If it reports formatting issues, run `rustup run nightly cargo fmt --all` and include the changes.

- [ ] **Step 6: Commit**

```bash
git add crates/harmony-node/src/main.rs
git commit -m "feat(node): thread disk_entries and disk_quota through startup"
```

---

## Verification Checklist

After all tasks are complete:

- [ ] `cargo clippy --workspace --all-targets -- -D warnings` passes
- [ ] `cargo test --workspace` passes
- [ ] `rustup run nightly cargo fmt --all -- --check` passes
- [ ] `RemoveFromDisk` actions are emitted when over quota (unit tests prove this)
- [ ] Pinned books are never evicted (unit test proves this)
- [ ] No eviction when `disk_quota` is `None` (unit test proves this)
- [ ] Safety valve prevents infinite loops (unit test proves this)
- [ ] LRU ordering is refreshed on disk reads (unit test proves this)
- [ ] Duplicate persists don't double-count bytes (unit test proves this)
