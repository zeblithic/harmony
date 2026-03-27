# S3 Fallback Resolver Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** When a CAS content query misses cache and disk for durable CIDs, fall back to S3 via the existing harmony-s3 library.

**Architecture:** StorageTier emits `S3Lookup` on durable cache+disk miss. NodeRuntime converts to `RuntimeAction::S3Lookup`. Event loop spawns an async `s3_library.get_book()` call. Result routes back through `S3ReadComplete`/`S3ReadFailed` events. StorageTier caches the book and serves the query reply.

**Tech Stack:** Rust, harmony-s3 (AWS SDK), harmony-content (StorageTier), tokio (async spawn)

**Spec:** `docs/superpowers/specs/2026-03-26-s3-fallback-design.md`

**Test command:** `cargo test -p harmony-content` and `cargo test -p harmony-node`
**Lint command:** `cargo clippy -p harmony-content -p harmony-node`

---

## File Structure

| File | Responsibility | Change |
|------|---------------|--------|
| `crates/harmony-content/src/storage_tier.rs` | S3 fallback emission + S3ReadComplete/Failed handlers | Modify |
| `crates/harmony-node/src/runtime.rs` | RuntimeAction/Event variants, wire S3 actions | Modify |
| `crates/harmony-node/src/event_loop.rs` | Async S3 fetch, completion channel, select! arm | Modify |

---

### Task 1: StorageTier — S3Lookup emission and S3ReadComplete/Failed handlers

Add `s3_enabled` flag, emit `S3Lookup` on durable cache+disk miss, handle S3 read results.

**Files:**
- Modify: `crates/harmony-content/src/storage_tier.rs`

- [ ] **Step 1: Write failing tests**

Add to the test module in `storage_tier.rs`:

```rust
#[test]
fn s3_lookup_emitted_on_durable_miss_when_s3_enabled() {
    let mut tier = make_tier_with_policy(ContentPolicy::default());
    tier.enable_s3();
    let (cid, _data) = cid_with_class(b"durable doc", false, false); // PublicDurable
    let actions = tier.handle(StorageTierEvent::ContentQuery { query_id: 42, cid });
    assert_eq!(actions.len(), 1);
    assert!(matches!(&actions[0], StorageTierAction::S3Lookup { query_id: 42, .. }));
}

#[test]
fn s3_lookup_not_emitted_for_ephemeral() {
    let mut tier = make_tier_with_policy(ContentPolicy::default());
    tier.enable_s3();
    let (cid, _data) = cid_with_class(b"ephemeral", false, true); // PublicEphemeral
    let actions = tier.handle(StorageTierEvent::ContentQuery { query_id: 42, cid });
    assert!(actions.is_empty(), "Ephemeral content should not trigger S3Lookup");
}

#[test]
fn s3_lookup_not_emitted_when_s3_disabled() {
    let tier = make_tier_with_policy(ContentPolicy::default());
    // s3 NOT enabled
    let (cid, _data) = cid_with_class(b"durable doc", false, false);
    let actions = tier.handle(StorageTierEvent::ContentQuery { query_id: 42, cid });
    assert!(actions.is_empty());
}

#[test]
fn s3_lookup_not_emitted_on_cache_hit() {
    let mut tier = make_tier_with_policy(ContentPolicy::default());
    tier.enable_s3();
    let (cid, data) = cid_with_class(b"cached doc", false, false);
    // Store in cache first.
    tier.handle(StorageTierEvent::TransitContent { cid, data });
    // Query should hit cache, not S3.
    let actions = tier.handle(StorageTierEvent::ContentQuery { query_id: 42, cid });
    assert!(actions.iter().any(|a| matches!(a, StorageTierAction::SendReply { .. })));
    assert!(!actions.iter().any(|a| matches!(a, StorageTierAction::S3Lookup { .. })));
}

#[test]
fn s3_lookup_not_emitted_on_disk_hit() {
    let mut tier = make_tier_with_policy(ContentPolicy::default());
    tier.enable_s3();
    tier.enable_disk(vec![]);
    let (cid, _data) = cid_with_class(b"disk doc", false, false);
    // Manually insert into disk_index.
    tier.disk_index_insert(cid);
    let actions = tier.handle(StorageTierEvent::ContentQuery { query_id: 42, cid });
    // Should emit DiskLookup, not S3Lookup.
    assert!(actions.iter().any(|a| matches!(a, StorageTierAction::DiskLookup { .. })));
    assert!(!actions.iter().any(|a| matches!(a, StorageTierAction::S3Lookup { .. })));
}

#[test]
fn s3_read_complete_caches_and_replies() {
    let mut tier = make_tier_with_policy(ContentPolicy::default());
    tier.enable_s3();
    let (cid, data) = cid_with_class(b"s3 fetched book", false, false);
    let actions = tier.handle(StorageTierEvent::S3ReadComplete {
        cid,
        query_id: 42,
        data: data.clone(),
    });
    // Should reply with data.
    assert!(actions.iter().any(|a| matches!(a, StorageTierAction::SendReply { query_id: 42, .. })));
    // Book should now be in cache.
    let query_actions = tier.handle(StorageTierEvent::ContentQuery { query_id: 43, cid });
    assert!(query_actions.iter().any(|a| matches!(a, StorageTierAction::SendReply { query_id: 43, .. })));
}

#[test]
fn s3_read_complete_with_bad_data_replies_empty() {
    let mut tier = make_tier_with_policy(ContentPolicy::default());
    tier.enable_s3();
    let (cid, _data) = cid_with_class(b"original", false, false);
    let actions = tier.handle(StorageTierEvent::S3ReadComplete {
        cid,
        query_id: 42,
        data: b"corrupted".to_vec(),
    });
    assert_eq!(actions.len(), 1);
    match &actions[0] {
        StorageTierAction::SendReply { query_id, payload } => {
            assert_eq!(*query_id, 42);
            assert!(payload.is_empty());
        }
        _ => panic!("Expected empty SendReply"),
    }
}

#[test]
fn s3_read_failed_replies_empty() {
    let mut tier = make_tier_with_policy(ContentPolicy::default());
    let actions = tier.handle(StorageTierEvent::S3ReadFailed { cid: ContentId::from_bytes([0; 32]), query_id: 42 });
    assert_eq!(actions.len(), 1);
    match &actions[0] {
        StorageTierAction::SendReply { query_id, payload } => {
            assert_eq!(*query_id, 42);
            assert!(payload.is_empty());
        }
        _ => panic!("Expected empty SendReply"),
    }
}
```

**Note:** The `disk_index_insert` test helper may not exist — add a `pub(crate) fn disk_index_insert(&mut self, cid: ContentId)` if needed for testing. Or use `enable_disk(vec![cid])` to populate the index. Check what helpers already exist.

- [ ] **Step 2: Add S3 event/action variants**

Add to `StorageTierEvent`:
```rust
/// S3 read completed — book fetched from remote storage.
S3ReadComplete { cid: ContentId, query_id: u64, data: Vec<u8> },
/// S3 read failed — book not found or network error.
S3ReadFailed { cid: ContentId, query_id: u64 },
```

Add to `StorageTierAction`:
```rust
/// Fall back to S3 for a durable CID not found in cache or on disk.
S3Lookup { cid: ContentId, query_id: u64 },
```

- [ ] **Step 3: Add `s3_enabled` field and `enable_s3()` method**

Add `s3_enabled: bool` field to `StorageTier` (default false in constructor). Add:

```rust
/// Enable S3 fallback for durable content queries.
pub fn enable_s3(&mut self) {
    self.s3_enabled = true;
}
```

- [ ] **Step 4: Emit S3Lookup on cache+disk miss**

In `handle_content_query`, after the disk_index check (which returns `DiskLookup` if the CID is indexed), modify the final `vec![]` return:

```rust
None => {
    self.metrics.cache_misses += 1;
    if self.disk_enabled && self.disk_index.contains(cid) {
        return vec![StorageTierAction::DiskLookup { cid: *cid, query_id }];
    }
    // S3 fallback for durable content when archivist is configured.
    if self.s3_enabled && Self::is_durable_class(cid) {
        return vec![StorageTierAction::S3Lookup { cid: *cid, query_id }];
    }
    vec![]
}
```

- [ ] **Step 5: Handle S3ReadComplete**

Add match arm in the main `handle` method:

```rust
StorageTierEvent::S3ReadComplete { cid, query_id, data } => {
    if !Self::verify_cid(&cid, &data) {
        self.metrics.disk_read_failures += 1; // reuse metric for now
        return vec![StorageTierAction::SendReply { query_id, payload: vec![] }];
    }
    // Pre-warm frequency so the S3-fetched CID survives admission challenge.
    self.cache.warm_frequency(&cid, 5);
    self.cache.store(cid, data.clone());
    self.mutations_since_broadcast = self.mutations_since_broadcast.saturating_add(1);

    let mut actions = vec![StorageTierAction::SendReply {
        query_id,
        payload: data.clone(),
    }];

    // Persist to disk if enabled.
    if self.disk_enabled && Self::is_durable_class(&cid) && !self.disk_index.contains(&cid) {
        self.disk_index.insert(cid);
        actions.push(StorageTierAction::PersistToDisk { cid, data: data.clone() });
    }

    // Announce to mesh if policy allows.
    if self.should_announce(&cid) {
        let key_expr = harmony_zenoh::namespace::content::publish_key(
            &hex::encode(cid.to_bytes()),
        );
        actions.push(StorageTierAction::AnnounceContent {
            key_expr,
            payload: data,
        });
    }

    if self.mutations_since_broadcast >= self.filter_config.mutation_threshold {
        actions.push(self.rebuild_filter());
    }

    actions
}
```

- [ ] **Step 6: Handle S3ReadFailed**

```rust
StorageTierEvent::S3ReadFailed { query_id, .. } => {
    vec![StorageTierAction::SendReply { query_id, payload: vec![] }]
}
```

- [ ] **Step 7: Run tests**

Run: `cargo test -p harmony-content -- --nocapture`
Expected: ALL PASS

- [ ] **Step 8: Run clippy and commit**

Run: `cargo clippy -p harmony-content`

```bash
git add crates/harmony-content/src/storage_tier.rs
git commit -m "feat(content): add S3 fallback — emit S3Lookup on durable miss, handle S3ReadComplete/Failed"
```

---

### Task 2: Wire S3 actions through NodeRuntime

Add RuntimeAction/Event variants and wire the StorageTier actions.

**Files:**
- Modify: `crates/harmony-node/src/runtime.rs`

- [ ] **Step 1: Add RuntimeAction variant**

Add to `RuntimeAction`:

```rust
/// Fetch a CAS book from S3 (spawned as async task by event loop).
S3Lookup { cid: ContentId, query_id: u64 },
```

- [ ] **Step 2: Add RuntimeEvent variants**

Add to `RuntimeEvent`:

```rust
/// S3 fetch completed.
S3ReadComplete { cid: ContentId, query_id: u64, data: Vec<u8> },
/// S3 fetch failed.
S3ReadFailed { cid: ContentId, query_id: u64 },
```

- [ ] **Step 3: Wire StorageTierAction::S3Lookup**

In `dispatch_storage_actions` (or wherever StorageTierActions are converted), add:

```rust
StorageTierAction::S3Lookup { cid, query_id } => {
    out.push(RuntimeAction::S3Lookup { cid, query_id });
}
```

- [ ] **Step 4: Wire RuntimeEvent::S3ReadComplete/Failed**

In `push_event`, add handling:

```rust
RuntimeEvent::S3ReadComplete { cid, query_id, data } => {
    let actions = self.storage.handle(StorageTierEvent::S3ReadComplete { cid, query_id, data });
    self.dispatch_storage_actions_inline(actions);
}
RuntimeEvent::S3ReadFailed { cid, query_id } => {
    let actions = self.storage.handle(StorageTierEvent::S3ReadFailed { cid, query_id });
    self.dispatch_storage_actions_inline(actions);
}
```

- [ ] **Step 5: Enable S3 in NodeRuntime::new()**

Add `s3_enabled: bool` to `NodeConfig`. In `NodeRuntime::new()`, after the `enable_disk` call:

```rust
if config.s3_enabled {
    rt.storage.enable_s3();
}
```

Add `s3_enabled: false` to NodeConfig Default and all test construction sites.

- [ ] **Step 6: Run tests, clippy, commit**

Run: `cargo test -p harmony-node -- --nocapture`
Run: `cargo clippy -p harmony-node`

```bash
git add crates/harmony-node/src/runtime.rs
git commit -m "feat(node): wire S3Lookup/S3ReadComplete/S3ReadFailed through NodeRuntime"
```

---

### Task 3: Event loop — async S3 fetch and completion channel

Handle `RuntimeAction::S3Lookup` by spawning an async S3 get, route results back.

**Files:**
- Modify: `crates/harmony-node/src/event_loop.rs`
- Modify: `crates/harmony-node/src/main.rs`

- [ ] **Step 1: Create a second S3Library for reads**

In `event_loop.rs`, after the existing archivist S3Library creation (line 262-288), create a read-side S3Library and store as `Option<Arc<harmony_s3::S3Library>>`:

```rust
#[cfg(feature = "archivist")]
let s3_read_library: Option<std::sync::Arc<harmony_s3::S3Library>> = if let Some(ref archivist) = archivist_config {
    match harmony_s3::S3Library::new(
        archivist.bucket.clone(),
        archivist.prefix.clone(),
        archivist.region.clone(),
    ).await {
        Ok(s3) => Some(std::sync::Arc::new(s3)),
        Err(e) => {
            tracing::warn!(err = %e, "S3 read library failed to init — S3 fallback disabled");
            None
        }
    }
} else {
    None
};

#[cfg(not(feature = "archivist"))]
let s3_read_library: Option<std::sync::Arc<()>> = None; // placeholder type
```

- [ ] **Step 2: Add S3 completion channel and types**

```rust
use harmony_content::cid::ContentId;

enum S3IoResult {
    ReadComplete { cid: ContentId, query_id: u64, data: Vec<u8> },
    ReadFailed { cid: ContentId, query_id: u64 },
}

let (s3_tx, mut s3_rx) = mpsc::channel::<S3IoResult>(64);
```

- [ ] **Step 3: Handle RuntimeAction::S3Lookup in dispatch_action**

Add `s3_read_library` and `s3_tx` parameters to `dispatch_action`. Add the match arm:

```rust
RuntimeAction::S3Lookup { cid, query_id } => {
    #[cfg(feature = "archivist")]
    if let Some(ref s3) = s3_read_library {
        let s3 = s3.clone();
        let tx = s3_tx.clone();
        tokio::spawn(async move {
            match s3.get_book(&cid.to_bytes()).await {
                Ok(Some(data)) => {
                    let _ = tx.send(S3IoResult::ReadComplete { cid, query_id, data }).await;
                }
                Ok(None) => {
                    tracing::debug!(cid = %hex::encode(&cid.to_bytes()[..8]), "S3 book not found");
                    let _ = tx.send(S3IoResult::ReadFailed { cid, query_id }).await;
                }
                Err(e) => {
                    tracing::warn!(cid = %hex::encode(&cid.to_bytes()[..8]), err = %e, "S3 fetch failed");
                    let _ = tx.send(S3IoResult::ReadFailed { cid, query_id }).await;
                }
            }
        });
    }
}
```

- [ ] **Step 4: Add select! arm for S3 completions**

In the main `tokio::select!` loop, add:

```rust
Some(s3_result) = s3_rx.recv() => {
    match s3_result {
        S3IoResult::ReadComplete { cid, query_id, data } => {
            runtime.push_event(RuntimeEvent::S3ReadComplete { cid, query_id, data });
        }
        S3IoResult::ReadFailed { cid, query_id } => {
            runtime.push_event(RuntimeEvent::S3ReadFailed { cid, query_id });
        }
    }
}
```

- [ ] **Step 5: Wire s3_enabled in main.rs**

In `main.rs`, set `s3_enabled` in NodeConfig:

```rust
s3_enabled: archivist_config.is_some(),
```

(The `archivist_config` variable already exists — it's resolved from config_file before NodeConfig construction.)

- [ ] **Step 6: Run tests, clippy, commit**

Run: `cargo test -p harmony-node -- --nocapture`
Run: `cargo clippy -p harmony-node`

```bash
git add crates/harmony-node/src/event_loop.rs crates/harmony-node/src/main.rs
git commit -m "feat(node): wire async S3 fetch via spawn + completion channel in event loop"
```

---

### Task 4: Final integration test

Run full test suite and verify everything compiles with and without the archivist feature.

- [ ] **Step 1: Run harmony-content tests**

Run: `cargo test -p harmony-content -- --nocapture`
Expected: ALL PASS

- [ ] **Step 2: Run harmony-node tests (without archivist)**

Run: `cargo test -p harmony-node -- --nocapture`
Expected: ALL PASS

- [ ] **Step 3: Run harmony-node tests (with archivist)**

Run: `cargo test -p harmony-node --features archivist -- --nocapture`
Expected: ALL PASS (or skip if archivist requires AWS credentials)

- [ ] **Step 4: Clippy both modes**

Run: `cargo clippy -p harmony-content -p harmony-node`
Run: `cargo clippy -p harmony-node --features archivist`
Expected: Clean

- [ ] **Step 5: Final commit if needed**

```bash
git add -A
git commit -m "chore: final cleanup after S3 fallback integration"
```
