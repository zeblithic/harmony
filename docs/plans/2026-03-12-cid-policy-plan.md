# CID Storage/Publishing Policy Enforcement — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Enforce storage, eviction, and Zenoh publishing rules based on the `(encrypted, ephemeral)` classification bits in every ContentId.

**Architecture:** Policy-aware StorageTier (Approach B) — extend `StorageTier` to consult a `ContentPolicy` config at three decision points: admission, eviction, and announcement. Content class is derived from the two leading bits of byte 0 in the CID. `EncryptedEphemeral` content never enters StorageTier at all — the `NodeRuntime` gates it at the routing layer.

**Tech Stack:** Rust, `no_std`-compatible (`harmony-content`), `std` (`harmony-node`), clap CLI, W-TinyLFU cache

**Design doc:** `docs/plans/2026-03-12-cid-policy-design.md`

---

### Task 1: ContentClass enum and content_class() method

**Files:**
- Modify: `crates/harmony-content/src/cid.rs`

**Context:** `ContentFlags` already exists at `cid.rs:17-22` with `encrypted: bool` and `ephemeral: bool`. `ContentId` has a `flags()` method that extracts these from byte 0. We need a `ContentClass` enum and a `content_class()` method on `ContentId`.

**Step 1: Write the failing tests**

Add to the existing `#[cfg(test)] mod tests` block in `cid.rs`:

```rust
#[test]
fn content_class_public_durable() {
    let flags = ContentFlags { encrypted: false, ephemeral: false, alt_hash: false };
    let cid = ContentId::for_blob(b"test", flags).unwrap();
    assert_eq!(cid.content_class(), ContentClass::PublicDurable);
}

#[test]
fn content_class_public_ephemeral() {
    let flags = ContentFlags { encrypted: false, ephemeral: true, alt_hash: false };
    let cid = ContentId::for_blob(b"test", flags).unwrap();
    assert_eq!(cid.content_class(), ContentClass::PublicEphemeral);
}

#[test]
fn content_class_encrypted_durable() {
    let flags = ContentFlags { encrypted: true, ephemeral: false, alt_hash: false };
    let cid = ContentId::for_blob(b"test", flags).unwrap();
    assert_eq!(cid.content_class(), ContentClass::EncryptedDurable);
}

#[test]
fn content_class_encrypted_ephemeral() {
    let flags = ContentFlags { encrypted: true, ephemeral: true, alt_hash: false };
    let cid = ContentId::for_blob(b"test", flags).unwrap();
    assert_eq!(cid.content_class(), ContentClass::EncryptedEphemeral);
}
```

**Step 2: Run tests to verify they fail**

Run: `cargo test -p harmony-content content_class_public_durable -- --no-capture`
Expected: FAIL — `ContentClass` not found

**Step 3: Write minimal implementation**

Add above `ContentFlags` in `cid.rs`:

```rust
/// Content class derived from the two leading classification bits of a CID.
///
/// The `(encrypted, ephemeral)` bits in `ContentFlags` (byte 0 of every CID)
/// define four content classes with distinct storage and publishing policies.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ContentClass {
    /// `(false, false)` — Valuable public information. Disk-backed, LFU-managed.
    PublicDurable,
    /// `(false, true)` — Disposable-first content. Memory-only, evict first.
    PublicEphemeral,
    /// `(true, false)` — Content with intentional relationship. Configurable per-device.
    EncryptedDurable,
    /// `(true, true)` — Maximum privacy. Never persists, never enters Zenoh.
    EncryptedEphemeral,
}
```

Add method on `ContentId` (inside the existing `impl ContentId` block):

```rust
/// Derive the content class from this CID's classification flags.
pub fn content_class(&self) -> ContentClass {
    let flags = self.flags();
    match (flags.encrypted, flags.ephemeral) {
        (false, false) => ContentClass::PublicDurable,
        (false, true)  => ContentClass::PublicEphemeral,
        (true, false)  => ContentClass::EncryptedDurable,
        (true, true)   => ContentClass::EncryptedEphemeral,
    }
}
```

Export `ContentClass` from `lib.rs`:

```rust
pub use cid::{ContentClass, ContentFlags, ContentId};
```

**Step 4: Run tests to verify they pass**

Run: `cargo test -p harmony-content content_class -- --no-capture`
Expected: 4 tests PASS

**Step 5: Commit**

```bash
git add crates/harmony-content/src/cid.rs crates/harmony-content/src/lib.rs
git commit -m "feat(content): add ContentClass enum and content_class() method"
```

---

### Task 2: ContentPolicy struct

**Files:**
- Modify: `crates/harmony-content/src/storage_tier.rs`

**Context:** `StorageTier` currently has `StorageBudget` for capacity config. We need `ContentPolicy` for per-class behavior config. The struct lives next to `StorageBudget` in `storage_tier.rs`.

**Step 1: Write the failing test**

Add to `storage_tier.rs` tests:

```rust
#[test]
fn content_policy_defaults_are_conservative() {
    let policy = ContentPolicy::default();
    // Conservative: don't store or announce encrypted content by default.
    assert!(!policy.encrypted_durable_persist);
    assert!(!policy.encrypted_durable_announce);
    // Public ephemeral announces by default (opt-out).
    assert!(policy.public_ephemeral_announce);
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p harmony-content content_policy_defaults -- --no-capture`
Expected: FAIL — `ContentPolicy` not found

**Step 3: Write minimal implementation**

Add after `StorageBudget` in `storage_tier.rs`:

```rust
use crate::cid::ContentClass;

/// Per-class storage and publishing policy.
///
/// Controls how each content class is handled at the admission and
/// announcement decision points. Classes not covered by explicit flags
/// have hardcoded behavior:
/// - `PublicDurable (00)`: always persist, always announce.
/// - `EncryptedEphemeral (11)`: never reaches StorageTier (gated at runtime).
#[derive(Debug, Clone)]
pub struct ContentPolicy {
    /// Whether to persist encrypted durable (10) content.
    pub encrypted_durable_persist: bool,
    /// Whether to announce encrypted durable (10) content on Zenoh.
    pub encrypted_durable_announce: bool,
    /// Whether to announce public ephemeral (01) content on Zenoh.
    pub public_ephemeral_announce: bool,
}

impl Default for ContentPolicy {
    fn default() -> Self {
        Self {
            encrypted_durable_persist: false,
            encrypted_durable_announce: false,
            public_ephemeral_announce: true,
        }
    }
}
```

**Step 4: Run test to verify it passes**

Run: `cargo test -p harmony-content content_policy_defaults -- --no-capture`
Expected: PASS

**Step 5: Commit**

```bash
git add crates/harmony-content/src/storage_tier.rs
git commit -m "feat(content): add ContentPolicy struct with conservative defaults"
```

---

### Task 3: Wire ContentPolicy into StorageTier

**Files:**
- Modify: `crates/harmony-content/src/storage_tier.rs`

**Context:** `StorageTier::new()` currently takes `(store, budget)`. We need to add `policy: ContentPolicy` and store it on the struct. This task ONLY wires the field — enforcement logic comes in Tasks 4–6.

**Step 1: Write the failing test**

Add to `storage_tier.rs` tests:

```rust
#[test]
fn storage_tier_accepts_policy() {
    let budget = StorageBudget {
        cache_capacity: 100,
        max_pinned_bytes: 1_000_000,
    };
    let policy = ContentPolicy::default();
    let (tier, _) = StorageTier::new(MemoryBlobStore::new(), budget, policy);
    // Should compile and create successfully.
    assert_eq!(tier.metrics().queries_served, 0);
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p harmony-content storage_tier_accepts_policy -- --no-capture`
Expected: FAIL — `new()` takes 2 arguments, not 3

**Step 3: Write minimal implementation**

1. Add `policy: ContentPolicy` field to `StorageTier` struct.
2. Change `StorageTier::new()` signature to accept `policy: ContentPolicy`.
3. Store `policy` on the struct.
4. Add a `pub fn policy(&self) -> &ContentPolicy` accessor.
5. Update ALL existing tests and call sites to pass `ContentPolicy::default()` as the third argument to `new()`.
6. Update `NodeRuntime::new()` in `crates/harmony-node/src/runtime.rs` to pass `ContentPolicy::default()`.

**Step 4: Run tests to verify they pass**

Run: `cargo test -p harmony-content -- --no-capture`
Expected: ALL existing tests PASS (no behavior change, just plumbing)

Also: `cargo test -p harmony-node -- --no-capture`
Expected: ALL existing tests PASS

**Step 5: Commit**

```bash
git add crates/harmony-content/src/storage_tier.rs crates/harmony-node/src/runtime.rs crates/harmony-node/src/main.rs
git commit -m "refactor(content): wire ContentPolicy into StorageTier constructor"
```

---

### Task 4: Admission gating by content class

**Files:**
- Modify: `crates/harmony-content/src/storage_tier.rs`

**Context:** `handle_transit()` (line 180) currently does: verify_cid → should_admit → store_preadmitted → announce. `handle_publish()` (line 202) does: verify_cid → store → announce. We need to add class-based admission checks BEFORE the W-TinyLFU admission check.

Policy rules from the design:
- `EncryptedEphemeral (11)` → always reject (guard — should never arrive)
- `EncryptedDurable (10)` + `!encrypted_durable_persist` → reject
- Everything else → proceed to existing logic

Both `handle_transit` AND `handle_publish` must respect class policy. `PublishContent` still bypasses W-TinyLFU but must respect class rejection.

**Step 1: Write the failing tests**

Add to `storage_tier.rs` tests. Helper function first:

```rust
fn make_tier_with_policy(policy: ContentPolicy) -> StorageTier<MemoryBlobStore> {
    let budget = StorageBudget {
        cache_capacity: 100,
        max_pinned_bytes: 1_000_000,
    };
    let (tier, _) = StorageTier::new(MemoryBlobStore::new(), budget, policy);
    tier
}

fn cid_with_class(data: &[u8], encrypted: bool, ephemeral: bool) -> (ContentId, Vec<u8>) {
    let flags = crate::cid::ContentFlags { encrypted, ephemeral, alt_hash: false };
    let cid = ContentId::for_blob(data, flags).unwrap();
    (cid, data.to_vec())
}
```

```rust
#[test]
fn transit_rejects_encrypted_ephemeral() {
    let mut tier = make_tier_with_policy(ContentPolicy::default());
    let (cid, data) = cid_with_class(b"secret stream", true, true);
    let actions = tier.handle(StorageTierEvent::TransitContent { cid, data });
    assert!(actions.is_empty());
    assert_eq!(tier.metrics().transit_rejected, 1);
}

#[test]
fn publish_rejects_encrypted_ephemeral() {
    let mut tier = make_tier_with_policy(ContentPolicy::default());
    let (cid, data) = cid_with_class(b"secret stream", true, true);
    let actions = tier.handle(StorageTierEvent::PublishContent { cid, data });
    assert!(actions.is_empty());
    assert_eq!(tier.metrics().publishes_rejected, 1);
}

#[test]
fn transit_rejects_encrypted_durable_when_policy_off() {
    let policy = ContentPolicy {
        encrypted_durable_persist: false,
        ..ContentPolicy::default()
    };
    let mut tier = make_tier_with_policy(policy);
    let (cid, data) = cid_with_class(b"encrypted file", true, false);
    let actions = tier.handle(StorageTierEvent::TransitContent { cid, data });
    assert!(actions.is_empty());
    assert_eq!(tier.metrics().transit_rejected, 1);
}

#[test]
fn transit_admits_encrypted_durable_when_policy_on() {
    let policy = ContentPolicy {
        encrypted_durable_persist: true,
        ..ContentPolicy::default()
    };
    let mut tier = make_tier_with_policy(policy);
    let (cid, data) = cid_with_class(b"encrypted file", true, false);
    let actions = tier.handle(StorageTierEvent::TransitContent { cid, data });
    assert_eq!(actions.len(), 1);
    assert!(matches!(&actions[0], StorageTierAction::AnnounceContent { .. }));
    assert_eq!(tier.metrics().transit_stored, 1);
}

#[test]
fn transit_admits_public_durable() {
    let mut tier = make_tier_with_policy(ContentPolicy::default());
    let (cid, data) = cid_with_class(b"public doc", false, false);
    let actions = tier.handle(StorageTierEvent::TransitContent { cid, data });
    assert_eq!(actions.len(), 1);
    assert_eq!(tier.metrics().transit_stored, 1);
}

#[test]
fn transit_admits_public_ephemeral() {
    let mut tier = make_tier_with_policy(ContentPolicy::default());
    let (cid, data) = cid_with_class(b"live stream", false, true);
    let actions = tier.handle(StorageTierEvent::TransitContent { cid, data });
    assert_eq!(actions.len(), 1);
    assert_eq!(tier.metrics().transit_stored, 1);
}
```

**Step 2: Run tests to verify they fail**

Run: `cargo test -p harmony-content transit_rejects_encrypted_ephemeral -- --no-capture`
Expected: FAIL — encrypted ephemeral content is currently admitted

**Step 3: Write minimal implementation**

Add a private method to `StorageTier`:

```rust
/// Check whether a CID's content class is admissible under the current policy.
///
/// Returns `true` if the content should proceed to W-TinyLFU / storage.
/// Returns `false` if the content class is rejected by policy.
fn class_admits(&self, cid: &ContentId) -> bool {
    match cid.content_class() {
        ContentClass::EncryptedEphemeral => false,
        ContentClass::EncryptedDurable => self.policy.encrypted_durable_persist,
        ContentClass::PublicDurable | ContentClass::PublicEphemeral => true,
    }
}
```

Insert the class check at the TOP of `handle_transit()` (after `verify_cid`, before `should_admit`):

```rust
fn handle_transit(&mut self, cid: ContentId, data: Vec<u8>) -> Vec<StorageTierAction> {
    if !Self::verify_cid(&cid, &data) {
        self.metrics.transit_rejected += 1;
        return vec![];
    }
    if !self.class_admits(&cid) {
        self.metrics.transit_rejected += 1;
        return vec![];
    }
    // ... existing should_admit / store_preadmitted / announce logic
}
```

Same pattern for `handle_publish()` (after `verify_cid`, before `store`):

```rust
fn handle_publish(&mut self, cid: ContentId, data: Vec<u8>) -> Vec<StorageTierAction> {
    if !Self::verify_cid(&cid, &data) {
        self.metrics.publishes_rejected += 1;
        return vec![];
    }
    if !self.class_admits(&cid) {
        self.metrics.publishes_rejected += 1;
        return vec![];
    }
    // ... existing store / announce logic
}
```

**Step 4: Run tests to verify they pass**

Run: `cargo test -p harmony-content -- --no-capture`
Expected: ALL tests PASS (new + existing)

**Step 5: Commit**

```bash
git add crates/harmony-content/src/storage_tier.rs
git commit -m "feat(content): admission gating by content class and policy"
```

---

### Task 5: Class-aware eviction priority

**Files:**
- Modify: `crates/harmony-content/src/cache.rs`

**Context:** The W-TinyLFU admission challenge in `admission_challenge()` (line 270) currently compares candidate vs victim purely by frequency. The design calls for class-based tier ordering:
1. PublicEphemeral (01) — evicted first
2. EncryptedDurable (10) — evicted second
3. PublicDurable (00) — evicted last

Rule: if the candidate's class is higher-priority (lower eviction order) than the victim's, the victim is evicted regardless of frequency.

**Step 1: Write the failing tests**

Add to `cache.rs` tests:

```rust
#[test]
fn ephemeral_evicted_before_durable() {
    // Fill cache with a mix of PublicDurable and PublicEphemeral.
    // Under pressure, PublicEphemeral items should be evicted first.
    //
    // Capacity 5: window=1, protected=1, probation=3.
    let store = MemoryBlobStore::new();
    let mut cs = ContentStore::new(store, 5);

    // Insert a PublicDurable item.
    let durable_flags = ContentFlags { encrypted: false, ephemeral: false, alt_hash: false };
    let durable_cid = cs.insert_with_flags(b"durable-data", durable_flags).unwrap();

    // Insert a PublicEphemeral item.
    let ephemeral_flags = ContentFlags { encrypted: false, ephemeral: true, alt_hash: false };
    let ephemeral_cid = cs.insert_with_flags(b"ephemeral-data", ephemeral_flags).unwrap();

    // Both should be in cache.
    assert!(cs.is_admitted(&durable_cid));
    assert!(cs.is_admitted(&ephemeral_cid));

    // Give the ephemeral item MORE frequency than durable (to prove class trumps frequency).
    for _ in 0..10 {
        cs.record_access(&ephemeral_cid);
    }

    // Fill cache to trigger eviction pressure.
    for i in 0..10 {
        let data = format!("pressure-{i}");
        cs.insert(data.as_bytes()).unwrap();
    }

    // Durable should survive, ephemeral should be evicted (class priority trumps frequency).
    assert!(
        cs.is_admitted(&durable_cid),
        "PublicDurable should survive eviction pressure"
    );
    assert!(
        !cs.is_admitted(&ephemeral_cid),
        "PublicEphemeral should be evicted before PublicDurable even with higher frequency"
    );
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p harmony-content ephemeral_evicted_before_durable -- --no-capture`
Expected: FAIL — ephemeral item survives because it has higher frequency

**Step 3: Write minimal implementation**

Add an `eviction_priority()` helper (higher number = evicted first):

```rust
use crate::cid::ContentClass;

impl ContentClass {
    /// Eviction priority: higher values are evicted first under pressure.
    ///
    /// PublicEphemeral (disposable-first) > EncryptedDurable > PublicDurable (most valuable).
    /// EncryptedEphemeral never enters the cache, so its priority is irrelevant.
    pub fn eviction_priority(self) -> u8 {
        match self {
            ContentClass::PublicEphemeral => 2,
            ContentClass::EncryptedDurable => 1,
            ContentClass::PublicDurable => 0,
            ContentClass::EncryptedEphemeral => u8::MAX, // unreachable in cache
        }
    }
}
```

Place `eviction_priority()` in `cid.rs` (on the `ContentClass` enum's impl block).

Then modify `admission_challenge()` in `cache.rs` to check class priority before frequency:

```rust
fn admission_challenge(&mut self, candidate: ContentId) {
    if self.probation.len() < self.probation.capacity() {
        self.probation.insert(candidate);
        return;
    }

    if self.pinned.contains(&candidate) {
        let Some(victim) = self.probation.peek_lru_excluding(&self.pinned) else {
            return;
        };
        self.probation.remove(&victim);
        self.store_remove(&victim);
        self.probation.insert(candidate);
        return;
    }

    let Some(victim) = self.probation.peek_lru_excluding(&self.pinned) else {
        self.store_remove(&candidate);
        return;
    };

    let candidate_class = candidate.content_class().eviction_priority();
    let victim_class = victim.content_class().eviction_priority();

    if candidate_class < victim_class {
        // Candidate is higher-value class — evict victim regardless of frequency.
        self.probation.remove(&victim);
        self.store_remove(&victim);
        self.probation.insert(candidate);
    } else if candidate_class > victim_class {
        // Candidate is lower-value class — drop it regardless of frequency.
        self.store_remove(&candidate);
    } else {
        // Same class — fall back to frequency comparison.
        let candidate_freq = self.sketch.estimate(&candidate);
        let victim_freq = self.sketch.estimate(&victim);
        if candidate_freq > victim_freq {
            self.probation.remove(&victim);
            self.store_remove(&victim);
            self.probation.insert(candidate);
        } else {
            self.store_remove(&candidate);
        }
    }
}
```

Also update `should_admit()` to factor in class priority when comparing against the probation victim:

```rust
pub fn should_admit(&mut self, cid: &ContentId) -> bool {
    if self.window.contains(cid) || self.probation.contains(cid) || self.protected.contains(cid) {
        return true;
    }
    self.sketch.increment(cid);
    if self.probation.len() < self.probation.capacity() {
        return true;
    }

    let candidate_class = cid.content_class().eviction_priority();
    match self.probation.peek_lru_excluding(&self.pinned) {
        Some(victim) => {
            let victim_class = victim.content_class().eviction_priority();
            if candidate_class < victim_class {
                true // higher-value class always admitted
            } else if candidate_class > victim_class {
                false // lower-value class always rejected
            } else {
                self.sketch.estimate(cid) > self.sketch.estimate(&victim)
            }
        }
        None => true,
    }
}
```

**Step 4: Run tests to verify they pass**

Run: `cargo test -p harmony-content -- --no-capture`
Expected: ALL tests PASS

**Step 5: Commit**

```bash
git add crates/harmony-content/src/cid.rs crates/harmony-content/src/cache.rs
git commit -m "feat(content): class-aware eviction priority in W-TinyLFU"
```

---

### Task 6: Announcement gating by content class

**Files:**
- Modify: `crates/harmony-content/src/storage_tier.rs`

**Context:** After successful storage, `handle_transit()` and `handle_publish()` unconditionally call `make_announce_action()`. The design requires conditional announcement:
- PublicDurable (00) → always announce
- PublicEphemeral (01) → announce if `public_ephemeral_announce`
- EncryptedDurable (10) → announce if `encrypted_durable_announce`
- EncryptedEphemeral (11) → never (unreachable — rejected at admission)

**Step 1: Write the failing tests**

```rust
#[test]
fn transit_public_ephemeral_no_announce_when_policy_off() {
    let policy = ContentPolicy {
        public_ephemeral_announce: false,
        ..ContentPolicy::default()
    };
    let mut tier = make_tier_with_policy(policy);
    let (cid, data) = cid_with_class(b"live stream", false, true);
    let actions = tier.handle(StorageTierEvent::TransitContent { cid, data });
    // Should store but NOT announce.
    assert!(actions.is_empty(), "no announce when policy off");
    assert_eq!(tier.metrics().transit_stored, 1);
}

#[test]
fn transit_public_ephemeral_announces_when_policy_on() {
    let policy = ContentPolicy {
        public_ephemeral_announce: true,
        ..ContentPolicy::default()
    };
    let mut tier = make_tier_with_policy(policy);
    let (cid, data) = cid_with_class(b"live stream", false, true);
    let actions = tier.handle(StorageTierEvent::TransitContent { cid, data });
    assert_eq!(actions.len(), 1);
    assert!(matches!(&actions[0], StorageTierAction::AnnounceContent { .. }));
}

#[test]
fn transit_encrypted_durable_no_announce_when_policy_off() {
    let policy = ContentPolicy {
        encrypted_durable_persist: true,  // must persist to reach announce
        encrypted_durable_announce: false,
        ..ContentPolicy::default()
    };
    let mut tier = make_tier_with_policy(policy);
    let (cid, data) = cid_with_class(b"encrypted doc", true, false);
    let actions = tier.handle(StorageTierEvent::TransitContent { cid, data });
    assert!(actions.is_empty(), "stored but not announced");
    assert_eq!(tier.metrics().transit_stored, 1);
}

#[test]
fn transit_encrypted_durable_announces_when_policy_on() {
    let policy = ContentPolicy {
        encrypted_durable_persist: true,
        encrypted_durable_announce: true,
        ..ContentPolicy::default()
    };
    let mut tier = make_tier_with_policy(policy);
    let (cid, data) = cid_with_class(b"encrypted doc", true, false);
    let actions = tier.handle(StorageTierEvent::TransitContent { cid, data });
    assert_eq!(actions.len(), 1);
    assert!(matches!(&actions[0], StorageTierAction::AnnounceContent { .. }));
}

#[test]
fn transit_public_durable_always_announces() {
    let mut tier = make_tier_with_policy(ContentPolicy::default());
    let (cid, data) = cid_with_class(b"valuable doc", false, false);
    let actions = tier.handle(StorageTierEvent::TransitContent { cid, data });
    assert_eq!(actions.len(), 1);
    assert!(matches!(&actions[0], StorageTierAction::AnnounceContent { .. }));
}
```

**Step 2: Run tests to verify they fail**

Run: `cargo test -p harmony-content transit_public_ephemeral_no_announce -- --no-capture`
Expected: FAIL — currently always announces

**Step 3: Write minimal implementation**

Add a private method to `StorageTier`:

```rust
/// Check whether a CID's content class should be announced on Zenoh.
fn should_announce(&self, cid: &ContentId) -> bool {
    match cid.content_class() {
        ContentClass::PublicDurable => true,
        ContentClass::PublicEphemeral => self.policy.public_ephemeral_announce,
        ContentClass::EncryptedDurable => self.policy.encrypted_durable_announce,
        ContentClass::EncryptedEphemeral => false,
    }
}
```

Replace the unconditional `vec![self.make_announce_action(&cid)]` at the end of `handle_transit()` and `handle_publish()` with:

```rust
if self.should_announce(&cid) {
    vec![self.make_announce_action(&cid)]
} else {
    vec![]
}
```

**Step 4: Run tests to verify they pass**

Run: `cargo test -p harmony-content -- --no-capture`
Expected: ALL tests PASS

**Step 5: Commit**

```bash
git add crates/harmony-content/src/storage_tier.rs
git commit -m "feat(content): announcement gating by content class and policy"
```

---

### Task 7: TieredBlobStore with disk actions/events (sans-I/O)

**Files:**
- Modify: `crates/harmony-content/src/storage_tier.rs`

**Context:** The design calls for `PersistToDisk`/`RemoveFromDisk` actions and a `DiskReadComplete` event. The StorageTier needs to emit disk persistence actions for durable classes and handle disk read responses for cache misses on items known to be on disk.

Currently `StorageTierAction` has: `SendReply`, `AnnounceContent`, `SendStatsReply`, `DeclareQueryables`, `DeclareSubscribers`.
Currently `StorageTierEvent` has: `ContentQuery`, `TransitContent`, `PublishContent`, `StatsQuery`.

**Step 1: Write the failing tests**

```rust
#[test]
fn transit_public_durable_emits_persist_to_disk() {
    let mut tier = make_tier_with_policy(ContentPolicy::default());
    let (cid, data) = cid_with_class(b"durable content", false, false);
    let actions = tier.handle(StorageTierEvent::TransitContent { cid, data: data.clone() });
    let persist_actions: Vec<_> = actions.iter()
        .filter(|a| matches!(a, StorageTierAction::PersistToDisk { .. }))
        .collect();
    assert_eq!(persist_actions.len(), 1, "durable content should emit PersistToDisk");
    match &persist_actions[0] {
        StorageTierAction::PersistToDisk { cid: pcid, data: pdata } => {
            assert_eq!(*pcid, cid);
            assert_eq!(pdata, &data);
        }
        _ => unreachable!(),
    }
}

#[test]
fn transit_public_ephemeral_no_persist_to_disk() {
    let mut tier = make_tier_with_policy(ContentPolicy::default());
    let (cid, data) = cid_with_class(b"ephemeral content", false, true);
    let actions = tier.handle(StorageTierEvent::TransitContent { cid, data });
    let persist_actions: Vec<_> = actions.iter()
        .filter(|a| matches!(a, StorageTierAction::PersistToDisk { .. }))
        .collect();
    assert!(persist_actions.is_empty(), "ephemeral content should NOT emit PersistToDisk");
}

#[test]
fn disk_read_complete_serves_queued_query() {
    let mut tier = make_tier_with_policy(ContentPolicy::default());
    let (cid, data) = cid_with_class(b"on-disk content", false, false);

    // Store and let it be persisted.
    tier.handle(StorageTierEvent::TransitContent { cid, data: data.clone() });

    // Simulate eviction from memory cache — query produces miss → DiskLookup.
    // For now, clear the in-memory cache by overflowing it.
    // (The real flow: query miss → check disk_index → emit DiskLookup → DiskReadComplete)
    // This test verifies DiskReadComplete re-populates cache and serves reply.
    let actions = tier.handle(StorageTierEvent::DiskReadComplete {
        cid,
        query_id: 42,
        data: data.clone(),
    });

    let reply_actions: Vec<_> = actions.iter()
        .filter(|a| matches!(a, StorageTierAction::SendReply { .. }))
        .collect();
    assert_eq!(reply_actions.len(), 1);
    match &reply_actions[0] {
        StorageTierAction::SendReply { query_id, payload } => {
            assert_eq!(*query_id, 42);
            assert_eq!(payload, &data);
        }
        _ => unreachable!(),
    }
}
```

**Step 2: Run tests to verify they fail**

Run: `cargo test -p harmony-content transit_public_durable_emits_persist -- --no-capture`
Expected: FAIL — `PersistToDisk` variant doesn't exist

**Step 3: Write minimal implementation**

Add new variants to `StorageTierAction`:

```rust
/// Persist content to disk (durable classes only). The runtime handles actual I/O.
PersistToDisk { cid: ContentId, data: Vec<u8> },
/// Remove content from disk. The runtime handles actual I/O.
RemoveFromDisk { cid: ContentId },
/// Request disk read for a CID known to be on disk but evicted from memory.
DiskLookup { cid: ContentId, query_id: u64 },
```

Add new variant to `StorageTierEvent`:

```rust
/// Disk read completed — runtime delivers data read from disk.
DiskReadComplete { cid: ContentId, query_id: u64, data: Vec<u8> },
```

Add a `disk_index: HashSet<ContentId>` field to `StorageTier`.

Add a private method to check if a class is durable:

```rust
fn is_durable_class(cid: &ContentId) -> bool {
    matches!(cid.content_class(), ContentClass::PublicDurable | ContentClass::EncryptedDurable)
}
```

In `handle_transit()` and `handle_publish()`, after storing, emit `PersistToDisk` for durable classes:

```rust
// After store_preadmitted / store:
if Self::is_durable_class(&cid) {
    self.disk_index.insert(cid);
    actions.push(StorageTierAction::PersistToDisk { cid, data });
}
```

In `handle_content_query()`, on cache miss, check `disk_index`:

```rust
None => {
    self.metrics.cache_misses += 1;
    if self.disk_index.contains(cid) {
        vec![StorageTierAction::DiskLookup { cid: *cid, query_id }]
    } else {
        vec![]
    }
}
```

Handle `DiskReadComplete` in the `handle()` match:

```rust
StorageTierEvent::DiskReadComplete { cid, query_id, data } => {
    // Re-cache the data from disk.
    self.cache.store(cid, data.clone());
    vec![StorageTierAction::SendReply { query_id, payload: data }]
}
```

**Step 4: Run tests to verify they pass**

Run: `cargo test -p harmony-content -- --no-capture`
Expected: ALL tests PASS

**Step 5: Commit**

```bash
git add crates/harmony-content/src/storage_tier.rs
git commit -m "feat(content): sans-I/O disk persistence actions and events"
```

---

### Task 8: CLI flags for ContentPolicy

**Files:**
- Modify: `crates/harmony-node/src/main.rs`
- Modify: `crates/harmony-node/src/runtime.rs`

**Context:** `Commands::Run` currently has `cache_capacity` and `compute_budget` args. We need three new CLI flags:
- `--encrypted-durable-persist` (default: false)
- `--encrypted-durable-announce` (default: false)
- `--public-ephemeral-announce` (default: true)

`NodeConfig` in `runtime.rs` needs a `content_policy: ContentPolicy` field so the runtime can pass it to `StorageTier::new()`.

**Step 1: Write the failing tests**

Add to `main.rs` tests:

```rust
#[test]
fn cli_parses_run_with_policy_flags() {
    let cli = Cli::try_parse_from([
        "harmony", "run",
        "--encrypted-durable-persist",
        "--encrypted-durable-announce",
        "--no-public-ephemeral-announce",
    ]).unwrap();
    if let Commands::Run {
        encrypted_durable_persist,
        encrypted_durable_announce,
        public_ephemeral_announce,
        ..
    } = cli.command {
        assert!(encrypted_durable_persist);
        assert!(encrypted_durable_announce);
        assert!(!public_ephemeral_announce);
    } else {
        panic!("expected Run command");
    }
}

#[test]
fn cli_policy_defaults() {
    let cli = Cli::try_parse_from(["harmony", "run"]).unwrap();
    if let Commands::Run {
        encrypted_durable_persist,
        encrypted_durable_announce,
        public_ephemeral_announce,
        ..
    } = cli.command {
        assert!(!encrypted_durable_persist);
        assert!(!encrypted_durable_announce);
        assert!(public_ephemeral_announce);
    } else {
        panic!("expected Run command");
    }
}
```

Add to `runtime.rs` tests:

```rust
#[test]
fn runtime_uses_content_policy() {
    use harmony_content::storage_tier::ContentPolicy;
    let config = NodeConfig {
        storage_budget: StorageBudget { cache_capacity: 100, max_pinned_bytes: 1_000_000 },
        compute_budget: InstructionBudget { fuel: 1000 },
        schedule: Default::default(),
        content_policy: ContentPolicy {
            encrypted_durable_persist: true,
            encrypted_durable_announce: true,
            public_ephemeral_announce: false,
        },
    };
    let (rt, _) = NodeRuntime::new(config, MemoryBlobStore::new());
    assert_eq!(rt.storage_queue_len(), 0);
}
```

**Step 2: Run tests to verify they fail**

Run: `cargo test -p harmony-node cli_parses_run_with_policy -- --no-capture`
Expected: FAIL — fields don't exist

**Step 3: Write minimal implementation**

1. Add `content_policy: ContentPolicy` to `NodeConfig`.
2. Update `NodeRuntime::new()` to pass `config.content_policy` to `StorageTier::new()`.
3. Update `make_runtime()` test helper in `runtime.rs` to include `content_policy: ContentPolicy::default()`.
4. Add CLI args to `Commands::Run`:

```rust
/// Accept encrypted durable (10) content for storage
#[arg(long, default_value_t = false)]
encrypted_durable_persist: bool,
/// Announce encrypted durable (10) content on Zenoh
#[arg(long, default_value_t = false)]
encrypted_durable_announce: bool,
/// Announce public ephemeral (01) content on Zenoh
#[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
public_ephemeral_announce: bool,
```

5. Wire the CLI values into `ContentPolicy` in the `Commands::Run` handler.

**Step 4: Run tests to verify they pass**

Run: `cargo test -p harmony-node -- --no-capture`
Expected: ALL tests PASS

Also: `cargo clippy --workspace`
Expected: zero warnings

**Step 5: Commit**

```bash
git add crates/harmony-node/src/main.rs crates/harmony-node/src/runtime.rs
git commit -m "feat(node): CLI flags for content policy configuration"
```

---

### Task 9: Integration tests

**Files:**
- Modify: `crates/harmony-content/src/storage_tier.rs` (add tests to existing test module)

**Context:** The design calls for integration tests covering full lifecycle scenarios. These exercise multiple policy enforcement points together.

**Step 1: Write integration tests**

```rust
#[test]
fn full_durable_lifecycle_store_evict_disk_serve() {
    // Store durable content → verify PersistToDisk emitted →
    // query after memory eviction → DiskLookup emitted →
    // DiskReadComplete → reply served.
    let mut tier = make_tier_with_policy(ContentPolicy::default());
    let (cid, data) = cid_with_class(b"durable lifecycle", false, false);

    // Store via transit.
    let actions = tier.handle(StorageTierEvent::TransitContent { cid, data: data.clone() });
    assert!(actions.iter().any(|a| matches!(a, StorageTierAction::PersistToDisk { .. })));
    assert!(actions.iter().any(|a| matches!(a, StorageTierAction::AnnounceContent { .. })));

    // Query should hit in-memory cache.
    let query_actions = tier.handle(StorageTierEvent::ContentQuery { query_id: 1, cid });
    assert!(query_actions.iter().any(|a| matches!(a, StorageTierAction::SendReply { .. })));

    // Simulate: DiskReadComplete after eviction (pretend cache evicted it).
    let disk_actions = tier.handle(StorageTierEvent::DiskReadComplete {
        cid, query_id: 99, data: data.clone(),
    });
    assert!(disk_actions.iter().any(|a| matches!(a, StorageTierAction::SendReply { query_id: 99, .. })));
}

#[test]
fn mixed_class_eviction_ephemeral_first() {
    // Fill cache with PublicDurable + PublicEphemeral, verify ephemeral evicted first.
    let budget = StorageBudget { cache_capacity: 5, max_pinned_bytes: 1_000_000 };
    let (mut tier, _) = StorageTier::new(
        MemoryBlobStore::new(), budget, ContentPolicy::default(),
    );

    // Store 2 durable items.
    let durable_flags = crate::cid::ContentFlags { encrypted: false, ephemeral: false, alt_hash: false };
    let d1_data = b"durable-1";
    let d1_cid = ContentId::for_blob(d1_data, durable_flags).unwrap();
    tier.handle(StorageTierEvent::PublishContent { cid: d1_cid, data: d1_data.to_vec() });

    let d2_data = b"durable-2";
    let d2_cid = ContentId::for_blob(d2_data, durable_flags).unwrap();
    tier.handle(StorageTierEvent::PublishContent { cid: d2_cid, data: d2_data.to_vec() });

    // Store 2 ephemeral items.
    let eph_flags = crate::cid::ContentFlags { encrypted: false, ephemeral: true, alt_hash: false };
    let e1_data = b"ephemeral-1";
    let e1_cid = ContentId::for_blob(e1_data, eph_flags).unwrap();
    tier.handle(StorageTierEvent::PublishContent { cid: e1_cid, data: e1_data.to_vec() });

    let e2_data = b"ephemeral-2";
    let e2_cid = ContentId::for_blob(e2_data, eph_flags).unwrap();
    tier.handle(StorageTierEvent::PublishContent { cid: e2_cid, data: e2_data.to_vec() });

    // Fill cache to trigger evictions.
    for i in 0..10 {
        let data = format!("pressure-{i}");
        let cid = ContentId::for_blob(data.as_bytes(), durable_flags).unwrap();
        tier.handle(StorageTierEvent::TransitContent { cid, data: data.into_bytes() });
    }

    // Durable items should survive; at least one ephemeral should be gone.
    let d1_hit = tier.handle(StorageTierEvent::ContentQuery { query_id: 1, cid: d1_cid });
    let d2_hit = tier.handle(StorageTierEvent::ContentQuery { query_id: 2, cid: d2_cid });
    // Durable items either in memory (SendReply) or on disk (DiskLookup) — either way, tracked.
    assert!(
        !d1_hit.is_empty() || !d2_hit.is_empty(),
        "at least one durable item should be retrievable"
    );
}

#[test]
fn policy_toggle_encrypted_durable() {
    // With policy off: encrypted durable rejected.
    let policy_off = ContentPolicy {
        encrypted_durable_persist: false,
        ..ContentPolicy::default()
    };
    let mut tier_off = make_tier_with_policy(policy_off);
    let (cid, data) = cid_with_class(b"encrypted file", true, false);
    let actions = tier_off.handle(StorageTierEvent::TransitContent { cid, data: data.clone() });
    assert!(actions.is_empty());
    assert_eq!(tier_off.metrics().transit_rejected, 1);

    // With policy on: encrypted durable admitted.
    let policy_on = ContentPolicy {
        encrypted_durable_persist: true,
        encrypted_durable_announce: true,
        ..ContentPolicy::default()
    };
    let mut tier_on = make_tier_with_policy(policy_on);
    let actions = tier_on.handle(StorageTierEvent::TransitContent { cid, data });
    assert!(!actions.is_empty());
    assert_eq!(tier_on.metrics().transit_stored, 1);
}
```

**Step 2: Run tests**

Run: `cargo test -p harmony-content -- --no-capture`
Expected: ALL tests PASS

**Step 3: Run full workspace quality gates**

Run: `cargo test --workspace`
Run: `cargo clippy --workspace`
Expected: ALL PASS, zero warnings

**Step 4: Commit**

```bash
git add crates/harmony-content/src/storage_tier.rs
git commit -m "test(content): integration tests for CID policy enforcement"
```

---

### Task 10: Final verification

**Step 1: Run full test suite**

Run: `cargo test --workspace`
Expected: ALL tests pass

**Step 2: Run clippy**

Run: `cargo clippy --workspace`
Expected: zero warnings

**Step 3: Run format check**

Run: `cargo fmt --all -- --check`
Expected: no formatting issues
