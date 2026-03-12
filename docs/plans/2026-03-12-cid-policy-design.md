# CID Storage/Publishing Policy Enforcement Design

> **Bead:** harmony-8fg — Storage/publishing policy enforcement for CID classification bits

**Goal:** Enforce storage, eviction, and Zenoh publishing rules based on the leading 2 bits of ContentId (`encrypted`, `ephemeral`), turning the existing classification flags into actionable policy at the StorageTier level.

**Approach:** Policy-aware StorageTier (Approach B) — extend StorageTier to consult a ContentPolicy config at three decision points: admission, eviction, and announcement.

## Content Classes

The `(encrypted, ephemeral)` bits in `ContentFlags` (byte 0 of every CID) define four content classes:

| Bits | Class | Storage | Zenoh | Reticulum |
|------|-------|---------|-------|-----------|
| `00` | PublicDurable | Disk-backed (LFU-managed) | Always announce | Allowed |
| `01` | PublicEphemeral | Memory-only, evict-first | Default on, opt-out | Allowed |
| `10` | EncryptedDurable | Configurable per-device | Configurable | Allowed |
| `11` | EncryptedEphemeral | Never persist | Never announce | P2P only |

### Class Semantics

**PublicDurable (00):** Valuable public information. When it crosses your router, keep a copy on disk so whoever sent it doesn't have to send it again. LFU/LRU manages disk utilization — this is not "pinning" (which means "never evict, period"), it's "worth keeping as long as space allows."

**PublicEphemeral (01):** Disposable-first content. Live streams, broadcasts, IoT telemetry, status reports. Best-effort resharing if you have capacity and someone downstream wants it. When you need to free space, this is what goes first and no one should be upset.

**EncryptedDurable (10):** Content with an intentional relationship behind it. Either you're the consumer (you hold the ciphertext, a platform holds the decryption key) or you're the custodian (a friend entrusted it to you for safekeeping). If neither applies, it's stray data — the conservative default is to reject it. Configurable per-device because the decision depends on context.

**EncryptedEphemeral (11):** Maximum privacy. Encrypted streams/calls, OTPs, session keys. Never touches disk, never enters Zenoh, lives only in volatile memory. The overhead of keeping it off Zenoh and off disk is exactly what protecting privacy looks like. Handled by direct P2P transport (Reticulum path), bypassing StorageTier entirely.

## Data Model

```rust
/// Content class derived from the two leading classification bits of a CID.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ContentClass {
    PublicDurable,       // 00
    PublicEphemeral,     // 01
    EncryptedDurable,    // 10
    EncryptedEphemeral,  // 11
}

impl ContentId {
    pub fn content_class(&self) -> ContentClass {
        let flags = self.flags();
        match (flags.encrypted, flags.ephemeral) {
            (false, false) => ContentClass::PublicDurable,
            (false, true)  => ContentClass::PublicEphemeral,
            (true, false)  => ContentClass::EncryptedDurable,
            (true, true)   => ContentClass::EncryptedEphemeral,
        }
    }
}
```

## Policy Configuration

```rust
/// Per-class storage and publishing policy.
#[derive(Debug, Clone)]
pub struct ContentPolicy {
    /// 10-class: whether to persist encrypted durable content.
    pub encrypted_durable_persist: bool,
    /// 10-class: whether to announce encrypted durable content on Zenoh.
    pub encrypted_durable_announce: bool,
    /// 01-class: whether to announce public ephemeral content on Zenoh.
    pub public_ephemeral_announce: bool,
}
```

CLI flags (fed to `ContentPolicy`):
- `--encrypted-durable-persist <bool>` (default: `false`)
- `--encrypted-durable-announce <bool>` (default: `false`)
- `--public-ephemeral-announce <bool>` (default: `true`)

Defaults for `10`-class are conservative: don't store or announce encrypted content unless opted in. The struct is designed so richer rules (tag-based filtering, trust thresholds) can be added later without changing the StorageTier interface.

## StorageTier Policy Enforcement

Three decision points, all within the existing sans-I/O event/action flow:

### 1. Admission

When `TransitContent` or `PublishContent` arrives, classify and check policy before W-TinyLFU:

- `EncryptedEphemeral` → reject (guard — should never arrive)
- `EncryptedDurable` + `!encrypted_durable_persist` → reject
- Otherwise → proceed to W-TinyLFU `should_admit()`

`PublishContent` still bypasses W-TinyLFU admission (local apps) but respects class policy.

### 2. Eviction Priority

Class-based tier ordering, frequency breaks ties within a tier:

1. **PublicEphemeral (01)** — evicted first ("disposable first")
2. **EncryptedDurable (10)** — evicted second
3. **PublicDurable (00)** — evicted last (most valuable to the network)

In the W-TinyLFU admission challenge: if the candidate is higher-priority class than the victim, the victim is evicted regardless of frequency.

### 3. Announcement Gating

After successful storage, conditionally emit `AnnounceContent`:

- `PublicDurable (00)` → always announce
- `PublicEphemeral (01)` → announce if `public_ephemeral_announce`
- `EncryptedDurable (10)` → announce if `encrypted_durable_announce`
- `EncryptedEphemeral (11)` → never (unreachable)

## Disk-Backed Storage (Sans-I/O)

Two-tier BlobStore with memory cache + disk persistence:

```rust
pub struct TieredBlobStore {
    /// Hot cache — memory-resident items (all classes).
    memory: HashMap<ContentId, Vec<u8>>,
    /// Cold storage index — CIDs known to be on disk (durable classes only).
    disk_index: HashSet<ContentId>,
}
```

New actions and events for sans-I/O disk operations:

```rust
// New actions
StorageTierAction::PersistToDisk { cid, data }
StorageTierAction::RemoveFromDisk { cid }

// New events
StorageTierEvent::DiskReadComplete { cid, data }
```

**Durable flow:** Admitted → memory + `PersistToDisk` → evicted from memory → stays in `disk_index` → query hits disk index → runtime reads from disk → `DiskReadComplete` → re-cache + serve.

**Ephemeral flow:** Admitted → memory only → evicted → gone.

Actual file I/O is the runtime's responsibility (sans-I/O boundary). This bead defines the actions/events; the runtime wiring is minimal.

## EncryptedEphemeral Routing

`11`-class content never enters StorageTier. The routing decision happens in `NodeRuntime::push_event`:

```
Inbound content → content_class() == EncryptedEphemeral?
  → route to Reticulum P2P path (Tier 1 only)
  → never enqueue for Tier 2 (storage)
```

No new Reticulum transport code in this bead — the existing `pack_for_transport` / `unpack_from_transport` already handles CID + blob. PQ-encrypted P2P transport is a separate concern building on the hybrid KEM work (PR #65).

## What Changes Where

| Crate | File | Change |
|-------|------|--------|
| `harmony-content` | `cid.rs` | `ContentClass` enum, `ContentId::content_class()` |
| `harmony-content` | `storage_tier.rs` | `ContentPolicy`, admission/eviction/announcement gating, `PersistToDisk`/`RemoveFromDisk`/`DiskReadComplete` |
| `harmony-content` | `cache.rs` | Class-aware eviction priority in W-TinyLFU challenge |
| `harmony-node` | `runtime.rs` | `EncryptedEphemeral` dispatch gate, wire `ContentPolicy` |
| `harmony-node` | `main.rs` | CLI flags for policy config |

## Non-Goals

- Actual disk I/O implementation in the runtime (sans-I/O boundary)
- PQ-encrypted P2P transport for `11`-class (separate bead)
- Tag-based or trust-based `10`-class filtering (future `ContentPolicy` refinement)
- `11`-class kernel memory isolation (harmony-os concern)
- Config file format (CLI flags for now, TOML/YAML later)

## Testing

**Unit tests:**
- `content_class()` — all four `(encrypted, ephemeral)` combinations
- Admission gating — each class admitted/rejected per policy
- Eviction priority — `01` evicted before `00` under pressure
- Announcement gating — each class announced/suppressed per policy
- Disk actions — `PersistToDisk`/`RemoveFromDisk` emitted at correct lifecycle points
- `DiskReadComplete` — re-populates cache and serves queued reply

**Integration tests:**
- Full durable lifecycle: store → evict from memory → disk index retains → query → disk read → serve
- Mixed-class eviction: fill cache with `00` + `01`, verify `01` evicted first
- Policy toggle: `10`-class admitted when policy on, rejected when off
