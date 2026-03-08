# Roxy: Self-Sovereign Content Licensing & Distribution

**Status:** Approved design. Not yet implemented.

**Goal:** A library crate (`harmony-roxy`) that gives any Harmony app content licensing primitives — enabling artists to publish, license, and distribute content with zero middlemen, 100% revenue to creators, and automatic cache lifecycle management.

**Color mapping:** Red (manifest authoring), Blue (key wrapping/encryption), Yellow (catalog discovery), Cyan (UCAN access control), Green (cache eviction), Magenta (artist/consumer UX).

---

## Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Payment model | Capability-as-payment (UCAN-only) | Payment is out of scope — Roxy enforces "do you have a valid token?" and a separate crate/platform handles SOC 2 compliant payment. Decouples licensing from payment complexity. |
| Content encryption | Per-content symmetric key, wrapped per-consumer | Artist generates one ChaCha20-Poly1305 key per content piece. Consumers receive the key wrapped via ECDH with their X25519 public key. Cheap (one ECDH + 32-byte encryption per grant), maps to existing harmony-identity/crypto primitives. |
| Cache eviction | Immediate on expiry | No grace period. Green wipes cached key and blob the moment the UCAN expires. Consumers are notified a configurable window before expiry. Auto-renew handles seamless continuity. Honest, no dark patterns. |
| License terms storage | Separate License Manifest (CID-addressed blob) | UCAN tokens stay lightweight (authorization only). Rich terms live in a signed, immutable manifest blob that Yellow can index for search. New terms = new manifest CID; existing grants reference old terms. |
| Content granularity | CID or bundle CID | A manifest can reference a single content CID or a bundle CID. Licensing a bundle grants access to all children. One UCAN covers a whole album. Artists can also license individual tracks separately. |
| Discovery | Zenoh key expressions | Artists publish catalogs to structured Zenoh patterns. Reticulum handles identity/presence ("this artist is online"); Zenoh handles catalog browsing and subscription matching. Yellow indexes manifests for semantic search. |
| Architecture | Library crate + app | `harmony-roxy` is a sans-I/O library crate. Protocol logic as pure state machines. Any Harmony app can embed licensing (glitch premium content, mail attachments, etc.). App layer wires to real I/O. |

---

## License Manifest

The core data structure artists publish to describe their content's licensing terms. Serialized as CBOR, content-addressed (gets its own CID), signed by the artist.

```rust
LicenseManifest {
    // Identity
    creator:          [u8; 16],       // Artist's address hash
    content_cid:      ContentId,      // Single CID or bundle CID
    manifest_version: u8,             // Schema version for Blue migrations

    // Terms
    license_type:     LicenseType,    // Free, OneTime, Subscription, Custom
    price:            Option<Price>,  // { amount: u64, currency: String, per: PricePer }
    duration:         Option<Duration>, // For subscriptions: access window per grant
    usage_rights:     UsageRights,    // Bitflags: STREAM, DOWNLOAD, REMIX, RESHARE
    expiry_notice:    u32,            // Seconds before expiry to notify consumer

    // Crypto
    content_key_cid:  ContentId,      // CID of the encrypted symmetric key blob
    signature:        [u8; 64],       // Ed25519 signature over the above fields
}
```

### LicenseType

- **`Free`** — No UCAN needed. Content key is published unencrypted. Green still tracks for dedup.
- **`OneTime`** — Pay once, access forever. UCAN has no `not_after`.
- **`Subscription`** — Recurring access windows. UCAN has `not_after`, consumer renews.
- **`Custom`** — Opaque terms referencing an external contract. Future-proofing for WASM policy workflows.

### UsageRights (bitflags)

- **`STREAM`** — Decrypt and play in real-time. No local persistence beyond cache.
- **`DOWNLOAD`** — Persist decrypted content locally. Green does NOT evict on expiry.
- **`REMIX`** — Derive new content from this content.
- **`RESHARE`** — Delegate access to others via UCAN delegation chain.

Manifests are immutable. Updated terms = new manifest CID. Old grants still reference old terms, so changing terms never breaks existing licenses.

---

## Key Wrapping & Access Flow

### Artist Publishes Content

```
1. Artist creates content (e.g., a song)
2. Generate random symmetric key K (ChaCha20-Poly1305)
3. Encrypt content with K → encrypted blob
4. Store encrypted blob → content_cid (encrypted flag set)
5. Store K encrypted with artist's own X25519 key → content_key_cid
6. Create & sign LicenseManifest → manifest_cid
7. Publish manifest to Zenoh: roxy/catalog/{artist_hash}/music/{manifest_cid}
```

### Consumer Acquires Access

```
1. Consumer discovers manifest via Zenoh subscription or Yellow search
2. Consumer acquires UCAN token (payment out-of-band)
   - UCAN fields: issuer=artist, audience=consumer, capability=Content,
     resource=manifest_cid, not_before/not_after=time bounds
3. Consumer presents UCAN to artist's node (or delegated caching node)
4. Node verifies UCAN chain, checks revocation set via Zenoh
5. Node unwraps K with artist's private key, re-wraps K with consumer's
   X25519 public key via ECDH → wrapped_key
6. Consumer unwraps with their own X25519 private key → K
7. Consumer fetches encrypted blob (from artist, cache, or nearby peer)
8. Consumer decrypts with K, plays content
```

### Key Distribution Notes

- The artist's node (or a delegated node) must be online for step 5, but only **once per grant**, not per play. The consumer caches K locally.
- Delegation: artist issues a UCAN to a delegate node authorizing it to wrap keys on their behalf. The delegate holds wrapped copies of K, not the raw key.
- K is never transmitted in the clear. Always wrapped per-recipient via ECDH.

---

## Zenoh Key Expressions & Discovery

### Namespace Hierarchy

```
roxy/catalog/{artist_hash}/{content_type}/{manifest_cid}  # published content
roxy/catalog/{artist_hash}/meta                            # artist profile
roxy/license/{consumer_hash}/{manifest_cid}                # active grants
roxy/revocation/{artist_hash}/{ucan_hash}                   # revocation notices
```

### Content Types (extensible)

```
music, video, text, image, software, dataset, bundle
```

### Consumer Subscription Patterns

| Intent | Zenoh pattern |
|---|---|
| All music from one artist | `roxy/catalog/{artist_hash}/music/**` |
| All content from one artist | `roxy/catalog/{artist_hash}/**` |
| All free music | `roxy/catalog/*/music/**` + filter `LicenseType::Free` |
| My active licenses | `roxy/license/{my_hash}/**` |
| Revocations affecting me | `roxy/revocation/**` (filtered client-side) |

### Artist Profile

Published at `roxy/catalog/{artist_hash}/meta` as a CID-addressed blob: display name, bio, avatar CID, links, and a list of published manifest CIDs. This is what a "Roxy artist page" renders from.

### Revocation Flow

Artist publishes to `roxy/revocation/{artist_hash}/{ucan_hash}`. All nodes holding that grant see it immediately via Zenoh. Green wipes the cached key and evicts the content.

---

## Cache Lifecycle State Machine

Green-managed, sans-I/O. Returns actions for the caller to execute.

### States

```
         ┌──────────┐
         │  Empty   │ ── acquire UCAN ──→ ┌──────────┐
         └──────────┘                      │  Active  │
                                           └────┬─────┘
              ┌────────────────────────────────┘
              │ (not_after - expiry_notice) reached
              ▼
         ┌──────────┐
         │ Expiring │ ── auto-renew succeeds ──→ Active
         └────┬─────┘
              │ not_after reached, no renewal
              ▼
         ┌──────────┐
         │  Evict   │ ── wipe K, delete blob ──→ Empty
         └──────────┘
```

### CacheEntry

```rust
CacheEntry {
    manifest_cid:   ContentId,
    content_cid:    ContentId,
    wrapped_key:    Vec<u8>,        // Consumer-specific wrapped K
    ucan_not_after: Option<f64>,    // None = perpetual (OneTime/Free)
    expiry_notice:  u32,            // Seconds before expiry to notify
    state:          CacheState,     // Active, Expiring, Evict
    auto_renew:     bool,           // Consumer preference
}
```

### CacheActions

```rust
CacheAction::NotifyExpiring { manifest_cid, seconds_remaining }
CacheAction::RequestRenewal { manifest_cid }      // if auto_renew
CacheAction::WipeKey { manifest_cid }             // zero the cached K
CacheAction::EvictContent { content_cid }          // delete encrypted blob
CacheAction::RenewalSucceeded { manifest_cid, new_not_after }
CacheAction::RenewalFailed { manifest_cid, reason }
```

The `CacheManager` exposes `tick(now: f64) -> Vec<CacheAction>` — same sans-I/O pattern as the rest of harmony. For `Free` and `OneTime` licenses, `ucan_not_after` is `None` — Green never time-evicts these (W-TinyLFU cache pressure can still deprioritize unpinned free content).

---

## Crate Structure

### New Crate: `harmony-roxy`

```
harmony-roxy/
├── src/
│   ├── lib.rs
│   ├── manifest.rs      # LicenseManifest, LicenseType, UsageRights, Price
│   ├── keywrap.rs       # wrap_key, unwrap_key (ECDH + ChaCha20)
│   ├── cache.rs         # CacheManager, CacheEntry, CacheState, CacheAction
│   ├── catalog.rs       # Zenoh key expression builders, catalog types
│   └── types.rs         # Shared types (artist profile, etc.)
└── Cargo.toml
```

### Dependency Graph

```
harmony-crypto
  └── harmony-identity
      ├── harmony-content    (ContentId, BlobStore, bundles)
      ├── harmony-zenoh      (key expressions, PubSubRouter)
      └── harmony-roxy       (NEW — depends on all three above)
```

### What harmony-roxy Owns

- License manifest types + CBOR serialization
- Key wrapping/unwrapping (thin wrapper over harmony-identity ECDH + harmony-crypto ChaCha20)
- Cache lifecycle state machine
- Zenoh key expression patterns for `roxy/**`
- UCAN resource format for content licenses

### What harmony-roxy Does NOT Own

- Payment (separate crate/service)
- UI (app layer)
- Content storage (uses harmony-content's BlobStore trait)
- Identity management (uses harmony-identity)
- Network transport (caller provides via sans-I/O pattern)

---

## The Self-Sovereign Artist Story

### "I want to share my album"

1. Artist opens Roxy (or any Harmony app with roxy support)
2. Drops 12 audio files into the upload area
3. Roxy chunks, encrypts, and stores each track → 12 content CIDs
4. Roxy bundles them into an album bundle → 1 album CID
5. Artist sets terms: Subscription, 5 cents/month, STREAM+DOWNLOAD, 3-day notice
6. Roxy creates & signs the LicenseManifest, publishes to Zenoh
7. Done. No middleman. No 30% cut. No approval process.

### "I want to listen to this album"

1. Consumer discovers album via Yellow search or Zenoh subscription
2. Sees terms: 12 tracks, 5 cents/month, STREAM+DOWNLOAD
3. Acquires UCAN token (payment out-of-band)
4. Presents UCAN → receives wrapped key → decrypts and plays
5. 3 days before expiry: notification. Auto-renew ON? Seamless. OFF? "Renew now?"
6. Expiry: Green wipes key + blob. Clean. No lingering access.

### "I want to give this away for free"

1. Same flow, but `LicenseType::Free`
2. Content key is published unencrypted. No UCAN needed.
3. Yellow indexes as free content. Green deduplicates across the mesh.

### The Promise

- 100% revenue to the artist
- No approval gate — publish instantly
- No geographic restrictions (unless artist explicitly sets them)
- No algorithm deciding promotion — Yellow surfaces what consumers search for, weighted by creator trust score
- Artist controls their own infrastructure without realizing it — their node is their server, the mesh is their CDN
