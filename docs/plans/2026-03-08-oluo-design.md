# Oluo — Harmony's Semantic Search & Knowledge Index

**Status:** Approved design — future implementation
**Date:** 2026-03-08
**Scope:** Evolve `harmony-semantic` crate + new `harmony-oluo` crate
**Supersedes:** Extends [HSI design](2026-03-07-harmony-semantic-index-design.md) (sidecar format evolution)
**Related:** [Wylene design](2026-03-08-wylene-fluid-interface-design.md) (Oluo contract), [Content addressing](2026-03-02-content-addressing-design.md) (CID infrastructure)

## Vision

Oluo is Harmony's librarian — the Yellow service responsible for indexing,
organizing, and retrieving content across the network. Named for the wise,
sage-like guardian of information, Oluo maintains both personal and global
knowledge indices using matryoshka semantic vector embeddings.

Oluo builds on the HSI (Harmony Semantic Index) foundation, evolving the
compact binary sidecar format into an enriched metadata system and adding
a hierarchical index structure for scalable search.

### Key properties

- **Enriched sidecars** — The 288-byte HSI binary embedding header is
  retained for hardware-speed vector scanning. A CBOR-encoded metadata
  trailer adds timestamps, locations, tags, descriptions, and arbitrary
  extensions. One format for both embeddings and metadata.
- **Layered overlays** — Multiple contributors enrich the same content
  independently. AI face detectors, user tags, and automated classifiers
  each produce overlay sidecars that Oluo merges at query time. No
  coordination, no mutable state.
- **Adaptive-depth embedding trie** — A binary space partition tree over
  embedding bits, with collection blobs as leaves. Dense semantic regions
  split deeper, sparse regions stay shallow. Deterministic: same content
  produces same trie on any node.
- **Tiered search scope** — Personal (local, microseconds) → Community
  (subscribed groups, milliseconds) → Network-wide (Zenoh fanout, seconds).
  Matryoshka tiers naturally match bandwidth budgets at each level.
- **Defense-in-depth privacy** — Jain (Green service) gates what Oluo
  sees on ingest and filters what Wylene shows on retrieval. The sidecar
  self-declares its privacy tier. Oluo independently verifies. Belt and
  suspenders.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                    Wylene (Blue)                         │
│  User intent → SearchIntent → present results           │
└──────────────────────────┬──────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────┐
│                    Jain (Green)                          │
│  Ingest gate (approve/reject) · Retrieval filter        │
│  (context-appropriate results only)                     │
└──────────┬───────────────────────────────┬──────────────┘
           │ ingest                        │ filter
┌──────────▼───────────────────────────────▼──────────────┐
│                    Oluo (Yellow)                         │
│  Embedding index · trie traversal · overlay merge       │
│  · scope escalation · ranking                           │
└──────────────────────────┬──────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────┐
│               harmony-semantic (data layer)              │
│  Enriched sidecar format · binary vector ops            │
│  · collection blobs · CBOR metadata · overlay merge     │
└─────────────────────────────────────────────────────────┘
```

Data flow:

```
INGEST:  new content → Jain (gate) → Oluo (embed + index)
SEARCH:  user → Wylene (intent) → Oluo (search) → Jain (filter) → Wylene (present)
```

## Enriched Sidecar Format

The existing 288-byte HSI sidecar becomes the fixed header of an enriched
sidecar. A CBOR-encoded metadata trailer follows it.

### Layout

```
┌─────────────────────────────────────────────────────────┐
│ FIXED HEADER (288 bytes) — evolved from HSI v1          │
│                                                          │
│  0:4    Magic: 0x48 0x53 0x49 0x02 ("HSI" + version 2)  │
│  4:4    Model fingerprint                                │
│  8:32   Target CID (the content this sidecar describes)  │
│ 40:8    Tier 1 — 64-bit binary vector                    │
│ 48:16   Tier 2 — 128-bit binary vector                   │
│ 64:32   Tier 3 — 256-bit binary vector                   │
│ 96:64   Tier 4 — 512-bit binary vector                   │
│160:128  Tier 5 — 1024-bit binary vector                  │
├─────────────────────────────────────────────────────────┤
│ METADATA TRAILER (variable length, CBOR-encoded)        │
│                                                          │
│  288:4  Trailer length (u32 big-endian, byte count)      │
│  292:N  CBOR map                                         │
│         ├─ privacy_tier: u8 (0=pub-durable, 1=pub-ephem, │
│         │                    2=enc-durable, 3=enc-ephem)  │
│         ├─ created_at: u64 (Unix millis)                 │
│         ├─ content_type: text (MIME type)                │
│         ├─ language: text (BCP-47 tag)                   │
│         ├─ geo: [f64, f64] (lat, lon) — optional         │
│         ├─ description: text — optional                  │
│         ├─ tags: [text] — optional                       │
│         ├─ refs: [bytes(32)] — CIDs of related content   │
│         ├─ source_device: text — optional                │
│         └─ ext: map<text, any> — freeform extension      │
└─────────────────────────────────────────────────────────┘
```

**Key properties:**

- Version bumped to `0x02` to distinguish from pure-embedding v1 sidecars.
- Fixed header at known offsets — binary vector scan reads bytes 40..288
  directly, no parsing needed.
- Trailer length at byte 288 — if absent (file is exactly 288 bytes),
  it's a v1 sidecar with no metadata (backward compatible).
- CBOR chosen over MessagePack for its IETF standardization (RFC 8949)
  and native support for tagged types (timestamps, CIDs).
- Well-known fields have short CBOR keys for compactness; `ext` map
  allows arbitrary additions without schema changes.

### CID type allocation

Enriched sidecars use `CidType::ReservedA` (same as HSI v1 sidecars).
The magic bytes and version field disambiguate v1 vs v2 format. Sidecar
CIDs are distinguishable from regular content blobs at the CID level.

## Privacy Tiers & Indexing Behavior

Four privacy tiers, each with distinct indexing semantics:

```
Tier              Value   Jain gate              Oluo indexing
────────────────  ─────   ─────────────────────  ──────────────────────
Public-Durable    0       Always passes          Full: all tiers, all
                                                 metadata, collection
                                                 placement

Public-Ephemeral  1       Passes with TTL hint   Lightweight: Tier 1
                                                 only, minimal metadata,
                                                 no collection placement,
                                                 auto-evict on TTL

Encrypted-Durable 2       Owner permission or    Full (on decrypted
                          delegation required    content), index entries
                                                 encrypted with owner keys

Encrypted-Ephem.  3       BLOCKED — never        N/A — Oluo never sees it
                          reaches Oluo
```

**Defense in depth:** Even if a bug in Jain allows an encrypted-ephemeral
sidecar through, Oluo checks the `privacy_tier` field and refuses to
index tier 3.

**Encrypted-durable indexing:** The sidecar itself is encrypted
(ChaCha20-Poly1305 via Nakaiah). Only the owner's node (or authorized
delegates) can decrypt and index it. Index entries pointing to
encrypted-durable content are also encrypted — Oluo's index doesn't
leak what's in your private durable store.

## Hierarchical Index — Adaptive-Depth Embedding Trie

The core index structure is a trie over binary embedding bits, with
collection blobs as leaves.

### Structure

```
                        Root (all content)
                       /                  \
                 bit 0 = 0            bit 0 = 1
                /        \           /         \
          bit 1 = 0   bit 1 = 1  bit 1 = 0   bit 1 = 1
              │           │         │            │
           [leaf]      [leaf]    [leaf]        /    \
           (42)        (198)     (7)       bit 2=0  bit 2=1
                                              │        │
                                           [leaf]   [leaf]
                                           (201)    (187)
```

**Splitting rule:** When a leaf collection exceeds 256 entries, split on
the next embedding bit. The split bit position is stored in the internal
node. Dense semantic regions (many photos, lots of code) split deeper.
Sparse regions stay shallow.

### Blob formats

```
Internal Node (106 bytes):
  0:4    Magic: 0x48 0x53 0x4E 0x01 ("HSN" + version 1)
  4:4    Model fingerprint
  8:2    Split bit position (u16)
 10:32   Child-0 CID (left subtree / leaf collection)
 42:32   Child-1 CID (right subtree / leaf collection)
 74:32   Centroid: 256-bit binary vector of this subtree

Leaf = Collection Blob (existing HSC format, up to 256 entries):
  0:4    Magic: 0x48 0x53 0x43 0x01 ("HSC" + version 1)
  4:4    Model fingerprint
  8:4    Entry count (u32, max 256)
 12:32   Centroid: 256-bit binary vector of the cluster center
 44:N×64 Entries: [32-byte CID + 32-byte Tier 3 vector] × N
         (max ~16,428 bytes for 256 entries)
```

Both internal nodes and leaf collections are content-addressed blobs
stored as `CidType::ReservedA`.

### Search traversal

1. Start at root. Check query's bit at the split position.
2. Descend into the matching child (the "near" side).
3. Also check the "far" child if its centroid's Hamming distance is
   within threshold (backtracking for better recall).
4. At leaf level, scan the collection blob with XOR+POPCNT at the
   appropriate tier.
5. Merge results across visited leaves, rank by Hamming distance.

This is a binary space partition tree where partition planes are
individual embedding dimensions — natural for binary-quantized vectors.

### Determinism

Two nodes indexing the same set of sidecars produce the same trie.
Split decisions are deterministic (based on entry count thresholds and
embedding bit values). Trie nodes are content-addressable and
replicable. A community can share its index root CID, and any member
can fetch and verify the entire trie.

### Progressive resolution during traversal

```
Network fanout (Tier 1, 64-bit):
  → Coarsest traversal, minimal bandwidth
  → "Are there any photos in this region of semantic space?"

Community search (Tier 3, 256-bit):
  → Medium precision, scans community collection blobs
  → "Find relevant discussions in our team's index"

Personal search (Tier 3–5, 256–1024-bit):
  → Full precision, local collection blobs
  → "Show me that article about mesh networking"
```

## Overlay Merge Semantics

Multiple sidecars can target the same content CID. Each is an
independent content-addressed blob. At query time, Oluo merges them.

### Merge rules

```
EMBEDDING TIERS (fixed header):
  → Use the sidecar matching the blessed model fingerprint
  → If multiple match, use the one with the latest created_at
  → Foreign-model sidecars: stored, flagged, not used for
    vector search (metadata still merged)

WELL-KNOWN METADATA FIELDS:
  → privacy_tier:   Most restrictive wins (max value)
  → created_at:     Earliest timestamp preserved (provenance)
  → content_type:   First non-null wins
  → language:       Union — content can be multilingual
  → geo:            Latest wins (most recent location knowledge)
  → description:    Concatenate with separator, deduplicate
  → tags:           Union (set merge, no duplicates)
  → refs:           Union (set merge)
  → source_device:  First non-null wins

EXTENSION MAP (ext):
  → Union merge — all keys from all overlays
  → Key conflicts: latest created_at wins
```

### Example

```
Original sidecar (by content creator):
  embedding: [0x3A, 0x7F, ...]
  content_type: "image/jpeg"
  created_at: 1741234567000
  tags: ["vacation", "beach"]

Overlay 1 (by AI face detector, Yellow service):
  tags: ["jake", "husband"]
  ext: { "faces": [<cid_jake>, <cid_husband>], "face_count": 2 }

Overlay 2 (by user, manual tag):
  description: "Sunset at Kailua Beach"
  tags: ["hawaii", "sunset"]
  geo: [21.3969, -157.7261]

Merged result at query time:
  embedding: [0x3A, 0x7F, ...]
  content_type: "image/jpeg"
  created_at: 1741234567000
  tags: ["vacation", "beach", "jake", "husband", "hawaii", "sunset"]
  description: "Sunset at Kailua Beach"
  geo: [21.3969, -157.7261]
  ext: { "faces": [...], "face_count": 2 }
```

### Overlay discovery

Overlays reference the same target CID in their fixed header (bytes
8..40). Oluo discovers overlays via a reverse index:
`target_cid → [sidecar_cid_1, sidecar_cid_2, ...]`. This mirrors
the Flatpack reverse lookup registry from the content-addressing design.

## Jain↔Oluo Contract

Two trait boundaries define the contract. These are the semantic
interfaces — runtime transport is Zenoh pub/sub.

### Ingest Gate

```rust
/// Jain's decision on whether content should be indexed.
pub enum IngestDecision {
    /// Index fully — all tiers, all metadata, collection placement.
    IndexFull,
    /// Index lightweight — Tier 1 only, minimal metadata, auto-evict after TTL.
    IndexLightweight { ttl_secs: u64 },
    /// Reject — do not index. Oluo must not retain any trace.
    Reject,
}

/// Jain calls this contract to submit content for indexing.
/// Oluo never solicits content — Jain pushes.
pub trait IngestGate {
    /// Submit a sidecar for indexing. Jain has already made the
    /// gate decision and attaches it.
    fn submit(&mut self, sidecar: EnrichedSidecar, decision: IngestDecision)
        -> Vec<OluoAction>;
}
```

**Zenoh transport:** Jain publishes to `oluo/ingest/{content_type}/**`
with the `IngestDecision` encoded in the payload header. Oluo subscribes.

### Retrieval Filter

```rust
/// A search result before Jain filtering.
pub struct RawSearchResult {
    pub target_cid: ContentId,
    pub score: f32,
    pub metadata: SidecarMetadata,
    pub overlays: Vec<ContentId>,
}

/// A search result after Jain filtering.
pub struct FilteredSearchResult {
    pub target_cid: ContentId,
    pub score: f32,
    pub metadata: SidecarMetadata,
}

/// Context Jain uses to make filtering decisions.
pub struct RetrievalContext {
    pub requester: IdentityId,
    pub social_context: Option<Vec<IdentityId>>,
    pub device_context: DeviceContext,
}

/// Jain filters raw results for context-appropriateness.
pub trait RetrievalFilter {
    /// Filter and optionally re-rank results.
    /// May remove results entirely or demote scores.
    fn filter(
        &self,
        results: Vec<RawSearchResult>,
        context: RetrievalContext,
    ) -> Vec<FilteredSearchResult>;
}
```

**Zenoh transport:** Oluo publishes raw results to
`oluo/results/{query_id}`. Jain subscribes, filters, and republishes
to `wylene/results/{query_id}`. Wylene never sees unfiltered results.

### Zenoh key expressions

```
INGEST:
  jain/approved/{content_type}/**   Jain → Oluo approved content
  oluo/ingest/{content_type}/**     Oluo ingest subscription

SEARCH:
  wylene/search/{user_hash}         Wylene → Oluo search request
  oluo/query/{user_hash}            Oluo query subscription
  oluo/results/{query_id}           Oluo → Jain raw results
  jain/filter/{query_id}            Jain filtering
  jain/filtered/{query_id}          Jain → Wylene filtered results
  wylene/results/{query_id}         Wylene result subscription

INDEX MANAGEMENT:
  oluo/index/stats/{node_id}        Index health metrics
  oluo/index/sync/{community_id}    Community index replication
```

## Tiered Search Scope

Three search levels, each widening the radius.

### Personal (local node)

```
Scope:    Your node's local index trie
Tier:     3–5 (256–1024 bit) — full precision
Latency:  Microseconds (in-memory trie traversal + XOR+POPCNT)
Results:  Everything Jain has approved for your index
```

Always searched first.

### Community (subscribed groups)

```
Scope:    Index trie root CIDs shared by opted-in communities
Tier:     3 (256-bit) — good precision, moderate bandwidth
Latency:  Milliseconds (fetch remote collection blobs by CID)
Transport: Subscribe to oluo/index/sync/{community_id}
           Members publish trie root CID updates
           Your node fetches and caches trie structure (not content)
```

Searched by default alongside personal.

### Network-wide (live fanout)

```
Scope:    Any reachable Oluo instance on the Zenoh network
Tier:     1 (64-bit) — coarsest, minimal bandwidth (8 bytes per query)
Latency:  Seconds (Zenoh query fanout, wait for responses)
Transport: Zenoh queryable at oluo/query/network/**
           Responding nodes return top-K at Tier 1
           Requester re-ranks with higher tiers if sidecars fetchable
```

Only on explicit request.

### Scope escalation example

```
User: "show me that mesh networking article"
  → Wylene sends SearchIntent to Oluo
  → Oluo searches Personal (Tier 5, microseconds)
  → Found 3 strong matches → return immediately

User: "find papers on delay-tolerant networking"
  → Personal search: 0 matches
  → Community search: 2 weak matches
  → Wylene: "I found a couple things in our team's index,
     but want me to search the wider network?"
  → User: "yes"
  → Network fanout (Tier 1): 12 matches from 4 nodes
  → Fetch Tier 3 sidecars for re-ranking → return top 5
```

## Crate Structure

### `harmony-semantic` (data formats & vector ops)

Evolves the existing HSI design. `no_std + alloc` compatible.

```
harmony-semantic/
  src/
    lib.rs            — public API, feature gates
    sidecar.rs        — EnrichedSidecar encode/decode (header + CBOR trailer)
    metadata.rs       — SidecarMetadata, well-known fields, PrivacyTier enum
    overlay.rs        — overlay merge logic (combine sidecars for same target)
    collection.rs     — collection blob encode/decode (HSC format)
    quantize.rs       — float32 → binary quantization + tier packing
    distance.rs       — Hamming distance (XOR + POPCNT, SIMD-aware)
    fingerprint.rs    — model fingerprint computation
    tier.rs           — EmbeddingTier enum (T1..T5), tier selection
    error.rs          — SemanticError type
```

**Dependencies:** `harmony-crypto` (SHA-256, BLAKE3), `harmony-content`
(`ContentId`, `CidType::ReservedA`), `ciborium` (std-gated, CBOR).
No ML framework, no runtime, no I/O.

### `harmony-oluo` (search engine)

The librarian. `no_std + alloc` for core logic (sans-I/O state machine).

```
harmony-oluo/
  src/
    lib.rs            — public API
    engine.rs         — OluoEngine sans-I/O state machine (Event → Action)
    trie.rs           — adaptive-depth embedding trie
    search.rs         — query execution (traversal, backtracking, merging)
    scope.rs          — SearchScope enum, escalation logic
    ingest.rs         — IngestGate trait, IngestDecision enum
    filter.rs         — RetrievalFilter trait, RetrievalContext
    ranking.rs        — score computation, overlay-aware result merging
    zenoh_keys.rs     — Zenoh key expression constants (oluo/**)
    error.rs          — OluoError type
```

**Dependencies:** `harmony-semantic`, `harmony-crypto`, `harmony-content`.
No direct dependency on Jain/Wylene crates — contract is via traits.

### Dependency graph

```
harmony-crypto
  └── harmony-content
        └── harmony-semantic    (sidecar format, vector ops)
              └── harmony-oluo  (search engine, trie index, Jain contract)
```

### Sans-I/O pattern

```rust
pub enum OluoEvent {
    Ingest { sidecar: EnrichedSidecar, decision: IngestDecision },
    Search { query_id: u64, query: SearchQuery },
    SyncReceived { community_id: [u8; 32], trie_root: ContentId },
    EvictExpired { now: u64 },
}

pub enum OluoAction {
    IndexUpdated { trie_root: ContentId },
    SearchResults { query_id: u64, results: Vec<RawSearchResult> },
    FetchTrieNode { cid: ContentId },
    FetchSidecar { cid: ContentId },
    PublishTrieRoot { community_id: [u8; 32], root: ContentId },
    PersistBlob { cid: ContentId, data: Vec<u8> },
}
```

## YAGNI Boundaries

| Out of scope | Why | When it might come |
|---|---|---|
| Embedding model integration | Runs externally, crate stays `no_std` | Never in-crate |
| Jain implementation | We define contract traits, not filtering logic | Separate design |
| Wylene↔Oluo query protocol | Already partially spec'd in Wylene design | After Oluo ships |
| Distributed consensus | Deterministic tries converge naturally | Likely never |
| Float32 re-ranking | Binary-only sufficient for launch | When precision matters |
| Cross-model translation | Overlays handle this more simply | If convergence insufficient |
| Proactive suggestions | Requires user model integration | After user model design |
| Network-wide search protocol | Personal + community come first | After community sync |
| Encrypted index entries | Requires Nakaiah integration | After Nakaiah design |

## Implementation Priority

```
Phase 1 — harmony-semantic:
  Enriched sidecar format, CBOR metadata, binary vector ops,
  collection blobs, overlay merge, privacy tier enum

Phase 2 — harmony-oluo:
  Adaptive-depth trie, personal search (local only),
  IngestGate/RetrievalFilter traits, sans-I/O engine

Phase 3 (future):
  Community index sync, network fanout, Jain integration,
  encrypted-durable indexing
```

## Design Principles

- **Content-addressed everything.** Sidecars, overlays, collection
  blobs, trie nodes — all have CIDs. No special storage paths.
  Replicate via existing content infrastructure.
- **Progressive resolution.** Every operation works at any tier.
  Constrained devices use Tier 1. Search servers use Tier 5.
  Same code path, different precision.
- **Deterministic indexing.** Same content produces same trie on
  any node. Enables verifiable community indices.
- **Language as a bridge.** The Qwen3 backbone's Chinese-English
  capability is intentional. Semantic search should work across
  language barriers.
- **Trust through redundancy.** Jain gates. Sidecar self-declares.
  Oluo verifies. Three independent checks for privacy compliance.
- **Sans-I/O.** Core logic is a pure state machine. The caller
  provides storage, networking, and time. Testable, portable,
  runtime-agnostic.
