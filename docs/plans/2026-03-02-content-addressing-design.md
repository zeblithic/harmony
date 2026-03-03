# Content-Addressing Architecture Design

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Design Harmony's content-addressing layer — a 32-byte CID format, blob/bundle Merkle DAG, content-defined chunking, W-TinyLFU caching, sidecar storage architecture, and reverse-lookup registry.

**Architecture:** A new `harmony-content` crate providing pure data structures and algorithms (sans-I/O) for content-addressed storage. Two data primitives — blobs (<=1MB leaf data) and bundles (arrays of CIDs) — compose into recursive Merkle DAGs rooted in a single 32-byte Content Identifier. Transport-agnostic: bridges to Reticulum resource transfer and Zenoh pub/sub/query are thin adapter layers.

**Tech Stack:** Rust, SHA-256 (from `harmony-crypto`), Gear hash (FastCDC), Count-Min Sketch (W-TinyLFU)

---

## 1. The 32-Byte Content Identifier (CID)

Every piece of content in Harmony is identified by a 32-byte (256-bit) Content Identifier. The CID is self-describing: it encodes the cryptographic hash, the payload size, the structural type (blob vs bundle vs special), and an integrity checksum — all in a single cache-line-aligned struct.

### Byte Layout

| Offset | Size | Field |
|---|---|---|
| `0x00–0x1B` | 28 bytes (224 bits) | **Content Hash** — first 224 bits of SHA-256 digest |
| `0x1C–0x1F` | 4 bytes (32 bits) | **Size + Type Tag** — big-endian u32 |

**Last 4 bytes (big-endian u32):**

- Bits `[31:12]` (20 bits): Payload size in bytes (max 1,048,575 = ~1MB)
- Bits `[11:0]` (12 bits): Type tag + checksum (unary-encoded)

### Hash Truncation Security

Truncating SHA-256 to 224 bits yields a collision resistance bound of 2^112 operations. At one trillion files ingested per second, it would take billions of years to reach even a fractional collision probability. NIST formally standardized SHA-224 at this same security level. The first 28 bytes of a SHA-256 digest retain uniform distribution and avalanche properties.

### The Unary Type Tag

The 12-bit type/checksum field uses unary prefix coding. The number of leading 1-bits before the first 0-bit encodes the type. Remaining bits are a fast integrity checksum (e.g., CRC) over the preceding 31 bytes + type bits.

| Prefix | Type | Checksum Bits | Strength |
|---|---|---|---|
| `0xxx-xxxx-xxxx` | **Blob** (leaf data) | 11 bits | 1/2048 |
| `10xx-xxxx-xxxx` | **Bundle L1** | 10 bits | 1/1024 |
| `110x-xxxx-xxxx` | **Bundle L2** | 9 bits | 1/512 |
| `1110-xxxx-xxxx` | **Bundle L3** | 8 bits | 1/256 |
| `1111-0xxx-xxxx` | **Bundle L4** | 7 bits | 1/128 |
| `1111-10xx-xxxx` | **Bundle L5** | 6 bits | 1/64 |
| `1111-110x-xxxx` | **Bundle L6** | 5 bits | 1/32 |
| `1111-1110-xxxx` | **Bundle L7** | 4 bits | 1/16 |
| `1111-1111-0xxx` | **Inline Metadata** | 3 bits | 1/8 |
| `1111-1111-10xx` | **Reserved A** | 2 bits | 1/4 |
| `1111-1111-110x` | **Reserved B** | 1 bit | 1/2 |
| `1111-1111-1110` | **Reserved C** | 0 bits | — |
| `1111-1111-1111` | **Reserved D** | 0 bits | — |

### Hardware Parsing

Extracting the type uses a single CPU `LZCNT` (count leading zeros) instruction on the inverted 12-bit field — no branching. The 32-byte struct aligns perfectly with AVX2 256-bit registers for SIMD-accelerated scanning and comparison. Two CIDs fit exactly in one 64-byte CPU cache line.

### Address Space Math

A 1MB bundle holds `1,048,575 / 32 = 32,767` CIDs.

| Level | Max Addressable Data | Example Scale |
|---|---|---|
| Blob (L0) | 1 MB | A single chunk |
| Bundle L1 | ~32 GB | A Blu-ray disc |
| Bundle L2 | ~1 PB | A large data center |
| Bundle L3 | ~32 EB | More than all data on Earth |
| Bundle L4+ | Theoretical | Composition headroom |

### Inline Metadata CID (`1111-1111-0xxx`)

When the type tag is `1111-1111-0`, the 28-byte "hash" field is repurposed as inline metadata. This appears as the first entry in a root bundle:

| Offset | Size | Field |
|---|---|---|
| `0x00–0x07` | 8 bytes | Total uncompressed file size (u64) |
| `0x08–0x0B` | 4 bytes | Total chunk count (u32) |
| `0x0C–0x13` | 8 bytes | Creation timestamp (u64, Unix epoch ms) |
| `0x14–0x1B` | 8 bytes | MIME type or extension (packed ASCII or hash) |

This solves the "zip bomb" problem — a client reads the first 32 bytes of a root bundle and knows the total file size before committing to download the full tree.

## 2. Data Model — Blobs, Bundles, and the Recursive Merkle DAG

### Two Primitives

**Blob** — A leaf node. An opaque byte array up to 1,048,575 bytes. Its CID's content hash is `SHA-256(data)[:28]`, its type tag starts with `0`, and its payload size equals the byte length of the data.

**Bundle** — An interior node. A byte array containing a flat, densely-packed array of 32-byte CIDs. No headers, no framing, no length prefixes. The bundle's own CID is `SHA-256(bundle_bytes)[:28]`, its type tag encodes its depth level, and its payload size equals the byte length of the CID array.

### Depth Invariant

A bundle's depth level is encoded in its type tag. The **only** structural rule is:

> **Every child CID in a bundle must have a strictly lower depth than the bundle itself.**

A blob has depth 0. A bundle's depth must be at least `max(child_depths) + 1`, but may be higher to leave room for future composition. Sparse depth is explicitly valid — an L4 bundle containing only L2 children is allowed. You trade checksum bits for composition flexibility, and 7 levels is plenty of room.

| Valid | Example |
|---|---|
| Yes | L4 bundle containing [L2, L2, L1, blob] |
| Yes | L3 bundle containing [blob, blob, blob] (sparse) |
| Yes | L2 bundle containing [L1, L1, blob] (mixed) |
| No | L2 bundle containing [L2, ...] (child depth not strictly less) |
| No | L1 bundle containing [L3, ...] (child depth exceeds parent) |

On ingest, the builder sets the depth to `max(child_depths) + 1` by default. On receive, the validator rejects any child with `depth >= parent_depth`. If a tree grows deeper than ~4 through repeated composition, repacking for efficiency is an optional offline optimization, never a requirement for correctness.

### Constructing a Merkle DAG (Ingest)

```
1. Split into <=1MB chunks (content-defined chunking, see Section 3)
2. Hash each chunk → Blob CID
3. If only 1 chunk → done, the blob CID is the root
4. Concatenate blob CIDs into groups of <=32,767 → L1 Bundles
5. If only 1 L1 bundle → done, the L1 CID is the root
6. Concatenate L1 CIDs into groups of <=32,767 → L2 Bundles
7. Repeat until a single root CID remains
```

For a root bundle, the first CID in the array is an Inline Metadata CID carrying total file size, chunk count, timestamp, and MIME type.

### Examples

**100MB file:**

```
100MB → 100 blobs (~1MB each)
      → 1 L1 bundle: [inline_metadata][blob_0]...[blob_99] = 3,232 bytes
      → Root = the L1 CID
```

**1TB file:**

```
1TB → ~1,000,000 blobs
    → ~31 L1 bundles (each ~32,767 blob CIDs)
    → 1 L2 bundle: [inline_metadata][L1_0]...[L1_30] = 1,024 bytes
    → Root = the L2 CID
```

### Structural Sharing

Identical chunks across different files produce identical CIDs. Two versions of a document that differ by one paragraph share all unchanged chunk CIDs. The network only transfers and stores the changed blobs plus a new root bundle.

### Bundle Uniformity

A bundle is just an array of 32-byte structs. No mixed formats, no variable-length records. This means:

- SIMD-friendly scanning (AVX2 loads one CID per instruction)
- Trivial serialization (the bundle *is* its wire format)
- Zero-copy parsing: `&[u8]` → `&[ContentId]` is a reinterpret
- Inline metadata CIDs participate uniformly, distinguished solely by type tag bits

## 3. Content-Defined Chunking

### Why CDC, Not Fixed-Size

Fixed-size chunking at 1MB boundaries means inserting a single byte at the beginning shifts every chunk boundary — zero deduplication. Content-defined chunking uses a rolling hash to find "natural" split points determined by the data itself. Boundaries are content-derived, so insertions only affect the chunk containing the edit.

### Algorithm: Gear Hash (FastCDC)

| Algorithm | Speed | Notes |
|---|---|---|
| Rabin fingerprint | ~400 MB/s | Good boundaries, expensive modular arithmetic |
| Gear hash (FastCDC) | ~2+ GB/s | Comparable quality, simpler, SIMD-friendly |

Gear hash: one multiply-and-XOR per byte with a lookup table. FastCDC demonstrates 2–10x speedup over Rabin with equivalent deduplication ratios.

### Parameters

| Parameter | Value | Rationale |
|---|---|---|
| Minimum chunk | 256 KB | Prevents pathological micro-chunking |
| Average chunk | 512 KB | Good deduplication granularity |
| Maximum chunk | 1 MB (1,048,575 bytes) | CID size field limit |
| Window size | 48 bytes | Standard for FastCDC |
| Bit mask | 19 bits (`0x7FFFF`) | Targets ~512KB average (`2^19 = 524,288`) |

### Chunking Pipeline

```
Raw data stream
  → Gear hash rolling window scans byte-by-byte
  → When hash & MASK == 0, cut here (natural boundary)
  → If 1MB reached without a cut, force-cut (hard maximum)
  → If cut found before 256KB, skip it (hard minimum)
  → Each chunk → SHA-256 → Blob CID
  → Collect CIDs → build bundle tree (Section 2)
```

### Deduplication Example

50MB document, 3KB edit in the middle:

```
Version 1: [chunk_0][chunk_1][chunk_2][...][chunk_98]
Version 2: [chunk_0][chunk_1][chunk_2'][...][chunk_98]
                              ^^^^^^^^ only this changed
```

98 of 99 chunks share CIDs. Network transfers only `chunk_2'` (<=1MB) plus a new root bundle (a few KB).

### Relationship to Transport

Content-defined chunking happens before encryption. Chunk on plaintext for good deduplication, then encrypt each chunk independently at the transport layer via `LinkCrypto`.

## 4. Wire Format and Transport Integration

### Transport-Agnostic Content Layer

The content layer produces CIDs and blobs. Three transport mechanisms carry content, selected based on path availability:

| Transport | When Used | Bandwidth |
|---|---|---|
| **Reticulum resource transfer** | Last resort, constrained links | Low (500B MTU) |
| **Zenoh pub/sub + query/reply** | Normal operation, LAN/WAN | High (line-speed) |
| **Direct QUIC (Iroh-style)** | Bulk swarming, multi-peer parallel | Very high |

### Content Retrieval Flow

```
1. Client has root CID
2. Query: get("harmony/content/{hex(root_cid)}")
3. Nearest node with the blob replies
4. If root is a bundle:
   a. Parse inline metadata CID → total size, chunk count
   b. For each child CID: check local store, fetch if missing
   c. Recurse for child bundles
5. All leaf blobs collected → reassemble original data
```

### Zenoh Key Expression Mapping

```
harmony/content/{cid_hex}          — blob/bundle data
harmony/content/{cid_hex}/meta     — optional out-of-band metadata
harmony/announce/{cid_hex}         — "I have this content" presence
```

Nodes register as Queryables on `harmony/content/**`. Zenoh's interest-based routing finds the nearest node. Multiple nodes registering for the same CID get natural load-balancing.

### Reticulum Resource Transfer Bridge

Each blob (<=1MB) maps 1:1 to a single resource transfer:

| Content Layer | Transport Layer |
|---|---|
| Blob (<=1MB) | Single ResourceSender/ResourceReceiver session |
| Bundle (<=1MB of CIDs) | Single ResourceSender/ResourceReceiver session |
| Full file (any size) | Orchestrator walks DAG, transfers each blob independently |

The CID content hash verifies plaintext integrity. The resource transfer's `resource_hash` verifies encrypted wire payload integrity. These are independent layers.

### Delta-Encoding for Bundle Updates

When a file changes and a new root bundle is generated, most child CIDs are identical. Delta-encoding avoids redundant transfer.

**Opcode format:**

| Opcode | Meaning |
|---|---|
| `0x00 [offset:u32] [length:u32]` | COPY length bytes from old bundle at offset |
| `0x01 [length:u16] [data...]` | INSERT length bytes of new data |

Delta is used only when the receiver already has the previous bundle version (signaled by including the old CID in the request). If the delta would be larger than the full bundle, send the full bundle instead. This is an optimization, never a requirement.

### Content Announcements

```
Zenoh put("harmony/announce/{root_cid_hex}", metadata)
```

Subscribers on `harmony/announce/**` receive notifications. Interested peers pull content on demand. For Reticulum-only paths, the root CID is embedded in a Reticulum announce packet's `app_data` field.

## 5. Caching Architecture — W-TinyLFU

### Why Not LRU or LFU

**LRU fails:** A sequential scan (backup, indexing) touches thousands of blobs, evicts all hot content, then those scanned blobs are never accessed again. One scan poisons the entire cache.

**LFU fails:** Tracking exact frequency for millions of CIDs consumes enormous memory. Items that were popular last month accumulate unassailable frequency scores and squat in the cache forever.

### W-TinyLFU: The Hybrid

```
┌─────────────────────────────────────────────────────────┐
│                    TinyLFU Frequency Sketch              │
│          (Count-Min Sketch, ~4 bits per counter)         │
│         Approximates access frequency for ALL CIDs       │
│         Periodic halving prevents "stuck items"          │
└────────────────────────┬────────────────────────────────┘
                         │ frequency query
          ┌──────────────┴──────────────┐
          │                             │
   ┌──────▼──────┐            ┌─────────▼─────────┐
   │ Window Cache │            │    Main Cache      │
   │   (1% cap)   │──admit?──▶│    (99% cap)       │
   │  Simple LRU   │           │  Segmented LRU     │
   └──────────────┘            └───────────────────┘
```

**Flow:**

1. Every access increments the CID's counter in the Count-Min Sketch
2. New content enters the Window Cache (1% capacity, plain LRU)
3. When Window is full, eviction candidate faces an admission challenge against the least-valuable Main Cache item:
   - Candidate frequency > Main victim frequency → promote to Main, evict victim
   - Otherwise → drop candidate, Main untouched
4. Periodic aging: all sketch counters right-shifted by 1 (halved), decaying historical popularity

**Sketch sizing (100,000 cached blobs, up to 100GB):**

| Parameter | Value |
|---|---|
| Sketch counters | ~200,000 (2x overprovisioning) |
| Counter width | 4 bits |
| Hash functions | 4 |
| Total sketch memory | ~400 KB |
| Halving threshold | Every 100,000 accesses |

400KB of metadata to manage 100GB of cached content.

### Storage Tiers

W-TinyLFU operates independently per tier:

| Tier | Medium | Typical Size |
|---|---|---|
| Hot | RAM | 1–16 GB |
| Warm | NVMe/SSD | 100 GB – 1 TB |
| Cold | HDD / Network | 1 TB+ |

### Pinning

Users pin content to exempt it from eviction. A pinned CID (and all reachable children) is marked permanent. Pinning shifts storage burden to the pinning node — the "sovereign data" principle.

## 6. Sidecar Architecture — Decoupling Forwarding from Storage

### The Problem with On-Path Caching

Every-router caching (traditional ICN) degrades router performance with disk I/O, wastes storage with redundant copies along paths, and mixes packet-switching concerns with blob-management concerns.

### The Sidecar Model

```
┌─────────────────────────┐     ┌─────────────────────────┐
│       Router             │     │     Storage Sidecar      │
│                          │     │                          │
│  • Forwarding (FIB)      │     │  • W-TinyLFU cache       │
│  • Interest tracking     │◄───►│  • Blob store (RAM/SSD)  │
│  • Zenoh routing engine  │link │  • CID index             │
│  • Line-speed operation  │     │  • Queryable registrar   │
│                          │     │                          │
│  Does NOT store blobs    │     │  Does NOT route packets  │
└─────────────────────────┘     └─────────────────────────┘
```

Connected by fast local link (localhost, Unix socket, shared memory). Separate logical roles, may share physical hardware.

### Ingest and Retrieval

**Ingest:** Router mirrors transiting blobs to sidecar (async, non-blocking). Sidecar applies W-TinyLFU admission. If admitted, stores blob and registers Zenoh Queryable.

**Retrieval:** Remote peer queries `harmony/content/{cid}`. Router's FIB knows the local sidecar has a matching Queryable. Routes query to sidecar. Sidecar replies with blob. Router forwards reply.

### Queryable Registration — Prefix Sharding

Sidecars register a small number of prefix-based Queryables, not one per CID:

```
harmony/content/0**  through  harmony/content/f**
```

16 registrations cover the entire CID space. On query, sidecar checks local index. If it has the blob, replies. If not, returns nothing and the router falls through upstream.

### Bloom Filter Announcements

Sidecars periodically broadcast Bloom filters of their cached CID set (~120 KB for 100,000 CIDs at 0.1% false positive). Nearby routers check the filter before routing queries to sidecars, skipping those that definitely don't have the content.

### Topology Flexibility

| Deployment | Router | Sidecar |
|---|---|---|
| Home node | Harmony process on a Raspberry Pi | Same process, in-memory (512MB) |
| Community server | Dedicated routing process | Separate process, NVMe (1TB) |
| Municipal mesh | Hardware appliance | Rack-mounted storage on 10GbE |
| Constrained device | LoRa gateway (Reticulum) | None (pure relay, no caching) |

## 7. Reverse Lookup Registry (Flatpack)

### The Problem

Content-addressing is forward-looking: root CID → walk tree → find blobs. The reverse — "which bundles contain this blob?" — has no answer in the data structure. Blobs have no upward pointers.

### Use Cases

- Deduplication discovery
- Impact analysis (blob unavailable → what roots affected?)
- Garbage collection (is this blob referenced by any pinned root?)
- Version tracking

### Architecture

Flatpack is a distributed inverted index: `blob_cid → [bundle_cid_a, bundle_cid_b, ...]`. It's an auxiliary service built on top of the content and routing layers.

**Index construction (per sidecar):**

```
When a bundle is cached:
  1. Parse the bundle's CID array
  2. For each child CID: insert child_cid → bundle_cid into local index
  3. Maintain Queryable on flatpack prefix
```

**Query:** `get("harmony/flatpack/{child_cid_hex}")` → returns known bundle CIDs containing that child.

**Prefix sharding:** `harmony/flatpack/0**` through `harmony/flatpack/f**`.

### Cuckoo Filter Broadcasts

Sidecars broadcast Cuckoo filters (~150 KB for 100,000 CIDs at 0.1% false positive) of their flatpack index. Cuckoo filters support deletion (unlike Bloom), which is needed as bundles are evicted from cache.

### Consistency Model

Flatpack is **eventually consistent** and **best-effort**:

- Sidecars only index bundles they've cached
- Different sidecars may return different results
- No results doesn't mean unreferenced — just that no nearby sidecar has seen the referencing bundle
- The authoritative answer always comes from walking the DAG forward

### Garbage Collection Integration

Before evicting a blob, optionally check local flatpack: is the blob referenced by any pinned bundle? If yes, skip eviction.

## 8. Implementation Roadmap

### New Crate: `harmony-content`

```
harmony-crypto
  └── harmony-identity
  │     └── harmony-reticulum
  └── harmony-content   ← NEW (depends only on harmony-crypto)
harmony-zenoh
```

Pure data structures and algorithms. No I/O, no async. Sans-I/O.

### Modules

| Module | Contents |
|---|---|
| `cid.rs` | `ContentId` struct, type tag parsing, hash computation, inline metadata |
| `blob.rs` | Blob validation, `BlobStore` trait |
| `bundle.rs` | Bundle construction, depth validation, child iteration, inline metadata |
| `chunker.rs` | Gear hash, FastCDC chunking with min/max/avg parameters |
| `dag.rs` | Merkle DAG builder (file → root CID), walker (root → ordered blobs) |
| `cache.rs` | W-TinyLFU: Count-Min Sketch, Window LRU, admission challenge |
| `delta.rs` | Bundle delta-encoding: COPY/INSERT opcodes |
| `store.rs` | `ContentStore` trait, in-memory reference implementation |

### Phased Build Order

| Phase | Scope | Priority | Depends On | Tests |
|---|---|---|---|---|
| 1 | CID format | P1 | — | ~15 |
| 2 | Chunking engine | P1 | Phase 1 | ~10 |
| 3 | Blob store + bundle construction | P1 | Phase 1 | ~12 |
| 4 | Merkle DAG builder + walker | P1 | Phases 2, 3 | ~15 |
| 5 | W-TinyLFU cache | P2 | Phase 3 | ~12 |
| 6 | Delta encoding | P2 | Phase 3 | ~8 |
| 7 | Transport bridges (Reticulum, Zenoh, Flatpack) | P3 | Phase 4 + existing crates | ~10 |

### Not Building (YAGNI)

- Compression (bz2, zstd) — orthogonal, future layer
- Erasure coding (Reed-Solomon) — not needed for initial design
- Swarming/BitTorrent-style parallel download — future optimization
- Push-based replication — pull-only initially
- Range requests within blobs — blobs are atomic initially
- Cooperative caching protocols — each node manages independently
- Cache placement optimization — emergent via independent W-TinyLFU
- Variable-size bitmask adaptation — marginal dedup gains
- Global consensus on reverse mappings — eventual consistency sufficient
