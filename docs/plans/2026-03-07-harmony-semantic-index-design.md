# Harmony Semantic Index (HSI) Design

**Status:** Approved design — future implementation
**Date:** 2026-03-07
**Scope:** New `harmony-semantic` crate + CID type allocation

## Vision

The Harmony Semantic Index (HSI) is a decentralized, content-addressed
semantic tagging system. Every piece of content in the Harmony network
gets a compact binary embedding vector stored as a sidecar blob, enabling
nearest-neighbor search, semantic navigation, and concept-space browsing
without centralized indices.

Think of it as a next-generation Dewey Decimal System: instead of
hierarchical categories assigned by librarians, content self-organizes
into a continuous geometric space where proximity equals semantic
similarity.

### Key properties

- **Binary-quantized MRL vectors** — Matryoshka Representation Learning
  front-loads semantic information into early dimensions, enabling
  progressive resolution from 64 bits up to 1024 bits. Binary
  quantization (float > 0 → 1, ≤ 0 → 0) yields 32× compression
  over float32 while retaining ~80% retrieval performance at 256 bits.
- **Hardware-aligned tiers** — 64-bit (GPR), 128-bit (SSE), 256-bit
  (AVX2), 512-bit (AVX-512), 1024-bit (2×AVX-512). Comparison is
  XOR + POPCNT — approximately 2 CPU cycles per operation.
- **Content-addressed sidecars** — Semantic tags are ordinary Harmony
  blobs with their own CIDs, linked to the content they describe.
  They replicate, cache, and garbage-collect using the existing
  content infrastructure.
- **Model-agnostic** — A 4-byte model fingerprint in every sidecar
  enables mixed-model coexistence and graceful migration without
  re-indexing the entire network.
- **256-bit dual-address thesis** — A 256-bit semantic vector alongside
  a 256-bit CID gives every piece of content dual coordinates: identity
  (what it *is*) and meaning (what it's *about*).

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                    Application Layer                     │
│  Search queries · browse-by-concept · "more like this"  │
└──────────────────────────┬──────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────┐
│                    Navigation Layer                      │
│  Basis vectors · vector arithmetic · two-stage retrieval │
│  (coarse binary scan → fine float32 re-rank)            │
└──────────────────────────┬──────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────┐
│                     Storage Layer                        │
│  Sidecar blobs · collection blobs · self-tagged chunks  │
│  (all content-addressed, all replicate via athenaeum)   │
└─────────────────────────────────────────────────────────┘
```

Three layers, each independently useful:

1. **Storage** — Binary sidecar blobs alongside content. Works even if
   no navigation or search exists yet — the vectors are there for
   future use.
2. **Navigation** — Basis vectors define conceptual axes (e.g.
   "technical ↔ artistic", "theoretical ↔ applied"). Vector arithmetic
   enables compositional queries ("Python + machine learning − Java").
3. **Application** — User-facing features built on the navigation
   layer: semantic search, concept browsing, recommendation.

## Sidecar Blob Format

A sidecar blob is a compact binary structure stored as a regular Harmony
blob (using `CidType::ReservedA` — see CID type allocation below). It
contains the multi-tier MRL binary vectors for a single piece of content.

### Layout (284 bytes)

```
Offset  Size   Field
──────  ─────  ─────────────────────────────────
  0       4    Magic: 0x48 0x53 0x49 0x01 ("HSI" + version 1)
  4       4    Model fingerprint (first 4 bytes of SHA-256(model_id))
  8      32    Target CID (the content this sidecar describes)
 40       8    Tier 1: 64-bit binary vector
 48      16    Tier 2: 128-bit binary vector
 64      32    Tier 3: 256-bit binary vector
 96      64    Tier 4: 512-bit binary vector
160     128    Tier 5: 1024-bit binary vector (optional — zero-filled
               if model outputs fewer dimensions)
──────  ─────
288     total
```

The tiers are **not** independent embeddings — they are nested prefixes
of the same MRL vector. Tier 1 is the first 64 dimensions, Tier 2 is
the first 128, and so on. This is the fundamental insight of Matryoshka
Representation Learning: the model is trained so that truncating to
any prefix retains as much information as possible.

**Tier selection at query time:** A node can choose which tier to use
based on its hardware and latency budget. A constrained device scans
Tier 1 (64-bit, single GPR comparison). A desktop scans Tier 3
(256-bit, single AVX2 comparison). A search server uses Tier 5 for
maximum precision.

### CID type allocation

Sidecar blobs use `CidType::ReservedA` (tag prefix `1111_1111_10`,
2-bit checksum). This is the first of four reserved CID type slots.
The CID's 20-bit size field holds the sidecar payload size (288 bytes),
and the 224-bit hash is the truncated SHA-256 of the sidecar content.

This means sidecar CIDs are distinguishable from regular blobs at the
CID level — a node can tell whether a CID refers to content or to
semantic metadata without fetching the blob.

## Collection Blobs

A collection blob groups up to 256 content CIDs that are semantically
near each other, forming a navigable cluster. Collections enable
browsing ("show me everything near this concept") without scanning
the entire content store.

### Layout

```
Offset  Size      Field
──────  ────────  ──────────────────────────
  0       4       Magic: 0x48 0x53 0x43 0x01 ("HSC" + version 1)
  4       4       Model fingerprint
  8       4       Entry count (u32, max 256)
 12      32       Centroid: 256-bit binary vector of the cluster center
 44       N×64    Entries: [32-byte CID + 32-byte Tier 3 vector] × N
──────  ────────
Total:   44 + N×64  (max ≈ 16,428 bytes for 256 entries)
```

Collections are also stored as `CidType::ReservedA` blobs (they share
the same CID type as sidecars — the magic bytes disambiguate).

**Collection lifecycle:**
- Created by any node that accumulates enough semantically similar content.
- Immutable once created (content-addressed).
- New content near the centroid triggers a new collection version.
- Old collections remain valid and addressable — they just describe a
  historical cluster state.

## Embedding Pipeline

```
Content  ──►  Chunking  ──►  Embedding Model  ──►  MRL Vector (float32)
                                                         │
                                              Binary Quantization
                                              (> 0 → 1, ≤ 0 → 0)
                                                         │
                                              Pack into Tier Buffers
                                              (64 / 128 / 256 / 512 / 1024 bit)
                                                         │
                                              Sidecar Blob  ──►  Content Store
```

### Model choice

**Primary:** `pplx-embed-v1-0.6B` (Perplexity, Qwen3 backbone, 0.6B
params, MIT license). Purpose-built for binary quantization with MRL
training. Produces 1024-dimensional float32 vectors that quantize to
binary with minimal quality loss.

**Why this model:**
- Explicitly trained for binary quantization (not retrofitted).
- Qwen3 backbone provides strong Chinese-English cross-lingual
  capability, aligning with Harmony's goal of bridging information
  across language barriers.
- 0.6B parameters — small enough to run on consumer hardware, large
  enough for high-quality embeddings.
- MIT license — compatible with both Harmony's Apache-2.0/MIT core
  and harmony-os's GPL-2.0-or-later.

**Fallback / future:** As open-weight models improve (e.g. Qwen3.5
embedding variants), the model fingerprint system allows seamless
migration. Nodes can re-embed their local content with a new model and
publish updated sidecars — the old sidecars remain valid for nodes
still using the old model.

### The Platonic Representation Hypothesis

Different embedding models trained on similar data converge toward the
same geometric structure (the "Platonic representation"). This means
binary vectors from different models, while not bit-identical, will
have high Hamming similarity for semantically similar content. The
model fingerprint exists for precision, but approximate cross-model
search works surprisingly well in practice.

## Semantic Navigation

### Basis vectors

A basis vector is a unit-length direction in embedding space that
represents a conceptual axis. Examples:

- `science ↔ art` — separates scientific papers from creative works
- `theoretical ↔ applied` — separates theory from practice
- `beginner ↔ expert` — separates introductory from advanced content

Basis vectors are computed by embedding representative text for each
pole and taking the difference vector (normalized). They are published
as regular sidecars with a well-known target CID (a sentinel value).

**Navigation:** Project any content's embedding onto a basis vector to
get a scalar position along that axis. This enables browsing like
"show me content that is more theoretical" or "find the most technical
items in this cluster."

### Vector arithmetic

Compositional queries via vector addition and subtraction:

- `"Python" + "machine learning"` → finds Python ML content
- `"cooking" − "meat" + "plant"` → finds vegan recipes
- `content_vector + basis_direction × 0.5` → explore nearby concepts

### Two-stage retrieval

1. **Coarse scan** — Hamming distance over Tier 3 (256-bit) binary
   vectors. XOR + POPCNT. Scan millions of vectors per second on
   commodity hardware. Return top-K candidates.
2. **Fine re-rank** — If full float32 vectors are available (from
   self-tagged athenaeum chunks or fetched on demand), re-rank the
   top-K using cosine similarity for maximum precision.

For most use cases, the coarse scan alone is sufficient. The re-rank
step is optional and only needed when precision matters more than
latency (e.g. academic search).

## Self-Tagging

Content stored in the athenaeum (Harmony's content-addressed storage)
can optionally embed its own semantic vector in a reserved region of
its first or last 4 KiB chunk.

### Mechanism

The first 4 KiB chunk of an athenaeum file can contain:
- The standard chunk payload
- A trailing self-tag region (if the chunk has room)

The self-tag uses the existing CID type system for signaling: a chunk
whose CID uses `CidType::ReservedA` indicates it contains a self-tag.
Nodes that don't understand self-tags treat it as an opaque blob —
backward compatible.

### Self-tag format

```
Offset  Size    Field
──────  ──────  ──────────────────────────
  0       4     Magic: 0x48 0x53 0x54 0x01 ("HST" + version 1)
  4       4     Model fingerprint
  8    4096     Full float32 MRL vector (1024 dimensions × 4 bytes)
4104     128    Tier 5: 1024-bit binary vector (redundant but cheap)
──────  ──────
4232    total
```

Self-tags are optional and primarily useful for high-value content
that benefits from the full float32 vector for precise re-ranking.
Most content only gets a sidecar (288 bytes of binary vectors).

## Future-Proofing

### Model migration

When a better embedding model becomes available:

1. Nodes that adopt the new model re-embed their local content and
   publish new sidecars with the new model fingerprint.
2. Old sidecars remain valid — they still have correct CIDs and
   hash-verified content. Nodes using the old model can still use them.
3. Over time, the network converges on the new model as nodes update.
4. The Platonic Representation Hypothesis suggests that cross-model
   approximate search degrades gracefully rather than failing.

### Mixed-model coexistence

The model fingerprint in every sidecar and collection enables:
- **Filtering:** "Only show results from model X" for precision.
- **Mixing:** Approximate search across models using the Platonic
  convergence property.
- **Migration tracking:** Network-wide statistics on model adoption.

### Reserved capacity

- `CidType::ReservedB` through `ReservedD` remain available for future
  semantic features (e.g. multimodal embeddings, temporal vectors).
- The sidecar version byte (currently 0x01) allows format evolution.
- Tier 5 (1024-bit) can be zero-filled for models with fewer
  dimensions, maintaining format compatibility.

## Crate & Bead Structure

### New crate: `harmony-semantic`

```
harmony-semantic/
  src/
    lib.rs          — public API
    sidecar.rs      — sidecar blob encode/decode
    collection.rs   — collection blob encode/decode
    quantize.rs     — float32 → binary quantization + tier packing
    distance.rs     — Hamming distance (XOR + POPCNT, SIMD-aware)
    fingerprint.rs  — model fingerprint computation
```

**Dependencies:** `harmony-crypto` (for SHA-256 hashing),
`harmony-content` (for CID construction with ReservedA).

**No runtime dependency on any ML framework.** The embedding model
runs externally (Python, ONNX, etc.) and feeds float32 vectors into
`harmony-semantic` for quantization and storage. This keeps the crate
`no_std`-compatible.

### Suggested beads

1. **Sidecar blob codec** — Encode/decode the 288-byte sidecar format,
   including model fingerprint and multi-tier binary vectors.
   Allocate `CidType::ReservedA` for semantic blobs.

2. **Binary quantization + Hamming distance** — Float32 → binary
   quantization with MRL tier packing. SIMD-aware Hamming distance
   (with scalar fallback). Benchmarks.

3. **Collection blob codec** — Encode/decode collection blobs. Centroid
   computation. Entry management (add/remove CIDs).

4. **Semantic navigation primitives** — Basis vector projection.
   Vector arithmetic (add, subtract, scale). Two-stage retrieval
   (coarse binary scan + optional float32 re-rank).

## Design Principles

- **YAGNI:** No ML model integration in the crate. No index structures
  beyond flat scan. No distributed query protocol. Those come later
  when we have real usage data.
- **Content-addressed everything:** Sidecars, collections, and
  self-tags are all ordinary content blobs. No special storage paths.
- **Progressive resolution:** Every operation works at any tier.
  Constrained devices use Tier 1. Search servers use Tier 5. The
  same code path, different precision.
- **Language as a bridge:** The Qwen3 backbone's Chinese-English
  capability is a feature, not a coincidence. The internet should be
  useful to as many people as possible, and semantic search that works
  across languages is a concrete step toward that goal.
