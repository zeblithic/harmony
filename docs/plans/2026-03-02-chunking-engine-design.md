# Content-Defined Chunking Engine Design

**Goal:** Implement a Gear hash / FastCDC content-defined chunker for `harmony-content` that splits arbitrary byte streams into deduplication-friendly chunks with deterministic, content-derived boundaries.

**Architecture:** A stateful `Chunker` struct (sans-I/O, no internal data buffer) that scans bytes via a rolling Gear hash and emits cut points when the hash matches a target mask. Configurable min/avg/max chunk sizes with production defaults of 256KB/512KB/1MB. A `chunk_all()` convenience function wraps the streaming API for one-shot use.

---

## Gear Hash

Rolling hash: one shift-and-XOR per byte using a 256-entry `u64` lookup table.

```
hash = (hash << 1).wrapping_add(GEAR_TABLE[byte as usize]);
```

The lookup table is generated at compile time using a const fn SplitMix64 PRNG with a fixed seed. Deterministic, reproducible, no external data.

A chunk boundary is found when `hash & mask == 0`, where `mask = avg_chunk_size - 1` (works because avg is a power of 2; for 512KB: `0x7_FFFF`).

**FastCDC two-mask optimization:** Within the min→normalization region (min to min+avg/2), use a stricter mask (fewer boundaries). Beyond normalization through max, use the standard mask. This produces a tighter distribution around the target average.

Within the 0→min region, skip hash checks entirely (boundaries would be rejected anyway).

## API

```rust
pub struct ChunkerConfig {
    pub min_chunk: usize,   // Hard minimum — no cuts below this
    pub avg_chunk: usize,   // Target average (must be power of 2)
    pub max_chunk: usize,   // Hard maximum — force cut here
}

impl ChunkerConfig {
    pub const DEFAULT: Self = Self {
        min_chunk: 256 * 1024,    // 256 KB
        avg_chunk: 512 * 1024,    // 512 KB
        max_chunk: 1_048_575,     // MAX_PAYLOAD_SIZE
    };
}

pub struct Chunker {
    config: ChunkerConfig,
    hash: u64,
    pos: usize,          // bytes consumed since last cut point
}

impl Chunker {
    pub fn new(config: ChunkerConfig) -> Result<Self, ContentError>;
    pub fn feed(&mut self, data: &[u8]) -> Vec<usize>;
    pub fn finalize(&mut self) -> Option<usize>;
}

pub fn chunk_all(data: &[u8], config: &ChunkerConfig) -> Vec<Range<usize>>;
```

- `feed()` returns relative cut points within the slice passed. Caller accumulates absolute offsets.
- `finalize()` emits trailing bytes as the last chunk (even if below min).
- `chunk_all()` wraps `Chunker` internally, returns `Vec<Range<usize>>`.

## Error Handling

Single error variant: `ContentError::InvalidChunkerConfig { reason: &'static str }`.

Validation at construction:
- `min_chunk == 0` → error
- `avg_chunk` not a power of 2 → error
- `min >= avg` or `avg >= max` → error
- `max > MAX_PAYLOAD_SIZE` → error

Edge cases (handled, not errors):
- Empty input → no chunks
- Input smaller than min → single chunk
- Tail after last cut → always emitted by finalize
- Pathological data → min/max bounds enforce chunk sizes

## Testing (~10-12 tests)

Using small chunk sizes (min=64, avg=128, max=256) for fast tests.

**Determinism & correctness:**
- Same data → same cut points
- All chunks respect min/max bounds
- Chunk sizes cluster around average
- Streaming matches one-shot

**Boundary enforcement:**
- Input < min → single chunk
- Input at max → forced cut
- Input of max+1 → two chunks
- Empty input → no chunks

**Deduplication property:**
- Insert bytes in middle, re-chunk — only nearby chunks change

**Gear hash unit tests:**
- Table generation deterministic
- Hash changes on any byte change
