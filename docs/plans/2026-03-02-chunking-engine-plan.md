# Content-Defined Chunking Engine Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement a Gear hash / FastCDC content-defined chunker in `harmony-content` that splits byte streams into deduplication-friendly chunks with deterministic, content-derived boundaries.

**Architecture:** A new `chunker.rs` module in `harmony-content` containing: (1) a compile-time-generated Gear hash lookup table via const fn SplitMix64, (2) a `ChunkerConfig` struct with validation, (3) a stateful `Chunker` struct implementing the FastCDC two-mask algorithm with streaming `feed()`/`finalize()` API, and (4) a `chunk_all()` convenience function. No I/O, no allocations in the hot path — pure sans-I/O data processing.

**Tech Stack:** Rust, const fn (compile-time table generation), existing `harmony-content` crate infrastructure (`ContentError`, `cid::MAX_PAYLOAD_SIZE`)

---

### Task 1: Gear Hash Lookup Table (const fn generation)

**Files:**
- Create: `crates/harmony-content/src/chunker.rs`
- Modify: `crates/harmony-content/src/lib.rs` (add `pub mod chunker;`)

**Context:** The Gear hash maps each byte to a u64 via a 256-entry lookup table. We generate this at compile time using a const fn SplitMix64 PRNG with a fixed seed. SplitMix64 is a simple, well-distributed PRNG: `state = state.wrapping_add(0x9e3779b97f4a7c15); z = state; z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9); z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb); z ^ (z >> 31)`. The seed value doesn't matter as long as it's fixed — we use `0` for simplicity.

**Step 1: Write the failing test**

Add to `crates/harmony-content/src/chunker.rs`:

```rust
use crate::cid::MAX_PAYLOAD_SIZE;
use crate::error::ContentError;
use std::ops::Range;

/// Fixed seed for the SplitMix64 PRNG used to generate the Gear hash table.
const SPLITMIX_SEED: u64 = 0;

/// Generate a single SplitMix64 value from state, returning (value, next_state).
const fn splitmix64(state: u64) -> (u64, u64) {
    let s = state.wrapping_add(0x9e3779b97f4a7c15);
    let mut z = s;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
    z = z ^ (z >> 31);
    (z, s)
}

/// Compile-time generated 256-entry Gear hash lookup table.
const fn generate_gear_table() -> [u64; 256] {
    let mut table = [0u64; 256];
    let mut state = SPLITMIX_SEED;
    let mut i = 0;
    while i < 256 {
        let (value, next_state) = splitmix64(state);
        table[i] = value;
        state = next_state;
        i += 1;
    }
    table
}

/// The Gear hash lookup table — 256 pseudo-random u64 values.
const GEAR_TABLE: [u64; 256] = generate_gear_table();

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gear_table_is_deterministic() {
        // Re-generate and compare — must be identical every time.
        let table2 = generate_gear_table();
        let mut i = 0;
        while i < 256 {
            assert_eq!(GEAR_TABLE[i], table2[i], "mismatch at index {}", i);
            i += 1;
        }
    }

    #[test]
    fn gear_table_has_no_obvious_patterns() {
        // No two adjacent entries should be equal (extremely unlikely with good PRNG).
        for i in 0..255 {
            assert_ne!(
                GEAR_TABLE[i],
                GEAR_TABLE[i + 1],
                "adjacent entries {} and {} are equal",
                i,
                i + 1
            );
        }
        // No entry should be zero (extremely unlikely with good PRNG).
        for (i, &val) in GEAR_TABLE.iter().enumerate() {
            assert_ne!(val, 0, "entry {} is zero", i);
        }
    }

    #[test]
    fn gear_table_first_entry_is_stable() {
        // Pin the first entry so we detect accidental seed/algorithm changes.
        // SplitMix64 with state 0: state becomes 0x9e3779b97f4a7c15,
        // then z = 0x9e3779b97f4a7c15, transformations applied.
        assert_eq!(GEAR_TABLE[0], 0xe220a8397b1dcdaf);
    }
}
```

Add to `crates/harmony-content/src/lib.rs`:

```rust
pub mod blob;
pub mod bundle;
pub mod chunker;
pub mod cid;
pub mod error;
```

**Step 2: Run test to verify it passes**

Run: `cargo test -p harmony-content gear_table`
Expected: 3 tests PASS

**Step 3: Commit**

```bash
git add crates/harmony-content/src/chunker.rs crates/harmony-content/src/lib.rs
git commit -m "feat(content): add Gear hash lookup table with const fn SplitMix64 generation"
```

---

### Task 2: ChunkerConfig with Validation

**Files:**
- Modify: `crates/harmony-content/src/chunker.rs`
- Modify: `crates/harmony-content/src/error.rs` (add `InvalidChunkerConfig` variant)

**Context:** `ChunkerConfig` holds min/avg/max chunk sizes. Validation: `min > 0`, `avg` is a power of 2, `min < avg < max`, `max <= MAX_PAYLOAD_SIZE`. The error variant uses `&'static str` for the reason (no allocation needed — these are fixed messages).

**Step 1: Add the error variant**

Add to `crates/harmony-content/src/error.rs` after the `EmptyBundle` variant:

```rust
    #[error("invalid chunker config: {reason}")]
    InvalidChunkerConfig { reason: &'static str },
```

**Step 2: Write the failing test**

Add to `crates/harmony-content/src/chunker.rs` (after the Gear table code, before `#[cfg(test)]`):

```rust
/// Configuration for content-defined chunking.
///
/// Defines the minimum, average, and maximum chunk sizes.
/// The average must be a power of 2 (used as a bit mask for boundary detection).
#[derive(Debug, Clone, Copy)]
pub struct ChunkerConfig {
    /// Hard minimum chunk size — no cut points below this.
    pub min_chunk: usize,
    /// Target average chunk size — must be a power of 2.
    pub avg_chunk: usize,
    /// Hard maximum chunk size — force a cut at this boundary.
    pub max_chunk: usize,
}

impl ChunkerConfig {
    /// Production defaults: 256KB min, 512KB avg, ~1MB max.
    pub const DEFAULT: Self = Self {
        min_chunk: 256 * 1024,
        avg_chunk: 512 * 1024,
        max_chunk: MAX_PAYLOAD_SIZE,
    };

    /// Validate the configuration.
    pub fn validate(&self) -> Result<(), ContentError> {
        if self.min_chunk == 0 {
            return Err(ContentError::InvalidChunkerConfig {
                reason: "min_chunk must be > 0",
            });
        }
        if !self.avg_chunk.is_power_of_two() {
            return Err(ContentError::InvalidChunkerConfig {
                reason: "avg_chunk must be a power of 2",
            });
        }
        if self.min_chunk >= self.avg_chunk {
            return Err(ContentError::InvalidChunkerConfig {
                reason: "min_chunk must be < avg_chunk",
            });
        }
        if self.avg_chunk >= self.max_chunk {
            return Err(ContentError::InvalidChunkerConfig {
                reason: "avg_chunk must be < max_chunk",
            });
        }
        if self.max_chunk > MAX_PAYLOAD_SIZE {
            return Err(ContentError::InvalidChunkerConfig {
                reason: "max_chunk must be <= MAX_PAYLOAD_SIZE",
            });
        }
        Ok(())
    }
}
```

Add tests:

```rust
    #[test]
    fn config_default_is_valid() {
        ChunkerConfig::DEFAULT.validate().unwrap();
    }

    #[test]
    fn config_rejects_zero_min() {
        let config = ChunkerConfig {
            min_chunk: 0,
            avg_chunk: 128,
            max_chunk: 256,
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn config_rejects_non_power_of_two_avg() {
        let config = ChunkerConfig {
            min_chunk: 64,
            avg_chunk: 100,
            max_chunk: 256,
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn config_rejects_min_ge_avg() {
        let config = ChunkerConfig {
            min_chunk: 128,
            avg_chunk: 128,
            max_chunk: 256,
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn config_rejects_avg_ge_max() {
        let config = ChunkerConfig {
            min_chunk: 64,
            avg_chunk: 256,
            max_chunk: 256,
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn config_rejects_max_exceeding_payload() {
        let config = ChunkerConfig {
            min_chunk: 64,
            avg_chunk: 128,
            max_chunk: MAX_PAYLOAD_SIZE + 1,
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn config_small_valid() {
        let config = ChunkerConfig {
            min_chunk: 64,
            avg_chunk: 128,
            max_chunk: 256,
        };
        config.validate().unwrap();
    }
```

**Step 3: Run tests to verify they pass**

Run: `cargo test -p harmony-content config_`
Expected: 7 tests PASS

**Step 4: Commit**

```bash
git add crates/harmony-content/src/chunker.rs crates/harmony-content/src/error.rs
git commit -m "feat(content): add ChunkerConfig with validation"
```

---

### Task 3: Chunker Struct — Core FastCDC Algorithm

**Files:**
- Modify: `crates/harmony-content/src/chunker.rs`

**Context:** The `Chunker` struct implements the FastCDC two-mask algorithm. Key details:

- **Masks:** `mask_s` (strict, for the region between min and normalization point) has more bits set, producing fewer boundaries. `mask_l` (loose, for the region between normalization and max) has fewer bits set, producing more boundaries. `mask_l = avg_chunk - 1` (standard mask). `mask_s = mask_l << 1 | 1` (one extra bit — roughly halves boundary probability).
- **Normalization point:** `min_chunk + avg_chunk / 2`. This shifts the chunk size distribution closer to the target average.
- **Regions:** `[0, min)` — skip hash checks. `[min, norm)` — use `mask_s` (strict). `[norm, max)` — use `mask_l` (loose). `max` — force cut.
- **State:** `hash: u64` (rolling Gear hash), `pos: usize` (bytes since last cut).
- **`feed()`** returns relative cut offsets within the provided slice. After each cut, `pos` resets to 0 and `hash` resets to 0.
- **`finalize()`** returns the remaining byte count (the tail chunk), or `None` if `pos == 0`.

**Step 1: Write the implementation and tests**

Add to `crates/harmony-content/src/chunker.rs` (after `ChunkerConfig`):

```rust
/// Content-defined chunker using the Gear hash / FastCDC algorithm.
///
/// Scans bytes through a rolling hash and emits cut points at content-derived
/// boundaries. No internal data buffer — the caller owns the bytes.
///
/// Two-mask FastCDC: uses a strict mask between min and normalization point,
/// and a loose mask between normalization and max, to produce a tighter
/// distribution around the target average.
pub struct Chunker {
    min_chunk: usize,
    max_chunk: usize,
    norm_point: usize,
    mask_s: u64,
    mask_l: u64,
    hash: u64,
    pos: usize,
}

impl Chunker {
    /// Create a new chunker with the given configuration.
    pub fn new(config: ChunkerConfig) -> Result<Self, ContentError> {
        config.validate()?;
        let mask_l = (config.avg_chunk as u64) - 1;
        let mask_s = (mask_l << 1) | 1;
        let norm_point = config.min_chunk + config.avg_chunk / 2;
        Ok(Chunker {
            min_chunk: config.min_chunk,
            max_chunk: config.max_chunk,
            norm_point,
            mask_s,
            mask_l,
            hash: 0,
            pos: 0,
        })
    }

    /// Scan `data` and return cut point offsets relative to the start of `data`.
    ///
    /// Each returned offset marks the end of a chunk (exclusive). The caller
    /// is responsible for tracking absolute positions across multiple `feed()` calls.
    ///
    /// After the last cut point in the returned vec, any remaining bytes in `data`
    /// are buffered internally (tracked by `pos`). Call `feed()` again with more
    /// data, or `finalize()` to emit the tail chunk.
    pub fn feed(&mut self, data: &[u8]) -> Vec<usize> {
        let mut cuts = Vec::new();
        let mut i = 0;

        while i < data.len() {
            // Always advance the hash (even in the skip region, to maintain state)
            self.hash = (self.hash << 1).wrapping_add(GEAR_TABLE[data[i] as usize]);
            self.pos += 1;
            i += 1;

            // Force cut at max
            if self.pos >= self.max_chunk {
                cuts.push(i);
                self.hash = 0;
                self.pos = 0;
                continue;
            }

            // Skip boundary checks below min
            if self.pos < self.min_chunk {
                continue;
            }

            // Two-mask FastCDC:
            // - Between min and normalization: strict mask (fewer boundaries)
            // - Between normalization and max: loose mask (more boundaries)
            let mask = if self.pos < self.norm_point {
                self.mask_s
            } else {
                self.mask_l
            };

            if self.hash & mask == 0 {
                cuts.push(i);
                self.hash = 0;
                self.pos = 0;
            }
        }

        cuts
    }

    /// Signal end-of-input. Returns the size of the trailing chunk,
    /// or `None` if there are no remaining bytes.
    pub fn finalize(&mut self) -> Option<usize> {
        if self.pos == 0 {
            return None;
        }
        let remaining = self.pos;
        self.hash = 0;
        self.pos = 0;
        Some(remaining)
    }
}
```

Add tests:

```rust
    /// Helper: small config for fast tests.
    fn small_config() -> ChunkerConfig {
        ChunkerConfig {
            min_chunk: 64,
            avg_chunk: 128,
            max_chunk: 256,
        }
    }

    #[test]
    fn chunker_deterministic() {
        let config = small_config();
        let data: Vec<u8> = (0..1024).map(|i| (i % 256) as u8).collect();

        let mut c1 = Chunker::new(config).unwrap();
        let cuts1 = c1.feed(&data);
        let tail1 = c1.finalize();

        let mut c2 = Chunker::new(config).unwrap();
        let cuts2 = c2.feed(&data);
        let tail2 = c2.finalize();

        assert_eq!(cuts1, cuts2);
        assert_eq!(tail1, tail2);
    }

    #[test]
    fn chunker_respects_min_max() {
        let config = small_config();
        let data: Vec<u8> = (0..4096).map(|i| (i * 7 % 256) as u8).collect();

        let mut chunker = Chunker::new(config).unwrap();
        let cuts = chunker.feed(&data);
        let _tail = chunker.finalize();

        // Verify all chunk sizes are within bounds
        let mut prev = 0;
        for &cut in &cuts {
            let chunk_size = cut - prev;
            assert!(
                chunk_size >= config.min_chunk,
                "chunk size {} < min {}",
                chunk_size,
                config.min_chunk
            );
            assert!(
                chunk_size <= config.max_chunk,
                "chunk size {} > max {}",
                chunk_size,
                config.max_chunk
            );
            prev = cut;
        }
        // Tail chunk can be smaller than min (it's the remainder)
    }

    #[test]
    fn chunker_small_input_single_chunk() {
        let config = small_config();
        let data = b"tiny";

        let mut chunker = Chunker::new(config).unwrap();
        let cuts = chunker.feed(data);
        let tail = chunker.finalize();

        assert!(cuts.is_empty(), "no cuts for data smaller than min_chunk");
        assert_eq!(tail, Some(4));
    }

    #[test]
    fn chunker_empty_input() {
        let config = small_config();
        let mut chunker = Chunker::new(config).unwrap();
        let cuts = chunker.feed(&[]);
        let tail = chunker.finalize();

        assert!(cuts.is_empty());
        assert_eq!(tail, None);
    }

    #[test]
    fn chunker_forces_cut_at_max() {
        let config = small_config(); // max = 256
        // All-zero data won't produce natural boundaries (hash stays 0,
        // but 0 & mask == 0 so it would match — use 0xFF to avoid that).
        // Actually, let's use data that specifically avoids natural cuts.
        // We need data where hash & mask != 0 for all positions in [min, max).
        // Simplest: max+1 bytes of data, verify we get at least one cut.
        let data = vec![0x42u8; config.max_chunk + 1];

        let mut chunker = Chunker::new(config).unwrap();
        let cuts = chunker.feed(&data);

        assert!(
            !cuts.is_empty(),
            "should have at least one cut for data > max_chunk"
        );
        // First cut must be at or before max_chunk
        assert!(cuts[0] <= config.max_chunk);
    }
```

**Step 2: Run tests to verify they pass**

Run: `cargo test -p harmony-content chunker`
Expected: All chunker tests PASS

**Step 3: Commit**

```bash
git add crates/harmony-content/src/chunker.rs
git commit -m "feat(content): add Chunker struct with FastCDC two-mask algorithm"
```

---

### Task 4: `chunk_all()` Convenience Function + Streaming Equivalence

**Files:**
- Modify: `crates/harmony-content/src/chunker.rs`

**Context:** `chunk_all()` takes a complete `&[u8]` and returns `Vec<Range<usize>>` — byte ranges for each chunk. It wraps `Chunker` internally. The critical property: `chunk_all(data)` must produce identical boundaries to `feed(data) + finalize()`.

**Step 1: Write the implementation and tests**

Add to `crates/harmony-content/src/chunker.rs` (after the `Chunker` impl):

```rust
/// Split data into chunks using content-defined chunking, returning byte ranges.
///
/// This is a convenience wrapper around `Chunker` for when all data is available
/// upfront. Returns the same boundaries as `Chunker::feed()` + `finalize()`.
pub fn chunk_all(data: &[u8], config: &ChunkerConfig) -> Result<Vec<Range<usize>>, ContentError> {
    let mut chunker = Chunker::new(*config)?;
    let cuts = chunker.feed(data);
    let tail = chunker.finalize();

    let mut ranges = Vec::with_capacity(cuts.len() + 1);
    let mut start = 0;
    for cut in cuts {
        ranges.push(start..cut);
        start = cut;
    }
    if let Some(remaining) = tail {
        ranges.push(start..start + remaining);
    }
    ranges
}
```

Add tests:

```rust
    #[test]
    fn chunk_all_matches_streaming() {
        let config = small_config();
        let data: Vec<u8> = (0..2048).map(|i| (i * 13 % 256) as u8).collect();

        // One-shot
        let ranges = chunk_all(&data, &config).unwrap();

        // Streaming
        let mut chunker = Chunker::new(config).unwrap();
        let cuts = chunker.feed(&data);
        let tail = chunker.finalize();

        // Reconstruct ranges from streaming output
        let mut streaming_ranges = Vec::new();
        let mut start = 0;
        for cut in cuts {
            streaming_ranges.push(start..cut);
            start = cut;
        }
        if let Some(remaining) = tail {
            streaming_ranges.push(start..start + remaining);
        }

        assert_eq!(ranges, streaming_ranges);
    }

    #[test]
    fn chunk_all_covers_entire_input() {
        let config = small_config();
        let data: Vec<u8> = (0..2048).map(|i| (i * 31 % 256) as u8).collect();
        let ranges = chunk_all(&data, &config).unwrap();

        // Ranges must be contiguous and cover the entire input
        assert_eq!(ranges.first().unwrap().start, 0);
        assert_eq!(ranges.last().unwrap().end, data.len());
        for window in ranges.windows(2) {
            assert_eq!(window[0].end, window[1].start, "gap between chunks");
        }
    }

    #[test]
    fn chunk_all_empty_input() {
        let config = small_config();
        let ranges = chunk_all(&[], &config).unwrap();
        assert!(ranges.is_empty());
    }

    #[test]
    fn chunk_all_streaming_multi_feed_equivalence() {
        let config = small_config();
        let data: Vec<u8> = (0..2048).map(|i| (i * 13 % 256) as u8).collect();

        // One-shot
        let ranges = chunk_all(&data, &config).unwrap();

        // Multi-feed streaming: feed in small increments
        let mut chunker = Chunker::new(config).unwrap();
        let mut all_cuts: Vec<usize> = Vec::new();
        let mut fed = 0;
        for chunk in data.chunks(100) {
            let cuts = chunker.feed(chunk);
            for cut in cuts {
                all_cuts.push(fed + cut);
            }
            fed += chunk.len();
        }
        let tail = chunker.finalize();

        let mut streaming_ranges = Vec::new();
        let mut start = 0;
        for cut in all_cuts {
            streaming_ranges.push(start..cut);
            start = cut;
        }
        if let Some(remaining) = tail {
            streaming_ranges.push(start..start + remaining);
        }

        assert_eq!(ranges, streaming_ranges);
    }
```

**Step 2: Run tests to verify they pass**

Run: `cargo test -p harmony-content chunk_all`
Expected: 4 tests PASS

**Step 3: Commit**

```bash
git add crates/harmony-content/src/chunker.rs
git commit -m "feat(content): add chunk_all() convenience function with streaming equivalence"
```

---

### Task 5: Deduplication Property Test

**Files:**
- Modify: `crates/harmony-content/src/chunker.rs`

**Context:** The whole point of CDC: inserting bytes in the middle should only change chunks near the edit point. Chunks far from the edit should have identical boundaries. This test verifies the deduplication property that justifies choosing CDC over fixed-size chunking.

**Step 1: Write the test**

```rust
    #[test]
    fn deduplication_insert_only_affects_nearby_chunks() {
        let config = small_config();

        // Create original data
        let original: Vec<u8> = (0..2048).map(|i| (i * 37 % 256) as u8).collect();
        let original_ranges = chunk_all(&original, &config).unwrap();
        let original_chunks: Vec<&[u8]> =
            original_ranges.iter().map(|r| &original[r.clone()]).collect();

        // Insert 10 bytes near the middle
        let insert_pos = 1024;
        let mut modified = original[..insert_pos].to_vec();
        modified.extend_from_slice(&[0xDE; 10]);
        modified.extend_from_slice(&original[insert_pos..]);

        let modified_ranges = chunk_all(&modified, &config).unwrap();
        let modified_chunks: Vec<&[u8]> =
            modified_ranges.iter().map(|r| &modified[r.clone()]).collect();

        // Count how many chunks from the original appear unchanged in the modified version
        let mut shared = 0;
        for orig_chunk in &original_chunks {
            if modified_chunks.contains(orig_chunk) {
                shared += 1;
            }
        }

        // At least some chunks should be shared (CDC property).
        // With small config and 2KB data, we expect most chunks to be reused.
        assert!(
            shared > 0,
            "CDC should preserve some chunks after a small edit, but none were shared"
        );
        // The total number of chunks shouldn't wildly change
        let count_diff =
            (original_ranges.len() as isize - modified_ranges.len() as isize).unsigned_abs();
        assert!(
            count_diff <= 2,
            "chunk count changed by {} — expected at most 2 for a small insertion",
            count_diff
        );
    }
```

**Step 2: Run tests to verify they pass**

Run: `cargo test -p harmony-content deduplication`
Expected: 1 test PASS

**Step 3: Run the full test suite**

Run: `cargo test -p harmony-content`
Expected: All tests PASS (existing + new chunker tests)

Run: `cargo clippy -p harmony-content`
Expected: No warnings

Run: `cargo fmt --all -- --check`
Expected: No formatting issues

**Step 4: Commit**

```bash
git add crates/harmony-content/src/chunker.rs
git commit -m "test(content): add CDC deduplication property test"
```
