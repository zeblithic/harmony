# KV Cache Compression Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add feature-gated 3-bit KV cache compression to `InferenceCache`, reducing memory ~5x between forward passes on edge devices.

**Architecture:** `InferenceCache` gains `compress()`/`decompress()` methods behind a `kv-compress` feature flag. Compression uses a simple uniform 3-bit quantizer (per-vector min/scale + packed 3-bit indices). A two-phase commit pattern ensures atomicity — on error, the cache rolls back to its pre-operation state. `forward()` gains a safety guard rejecting compressed caches.

**Tech Stack:** Rust, `candle-core` (Tensor ↔ f32 bridge), `half` (f16 conversion), `thiserror`

**Spec:** `docs/superpowers/specs/2026-03-27-turboquant-kv-compression-design.md`

**Spec deviation:** The spec references a `tq-kv` crate (actual: `turbo-quant` on crates.io). Investigation revealed `turbo-quant`'s `PolarCode` in-memory format uses lossless f32 radii + u16 angle indices = ~384 bytes per dim-128 vector, which is **1.5x larger** than the f16 original (256 bytes). For Phase 1 (compress/decompress per forward pass), we use a simple 3-bit uniform quantizer instead, achieving ~4.6x compression. The `turbo-quant` crate's unbiased inner-product estimation is only needed for Phase 2 fused attention (harmony-51dy), where attention scores are computed directly on compressed data without decompression.

---

## File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `crates/harmony-inference/Cargo.toml` | Modify | Add `kv-compress` feature flag |
| `crates/harmony-inference/src/error.rs` | Modify | Add `CompressionFailed`, `CacheCompressed` variants |
| `crates/harmony-inference/src/kv_compress.rs` | Create | `CompressedKvLayer`, 3-bit quantizer, pack/unpack, tensor bridge |
| `crates/harmony-inference/src/lib.rs` | Modify | New fields on `InferenceCache`, `compress()`/`decompress()`/`is_compressed()`/`memory_bytes()` methods, module declaration |
| `crates/harmony-inference/src/engine.rs` | Modify | `forward()` safety guard |

---

### Task 1: Feature Flag Setup

**Files:**
- Modify: `crates/harmony-inference/Cargo.toml`

No new dependencies needed — `candle-core` and `half` are already available in the workspace.

- [ ] **Step 1: Add feature flag**

In `crates/harmony-inference/Cargo.toml`, add to `[features]`:

```toml
[features]
default = []
cuda = ["candle-core/cuda", "candle-nn/cuda"]
metal = ["candle-core/metal", "candle-nn/metal"]
kv-compress = []
```

- [ ] **Step 2: Verify compilation**

Run: `cargo check -p harmony-inference`
Expected: OK

Run: `cargo check -p harmony-inference --features kv-compress`
Expected: OK

- [ ] **Step 3: Commit**

```bash
git add crates/harmony-inference/Cargo.toml
git commit -m "feat(inference): add kv-compress feature flag

Empty feature flag for KV cache compression. No code changes yet."
```

---

### Task 2: Error Variants

**Files:**
- Modify: `crates/harmony-inference/src/error.rs`

- [ ] **Step 1: Write failing tests**

Add to `crates/harmony-inference/src/error.rs` at the bottom:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(feature = "kv-compress")]
    fn compression_failed_displays_message() {
        let err = InferenceError::CompressionFailed("bad tensor".into());
        assert_eq!(err.to_string(), "compression failed: bad tensor");
    }

    #[test]
    #[cfg(feature = "kv-compress")]
    fn cache_compressed_displays_message() {
        let err = InferenceError::CacheCompressed;
        assert!(err.to_string().contains("compressed"));
    }
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test -p harmony-inference --features kv-compress -- error::tests`
Expected: FAIL — variants don't exist yet

- [ ] **Step 3: Add error variants**

In `crates/harmony-inference/src/error.rs`, add before the closing `}` of `InferenceError`:

```rust
    /// KV cache compression or decompression failed.
    #[cfg(feature = "kv-compress")]
    #[error("compression failed: {0}")]
    CompressionFailed(String),

    /// Forward pass attempted while cache is compressed — call decompress() first.
    #[cfg(feature = "kv-compress")]
    #[error("cache is compressed — call decompress() before forward()")]
    CacheCompressed,
```

- [ ] **Step 4: Run tests**

Run: `cargo test -p harmony-inference --features kv-compress -- error::tests`
Expected: PASS

Run: `cargo test -p harmony-inference`
Expected: PASS (feature-gated tests not compiled)

- [ ] **Step 5: Commit**

```bash
git add crates/harmony-inference/src/error.rs
git commit -m "feat(inference): add CompressionFailed and CacheCompressed error variants

Feature-gated behind kv-compress. CompressionFailed wraps quantization errors;
CacheCompressed prevents forward() on a compressed cache."
```

---

### Task 3: kv_compress Module — 3-Bit Quantizer Core

**Files:**
- Create: `crates/harmony-inference/src/kv_compress.rs`
- Modify: `crates/harmony-inference/src/lib.rs` (module declaration only)

This task implements the quantizer without any Tensor dependencies — pure `&[f32]` ↔ packed bytes.

- [ ] **Step 1: Create kv_compress.rs with types and pack/unpack**

Create `crates/harmony-inference/src/kv_compress.rs`:

```rust
//! 3-bit uniform KV cache compression.
//!
//! Compresses f16 KV cache tensors to ~52 bytes per 128-dim vector (vs 256
//! bytes at f16), a ~4.9x reduction. Each vector is independently quantized:
//! `min` and `scale` header (8 bytes f32) + 3-bit packed indices.
//!
//! Feature-gated behind `kv-compress`.

use crate::error::InferenceError;
use candle_core::{DType, Device, Tensor};

/// Maximum quantization index (2^3 - 1 = 7, i.e., 8 levels 0..=7).
const MAX_INDEX: u8 = 7;

/// A single compressed vector: per-vector min/scale + 3-bit packed indices.
#[derive(Clone, Debug)]
pub(crate) struct QuantizedVec {
    /// Minimum value of the original vector (dequant base).
    min: f32,
    /// Scale factor: `(max - min) / 7.0`. Zero if vector is constant.
    scale: f32,
    /// 3-bit packed quantization indices. `ceil(dim * 3 / 8)` bytes.
    packed: Vec<u8>,
}

impl QuantizedVec {
    /// Approximate memory usage in bytes.
    pub(crate) fn byte_size(&self) -> usize {
        // 4 (min) + 4 (scale) + packed data
        8 + self.packed.len()
    }
}

/// Compressed representation of one layer's K and V tensors.
#[derive(Clone, Debug)]
pub(crate) struct CompressedKvLayer {
    /// Compressed key vectors, row-major: [head0_tok0, head0_tok1, ..., headH_tokT].
    pub(crate) k: Vec<QuantizedVec>,
    /// Compressed value vectors, same layout as k.
    pub(crate) v: Vec<QuantizedVec>,
    /// Sequence length at compression time (for tensor shape reconstruction).
    pub(crate) seq_len: usize,
}

impl CompressedKvLayer {
    /// Approximate memory usage in bytes.
    pub(crate) fn byte_size(&self) -> usize {
        self.k.iter().map(|q| q.byte_size()).sum::<usize>()
            + self.v.iter().map(|q| q.byte_size()).sum::<usize>()
    }
}

// ── 3-bit packing ────────────────────────────────────────────────────

/// Pack 3-bit values (each 0..7) into bytes, 8 values per 3 bytes.
///
/// Bit layout within each 3-byte group (little-endian):
/// ```text
/// byte 0: [v0₂ v0₁ v0₀ | v1₂ v1₁ v1₀ | v2₁ v2₀]
/// byte 1: [v2₂ | v3₂ v3₁ v3₀ | v4₂ v4₁ v4₀ | v5₀]
/// byte 2: [v5₂ v5₁ | v6₂ v6₁ v6₀ | v7₂ v7₁ v7₀]
/// ```
fn pack_3bit(values: &[u8]) -> Vec<u8> {
    let num_bytes = (values.len() * 3 + 7) / 8;
    let mut packed = vec![0u8; num_bytes];
    for (i, &v) in values.iter().enumerate() {
        let bit_offset = i * 3;
        let byte_idx = bit_offset / 8;
        let bit_idx = bit_offset % 8;
        packed[byte_idx] |= (v & 0x7) << bit_idx;
        // Handle straddling a byte boundary (bit_idx > 5)
        if bit_idx > 5 && byte_idx + 1 < packed.len() {
            packed[byte_idx + 1] |= (v & 0x7) >> (8 - bit_idx);
        }
    }
    packed
}

/// Unpack 3-bit values from packed bytes.
fn unpack_3bit(packed: &[u8], count: usize) -> Vec<u8> {
    let mut values = Vec::with_capacity(count);
    for i in 0..count {
        let bit_offset = i * 3;
        let byte_idx = bit_offset / 8;
        let bit_idx = bit_offset % 8;
        let mut v = (packed[byte_idx] >> bit_idx) & 0x7;
        // Handle straddling
        if bit_idx > 5 && byte_idx + 1 < packed.len() {
            v |= (packed[byte_idx + 1] << (8 - bit_idx)) & 0x7;
        }
        values.push(v);
    }
    values
}

// ── Per-vector quantize/dequantize ───────────────────────────────────

/// Quantize an f32 slice to a QuantizedVec (3-bit uniform).
fn quantize_vec(data: &[f32]) -> QuantizedVec {
    if data.is_empty() {
        return QuantizedVec { min: 0.0, scale: 0.0, packed: vec![] };
    }

    let min = data.iter().copied().fold(f32::INFINITY, f32::min);
    let max = data.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let range = max - min;
    let scale = if range > 0.0 { range / MAX_INDEX as f32 } else { 0.0 };

    let indices: Vec<u8> = data.iter().map(|&v| {
        if scale == 0.0 {
            0
        } else {
            ((v - min) / scale).round().clamp(0.0, MAX_INDEX as f32) as u8
        }
    }).collect();

    QuantizedVec {
        min,
        scale,
        packed: pack_3bit(&indices),
    }
}

/// Dequantize a QuantizedVec back to f32 values.
fn dequantize_vec(qv: &QuantizedVec, dim: usize) -> Vec<f32> {
    let indices = unpack_3bit(&qv.packed, dim);
    indices.iter().map(|&idx| qv.min + idx as f32 * qv.scale).collect()
}

// ── Tensor bridge ────────────────────────────────────────────────────

/// Compress an f16 tensor `[1, num_kv_heads, seq_len, head_dim]` to quantized vecs.
///
/// Returns (quantized_vecs, seq_len). One QuantizedVec per (head, token) pair,
/// stored row-major: [head0_tok0, head0_tok1, ..., headH_tokT].
pub(crate) fn compress_tensor(
    tensor: &Tensor,
) -> Result<(Vec<QuantizedVec>, usize), InferenceError> {
    let (_batch, num_heads, seq_len, _head_dim) = tensor.dims4()
        .map_err(|e| InferenceError::CompressionFailed(format!("unexpected tensor shape: {e}")))?;

    let total_vecs = num_heads * seq_len;
    let flat = tensor
        .to_dtype(DType::F32)
        .and_then(|t| t.reshape((total_vecs, _head_dim)))
        .map_err(|e| InferenceError::CompressionFailed(e.to_string()))?;
    let rows = flat
        .to_vec2::<f32>()
        .map_err(|e| InferenceError::CompressionFailed(e.to_string()))?;

    let vecs: Vec<QuantizedVec> = rows.iter().map(|row| quantize_vec(row)).collect();
    Ok((vecs, seq_len))
}

/// Decompress quantized vecs back to an f16 tensor `[1, num_kv_heads, seq_len, head_dim]`.
pub(crate) fn decompress_tensor(
    vecs: &[QuantizedVec],
    num_kv_heads: usize,
    seq_len: usize,
    head_dim: usize,
    device: &Device,
) -> Result<Tensor, InferenceError> {
    let expected = num_kv_heads * seq_len;
    if vecs.len() != expected {
        return Err(InferenceError::CompressionFailed(format!(
            "expected {expected} quantized vecs, got {}", vecs.len()
        )));
    }

    let mut flat = Vec::with_capacity(expected * head_dim);
    for qv in vecs {
        flat.extend_from_slice(&dequantize_vec(qv, head_dim));
    }

    Tensor::from_vec(flat, (1, num_kv_heads, seq_len, head_dim), device)
        .and_then(|t| t.to_dtype(DType::F16))
        .map_err(|e| InferenceError::CompressionFailed(e.to_string()))
}

// ── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pack_unpack_roundtrip() {
        let values: Vec<u8> = (0..16).map(|i| i % 8).collect();
        let packed = pack_3bit(&values);
        let unpacked = unpack_3bit(&packed, values.len());
        assert_eq!(values, unpacked);
    }

    #[test]
    fn pack_unpack_all_sevens() {
        let values = vec![7u8; 24];
        let packed = pack_3bit(&values);
        let unpacked = unpack_3bit(&packed, 24);
        assert_eq!(values, unpacked);
    }

    #[test]
    fn pack_unpack_all_zeros() {
        let values = vec![0u8; 128];
        let packed = pack_3bit(&values);
        assert_eq!(packed.len(), 48); // 128 * 3 / 8 = 48
        let unpacked = unpack_3bit(&packed, 128);
        assert_eq!(values, unpacked);
    }

    #[test]
    fn quantize_dequantize_roundtrip() {
        let data: Vec<f32> = (0..128).map(|i| i as f32 / 128.0).collect();
        let qv = quantize_vec(&data);
        let restored = dequantize_vec(&qv, 128);

        assert_eq!(restored.len(), 128);
        let max_err: f32 = data.iter().zip(restored.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        // 3-bit quantization of [0, 1] range: step = 1/7 ≈ 0.143
        // Max error should be < half step ≈ 0.072
        assert!(max_err < 0.08, "max error {max_err} too large");
    }

    #[test]
    fn quantize_constant_vector() {
        let data = vec![42.0_f32; 128];
        let qv = quantize_vec(&data);
        let restored = dequantize_vec(&qv, 128);
        // Constant vector: all values should be exactly 42.0
        assert!(restored.iter().all(|&v| (v - 42.0).abs() < 1e-6));
    }

    #[test]
    fn quantize_empty_vector() {
        let qv = quantize_vec(&[]);
        assert_eq!(qv.byte_size(), 8); // just min + scale
        let restored = dequantize_vec(&qv, 0);
        assert!(restored.is_empty());
    }

    #[test]
    fn quantized_vec_byte_size() {
        let data = vec![1.0_f32; 128];
        let qv = quantize_vec(&data);
        // 8 (header) + 48 (128 * 3 / 8) = 56 bytes
        assert_eq!(qv.byte_size(), 56);
    }

    #[test]
    fn compressed_kv_layer_byte_size() {
        let data = vec![1.0_f32; 128];
        let qv = quantize_vec(&data);
        let layer = CompressedKvLayer {
            k: vec![qv.clone(); 8],  // 8 heads × 1 token
            v: vec![qv; 8],
            seq_len: 1,
        };
        // 2 tensors * 8 * 56 = 896 bytes
        assert_eq!(layer.byte_size(), 896);
    }

    #[test]
    fn compress_tensor_roundtrip() {
        let shape = (1, 8, 4, 128);
        let tensor = Tensor::rand(0f32, 1f32, shape, &Device::Cpu)
            .unwrap()
            .to_dtype(DType::F16)
            .unwrap();

        let (vecs, seq_len) = compress_tensor(&tensor).unwrap();
        assert_eq!(seq_len, 4);
        assert_eq!(vecs.len(), 32); // 8 heads * 4 tokens

        let restored = decompress_tensor(&vecs, 8, 4, 128, &Device::Cpu).unwrap();
        assert_eq!(restored.dims4().unwrap(), (1, 8, 4, 128));
        assert_eq!(restored.dtype(), DType::F16);
    }
}
```

- [ ] **Step 2: Add module declaration to lib.rs**

In `crates/harmony-inference/src/lib.rs`, add after the `pub(crate) mod qwen3_ext;` line:

```rust
#[cfg(feature = "kv-compress")]
pub(crate) mod kv_compress;
```

- [ ] **Step 3: Run tests**

Run: `cargo test -p harmony-inference --features kv-compress -- kv_compress::tests`
Expected: PASS (9 tests)

Run: `cargo test -p harmony-inference`
Expected: PASS (module not compiled without feature)

- [ ] **Step 4: Commit**

```bash
git add crates/harmony-inference/src/kv_compress.rs crates/harmony-inference/src/lib.rs
git commit -m "feat(inference): add 3-bit uniform KV cache quantizer

Per-vector min/scale header + packed 3-bit indices. 56 bytes per 128-dim
vector vs 256 bytes f16 = ~4.6x compression. Includes pack/unpack,
quantize/dequantize, and candle Tensor bridge."
```

---

### Task 4: InferenceCache New Fields, Constructor, and is_compressed()

**Files:**
- Modify: `crates/harmony-inference/src/lib.rs`

- [ ] **Step 1: Write failing tests**

Add to the `cache_tests` module in `crates/harmony-inference/src/lib.rs`:

```rust
    #[test]
    #[cfg(feature = "kv-compress")]
    fn new_cache_is_not_compressed() {
        let cache = InferenceCache::new(28, 128, 8);
        assert!(!cache.is_compressed());
    }

    #[test]
    #[cfg(feature = "kv-compress")]
    fn new_cache_compressed_vec_matches_layers() {
        let cache = InferenceCache::new(28, 128, 8);
        assert_eq!(cache.compressed.len(), 28);
        assert!(cache.compressed.iter().all(|c| c.is_none()));
    }
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test -p harmony-inference --features kv-compress -- cache_tests`
Expected: FAIL — fields don't exist yet

- [ ] **Step 3: Add fields and update constructor**

In `crates/harmony-inference/src/lib.rs`, add to the `InferenceCache` struct before the closing `}`:

```rust
    /// Per-layer compressed K/V state. Populated by compress(), consumed by decompress().
    #[cfg(feature = "kv-compress")]
    pub(crate) compressed: Vec<Option<kv_compress::CompressedKvLayer>>,

    /// Whether the cache is currently in compressed form.
    #[cfg(feature = "kv-compress")]
    pub(crate) is_compressed: bool,
```

Update `InferenceCache::new()`:

```rust
    pub fn new(num_layers: usize, head_dim: usize, num_kv_heads: usize) -> Self {
        Self {
            layers: (0..num_layers).map(|_| None).collect(),
            position: 0,
            num_layers,
            head_dim,
            num_kv_heads,
            #[cfg(feature = "kv-compress")]
            compressed: (0..num_layers).map(|_| None).collect(),
            #[cfg(feature = "kv-compress")]
            is_compressed: false,
        }
    }
```

Add a new impl block after the existing `impl InferenceCache`:

```rust
#[cfg(feature = "kv-compress")]
impl InferenceCache {
    /// Whether the cache is currently in compressed form.
    pub fn is_compressed(&self) -> bool {
        self.is_compressed
    }
}
```

- [ ] **Step 4: Run tests**

Run: `cargo test -p harmony-inference --features kv-compress -- cache_tests`
Expected: PASS

Run: `cargo test -p harmony-inference`
Expected: PASS (existing tests unaffected)

- [ ] **Step 5: Commit**

```bash
git add crates/harmony-inference/src/lib.rs
git commit -m "feat(inference): add compressed/is_compressed fields to InferenceCache

Feature-gated fields for kv-compress: compressed Vec and is_compressed flag.
Constructor initializes both to empty/false."
```

---

### Task 5: compress() Implementation

**Files:**
- Modify: `crates/harmony-inference/src/lib.rs`

- [ ] **Step 1: Write failing tests**

Add a new test module in `crates/harmony-inference/src/lib.rs`:

```rust
#[cfg(test)]
#[cfg(feature = "kv-compress")]
mod kv_compress_cache_tests {
    use super::*;
    use candle_core::{DType, Device, Tensor};

    /// Create a cache with `n_tokens` of random f16 KV tensors in layer 0.
    fn cache_with_data(num_layers: usize, num_kv_heads: usize, head_dim: usize, n_tokens: usize) -> InferenceCache {
        let mut cache = InferenceCache::new(num_layers, head_dim, num_kv_heads);
        if n_tokens > 0 {
            let shape = (1, num_kv_heads, n_tokens, head_dim);
            let k = Tensor::rand(0f32, 1f32, shape, &Device::Cpu)
                .unwrap()
                .to_dtype(DType::F16)
                .unwrap();
            let v = Tensor::rand(0f32, 1f32, shape, &Device::Cpu)
                .unwrap()
                .to_dtype(DType::F16)
                .unwrap();
            cache.layers[0] = Some((k, v));
            cache.position = n_tokens;
        }
        cache
    }

    #[test]
    fn compress_empty_cache_succeeds() {
        let mut cache = InferenceCache::new(2, 128, 8);
        assert!(cache.compress().is_ok());
        assert!(cache.is_compressed());
    }

    #[test]
    fn compress_noop_when_already_compressed() {
        let mut cache = InferenceCache::new(2, 128, 8);
        cache.compress().unwrap();
        assert!(cache.compress().is_ok());
        assert!(cache.is_compressed());
    }

    #[test]
    fn compress_frees_full_precision_tensors() {
        let mut cache = cache_with_data(2, 8, 128, 16);
        assert!(cache.layers[0].is_some());

        cache.compress().unwrap();

        assert!(cache.is_compressed());
        assert!(cache.layers[0].is_none());
        assert!(cache.layers[1].is_none());
        assert!(cache.compressed[0].is_some());
        assert!(cache.compressed[1].is_none()); // was unpopulated
    }
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test -p harmony-inference --features kv-compress -- kv_compress_cache_tests`
Expected: FAIL — `compress()` doesn't exist

- [ ] **Step 3: Implement compress()**

Add to the `#[cfg(feature = "kv-compress")] impl InferenceCache` block:

```rust
    /// Compress all populated layers to 3-bit quantized format.
    /// Frees full-precision tensors to reclaim memory.
    /// No-op if already compressed.
    /// Atomic: on error, cache remains in uncompressed state.
    pub fn compress(&mut self) -> Result<(), InferenceError> {
        if self.is_compressed {
            return Ok(());
        }

        // Phase 1: compress all populated layers into a temporary vec.
        // If any layer fails, self is untouched.
        let mut new_compressed = Vec::with_capacity(self.num_layers);
        for layer in &self.layers {
            match layer {
                Some((k, v)) => {
                    let (k_vecs, seq_len) = kv_compress::compress_tensor(k)?;
                    let (v_vecs, _) = kv_compress::compress_tensor(v)?;
                    new_compressed.push(Some(kv_compress::CompressedKvLayer {
                        k: k_vecs,
                        v: v_vecs,
                        seq_len,
                    }));
                }
                None => new_compressed.push(None),
            }
        }

        // Phase 2: commit (cannot fail — just moving data).
        self.compressed = new_compressed;
        for layer in &mut self.layers {
            *layer = None;
        }
        self.is_compressed = true;
        Ok(())
    }
```

- [ ] **Step 4: Run tests**

Run: `cargo test -p harmony-inference --features kv-compress -- kv_compress_cache_tests`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add crates/harmony-inference/src/lib.rs
git commit -m "feat(inference): implement compress() with atomic two-phase commit

Compresses populated KV layers to 3-bit quantized vecs, frees f16 tensors.
Phase 1 builds temp vec (rollback on error), Phase 2 commits."
```

---

### Task 6: decompress() Implementation

**Files:**
- Modify: `crates/harmony-inference/src/lib.rs`

- [ ] **Step 1: Write failing tests**

Add to `kv_compress_cache_tests`:

```rust
    #[test]
    fn decompress_noop_when_uncompressed() {
        let mut cache = cache_with_data(2, 8, 128, 16);
        assert!(cache.decompress().is_ok());
        assert!(!cache.is_compressed());
        assert!(cache.layers[0].is_some());
    }

    #[test]
    fn compress_decompress_roundtrip() {
        let mut cache = cache_with_data(2, 8, 128, 16);
        let orig_k = cache.layers[0].as_ref().unwrap().0
            .to_dtype(DType::F32).unwrap()
            .flatten_all().unwrap()
            .to_vec1::<f32>().unwrap();

        cache.compress().unwrap();
        cache.decompress().unwrap();
        assert!(!cache.is_compressed());

        let (k, v) = cache.layers[0].as_ref().expect("layer 0 should be restored");
        assert_eq!(k.dims4().unwrap(), (1, 8, 16, 128));
        assert_eq!(v.dims4().unwrap(), (1, 8, 16, 128));
        assert_eq!(k.dtype(), DType::F16);

        let restored_k = k.to_dtype(DType::F32).unwrap()
            .flatten_all().unwrap()
            .to_vec1::<f32>().unwrap();
        assert_eq!(orig_k.len(), restored_k.len());
        let max_err: f32 = orig_k.iter().zip(restored_k.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        // 3-bit on [0,1]: step=1/7≈0.143, max error < half step + f16 rounding
        assert!(max_err < 0.15, "max reconstruction error {max_err} too large");
    }

    #[test]
    fn is_compressed_state_tracking() {
        let mut cache = cache_with_data(2, 8, 128, 4);
        assert!(!cache.is_compressed());
        cache.compress().unwrap();
        assert!(cache.is_compressed());
        cache.decompress().unwrap();
        assert!(!cache.is_compressed());
        cache.compress().unwrap();
        assert!(cache.is_compressed());
    }
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test -p harmony-inference --features kv-compress -- kv_compress_cache_tests`
Expected: FAIL — `decompress()` doesn't exist

- [ ] **Step 3: Implement decompress()**

Add to the `#[cfg(feature = "kv-compress")] impl InferenceCache` block:

```rust
    /// Decompress all layers back to full-precision f16 tensors.
    /// No-op if not compressed.
    /// Atomic: on error, cache remains in compressed state.
    pub fn decompress(&mut self) -> Result<(), InferenceError> {
        if !self.is_compressed {
            return Ok(());
        }

        let device = candle_core::Device::Cpu;

        // Phase 1: decompress into temporary vec.
        let mut new_layers: Vec<Option<(Tensor, Tensor)>> = Vec::with_capacity(self.num_layers);
        for comp in &self.compressed {
            match comp {
                Some(c) => {
                    let k = kv_compress::decompress_tensor(
                        &c.k, self.num_kv_heads, c.seq_len, self.head_dim, &device,
                    )?;
                    let v = kv_compress::decompress_tensor(
                        &c.v, self.num_kv_heads, c.seq_len, self.head_dim, &device,
                    )?;
                    new_layers.push(Some((k, v)));
                }
                None => new_layers.push(None),
            }
        }

        // Phase 2: commit.
        self.layers = new_layers;
        for comp in &mut self.compressed {
            *comp = None;
        }
        self.is_compressed = false;
        Ok(())
    }
```

- [ ] **Step 4: Run tests**

Run: `cargo test -p harmony-inference --features kv-compress -- kv_compress_cache_tests`
Expected: PASS (6 tests)

- [ ] **Step 5: Commit**

```bash
git add crates/harmony-inference/src/lib.rs
git commit -m "feat(inference): implement decompress() with atomic two-phase commit

Reconstructs f16 tensors from 3-bit quantized vecs. Same two-phase
pattern: build temp vec, commit on success, rollback on error."
```

---

### Task 7: memory_bytes() Implementation

**Files:**
- Modify: `crates/harmony-inference/src/lib.rs`

- [ ] **Step 1: Write failing tests**

Add to `kv_compress_cache_tests`:

```rust
    #[test]
    fn memory_bytes_empty_cache() {
        let cache = InferenceCache::new(2, 128, 8);
        assert_eq!(cache.memory_bytes(), 0);
    }

    #[test]
    fn memory_bytes_uncompressed() {
        let cache = cache_with_data(2, 8, 128, 16);
        let bytes = cache.memory_bytes();
        // Layer 0: K + V = 2 * 1 * 8 * 16 * 128 * 2 bytes = 65536
        assert_eq!(bytes, 65536);
    }

    #[test]
    fn compress_reduces_memory() {
        let mut cache = cache_with_data(2, 8, 128, 16);
        let before = cache.memory_bytes();
        cache.compress().unwrap();
        let after = cache.memory_bytes();
        assert!(after < before, "compressed {after} should be < uncompressed {before}");
        let ratio = before as f64 / after as f64;
        // 3-bit uniform: 56 bytes/vec vs 256 bytes/vec ≈ 4.6x
        assert!(ratio > 3.0, "compression ratio {ratio:.1}x too low");
    }

    #[test]
    fn memory_bytes_restored_matches_original() {
        let mut cache = cache_with_data(2, 8, 128, 16);
        let original = cache.memory_bytes();
        cache.compress().unwrap();
        cache.decompress().unwrap();
        assert_eq!(cache.memory_bytes(), original);
    }
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test -p harmony-inference --features kv-compress -- kv_compress_cache_tests::memory`
Expected: FAIL — `memory_bytes()` doesn't exist

- [ ] **Step 3: Implement memory_bytes()**

Add to the `#[cfg(feature = "kv-compress")] impl InferenceCache` block:

```rust
    /// Approximate memory usage in bytes (accounts for compression state).
    /// Unpopulated layers contribute 0 bytes.
    pub fn memory_bytes(&self) -> usize {
        let mut total = 0;
        for (i, layer) in self.layers.iter().enumerate() {
            if let Some((k, v)) = layer {
                total += (k.elem_count() + v.elem_count()) * k.dtype().size_in_bytes();
            }
            if let Some(comp) = &self.compressed[i] {
                total += comp.byte_size();
            }
        }
        total
    }
```

- [ ] **Step 4: Run tests**

Run: `cargo test -p harmony-inference --features kv-compress -- kv_compress_cache_tests`
Expected: PASS (10 tests)

- [ ] **Step 5: Commit**

```bash
git add crates/harmony-inference/src/lib.rs
git commit -m "feat(inference): implement memory_bytes() for compression-aware reporting

Returns total bytes across layers, counting f16 tensor bytes when
uncompressed and quantized vec bytes when compressed."
```

---

### Task 8: forward() Safety Guard

**Files:**
- Modify: `crates/harmony-inference/src/engine.rs`

- [ ] **Step 1: Write failing test**

Add to the test module in `crates/harmony-inference/src/engine.rs`:

```rust
    #[test]
    #[cfg(feature = "kv-compress")]
    fn forward_while_compressed_errors() {
        let engine = QwenEngine::new(Device::Cpu);
        let mut cache = InferenceCache::new(28, 128, 8);
        cache.compress().unwrap();
        assert!(cache.is_compressed());

        let result = engine.forward(&[1, 2, 3], &mut cache);
        assert!(
            matches!(result, Err(InferenceError::CacheCompressed)),
            "expected CacheCompressed, got {result:?}"
        );
        assert!(cache.is_compressed());
    }
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p harmony-inference --features kv-compress -- tests::forward_while_compressed`
Expected: FAIL — returns `ModelNotLoaded` instead of `CacheCompressed`

- [ ] **Step 3: Add safety guard**

In `crates/harmony-inference/src/engine.rs`, in the `forward()` method, add after the empty-tokens check (after line 100) and before the `model.as_ref()` line:

```rust
        #[cfg(feature = "kv-compress")]
        if cache.is_compressed() {
            return Err(InferenceError::CacheCompressed);
        }
```

- [ ] **Step 4: Run tests**

Run: `cargo test -p harmony-inference --features kv-compress -- tests::forward_while_compressed`
Expected: PASS

Run: `cargo test -p harmony-inference`
Expected: PASS (guard not compiled without feature)

- [ ] **Step 5: Commit**

```bash
git add crates/harmony-inference/src/engine.rs
git commit -m "feat(inference): add forward() safety guard for compressed cache

Returns CacheCompressed if cache.is_compressed(), preventing the model
from seeing None layers after compression."
```

---

### Task 9: Atomicity and Edge Case Tests

**Files:**
- Modify: `crates/harmony-inference/src/lib.rs` (tests only)

- [ ] **Step 1: Add atomicity test**

Add to `kv_compress_cache_tests`:

```rust
    #[test]
    fn compress_atomic_on_error() {
        let mut cache = InferenceCache::new(2, 128, 8);

        // Layer 0: correct shape [1, 8, 4, 128]
        let shape_ok = (1, 8, 4, 128);
        let k0 = Tensor::rand(0f32, 1f32, shape_ok, &Device::Cpu)
            .unwrap().to_dtype(DType::F16).unwrap();
        let v0 = Tensor::rand(0f32, 1f32, shape_ok, &Device::Cpu)
            .unwrap().to_dtype(DType::F16).unwrap();
        cache.layers[0] = Some((k0, v0));

        // Layer 1: 3D tensor (wrong rank) — will fail dims4()
        let k1 = Tensor::rand(0f32, 1f32, (8, 4, 128), &Device::Cpu)
            .unwrap().to_dtype(DType::F16).unwrap();
        let v1 = Tensor::rand(0f32, 1f32, (8, 4, 128), &Device::Cpu)
            .unwrap().to_dtype(DType::F16).unwrap();
        cache.layers[1] = Some((k1, v1));

        let result = cache.compress();
        assert!(result.is_err(), "should fail on malformed tensor");

        // Atomic: cache must remain fully uncompressed
        assert!(!cache.is_compressed());
        assert!(cache.layers[0].is_some());
        assert!(cache.layers[1].is_some());
        assert!(cache.compressed.iter().all(|c| c.is_none()));
    }

    #[test]
    fn double_roundtrip_preserves_shape() {
        let mut cache = cache_with_data(2, 8, 128, 8);

        cache.compress().unwrap();
        cache.decompress().unwrap();
        cache.compress().unwrap();
        cache.decompress().unwrap();

        assert!(!cache.is_compressed());
        let (k, v) = cache.layers[0].as_ref().unwrap();
        assert_eq!(k.dims4().unwrap(), (1, 8, 8, 128));
        assert_eq!(v.dims4().unwrap(), (1, 8, 8, 128));
        assert_eq!(k.dtype(), DType::F16);
    }

    #[test]
    fn compress_with_partial_layers() {
        let mut cache = InferenceCache::new(4, 128, 8);
        let shape = (1, 8, 4, 128);
        let k = Tensor::rand(0f32, 1f32, shape, &Device::Cpu)
            .unwrap().to_dtype(DType::F16).unwrap();
        let v = Tensor::rand(0f32, 1f32, shape, &Device::Cpu)
            .unwrap().to_dtype(DType::F16).unwrap();
        cache.layers[0] = Some((k.clone(), v.clone()));
        cache.layers[2] = Some((k, v));

        cache.compress().unwrap();
        assert!(cache.compressed[0].is_some());
        assert!(cache.compressed[1].is_none());
        assert!(cache.compressed[2].is_some());
        assert!(cache.compressed[3].is_none());

        cache.decompress().unwrap();
        assert!(cache.layers[0].is_some());
        assert!(cache.layers[1].is_none());
        assert!(cache.layers[2].is_some());
        assert!(cache.layers[3].is_none());
    }
```

- [ ] **Step 2: Run all tests**

Run: `cargo test -p harmony-inference --features kv-compress`
Expected: ALL PASS

- [ ] **Step 3: Run tests without feature**

Run: `cargo test -p harmony-inference`
Expected: ALL PASS

- [ ] **Step 4: Run clippy**

Run: `cargo clippy -p harmony-inference --features kv-compress -- -D warnings`
Expected: No warnings

Run: `cargo clippy -p harmony-inference -- -D warnings`
Expected: No warnings

- [ ] **Step 5: Commit**

```bash
git add crates/harmony-inference/src/lib.rs
git commit -m "test(inference): add atomicity and edge case tests for KV compression

Tests: compress_atomic_on_error (malformed tensor triggers rollback),
double_roundtrip_preserves_shape, compress_with_partial_layers."
```

---

## Test Matrix Summary

| Test | What | Task |
|------|------|------|
| `pack_unpack_roundtrip` | 3-bit packing is lossless | 3 |
| `pack_unpack_all_sevens` | Boundary value packing | 3 |
| `pack_unpack_all_zeros` | Zero packing + correct size | 3 |
| `quantize_dequantize_roundtrip` | Per-vector error within tolerance | 3 |
| `quantize_constant_vector` | Constant vectors handled (scale=0) | 3 |
| `quantize_empty_vector` | Empty input handled | 3 |
| `quantized_vec_byte_size` | 56 bytes for dim-128 | 3 |
| `compressed_kv_layer_byte_size` | Layer byte accounting | 3 |
| `compress_tensor_roundtrip` | Tensor ↔ quantized vec bridge | 3 |
| `compress_empty_cache_succeeds` | Empty cache compress OK | 5 |
| `compress_noop_when_already_compressed` | Double-compress is no-op | 5 |
| `compress_frees_full_precision_tensors` | Layers freed, compressed populated | 5 |
| `decompress_noop_when_uncompressed` | Decompress on uncompressed no-op | 6 |
| `compress_decompress_roundtrip` | Approximate equality after roundtrip | 6 |
| `is_compressed_state_tracking` | Flag toggles through cycle | 6 |
| `memory_bytes_empty_cache` | 0 bytes for empty cache | 7 |
| `memory_bytes_uncompressed` | Correct byte count for f16 | 7 |
| `compress_reduces_memory` | Compressed < uncompressed, >3x ratio | 7 |
| `memory_bytes_restored_matches_original` | Restored = original bytes | 7 |
| `forward_while_compressed_errors` | CacheCompressed error | 8 |
| `compress_atomic_on_error` | Malformed tensor triggers rollback | 9 |
| `double_roundtrip_preserves_shape` | Shape/dtype preserved through 2 cycles | 9 |
| `compress_with_partial_layers` | Sparse layer population correct | 9 |
