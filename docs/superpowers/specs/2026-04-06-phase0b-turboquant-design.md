# Phase 0b: TurboQuant KV Cache Compression — Design Spec

**Epic:** harmony-ct87 / harmony-iwsh / harmony-3570
**Phase:** 0b (parallel with 0a, 0c, 0d)
**Status:** Design approved
**Date:** 2026-04-06
**Supersedes:** `2026-03-27-turboquant-kv-compression-design.md` (which assumed external `tq-kv` crate)

## Goal

Replace the 3-bit uniform quantization internals of `kv_compress.rs` with a
three-stage PolarQuant + QJL pipeline. Achieves 42 bytes/vector at head_dim=80
(vs 56 bytes current) with better reconstruction quality and no per-vector
metadata overhead. External API shape unchanged; feature flag stays `kv-compress`.

## Architecture

### Three-Stage Pipeline

```
Compress:
  input vector [head_dim] (f32)
    → Stage 1: Rotate (Q @ vector, precomputed random orthogonal matrix)
    → Stage 2: PolarQuant (extract f16 radius, Lloyd-Max quantize unit coords)
    → Stage 3: QJL (sign-bit projection of quantization residual)
  output: TurboQuantVec { radius, angles_packed, signs_packed }

Decompress:
  TurboQuantVec
    → Stage 3: Reconstruct residual correction from sign bits
    → Stage 2: Dequantize unit coords from codebook, scale by radius
    → Stage 1: Inverse rotate (Q^T @ reconstructed)
  output: vector [head_dim] (f32)
```

### Stage 1: Random Orthogonal Preconditioning

Smooths channel-wise outliers so all dimensions have similar distributions,
making the global codebook effective across all channels.

- Generate a random `[head_dim, head_dim]` matrix from a deterministic seed
- Orthogonalize via modified Gram-Schmidt (numerically stable, hand-rolled)
- Store both Q and Q^T (transpose = inverse for orthogonal matrices)
- Compress: `rotated = Q @ vector`
- Decompress: `original = Q^T @ reconstructed`

The seed is fixed per `TurboQuantConfig` so compress/decompress are always paired.
The matrix is generated in f64 for numerical precision, stored as f32 for use.

### Stage 2: PolarQuant (Cartesian Lloyd-Max)

Decomposes each rotated vector into magnitude and direction, then quantizes the
direction using a global codebook. Eliminates per-vector min/scale metadata.

1. Compute `radius = L2_norm(rotated_vector)` — stored as f16 (2 bytes)
2. Normalize: `unit_vector = rotated_vector / radius`
3. Quantize each coordinate of the unit vector to 3 bits using Lloyd-Max
   codebook for Gaussian distribution (rotation makes coordinates ~i.i.d. Gaussian)
4. Pack into `ceil(head_dim * 3 / 8)` bytes

**Why Cartesian instead of hyperspherical angles:** The rotation already
decorrelates dimensions into approximately i.i.d. Gaussian. Quantizing Cartesian
coordinates directly with a global Gaussian codebook gives the same benefit (no
per-vector metadata) without expensive trig operations on the hot path. The key
insight from PolarQuant is the *global codebook* — not the coordinate system.

**Lloyd-Max codebook:** Pre-computed for the standard normal distribution N(0,1).
8 levels (3 bits). Two arrays:
- `boundaries[7]` — decision thresholds between levels
- `centroids[8]` — reconstruction values (conditional means)

These are constants derived from the Gaussian CDF. Can be computed at init or
hardcoded. The codebook is tiny (~60 bytes) and shared across all vectors.

**Radius special case:** If radius is zero (zero vector), skip quantization —
store radius=0 and all-zero packed data. Decompression returns a zero vector.

### Stage 3: QJL Error Correction

Captures quantization error as 1-bit sign projections for unbiased inner product
estimation. Attention scores computed on decompressed vectors are statistically
correct.

1. Compute residual: `residual = rotated_vector - reconstructed_vector`
   (where `reconstructed = radius * dequantized_unit_vector`)
2. Project: `projected = JL_matrix @ residual`
3. Store sign bits: `signs[i] = projected[i] > 0 ? 1 : 0`
4. Pack into `ceil(head_dim / 8)` bytes

**JL matrix:** Random +-1 entries (Rademacher distribution), deterministic seed.
Stored as packed bits: `head_dim * head_dim / 8` bytes = 800 bytes for dim=80.

**Decompression correction:**
```
scale = radius * EXPECTED_RELATIVE_ERROR
corrected = reconstructed + scale * JL_matrix^T @ sign_vector_scaled
```
where `EXPECTED_RELATIVE_ERROR` is a global constant derived from the expected
quantization error of 3-bit Lloyd-Max on Gaussian data. The scale factor is
derived from the stored radius, adding zero extra bytes per vector.

**Sign vector scaling:** The packed sign bits are decoded as +-1 values
(not 0/1) for the matrix multiply, so the correction is symmetric.

## Byte Budget (head_dim=80)

| Component | Bytes | Description |
|-----------|-------|-------------|
| Radius | 2 | f16 magnitude |
| Angles | 30 | ceil(80 * 3 / 8) packed 3-bit Lloyd-Max indices |
| QJL signs | 10 | ceil(80 / 8) packed 1-bit sign projections |
| **Total** | **42** | **per vector** |

Comparison at head_dim=80:

| Format | Bytes/vector | Compression vs f16 |
|--------|-------------|-------------------|
| f16 | 160 | 1.0x |
| Current 3-bit uniform | 56 | 2.9x |
| TurboQuant | 42 | 3.8x |

Memory at 32K context, 24 layers, GQA-8:

| Format | Memory |
|--------|--------|
| f16 | 1.9 GB |
| Current 3-bit | 420 MB |
| TurboQuant | 315 MB |

## State Management

### TurboQuantConfig

```rust
pub struct TurboQuantConfig {
    pub head_dim: usize,
    pub seed: u64,        // deterministic seed for reproducible matrices
}
```

### TurboQuantState

Initialized once at model load. Immutable after creation. Multiple caches
(different conversations) share the same state via `&TurboQuantState`.

```rust
pub(crate) struct TurboQuantState {
    config: TurboQuantConfig,
    ortho_matrix: Vec<f32>,      // [head_dim * head_dim], row-major Q
    ortho_transpose: Vec<f32>,   // [head_dim * head_dim], row-major Q^T
    codebook: LloydMaxCodebook,  // boundaries[7] + centroids[8]
    jl_matrix: Vec<u8>,          // packed +-1 bits, [head_dim * head_dim / 8]
    expected_relative_error: f32, // precomputed from codebook
}
```

### Lifetime in the System

```
Model load:
  engine.load_gguf(...)
  let tq_state = TurboQuantState::new(&TurboQuantConfig { head_dim: 80, seed: 42 })?;

Generate loop:
  cache.decompress(&tq_state)?;     // TurboQuantState passed by reference
  let logits = engine.forward(...)?;
  cache.compress(&tq_state)?;
```

The engine or caller owns `TurboQuantState`. Caches borrow it. This matches
the existing pattern where the engine is stateless and caches are per-conversation.

## File Structure

Convert `kv_compress.rs` to a module directory:

| File | Responsibility | Approximate size |
|------|----------------|-----------------|
| `kv_compress/mod.rs` | Public API: `TurboQuantState`, `TurboQuantConfig`, `TurboQuantVec`, `CompressedKvLayer`, `compress_tensor()`, `decompress_tensor()` | ~200 lines |
| `kv_compress/orthogonal.rs` | `generate_orthogonal_matrix(dim, seed)` — seeded RNG + modified Gram-Schmidt | ~100 lines |
| `kv_compress/lloyd_max.rs` | `LloydMaxCodebook` — Gaussian codebook, `quantize()`, `dequantize()` | ~80 lines |
| `kv_compress/packing.rs` | `pack_3bit()`, `unpack_3bit()`, `pack_1bit()`, `unpack_1bit()` — extracted and extended from current code | ~80 lines |
| `kv_compress/qjl.rs` | `JlMatrix` — Rademacher matrix generation, `encode_signs()`, `decode_correction()` | ~100 lines |

Total: ~560 lines across 5 files (vs 332 lines currently in one file).

## API Changes

### Internal API (within kv_compress module)

```rust
impl TurboQuantState {
    pub fn new(config: &TurboQuantConfig) -> Result<Self, InferenceError>;

    pub fn compress_tensor(
        &self,
        tensor: &Tensor,
    ) -> Result<(Vec<TurboQuantVec>, usize), InferenceError>;

    pub fn decompress_tensor(
        &self,
        vecs: &[TurboQuantVec],
        num_kv_heads: usize,
        seq_len: usize,
        head_dim: usize,
        device: &Device,
    ) -> Result<Tensor, InferenceError>;
}
```

### TurboQuantVec (replaces QuantizedVec)

```rust
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TurboQuantVec {
    pub radius: u16,       // f16 bits (magnitude)
    pub angles: Vec<u8>,   // 3-bit packed Lloyd-Max indices
    pub signs: Vec<u8>,    // 1-bit packed QJL sign projections
}
```

### CompressedKvLayer (unchanged shape)

```rust
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CompressedKvLayer {
    pub k: Vec<TurboQuantVec>,
    pub v: Vec<TurboQuantVec>,
    pub seq_len: usize,
}
```

### InferenceCache changes (lib.rs)

The `compress()` and `decompress()` methods gain a `&TurboQuantState` parameter:

```rust
pub fn compress(&mut self, tq: &TurboQuantState) -> Result<(), InferenceError>;
pub fn decompress(&mut self, tq: &TurboQuantState) -> Result<(), InferenceError>;
```

The `InferenceCache` struct itself does not change — it still holds
`Vec<Option<CompressedKvLayer>>` and `is_compressed: bool`.

## Serialization

`TurboQuantVec` uses `serde` + `postcard` (same as current `QuantizedVec`).
The serialized format is NOT backward-compatible with the old format. This is
expected — there is no production data in the old format that needs migration.

`serialize_compressed()` and `deserialize_compressed()` on `InferenceCache`
continue to work unchanged (they serialize `CompressedKvLayer` which now
contains `TurboQuantVec` instead of `QuantizedVec`).

## Error Handling

No new error variants needed. The existing `InferenceError::CompressionFailed(String)`
covers all failure modes:
- Matrix generation failure (degenerate seed)
- Tensor shape mismatch
- Zero-length vectors

## Scope Boundary

**In scope (0b):**
- `TurboQuantConfig` and `TurboQuantState` with `new()`
- Random orthogonal matrix generation (modified Gram-Schmidt, seeded)
- Lloyd-Max codebook for Gaussian distribution (3-bit, 8 levels)
- PolarQuant compress/decompress (radius + Cartesian Lloyd-Max)
- QJL sign-bit encode/decode with radius-derived correction scale
- 3-bit and 1-bit packing/unpacking
- `TurboQuantVec` replacing `QuantizedVec`
- `compress_tensor()` and `decompress_tensor()` as methods on state
- Update `InferenceCache::compress()`/`decompress()` signatures in lib.rs
- Update all existing kv-compress tests
- New tests for each submodule
- Serde serialization roundtrip

**Out of scope (deferred):**
- Fused attention on compressed keys
- SIMD/NEON optimized matrix multiply
- Selective layer compression
- KV cache serialization for CAS distribution
- Backward compatibility with old serialized format

## Testing Strategy

### Per-submodule tests

| Module | Tests |
|--------|-------|
| `orthogonal` | Matrix is orthogonal (Q @ Q^T ≈ I), deterministic from seed, different seeds produce different matrices |
| `lloyd_max` | Codebook centroids are ordered, quantize/dequantize roundtrip error bounded, boundary conditions |
| `packing` | 3-bit pack/unpack roundtrip (carried from current), 1-bit pack/unpack roundtrip (new) |
| `qjl` | JL matrix is deterministic from seed, sign encode/decode roundtrip, correction reduces error |
| `mod` | Full compress/decompress roundtrip (error bounded), byte size matches budget, zero vector handling |

### Integration tests (lib.rs, existing tests updated)

All existing `kv_compress_cache_tests` updated to pass `&TurboQuantState`:
- `compress_decompress_roundtrip`
- `compress_reduces_memory`
- `compress_noop_when_compressed`
- `decompress_noop_when_uncompressed`
- `compress_atomic_on_error`
- `serialize_deserialize_roundtrip`
- etc.
