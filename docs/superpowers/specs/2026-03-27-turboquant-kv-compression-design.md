# TurboQuant KV Cache Compression for Edge Inference

**Date:** 2026-03-27
**Status:** Draft
**Bead:** harmony-p3zv
**Depends on:** harmony-6b0 (externalized KV cache), harmony-471e (tq-kv crate evaluation)

## Problem

On a 512MB edge device running Qwen3-0.6B, the KV cache is the primary memory bottleneck during autoregressive generation. Qwen3-0.6B uses f16 tensors in the KV cache. At 2048 context tokens, the KV cache consumes ~115MB (28 layers × 8 KV heads × 2048 tokens × 128 head_dim × 2 tensors × 2 bytes/f16). TurboQuant compresses this to ~3 bits per channel (~22MB), a ~5.3x reduction, enabling proportionally longer context windows within the same memory budget.

## Constraints

- **Feature-gated.** All TurboQuant code behind `#[cfg(feature = "turbo-quant")]`. Without the feature, the crate compiles and works identically — compress/decompress methods don't exist, and the caller never calls them. The architecture diagram shows the feature-enabled loop; without the feature, the loop is just `forward → sample → repeat`.
- **Phase 1: compress/decompress per forward pass.** No attention code changes. Fused attention on compressed keys is a follow-up optimization.
- **Lossy but unbiased.** TurboQuant provides provably unbiased inner-product estimation. Output quality degrades slightly but attention scores remain statistically correct.
- **Atomic compress/decompress.** On error, the cache rolls back to its pre-operation state. No partial states.
- **`tq-kv` crate.** candle feature for direct Tensor integration. MIT OR Apache-2.0.

## Architecture

```
Generate loop (caller, with turbo-quant feature enabled):
  loop {
      cache.decompress()?;              // restore to full precision (no-op if uncompressed)
      let logits = engine.forward(&tokens, &mut cache)?;  // appends new K/V to cache
      cache.compress()?;                // compress to 3-bit
      let token = sample(logits);
      // cache is now compressed, using ~5x less memory between tokens
  }
```

Compression happens entirely at the `InferenceCache` level — the model's forward pass sees full-precision tensors every time. The caller controls when compression happens.

**Safety guard:** `forward()` checks `is_compressed` and returns `InferenceError::CacheCompressed` if the cache hasn't been decompressed. This prevents silent data loss from a missed `decompress()` call.

## Integration Point

`InferenceCache` (in `harmony-inference/src/lib.rs`) already has:

```rust
pub struct InferenceCache {
    pub layers: Vec<Option<(Tensor, Tensor)>>,  // per-layer (K, V) pairs
    pub position: usize,
    pub num_layers: usize,
    pub head_dim: usize,       // 128 for Qwen3-0.6B
    pub num_kv_heads: usize,   // 8 for Qwen3-0.6B
}
```

Tensor shape per layer: `[1, num_kv_heads, seq_len, head_dim]` = `[1, 8, seq_len, 128]`.
Tensor dtype: f16 (DType::F16 in `qwen3_ext.rs`).

## New Types

```rust
/// Compressed representation of one layer's K and V tensors.
/// Opaque — only tq-kv knows the internal format.
/// Uses the same `tq_kv::CompressedTensor` type for both K and V
/// (tq-kv's compression is data-oblivious and uses the same algorithm for both).
#[cfg(feature = "turbo-quant")]
pub struct CompressedKvLayer {
    pub k: tq_kv::CompressedTensor,
    pub v: tq_kv::CompressedTensor,
}
```

**Note:** The exact type name (`CompressedTensor`, `CompressedKeys`, etc.) depends on what `tq-kv` exports. The implementer should check `tq-kv`'s public API with the `candle` feature enabled and use the correct type. If separate types exist for K and V, use them.

## New Fields on InferenceCache

```rust
pub struct InferenceCache {
    // ... existing fields ...

    /// Per-layer compressed K/V state. Populated by compress(), consumed by decompress().
    #[cfg(feature = "turbo-quant")]
    compressed: Vec<Option<CompressedKvLayer>>,

    /// Whether the cache is currently in compressed form.
    /// When true, `layers` entries are None (memory freed) and `compressed` holds the data.
    /// `forward()` checks this flag and returns an error if true.
    #[cfg(feature = "turbo-quant")]
    is_compressed: bool,
}
```

When compressed: `layers[i]` is set to `None` (freeing the full-precision tensors) and `compressed[i]` holds the 3-bit representation. When decompressed: `layers[i]` is restored and `compressed[i]` is set to `None`.

Layers where both `layers[i]` and `compressed[i]` are `None` are unpopulated (not yet seen by forward pass) and contribute 0 bytes to `memory_bytes()`.

## New Methods

```rust
#[cfg(feature = "turbo-quant")]
impl InferenceCache {
    /// Compress all populated layers to 3-bit TurboQuant format.
    /// Frees full-precision tensors to reclaim memory.
    /// No-op if already compressed.
    /// Atomic: on error, cache remains in uncompressed state.
    pub fn compress(&mut self) -> Result<(), InferenceError>;

    /// Decompress all layers back to full-precision tensors.
    /// No-op if not compressed.
    /// Atomic: on error, cache remains in compressed state.
    pub fn decompress(&mut self) -> Result<(), InferenceError>;

    /// Whether the cache is currently compressed.
    pub fn is_compressed(&self) -> bool;

    /// Approximate memory usage in bytes (accounts for compression state).
    /// Unpopulated layers contribute 0 bytes.
    pub fn memory_bytes(&self) -> usize;
}
```

## forward() Safety Guard

In the `forward()` method (both `QwenEngine` and the trait default), add at the top:

```rust
#[cfg(feature = "turbo-quant")]
if cache.is_compressed() {
    return Err(InferenceError::CacheCompressed);
}
```

This prevents the model from seeing `None` in `layers[i]` and silently discarding cached KV state. The caller must call `decompress()` first.

### compress() implementation (atomic)

```rust
pub fn compress(&mut self) -> Result<(), InferenceError> {
    if self.is_compressed { return Ok(()); }  // no-op

    // Phase 1: compress all layers into a temporary vec (don't modify self yet)
    let mut new_compressed = Vec::with_capacity(self.num_layers);
    for layer in &self.layers {
        match layer {
            Some((k, v)) => {
                let ck = tq_kv::compress(&k).map_err(|e| InferenceError::CompressionFailed(e.to_string()))?;
                let cv = tq_kv::compress(&v).map_err(|e| InferenceError::CompressionFailed(e.to_string()))?;
                new_compressed.push(Some(CompressedKvLayer { k: ck, v: cv }));
            }
            None => new_compressed.push(None),
        }
    }

    // Phase 2: commit (cannot fail — just moving data)
    self.compressed = new_compressed;
    for layer in &mut self.layers {
        *layer = None;  // free full-precision tensors
    }
    self.is_compressed = true;
    Ok(())
}
```

If Phase 1 fails at any layer, `self` is untouched — the cache remains uncompressed with all original tensors intact.

### decompress() implementation (atomic)

```rust
pub fn decompress(&mut self) -> Result<(), InferenceError> {
    if !self.is_compressed { return Ok(()); }  // no-op

    // Phase 1: decompress into temporary vec
    let mut new_layers = Vec::with_capacity(self.num_layers);
    for comp in &self.compressed {
        match comp {
            Some(c) => {
                let k = tq_kv::decompress(&c.k).map_err(|e| InferenceError::CompressionFailed(e.to_string()))?;
                let v = tq_kv::decompress(&c.v).map_err(|e| InferenceError::CompressionFailed(e.to_string()))?;
                new_layers.push(Some((k, v)));
            }
            None => new_layers.push(None),
        }
    }

    // Phase 2: commit
    self.layers = new_layers;
    for comp in &mut self.compressed {
        *comp = None;  // free compressed data
    }
    self.is_compressed = false;
    Ok(())
}
```

### memory_bytes() implementation

```rust
pub fn memory_bytes(&self) -> usize {
    let mut total = 0;
    for (i, layer) in self.layers.iter().enumerate() {
        if let Some((k, v)) = layer {
            // Full-precision: elem_count * dtype_size
            total += (k.elem_count() + v.elem_count()) * k.dtype().size_in_bytes();
        }
        if let Some(comp) = &self.compressed[i] {
            total += comp.k.byte_size() + comp.v.byte_size();
        }
        // Both None = unpopulated layer, contributes 0 bytes
    }
    total
}
```

## Constructor Changes

`InferenceCache::new()` initializes the new fields:

```rust
#[cfg(feature = "turbo-quant")]
compressed: (0..num_layers).map(|_| None).collect(),
#[cfg(feature = "turbo-quant")]
is_compressed: false,
```

`CompressedKvLayer` and its contained `tq_kv` types implement `Drop` automatically (Rust default). Dropping a compressed cache frees all compressed data without leaking.

## Cargo.toml Changes

```toml
[features]
default = []
turbo-quant = ["dep:tq-kv"]

[dependencies]
tq-kv = { version = "0.1", optional = true, features = ["candle"] }
```

**Note:** `no_std` feature omitted when `candle` is enabled — candle requires std. The `tq-kv` no_std core is relevant for future non-candle integration (e.g., direct buffer manipulation on MIPS without candle).

## Error Handling

New `InferenceError` variants:

```rust
pub enum InferenceError {
    // ... existing variants ...
    #[cfg(feature = "turbo-quant")]
    CompressionFailed(String),
    #[cfg(feature = "turbo-quant")]
    CacheCompressed,  // forward() called while cache is compressed
}
```

## Testing Strategy

All tests require the `turbo-quant` feature: `cargo test -p harmony-inference --features turbo-quant`.

| Test | What |
|------|------|
| `compress_decompress_roundtrip` | Compress then decompress, verify K/V tensors are approximately equal (allow small floating-point error from lossy compression) |
| `compress_reduces_memory` | Verify `memory_bytes()` decreases after compression (~5x reduction for f16) |
| `decompress_noop_when_uncompressed` | Decompress on uncompressed cache returns Ok, no state change |
| `compress_noop_when_compressed` | Compress on already-compressed cache returns Ok, no state change |
| `is_compressed_state_tracking` | Verify flag toggles correctly through compress/decompress cycle |
| `compress_empty_cache` | Compress with no populated layers succeeds (no-op) |
| `forward_while_compressed_errors` | Call forward() on compressed cache → CacheCompressed error. Verify no data loss (cache still compressed, can decompress). |
| `forward_with_compression_cycle` | Full generate loop: forward → compress → decompress → forward. Verify the model produces valid output (not NaN/Inf) |
| `memory_bytes_accuracy` | Verify memory_bytes returns plausible values in both states |
| `compress_atomic_on_error` | If compression fails mid-way (mock failure), verify cache remains fully uncompressed |

## Out of Scope

- Fused attention on compressed keys (`tq_kv::fused_attention_scores`) — follow-up optimization
- PolarQuant-only fallback for MIPS (skip QJL residual) — follow-up for constrained targets
- MIPS DSP ASE optimized rotation matrices — follow-up (harmony-swpv)
- Selective layer compression (compress only some layers) — YAGNI for now
- KV cache serialization for CAS distribution (harmony-hbf0) — separate bead
