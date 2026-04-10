//! Paged KV cache with block-level allocation.
//!
//! Replaces contiguous KV cache allocation with fixed-size pages (blocks)
//! managed via a free-list allocator, inspired by vLLM's PagedAttention.
//!
//! # Motivation
//!
//! The default [`InferenceCache`](crate::InferenceCache) stores one contiguous
//! tensor per layer, grown by `Tensor::cat` on every decoded token. This has
//! two problems on edge devices:
//!
//! 1. Each concatenation allocates a new tensor and copies all previous data.
//! 2. The contiguous allocation can fail even when total free memory suffices
//!    (fragmentation).
//!
//! Paged allocation fixes both: small fixed-size blocks are allocated on
//! demand from a reusable pool, and `gather()` assembles the full K/V
//! sequence only when attention needs it.
//!
//! # Layout
//!
//! Each [`KvBlock`] stores K and V data for one layer, for up to `page_size`
//! tokens. The raw f32 layout is `[num_kv_heads][page_size][head_dim]`,
//! matching the existing `[1, num_kv_heads, seq_len, head_dim]` tensor
//! convention so that `gather()` is a simple concatenation with no transpose.
//!
//! # Usage
//!
//! ```ignore
//! use harmony_inference::paged_kv::{PagedKvConfig, PagedKvCache};
//!
//! let config = PagedKvConfig::new(16, 2048);
//! let mut cache = PagedKvCache::new(config, num_layers, head_dim, num_kv_heads);
//!
//! // During forward pass, for each layer:
//! cache.append(layer, &k_data, &v_data)?;
//! let (k, v) = cache.gather(layer, &device)?;
//! // ... compute attention with (k, v) ...
//!
//! // After all layers, advance position:
//! cache.advance(1)?;
//!
//! // Speculative decode rollback:
//! cache.truncate(cache.position() - rejected_count)?;
//! ```

use crate::error::InferenceError;
use candle_core::{Device, Tensor};

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

/// Configuration for paged KV cache.
#[derive(Debug, Clone)]
pub struct PagedKvConfig {
    /// Number of token positions per page (block).
    pub page_size: usize,
    /// Maximum number of pages in the pool (caps total memory).
    pub max_pages: usize,
}

impl PagedKvConfig {
    /// Create a new config.
    ///
    /// # Panics
    ///
    /// Panics if `page_size < 1` or `max_pages < 1`.
    pub fn new(page_size: usize, max_pages: usize) -> Self {
        assert!(page_size >= 1, "page_size must be >= 1");
        assert!(max_pages >= 1, "max_pages must be >= 1");
        Self { page_size, max_pages }
    }
}

impl Default for PagedKvConfig {
    fn default() -> Self {
        Self {
            page_size: 16,
            max_pages: 2048,
        }
    }
}

// ---------------------------------------------------------------------------
// Block allocator
// ---------------------------------------------------------------------------

/// Free-list allocator for reusable block IDs.
///
/// Manages a pool of `total` block indices `[0, total)`. Allocation pops
/// from the free list (LIFO for temporal locality); freeing pushes back.
///
/// An allocation bitmap prevents double-free corruption — `free()` panics
/// if the block is not currently allocated.
#[derive(Debug)]
pub(crate) struct BlockAllocator {
    free_list: Vec<usize>,
    in_use: Vec<bool>,
    total: usize,
}

impl BlockAllocator {
    /// Create an allocator with `total` blocks, all initially free.
    pub(crate) fn new(total: usize) -> Self {
        // Reverse so pop() yields 0, 1, 2, ... in order.
        Self {
            free_list: (0..total).rev().collect(),
            in_use: vec![false; total],
            total,
        }
    }

    /// Allocate one block. Returns `None` if the pool is exhausted.
    pub(crate) fn alloc(&mut self) -> Option<usize> {
        let id = self.free_list.pop()?;
        self.in_use[id] = true;
        Some(id)
    }

    /// Return a block to the pool.
    ///
    /// # Panics
    ///
    /// Panics if `id >= total` or the block is not currently allocated
    /// (double-free).
    pub(crate) fn free(&mut self, id: usize) {
        assert!(id < self.total, "block id {id} out of range [0, {})", self.total);
        assert!(self.in_use[id], "double free of block id {id}");
        self.in_use[id] = false;
        self.free_list.push(id);
    }

    /// Number of currently allocated blocks.
    pub(crate) fn allocated(&self) -> usize {
        self.total - self.free_list.len()
    }

    /// Number of available blocks.
    pub(crate) fn free_count(&self) -> usize {
        self.free_list.len()
    }

    /// Total pool size.
    #[cfg(test)]
    pub(crate) fn total(&self) -> usize {
        self.total
    }
}

// ---------------------------------------------------------------------------
// KV block (internal)
// ---------------------------------------------------------------------------

/// One page of K/V data for a single transformer layer.
///
/// Layout: `[num_kv_heads][page_size][head_dim]` in row-major order.
/// Only the first `used` token positions per head are populated.
struct KvBlock {
    k: Vec<f32>,
    v: Vec<f32>,
    used: usize,
}

impl KvBlock {
    /// Create a zero-initialized block with capacity for `page_size` tokens.
    fn new(num_kv_heads: usize, page_size: usize, head_dim: usize) -> Self {
        let cap = num_kv_heads * page_size * head_dim;
        Self {
            k: vec![0.0; cap],
            v: vec![0.0; cap],
            used: 0,
        }
    }

    /// Append one token's K/V data into this block.
    ///
    /// `k_data` and `v_data` each have length `num_kv_heads * head_dim`,
    /// laid out as `[head][dim]`.
    ///
    /// Scatters into the correct position for each head in the
    /// `[head][page_size][dim]` layout.
    fn append_token(
        &mut self,
        k_data: &[f32],
        v_data: &[f32],
        num_kv_heads: usize,
        page_size: usize,
        head_dim: usize,
    ) {
        debug_assert_eq!(k_data.len(), num_kv_heads * head_dim);
        debug_assert_eq!(v_data.len(), num_kv_heads * head_dim);
        debug_assert!(self.used < page_size);

        for h in 0..num_kv_heads {
            let dst_off = h * page_size * head_dim + self.used * head_dim;
            let src_off = h * head_dim;
            self.k[dst_off..dst_off + head_dim]
                .copy_from_slice(&k_data[src_off..src_off + head_dim]);
            self.v[dst_off..dst_off + head_dim]
                .copy_from_slice(&v_data[src_off..src_off + head_dim]);
        }
        self.used += 1;
    }

    /// Memory footprint in bytes (both K and V).
    fn memory_bytes(&self) -> usize {
        (self.k.len() + self.v.len()) * std::mem::size_of::<f32>()
    }
}

// ---------------------------------------------------------------------------
// Paged KV cache
// ---------------------------------------------------------------------------

/// Paged KV cache with block-level allocation.
///
/// Stores KV data in fixed-size pages managed by a [`BlockAllocator`].
/// Each transformer layer has its own page table (list of block IDs in
/// logical order). The `gather()` method assembles pages into contiguous
/// tensors matching the `[1, num_kv_heads, seq_len, head_dim]` convention.
///
/// Per-layer token counts are tracked internally to detect caller errors
/// (e.g. forgetting to append to a layer before calling `advance()`).
pub struct PagedKvCache {
    config: PagedKvConfig,
    allocator: BlockAllocator,
    /// Physical block storage indexed by block ID. Freed blocks are retained
    /// with `used = 0` to avoid reallocation on reuse.
    pool: Vec<Option<KvBlock>>,
    /// Per-layer page tables: `layer_tables[layer]` = block IDs in order.
    layer_tables: Vec<Vec<usize>>,
    /// Total tokens committed via `advance()` (position offset for RoPE).
    position: usize,
    /// Per-layer token counts (incremented by `append()`, validated by `advance()`).
    per_layer_tokens: Vec<usize>,
    num_layers: usize,
    head_dim: usize,
    num_kv_heads: usize,
}

impl PagedKvCache {
    /// Create an empty paged KV cache.
    pub fn new(
        config: PagedKvConfig,
        num_layers: usize,
        head_dim: usize,
        num_kv_heads: usize,
    ) -> Self {
        let max_pages = config.max_pages;
        Self {
            config,
            allocator: BlockAllocator::new(max_pages),
            pool: (0..max_pages).map(|_| None).collect(),
            layer_tables: (0..num_layers).map(|_| Vec::new()).collect(),
            position: 0,
            per_layer_tokens: vec![0; num_layers],
            num_layers,
            head_dim,
            num_kv_heads,
        }
    }

    /// Append one token's K/V data for a single layer.
    ///
    /// `k_data` and `v_data` each have `num_kv_heads * head_dim` f32 values,
    /// laid out as `[head][dim]`. Allocates a new page if the current one is
    /// full. Freed pages are reused without reallocation.
    ///
    /// # Errors
    ///
    /// Returns `InferenceError` if the block pool is exhausted or the layer
    /// index is out of range.
    pub fn append(
        &mut self,
        layer: usize,
        k_data: &[f32],
        v_data: &[f32],
    ) -> Result<(), InferenceError> {
        if layer >= self.num_layers {
            return Err(InferenceError::PagedKvCacheFailed(format!(
                "layer index {layer} out of range [0, {})",
                self.num_layers,
            )));
        }

        let expected = self.num_kv_heads * self.head_dim;
        if k_data.len() != expected || v_data.len() != expected {
            return Err(InferenceError::PagedKvCacheFailed(format!(
                "append data length mismatch: expected {expected}, got k={} v={}",
                k_data.len(),
                v_data.len(),
            )));
        }

        let page_size = self.config.page_size;
        let table = &mut self.layer_tables[layer];

        // Check if we need a new page.
        let need_new_page = table.is_empty()
            || self.pool[*table.last().unwrap()]
                .as_ref()
                .map_or(true, |b| b.used >= page_size);

        if need_new_page {
            let block_id = self.allocator.alloc().ok_or_else(|| {
                InferenceError::PagedKvCacheFailed(format!(
                    "block pool exhausted ({} pages allocated)",
                    self.allocator.allocated(),
                ))
            })?;
            // Reuse existing block allocation if available (freed blocks
            // retain their Vec capacity), otherwise create a new one.
            if self.pool[block_id].is_none() {
                self.pool[block_id] =
                    Some(KvBlock::new(self.num_kv_heads, page_size, self.head_dim));
            }
            table.push(block_id);
        }

        let block_id = *table.last().unwrap();
        self.pool[block_id]
            .as_mut()
            .unwrap()
            .append_token(k_data, v_data, self.num_kv_heads, page_size, self.head_dim);

        self.per_layer_tokens[layer] += 1;

        Ok(())
    }

    /// Advance the position counter after all layers have been appended.
    ///
    /// Call this once per decoded token (after appending to every layer),
    /// not once per layer. Validates that every layer has exactly
    /// `position + count` tokens appended.
    ///
    /// # Errors
    ///
    /// Returns `InferenceError` if any layer's token count doesn't match
    /// the expected `position + count`.
    pub fn advance(&mut self, count: usize) -> Result<(), InferenceError> {
        let expected = self.position + count;
        for (i, &layer_count) in self.per_layer_tokens.iter().enumerate() {
            if layer_count != expected {
                return Err(InferenceError::PagedKvCacheFailed(format!(
                    "advance({count}): layer {i} has {layer_count} tokens, \
                     expected {expected}",
                )));
            }
        }
        self.position = expected;
        Ok(())
    }

    /// Gather all pages for a layer into contiguous K/V tensors.
    ///
    /// Returns `(K, V)` with shape `[1, num_kv_heads, seq_len, head_dim]`
    /// in F32, matching the convention expected by attention.
    ///
    /// # Errors
    ///
    /// Returns `InferenceError` if the layer index is out of range.
    pub fn gather(
        &self,
        layer: usize,
        device: &Device,
    ) -> Result<(Tensor, Tensor), InferenceError> {
        if layer >= self.num_layers {
            return Err(InferenceError::PagedKvCacheFailed(format!(
                "layer index {layer} out of range [0, {})",
                self.num_layers,
            )));
        }

        let table = &self.layer_tables[layer];
        if table.is_empty() {
            return Err(InferenceError::PagedKvCacheFailed(
                "gather called on empty layer — no tokens appended".into(),
            ));
        }

        let page_size = self.config.page_size;
        let head_dim = self.head_dim;
        let num_kv_heads = self.num_kv_heads;

        // Total token count across all pages for this layer.
        let total_tokens: usize = table
            .iter()
            .map(|&id| self.pool[id].as_ref().unwrap().used)
            .sum();

        // Assemble per-head contiguous data: [num_kv_heads][total_tokens][head_dim]
        let total = num_kv_heads * total_tokens * head_dim;
        let mut k_out = Vec::with_capacity(total);
        let mut v_out = Vec::with_capacity(total);

        for h in 0..num_kv_heads {
            for &block_id in table {
                let block = self.pool[block_id].as_ref().unwrap();
                let src_off = h * page_size * head_dim;
                let len = block.used * head_dim;
                k_out.extend_from_slice(&block.k[src_off..src_off + len]);
                v_out.extend_from_slice(&block.v[src_off..src_off + len]);
            }
        }

        let shape = (1, num_kv_heads, total_tokens, head_dim);
        let k = Tensor::from_vec(k_out, shape, device)
            .map_err(|e| InferenceError::ForwardFailed(e.to_string()))?;
        let v = Tensor::from_vec(v_out, shape, device)
            .map_err(|e| InferenceError::ForwardFailed(e.to_string()))?;

        Ok((k, v))
    }

    /// Truncate the cache to `new_position` tokens, freeing trailing pages.
    ///
    /// Used for speculative decode rollback: after verification rejects some
    /// draft tokens, truncate to the last accepted position. Freed blocks
    /// retain their heap allocations for zero-cost reuse on the next append.
    ///
    /// # Errors
    ///
    /// Returns `InferenceError` if `new_position > self.position`.
    pub fn truncate(&mut self, new_position: usize) -> Result<(), InferenceError> {
        if new_position > self.position {
            return Err(InferenceError::PagedKvCacheFailed(format!(
                "truncate position {new_position} exceeds current position {}",
                self.position,
            )));
        }
        if new_position == self.position {
            return Ok(());
        }

        let page_size = self.config.page_size;

        // How many full pages to keep, and the remainder in the last page.
        let keep_pages = (new_position + page_size - 1) / page_size; // ceil div
        let last_page_used = if new_position == 0 {
            0
        } else {
            let rem = new_position % page_size;
            if rem == 0 { page_size } else { rem }
        };

        for table in &mut self.layer_tables {
            // Free pages beyond the keep range. Retain block allocations.
            while table.len() > keep_pages {
                let block_id = table.pop().unwrap();
                if let Some(block) = self.pool[block_id].as_mut() {
                    block.used = 0;
                }
                self.allocator.free(block_id);
            }

            // Adjust `used` on the new last page (if any).
            // Never increase `used` beyond its current value.
            if let Some(&last_id) = table.last() {
                if let Some(block) = self.pool[last_id].as_mut() {
                    block.used = last_page_used.min(block.used);
                }
            }
        }

        for count in &mut self.per_layer_tokens {
            *count = new_position;
        }
        self.position = new_position;
        Ok(())
    }

    /// Current token position (total tokens appended and not truncated).
    pub fn position(&self) -> usize {
        self.position
    }

    /// Whether the cache is empty (no tokens appended).
    pub fn is_empty(&self) -> bool {
        self.position == 0
    }

    /// Number of pages currently allocated across all layers.
    pub fn pages_allocated(&self) -> usize {
        self.allocator.allocated()
    }

    /// Number of free pages remaining in the pool.
    pub fn pages_free(&self) -> usize {
        self.allocator.free_count()
    }

    /// Total memory used by actively assigned blocks (bytes).
    ///
    /// Only counts blocks currently in page tables, not freed blocks
    /// retained in the pool for reuse.
    pub fn memory_bytes(&self) -> usize {
        self.layer_tables
            .iter()
            .flat_map(|table| table.iter())
            .map(|&id| self.pool[id].as_ref().unwrap().memory_bytes())
            .sum()
    }

    /// Maximum token capacity given the pool size and page size.
    ///
    /// Each layer needs its own set of pages, so the per-layer capacity is
    /// `max_pages / num_layers * page_size`.
    pub fn max_token_capacity(&self) -> usize {
        if self.num_layers == 0 {
            return 0;
        }
        (self.config.max_pages / self.num_layers) * self.config.page_size
    }

    /// Reference to the underlying config.
    pub fn config(&self) -> &PagedKvConfig {
        &self.config
    }

    /// Reset the cache: free all blocks and reset position to 0.
    ///
    /// Freed blocks retain their heap allocations for reuse.
    pub fn reset(&mut self) {
        for table in &mut self.layer_tables {
            for &block_id in table.iter() {
                if let Some(block) = self.pool[block_id].as_mut() {
                    block.used = 0;
                }
                self.allocator.free(block_id);
            }
            table.clear();
        }
        for count in &mut self.per_layer_tokens {
            *count = 0;
        }
        self.position = 0;
    }
}

impl std::fmt::Debug for PagedKvCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PagedKvCache")
            .field("page_size", &self.config.page_size)
            .field("max_pages", &self.config.max_pages)
            .field("position", &self.position)
            .field("pages_allocated", &self.allocator.allocated())
            .field("pages_free", &self.allocator.free_count())
            .field("num_layers", &self.num_layers)
            .field("head_dim", &self.head_dim)
            .field("num_kv_heads", &self.num_kv_heads)
            .finish()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    // -- Config tests --

    #[test]
    fn default_config_values() {
        let config = PagedKvConfig::default();
        assert_eq!(config.page_size, 16);
        assert_eq!(config.max_pages, 2048);
    }

    #[test]
    fn config_new_validates() {
        let config = PagedKvConfig::new(32, 1024);
        assert_eq!(config.page_size, 32);
        assert_eq!(config.max_pages, 1024);
    }

    #[test]
    #[should_panic(expected = "page_size must be >= 1")]
    fn config_rejects_zero_page_size() {
        PagedKvConfig::new(0, 100);
    }

    #[test]
    #[should_panic(expected = "max_pages must be >= 1")]
    fn config_rejects_zero_max_pages() {
        PagedKvConfig::new(16, 0);
    }

    // -- BlockAllocator tests --

    #[test]
    fn allocator_initial_state() {
        let alloc = BlockAllocator::new(10);
        assert_eq!(alloc.total(), 10);
        assert_eq!(alloc.free_count(), 10);
        assert_eq!(alloc.allocated(), 0);
    }

    #[test]
    fn allocator_alloc_and_free() {
        let mut alloc = BlockAllocator::new(3);
        let a = alloc.alloc().unwrap();
        let b = alloc.alloc().unwrap();
        assert_eq!(alloc.allocated(), 2);
        assert_eq!(alloc.free_count(), 1);

        // IDs should be unique.
        assert_ne!(a, b);

        alloc.free(a);
        assert_eq!(alloc.allocated(), 1);
        assert_eq!(alloc.free_count(), 2);

        // Freed ID is reusable.
        let c = alloc.alloc().unwrap();
        assert_eq!(c, a); // LIFO: last freed = first reused
    }

    #[test]
    fn allocator_exhaustion() {
        let mut alloc = BlockAllocator::new(2);
        assert!(alloc.alloc().is_some());
        assert!(alloc.alloc().is_some());
        assert!(alloc.alloc().is_none());
    }

    #[test]
    #[should_panic(expected = "double free")]
    fn allocator_rejects_double_free() {
        let mut alloc = BlockAllocator::new(3);
        let a = alloc.alloc().unwrap();
        alloc.free(a);
        alloc.free(a);
    }

    #[test]
    #[should_panic(expected = "out of range")]
    fn allocator_rejects_out_of_range_free() {
        let mut alloc = BlockAllocator::new(3);
        alloc.free(5);
    }

    // -- PagedKvCache core tests --

    /// Helper: create flat K/V data for one token with a recognizable pattern.
    fn token_kv(num_kv_heads: usize, head_dim: usize, val: f32) -> (Vec<f32>, Vec<f32>) {
        let n = num_kv_heads * head_dim;
        (vec![val; n], vec![val + 100.0; n])
    }

    #[test]
    fn append_and_gather_single_token() {
        let config = PagedKvConfig::new(4, 100);
        let (layers, heads, dim) = (2, 2, 4);
        let mut cache = PagedKvCache::new(config, layers, dim, heads);

        let (k, v) = token_kv(heads, dim, 1.0);
        cache.append(0, &k, &v).unwrap();
        cache.append(1, &k, &v).unwrap();
        cache.advance(1).unwrap();

        assert_eq!(cache.position(), 1);
        assert!(!cache.is_empty());

        let (kt, vt) = cache.gather(0, &Device::Cpu).unwrap();
        assert_eq!(kt.dims(), &[1, heads, 1, dim]);
        assert_eq!(vt.dims(), &[1, heads, 1, dim]);

        // Verify values.
        let k_vals: Vec<f32> = kt.flatten_all().unwrap().to_vec1().unwrap();
        assert!(k_vals.iter().all(|&x| (x - 1.0).abs() < 1e-6));
        let v_vals: Vec<f32> = vt.flatten_all().unwrap().to_vec1().unwrap();
        assert!(v_vals.iter().all(|&x| (x - 101.0).abs() < 1e-6));
    }

    #[test]
    fn append_spans_multiple_pages() {
        let page_size = 2;
        let config = PagedKvConfig::new(page_size, 100);
        let (layers, heads, dim) = (1, 1, 4);
        let mut cache = PagedKvCache::new(config, layers, dim, heads);

        // Append 5 tokens → needs 3 pages (2 + 2 + 1).
        for i in 0..5 {
            let (k, v) = token_kv(heads, dim, i as f32);
            cache.append(0, &k, &v).unwrap();
            cache.advance(1).unwrap();
        }

        assert_eq!(cache.position(), 5);
        assert_eq!(cache.layer_tables[0].len(), 3); // 3 pages

        let (kt, _) = cache.gather(0, &Device::Cpu).unwrap();
        assert_eq!(kt.dims(), &[1, 1, 5, dim]);

        // Values should be [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, ..., 4, 4, 4, 4].
        let vals: Vec<f32> = kt.flatten_all().unwrap().to_vec1().unwrap();
        for (t, chunk) in vals.chunks(dim).enumerate() {
            let expected = t as f32;
            assert!(
                chunk.iter().all(|&x| (x - expected).abs() < 1e-6),
                "token {t}: expected {expected}, got {:?}",
                chunk
            );
        }
    }

    #[test]
    fn gather_errors_on_empty_layer() {
        let config = PagedKvConfig::new(4, 100);
        let cache = PagedKvCache::new(config, 2, 4, 2);
        let err = cache.gather(0, &Device::Cpu).unwrap_err();
        assert!(err.to_string().contains("empty layer"));
    }

    #[test]
    fn gather_errors_on_invalid_layer() {
        let config = PagedKvConfig::new(4, 100);
        let cache = PagedKvCache::new(config, 2, 4, 2);
        let err = cache.gather(5, &Device::Cpu).unwrap_err();
        assert!(err.to_string().contains("out of range"));
    }

    #[test]
    fn append_errors_on_invalid_layer() {
        let config = PagedKvConfig::new(4, 100);
        let mut cache = PagedKvCache::new(config, 2, 4, 2);
        let (k, v) = token_kv(2, 4, 0.0);
        let err = cache.append(5, &k, &v).unwrap_err();
        assert!(err.to_string().contains("out of range"));
    }

    #[test]
    fn append_errors_on_data_length_mismatch() {
        let config = PagedKvConfig::new(4, 100);
        let mut cache = PagedKvCache::new(config, 1, 4, 2);
        let err = cache.append(0, &[1.0; 3], &[2.0; 8]).unwrap_err();
        assert!(err.to_string().contains("length mismatch"));
    }

    #[test]
    fn pool_exhaustion_error() {
        // Only 2 pages total, 1 layer, page_size=2 → max 4 tokens.
        let config = PagedKvConfig::new(2, 2);
        let mut cache = PagedKvCache::new(config, 1, 2, 1);
        let (k, v) = token_kv(1, 2, 0.0);

        // 4 tokens fit in 2 pages.
        for _ in 0..4 {
            cache.append(0, &k, &v).unwrap();
            cache.advance(1).unwrap();
        }

        // 5th token needs a 3rd page → exhaustion.
        let err = cache.append(0, &k, &v).unwrap_err();
        assert!(err.to_string().contains("exhausted"));
    }

    // -- Advance validation tests --

    #[test]
    fn advance_validates_layer_tokens() {
        let config = PagedKvConfig::new(4, 100);
        let mut cache = PagedKvCache::new(config, 2, 4, 2);
        let (k, v) = token_kv(2, 4, 1.0);

        // Only append to layer 0, not layer 1.
        cache.append(0, &k, &v).unwrap();

        // advance should fail because layer 1 hasn't been appended.
        let err = cache.advance(1).unwrap_err();
        assert!(err.to_string().contains("layer 1"));
    }

    #[test]
    fn advance_succeeds_when_all_layers_match() {
        let config = PagedKvConfig::new(4, 100);
        let mut cache = PagedKvCache::new(config, 3, 4, 2);
        let (k, v) = token_kv(2, 4, 1.0);

        for layer in 0..3 {
            cache.append(layer, &k, &v).unwrap();
        }
        cache.advance(1).unwrap();
        assert_eq!(cache.position(), 1);
    }

    // -- Truncation tests --

    #[test]
    fn truncate_within_page() {
        let config = PagedKvConfig::new(4, 100);
        let (layers, heads, dim) = (1, 1, 2);
        let mut cache = PagedKvCache::new(config, layers, dim, heads);

        for i in 0..3 {
            let (k, v) = token_kv(heads, dim, i as f32);
            cache.append(0, &k, &v).unwrap();
            cache.advance(1).unwrap();
        }
        assert_eq!(cache.position(), 3);

        cache.truncate(2).unwrap();
        assert_eq!(cache.position(), 2);

        let (kt, _) = cache.gather(0, &Device::Cpu).unwrap();
        assert_eq!(kt.dims(), &[1, 1, 2, dim]);
    }

    #[test]
    fn truncate_frees_trailing_pages() {
        let config = PagedKvConfig::new(2, 100);
        let (layers, heads, dim) = (1, 1, 2);
        let mut cache = PagedKvCache::new(config, layers, dim, heads);

        // 4 tokens → 2 pages.
        for i in 0..4 {
            let (k, v) = token_kv(heads, dim, i as f32);
            cache.append(0, &k, &v).unwrap();
            cache.advance(1).unwrap();
        }
        assert_eq!(cache.pages_allocated(), 2);

        // Truncate to 1 token → should free second page.
        cache.truncate(1).unwrap();
        assert_eq!(cache.position(), 1);
        assert_eq!(cache.pages_allocated(), 1);
        assert_eq!(cache.layer_tables[0].len(), 1);
    }

    #[test]
    fn truncate_to_zero_frees_all() {
        let config = PagedKvConfig::new(2, 100);
        let (layers, heads, dim) = (1, 1, 2);
        let mut cache = PagedKvCache::new(config, layers, dim, heads);

        for i in 0..3 {
            let (k, v) = token_kv(heads, dim, i as f32);
            cache.append(0, &k, &v).unwrap();
            cache.advance(1).unwrap();
        }

        cache.truncate(0).unwrap();
        assert_eq!(cache.position(), 0);
        assert!(cache.is_empty());
        assert_eq!(cache.pages_allocated(), 0);
        assert!(cache.layer_tables[0].is_empty());
    }

    #[test]
    fn truncate_noop_at_current_position() {
        let config = PagedKvConfig::new(4, 100);
        let mut cache = PagedKvCache::new(config, 1, 2, 1);
        let (k, v) = token_kv(1, 2, 0.0);
        cache.append(0, &k, &v).unwrap();
        cache.advance(1).unwrap();

        cache.truncate(1).unwrap(); // no-op
        assert_eq!(cache.position(), 1);
    }

    #[test]
    fn truncate_errors_beyond_position() {
        let config = PagedKvConfig::new(4, 100);
        let mut cache = PagedKvCache::new(config, 1, 2, 1);
        let err = cache.truncate(5).unwrap_err();
        assert!(err.to_string().contains("exceeds current position"));
    }

    #[test]
    fn truncate_at_page_boundary() {
        let config = PagedKvConfig::new(2, 100);
        let (layers, heads, dim) = (1, 1, 2);
        let mut cache = PagedKvCache::new(config, layers, dim, heads);

        // 4 tokens → 2 full pages.
        for i in 0..4 {
            let (k, v) = token_kv(heads, dim, i as f32);
            cache.append(0, &k, &v).unwrap();
            cache.advance(1).unwrap();
        }

        // Truncate to exactly 2 → keep first page, free second.
        cache.truncate(2).unwrap();
        assert_eq!(cache.position(), 2);
        assert_eq!(cache.pages_allocated(), 1);

        let (kt, _) = cache.gather(0, &Device::Cpu).unwrap();
        assert_eq!(kt.dims(), &[1, 1, 2, dim]);
    }

    // -- Multi-layer tests --

    #[test]
    fn multi_layer_independent_tables() {
        let config = PagedKvConfig::new(2, 100);
        let (layers, heads, dim) = (3, 1, 2);
        let mut cache = PagedKvCache::new(config, layers, dim, heads);

        for _ in 0..3 {
            let (k, v) = token_kv(heads, dim, 1.0);
            for layer in 0..layers {
                cache.append(layer, &k, &v).unwrap();
            }
            cache.advance(1).unwrap();
        }

        // Each layer should have 2 pages (page_size=2, 3 tokens).
        for layer in 0..layers {
            assert_eq!(cache.layer_tables[layer].len(), 2);
            let (kt, _) = cache.gather(layer, &Device::Cpu).unwrap();
            assert_eq!(kt.dims(), &[1, 1, 3, dim]);
        }

        // 3 layers × 2 pages each = 6 pages allocated.
        assert_eq!(cache.pages_allocated(), 6);
    }

    // -- Reset test --

    #[test]
    fn reset_clears_everything() {
        let config = PagedKvConfig::new(2, 100);
        let mut cache = PagedKvCache::new(config, 2, 4, 2);

        let (k, v) = token_kv(2, 4, 1.0);
        for _ in 0..3 {
            cache.append(0, &k, &v).unwrap();
            cache.append(1, &k, &v).unwrap();
            cache.advance(1).unwrap();
        }

        cache.reset();
        assert_eq!(cache.position(), 0);
        assert!(cache.is_empty());
        assert_eq!(cache.pages_allocated(), 0);
        assert_eq!(cache.pages_free(), 100);
    }

    // -- Memory and capacity tests --

    #[test]
    fn memory_bytes_scales_with_pages() {
        let config = PagedKvConfig::new(4, 100);
        let (heads, dim) = (2, 8);
        let mut cache = PagedKvCache::new(config, 1, dim, heads);

        assert_eq!(cache.memory_bytes(), 0);

        let (k, v) = token_kv(heads, dim, 0.0);
        cache.append(0, &k, &v).unwrap();
        cache.advance(1).unwrap();

        // One page: 2 * (heads * page_size * dim * sizeof(f32))
        let expected = 2 * heads * 4 * dim * 4; // 2 sides × 2 heads × 4 slots × 8 dim × 4 bytes
        assert_eq!(cache.memory_bytes(), expected);
    }

    #[test]
    fn max_token_capacity() {
        let config = PagedKvConfig::new(16, 2048);
        let cache = PagedKvCache::new(config, 32, 64, 4);
        // 2048 pages / 32 layers = 64 pages per layer × 16 tokens = 1024
        assert_eq!(cache.max_token_capacity(), 1024);
    }

    // -- Reuse after truncation --

    #[test]
    fn freed_pages_reusable_after_truncate() {
        // 4 total pages, 1 layer, page_size=2 → max 8 tokens if no multi-layer.
        let config = PagedKvConfig::new(2, 4);
        let (heads, dim) = (1, 2);
        let mut cache = PagedKvCache::new(config, 1, dim, heads);

        // Fill 4 tokens → 2 pages.
        for i in 0..4 {
            let (k, v) = token_kv(heads, dim, i as f32);
            cache.append(0, &k, &v).unwrap();
            cache.advance(1).unwrap();
        }
        assert_eq!(cache.pages_free(), 2);

        // Truncate to 2 → frees 1 page.
        cache.truncate(2).unwrap();
        assert_eq!(cache.pages_free(), 3);

        // Can now append 4 more tokens (2 in existing half-page + 2 pages × 2).
        // Actually: current page has used=2 (full), so we need new pages for tokens 3-5.
        for i in 0..4 {
            let (k, v) = token_kv(heads, dim, (10 + i) as f32);
            cache.append(0, &k, &v).unwrap();
            cache.advance(1).unwrap();
        }
        assert_eq!(cache.position(), 6);

        let (kt, _) = cache.gather(0, &Device::Cpu).unwrap();
        assert_eq!(kt.dims(), &[1, 1, 6, dim]);
    }

    #[test]
    fn truncate_retains_block_allocations() {
        let config = PagedKvConfig::new(2, 10);
        let (heads, dim) = (1, 2);
        let mut cache = PagedKvCache::new(config, 1, dim, heads);

        // Append 4 tokens (2 pages), then truncate to 0.
        for i in 0..4 {
            let (k, v) = token_kv(heads, dim, i as f32);
            cache.append(0, &k, &v).unwrap();
            cache.advance(1).unwrap();
        }
        cache.truncate(0).unwrap();

        // Pool entries should still exist (blocks retained for reuse).
        // Re-appending should reuse those allocations, not create new ones.
        for i in 0..4 {
            let (k, v) = token_kv(heads, dim, i as f32);
            cache.append(0, &k, &v).unwrap();
            cache.advance(1).unwrap();
        }
        assert_eq!(cache.position(), 4);
        assert_eq!(cache.pages_allocated(), 2);
    }

    // -- Multi-head gather correctness --

    #[test]
    fn multi_head_gather_preserves_head_ordering() {
        let config = PagedKvConfig::new(2, 100);
        let (heads, dim) = (3, 2);
        let mut cache = PagedKvCache::new(config, 1, dim, heads);

        // Append 3 tokens with distinguishable per-head data.
        for t in 0..3u32 {
            // K data: head h, dim d → (h+1) * 10 + t
            let mut k_data = vec![0.0f32; heads * dim];
            let mut v_data = vec![0.0f32; heads * dim];
            for h in 0..heads {
                for d in 0..dim {
                    k_data[h * dim + d] = (h as f32 + 1.0) * 10.0 + t as f32;
                    v_data[h * dim + d] = (h as f32 + 1.0) * 100.0 + t as f32;
                }
            }
            cache.append(0, &k_data, &v_data).unwrap();
            cache.advance(1).unwrap();
        }

        let (kt, _vt) = cache.gather(0, &Device::Cpu).unwrap();
        assert_eq!(kt.dims(), &[1, heads, 3, dim]);

        // Verify: kt[0, h, t, :] should all be (h+1)*10 + t.
        let k_flat: Vec<f32> = kt.flatten_all().unwrap().to_vec1().unwrap();
        for h in 0..heads {
            for t in 0..3 {
                let expected = (h as f32 + 1.0) * 10.0 + t as f32;
                for d in 0..dim {
                    let idx = h * 3 * dim + t * dim + d;
                    assert!(
                        (k_flat[idx] - expected).abs() < 1e-6,
                        "k[0,{h},{t},{d}] = {}, expected {expected}",
                        k_flat[idx]
                    );
                }
            }
        }
    }

    // -- Error variant test --

    #[test]
    fn errors_use_paged_kv_cache_variant() {
        let config = PagedKvConfig::new(4, 100);
        let cache = PagedKvCache::new(config, 2, 4, 2);
        let err = cache.gather(5, &Device::Cpu).unwrap_err();
        // Should use PagedKvCacheFailed, not SpeculativeDecodeFailed.
        assert!(err.to_string().starts_with("paged KV cache error:"));
    }

    // -- Debug formatting --

    #[test]
    fn debug_format() {
        let config = PagedKvConfig::new(16, 2048);
        let cache = PagedKvCache::new(config, 32, 64, 4);
        let debug = format!("{:?}", cache);
        assert!(debug.contains("PagedKvCache"));
        assert!(debug.contains("page_size: 16"));
    }
}
