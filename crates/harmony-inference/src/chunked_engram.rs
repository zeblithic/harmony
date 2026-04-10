//! Chunked periodic Engram injection during decode (Strategy C).
//!
//! During autoregressive decoding, each forward pass processes a single token
//! (seq_len=1). The Engram bridge can't form N-grams from one token, so decode
//! steps normally get zero Engram injection. The [`ChunkedEngramScheduler`]
//! solves this by maintaining a sliding window of recent tokens and refreshing
//! Engram embeddings every K decode steps, amortizing shard fetch latency over
//! K tokens. Follows the RETRO chunked cross-attention heuristic.
//!
//! # Usage
//!
//! ```ignore
//! let mut scheduler = ChunkedEngramScheduler::new(ChunkedEngramConfig {
//!     chunk_size: 8,
//!     max_window: 10,
//! });
//! scheduler.seed(&prompt_tokens, Some(prefill_last_embedding));
//!
//! // Decode loop:
//! if scheduler.push_token(token) == StepResult::ChunkBoundary {
//!     let request = scheduler.prepare_request(&client)?;
//!     let missing = scheduler.missing_shards(&request);
//!     let new_data = fetcher.fetch(&missing)?;
//!     scheduler.resolve(&request, new_data, &client, &device)?;
//! }
//! let embedding = scheduler.cached_embedding(); // [1, 1, engram_dim]
//! ```

use std::collections::{HashMap, VecDeque};

use candle_core::{Device, Tensor};
use harmony_engram::EngramClient;

use crate::engram_bridge::{self, EngramRequest, ShardRequest};
use crate::error::InferenceError;

/// Configuration for chunked Engram injection during decode.
#[derive(Debug, Clone)]
pub struct ChunkedEngramConfig {
    /// Chunk size K: decode steps between Engram refreshes. Must be >= 1.
    pub chunk_size: usize,
    /// Max tokens retained in the sliding window buffer.
    /// Default: `chunk_size + 2` (trigram context for the chunk's first token).
    pub max_window: usize,
}

impl ChunkedEngramConfig {
    /// Create a config with the given chunk size and default max_window.
    pub fn new(chunk_size: usize) -> Self {
        assert!(chunk_size >= 1, "chunk_size must be >= 1");
        Self {
            chunk_size,
            max_window: chunk_size + 2,
        }
    }
}

/// Result of [`ChunkedEngramScheduler::push_token`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StepResult {
    /// Not a chunk boundary; continue with the cached embedding.
    Continue,
    /// Chunk boundary reached; caller should fetch shards and call
    /// [`ChunkedEngramScheduler::resolve`].
    ChunkBoundary,
}

/// Trait for fetching Engram shards by CID.
///
/// Implementations bridge the synchronous decode loop to the asynchronous
/// event loop. The simplest implementation sends shard requests over a
/// channel and blocks on a oneshot for the results.
pub trait ShardFetcher {
    /// Fetch shard data for the given requests. Blocks until all data is
    /// available. Returns a map from `shard_index` to raw shard bytes.
    fn fetch(&self, shards: &[&ShardRequest]) -> Result<HashMap<u64, Vec<u8>>, InferenceError>;
}

/// Manages chunked periodic Engram injection during decode.
///
/// Sans-I/O: produces [`EngramRequest`]s for the caller to fulfill, caches
/// shard data across chunks, and provides a cached embedding tensor for each
/// decode step. The caller orchestrates shard fetching via [`ShardFetcher`]
/// or manual request/resolve calls.
///
/// # Lifecycle
///
/// 1. [`seed`](Self::seed) with prefill tokens and optional initial embedding
/// 2. For each decode step: [`push_token`](Self::push_token)
/// 3. At chunk boundaries: [`prepare_request`](Self::prepare_request) →
///    fetch missing shards → [`resolve`](Self::resolve)
/// 4. Use [`cached_embedding`](Self::cached_embedding) for `forward_with_engram`
pub struct ChunkedEngramScheduler {
    config: ChunkedEngramConfig,
    /// Sliding window of recent tokens for N-gram context.
    token_buffer: VecDeque<u32>,
    /// Cached embedding for current chunk: `[1, 1, engram_dim]`.
    cached_embedding: Option<Tensor>,
    /// Decode steps since the last refresh.
    steps_since_refresh: usize,
    /// Persistent shard data cache: `shard_index → raw bytes`.
    shard_cache: HashMap<u64, Vec<u8>>,
}

impl ChunkedEngramScheduler {
    /// Create a new scheduler with the given configuration.
    ///
    /// # Panics
    ///
    /// Panics if `config.chunk_size` is 0 or `config.max_window` is 0.
    pub fn new(config: ChunkedEngramConfig) -> Self {
        assert!(config.chunk_size >= 1, "chunk_size must be >= 1");
        assert!(config.max_window >= 1, "max_window must be >= 1");
        Self {
            config,
            token_buffer: VecDeque::new(),
            cached_embedding: None,
            steps_since_refresh: 0,
            shard_cache: HashMap::new(),
        }
    }

    /// Seed the token buffer from prefill tokens.
    ///
    /// Retains the last `max_window` tokens from the provided slice.
    /// Optionally sets an initial cached embedding (typically the last
    /// position's embedding from the prefill Engram resolution) so that
    /// the first `chunk_size` decode steps have Engram context.
    pub fn seed(&mut self, tokens: &[u32], initial_embedding: Option<Tensor>) {
        self.token_buffer.clear();
        let start = tokens.len().saturating_sub(self.config.max_window);
        for &t in &tokens[start..] {
            self.token_buffer.push_back(t);
        }
        self.cached_embedding = initial_embedding;
        self.steps_since_refresh = 0;
    }

    /// Register a newly decoded token.
    ///
    /// Appends the token to the sliding window buffer (trimming to
    /// `max_window`) and increments the step counter. Returns
    /// [`StepResult::ChunkBoundary`] every `chunk_size` steps.
    pub fn push_token(&mut self, token: u32) -> StepResult {
        self.token_buffer.push_back(token);
        // Trim buffer to max_window.
        while self.token_buffer.len() > self.config.max_window {
            self.token_buffer.pop_front();
        }
        self.steps_since_refresh += 1;
        if self.steps_since_refresh >= self.config.chunk_size {
            StepResult::ChunkBoundary
        } else {
            StepResult::Continue
        }
    }

    /// Prepare an [`EngramRequest`] from the current token window.
    ///
    /// Extracts bigrams and trigrams from the full buffer and returns the
    /// request with all required shards (including those already cached).
    /// Use [`missing_shards`](Self::missing_shards) to filter.
    pub fn prepare_request(
        &self,
        client: &EngramClient,
    ) -> Result<EngramRequest, InferenceError> {
        let window: Vec<u32> = self.token_buffer.iter().copied().collect();
        engram_bridge::prepare_engram_request(client, &window)
            .map_err(|e| InferenceError::EngramResolutionFailed(e.to_string()))
    }

    /// Return shard requests not already present in the internal cache.
    pub fn missing_shards<'a>(&self, request: &'a EngramRequest) -> Vec<&'a ShardRequest> {
        request
            .required_shards
            .iter()
            .filter(|s| !self.shard_cache.contains_key(&s.shard_index))
            .collect()
    }

    /// Supply newly-fetched shard data and resolve embeddings.
    ///
    /// Merges `new_shards` into the persistent cache, resolves the full
    /// embedding tensor from the token window, extracts the last position's
    /// embedding as `[1, 1, engram_dim]`, and caches it for subsequent
    /// decode steps. Resets the step counter.
    pub fn resolve(
        &mut self,
        request: &EngramRequest,
        new_shards: HashMap<u64, Vec<u8>>,
        client: &EngramClient,
        device: &Device,
    ) -> Result<(), InferenceError> {
        // Stage merged shard data so self.shard_cache is unchanged on error.
        let mut merged = self.shard_cache.clone();
        for (idx, data) in new_shards {
            merged.insert(idx, data);
        }

        // Resolve full embeddings: [1, window_len, engram_dim].
        let embeddings =
            engram_bridge::resolve_engram_embeddings(client, request, &merged, device)
                .map_err(|e| InferenceError::EngramResolutionFailed(e.to_string()))?;

        // Extract last position: [1, 1, engram_dim].
        let seq_len = request.seq_len;
        if seq_len == 0 {
            return Err(InferenceError::EngramResolutionFailed(
                "empty token window — cannot resolve embedding".into(),
            ));
        }
        let last_embedding = embeddings
            .narrow(1, seq_len - 1, 1)
            .map_err(|e| InferenceError::EngramResolutionFailed(e.to_string()))?;

        // Commit: resolution succeeded, update all mutable state.
        self.shard_cache = merged;
        self.cached_embedding = Some(last_embedding);
        self.steps_since_refresh = 0;
        Ok(())
    }

    /// Access the current token buffer (for latent projection).
    pub fn token_buffer(&self) -> &VecDeque<u32> {
        &self.token_buffer
    }

    /// Get the cached embedding for the current decode step.
    ///
    /// Returns `Some(&Tensor)` with shape `[1, 1, engram_dim]` if an
    /// embedding has been resolved (via [`seed`](Self::seed) or
    /// [`resolve`](Self::resolve)), or `None` otherwise.
    pub fn cached_embedding(&self) -> Option<&Tensor> {
        self.cached_embedding.as_ref()
    }

    /// Prepare an Engram request using latent projection instead of token hashing.
    ///
    /// When a [`LatentProjection`](crate::latent_projection::LatentProjection) is
    /// available, projects the token embeddings through the MLP and generates
    /// binary LSH keys instead of xxhash64 token-byte keys.
    ///
    /// The caller provides `embeddings` for the current token buffer (from
    /// `engine.token_embeddings()`).
    pub fn prepare_request_latent(
        &self,
        client: &EngramClient,
        projection: &crate::latent_projection::LatentProjection,
        embeddings: &candle_core::Tensor,
    ) -> Result<EngramRequest, InferenceError> {
        let window_len = self.token_buffer.len();
        let (keys, positions) = projection
            .project_ngrams(embeddings, window_len)
            .map_err(|e| InferenceError::EngramResolutionFailed(e.to_string()))?;
        engram_bridge::prepare_engram_request_latent(
            client.config(),
            &keys,
            &positions,
            window_len,
        )
        .map_err(|e| InferenceError::EngramResolutionFailed(e.to_string()))
    }

    /// Number of shards currently in the persistent cache.
    pub fn shard_cache_len(&self) -> usize {
        self.shard_cache.len()
    }

    /// Number of decode steps since the last refresh.
    pub fn steps_since_refresh(&self) -> usize {
        self.steps_since_refresh
    }

    /// Steps remaining until the next chunk boundary Engram refresh.
    pub fn steps_until_boundary(&self) -> usize {
        self.config.chunk_size.saturating_sub(self.steps_since_refresh)
    }

    /// Clear all state (token buffer, cached embedding, shard cache).
    pub fn reset(&mut self) {
        self.token_buffer.clear();
        self.cached_embedding = None;
        self.steps_since_refresh = 0;
        self.shard_cache.clear();
    }
}

/// Convenience function: push a token and automatically resolve at chunk
/// boundaries using the provided [`ShardFetcher`].
///
/// Combines [`ChunkedEngramScheduler::push_token`], [`prepare_request`],
/// [`missing_shards`], and [`resolve`] into a single call. After this
/// returns, [`ChunkedEngramScheduler::cached_embedding`] is up to date.
pub fn step_decode(
    scheduler: &mut ChunkedEngramScheduler,
    token: u32,
    client: &EngramClient,
    fetcher: &dyn ShardFetcher,
    device: &Device,
) -> Result<(), InferenceError> {
    if scheduler.push_token(token) == StepResult::ChunkBoundary {
        let request = scheduler.prepare_request(client)?;
        let missing = scheduler.missing_shards(&request);
        let new_data = if missing.is_empty() {
            HashMap::new()
        } else {
            fetcher.fetch(&missing)?
        };
        scheduler.resolve(&request, new_data, client, device)?;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    use harmony_engram::EngramConfig;

    /// Embedding dimension used by test_client().
    const EMBEDDING_DIM: usize = 4;

    /// Create a minimal EngramClient for testing.
    /// Mirrors the test_client() in engram_bridge.rs tests.
    fn test_client() -> EngramClient {
        let config = EngramConfig {
            version: "test".into(),
            embedding_dim: EMBEDDING_DIM,
            dtype_bytes: 2, // f16
            num_heads: 2,
            shard_size: 100,
            num_shards: 10,
            total_entries: 1000,
            hash_seeds: vec![42, 99],
        };
        let shard_cids: Vec<[u8; 32]> = (0..10)
            .map(|i| {
                let mut cid = [0u8; 32];
                cid[0] = i as u8;
                cid
            })
            .collect();
        EngramClient::from_manifest(config, shard_cids)
    }

    /// Create mock shard data: all zeros (valid f16 zero = 0x0000).
    fn zero_shard_data(num_shards: usize, shard_bytes: usize) -> HashMap<u64, Vec<u8>> {
        (0..num_shards as u64)
            .map(|i| (i, vec![0u8; shard_bytes]))
            .collect()
    }

    // ── Config validation ───────────────────────────────────────────────

    #[test]
    #[should_panic(expected = "chunk_size must be >= 1")]
    fn chunk_size_zero_panics() {
        ChunkedEngramConfig::new(0);
    }

    #[test]
    #[should_panic(expected = "max_window must be >= 1")]
    fn max_window_zero_panics() {
        ChunkedEngramScheduler::new(ChunkedEngramConfig {
            chunk_size: 4,
            max_window: 0,
        });
    }

    #[test]
    fn default_max_window() {
        let cfg = ChunkedEngramConfig::new(8);
        assert_eq!(cfg.chunk_size, 8);
        assert_eq!(cfg.max_window, 10); // 8 + 2
    }

    // ── Boundary detection ──────────────────────────────────────────────

    #[test]
    fn push_token_returns_continue_before_boundary() {
        let mut scheduler = ChunkedEngramScheduler::new(ChunkedEngramConfig::new(4));
        assert_eq!(scheduler.push_token(1), StepResult::Continue);
        assert_eq!(scheduler.push_token(2), StepResult::Continue);
        assert_eq!(scheduler.push_token(3), StepResult::Continue);
    }

    #[test]
    fn push_token_returns_chunk_boundary_at_k() {
        let mut scheduler = ChunkedEngramScheduler::new(ChunkedEngramConfig::new(4));
        assert_eq!(scheduler.push_token(1), StepResult::Continue);
        assert_eq!(scheduler.push_token(2), StepResult::Continue);
        assert_eq!(scheduler.push_token(3), StepResult::Continue);
        assert_eq!(scheduler.push_token(4), StepResult::ChunkBoundary);
    }

    #[test]
    fn boundary_resets_after_resolve() {
        let client = test_client();
        let device = Device::Cpu;
        let mut scheduler = ChunkedEngramScheduler::new(ChunkedEngramConfig::new(3));
        // Seed with enough context for N-grams.
        scheduler.seed(&[10, 20], None);

        // Push 3 tokens → boundary.
        assert_eq!(scheduler.push_token(1), StepResult::Continue);
        assert_eq!(scheduler.push_token(2), StepResult::Continue);
        assert_eq!(scheduler.push_token(3), StepResult::ChunkBoundary);

        // Resolve to reset counter.
        // shard_size(100) * vector_bytes(4*2=8) = 800 bytes per shard
        let request = scheduler.prepare_request(&client).unwrap();
        let shard_data = zero_shard_data(10, 800);
        scheduler.resolve(&request, shard_data, &client, &device).unwrap();

        // Next 3 tokens: continue, continue, boundary.
        assert_eq!(scheduler.push_token(4), StepResult::Continue);
        assert_eq!(scheduler.push_token(5), StepResult::Continue);
        assert_eq!(scheduler.push_token(6), StepResult::ChunkBoundary);
    }

    // ── Token buffer ────────────────────────────────────────────────────

    #[test]
    fn seed_populates_buffer() {
        let mut scheduler = ChunkedEngramScheduler::new(ChunkedEngramConfig::new(4));
        scheduler.seed(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], None);
        // max_window = 6, so last 6 tokens retained.
        let buf: Vec<u32> = scheduler.token_buffer.iter().copied().collect();
        assert_eq!(buf, vec![5, 6, 7, 8, 9, 10]);
    }

    #[test]
    fn buffer_trims_to_max_window() {
        let mut scheduler = ChunkedEngramScheduler::new(ChunkedEngramConfig::new(3));
        // max_window = 5
        scheduler.seed(&[10, 20, 30], None);
        // Push tokens beyond max_window.
        for t in 1..=10 {
            scheduler.push_token(t);
        }
        assert_eq!(scheduler.token_buffer.len(), 5);
        let buf: Vec<u32> = scheduler.token_buffer.iter().copied().collect();
        assert_eq!(buf, vec![6, 7, 8, 9, 10]);
    }

    // ── Request preparation ─────────────────────────────────────────────

    #[test]
    fn prepare_request_produces_ngrams() {
        let client = test_client();
        let mut scheduler = ChunkedEngramScheduler::new(ChunkedEngramConfig::new(4));
        scheduler.seed(&[1, 2, 3, 4], None);

        let request = scheduler.prepare_request(&client).unwrap();
        // 4 tokens → 3 bigrams + 2 trigrams = 5 lookups
        assert_eq!(request.lookups.len(), 5);
        assert_eq!(request.seq_len, 4);
    }

    #[test]
    fn missing_shards_filters_cached() {
        let client = test_client();
        let mut scheduler = ChunkedEngramScheduler::new(ChunkedEngramConfig::new(4));
        scheduler.seed(&[1, 2, 3, 4], None);

        let request = scheduler.prepare_request(&client).unwrap();
        let all_missing = scheduler.missing_shards(&request);
        assert_eq!(all_missing.len(), request.required_shards.len());

        // Pre-populate shard cache with all required shards.
        for s in &request.required_shards {
            scheduler.shard_cache.insert(s.shard_index, vec![0u8; 800]);
        }
        let none_missing = scheduler.missing_shards(&request);
        assert!(none_missing.is_empty());
    }

    // ── Resolution ──────────────────────────────────────────────────────

    #[test]
    fn resolve_caches_last_position_embedding() {
        let client = test_client();
        let device = Device::Cpu;
        let mut scheduler = ChunkedEngramScheduler::new(ChunkedEngramConfig::new(3));
        scheduler.seed(&[10, 20], None);
        scheduler.push_token(1);
        scheduler.push_token(2);
        scheduler.push_token(3);

        let request = scheduler.prepare_request(&client).unwrap();
        let shard_data = zero_shard_data(10, 800);
        scheduler.resolve(&request, shard_data, &client, &device).unwrap();

        let emb = scheduler.cached_embedding().expect("should have embedding");
        // Shape: [1, 1, embedding_dim=4]
        assert_eq!(emb.dims(), &[1, 1, 4]);
    }

    #[test]
    fn resolve_merges_into_shard_cache() {
        let client = test_client();
        let device = Device::Cpu;
        let mut scheduler = ChunkedEngramScheduler::new(ChunkedEngramConfig::new(2));
        scheduler.seed(&[1, 2, 3], None);
        scheduler.push_token(4);
        scheduler.push_token(5);

        assert_eq!(scheduler.shard_cache_len(), 0);

        let request = scheduler.prepare_request(&client).unwrap();
        let shard_data = zero_shard_data(10, 800);
        let num_shards = shard_data.len();
        scheduler.resolve(&request, shard_data, &client, &device).unwrap();

        assert_eq!(scheduler.shard_cache_len(), num_shards);
    }

    #[test]
    fn resolve_with_all_cached_shards() {
        let client = test_client();
        let device = Device::Cpu;
        let mut scheduler = ChunkedEngramScheduler::new(ChunkedEngramConfig::new(2));
        scheduler.seed(&[1, 2, 3], None);

        // Pre-populate shard cache.
        for i in 0..10u64 {
            scheduler.shard_cache.insert(i, vec![0u8; 800]);
        }

        scheduler.push_token(4);
        scheduler.push_token(5);

        let request = scheduler.prepare_request(&client).unwrap();
        let missing = scheduler.missing_shards(&request);
        assert!(missing.is_empty());

        // Resolve with empty new_shards — uses only cached data.
        scheduler
            .resolve(&request, HashMap::new(), &client, &device)
            .unwrap();

        assert!(scheduler.cached_embedding().is_some());
    }

    // ── Seed with embedding ─────────────────────────────────────────────

    #[test]
    fn seed_with_initial_embedding() {
        let mut scheduler = ChunkedEngramScheduler::new(ChunkedEngramConfig::new(4));
        let emb = Tensor::zeros((1, 1, 4), candle_core::DType::F32, &Device::Cpu).unwrap();
        scheduler.seed(&[1, 2, 3], Some(emb));
        assert!(scheduler.cached_embedding().is_some());
        assert_eq!(scheduler.cached_embedding().unwrap().dims(), &[1, 1, 4]);
    }

    #[test]
    fn seed_without_embedding_returns_none() {
        let mut scheduler = ChunkedEngramScheduler::new(ChunkedEngramConfig::new(4));
        scheduler.seed(&[1, 2, 3], None);
        assert!(scheduler.cached_embedding().is_none());
    }

    // ── step_decode convenience ─────────────────────────────────────────

    /// Mock ShardFetcher that returns zero shard data.
    struct MockFetcher;

    impl ShardFetcher for MockFetcher {
        fn fetch(
            &self,
            shards: &[&ShardRequest],
        ) -> Result<HashMap<u64, Vec<u8>>, InferenceError> {
            Ok(shards
                .iter()
                .map(|s| (s.shard_index, vec![0u8; 800]))
                .collect())
        }
    }

    #[test]
    fn step_decode_combines_push_and_resolve() {
        let client = test_client();
        let device = Device::Cpu;
        let fetcher = MockFetcher;
        let mut scheduler = ChunkedEngramScheduler::new(ChunkedEngramConfig::new(3));
        scheduler.seed(&[10, 20], None);

        // Steps 1, 2: no resolution.
        step_decode(&mut scheduler, 1, &client, &fetcher, &device).unwrap();
        assert!(scheduler.cached_embedding().is_none());
        step_decode(&mut scheduler, 2, &client, &fetcher, &device).unwrap();
        assert!(scheduler.cached_embedding().is_none());

        // Step 3: chunk boundary → auto-resolve.
        step_decode(&mut scheduler, 3, &client, &fetcher, &device).unwrap();
        assert!(scheduler.cached_embedding().is_some());
        assert_eq!(
            scheduler.cached_embedding().unwrap().dims(),
            &[1, 1, 4],
        );
        assert_eq!(scheduler.steps_since_refresh(), 0);
    }

    // ── Reset ───────────────────────────────────────────────────────────

    #[test]
    fn reset_clears_all_state() {
        let client = test_client();
        let device = Device::Cpu;
        let mut scheduler = ChunkedEngramScheduler::new(ChunkedEngramConfig::new(2));
        scheduler.seed(&[1, 2, 3], None);
        scheduler.push_token(4);
        scheduler.push_token(5);
        let request = scheduler.prepare_request(&client).unwrap();
        scheduler
            .resolve(&request, zero_shard_data(10, 800), &client, &device)
            .unwrap();

        assert!(scheduler.cached_embedding().is_some());
        assert!(scheduler.shard_cache_len() > 0);

        scheduler.reset();

        assert!(scheduler.cached_embedding().is_none());
        assert_eq!(scheduler.shard_cache_len(), 0);
        assert_eq!(scheduler.steps_since_refresh(), 0);
        assert!(scheduler.token_buffer.is_empty());
    }

    // ── Latent projection ───────────────────────────────────────────────

    #[test]
    fn prepare_request_with_latent_projection() {
        let client = test_client();
        let mut scheduler = ChunkedEngramScheduler::new(ChunkedEngramConfig::new(3));
        scheduler.seed(&[10, 20, 30, 40, 50], None);

        // Create a tiny latent projection matching EMBEDDING_DIM
        let proj = crate::latent_projection::LatentProjection::new_random(
            EMBEDDING_DIM, 8, 4, &Device::Cpu,
        ).unwrap();

        // Token buffer should have 5 tokens (max_window = chunk_size + 2 = 5)
        let token_buf: Vec<u32> = scheduler.token_buffer().iter().copied().collect();
        let dummy_embeddings = Tensor::randn(
            0f32, 1.0,
            (1, token_buf.len(), EMBEDDING_DIM),
            &Device::Cpu,
        ).unwrap();

        let request = scheduler
            .prepare_request_latent(&client, &proj, &dummy_embeddings)
            .unwrap();

        // Should have lookups (bigrams + trigrams from the 5-token window)
        assert!(!request.lookups.is_empty());
        assert_eq!(request.seq_len, token_buf.len());
    }
}
