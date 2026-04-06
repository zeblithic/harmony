//! Block Attention Residuals: learned depth-wise attention at block boundaries.
//!
//! Partitions transformer layers into N contiguous blocks. Within each block,
//! standard additive residuals operate. At block boundaries, a softmax-weighted
//! attention over previous block summaries allows deep layers to selectively
//! recall early-layer features, solving PreNorm dilution.
//!
//! # Usage
//!
//! ```ignore
//! let config = BlockAttnResConfig { num_blocks: 8, layers_per_block: 3, hidden_dim: 1280 };
//! let attnres = BlockAttnRes::new(&config, &device)?;
//! // In the forward loop, call attnres.layer_output() after each layer
//! // and attnres.block_input() at each block boundary.
//! ```

use candle_core::{Device, Result, Tensor};

/// Configuration for Block Attention Residuals.
#[derive(Debug, Clone)]
pub struct BlockAttnResConfig {
    /// Number of blocks to partition layers into.
    pub num_blocks: usize,
    /// Number of transformer layers per block.
    pub layers_per_block: usize,
    /// Hidden dimension of the transformer.
    pub hidden_dim: usize,
}

impl BlockAttnResConfig {
    /// Total number of transformer layers.
    pub fn total_layers(&self) -> usize {
        self.num_blocks * self.layers_per_block
    }

    /// Returns which block a given layer index belongs to.
    pub fn block_of(&self, layer_idx: usize) -> usize {
        layer_idx / self.layers_per_block
    }

    /// Returns true if layer_idx is the first layer of a block (a boundary).
    pub fn is_block_start(&self, layer_idx: usize) -> bool {
        layer_idx % self.layers_per_block == 0
    }
}

/// Mutable state accumulated during a single forward pass.
///
/// Created fresh for each forward pass via [`BlockAttnRes::new_state()`].
/// Tracks block summaries and the running partial sum within the current block.
pub struct BlockAttnResState {
    /// Completed block summaries. `summaries[k]` is the hidden state at the
    /// end of block k. Grows as blocks complete.
    summaries: Vec<Tensor>,
    /// Accumulated residual within the current (incomplete) block.
    /// Reset at each block boundary.
    partial_sum: Option<Tensor>,
}

/// Block Attention Residuals module.
///
/// Holds the learned pseudo-query vectors (one per block boundary).
/// The forward-pass state is tracked separately in [`BlockAttnResState`]
/// so the module itself is stateless and reusable across sequences.
pub struct BlockAttnRes {
    /// Learned pseudo-query vectors, one per boundary.
    /// `queries[k]` is used at the boundary before block k+1.
    /// Length: num_blocks - 1 (block 0 has no preceding boundary).
    queries: Vec<Tensor>,
    config: BlockAttnResConfig,
}

impl BlockAttnRes {
    /// Create with random pseudo-query vectors (for testing / fresh init).
    pub fn new(config: &BlockAttnResConfig, device: &Device) -> Result<Self> {
        let mut queries = Vec::with_capacity(config.num_blocks.saturating_sub(1));
        for _ in 0..config.num_blocks.saturating_sub(1) {
            // Small random init — will be trained to meaningful values
            let q = (Tensor::randn(0f32, 1f32, (1, 1, config.hidden_dim), device)?
                * (1.0 / (config.hidden_dim as f64).sqrt()))?;
            queries.push(q);
        }
        Ok(Self {
            queries,
            config: config.clone(),
        })
    }

    /// Create from pre-loaded query tensors (for loading trained weights).
    ///
    /// `query_tensors`: Vec of tensors, each shape `[1, 1, hidden_dim]`.
    /// Length must be `num_blocks - 1`.
    pub fn from_tensors(
        config: &BlockAttnResConfig,
        query_tensors: Vec<Tensor>,
    ) -> Result<Self> {
        if query_tensors.len() != config.num_blocks.saturating_sub(1) {
            candle_core::bail!(
                "expected {} query tensors, got {}",
                config.num_blocks.saturating_sub(1),
                query_tensors.len()
            );
        }
        Ok(Self {
            queries: query_tensors,
            config: config.clone(),
        })
    }

    /// Create a fresh state for a new forward pass.
    pub fn new_state(&self) -> BlockAttnResState {
        BlockAttnResState {
            summaries: Vec::with_capacity(self.config.num_blocks),
            partial_sum: None,
        }
    }

    /// Called after each transformer layer completes.
    ///
    /// Accumulates the layer's hidden state into the current block's partial
    /// sum. When the last layer of a block completes, finalizes the block
    /// summary and resets the partial sum for the next block.
    ///
    /// # Arguments
    /// - `layer_idx`: 0-based index of the layer that just completed
    /// - `hidden_state`: the layer's output tensor `[batch, seq_len, hidden_dim]`
    /// - `state`: mutable forward-pass state
    pub fn notify_layer_output(
        &self,
        layer_idx: usize,
        hidden_state: &Tensor,
        state: &mut BlockAttnResState,
    ) -> Result<()> {
        // Accumulate into partial sum
        state.partial_sum = Some(match &state.partial_sum {
            Some(prev) => (prev + hidden_state)?,
            None => hidden_state.clone(),
        });

        // Check if this is the last layer in the current block
        let is_last_in_block = (layer_idx + 1) % self.config.layers_per_block == 0;
        if is_last_in_block {
            // Finalize: move partial_sum into summaries
            let summary = state
                .partial_sum
                .take()
                .expect("partial_sum must exist after accumulation");
            state.summaries.push(summary);
        }

        Ok(())
    }

    /// Compute the input to a block using depth-wise attention over previous
    /// block summaries.
    ///
    /// For block 0 (no preceding blocks), returns `hidden_state` unchanged.
    /// For block k > 0, computes:
    ///   `input = sum_{j=0..k-1} alpha_j * summary_j + alpha_k * hidden_state`
    ///   where `alpha = softmax(query_k . [summary_0, ..., summary_{k-1}, hidden_state])`
    ///
    /// # Arguments
    /// - `block_idx`: 0-based block index
    /// - `hidden_state`: current hidden state `[batch, seq_len, hidden_dim]`
    /// - `state`: forward-pass state with completed block summaries
    pub fn block_input(
        &self,
        block_idx: usize,
        hidden_state: &Tensor,
        state: &BlockAttnResState,
    ) -> Result<Tensor> {
        // Block 0 has no preceding blocks — pass through unchanged
        if block_idx == 0 {
            return Ok(hidden_state.clone());
        }

        // query_idx: block 1 uses queries[0], block 2 uses queries[1], etc.
        let query_idx = block_idx - 1;
        let query = &self.queries[query_idx]; // [1, 1, hidden_dim]

        // Collect candidates: all completed summaries + current hidden_state
        let num_candidates = state.summaries.len() + 1;

        // Compute attention scores: dot(query, candidate) for each candidate
        let mut scores = Vec::with_capacity(num_candidates);
        for summary in &state.summaries {
            // [batch, seq_len, hidden_dim] * [1, 1, hidden_dim] → sum → [batch, seq_len, 1]
            let score = summary.broadcast_mul(query)?.sum_keepdim(candle_core::D::Minus1)?;
            scores.push(score);
        }
        // Current hidden state as final candidate
        let current_score = hidden_state.broadcast_mul(query)?.sum_keepdim(candle_core::D::Minus1)?;
        scores.push(current_score);

        // Stack scores: [batch, seq_len, num_candidates]
        let score_refs: Vec<&Tensor> = scores.iter().collect();
        let stacked_scores = Tensor::cat(&score_refs, 2)?;

        // Softmax over candidates dimension (last dim)
        let weights = candle_nn::ops::softmax_last_dim(&stacked_scores)?;

        // Weighted sum of candidates
        let mut result = Tensor::zeros_like(hidden_state)?;
        for (i, summary) in state.summaries.iter().enumerate() {
            let w = weights.narrow(2, i, 1)?; // [batch, seq_len, 1]
            result = (result + w.broadcast_mul(summary)?)?;
        }
        // Add weighted current hidden state
        let w_current = weights.narrow(2, num_candidates - 1, 1)?;
        result = (result + w_current.broadcast_mul(hidden_state)?)?;

        Ok(result)
    }

    /// Reference to the config.
    pub fn config(&self) -> &BlockAttnResConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config() -> BlockAttnResConfig {
        BlockAttnResConfig {
            num_blocks: 8,
            layers_per_block: 3,
            hidden_dim: 64,
        }
    }

    #[test]
    fn config_total_layers() {
        let cfg = test_config();
        assert_eq!(cfg.total_layers(), 24);
    }

    #[test]
    fn config_block_of() {
        let cfg = test_config();
        assert_eq!(cfg.block_of(0), 0);
        assert_eq!(cfg.block_of(2), 0);
        assert_eq!(cfg.block_of(3), 1);
        assert_eq!(cfg.block_of(23), 7);
    }

    #[test]
    fn config_is_block_start() {
        let cfg = test_config();
        assert!(cfg.is_block_start(0));
        assert!(!cfg.is_block_start(1));
        assert!(!cfg.is_block_start(2));
        assert!(cfg.is_block_start(3));
        assert!(cfg.is_block_start(21));
        assert!(!cfg.is_block_start(22));
    }

    #[test]
    fn new_creates_correct_query_count() {
        let cfg = test_config();
        let module = BlockAttnRes::new(&cfg, &Device::Cpu).unwrap();
        // 8 blocks → 7 boundaries → 7 query vectors
        assert_eq!(module.queries.len(), 7);
    }

    #[test]
    fn new_state_starts_empty() {
        let cfg = test_config();
        let module = BlockAttnRes::new(&cfg, &Device::Cpu).unwrap();
        let state = module.new_state();
        assert!(state.summaries.is_empty());
        assert!(state.partial_sum.is_none());
    }

    #[test]
    fn from_tensors_validates_count() {
        let cfg = test_config();
        // Too few tensors
        let result = BlockAttnRes::from_tensors(&cfg, vec![]);
        assert!(result.is_err());

        // Correct count
        let tensors: Vec<Tensor> = (0..7)
            .map(|_| Tensor::zeros((1, 1, 64), candle_core::DType::F32, &Device::Cpu).unwrap())
            .collect();
        let result = BlockAttnRes::from_tensors(&cfg, tensors);
        assert!(result.is_ok());
    }

    #[test]
    fn notify_accumulates_partial_sum() {
        let cfg = BlockAttnResConfig {
            num_blocks: 2,
            layers_per_block: 2,
            hidden_dim: 4,
        };
        let module = BlockAttnRes::new(&cfg, &Device::Cpu).unwrap();
        let mut state = module.new_state();

        // Layer 0 output: [1, 3, 4] (batch=1, seq=3, dim=4)
        let h0 = Tensor::ones((1, 3, 4), candle_core::DType::F32, &Device::Cpu).unwrap();
        module.notify_layer_output(0, &h0, &mut state).unwrap();

        // Partial sum should exist but no summaries yet (block 0 not done)
        assert!(state.partial_sum.is_some());
        assert!(state.summaries.is_empty());
    }

    #[test]
    fn notify_finalizes_block_summary() {
        let cfg = BlockAttnResConfig {
            num_blocks: 2,
            layers_per_block: 2,
            hidden_dim: 4,
        };
        let module = BlockAttnRes::new(&cfg, &Device::Cpu).unwrap();
        let mut state = module.new_state();

        let h = Tensor::ones((1, 3, 4), candle_core::DType::F32, &Device::Cpu).unwrap();

        // Layer 0 (block 0, position 0)
        module.notify_layer_output(0, &h, &mut state).unwrap();
        assert!(state.summaries.is_empty());

        // Layer 1 (block 0, position 1 — last in block)
        module.notify_layer_output(1, &h, &mut state).unwrap();
        // Block 0 should now be finalized
        assert_eq!(state.summaries.len(), 1);
        // Partial sum should be reset for the next block
        assert!(state.partial_sum.is_none());
    }

    #[test]
    fn block_input_at_block_0_returns_hidden_state() {
        let cfg = BlockAttnResConfig {
            num_blocks: 2,
            layers_per_block: 2,
            hidden_dim: 4,
        };
        let module = BlockAttnRes::new(&cfg, &Device::Cpu).unwrap();
        let state = module.new_state();

        let h = Tensor::ones((1, 3, 4), candle_core::DType::F32, &Device::Cpu).unwrap();
        // Block 0 has no preceding blocks — should return input unchanged
        let result = module.block_input(0, &h, &state).unwrap();
        assert_eq!(result.dims(), &[1, 3, 4]);

        // Should be identical to input
        let diff: f32 = (&result - &h)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar()
            .unwrap();
        assert!(diff < 1e-6);
    }

    #[test]
    fn block_input_at_block_1_uses_attention() {
        let cfg = BlockAttnResConfig {
            num_blocks: 2,
            layers_per_block: 2,
            hidden_dim: 4,
        };
        let module = BlockAttnRes::new(&cfg, &Device::Cpu).unwrap();
        let mut state = module.new_state();

        let h = Tensor::ones((1, 3, 4), candle_core::DType::F32, &Device::Cpu).unwrap();

        // Complete block 0
        module.notify_layer_output(0, &h, &mut state).unwrap();
        module.notify_layer_output(1, &h, &mut state).unwrap();
        assert_eq!(state.summaries.len(), 1);

        // Block 1 input: should be a weighted combination of block 0 summary + current h
        let result = module.block_input(1, &h, &state).unwrap();
        assert_eq!(result.dims(), &[1, 3, 4]);
    }

    #[test]
    fn block_input_preserves_shape_through_all_blocks() {
        let cfg = BlockAttnResConfig {
            num_blocks: 4,
            layers_per_block: 2,
            hidden_dim: 8,
        };
        let module = BlockAttnRes::new(&cfg, &Device::Cpu).unwrap();
        let mut state = module.new_state();

        let h = Tensor::randn(0f32, 1f32, (1, 5, 8), &Device::Cpu).unwrap();

        for block_idx in 0..4 {
            let input = module.block_input(block_idx, &h, &state).unwrap();
            assert_eq!(input.dims(), &[1, 5, 8], "block {block_idx} input shape mismatch");

            // Simulate layers within this block
            let start = block_idx * 2;
            for layer in start..start + 2 {
                module.notify_layer_output(layer, &h, &mut state).unwrap();
            }
        }
        assert_eq!(state.summaries.len(), 4);
    }

    #[test]
    fn block_input_is_convex_combination() {
        // The output should be a weighted average of the candidates.
        // Verify by checking it falls within the bounding box of the candidates.
        let cfg = BlockAttnResConfig {
            num_blocks: 3,
            layers_per_block: 1,
            hidden_dim: 4,
        };
        let module = BlockAttnRes::new(&cfg, &Device::Cpu).unwrap();
        let mut state = module.new_state();

        // Block 0: all zeros
        let h0 = Tensor::zeros((1, 1, 4), candle_core::DType::F32, &Device::Cpu).unwrap();
        module.notify_layer_output(0, &h0, &mut state).unwrap();

        // Block 1: all ones
        let h1 = Tensor::ones((1, 1, 4), candle_core::DType::F32, &Device::Cpu).unwrap();
        module.notify_layer_output(1, &h1, &mut state).unwrap();

        // Block 2 input with hidden_state = all twos
        let h2 = (Tensor::ones((1, 1, 4), candle_core::DType::F32, &Device::Cpu).unwrap() * 2.0)
            .unwrap();
        let result = module.block_input(2, &h2, &state).unwrap();
        let result_vals: Vec<f32> = result.flatten_all().unwrap().to_vec1().unwrap();

        // Candidates are: [0,0,0,0], [1,1,1,1], [2,2,2,2]
        // Convex combination must be in [0, 2] for each element
        for (i, &v) in result_vals.iter().enumerate() {
            assert!(
                v >= -0.01 && v <= 2.01,
                "element {i} = {v} is outside convex hull [0, 2]"
            );
        }
    }

    #[test]
    fn zero_query_produces_uniform_weights() {
        // When the pseudo-query is zero, all dot products are zero,
        // so softmax produces uniform weights = 1/N for each candidate.
        let cfg = BlockAttnResConfig {
            num_blocks: 3,
            layers_per_block: 1,
            hidden_dim: 4,
        };
        let zero_queries: Vec<Tensor> = (0..2)
            .map(|_| {
                Tensor::zeros((1, 1, 4), candle_core::DType::F32, &Device::Cpu).unwrap()
            })
            .collect();
        let module = BlockAttnRes::from_tensors(&cfg, zero_queries).unwrap();
        let mut state = module.new_state();

        // Block 0 summary: all zeros
        let h0 = Tensor::zeros((1, 1, 4), candle_core::DType::F32, &Device::Cpu).unwrap();
        module.notify_layer_output(0, &h0, &mut state).unwrap();

        // Block 1 summary: all threes
        let h1 = (Tensor::ones((1, 1, 4), candle_core::DType::F32, &Device::Cpu).unwrap() * 3.0)
            .unwrap();
        module.notify_layer_output(1, &h1, &mut state).unwrap();

        // Block 2 input with hidden_state = all sixes
        let h2 = (Tensor::ones((1, 1, 4), candle_core::DType::F32, &Device::Cpu).unwrap() * 6.0)
            .unwrap();
        let result = module.block_input(2, &h2, &state).unwrap();
        let result_vals: Vec<f32> = result.flatten_all().unwrap().to_vec1().unwrap();

        // Uniform weights = 1/3 for each of [0, 3, 6] → expected = 3.0
        for (i, &v) in result_vals.iter().enumerate() {
            assert!(
                (v - 3.0).abs() < 1e-5,
                "element {i} = {v}, expected 3.0 (uniform average of 0, 3, 6)"
            );
        }
    }

    #[test]
    fn batch_and_seq_len_handled_independently() {
        let cfg = BlockAttnResConfig {
            num_blocks: 2,
            layers_per_block: 1,
            hidden_dim: 4,
        };
        let module = BlockAttnRes::new(&cfg, &Device::Cpu).unwrap();
        let mut state = module.new_state();

        // batch=2, seq_len=3
        let h0 = Tensor::randn(0f32, 1f32, (2, 3, 4), &Device::Cpu).unwrap();
        module.notify_layer_output(0, &h0, &mut state).unwrap();

        let h1 = Tensor::randn(0f32, 1f32, (2, 3, 4), &Device::Cpu).unwrap();
        let result = module.block_input(1, &h1, &state).unwrap();

        // Output shape must match input
        assert_eq!(result.dims(), &[2, 3, 4]);
    }

    #[test]
    fn single_block_config_has_no_queries() {
        let cfg = BlockAttnResConfig {
            num_blocks: 1,
            layers_per_block: 24,
            hidden_dim: 64,
        };
        let module = BlockAttnRes::new(&cfg, &Device::Cpu).unwrap();
        assert_eq!(module.queries.len(), 0);

        let mut state = module.new_state();
        let h = Tensor::ones((1, 1, 64), candle_core::DType::F32, &Device::Cpu).unwrap();

        // Block 0 always passes through
        let result = module.block_input(0, &h, &state).unwrap();
        let diff: f32 = (&result - &h)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar()
            .unwrap();
        assert!(diff < 1e-6);

        // All 24 layers go into one block
        for i in 0..24 {
            module.notify_layer_output(i, &h, &mut state).unwrap();
        }
        assert_eq!(state.summaries.len(), 1);
    }

    /// Simulates a complete forward pass through a 24-layer, 8-block transformer.
    #[test]
    fn full_forward_pass_simulation() {
        let cfg = BlockAttnResConfig {
            num_blocks: 8,
            layers_per_block: 3,
            hidden_dim: 64,
        };
        let module = BlockAttnRes::new(&cfg, &Device::Cpu).unwrap();
        let mut state = module.new_state();

        let batch = 1;
        let seq_len = 10;
        let mut h = Tensor::randn(0f32, 1f32, (batch, seq_len, 64), &Device::Cpu).unwrap();

        for layer_idx in 0..24 {
            // At block boundaries (except block 0), compute weighted input
            if cfg.is_block_start(layer_idx) {
                let block_idx = cfg.block_of(layer_idx);
                h = module.block_input(block_idx, &h, &state).unwrap();
            }

            // Simulate transformer layer (just add noise — we're testing AttnRes, not the transformer)
            let layer_out =
                Tensor::randn(0f32, 0.1f32, (batch, seq_len, 64), &Device::Cpu).unwrap();
            h = (&h + &layer_out).unwrap();

            // Notify AttnRes of this layer's output
            module.notify_layer_output(layer_idx, &h, &mut state).unwrap();
        }

        // All 8 blocks should be finalized
        assert_eq!(state.summaries.len(), 8);
        // Partial sum should be None (last block was finalized)
        assert!(state.partial_sum.is_none());
        // Output shape should be preserved
        assert_eq!(h.dims(), &[batch, seq_len, 64]);
    }

    /// Verifies that block 7 can "see" block 0's summary (the Engram-enriched state).
    #[test]
    fn deep_block_can_recall_early_block() {
        let cfg = BlockAttnResConfig {
            num_blocks: 4,
            layers_per_block: 1,
            hidden_dim: 4,
        };
        // Craft a query that strongly prefers the first candidate (block 0 summary)
        // by making it point in the same direction as block 0's data.
        let block0_direction =
            Tensor::new(&[1.0f32, 0.0, 0.0, 0.0], &Device::Cpu).unwrap()
                .reshape((1, 1, 4)).unwrap();
        let neutral_query =
            Tensor::zeros((1, 1, 4), candle_core::DType::F32, &Device::Cpu).unwrap();

        // queries[0] for block 1: neutral
        // queries[1] for block 2: neutral
        // queries[2] for block 3: strongly prefers direction [1,0,0,0]
        let queries = vec![
            neutral_query.clone(),
            neutral_query,
            (block0_direction * 10.0).unwrap(), // large magnitude → strong preference
        ];
        let module = BlockAttnRes::from_tensors(&cfg, queries).unwrap();
        let mut state = module.new_state();

        // Block 0 summary: [10, 0, 0, 0] — high component in direction [1,0,0,0]
        let h0 = Tensor::new(&[10.0f32, 0.0, 0.0, 0.0], &Device::Cpu)
            .unwrap()
            .reshape((1, 1, 4))
            .unwrap();
        module.notify_layer_output(0, &h0, &mut state).unwrap();

        // Block 1 summary: [0, 5, 0, 0] — orthogonal
        let h1 = Tensor::new(&[0.0f32, 5.0, 0.0, 0.0], &Device::Cpu)
            .unwrap()
            .reshape((1, 1, 4))
            .unwrap();
        module.notify_layer_output(1, &h1, &mut state).unwrap();

        // Block 2 summary: [0, 0, 5, 0] — orthogonal
        let h2 = Tensor::new(&[0.0f32, 0.0, 5.0, 0.0], &Device::Cpu)
            .unwrap()
            .reshape((1, 1, 4))
            .unwrap();
        module.notify_layer_output(2, &h2, &mut state).unwrap();

        // Block 3 input with hidden_state [0, 0, 0, 5]
        let h3 = Tensor::new(&[0.0f32, 0.0, 0.0, 5.0], &Device::Cpu)
            .unwrap()
            .reshape((1, 1, 4))
            .unwrap();
        let result = module.block_input(3, &h3, &state).unwrap();
        let vals: Vec<f32> = result.flatten_all().unwrap().to_vec1().unwrap();

        // The query strongly aligns with block 0's direction [1,0,0,0].
        // Block 0's summary [10,0,0,0] has dot product 10*10=100 with the query.
        // Others have dot product ≈ 0.
        // So the result should be heavily weighted toward block 0.
        // vals[0] should be much larger than vals[1], vals[2], vals[3]
        assert!(
            vals[0] > vals[1] && vals[0] > vals[2] && vals[0] > vals[3],
            "block 3 should strongly recall block 0; got {:?}",
            vals
        );
    }
}
