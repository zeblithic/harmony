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
}
