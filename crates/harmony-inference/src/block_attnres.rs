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
}
