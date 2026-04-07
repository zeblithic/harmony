//! UQ (Uncertainty Quantification) feature extraction.
//!
//! Extracts an 8-dimensional feature vector from per-layer L2 norms and output
//! logits. The features are consumed by [`crate::uq_head::UqHead`] for routing
//! decisions. All computation is pure floating-point — no tensor dependencies.
//!
//! # Features
//!
//! | Index | Description                                                       |
//! |-------|-------------------------------------------------------------------|
//! | f1    | L2 norm at `norm_layers[0]`                                       |
//! | f2    | L2 norm at `norm_layers[1]`                                       |
//! | f3    | L2 norm at `norm_layers[2]`                                       |
//! | f4    | L2 norm at `norm_layers[3]`                                       |
//! | f5    | Linear regression slope over f1–f4                                |
//! | f6    | Shannon entropy of logit distribution (after softmax)             |
//! | f7    | Top-k probability mass (after softmax, sorted descending)         |
//! | f8    | Stub: attention lookback (deferred, always 0.0)                   |

use crate::error::InferenceError;

/// Configuration for UQ feature extraction.
///
/// Specifies which layer indices to sample for L2 norm features (f1–f4)
/// and how many top probabilities to include in the top-k mass (f7).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UqFeatureConfig {
    /// Indices into the per-layer L2 norm slice for features f1–f4.
    /// Must have exactly 4 elements; each must be within bounds of the
    /// `layer_norms` slice passed to [`extract_uq_features`].
    pub norm_layers: [usize; 4],
    /// Number of top probabilities to sum for the top-k mass feature (f7).
    pub top_k_for_mass: usize,
}

impl Default for UqFeatureConfig {
    /// Default configuration for a 24-layer model (e.g. Qwen3-1.7B).
    ///
    /// Samples layers [0, 8, 16, 23] with top-10 mass.
    fn default() -> Self {
        Self {
            norm_layers: [0, 8, 16, 23],
            top_k_for_mass: 10,
        }
    }
}

impl UqFeatureConfig {
    /// Configuration for a small 8-layer model.
    ///
    /// Samples layers [0, 2, 5, 7] with top-10 mass.
    pub fn tiny() -> Self {
        Self {
            norm_layers: [0, 2, 5, 7],
            top_k_for_mass: 10,
        }
    }

    /// Derive a configuration from the number of transformer layers.
    ///
    /// | `num_layers` | `norm_layers`   |
    /// |--------------|-----------------|
    /// | 24           | [0, 8, 16, 23]  |
    /// | 8            | [0, 2, 5, 7]    |
    /// | other n      | evenly-spaced, clamped to [0, n-1] |
    ///
    /// The evenly-spaced formula is `[0, (n-1)/3, 2*(n-1)/3, n-1]` using
    /// integer division, then clamped so no index exceeds `n-1`.
    pub fn for_num_layers(num_layers: usize) -> Self {
        match num_layers {
            24 => Self::default(),
            8 => Self::tiny(),
            n => {
                let last = n.saturating_sub(1);
                let i1 = (last / 3).min(last);
                let i2 = (2 * last / 3).min(last);
                Self {
                    norm_layers: [0, i1, i2, last],
                    top_k_for_mass: 10,
                }
            }
        }
    }
}

/// Extract the 8-dimensional UQ feature vector.
///
/// # Arguments
///
/// * `layer_norms` — per-layer L2 norms from a forward pass, one per layer.
/// * `logits` — raw (pre-softmax) output logits from the final layer.
/// * `config` — which layers to sample and top-k parameter.
///
/// # Errors
///
/// Returns [`InferenceError::ForwardFailed`] if any index in
/// `config.norm_layers` is out of bounds for `layer_norms`.
pub fn extract_uq_features(
    layer_norms: &[f32],
    logits: &[f32],
    config: &UqFeatureConfig,
) -> Result<[f32; 8], InferenceError> {
    // Validate all norm_layer indices.
    for (pos, &idx) in config.norm_layers.iter().enumerate() {
        if idx >= layer_norms.len() {
            return Err(InferenceError::ForwardFailed(format!(
                "norm_layers[{pos}] = {idx} is out of bounds for layer_norms of length {}",
                layer_norms.len()
            )));
        }
    }

    // f1–f4: direct lookup.
    let f1 = layer_norms[config.norm_layers[0]];
    let f2 = layer_norms[config.norm_layers[1]];
    let f3 = layer_norms[config.norm_layers[2]];
    let f4 = layer_norms[config.norm_layers[3]];

    // f5: linear regression slope over f1–f4.
    // Slope = (3*(f4 - f1) + (f3 - f2)) / 10.0
    let f5 = (3.0 * (f4 - f1) + (f3 - f2)) / 10.0;

    // Compute softmax of logits for f6 and f7.
    let probs = softmax(logits);

    // f6: Shannon entropy — -sum(p * ln(p)).
    let f6 = shannon_entropy(&probs);

    // f7: top-k probability mass.
    let f7 = top_k_mass(&probs, config.top_k_for_mass);

    // f8: attention lookback (deferred).
    let f8 = 0.0_f32;

    Ok([f1, f2, f3, f4, f5, f6, f7, f8])
}

// ── Internal helpers ──────────────────────────────────────────────────────────

/// Compute the softmax of a slice of logits.
///
/// Uses the numerically stable max-subtraction trick. Returns an empty
/// `Vec` if `logits` is empty.
fn softmax(logits: &[f32]) -> Vec<f32> {
    if logits.is_empty() {
        return Vec::new();
    }
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|&x| (x - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.iter().map(|&e| e / sum).collect()
}

/// Shannon entropy of a probability distribution: `-sum(p * ln(p))`.
///
/// Values ≤ 0 are clamped to a small epsilon before taking the logarithm
/// to avoid -inf / NaN. The result is in nats.
fn shannon_entropy(probs: &[f32]) -> f32 {
    const EPS: f32 = 1e-10;
    probs
        .iter()
        .map(|&p| {
            let p = p.max(EPS);
            -p * p.ln()
        })
        .sum()
}

/// Sum the top-`k` largest probabilities (sorted descending).
///
/// If `k >= probs.len()`, returns the full sum (i.e. 1.0 for a valid
/// probability distribution). If `probs` is empty, returns 0.0.
fn top_k_mass(probs: &[f32], k: usize) -> f32 {
    if probs.is_empty() || k == 0 {
        return 0.0;
    }
    let mut sorted = probs.to_vec();
    // Sort descending (NaN-safe: treat NaN as smaller than any real value).
    sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Less));
    sorted.iter().take(k).sum()
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // Helper: build a config with custom norm_layers and default top_k_for_mass=10.
    fn cfg(norm_layers: [usize; 4]) -> UqFeatureConfig {
        UqFeatureConfig {
            norm_layers,
            top_k_for_mass: 10,
        }
    }

    // 1. Uniform logits → entropy ≈ ln(N).
    #[test]
    fn entropy_uniform_is_maximal() {
        let n = 100_usize;
        let logits = vec![0.0_f32; n]; // all equal → uniform after softmax
        let probs = softmax(&logits);
        let h = shannon_entropy(&probs);
        let expected = (n as f32).ln();
        assert!(
            (h - expected).abs() < 1e-4,
            "entropy {h} ≠ ln({n}) = {expected}"
        );
    }

    // 2. One dominant logit → entropy ≈ 0.
    #[test]
    fn entropy_one_hot_is_zero() {
        let mut logits = vec![0.0_f32; 100];
        logits[0] = 1000.0; // effectively one-hot after softmax
        let probs = softmax(&logits);
        let h = shannon_entropy(&probs);
        assert!(h < 1e-3, "entropy {h} should be near 0 for one-hot dist");
    }

    // 3. Peaked distribution → top-1 mass ≈ 1.0.
    #[test]
    fn top_k_mass_peaked() {
        let mut logits = vec![0.0_f32; 50];
        logits[0] = 1000.0;
        let probs = softmax(&logits);
        let mass = top_k_mass(&probs, 1);
        assert!(
            (mass - 1.0).abs() < 1e-4,
            "top-1 mass {mass} should be ≈ 1.0 for peaked dist"
        );
    }

    // 4. Uniform distribution → top-k mass = k/N.
    #[test]
    fn top_k_mass_uniform() {
        let n = 100_usize;
        let k = 20_usize;
        let logits = vec![0.0_f32; n];
        let probs = softmax(&logits);
        let mass = top_k_mass(&probs, k);
        let expected = k as f32 / n as f32;
        assert!(
            (mass - expected).abs() < 1e-4,
            "top-{k} mass {mass} ≠ {k}/{n} = {expected}"
        );
    }

    // 5. Increasing norms → positive slope.
    #[test]
    fn slope_positive_for_increasing_norms() {
        // 8-layer model; norm_layers = [0, 2, 5, 7]
        let layer_norms = vec![1.0_f32, 0.0, 2.0, 0.0, 0.0, 3.0, 0.0, 4.0];
        let config = UqFeatureConfig {
            norm_layers: [0, 2, 5, 7],
            top_k_for_mass: 1,
        };
        let logits = vec![1.0_f32; 4];
        let features = extract_uq_features(&layer_norms, &logits, &config).unwrap();
        let f5 = features[4];
        assert!(f5 > 0.0, "slope {f5} should be positive for norms [1,2,3,4]");
    }

    // 6. Decreasing norms → negative slope.
    #[test]
    fn slope_negative_for_decreasing_norms() {
        let layer_norms = vec![4.0_f32, 0.0, 3.0, 0.0, 0.0, 2.0, 0.0, 1.0];
        let config = UqFeatureConfig {
            norm_layers: [0, 2, 5, 7],
            top_k_for_mass: 1,
        };
        let logits = vec![1.0_f32; 4];
        let features = extract_uq_features(&layer_norms, &logits, &config).unwrap();
        let f5 = features[4];
        assert!(f5 < 0.0, "slope {f5} should be negative for norms [4,3,2,1]");
    }

    // 7. Constant norms → slope = 0.
    #[test]
    fn slope_zero_for_constant_norms() {
        let layer_norms = vec![5.0_f32, 0.0, 5.0, 0.0, 0.0, 5.0, 0.0, 5.0];
        let config = UqFeatureConfig {
            norm_layers: [0, 2, 5, 7],
            top_k_for_mass: 1,
        };
        let logits = vec![1.0_f32; 4];
        let features = extract_uq_features(&layer_norms, &logits, &config).unwrap();
        let f5 = features[4];
        assert!(
            f5.abs() < 1e-6,
            "slope {f5} should be 0 for constant norms"
        );
    }

    // 8. f8 is always 0.
    #[test]
    fn f8_always_zero() {
        let layer_norms = vec![1.0_f32, 2.0, 3.0, 4.0];
        let logits = vec![0.5_f32, 1.0, 2.0, 0.1];
        let config = cfg([0, 1, 2, 3]);
        let features = extract_uq_features(&layer_norms, &logits, &config).unwrap();
        assert_eq!(features[7], 0.0, "f8 must be 0.0 (stubbed)");
    }

    // 9. Out-of-bounds norm_layers index → returns error.
    #[test]
    fn norm_layers_out_of_bounds_returns_error() {
        let layer_norms = vec![1.0_f32, 2.0, 3.0]; // len = 3
        let logits = vec![0.0_f32; 10];
        let config = UqFeatureConfig {
            norm_layers: [0, 1, 2, 5], // index 5 is OOB
            top_k_for_mass: 5,
        };
        let result = extract_uq_features(&layer_norms, &logits, &config);
        assert!(
            matches!(result, Err(InferenceError::ForwardFailed(_))),
            "expected ForwardFailed, got {result:?}"
        );
    }

    // 10. Same inputs → same outputs (deterministic).
    #[test]
    fn features_are_deterministic() {
        let layer_norms = vec![1.1_f32, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8];
        let logits: Vec<f32> = (0..50).map(|i| i as f32 * 0.1).collect();
        let config = UqFeatureConfig {
            norm_layers: [0, 2, 5, 7],
            top_k_for_mass: 5,
        };
        let a = extract_uq_features(&layer_norms, &logits, &config).unwrap();
        let b = extract_uq_features(&layer_norms, &logits, &config).unwrap();
        assert_eq!(a, b, "feature extraction must be deterministic");
    }

    // 11. for_num_layers(24) → default [0, 8, 16, 23].
    #[test]
    fn for_num_layers_target() {
        let config = UqFeatureConfig::for_num_layers(24);
        assert_eq!(config.norm_layers, [0, 8, 16, 23]);
    }

    // 12. for_num_layers(8) → tiny [0, 2, 5, 7].
    #[test]
    fn for_num_layers_tiny() {
        let config = UqFeatureConfig::for_num_layers(8);
        assert_eq!(config.norm_layers, [0, 2, 5, 7]);
    }

    // 13. for_num_layers(4) → [0, 1, 2, 3].
    #[test]
    fn for_num_layers_4() {
        let config = UqFeatureConfig::for_num_layers(4);
        assert_eq!(config.norm_layers, [0, 1, 2, 3]);
    }
}
