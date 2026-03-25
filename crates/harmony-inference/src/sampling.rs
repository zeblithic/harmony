//! Pure sampling functions — no model state needed.

use crate::error::InferenceError;
use rand::Rng;

/// Return the index of the maximum logit (argmax).
pub fn sample_greedy(logits: &[f32]) -> Result<u32, InferenceError> {
    if logits.is_empty() {
        return Err(InferenceError::SamplingFailed("empty logits".into()));
    }
    let (idx, _) = logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap(); // safe: non-empty checked above
    Ok(idx as u32)
}

/// Apply temperature scaling in place: `logits[i] /= temperature`.
///
/// Temperature must be > 0.0. Values <= 0.0 are treated as a no-op
/// (use greedy sampling instead for deterministic decoding).
pub fn apply_temperature(logits: &mut [f32], temperature: f32) {
    if temperature <= 0.0 {
        return;
    }
    for l in logits.iter_mut() {
        *l /= temperature;
    }
}

/// Keep only the `k` highest logits; set the rest to `-inf`.
///
/// If `k == 0` or `k >= logits.len()`, this is a no-op.
pub fn apply_top_k(logits: &mut [f32], k: u32) {
    if k == 0 || k as usize >= logits.len() {
        return;
    }
    let mut indexed: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
    indexed.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    for &(idx, _) in &indexed[k as usize..] {
        logits[idx] = f32::NEG_INFINITY;
    }
}

/// Apply nucleus (top-p) filtering: sort by probability, keep tokens whose
/// cumulative probability mass ≤ `top_p`, set the rest to `-inf`.
///
/// `top_p` must be in `(0.0, 1.0)` to have an effect.
/// Values `>= 1.0` or `<= 0.0` are no-ops.
pub fn apply_top_p(logits: &mut [f32], top_p: f32) {
    if top_p >= 1.0 || top_p <= 0.0 {
        return;
    }
    let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|&l| (l - max_logit).exp()).collect();
    let sum: f32 = exps.iter().sum();

    let mut indexed: Vec<(usize, f32)> = exps.iter().map(|&e| e / sum).enumerate().collect();
    indexed.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut cumsum = 0.0;
    let mut keep = Vec::new();
    for (idx, prob) in &indexed {
        cumsum += prob;
        keep.push(*idx);
        if cumsum >= top_p {
            break;
        }
    }

    let keep_set: std::collections::HashSet<usize> = keep.into_iter().collect();
    for (i, l) in logits.iter_mut().enumerate() {
        if !keep_set.contains(&i) {
            *l = f32::NEG_INFINITY;
        }
    }
}

/// Apply repeat penalty for unique tokens seen in context.
///
/// For each *unique* token in `context_tokens`:
/// - If its logit is positive, divide by `penalty`
/// - If its logit is negative, multiply by `penalty`
///
/// Tokens appearing multiple times in context are penalized only once,
/// matching the standard llama.cpp / HF transformers behavior.
pub fn apply_repeat_penalty(logits: &mut [f32], penalty: f32, context_tokens: &[u32]) {
    if penalty == 1.0 || penalty <= 0.0 {
        return;
    }
    let mut seen = std::collections::HashSet::new();
    for &token in context_tokens {
        if !seen.insert(token) {
            continue;
        }
        if let Some(logit) = logits.get_mut(token as usize) {
            if *logit > 0.0 {
                *logit /= penalty;
            } else {
                *logit *= penalty;
            }
        }
    }
}

/// Sample a token from logits using the full pipeline.
///
/// Order: repeat penalty → temperature → top-k → top-p → weighted random.
/// If `temperature == 0.0` (greedy), repeat penalty is still applied before
/// argmax to avoid repetitive output.
pub fn sample(
    logits: &[f32],
    params: &crate::SamplingParams,
    context_tokens: &[u32],
    rng: &mut impl Rng,
) -> Result<u32, InferenceError> {
    if logits.is_empty() {
        return Err(InferenceError::SamplingFailed("empty logits".into()));
    }

    let mut logits = logits.to_vec();

    // Repeat penalty is always applied — even in greedy mode — to prevent
    // repetitive output. This matches llama.cpp and HF transformers behavior.
    apply_repeat_penalty(&mut logits, params.repeat_penalty, context_tokens);

    if params.temperature == 0.0 {
        return sample_greedy(&logits);
    }

    apply_temperature(&mut logits, params.temperature);
    apply_top_k(&mut logits, params.top_k);
    apply_top_p(&mut logits, params.top_p);

    sample_from_distribution(&logits, rng)
}

/// Sample a token from a logit distribution using weighted random selection.
///
/// Applies softmax internally, then draws from the resulting distribution.
fn sample_from_distribution(logits: &[f32], rng: &mut impl Rng) -> Result<u32, InferenceError> {
    let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if max_logit == f32::NEG_INFINITY {
        return Err(InferenceError::SamplingFailed(
            "all logits are -inf after filtering".into(),
        ));
    }

    let exps: Vec<f32> = logits.iter().map(|&l| (l - max_logit).exp()).collect();
    let sum: f32 = exps.iter().sum();

    let threshold = rng.gen::<f32>() * sum;
    let mut cumsum = 0.0;
    for (i, &exp) in exps.iter().enumerate() {
        cumsum += exp;
        if cumsum >= threshold {
            return Ok(i as u32);
        }
    }
    Ok((logits.len() - 1) as u32)
}

#[cfg(test)]
mod tests {
    use super::*;

    use rand::rngs::StdRng;
    use rand::SeedableRng;

    #[test]
    fn test_greedy_returns_argmax() {
        let logits = [1.0_f32, 3.0, 2.0, 0.5];
        assert_eq!(sample_greedy(&logits).unwrap(), 1);
    }

    #[test]
    fn test_greedy_first_max_on_tie() {
        let logits = [5.0_f32, 5.0, 1.0];
        let result = sample_greedy(&logits).unwrap();
        assert!(result == 0 || result == 1);
        assert_eq!(logits[result as usize], 5.0);
    }

    #[test]
    fn test_greedy_empty_logits() {
        let result = sample_greedy(&[]);
        assert!(matches!(result, Err(InferenceError::SamplingFailed(_))));
    }

    #[test]
    fn test_temperature_scales_logits() {
        let mut logits = [2.0_f32, 4.0, 6.0];
        apply_temperature(&mut logits, 2.0);
        assert_eq!(logits, [1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_temperature_one_is_identity() {
        let mut logits = [1.5_f32, 2.5, 3.5];
        let original = logits;
        apply_temperature(&mut logits, 1.0);
        assert_eq!(logits, original);
    }

    #[test]
    fn test_top_k_keeps_highest() {
        let mut logits = [1.0_f32, 4.0, 2.0, 3.0];
        apply_top_k(&mut logits, 2);
        assert_eq!(logits[0], f32::NEG_INFINITY);
        assert_eq!(logits[1], 4.0);
        assert_eq!(logits[2], f32::NEG_INFINITY);
        assert_eq!(logits[3], 3.0);
    }

    #[test]
    fn test_top_k_zero_is_noop() {
        let mut logits = [1.0_f32, 2.0, 3.0];
        let original = logits;
        apply_top_k(&mut logits, 0);
        assert_eq!(logits, original);
    }

    #[test]
    fn test_top_k_one_keeps_max_only() {
        let mut logits = [1.0_f32, 5.0, 3.0];
        apply_top_k(&mut logits, 1);
        assert_eq!(logits[0], f32::NEG_INFINITY);
        assert_eq!(logits[1], 5.0);
        assert_eq!(logits[2], f32::NEG_INFINITY);
    }

    #[test]
    fn test_top_k_ge_len_is_noop() {
        let mut logits = [1.0_f32, 2.0];
        let original = logits;
        apply_top_k(&mut logits, 5);
        assert_eq!(logits, original);
    }

    #[test]
    fn test_top_p_filters_low_probability() {
        let mut logits = [10.0_f32, 1.0, 0.0];
        apply_top_p(&mut logits, 0.9);
        assert_eq!(logits[0], 10.0);
        assert_eq!(logits[1], f32::NEG_INFINITY);
        assert_eq!(logits[2], f32::NEG_INFINITY);
    }

    #[test]
    fn test_top_p_one_is_noop() {
        let mut logits = [1.0_f32, 2.0, 3.0];
        let original = logits;
        apply_top_p(&mut logits, 1.0);
        assert_eq!(logits, original);
    }

    #[test]
    fn test_top_p_keeps_enough_mass() {
        let mut logits = [1.0_f32, 1.0, 1.0, 1.0];
        apply_top_p(&mut logits, 0.6);
        let kept = logits.iter().filter(|&&l| l != f32::NEG_INFINITY).count();
        assert!(kept >= 3, "should keep at least 3 tokens, kept {kept}");
    }

    #[test]
    fn test_repeat_penalty_reduces_positive_logits() {
        let mut logits = [1.0_f32, 4.0, 2.0];
        apply_repeat_penalty(&mut logits, 2.0, &[1]);
        assert_eq!(logits[0], 1.0);
        assert_eq!(logits[1], 2.0); // 4.0 / 2.0
        assert_eq!(logits[2], 2.0);
    }

    #[test]
    fn test_repeat_penalty_amplifies_negative_logits() {
        let mut logits = [1.0_f32, -2.0, 3.0];
        apply_repeat_penalty(&mut logits, 2.0, &[1]);
        assert_eq!(logits[1], -4.0); // -2.0 * 2.0
    }

    #[test]
    fn test_repeat_penalty_one_is_noop() {
        let mut logits = [1.0_f32, 2.0, 3.0];
        let original = logits;
        apply_repeat_penalty(&mut logits, 1.0, &[0, 1, 2]);
        assert_eq!(logits, original);
    }

    #[test]
    fn test_repeat_penalty_out_of_bounds_ignored() {
        let mut logits = [1.0_f32, 2.0];
        let original = logits;
        apply_repeat_penalty(&mut logits, 2.0, &[99]);
        assert_eq!(logits, original);
    }

    #[test]
    fn test_repeat_penalty_deduplicates_context() {
        // Token 1 appears 3 times — penalty should apply only once
        let mut logits = [1.0_f32, 8.0, 2.0];
        apply_repeat_penalty(&mut logits, 2.0, &[1, 1, 1]);
        assert_eq!(logits[1], 4.0); // 8.0 / 2.0 (once, not 8.0 / 8.0)
    }

    #[test]
    fn test_top_p_zero_is_noop() {
        let mut logits = [1.0_f32, 2.0, 3.0];
        let original = logits;
        apply_top_p(&mut logits, 0.0);
        assert_eq!(logits, original);
    }

    #[test]
    fn test_sample_greedy_applies_repeat_penalty() {
        // Without repeat penalty, token 1 (logit=5.0) wins.
        // With repeat penalty=10.0 on token 1, logit becomes 0.5, so token 2 (3.0) wins.
        let logits = [1.0_f32, 5.0, 3.0];
        let params = crate::SamplingParams {
            temperature: 0.0,
            repeat_penalty: 10.0,
            ..crate::SamplingParams::greedy()
        };
        let mut rng = StdRng::seed_from_u64(42);
        let result = sample(&logits, &params, &[1], &mut rng).unwrap();
        assert_eq!(result, 2, "greedy should respect repeat penalty");
    }

    #[test]
    fn test_sample_temperature_zero_is_greedy() {
        let logits = [1.0_f32, 5.0, 3.0];
        let params = crate::SamplingParams::greedy();
        let mut rng = StdRng::seed_from_u64(42);
        let result = sample(&logits, &params, &[], &mut rng).unwrap();
        assert_eq!(result, 1);
    }

    #[test]
    fn test_sample_returns_valid_token_id() {
        let logits = [1.0_f32, 2.0, 3.0, 4.0];
        let params = crate::SamplingParams::default();
        let mut rng = StdRng::seed_from_u64(42);
        let result = sample(&logits, &params, &[], &mut rng).unwrap();
        assert!(result < 4, "token ID {result} out of range");
    }

    #[test]
    fn test_sample_empty_logits_error() {
        let params = crate::SamplingParams::default();
        let mut rng = StdRng::seed_from_u64(42);
        let result = sample(&[], &params, &[], &mut rng);
        assert!(matches!(result, Err(InferenceError::SamplingFailed(_))));
    }

    #[test]
    fn test_sample_deterministic_with_same_seed() {
        let logits = [1.0_f32, 2.0, 3.0, 4.0, 5.0];
        let params = crate::SamplingParams {
            temperature: 0.8,
            top_p: 0.9,
            top_k: 3,
            repeat_penalty: 1.0,
            repeat_last_n: 64,
        };
        let r1 = sample(&logits, &params, &[], &mut StdRng::seed_from_u64(123)).unwrap();
        let r2 = sample(&logits, &params, &[], &mut StdRng::seed_from_u64(123)).unwrap();
        assert_eq!(r1, r2, "same seed must produce same result");
    }
}
