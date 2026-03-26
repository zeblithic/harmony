//! Speculative decoding verification logic.

/// Compute softmax probabilities from logits (numerically stable).
pub fn softmax(logits: &[f32]) -> Vec<f32> {
    if logits.is_empty() {
        return Vec::new();
    }
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|&x| (x - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    if sum == 0.0 {
        // Degenerate case: all logits are -inf
        return vec![0.0; logits.len()];
    }
    exps.into_iter().map(|e| e / sum).collect()
}

/// Check whether a draft token should be accepted (greedy criterion).
///
/// Accept if P_target(token) >= P_draft(token), where P_draft = exp(draft_logprob).
pub fn should_accept_draft(
    target_logits: &[f32],
    draft_token_id: u32,
    draft_logprob: f32,
) -> bool {
    let probs = softmax(target_logits);
    let p_target = probs.get(draft_token_id as usize).copied().unwrap_or(0.0);
    let p_draft = draft_logprob.exp();
    p_target >= p_draft
}

/// Sample the token with the highest probability (greedy) and return
/// its ID and log-probability.
pub fn sample_greedy_with_logprob(logits: &[f32]) -> (u32, f32) {
    let probs = softmax(logits);
    let (token_id, &p) = probs
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .expect("logits must not be empty");
    (token_id as u32, if p > 0.0 { p.ln() } else { f32::NEG_INFINITY })
}

/// Compute the log-probability of a specific token given logits.
pub fn logprob_of(logits: &[f32], token_id: u32) -> f32 {
    let probs = softmax(logits);
    let p = probs.get(token_id as usize).copied().unwrap_or(0.0);
    if p > 0.0 { p.ln() } else { f32::NEG_INFINITY }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn softmax_sums_to_one() {
        let logits = vec![1.0, 2.0, 3.0, 4.0];
        let probs = softmax(&logits);
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn softmax_empty() {
        assert!(softmax(&[]).is_empty());
    }

    #[test]
    fn softmax_single() {
        let probs = softmax(&[5.0]);
        assert_eq!(probs.len(), 1);
        assert!((probs[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn softmax_uniform() {
        let logits = vec![0.0; 4];
        let probs = softmax(&logits);
        for p in &probs {
            assert!((p - 0.25).abs() < 1e-6);
        }
    }

    #[test]
    fn accept_draft_high_target_prob() {
        // Draft token has logprob ln(0.3) ≈ -1.204
        // Target gives it probability 0.5 (> 0.3) → accept
        let mut logits = vec![0.0; 10];
        logits[3] = 2.0; // token 3 will have high probability
        let draft_logprob = (0.1f32).ln(); // draft gave it 0.1
        assert!(should_accept_draft(&logits, 3, draft_logprob));
    }

    #[test]
    fn reject_draft_low_target_prob() {
        // Draft token has high logprob but target gives it low probability
        let logits = vec![0.0; 10]; // uniform ≈ 0.1 each
        let draft_logprob = (0.9f32).ln(); // draft was very confident
        // P_target(token 3) ≈ 0.1 < 0.9 → reject
        assert!(!should_accept_draft(&logits, 3, draft_logprob));
    }

    #[test]
    fn accept_draft_equal_prob() {
        // When P_target == P_draft, should accept (>= criterion)
        let logits = vec![0.0; 4]; // uniform = 0.25 each
        let draft_logprob = (0.25f32).ln();
        assert!(should_accept_draft(&logits, 0, draft_logprob));
    }

    #[test]
    fn sample_greedy_picks_argmax() {
        let logits = vec![1.0, 5.0, 2.0, 0.5];
        let (token_id, logprob) = sample_greedy_with_logprob(&logits);
        assert_eq!(token_id, 1); // index 1 has highest logit
        assert!(logprob < 0.0); // logprob is negative (probability < 1)
        assert!(logprob > f32::NEG_INFINITY);
    }

    #[test]
    fn logprob_of_valid_token() {
        let logits = vec![0.0; 4]; // uniform = 0.25
        let lp = logprob_of(&logits, 2);
        assert!((lp - (0.25f32).ln()).abs() < 1e-5);
    }

    #[test]
    fn logprob_of_out_of_range() {
        let logits = vec![1.0, 2.0];
        let lp = logprob_of(&logits, 999);
        assert_eq!(lp, f32::NEG_INFINITY);
    }
}
