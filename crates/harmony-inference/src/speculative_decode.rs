//! Uncertainty-aware speculative decode scheduler (ZEB-60).
//!
//! Sans-I/O state machine that consumes UQ sigmoid confidence scores and
//! produces accept/reject/halt decisions for speculative decoding. Uses a
//! tripartite threshold system:
//!
//! - **tau_generate**: below this, skip speculation entirely (context too chaotic)
//! - **tau_chain**: cumulative product of draft confidences; stop drafting when below
//! - **tau_accept**: above this, accept draft WITHOUT full-model verification (ConfSpec bypass)
//!
//! The scheduler does not perform forward passes or sampling — the caller
//! orchestrates drafting, verification, and token emission.
//!
//! # Usage
//!
//! ```ignore
//! let mut scheduler = SpeculativeDecodeScheduler::new(SpecDecConfig::default());
//!
//! // After forward_full + classify_uncertainty:
//! let context = SpecContext {
//!     is_thinking: false,
//!     engram_steps_remaining: Some(engram_scheduler.steps_until_boundary()),
//! };
//!
//! if scheduler.begin_draft(confidence, &context) {
//!     // Draft loop
//!     loop {
//!         // ... run forward, get draft_token and draft_confidence ...
//!         if scheduler.push_draft(draft_token, draft_confidence)? == DraftAction::Halt {
//!             break;
//!         }
//!     }
//!     let bypass = scheduler.bypass_indices();
//!     // ... verify draft tokens against full model ...
//!     scheduler.complete(accepted_count)?;
//! }
//! ```
//!
//! # DraftTree
//!
//! Separate utility for generating attention masks for parallel verification.
//! Supports linear chains (current) and branching trees (future: TALON-style).
//!
//! References: ConfSpec (arXiv 2602.18447), TALON (arXiv 2601.07353)

use candle_core::{Device, Tensor};

use crate::error::InferenceError;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for uncertainty-aware speculative decoding.
///
/// Three thresholds control the speculation lifecycle:
/// - `tau_generate` gates entry (skip if confidence too low)
/// - `tau_chain` gates draft continuation (cumulative product)
/// - `tau_accept` gates verification bypass (ConfSpec)
#[derive(Debug, Clone)]
pub struct SpecDecConfig {
    /// Confidence below this skips speculation entirely. In `[0, 1)`.
    pub tau_generate: f32,
    /// Cumulative confidence product below this stops drafting. In `(0, 1)`.
    pub tau_chain: f32,
    /// Confidence above this bypasses verification (ConfSpec). In `(tau_generate, 1]`.
    pub tau_accept: f32,
    /// Maximum draft length (safety cap). Must be >= 2.
    pub max_draft_len: usize,
}

impl SpecDecConfig {
    /// Create a validated config. Panics on invalid thresholds.
    pub fn new(tau_generate: f32, tau_chain: f32, tau_accept: f32, max_draft_len: usize) -> Self {
        assert!(
            tau_generate >= 0.0 && tau_generate < 1.0,
            "tau_generate must be in [0, 1), got {tau_generate}"
        );
        assert!(
            tau_chain > 0.0 && tau_chain < 1.0,
            "tau_chain must be in (0, 1), got {tau_chain}"
        );
        assert!(
            tau_accept > tau_generate && tau_accept <= 1.0,
            "tau_accept must be in (tau_generate, 1], got {tau_accept}"
        );
        assert!(max_draft_len >= 2, "max_draft_len must be >= 2");
        Self {
            tau_generate,
            tau_chain,
            tau_accept,
            max_draft_len,
        }
    }
}

impl Default for SpecDecConfig {
    fn default() -> Self {
        Self::new(0.3, 0.2, 0.95, 8)
    }
}

// ---------------------------------------------------------------------------
// Context & Actions
// ---------------------------------------------------------------------------

/// External context for the speculation gate.
#[derive(Debug, Clone)]
pub struct SpecContext {
    /// True if the model is in continuous thought (COCONUT) mode.
    /// Speculation is disabled during think phases.
    pub is_thinking: bool,
    /// Steps until the next Engram chunk boundary refresh.
    /// `None` = no Engram scheduler active. Draft length is capped to this.
    pub engram_steps_remaining: Option<usize>,
}

/// Result of [`SpeculativeDecodeScheduler::push_draft`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DraftAction {
    /// Cumulative confidence still above threshold; continue drafting.
    Continue,
    /// Stop drafting — max length, confidence drop, or boundary reached.
    Halt,
}

// ---------------------------------------------------------------------------
// Metrics
// ---------------------------------------------------------------------------

/// Accumulated metrics for speculative decoding.
#[derive(Debug, Clone, Default)]
pub struct SpecDecMetrics {
    /// Total speculation attempts (begin_draft returned true).
    pub total_speculations: u64,
    /// Speculations that ran to completion (complete() called).
    pub completed_speculations: u64,
    /// Speculations that were canceled (cancel() called).
    pub canceled_speculations: u64,
    /// Total individual draft tokens produced across all speculations.
    pub total_drafts: u64,
    /// Draft tokens accepted by verification.
    pub accepted_drafts: u64,
    /// Verification passes bypassed via ConfSpec (confidence > tau_accept).
    pub bypassed_verifications: u64,
    /// Sum of draft lengths across completed speculations (for average).
    pub total_draft_length: u64,
    /// Times begin_draft returned false (confidence too low, thinking, etc).
    pub skipped_speculations: u64,
}

impl SpecDecMetrics {
    /// Fraction of draft tokens accepted by verification.
    pub fn acceptance_rate(&self) -> f64 {
        if self.total_drafts == 0 {
            return 0.0;
        }
        self.accepted_drafts as f64 / self.total_drafts as f64
    }

    /// Average draft length across completed speculations.
    pub fn avg_draft_length(&self) -> f64 {
        if self.completed_speculations == 0 {
            return 0.0;
        }
        self.total_draft_length as f64 / self.completed_speculations as f64
    }

    /// Fraction of draft tokens that bypassed verification (ConfSpec).
    pub fn verification_bypass_rate(&self) -> f64 {
        if self.total_drafts == 0 {
            return 0.0;
        }
        self.bypassed_verifications as f64 / self.total_drafts as f64
    }
}

// ---------------------------------------------------------------------------
// Scheduler
// ---------------------------------------------------------------------------

/// Internal state for the speculative decode state machine.
enum SpecDecState {
    Idle,
    Drafting {
        tokens: Vec<u32>,
        confidences: Vec<f32>,
        cumulative_confidence: f32,
        effective_max_len: usize,
    },
}

/// Sans-I/O speculative decode scheduler.
///
/// Consumes UQ confidence scores, produces accept/reject/halt decisions.
/// The caller handles all forward passes, sampling, and verification.
pub struct SpeculativeDecodeScheduler {
    config: SpecDecConfig,
    state: SpecDecState,
    metrics: SpecDecMetrics,
}

impl SpeculativeDecodeScheduler {
    /// Create a new scheduler with the given configuration.
    pub fn new(config: SpecDecConfig) -> Self {
        Self {
            config,
            state: SpecDecState::Idle,
            metrics: SpecDecMetrics::default(),
        }
    }

    /// Attempt to begin a speculative draft.
    ///
    /// Returns `true` if speculation should proceed, `false` if conditions
    /// aren't met (low confidence, thinking, near Engram boundary).
    ///
    /// The `confidence` argument is the UQ confidence of the *already-accepted*
    /// trigger token, not the first draft token.
    pub fn begin_draft(&mut self, confidence: f32, context: &SpecContext) -> bool {
        if !matches!(self.state, SpecDecState::Idle) {
            return false;
        }

        // COCONUT: no speculation during continuous thought
        if context.is_thinking {
            self.metrics.skipped_speculations += 1;
            return false;
        }

        // Confidence gate
        if confidence < self.config.tau_generate {
            self.metrics.skipped_speculations += 1;
            return false;
        }

        // Compute effective max draft length, capped by Engram boundary
        let effective_max_len = match context.engram_steps_remaining {
            Some(remaining) => self.config.max_draft_len.min(remaining),
            None => self.config.max_draft_len,
        };

        // Not worth speculating for fewer than 2 tokens
        if effective_max_len < 2 {
            self.metrics.skipped_speculations += 1;
            return false;
        }

        self.metrics.total_speculations += 1;
        self.state = SpecDecState::Drafting {
            tokens: Vec::with_capacity(effective_max_len),
            confidences: Vec::with_capacity(effective_max_len),
            cumulative_confidence: 1.0,
            effective_max_len,
        };
        true
    }

    /// Add a drafted token with its UQ confidence score.
    ///
    /// Returns [`DraftAction::Continue`] if more tokens can be drafted,
    /// or [`DraftAction::Halt`] if the draft should stop (max length,
    /// cumulative confidence below tau_chain).
    ///
    /// # Errors
    ///
    /// Returns [`InferenceError::SpeculativeDecodeFailed`] if called while idle.
    pub fn push_draft(
        &mut self,
        token: u32,
        confidence: f32,
    ) -> Result<DraftAction, InferenceError> {
        let (tokens, confidences, cumulative, effective_max_len) = match &mut self.state {
            SpecDecState::Drafting {
                tokens,
                confidences,
                cumulative_confidence,
                effective_max_len,
            } => (tokens, confidences, cumulative_confidence, *effective_max_len),
            SpecDecState::Idle => {
                return Err(InferenceError::SpeculativeDecodeFailed(
                    "push_draft called while idle — call begin_draft first".into(),
                ))
            }
        };

        *cumulative *= confidence;
        tokens.push(token);
        confidences.push(confidence);

        // Halt conditions: max length or cumulative confidence drop
        if tokens.len() >= effective_max_len || *cumulative < self.config.tau_chain {
            Ok(DraftAction::Halt)
        } else {
            Ok(DraftAction::Continue)
        }
    }

    /// Indices of draft tokens whose confidence exceeds `tau_accept`.
    ///
    /// The caller can skip full-model verification at these positions
    /// (ConfSpec bypass). Returns an empty vec if not drafting or no
    /// tokens qualify.
    pub fn bypass_indices(&self) -> Vec<usize> {
        match &self.state {
            SpecDecState::Drafting { confidences, .. } => confidences
                .iter()
                .enumerate()
                .filter(|(_, &c)| c >= self.config.tau_accept)
                .map(|(i, _)| i)
                .collect(),
            SpecDecState::Idle => Vec::new(),
        }
    }

    /// Complete the speculation after verification.
    ///
    /// `accepted` is the number of draft tokens the full model accepted
    /// (the longest matching prefix). Updates metrics and resets to Idle.
    ///
    /// # Errors
    ///
    /// Returns [`InferenceError::SpeculativeDecodeFailed`] if called while
    /// idle or if `accepted` exceeds the number of drafted tokens.
    pub fn complete(&mut self, accepted: usize) -> Result<(), InferenceError> {
        let (tokens, confidences) = match &self.state {
            SpecDecState::Drafting {
                tokens,
                confidences,
                ..
            } => (tokens.clone(), confidences.clone()),
            SpecDecState::Idle => {
                return Err(InferenceError::SpeculativeDecodeFailed(
                    "complete called while idle — no active draft".into(),
                ))
            }
        };

        if accepted > tokens.len() {
            return Err(InferenceError::SpeculativeDecodeFailed(format!(
                "accepted count {accepted} exceeds draft length {}",
                tokens.len()
            )));
        }

        let draft_len = tokens.len() as u64;
        self.metrics.completed_speculations += 1;
        self.metrics.total_drafts += draft_len;
        self.metrics.accepted_drafts += accepted as u64;
        self.metrics.total_draft_length += draft_len;

        // Count bypassed verifications among accepted tokens
        let bypassed = confidences
            .iter()
            .take(accepted)
            .filter(|&&c| c >= self.config.tau_accept)
            .count() as u64;
        self.metrics.bypassed_verifications += bypassed;

        self.state = SpecDecState::Idle;
        Ok(())
    }

    /// Cancel the current draft without verification.
    ///
    /// Records the cancellation in metrics and resets to Idle.
    /// No-op if already idle.
    pub fn cancel(&mut self) {
        if let SpecDecState::Drafting { tokens, .. } = &self.state {
            self.metrics.canceled_speculations += 1;
            self.metrics.total_drafts += tokens.len() as u64;
        }
        self.state = SpecDecState::Idle;
    }

    /// Peek at the current draft tokens, if drafting.
    pub fn draft_tokens(&self) -> Option<&[u32]> {
        match &self.state {
            SpecDecState::Drafting { tokens, .. } => Some(tokens),
            SpecDecState::Idle => None,
        }
    }

    /// Whether the scheduler is idle (no active draft).
    pub fn is_idle(&self) -> bool {
        matches!(self.state, SpecDecState::Idle)
    }

    /// Clear all state and metrics.
    pub fn reset(&mut self) {
        self.state = SpecDecState::Idle;
        self.metrics = SpecDecMetrics::default();
    }

    /// Read-only access to accumulated metrics.
    pub fn metrics(&self) -> &SpecDecMetrics {
        &self.metrics
    }

    /// Read-only access to configuration.
    pub fn config(&self) -> &SpecDecConfig {
        &self.config
    }
}

// ---------------------------------------------------------------------------
// DraftTree — attention mask generation for parallel verification
// ---------------------------------------------------------------------------

/// Tree structure for speculative draft tokens.
///
/// Used to generate attention masks for parallel verification of drafted
/// sequences. Supports linear chains (one candidate per position) and
/// branching trees (multiple candidates, future: TALON-style).
///
/// Each entry in `parents` is the index of that draft token's parent in
/// the tree. `None` means the token is a root (first draft position).
pub struct DraftTree {
    parents: Vec<Option<usize>>,
}

impl DraftTree {
    /// Create a linear chain of `len` draft tokens.
    ///
    /// ```ignore
    /// DraftTree::linear(4) → parents: [None, Some(0), Some(1), Some(2)]
    /// ```
    pub fn linear(len: usize) -> Self {
        let parents = (0..len)
            .map(|i| if i == 0 { None } else { Some(i - 1) })
            .collect();
        Self { parents }
    }

    /// Create a tree from explicit parent indices.
    ///
    /// # Panics
    ///
    /// Panics if any parent index is out of bounds or self-referential.
    pub fn from_parents(parents: Vec<Option<usize>>) -> Self {
        for (i, parent) in parents.iter().enumerate() {
            if let Some(p) = parent {
                assert!(*p < i, "parent index {p} must be < self index {i}");
            }
        }
        Self { parents }
    }

    /// Number of draft tokens in the tree.
    pub fn len(&self) -> usize {
        self.parents.len()
    }

    /// Whether the tree is empty.
    pub fn is_empty(&self) -> bool {
        self.parents.len() == 0
    }

    /// Ancestor indices for the given draft token, from root to parent.
    ///
    /// Does not include the token itself.
    pub fn ancestors(&self, index: usize) -> Vec<usize> {
        let mut result = Vec::new();
        let mut current = self.parents[index];
        while let Some(p) = current {
            result.push(p);
            current = self.parents[p];
        }
        result.reverse();
        result
    }

    /// Generate an additive attention mask for parallel verification.
    ///
    /// Returns a `[total_len, total_len]` f32 tensor where `total_len =
    /// context_len + tree.len()`. Values are `0.0` (attend) or
    /// `f32::NEG_INFINITY` (masked), matching the standard additive mask
    /// convention used by causal attention.
    ///
    /// Mask rules:
    /// - Context tokens use standard causal masking (attend to self + past).
    /// - Draft tokens attend to all context tokens, their ancestors in the
    ///   tree, and themselves — but NOT to siblings or other branches.
    pub fn verification_mask(
        &self,
        context_len: usize,
        device: &Device,
    ) -> candle_core::Result<Tensor> {
        let total_len = context_len + self.parents.len();
        let mut mask = vec![f32::NEG_INFINITY; total_len * total_len];

        // Context tokens: standard causal mask
        for i in 0..context_len {
            for j in 0..=i {
                mask[i * total_len + j] = 0.0;
            }
        }

        // Draft tokens: attend to all context + ancestors + self
        for d in 0..self.parents.len() {
            let row = context_len + d;

            // Attend to all context tokens
            for j in 0..context_len {
                mask[row * total_len + j] = 0.0;
            }

            // Attend to ancestors in the tree
            for &anc in &self.ancestors(d) {
                mask[row * total_len + context_len + anc] = 0.0;
            }

            // Attend to self
            mask[row * total_len + row] = 0.0;
        }

        Tensor::from_vec(mask, (total_len, total_len), device)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn default_context() -> SpecContext {
        SpecContext {
            is_thinking: false,
            engram_steps_remaining: None,
        }
    }

    // -- Config validation --

    #[test]
    fn default_config_values() {
        let config = SpecDecConfig::default();
        assert!((config.tau_generate - 0.3).abs() < f32::EPSILON);
        assert!((config.tau_chain - 0.2).abs() < f32::EPSILON);
        assert!((config.tau_accept - 0.95).abs() < f32::EPSILON);
        assert_eq!(config.max_draft_len, 8);
    }

    #[test]
    #[should_panic(expected = "tau_generate must be in [0, 1)")]
    fn config_rejects_invalid_tau_generate() {
        SpecDecConfig::new(1.0, 0.2, 0.95, 8);
    }

    #[test]
    #[should_panic(expected = "tau_accept must be in (tau_generate, 1]")]
    fn tau_accept_must_exceed_tau_generate() {
        SpecDecConfig::new(0.5, 0.2, 0.4, 8); // tau_accept < tau_generate
    }

    #[test]
    #[should_panic(expected = "max_draft_len must be >= 2")]
    fn config_rejects_max_draft_len_one() {
        SpecDecConfig::new(0.3, 0.2, 0.95, 1);
    }

    // -- Draft flow --

    #[test]
    fn begin_draft_rejects_low_confidence() {
        let mut sched = SpeculativeDecodeScheduler::new(SpecDecConfig::default());
        assert!(!sched.begin_draft(0.1, &default_context())); // 0.1 < 0.3
        assert_eq!(sched.metrics().skipped_speculations, 1);
    }

    #[test]
    fn begin_draft_rejects_while_thinking() {
        let mut sched = SpeculativeDecodeScheduler::new(SpecDecConfig::default());
        let ctx = SpecContext {
            is_thinking: true,
            engram_steps_remaining: None,
        };
        assert!(!sched.begin_draft(0.9, &ctx));
        assert_eq!(sched.metrics().skipped_speculations, 1);
    }

    #[test]
    fn begin_draft_rejects_near_engram_boundary() {
        let mut sched = SpeculativeDecodeScheduler::new(SpecDecConfig::default());
        // remaining = 1, effective_max_len = min(8, 1) = 1 < 2
        let ctx = SpecContext {
            is_thinking: false,
            engram_steps_remaining: Some(1),
        };
        assert!(!sched.begin_draft(0.9, &ctx));
        assert_eq!(sched.metrics().skipped_speculations, 1);
    }

    #[test]
    fn begin_draft_caps_draft_at_engram_remaining() {
        let mut sched = SpeculativeDecodeScheduler::new(SpecDecConfig::default());
        let ctx = SpecContext {
            is_thinking: false,
            engram_steps_remaining: Some(3), // effective_max_len = min(8, 3) = 3
        };
        assert!(sched.begin_draft(0.9, &ctx));

        assert_eq!(sched.push_draft(1, 0.9).unwrap(), DraftAction::Continue); // len=1 < 3
        assert_eq!(sched.push_draft(2, 0.9).unwrap(), DraftAction::Continue); // len=2 < 3
        assert_eq!(sched.push_draft(3, 0.9).unwrap(), DraftAction::Halt); // len=3 = effective_max
    }

    #[test]
    fn push_draft_continues_while_cumulative_above_tau_chain() {
        let mut sched = SpeculativeDecodeScheduler::new(SpecDecConfig::default());
        assert!(sched.begin_draft(0.9, &default_context()));

        // cumulative: 1.0 * 0.8 = 0.8 > 0.2
        assert_eq!(sched.push_draft(10, 0.8).unwrap(), DraftAction::Continue);
        // cumulative: 0.8 * 0.7 = 0.56 > 0.2
        assert_eq!(sched.push_draft(11, 0.7).unwrap(), DraftAction::Continue);
        // cumulative: 0.56 * 0.5 = 0.28 > 0.2
        assert_eq!(sched.push_draft(12, 0.5).unwrap(), DraftAction::Continue);
        // cumulative: 0.28 * 0.5 = 0.14 < 0.2 → halt
        assert_eq!(sched.push_draft(13, 0.5).unwrap(), DraftAction::Halt);
    }

    #[test]
    fn push_draft_halts_at_max_length() {
        let config = SpecDecConfig::new(0.3, 0.01, 0.95, 3); // max_draft_len = 3
        let mut sched = SpeculativeDecodeScheduler::new(config);
        assert!(sched.begin_draft(0.9, &default_context()));

        assert_eq!(sched.push_draft(1, 0.9).unwrap(), DraftAction::Continue);
        assert_eq!(sched.push_draft(2, 0.9).unwrap(), DraftAction::Continue);
        assert_eq!(sched.push_draft(3, 0.9).unwrap(), DraftAction::Halt); // 3rd = max
    }

    #[test]
    fn push_draft_errors_while_idle() {
        let mut sched = SpeculativeDecodeScheduler::new(SpecDecConfig::default());
        assert!(sched.push_draft(1, 0.9).is_err());
    }

    // -- Verification --

    #[test]
    fn bypass_indices_empty_below_threshold() {
        let mut sched = SpeculativeDecodeScheduler::new(SpecDecConfig::default());
        assert!(sched.begin_draft(0.9, &default_context()));
        sched.push_draft(1, 0.5).unwrap();
        sched.push_draft(2, 0.8).unwrap();
        sched.push_draft(3, 0.9).unwrap(); // 0.9 < 0.95

        assert!(sched.bypass_indices().is_empty());
    }

    #[test]
    fn bypass_indices_includes_high_confidence() {
        let mut sched = SpeculativeDecodeScheduler::new(SpecDecConfig::default());
        assert!(sched.begin_draft(0.9, &default_context()));
        sched.push_draft(1, 0.5).unwrap();
        sched.push_draft(2, 0.99).unwrap(); // >= 0.95
        sched.push_draft(3, 0.97).unwrap(); // >= 0.95

        assert_eq!(sched.bypass_indices(), vec![1, 2]);
    }

    #[test]
    fn bypass_at_exact_tau_accept() {
        let mut sched = SpeculativeDecodeScheduler::new(SpecDecConfig::default());
        assert!(sched.begin_draft(0.9, &default_context()));
        sched.push_draft(1, 0.95).unwrap(); // exactly tau_accept — should bypass

        assert_eq!(sched.bypass_indices(), vec![0]);
    }

    #[test]
    fn complete_transitions_to_idle() {
        let mut sched = SpeculativeDecodeScheduler::new(SpecDecConfig::default());
        assert!(sched.begin_draft(0.9, &default_context()));
        sched.push_draft(1, 0.8).unwrap();
        sched.push_draft(2, 0.7).unwrap();
        assert!(!sched.is_idle());

        sched.complete(2).unwrap();
        assert!(sched.is_idle());
        assert!(sched.draft_tokens().is_none());
    }

    #[test]
    fn complete_errors_while_idle() {
        let mut sched = SpeculativeDecodeScheduler::new(SpecDecConfig::default());
        assert!(sched.complete(0).is_err());
    }

    #[test]
    fn complete_errors_on_accepted_exceeds_draft() {
        let mut sched = SpeculativeDecodeScheduler::new(SpecDecConfig::default());
        assert!(sched.begin_draft(0.9, &default_context()));
        sched.push_draft(1, 0.8).unwrap();
        sched.push_draft(2, 0.7).unwrap();

        assert!(sched.complete(3).is_err()); // only 2 drafted
    }

    // -- Cancel / Reset --

    #[test]
    fn cancel_resets_state_and_tracks_metric() {
        let mut sched = SpeculativeDecodeScheduler::new(SpecDecConfig::default());
        assert!(sched.begin_draft(0.9, &default_context()));
        sched.push_draft(1, 0.8).unwrap();
        sched.push_draft(2, 0.7).unwrap();

        sched.cancel();
        assert!(sched.is_idle());
        assert_eq!(sched.metrics().canceled_speculations, 1);
        assert_eq!(sched.metrics().total_drafts, 2);
    }

    #[test]
    fn cancel_does_not_inflate_avg_draft_length() {
        let mut sched = SpeculativeDecodeScheduler::new(SpecDecConfig::default());

        // Complete one 3-token draft
        assert!(sched.begin_draft(0.9, &default_context()));
        sched.push_draft(1, 0.8).unwrap();
        sched.push_draft(2, 0.7).unwrap();
        sched.push_draft(3, 0.6).unwrap();
        sched.complete(2).unwrap();

        // Cancel a 5-token draft
        assert!(sched.begin_draft(0.8, &default_context()));
        for i in 0..5 {
            sched.push_draft(10 + i, 0.9).unwrap();
        }
        sched.cancel();

        // avg_draft_length should be 3.0 (only the completed speculation),
        // not 8.0 (if cancel had added to total_draft_length)
        assert!((sched.metrics().avg_draft_length() - 3.0).abs() < 1e-10);
    }

    #[test]
    fn reset_clears_everything() {
        let mut sched = SpeculativeDecodeScheduler::new(SpecDecConfig::default());
        assert!(sched.begin_draft(0.9, &default_context()));
        sched.push_draft(1, 0.8).unwrap();
        sched.complete(1).unwrap();

        sched.reset();
        assert!(sched.is_idle());
        assert_eq!(sched.metrics().total_speculations, 0);
        assert_eq!(sched.metrics().completed_speculations, 0);
    }

    // -- Metrics --

    #[test]
    fn metrics_accumulate_across_speculations() {
        let mut sched = SpeculativeDecodeScheduler::new(SpecDecConfig::default());

        // Speculation 1: 3 tokens, 2 accepted
        assert!(sched.begin_draft(0.9, &default_context()));
        sched.push_draft(1, 0.8).unwrap();
        sched.push_draft(2, 0.7).unwrap();
        sched.push_draft(3, 0.6).unwrap();
        sched.complete(2).unwrap();

        // Speculation 2: 2 tokens, 2 accepted
        assert!(sched.begin_draft(0.8, &default_context()));
        sched.push_draft(4, 0.9).unwrap();
        sched.push_draft(5, 0.8).unwrap();
        sched.complete(2).unwrap();

        let m = sched.metrics();
        assert_eq!(m.total_speculations, 2);
        assert_eq!(m.completed_speculations, 2);
        assert_eq!(m.total_drafts, 5);
        assert_eq!(m.accepted_drafts, 4);
        assert_eq!(m.total_draft_length, 5);
    }

    #[test]
    fn computed_ratios_correct() {
        let mut sched = SpeculativeDecodeScheduler::new(SpecDecConfig::default());

        // 4 tokens drafted, 3 accepted, 1 bypassed (confidence >= 0.95)
        assert!(sched.begin_draft(0.9, &default_context()));
        sched.push_draft(1, 0.8).unwrap();
        sched.push_draft(2, 0.99).unwrap(); // bypass
        sched.push_draft(3, 0.7).unwrap();
        sched.push_draft(4, 0.6).unwrap();
        sched.complete(3).unwrap(); // accept first 3

        let m = sched.metrics();
        assert!((m.acceptance_rate() - 0.75).abs() < 1e-10); // 3/4
        assert!((m.avg_draft_length() - 4.0).abs() < 1e-10); // 4/1
        assert!((m.verification_bypass_rate() - 0.25).abs() < 1e-10); // 1/4
    }

    // -- DraftTree --

    #[test]
    fn linear_tree_ancestors_correct() {
        let tree = DraftTree::linear(4);
        assert_eq!(tree.len(), 4);
        assert!(!tree.is_empty());

        // Root has no ancestors
        assert!(tree.ancestors(0).is_empty());
        // Token 1: ancestor is 0
        assert_eq!(tree.ancestors(1), vec![0]);
        // Token 3: ancestors are 0, 1, 2
        assert_eq!(tree.ancestors(3), vec![0, 1, 2]);
    }

    #[test]
    fn verification_mask_shape_and_values() {
        let tree = DraftTree::linear(3);
        let device = Device::Cpu;
        let mask = tree.verification_mask(2, &device).unwrap();

        // Shape: [2+3, 2+3] = [5, 5]
        assert_eq!(mask.dims(), &[5, 5]);

        let data: Vec<f32> = mask.flatten_all().unwrap().to_vec1().unwrap();
        let at = |r: usize, c: usize| data[r * 5 + c];

        // Context tokens: causal mask
        assert_eq!(at(0, 0), 0.0); // ctx[0] attends to ctx[0]
        assert_eq!(at(0, 1), f32::NEG_INFINITY); // ctx[0] can't see ctx[1]
        assert_eq!(at(1, 0), 0.0); // ctx[1] attends to ctx[0]
        assert_eq!(at(1, 1), 0.0); // ctx[1] attends to ctx[1]

        // Draft token 0 (row 2): attends to all context + self
        assert_eq!(at(2, 0), 0.0); // context
        assert_eq!(at(2, 1), 0.0); // context
        assert_eq!(at(2, 2), 0.0); // self
        assert_eq!(at(2, 3), f32::NEG_INFINITY); // can't see draft 1
        assert_eq!(at(2, 4), f32::NEG_INFINITY); // can't see draft 2

        // Draft token 2 (row 4): attends to context + ancestors (0,1) + self
        assert_eq!(at(4, 0), 0.0); // context
        assert_eq!(at(4, 1), 0.0); // context
        assert_eq!(at(4, 2), 0.0); // ancestor: draft 0
        assert_eq!(at(4, 3), 0.0); // ancestor: draft 1
        assert_eq!(at(4, 4), 0.0); // self
    }

    #[test]
    fn branching_tree_masks_siblings() {
        // Tree:  0 (root)
        //       / \
        //      1   2  (siblings — should NOT attend to each other)
        let tree = DraftTree::from_parents(vec![None, Some(0), Some(0)]);
        let device = Device::Cpu;
        let mask = tree.verification_mask(1, &device).unwrap();

        // Shape: [1+3, 1+3] = [4, 4]
        assert_eq!(mask.dims(), &[4, 4]);

        let data: Vec<f32> = mask.flatten_all().unwrap().to_vec1().unwrap();
        let at = |r: usize, c: usize| data[r * 4 + c];

        // Draft 1 (row 2): context + ancestor 0 + self, NOT sibling 2
        assert_eq!(at(2, 0), 0.0); // context
        assert_eq!(at(2, 1), 0.0); // ancestor: draft 0
        assert_eq!(at(2, 2), 0.0); // self
        assert_eq!(at(2, 3), f32::NEG_INFINITY); // sibling: draft 2

        // Draft 2 (row 3): context + ancestor 0 + self, NOT sibling 1
        assert_eq!(at(3, 0), 0.0); // context
        assert_eq!(at(3, 1), 0.0); // ancestor: draft 0
        assert_eq!(at(3, 2), f32::NEG_INFINITY); // sibling: draft 1
        assert_eq!(at(3, 3), 0.0); // self
    }
}
