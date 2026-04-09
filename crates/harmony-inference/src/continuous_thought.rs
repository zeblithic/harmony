//! Continuous thought infrastructure — types and configuration.
//!
//! Phase 4a: Pause token (`<think>`) routing driven by the UQ Head.
//! The UQ Head classifies each generation step; when the model is uncertain,
//! the engine forces a `<think>` token instead of sampling, granting extra
//! forward passes for deliberation without emitting output.
//!
//! Phase 4b (future): Replaces pause tokens with true COCONUT continuous
//! thought — feeding the final hidden state back as the next input embedding,
//! bypassing the vocabulary projection entirely.

/// Routing decision for the generation loop.
///
/// Returned by [`crate::HarmonyEngine::route_thought`] to tell the caller
/// whether to emit a token, continue thinking, or abort generation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThoughtAction {
    /// Emit the sampled token to the output stream.
    Emit,
    /// Suppress output, force `<think>` token, continue deliberation.
    /// The `<think>` token runs a full forward pass (advancing the KV cache)
    /// but is filtered from the user-visible output stream.
    Think,
    /// Abort generation. Triggered by spectral collapse detection —
    /// hidden state norms collapsing toward zero, indicating the model
    /// has no signal. Escalate to the caller.
    Abort,
}

/// Configuration for continuous thought routing.
///
/// Controls when and how the UQ Head triggers extra thinking steps.
/// Loaded from GGUF metadata or set programmatically.
///
/// When `think_token_id` is `None`, continuous thought is disabled and
/// [`crate::HarmonyEngine::route_thought`] always returns [`ThoughtAction::Emit`].
#[derive(Debug, Clone, PartialEq)]
pub struct ContinuousThoughtConfig {
    /// Token ID for the `<think>` pause token. `None` = disabled.
    pub think_token_id: Option<u32>,
    /// Maximum consecutive think steps before forcing token emission.
    /// Prevents infinite loops. Default: 4.
    pub max_think_steps: u32,
    /// UQ confidence threshold above which the model is considered confident
    /// enough to emit a token. Below this threshold, the model thinks more.
    /// Default: 0.85.
    pub confidence_threshold: f32,
}

impl Default for ContinuousThoughtConfig {
    fn default() -> Self {
        Self {
            think_token_id: None,
            max_think_steps: 4,
            confidence_threshold: 0.85,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_values() {
        let config = ContinuousThoughtConfig::default();
        assert_eq!(config.think_token_id, None);
        assert_eq!(config.max_think_steps, 4);
        assert!((config.confidence_threshold - 0.85).abs() < f32::EPSILON);
    }

    #[test]
    fn config_with_think_token() {
        let config = ContinuousThoughtConfig {
            think_token_id: Some(32000),
            max_think_steps: 6,
            confidence_threshold: 0.9,
        };
        assert_eq!(config.think_token_id, Some(32000));
        assert_eq!(config.max_think_steps, 6);
        assert!((config.confidence_threshold - 0.9).abs() < f32::EPSILON);
    }

    #[test]
    fn thought_action_variants_are_distinct() {
        assert_ne!(ThoughtAction::Emit, ThoughtAction::Think);
        assert_ne!(ThoughtAction::Emit, ThoughtAction::Abort);
        assert_ne!(ThoughtAction::Think, ThoughtAction::Abort);
    }

    #[test]
    fn thought_action_is_copy() {
        let a = ThoughtAction::Think;
        let b = a; // Copy
        assert_eq!(a, b);
    }

    #[test]
    fn config_is_clone() {
        let config = ContinuousThoughtConfig {
            think_token_id: Some(100),
            max_think_steps: 3,
            confidence_threshold: 0.7,
        };
        let cloned = config.clone();
        assert_eq!(config, cloned);
    }

    #[test]
    fn disabled_config() {
        let config = ContinuousThoughtConfig::default();
        assert!(config.think_token_id.is_none());
    }
}
