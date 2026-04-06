//! Uncertainty Quantification Head — parallel metacognitive MLP monitor.
//!
//! Takes 8 pre-extracted features (hidden state norms, entropy, etc.) and
//! produces a 4-class uncertainty classification plus scalar confidence.
//! The caller uses the classification for routing decisions (emit, retrieve,
//! or abort). Feature extraction is external — this module is a pure classifier.

use candle_core::{Device, Result, Tensor};

/// Uncertainty classification output.
///
/// Discriminant values (0–3) match training label indices and are stored
/// as `u8` in serialized models. Use [`from_u8`](Self::from_u8) to parse.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum UqClass {
    /// Model is confident in its output. Action: emit token, continue.
    Confident = 0,
    /// High-volume uncertainty — many plausible candidates.
    /// Action: trigger Engram lookup with hidden state as semantic query.
    HighVolume = 1,
    /// Spectral collapse — hidden state norms collapsing toward zero.
    /// Action: abort generation, flag as unknowable, escalate.
    SpectralCollapse = 2,
    /// Ambiguous uncertainty. Conservative action: treat as HighVolume.
    Uncertain = 3,
}

impl UqClass {
    /// Parse a class from its `u8` discriminant (0–3).
    ///
    /// Returns `None` for values outside the valid range.
    pub fn from_u8(value: u8) -> Option<Self> {
        match value {
            0 => Some(Self::Confident),
            1 => Some(Self::HighVolume),
            2 => Some(Self::SpectralCollapse),
            3 => Some(Self::Uncertain),
            _ => None,
        }
    }
}

impl TryFrom<u8> for UqClass {
    type Error = u8;

    fn try_from(value: u8) -> core::result::Result<Self, Self::Error> {
        Self::from_u8(value).ok_or(value)
    }
}

/// Configuration for the Uncertainty Quantification Head.
#[derive(Debug, Clone)]
pub struct UqHeadConfig {
    /// Number of input features. Default: 8.
    pub num_features: usize,
    /// Hidden dimension of the classifier MLP. Default: 32.
    pub hidden_dim: usize,
    /// Number of output classes. Default: 4.
    pub num_classes: usize,
}

impl Default for UqHeadConfig {
    fn default() -> Self {
        Self {
            num_features: 8,
            hidden_dim: 32,
            num_classes: 4,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── UqClass parsing ──

    #[test]
    fn from_u8_valid_classes() {
        assert_eq!(UqClass::from_u8(0), Some(UqClass::Confident));
        assert_eq!(UqClass::from_u8(1), Some(UqClass::HighVolume));
        assert_eq!(UqClass::from_u8(2), Some(UqClass::SpectralCollapse));
        assert_eq!(UqClass::from_u8(3), Some(UqClass::Uncertain));
    }

    #[test]
    fn from_u8_invalid_returns_none() {
        assert_eq!(UqClass::from_u8(4), None);
        assert_eq!(UqClass::from_u8(255), None);
    }

    #[test]
    fn try_from_u8_valid() {
        assert_eq!(UqClass::try_from(0u8), Ok(UqClass::Confident));
        assert_eq!(UqClass::try_from(3u8), Ok(UqClass::Uncertain));
    }

    #[test]
    fn try_from_u8_invalid() {
        assert_eq!(UqClass::try_from(4u8), Err(4u8));
    }

    // ── UqHeadConfig ──

    #[test]
    fn default_config_values() {
        let cfg = UqHeadConfig::default();
        assert_eq!(cfg.num_features, 8);
        assert_eq!(cfg.hidden_dim, 32);
        assert_eq!(cfg.num_classes, 4);
    }
}
