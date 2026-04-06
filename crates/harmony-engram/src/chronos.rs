//! Chronos temporal decay — attenuates Engram embeddings by knowledge freshness.
//!
//! Stale facts decay toward zero using a Gaussian curve past their TTL.
//! When decay reaches zero, the attenuated embedding becomes a zero vector,
//! and `EngramGatedResidual` produces zero residual — the model proceeds
//! as if no Engram entry exists.

/// Chronos frequency tier — classifies how quickly knowledge expires.
///
/// Tier numbering (1–5) matches the spec and is stored as `u8` in Engram
/// table entries. Use [`from_u8`](Self::from_u8) to parse from stored values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum ChronosTier {
    /// Physical constants, mathematical axioms. Never expires.
    Eternal = 1,
    /// Historical facts, geography. TTL: ~10 years.
    NearEternal = 2,
    /// Political leaders, tech standards. TTL: ~1 year.
    Episodic = 3,
    /// Ages, populations. TTL: ~30 days.
    Regular = 4,
    /// Stock prices, live sensor data. Immediately stale.
    Ephemeral = 5,
}

impl ChronosTier {
    /// Default TTL in seconds for this tier.
    ///
    /// | Tier | TTL |
    /// |------|-----|
    /// | Eternal | `u32::MAX` (~136 years) |
    /// | NearEternal | 315,576,000 (~10 years) |
    /// | Episodic | 31,557,600 (~1 year) |
    /// | Regular | 2,592,000 (30 days) |
    /// | Ephemeral | 0 |
    pub fn default_ttl_seconds(self) -> u32 {
        match self {
            Self::Eternal => u32::MAX,
            Self::NearEternal => 315_576_000, // 10 * 365.25 * 86400
            Self::Episodic => 31_557_600,     // 365.25 * 86400
            Self::Regular => 2_592_000,       // 30 * 86400
            Self::Ephemeral => 0,
        }
    }

    /// Parse a tier from its `u8` discriminant (1–5).
    ///
    /// Returns `None` for values outside the valid range.
    pub fn from_u8(value: u8) -> Option<Self> {
        match value {
            1 => Some(Self::Eternal),
            2 => Some(Self::NearEternal),
            3 => Some(Self::Episodic),
            4 => Some(Self::Regular),
            5 => Some(Self::Ephemeral),
            _ => None,
        }
    }
}

/// Per-entry metadata for Chronos temporal decay.
///
/// Carries the timestamp and freshness tier needed to compute the decay
/// factor. The embedding itself is not included — the caller multiplies
/// the resolved embedding by the decay factor returned by [`temporal_decay`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EngramMetadata {
    /// Unix epoch timestamp when this entry was last updated.
    pub timestamp: u32,
    /// Frequency tier (1–5).
    pub tier: ChronosTier,
    /// Time-to-live in seconds. Usually derived from tier via
    /// [`ChronosTier::default_ttl_seconds`], but may be overridden.
    pub ttl_seconds: u32,
}

impl EngramMetadata {
    /// Create metadata with the tier's default TTL.
    pub fn new(timestamp: u32, tier: ChronosTier) -> Self {
        Self {
            timestamp,
            tier,
            ttl_seconds: tier.default_ttl_seconds(),
        }
    }

    /// Create metadata with a custom TTL override.
    pub fn with_ttl(timestamp: u32, tier: ChronosTier, ttl_seconds: u32) -> Self {
        Self {
            timestamp,
            tier,
            ttl_seconds,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Tier TTL defaults ──

    #[test]
    fn eternal_ttl_is_max() {
        assert_eq!(ChronosTier::Eternal.default_ttl_seconds(), u32::MAX);
    }

    #[test]
    fn near_eternal_ttl_is_ten_years() {
        // 10 * 365.25 * 24 * 3600 = 315_576_000
        assert_eq!(ChronosTier::NearEternal.default_ttl_seconds(), 315_576_000);
    }

    #[test]
    fn episodic_ttl_is_one_year() {
        // 365.25 * 24 * 3600 = 31_557_600
        assert_eq!(ChronosTier::Episodic.default_ttl_seconds(), 31_557_600);
    }

    #[test]
    fn regular_ttl_is_thirty_days() {
        // 30 * 24 * 3600 = 2_592_000
        assert_eq!(ChronosTier::Regular.default_ttl_seconds(), 2_592_000);
    }

    #[test]
    fn ephemeral_ttl_is_zero() {
        assert_eq!(ChronosTier::Ephemeral.default_ttl_seconds(), 0);
    }

    // ── Tier parsing ──

    #[test]
    fn from_u8_valid_tiers() {
        assert_eq!(ChronosTier::from_u8(1), Some(ChronosTier::Eternal));
        assert_eq!(ChronosTier::from_u8(2), Some(ChronosTier::NearEternal));
        assert_eq!(ChronosTier::from_u8(3), Some(ChronosTier::Episodic));
        assert_eq!(ChronosTier::from_u8(4), Some(ChronosTier::Regular));
        assert_eq!(ChronosTier::from_u8(5), Some(ChronosTier::Ephemeral));
    }

    #[test]
    fn from_u8_invalid_returns_none() {
        assert_eq!(ChronosTier::from_u8(0), None);
        assert_eq!(ChronosTier::from_u8(6), None);
        assert_eq!(ChronosTier::from_u8(255), None);
    }

    // ── Metadata construction ──

    #[test]
    fn new_derives_ttl_from_tier() {
        let meta = EngramMetadata::new(1_000_000, ChronosTier::Regular);
        assert_eq!(meta.timestamp, 1_000_000);
        assert_eq!(meta.tier, ChronosTier::Regular);
        assert_eq!(meta.ttl_seconds, 2_592_000);
    }

    #[test]
    fn with_ttl_overrides_default() {
        let meta = EngramMetadata::with_ttl(1_000_000, ChronosTier::Regular, 86_400);
        assert_eq!(meta.ttl_seconds, 86_400);
        assert_eq!(meta.tier, ChronosTier::Regular);
    }
}
