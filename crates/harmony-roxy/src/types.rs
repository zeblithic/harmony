//! Shared types: artist profile, price, licensing terms.

use alloc::string::String;
use bitflags::bitflags;
use serde::{Deserialize, Serialize};

/// Type of license an artist offers for content.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u8)]
pub enum LicenseType {
    /// No token needed. Content key is public.
    Free = 0,
    /// Pay once, access forever (UCAN with no expiry).
    OneTime = 1,
    /// Recurring access windows with expiry.
    Subscription = 2,
    /// Opaque terms referencing an external contract.
    Custom = 3,
}

/// How a price is charged.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u8)]
pub enum PricePer {
    /// One-time payment.
    Once = 0,
    /// Per calendar month.
    Month = 1,
    /// Per calendar year.
    Year = 2,
    /// Per access (each stream/download).
    Access = 3,
}

/// Price for a license.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Price {
    /// Amount in smallest currency unit (e.g., cents for USD).
    pub amount: u64,
    /// Currency identifier (e.g., "USD", "BTC", "HAR").
    pub currency: String,
    /// Billing period.
    pub per: PricePer,
}

bitflags! {
    /// What the consumer is allowed to do with licensed content.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct UsageRights: u8 {
        /// Decrypt and play in real-time.
        const STREAM   = 0b0001;
        /// Persist decrypted content locally.
        const DOWNLOAD = 0b0010;
        /// Derive new content from this content.
        const REMIX    = 0b0100;
        /// Delegate access to others via UCAN chain.
        const RESHARE  = 0b1000;
    }
}

// Manual serde implementation for UsageRights because bitflags 2.x
// requires the `serde` feature for derive-based serialization, which
// is not enabled in this workspace.
impl Serialize for UsageRights {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        self.bits().serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for UsageRights {
    /// Deserializes usage rights, silently masking off unknown bits.
    ///
    /// Uses `from_bits_truncate` for forward compatibility: an older node
    /// that doesn't recognize a new right (e.g., `BROADCAST = 0b10000`)
    /// will ignore it rather than rejecting the entire manifest. This is
    /// safe because an old node can't exercise a right its software doesn't
    /// implement.
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let bits = u8::deserialize(deserializer)?;
        Ok(UsageRights::from_bits_truncate(bits))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn usage_rights_bitflags() {
        let rights = UsageRights::STREAM | UsageRights::DOWNLOAD;
        assert!(rights.contains(UsageRights::STREAM));
        assert!(rights.contains(UsageRights::DOWNLOAD));
        assert!(!rights.contains(UsageRights::REMIX));
        assert!(!rights.contains(UsageRights::RESHARE));
    }

    #[test]
    fn usage_rights_serialization_round_trip() {
        let rights = UsageRights::STREAM | UsageRights::REMIX;
        let bytes = postcard::to_allocvec(&rights).unwrap();
        let decoded: UsageRights = postcard::from_bytes(&bytes).unwrap();
        assert_eq!(rights, decoded);
    }

    #[test]
    fn license_type_serialization_round_trip() {
        for lt in [
            LicenseType::Free,
            LicenseType::OneTime,
            LicenseType::Subscription,
            LicenseType::Custom,
        ] {
            let bytes = postcard::to_allocvec(&lt).unwrap();
            let decoded: LicenseType = postcard::from_bytes(&bytes).unwrap();
            assert_eq!(lt, decoded);
        }
    }

    #[test]
    fn price_serialization_round_trip() {
        let price = Price {
            amount: 500,
            currency: alloc::string::String::from("USD"),
            per: PricePer::Month,
        };
        let bytes = postcard::to_allocvec(&price).unwrap();
        let decoded: Price = postcard::from_bytes(&bytes).unwrap();
        assert_eq!(price.amount, decoded.amount);
        assert_eq!(price.currency, decoded.currency);
        assert_eq!(price.per, decoded.per);
    }
}
