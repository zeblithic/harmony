//! Zenoh-native identity discovery with signed announce records and
//! presence tracking.
//!
//! # Security
//!
//! Announce verification includes:
//! 1. Signature validity (cryptographic proof of authorship)
//! 2. Address-to-pubkey binding (`SHA256(enc_key || sig_key)[:16] == identity_hash`)
//! 3. Timestamp freshness (expiry + clock skew tolerance)
//!
//! The binding check prevents forged announces where an attacker substitutes
//! their own keys while claiming someone else's identity address. An announce
//! is rejected with [`DiscoveryError::AddressMismatch`] if the included
//! public keys do not hash to the claimed identity address.
//!
//! See [`verify_announce`] for the full verification pipeline.

#![cfg_attr(not(feature = "std"), no_std)]
extern crate alloc;

pub mod error;
pub mod manager;
pub mod record;
pub mod resolve;
pub mod verify;

pub use error::DiscoveryError;
pub use manager::{DiscoveryAction, DiscoveryEvent, DiscoveryManager};
pub use record::{AnnounceBuilder, AnnounceRecord, RoutingHint};
pub use resolve::OfflineResolver;
pub use verify::verify_announce;
