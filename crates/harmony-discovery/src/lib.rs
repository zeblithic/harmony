//! Zenoh-native identity discovery with signed announce records and
//! presence tracking.
//!
//! # Security
//!
//! **V1 limitation:** Announce verification checks signature validity
//! but does NOT verify that the included `public_key` hashes to the
//! claimed `identity_ref.hash`. This means an attacker with a valid
//! keypair can forge announces for arbitrary identity addresses. Do
//! not rely on announce records as proof of identity ownership until
//! address re-derivation is implemented in a future version. See
//! [`verify_announce`] for details.

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
