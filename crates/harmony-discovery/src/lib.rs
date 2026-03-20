#![cfg_attr(not(feature = "std"), no_std)]
extern crate alloc;

pub mod error;
pub mod record;

pub use error::DiscoveryError;
pub use record::{AnnounceBuilder, AnnounceRecord, RoutingHint};
