#![cfg_attr(not(feature = "std"), no_std)]
extern crate alloc;

pub mod error;
pub mod manager;
pub mod record;
pub mod verify;

pub use error::DiscoveryError;
pub use manager::{DiscoveryAction, DiscoveryEvent, DiscoveryManager};
pub use record::{AnnounceBuilder, AnnounceRecord, RoutingHint};
pub use verify::verify_announce;
