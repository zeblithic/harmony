#![cfg_attr(not(feature = "std"), no_std)]
extern crate alloc;

pub mod entropy;
pub mod error;
pub mod network;
pub mod persistence;

pub use entropy::EntropySource;
pub use error::PlatformError;
pub use network::NetworkInterface;
pub use persistence::{MemoryPersistentState, PersistentState};
