#![cfg_attr(not(feature = "std"), no_std)]
extern crate alloc;

pub mod endorsement;
pub mod error;
pub mod profile;
pub mod verify;

pub use endorsement::{EndorsementBuilder, EndorsementRecord};
pub use error::ProfileError;
pub use profile::{ProfileBuilder, ProfileRecord};
pub use verify::{verify_endorsement, verify_profile, ProfileKeyResolver};

#[cfg(any(test, feature = "test-utils"))]
pub use verify::MemoryKeyResolver;
