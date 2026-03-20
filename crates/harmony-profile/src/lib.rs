#![cfg_attr(not(feature = "std"), no_std)]
extern crate alloc;

pub mod error;
pub mod profile;

pub use error::ProfileError;
pub use profile::{ProfileBuilder, ProfileRecord};
