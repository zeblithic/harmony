#![cfg_attr(not(feature = "std"), no_std)]
extern crate alloc;

pub mod error;
pub mod score;

pub use error::TrustError;
pub use score::TrustScore;
