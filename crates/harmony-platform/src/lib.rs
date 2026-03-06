#![cfg_attr(not(feature = "std"), no_std)]
extern crate alloc;

pub mod entropy;
pub mod error;

pub use entropy::EntropySource;
