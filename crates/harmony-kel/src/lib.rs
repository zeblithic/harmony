#![cfg_attr(not(feature = "std"), no_std)]
extern crate alloc;

pub mod commitment;
pub mod error;

pub use commitment::{compute_commitment, verify_commitment};
pub use error::KelError;
