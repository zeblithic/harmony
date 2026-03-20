#![cfg_attr(not(feature = "std"), no_std)]
extern crate alloc;

pub mod claim;
pub mod error;

pub use claim::{Claim, SaltedClaim};
pub use error::CredentialError;
