#![cfg_attr(not(feature = "std"), no_std)]
extern crate alloc;

pub mod claim;
pub mod credential;
pub mod error;
pub mod status_list;

pub use claim::{Claim, SaltedClaim};
pub use credential::{Credential, CredentialBuilder};
pub use error::CredentialError;
pub use status_list::{StatusList, StatusListResolver};

#[cfg(any(test, feature = "test-utils"))]
pub use status_list::MemoryStatusListResolver;
