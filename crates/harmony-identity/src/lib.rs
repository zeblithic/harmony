#![cfg_attr(not(feature = "std"), no_std)]
extern crate alloc;

pub mod error;
pub mod identity;

pub use error::IdentityError;
pub use identity::{Identity, PrivateIdentity};
