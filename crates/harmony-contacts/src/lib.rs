#![cfg_attr(not(feature = "std"), no_std)]
extern crate alloc;

pub mod contact;
pub mod error;
pub mod store;

pub use contact::{Contact, PeeringPolicy, PeeringPriority};
pub use error::ContactError;
pub use store::ContactStore;
