#![cfg_attr(not(feature = "std"), no_std)]
extern crate alloc;

pub mod contact;
pub mod error;

pub use contact::{Contact, PeeringPolicy, PeeringPriority};
pub use error::ContactError;
