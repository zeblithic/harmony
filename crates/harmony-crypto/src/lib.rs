#![cfg_attr(not(feature = "std"), no_std)]
extern crate alloc;

pub mod aead;
pub mod capability;
pub mod error;
pub mod fernet;
pub mod hash;
pub mod hkdf;
pub mod hybrid_kem;
pub mod ml_dsa;
pub mod ml_kem;
pub mod sealed_box;
#[cfg(feature = "serde")]
mod serde_helpers;
pub mod x25519;

pub use error::CryptoError;
