#![cfg_attr(not(feature = "std"), no_std)]
extern crate alloc;

pub mod error;
pub mod identity;
pub mod ucan;

pub use error::IdentityError;
pub use identity::{Identity, PrivateIdentity};
pub use ucan::{
    verify_revocation, verify_token, CapabilityType, IdentityResolver, ProofResolver, Revocation,
    RevocationSet, UcanError, UcanToken,
};
#[cfg(any(test, feature = "test-utils"))]
pub use ucan::{MemoryIdentityStore, MemoryProofStore, MemoryRevocationSet};
