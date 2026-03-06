#![cfg_attr(not(feature = "std"), no_std)]
extern crate alloc;

pub mod error;
pub mod identity;
pub mod ucan;

pub use error::IdentityError;
pub use identity::{Identity, PrivateIdentity};
pub use ucan::{
    CapabilityType, IdentityResolver, MemoryIdentityStore, MemoryProofStore,
    MemoryRevocationSet, ProofResolver, Revocation, RevocationSet, UcanError, UcanToken,
};
