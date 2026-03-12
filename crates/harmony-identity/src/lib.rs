#![cfg_attr(not(feature = "std"), no_std)]
extern crate alloc;

pub mod error;
pub mod identity;
pub mod pq_identity;
pub mod ucan;

pub use error::IdentityError;
pub use identity::{Identity, PrivateIdentity};
pub use pq_identity::{PqIdentity, PqPrivateIdentity};
pub use ucan::{
    verify_revocation, verify_token, CapabilityType, CryptoSuite, IdentityResolver, PqUcanToken,
    ProofResolver, Revocation, RevocationSet, UcanError, UcanToken,
};
#[cfg(any(test, feature = "test-utils"))]
pub use ucan::{MemoryIdentityStore, MemoryProofStore, MemoryRevocationSet};
