#![cfg_attr(not(feature = "std"), no_std)]
extern crate alloc;

pub mod crypto_suite;
pub mod error;
pub mod identity;
pub mod identity_ref;
pub mod pq_identity;
pub mod ucan;

pub use crypto_suite::CryptoSuite;
pub use error::IdentityError;
pub use identity::{Identity, IdentityHash, PrivateIdentity};
pub use identity_ref::IdentityRef;
pub use pq_identity::{PqIdentity, PqPrivateIdentity};
pub use ucan::{
    verify_revocation, verify_token, CapabilityType, IdentityResolver, PqUcanToken, ProofResolver,
    Revocation, RevocationSet, UcanError, UcanToken,
};
#[cfg(any(test, feature = "test-utils"))]
pub use ucan::{MemoryIdentityStore, MemoryProofStore, MemoryRevocationSet};
