#![cfg_attr(not(feature = "std"), no_std)]
extern crate alloc;

pub mod claim;
pub mod credential;
pub mod disclosure;
pub mod error;
pub mod status_list;
pub mod verify;

pub use claim::{Claim, SaltedClaim};
pub use credential::{Credential, CredentialBuilder};
pub use disclosure::Presentation;
pub use error::CredentialError;
pub use status_list::{StatusList, StatusListResolver};
pub use verify::{
    verify_chain, verify_credential, verify_presentation, CredentialKeyResolver, CredentialResolver,
};

#[cfg(any(test, feature = "test-utils"))]
pub use status_list::MemoryStatusListResolver;
#[cfg(any(test, feature = "test-utils"))]
pub use verify::MemoryCredentialResolver;
#[cfg(any(test, feature = "test-utils"))]
pub use verify::MemoryKeyResolver;

#[cfg(feature = "jsonld")]
pub mod jsonld;

#[cfg(feature = "jsonld")]
pub use jsonld::identity_to_did_key;
#[cfg(feature = "jsonld")]
pub use jsonld::credential_to_jsonld;
