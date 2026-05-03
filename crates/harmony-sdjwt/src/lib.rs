#![cfg_attr(not(feature = "std"), no_std)]
extern crate alloc;

pub mod error;
#[cfg(feature = "std")]
pub mod key_binding;
pub mod parse;
pub mod types;
#[cfg(feature = "std")]
pub mod verify;

pub use error::SdJwtError;
#[cfg(feature = "std")]
pub use key_binding::verify_key_binding;
#[cfg(feature = "std")]
pub use parse::parse;
pub use parse::split_jws;
pub use types::{Disclosure, JwsHeader, JwtPayload, SdJwt};
#[cfg(feature = "std")]
pub use verify::{verify, verify_from_header};

#[cfg(feature = "credential")]
pub mod claims;

#[cfg(feature = "credential")]
pub use claims::{map_claims, verify_disclosures, VerifiedDisclosures};
