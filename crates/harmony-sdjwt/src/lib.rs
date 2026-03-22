#![cfg_attr(not(feature = "std"), no_std)]
extern crate alloc;

pub mod error;
pub mod parse;
pub mod types;
pub mod verify;

pub use error::SdJwtError;
pub use parse::signing_input;
#[cfg(feature = "std")]
pub use parse::parse;
pub use types::{Disclosure, JwsHeader, JwtPayload, SdJwt};
#[cfg(feature = "std")]
pub use verify::{verify, verify_from_header};
