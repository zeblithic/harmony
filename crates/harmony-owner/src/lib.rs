//! Two-tier owner→device identity binding.
//!
//! Spec: `docs/superpowers/specs/2026-04-25-harmony-owner-device-binding-design.md`.

pub mod cbor;
pub mod certs;
pub mod crdt;
pub mod error;
pub mod lifecycle;
pub mod pubkey_bundle;
pub mod signing;
pub mod state;
pub mod trust;

pub use error::OwnerError;
