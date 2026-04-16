//! Discovery-backed email → `IdentityHash` resolution for harmony-mail.
//!
//! See `docs/superpowers/specs/2026-04-15-smtp-rcpt-admission-design.md`
//! for the complete design. Module ordering mirrors the cryptographic
//! dependency order: `claim` is leaf (pure verification), `cache`,
//! `dns`, `http` compose onto it, `resolver` orchestrates.

#![deny(clippy::unwrap_used, clippy::expect_used)]
#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]
#![forbid(unsafe_code)]

pub mod cache;
pub mod claim;
pub mod dns;
pub mod http;
pub mod resolver;

#[cfg(any(test, feature = "test-support"))]
pub mod test_support;
