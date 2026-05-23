//! Pkarr (Public-Key Addressable Resource Records) publish + resolve over
//! BitTorrent Mainline DHT via HTTP-relay. Transport-agnostic at the value
//! layer — see [`record::PkarrRoutingRecord`].
//!
//! See `harmony-client/docs/specs/2026-05-23-zeb-321-phase2-discovery-bootstrap-design.md`
//! for the cohesive Phase 2 design this crate is the harmony-core half of.
//!
//! # Crates this crate intentionally does NOT depend on
//!
//! - `iroh` / `zenoh` — transport-agnostic at the value layer; downstream
//!   harmony-client wraps the opaque `routing_blob` with iroh-specific
//!   routing data.
//! - `harmony-client` — this crate ships in harmony-core for reuse by
//!   harmony-arch / harmony-glitch / etc.

#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

pub mod error;

pub use error::PkarrError;
