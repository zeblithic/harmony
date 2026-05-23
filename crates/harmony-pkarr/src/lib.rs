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

pub mod derive;
pub mod epoch;
pub mod error;
pub mod record;
#[cfg(any(test, feature = "test-fixtures"))]
pub mod testing;

pub use derive::{derive_ephemeral_key, PkarrCase};
pub use epoch::{current_epoch_id, epoch_start_ms, epoch_tolerance_window, EPOCH_DURATION_MS};
pub use error::PkarrError;
pub use record::{PkarrRoutingRecord, SKEW_TOLERANCE_MS};
