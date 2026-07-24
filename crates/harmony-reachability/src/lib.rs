//! Reachability announce record + a generic multi-device LWW kernel.
//!
//! Extracted from harmony-client (ZEB-744 / ZEB-571 item 7). The record is a
//! byte-stable CBOR value published to the pkarr DHT; the kernel is the generic
//! `(owner, node_id)`-keyed last-writer-wins substrate. App-specific reachability
//! policy (source arbitration, reconnect/liveness integration, pkarr refresh)
//! stays in the consuming app.
//!
//! This crate is transport- and identity-agnostic: no iroh, no pkarr, no
//! identity/crypto dependency. Signing over the record is the caller's concern.

pub mod canonical;

pub use canonical::{canonical_cbor_decode, canonical_cbor_encode, CborError};
