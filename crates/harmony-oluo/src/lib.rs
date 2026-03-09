// SPDX-License-Identifier: Apache-2.0 OR MIT
//! Oluo — Harmony's semantic search engine.
//!
//! An adaptive-depth embedding trie with binary vector indexing,
//! content-addressed sidecars, and sans-I/O state machine design.

#![no_std]
extern crate alloc;

pub mod error;
pub mod zenoh_keys;

pub use error::{OluoError, OluoResult};
