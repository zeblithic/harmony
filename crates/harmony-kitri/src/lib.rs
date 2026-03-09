// SPDX-License-Identifier: Apache-2.0 OR MIT
//! Kitri — Harmony's native programming model for durable distributed computation.
//!
//! Layer 0: sans-I/O core types, traits, and manifest parsing.
//! This crate defines the *protocol* of Kitri — no runtime, no I/O.

#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

pub mod error;
pub mod io;
pub mod event;
pub mod checkpoint;
pub mod manifest;
pub mod trust;
pub mod retry;
pub mod program;
pub mod dag;
pub mod staging;
pub mod compensation;

pub use error::{KitriError, KitriResult};
