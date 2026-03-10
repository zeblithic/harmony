// SPDX-License-Identifier: Apache-2.0 OR MIT
//! Athenaeum — 32-bit content-addressed page system.
//!
//! Translates 256-bit CID-addressed blobs into 32-bit-addressed
//! 4KB pages optimized for CPU cache lines and register widths.

#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

mod addr;
mod athenaeum;
// TODO: re-enable after migration to PageAddr
// mod encyclopedia;
mod hash;
mod volume;

pub use addr::{
    Algorithm, PageAddr, ALGO_COUNT, BOOK_MAX_SIZE, NULL_PAGE, PAGES_PER_BOOK, PAGE_SIZE,
};
pub use athenaeum::{Book, BookError};
// TODO: re-enable after migration to PageAddr
// pub use encyclopedia::{Encyclopedia, SPLIT_THRESHOLD};
pub use hash::{sha224_hash, sha256_hash};
pub use volume::{route_chunk, Volume, MAX_PARTITION_DEPTH};

/// Legacy alias for backward compatibility during transition.
pub const MAX_BLOB_SIZE: usize = BOOK_MAX_SIZE;
