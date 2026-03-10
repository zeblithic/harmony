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
// mod book;
// mod encyclopedia;
mod hash;
// TODO: re-enable after migration to PageAddr
// mod volume;

pub use addr::{
    Algorithm, PageAddr, ALGO_COUNT, BOOK_MAX_SIZE, NULL_PAGE, PAGES_PER_BOOK, PAGE_SIZE,
};
pub use athenaeum::{Book, BookError};
// TODO: re-enable after migration to PageAddr
// pub use book::{Book, BookEntry, BookError};
// pub use encyclopedia::{Encyclopedia, SPLIT_THRESHOLD};
pub use hash::{sha224_hash, sha256_hash};
// TODO: re-enable after migration to PageAddr
// pub use volume::{route_chunk, Volume, MAX_PARTITION_DEPTH};
