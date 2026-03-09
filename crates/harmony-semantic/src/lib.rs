// SPDX-License-Identifier: Apache-2.0 OR MIT
//! Harmony Semantic Index — binary vector embeddings, sidecar format,
//! and collection blobs for content-addressed semantic search.

#![no_std]
extern crate alloc;

pub mod error;
pub mod tier;

pub use error::{SemanticError, SemanticResult};
pub use tier::EmbeddingTier;
