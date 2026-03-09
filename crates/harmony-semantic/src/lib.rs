// SPDX-License-Identifier: Apache-2.0 OR MIT
//! Harmony Semantic Index — binary vector embeddings, sidecar format,
//! and collection blobs for content-addressed semantic search.

#![no_std]
extern crate alloc;

pub mod error;
pub mod fingerprint;
pub mod sidecar;
pub mod tier;

pub use error::{SemanticError, SemanticResult};
pub use fingerprint::{model_fingerprint, ModelFingerprint};
pub use sidecar::{SidecarHeader, SIDECAR_HEADER_SIZE, SIDECAR_V1_MAGIC, SIDECAR_V2_MAGIC};
pub use tier::EmbeddingTier;
