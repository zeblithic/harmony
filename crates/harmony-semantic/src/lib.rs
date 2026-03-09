// SPDX-License-Identifier: Apache-2.0 OR MIT
//! Harmony Semantic Index — binary vector embeddings, sidecar format,
//! and collection blobs for content-addressed semantic search.

#![no_std]
extern crate alloc;

pub mod collection;
pub mod distance;
pub mod error;
pub mod fingerprint;
pub mod metadata;
pub mod overlay;
pub mod quantize;
pub mod sidecar;
pub mod tier;

pub use collection::{CollectionBlob, CollectionEntry};
pub use distance::{hamming_distance, hamming_distance_fast};
pub use error::{SemanticError, SemanticResult};
pub use fingerprint::{model_fingerprint, ModelFingerprint};
pub use metadata::{PrivacyTier, SidecarMetadata};
pub use overlay::{merge_metadata, select_embedding_header};
pub use quantize::{pack_tiers, quantize_to_binary};
pub use sidecar::{SidecarHeader, SIDECAR_HEADER_SIZE, SIDECAR_V1_MAGIC, SIDECAR_V2_MAGIC};
pub use tier::EmbeddingTier;

#[cfg(feature = "std")]
pub use sidecar::EnrichedSidecar;
