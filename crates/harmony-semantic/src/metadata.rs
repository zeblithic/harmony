// SPDX-License-Identifier: Apache-2.0 OR MIT
//! Sidecar metadata — privacy tier and content annotations.

use alloc::collections::BTreeMap;
use alloc::string::String;
use alloc::vec::Vec;

/// Privacy tier for content indexing.
///
/// Used by Oluo (indexing decisions) and Jain (gate decisions).
/// Tiers are ordered from most-public to most-private.
#[derive(
    Debug,
    Clone,
    Copy,
    Default,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Hash,
    serde::Serialize,
    serde::Deserialize,
)]
pub enum PrivacyTier {
    /// Public, durable content — freely indexed and cached.
    #[default]
    PublicDurable = 0,
    /// Public, ephemeral content — indexed but not cached long-term.
    PublicEphemeral = 1,
    /// Encrypted, durable content — stored but not indexed in the clear.
    EncryptedDurable = 2,
    /// Encrypted, ephemeral content — neither indexed nor cached.
    EncryptedEphemeral = 3,
}

/// Sidecar metadata — parsed from the CBOR trailer.
///
/// All fields are optional; a v1 sidecar (no trailer) yields the
/// `Default` value (all `None`).
#[derive(Debug, Clone, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct SidecarMetadata {
    /// Privacy tier controlling indexing and gate behaviour.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub privacy_tier: Option<PrivacyTier>,
    /// Creation timestamp (Unix seconds).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub created_at: Option<u64>,
    /// MIME content type (e.g. "text/plain").
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content_type: Option<String>,
    /// BCP-47 language tag (e.g. "en-US").
    #[serde(skip_serializing_if = "Option::is_none")]
    pub language: Option<String>,
    /// Geographic coordinates (latitude, longitude).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub geo: Option<(f64, f64)>,
    /// Human-readable description.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// Free-form tags for categorisation.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tags: Option<Vec<String>>,
    /// References to other content CIDs.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub refs: Option<Vec<[u8; 32]>>,
    /// Source device identifier.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source_device: Option<String>,
    /// Extension map for future fields.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ext: Option<BTreeMap<String, Vec<u8>>>,
}
