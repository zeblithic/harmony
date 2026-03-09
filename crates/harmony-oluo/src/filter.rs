// SPDX-License-Identifier: Apache-2.0 OR MIT
//! Retrieval filter — Jain's output filtering interface for search results.

use alloc::vec::Vec;
use harmony_semantic::metadata::SidecarMetadata;

/// A raw search result before Jain filtering.
#[derive(Debug, Clone, PartialEq)]
pub struct RawSearchResult {
    /// CID of the content.
    pub target_cid: [u8; 32],
    /// Similarity score (0.0 = identical, 1.0 = maximally different).
    pub score: f32,
    /// Merged metadata from all overlays.
    pub metadata: SidecarMetadata,
    /// CIDs of the overlay sidecars that contributed to this result.
    pub overlays: Vec<[u8; 32]>,
}

/// A filtered search result after Jain processing.
#[derive(Debug, Clone, PartialEq)]
pub struct FilteredSearchResult {
    /// CID of the content.
    pub target_cid: [u8; 32],
    /// Similarity score (0.0 = identical, 1.0 = maximally different).
    pub score: f32,
    /// Filtered metadata (Jain may redact fields).
    pub metadata: SidecarMetadata,
}

/// Context about the requester, provided to the retrieval filter.
#[derive(Debug, Clone, Default)]
pub struct RetrievalContext {
    /// Identity of the requester (if known).
    pub requester: Option<[u8; 32]>,
    /// Social context hints (community membership, etc.).
    pub social_context: Option<[u8; 32]>,
    /// Device context hints.
    pub device_context: Option<[u8; 32]>,
}

/// The retrieval filter trait — implemented by Jain, consumed by Oluo.
///
/// After search produces raw results, Jain filters them based on
/// the requester's context, social relationships, and privacy rules.
pub trait RetrievalFilter {
    /// Filter raw search results, returning only those the requester
    /// is allowed to see, with metadata potentially redacted.
    fn filter(
        &self,
        results: &[RawSearchResult],
        context: &RetrievalContext,
    ) -> Vec<FilteredSearchResult>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn retrieval_context_default() {
        let ctx = RetrievalContext::default();
        assert!(ctx.requester.is_none());
        assert!(ctx.social_context.is_none());
        assert!(ctx.device_context.is_none());
    }
}
