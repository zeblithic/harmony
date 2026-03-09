// SPDX-License-Identifier: Apache-2.0 OR MIT
//! Search scope and query definitions.

use harmony_semantic::EmbeddingTier;

/// The scope of a search query.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SearchScope {
    /// Search only the local node's personal index.
    Personal,
    /// Search within a specific community's shared index.
    Community,
    /// Search across the entire reachable network.
    NetworkWide,
}

/// A search query submitted to the Oluo engine.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SearchQuery {
    /// Binary embedding vector at Tier 3 (256-bit, 32 bytes).
    pub embedding: [u8; 32],
    /// The tier to use for distance comparison.
    pub tier: EmbeddingTier,
    /// Scope of the search.
    pub scope: SearchScope,
    /// Maximum number of results to return.
    pub max_results: u32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn search_scope_equality() {
        assert_eq!(SearchScope::Personal, SearchScope::Personal);
        assert_eq!(SearchScope::Community, SearchScope::Community);
        assert_eq!(SearchScope::NetworkWide, SearchScope::NetworkWide);

        assert_ne!(SearchScope::Personal, SearchScope::Community);
        assert_ne!(SearchScope::Community, SearchScope::NetworkWide);
        assert_ne!(SearchScope::Personal, SearchScope::NetworkWide);
    }

    #[test]
    fn search_query_construction() {
        let embedding = [0xAB; 32];
        let query = SearchQuery {
            embedding,
            tier: EmbeddingTier::T3,
            scope: SearchScope::Community,
            max_results: 10,
        };

        assert_eq!(query.embedding, [0xAB; 32]);
        assert_eq!(query.tier, EmbeddingTier::T3);
        assert_eq!(query.scope, SearchScope::Community);
        assert_eq!(query.max_results, 10);
    }
}
