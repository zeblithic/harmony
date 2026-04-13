// SPDX-License-Identifier: Apache-2.0 OR MIT
//! Search scope and query definitions.

use harmony_semantic::EmbeddingTier;

/// The scope of a search query.
///
/// Ordered from narrowest to broadest: Personal < Community < NetworkWide.
/// Scopes are hierarchical: a Community query includes Personal entries,
/// and a NetworkWide query includes all entries.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum SearchScope {
    /// Search only the local node's personal index.
    Personal,
    /// Search within a specific community's shared index.
    Community,
    /// Search across the entire reachable network.
    NetworkWide,
}

impl SearchScope {
    /// Index into `scope_counts` array (Personal=0, Community=1, NetworkWide=2).
    pub(crate) fn index(self) -> usize {
        match self {
            Self::Personal => 0,
            Self::Community => 1,
            Self::NetworkWide => 2,
        }
    }
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
    fn search_scope_ordering() {
        assert!(SearchScope::Personal < SearchScope::Community);
        assert!(SearchScope::Community < SearchScope::NetworkWide);
        assert!(SearchScope::Personal < SearchScope::NetworkWide);

        // Widening: max picks the broadest scope
        assert_eq!(
            SearchScope::Personal.max(SearchScope::Community),
            SearchScope::Community
        );
        assert_eq!(
            SearchScope::Community.max(SearchScope::NetworkWide),
            SearchScope::NetworkWide
        );
        assert_eq!(
            SearchScope::Personal.max(SearchScope::Personal),
            SearchScope::Personal
        );
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
