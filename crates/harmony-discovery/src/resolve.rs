use harmony_identity::IdentityHash;

use crate::record::AnnounceRecord;

/// Resolve an identity's announce record from persistent storage.
///
/// Used as a fallback when no live Queryable answers. The content
/// layer can implement this by storing AnnounceRecords as content
/// objects.
///
/// Implementations should filter out expired records (`now >= expires_at`)
/// to avoid returning stale data that the caller would reject via
/// `verify_announce`.
///
/// The `DiscoveryManager` does not call this directly — the caller
/// (runtime integration layer) falls back to offline resolution
/// when a query yields no cached result.
pub trait OfflineResolver {
    fn resolve(&self, address: &IdentityHash, now: u64) -> Option<AnnounceRecord>;
}
