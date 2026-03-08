//! Legacy Zenoh content bridge — superseded by [`crate::storage_tier::StorageTier`].
//!
//! This module re-exports the canonical key expression helpers from
//! `harmony_zenoh::namespace` for callers that haven't migrated to StorageTier.

use alloc::string::String;

pub use harmony_zenoh::namespace::content::all_shard_patterns as content_queryable_key_exprs;

use crate::cid::ContentId;

/// Convert a CID to its content key expression.
///
/// Delegates to [`harmony_zenoh::namespace::content::fetch_key`].
pub fn cid_to_key_expr(cid: &ContentId) -> String {
    let hex_cid = hex::encode(cid.to_bytes());
    harmony_zenoh::namespace::content::fetch_key(&hex_cid)
}

/// Convert a CID to its announce key expression.
///
/// Delegates to [`harmony_zenoh::namespace::announce::key`].
pub fn cid_to_announce_key_expr(cid: &ContentId) -> String {
    let hex_cid = hex::encode(cid.to_bytes());
    harmony_zenoh::namespace::announce::key(&hex_cid)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cid_key_expr_matches_shard_structure() {
        let cid = ContentId::for_blob(b"shard routing test", crate::cid::ContentFlags::default())
            .unwrap();
        let key_expr = cid_to_key_expr(&cid);
        assert!(key_expr.starts_with("harmony/content/"));

        let shards = content_queryable_key_exprs();
        assert_eq!(shards.len(), 16);
        assert!(shards[0].starts_with("harmony/content/0/"));
    }

    #[test]
    fn announce_key_format() {
        let cid =
            ContentId::for_blob(b"announce test", crate::cid::ContentFlags::default()).unwrap();
        let key = cid_to_announce_key_expr(&cid);
        assert!(key.starts_with("harmony/announce/"));
    }
}
