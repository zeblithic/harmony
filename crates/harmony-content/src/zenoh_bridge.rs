//! Zenoh content bridge — wires ContentStore to queryables and content announcements.
//!
//! Provides helper functions for prefix-sharded queryable registration,
//! content query handling, and content announcement publishing.
//! Sans-I/O: the caller feeds events and executes returned actions.

use crate::blob::BlobStore;
use crate::cid::ContentId;

/// The 16 hex-digit prefixes for sharded queryable registration.
const HEX_PREFIXES: [char; 16] = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f',
];

/// Inbound events the caller feeds into the bridge.
#[derive(Debug, Clone)]
pub enum ContentBridgeEvent {
    /// A remote peer is querying for content we might have.
    ContentQuery {
        query_id: u64,
        cid: ContentId,
    },
    /// Local store ingested new content (trigger announcement).
    ContentStored {
        cid: ContentId,
    },
}

/// Outbound actions the bridge returns for the caller to execute.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ContentBridgeAction {
    /// Reply to a content query with the blob data.
    SendReply { query_id: u64, payload: Vec<u8> },
    /// Announce that we have this content on the network.
    Publish { key_expr: String, payload: Vec<u8> },
}

/// Generate the 16 prefix-sharded key expressions for content queryables.
///
/// Returns key expressions like `harmony/content/0/**`, `harmony/content/1/**`, etc.
pub fn content_queryable_key_exprs() -> Vec<String> {
    HEX_PREFIXES
        .iter()
        .map(|prefix| format!("harmony/content/{prefix}/**"))
        .collect()
}

/// Convert a CID to its content key expression: `harmony/content/{hex_cid}`.
pub fn cid_to_key_expr(cid: &ContentId) -> String {
    format!("harmony/content/{}", hex::encode(cid.to_bytes()))
}

/// Convert a CID to its announce key expression: `harmony/announce/{hex_cid}`.
pub fn cid_to_announce_key_expr(cid: &ContentId) -> String {
    format!("harmony/announce/{}", hex::encode(cid.to_bytes()))
}

/// Handle a content query: look up the CID in the store, return SendReply if found.
pub fn handle_content_query<S: BlobStore>(
    store: &S,
    query_id: u64,
    cid: &ContentId,
) -> Vec<ContentBridgeAction> {
    match store.get(cid) {
        Some(data) => vec![ContentBridgeAction::SendReply {
            query_id,
            payload: data.to_vec(),
        }],
        None => vec![],
    }
}

/// Handle new content being stored: produce an announcement action.
pub fn handle_content_stored(cid: &ContentId, size: u32) -> Vec<ContentBridgeAction> {
    let key_expr = cid_to_announce_key_expr(cid);
    // Minimal announcement payload: just the size as 4 bytes big-endian.
    let payload = size.to_be_bytes().to_vec();
    vec![ContentBridgeAction::Publish { key_expr, payload }]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::blob::MemoryBlobStore;

    #[test]
    fn query_returns_cached_content() {
        let mut store = MemoryBlobStore::new();
        let data = b"cached blob for query test";
        let cid = store.insert(data).unwrap();

        let actions = handle_content_query(&store, 42, &cid);
        assert_eq!(actions.len(), 1);
        match &actions[0] {
            ContentBridgeAction::SendReply { query_id, payload } => {
                assert_eq!(*query_id, 42);
                assert_eq!(payload.as_slice(), data.as_slice());
            }
            other => panic!("expected SendReply, got {other:?}"),
        }
    }

    #[test]
    fn query_miss_produces_no_reply() {
        let store = MemoryBlobStore::new();
        let cid = ContentId::for_blob(b"not in store").unwrap();

        let actions = handle_content_query(&store, 99, &cid);
        assert!(actions.is_empty(), "cache miss should produce no actions");
    }

    #[test]
    fn content_stored_triggers_announcement() {
        let cid = ContentId::for_blob(b"new content to announce").unwrap();
        let size = cid.payload_size();

        let actions = handle_content_stored(&cid, size);
        assert_eq!(actions.len(), 1);
        match &actions[0] {
            ContentBridgeAction::Publish { key_expr, payload } => {
                assert!(
                    key_expr.starts_with("harmony/announce/"),
                    "key_expr should start with harmony/announce/, got: {key_expr}"
                );
                // Payload should be 4 bytes (size as big-endian u32).
                assert_eq!(payload.len(), 4);
                let announced_size = u32::from_be_bytes(payload[..4].try_into().unwrap());
                assert_eq!(announced_size, size);
            }
            other => panic!("expected Publish, got {other:?}"),
        }
    }
}
