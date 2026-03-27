//! Node Trinity Zenoh key expression namespace.
//!
//! Typed constants and builders for the three-tier node architecture:
//!
//! | Tier | Prefix | Purpose |
//! |------|--------|---------|
//! | 1 | `harmony/reticulum/` | Router: packet forwarding, announces, links |
//! | 2 | `harmony/content/`, `harmony/announce/` | Storage: CAS fetch, caching, availability |
//! | 3 | `harmony/compute/`, `harmony/workflow/` | Compute: WASM execution, durable workflows |
//! | — | `harmony/presence/`, `harmony/community/`, etc. | Cross-tier and application layer |
//!
//! All queries work identically whether local, LAN, or remote — the
//! [`Locality`] enum tags replies with their origin for preference ordering.

/// Root prefix for all Harmony key expressions.
pub const ROOT: &str = "harmony";

// ── Locality ────────────────────────────────────────────────────────

/// Where a query reply originated.
///
/// Ordered so that `Local < Lan < Remote` — callers can prefer the
/// closest source with `min()`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Locality {
    /// From this node's own storage or compute.
    Local = 0,
    /// From a peer on the local network.
    Lan = 1,
    /// From a remote peer over the internet.
    Remote = 2,
}

impl Locality {
    /// Returns `true` if the reply came from this node.
    pub fn is_local(self) -> bool {
        self == Locality::Local
    }
}

// ── Tier 1: Router (Reticulum) ──────────────────────────────────────

/// Reticulum router key expressions.
pub mod reticulum {
    use alloc::{format, string::String};
    /// Base prefix: `harmony/reticulum`
    pub const PREFIX: &str = "harmony/reticulum";

    /// Path announcement prefix: `harmony/reticulum/announce`
    pub const ANNOUNCE: &str = "harmony/reticulum/announce";

    /// Bidirectional link prefix: `harmony/reticulum/link`
    pub const LINK: &str = "harmony/reticulum/link";

    /// Router diagnostics prefix: `harmony/reticulum/diagnostics`
    pub const DIAGNOSTICS: &str = "harmony/reticulum/diagnostics";

    // ── Subscription patterns ───────────────────────────────────

    /// Subscribe to all announces: `harmony/reticulum/announce/*`
    pub const ANNOUNCE_SUB: &str = "harmony/reticulum/announce/*";

    /// Subscribe to all link data: `harmony/reticulum/link/*`
    pub const LINK_SUB: &str = "harmony/reticulum/link/*";

    // ── Builders ────────────────────────────────────────────────

    /// `harmony/reticulum/announce/{dest_hash}`
    pub fn announce_key(dest_hash: &str) -> String {
        format!("{ANNOUNCE}/{dest_hash}")
    }

    /// `harmony/reticulum/link/{link_id}`
    pub fn link_key(link_id: &str) -> String {
        format!("{LINK}/{link_id}")
    }

    /// `harmony/reticulum/diagnostics/{node_addr}`
    pub fn diagnostics_key(node_addr: &str) -> String {
        format!("{DIAGNOSTICS}/{node_addr}")
    }
}

// ── Tier 2: Storage ─────────────────────────────────────────────────

/// Content storage key expressions.
///
/// Canonical source for all content-tier Zenoh key patterns. The
/// `harmony-content` crate depends on `harmony-zenoh` and re-exports
/// these helpers via `zenoh_bridge` and `StorageTier`.
pub mod content {
    use alloc::{format, string::String, vec::Vec};
    /// Base prefix: `harmony/content`
    pub const PREFIX: &str = "harmony/content";

    /// Explicit store request prefix: `harmony/content/publish`
    pub const PUBLISH: &str = "harmony/content/publish";

    /// Opportunistic cache transit prefix: `harmony/content/transit`
    pub const TRANSIT: &str = "harmony/content/transit";

    /// Cache metrics prefix: `harmony/content/stats`
    pub const STATS: &str = "harmony/content/stats";

    // ── Subscription patterns ───────────────────────────────────

    /// Subscribe to transit traffic: `harmony/content/transit/**`
    pub const TRANSIT_SUB: &str = "harmony/content/transit/**";

    /// Subscribe to publish requests: `harmony/content/publish/*`
    pub const PUBLISH_SUB: &str = "harmony/content/publish/*";

    /// The 16 hex-digit shard prefixes.
    pub const HEX_PREFIXES: [char; 16] = [
        '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f',
    ];

    // ── Builders ────────────────────────────────────────────────

    /// Shard queryable pattern: `harmony/content/{prefix}/**`
    pub fn shard_pattern(prefix: char) -> String {
        format!("{PREFIX}/{prefix}/**")
    }

    /// All 16 shard queryable patterns.
    pub fn all_shard_patterns() -> Vec<String> {
        HEX_PREFIXES.iter().map(|p| shard_pattern(*p)).collect()
    }

    /// Content fetch key: `harmony/content/{prefix}/{cid_hex}`
    ///
    /// The shard prefix is the **second** hex character of the CID (low nibble
    /// of byte 0).  The first hex character encodes ContentFlags (encrypted,
    /// ephemeral, alt_hash) and would cluster all default-flagged content into
    /// 2 of 16 shards.  The second character is pure hash bits — uniformly
    /// distributed regardless of flag combination.
    ///
    /// `cid_hex` is lowercased to match `hex::encode` convention.
    pub fn fetch_key(cid_hex: &str) -> String {
        let lower = cid_hex.to_ascii_lowercase();
        let prefix = lower.get(1..2).unwrap_or("");
        format!("{PREFIX}/{prefix}/{lower}")
    }

    /// Publish request key: `harmony/content/publish/{cid_hex}`
    pub fn publish_key(cid_hex: &str) -> String {
        format!("{PUBLISH}/{cid_hex}")
    }

    /// Transit key: `harmony/content/transit/{cid_hex}`
    pub fn transit_key(cid_hex: &str) -> String {
        format!("{TRANSIT}/{cid_hex}")
    }

    /// Stats queryable key: `harmony/content/stats/{node_addr}`
    pub fn stats_key(node_addr: &str) -> String {
        format!("{STATS}/{node_addr}")
    }
}

/// Content availability announcements.
pub mod announce {
    use alloc::{format, string::String};
    /// Base prefix: `harmony/announce`
    pub const PREFIX: &str = "harmony/announce";

    /// Subscribe to all announcements: `harmony/announce/*`
    pub const SUB: &str = "harmony/announce/*";

    /// Announcement key: `harmony/announce/{cid_hex}`
    pub fn key(cid_hex: &str) -> String {
        format!("{PREFIX}/{cid_hex}")
    }
}

/// Content filter broadcasts.
///
/// Nodes periodically broadcast Bloom filters of their cached CID set.
/// Receiving nodes check these filters before dispatching content queries,
/// skipping peers that definitely don't have the content.
pub mod filters {
    use alloc::{format, string::String};
    /// Base prefix: `harmony/filters`
    pub const PREFIX: &str = "harmony/filters";

    /// Content filter prefix: `harmony/filters/content`
    pub const CONTENT_PREFIX: &str = "harmony/filters/content";

    /// Subscribe to all content filters: `harmony/filters/content/**`
    pub const CONTENT_SUB: &str = "harmony/filters/content/**";

    /// Content filter key: `harmony/filters/content/{node_addr}`
    pub fn content_key(node_addr: &str) -> String {
        format!("{CONTENT_PREFIX}/{node_addr}")
    }

    /// Flatpack filter prefix: `harmony/filters/flatpack`
    pub const FLATPACK_PREFIX: &str = "harmony/filters/flatpack";

    /// Subscribe to all flatpack filters: `harmony/filters/flatpack/**`
    pub const FLATPACK_SUB: &str = "harmony/filters/flatpack/**";

    /// Flatpack filter key: `harmony/filters/flatpack/{node_addr}`
    pub fn flatpack_key(node_addr: &str) -> String {
        format!("{FLATPACK_PREFIX}/{node_addr}")
    }

    /// Memo filter prefix: `harmony/filters/memo`
    pub const MEMO_PREFIX: &str = "harmony/filters/memo";

    /// Subscribe to all memo filters: `harmony/filters/memo/**`
    pub const MEMO_SUB: &str = "harmony/filters/memo/**";

    /// Memo filter key: `harmony/filters/memo/{node_addr}`
    pub fn memo_key(node_addr: &str) -> String {
        format!("{MEMO_PREFIX}/{node_addr}")
    }

    /// Page filter prefix: `harmony/filters/page`
    pub const PAGE_PREFIX: &str = "harmony/filters/page";

    /// Subscribe to all page filters: `harmony/filters/page/**`
    pub const PAGE_SUB: &str = "harmony/filters/page/**";

    /// Page filter key: `harmony/filters/page/{node_addr}`
    pub fn page_key(node_addr: &str) -> String {
        format!("{PAGE_PREFIX}/{node_addr}")
    }
}

/// Flatpack reverse-lookup query key expressions.
///
/// Nodes query `harmony/flatpack/{child_cid_hex}` to discover which
/// bundles reference a given child CID. Peers with matching index
/// entries respond with their bundle CID lists.
///
/// These constants define the query/response path for on-demand reverse
/// lookups. The queryable handler and subscription wiring are not yet
/// implemented — this is scaffolding for a follow-on PR that adds the
/// request-response path. The filter broadcast path
/// (`filters::flatpack_key`) is already wired.
pub mod flatpack {
    use alloc::{format, string::String};
    /// Base prefix: `harmony/flatpack`
    pub const PREFIX: &str = "harmony/flatpack";

    /// Subscribe to all flatpack queries: `harmony/flatpack/**`
    pub const SUB: &str = "harmony/flatpack/**";

    /// Flatpack query key: `harmony/flatpack/{child_cid_hex}`
    pub fn query_key(child_cid_hex: &str) -> String {
        format!("{PREFIX}/{child_cid_hex}")
    }
}

// ── Tier 3: Compute ─────────────────────────────────────────────────

/// WASM compute key expressions.
pub mod compute {
    use alloc::{format, string::String};
    /// Base prefix: `harmony/compute`
    pub const PREFIX: &str = "harmony/compute";

    /// Execute activity prefix: `harmony/compute/activity`
    pub const ACTIVITY: &str = "harmony/compute/activity";

    /// Computation results prefix: `harmony/compute/result`
    pub const RESULT: &str = "harmony/compute/result";

    /// Advertised capacity prefix: `harmony/compute/capacity`
    pub const CAPACITY: &str = "harmony/compute/capacity";

    // ── Subscription patterns ───────────────────────────────────

    /// Subscribe to all activity requests: `harmony/compute/activity/*`
    pub const ACTIVITY_SUB: &str = "harmony/compute/activity/*";

    /// Key expression for inference queryable.
    pub const INFERENCE_ACTIVITY: &str = "harmony/compute/activity/inference";

    /// Base prefix for DSD verification queryable.
    pub const VERIFY_ACTIVITY: &str = "harmony/compute/activity/verify";

    /// Per-node verify key: `harmony/compute/activity/verify/{node_addr}`
    pub fn verify_key(node_addr: &str) -> String {
        format!("{VERIFY_ACTIVITY}/{node_addr}")
    }

    /// Key expression for DSD speculative inference queryable.
    pub const SPECULATIVE_ACTIVITY: &str = "harmony/compute/activity/speculative";

    /// Subscribe to all capacity advertisements: `harmony/compute/capacity/*`
    pub const CAPACITY_SUB: &str = "harmony/compute/capacity/*";

    /// Subscribe to all results: `harmony/compute/result/*`
    pub const RESULT_SUB: &str = "harmony/compute/result/*";

    // ── Builders ────────────────────────────────────────────────

    /// Activity queryable key: `harmony/compute/activity/{activity_type}`
    pub fn activity_key(activity_type: &str) -> String {
        format!("{ACTIVITY}/{activity_type}")
    }

    /// Result key: `harmony/compute/result/{workflow_id}`
    pub fn result_key(workflow_id: &str) -> String {
        format!("{RESULT}/{workflow_id}")
    }

    /// Capacity key: `harmony/compute/capacity/{node_addr}`
    pub fn capacity_key(node_addr: &str) -> String {
        format!("{CAPACITY}/{node_addr}")
    }
}

/// Durable workflow execution key expressions.
pub mod workflow {
    use alloc::{format, string::String};
    /// Base prefix: `harmony/workflow`
    pub const PREFIX: &str = "harmony/workflow";

    // ── Builders ────────────────────────────────────────────────

    /// Checkpoint key: `harmony/workflow/{workflow_id}/checkpoint`
    pub fn checkpoint_key(workflow_id: &str) -> String {
        format!("{PREFIX}/{workflow_id}/checkpoint")
    }

    /// History prefix: `harmony/workflow/{workflow_id}/history`
    pub fn history_key(workflow_id: &str) -> String {
        format!("{PREFIX}/{workflow_id}/history")
    }

    /// Subscribe to all history for a workflow: `harmony/workflow/{workflow_id}/history/**`
    pub fn history_sub(workflow_id: &str) -> String {
        format!("{PREFIX}/{workflow_id}/history/**")
    }
}

// ── Cross-tier: Presence ────────────────────────────────────────────

/// Liveliness token key expressions.
pub mod presence {
    use alloc::{format, string::String};
    /// Base prefix: `harmony/presence`
    pub const PREFIX: &str = "harmony/presence";

    /// Subscribe to all presence: `harmony/presence/**`
    pub const SUB: &str = "harmony/presence/**";

    /// Presence token key: `harmony/presence/{community_id}/{peer_addr}`
    pub fn token_key(community_id: &str, peer_addr: &str) -> String {
        format!("{PREFIX}/{community_id}/{peer_addr}")
    }

    /// Subscribe to all presence in a community: `harmony/presence/{community_id}/*`
    pub fn community_sub(community_id: &str) -> String {
        format!("{PREFIX}/{community_id}/*")
    }
}

// ── Application layer ───────────────────────────────────────────────

/// Community hub key expressions.
pub mod community {
    use alloc::{format, string::String};
    /// Base prefix: `harmony/community`
    pub const PREFIX: &str = "harmony/community";

    /// Channel key: `harmony/community/{hub_id}/channels/{channel_name}`
    pub fn channel_key(hub_id: &str, channel_name: &str) -> String {
        format!("{PREFIX}/{hub_id}/channels/{channel_name}")
    }
}

/// Direct message inbox key expressions.
pub mod messages {
    use alloc::{format, string::String};
    /// Base prefix: `harmony/messages`
    pub const PREFIX: &str = "harmony/messages";

    /// Inbox key: `harmony/messages/{peer_addr}/inbox`
    pub fn inbox_key(peer_addr: &str) -> String {
        format!("{PREFIX}/{peer_addr}/inbox")
    }
}

/// Peer profile key expressions.
pub mod profile {
    use alloc::{format, string::String};
    /// Base prefix: `harmony/profile`
    pub const PREFIX: &str = "harmony/profile";

    /// Subscribe to all profiles: `harmony/profile/*`
    pub const SUB: &str = "harmony/profile/*";

    /// Profile key: `harmony/profile/{peer_addr}`
    pub fn key(peer_addr: &str) -> String {
        format!("{PREFIX}/{peer_addr}")
    }
}

/// Identity announce and resolve key expressions.
pub mod identity {
    use alloc::{format, string::String};

    pub const PREFIX: &str = "harmony/identity";

    /// Subscribe to all identity announces.
    pub const ALL_ANNOUNCES: &str = "harmony/identity/*/announce";

    /// Register a queryable for all identity resolve requests.
    pub const ALL_RESOLVES: &str = "harmony/identity/*/resolve";

    /// Subscribe to all identity presence changes.
    pub const ALL_ALIVE: &str = "harmony/identity/*/alive";

    /// Key expression for a specific identity's announce channel.
    ///
    /// `address_hex` must be the 32-character lowercase hex encoding of
    /// the 16-byte `IdentityHash` (e.g. `"aa00bb11cc22dd33ee44ff5566778899"`).
    pub fn announce_key(address_hex: &str) -> String {
        format!("{PREFIX}/{address_hex}/announce")
    }

    /// Key expression for a specific identity's resolve endpoint.
    ///
    /// `address_hex` must be the 32-character lowercase hex encoding of
    /// the 16-byte `IdentityHash`.
    pub fn resolve_key(address_hex: &str) -> String {
        format!("{PREFIX}/{address_hex}/resolve")
    }

    /// Key expression for a specific identity's liveliness token.
    ///
    /// `address_hex` must be the 32-character lowercase hex encoding of
    /// the 16-byte `IdentityHash`.
    pub fn alive_key(address_hex: &str) -> String {
        format!("{PREFIX}/{address_hex}/alive")
    }

    /// Zenoh key expressions for the did:web gateway.
    pub mod web {
        use alloc::format;
        use alloc::string::String;

        /// Wildcard matching all did:web gateway queries.
        pub const ALL: &str = "harmony/identity/web/**";

        /// Build a key expression for a specific did:web DID.
        pub fn key(domain: &str, path: &str) -> String {
            if path.is_empty() {
                format!("{}/web/{}", super::PREFIX, domain)
            } else {
                format!("{}/web/{}/{}", super::PREFIX, domain, path)
            }
        }
    }
}

/// Authenticated discovery queryable namespace.
///
/// Peers query `harmony/discover/{identity_hash_hex}` with a PQ UCAN token
/// to retrieve full routing hints (including tunnel addresses). Without a
/// valid token, the queryable responds with public hints only.
///
/// Distinct from `harmony/identity/` which handles DID document resolution.
pub mod discover {
    use alloc::{format, string::String};

    /// Base prefix: `harmony/discover`
    pub const PREFIX: &str = "harmony/discover";

    /// Subscribe/queryable pattern: `harmony/discover/**`
    pub const SUB: &str = "harmony/discover/**";

    /// Key for a specific identity: `harmony/discover/{identity_hash_hex}`
    pub fn key(identity_hash_hex: &str) -> String {
        format!("{PREFIX}/{identity_hash_hex}")
    }
}

/// Endorsement record key expressions.
///
/// Endorsements are published at `harmony/endorsement/{endorser_hex}/{endorsee_hex}`.
/// Both `address_hex` parameters must be the 32-character lowercase hex encoding
/// of the 16-byte `IdentityHash`.
pub mod endorsement {
    use alloc::{format, string::String};

    pub const PREFIX: &str = "harmony/endorsement";

    /// All endorsements by a specific endorser.
    ///
    /// `endorser_hex` must be the 32-character lowercase hex encoding
    /// of the 16-byte `IdentityHash`.
    pub fn by_endorser(endorser_hex: &str) -> String {
        format!("{PREFIX}/{endorser_hex}/*")
    }

    /// All endorsements of a specific endorsee (query-only pattern).
    ///
    /// This pattern uses `*` in a non-terminal position, which works
    /// for Zenoh `session.get()` queries but may not be registerable
    /// as a subscriber in Zenoh 1.x. For subscription use cases,
    /// subscribe to all endorsements and filter at the application layer.
    ///
    /// `endorsee_hex` must be the 32-character lowercase hex encoding
    /// of the 16-byte `IdentityHash`.
    pub fn of_endorsee(endorsee_hex: &str) -> String {
        format!("{PREFIX}/*/{endorsee_hex}")
    }

    /// Key expression for a specific endorsement.
    ///
    /// Both parameters must be the 32-character lowercase hex encoding
    /// of the 16-byte `IdentityHash`.
    pub fn key(endorser_hex: &str, endorsee_hex: &str) -> String {
        format!("{PREFIX}/{endorser_hex}/{endorsee_hex}")
    }
}

/// Signed memo attestations for deterministic computation results.
///
/// Namespace: `harmony/memo/<input_cid_hex>/<output_cid_hex>/sign/<signer_id_hex>`
///
/// Each path level encodes the attestation: input (computation recipe),
/// output (result), signer (who verified). Multiple outputs under the same
/// input means disagreement — structurally visible in the path topology.
pub mod memo {
    use alloc::{format, string::String};

    pub const PREFIX: &str = "harmony/memo";
    pub const SUB: &str = "harmony/memo/**";

    /// Full attestation key: `harmony/memo/{input}/{output}/sign/{signer}`
    pub fn sign_key(input_hex: &str, output_hex: &str, signer_hex: &str) -> String {
        format!("{PREFIX}/{input_hex}/{output_hex}/sign/{signer_hex}")
    }

    /// All memos for an input: `harmony/memo/{input}/**`
    pub fn input_query(input_hex: &str) -> String {
        format!("{PREFIX}/{input_hex}/**")
    }

    /// All signers for a specific input→output: `harmony/memo/{input}/{output}/sign/**`
    pub fn output_query(input_hex: &str, output_hex: &str) -> String {
        format!("{PREFIX}/{input_hex}/{output_hex}/sign/**")
    }

    /// What does a specific signer say about this input: `harmony/memo/{input}/*/sign/{signer}`
    ///
    /// This pattern uses `*` in a non-terminal position, which works for
    /// Zenoh `session.get()` queries but may not be registerable as a
    /// subscriber in Zenoh 1.x. For subscription use cases, subscribe to
    /// `harmony/memo/{input}/**` and filter at the application layer.
    pub fn signer_query(input_hex: &str, signer_hex: &str) -> String {
        format!("{PREFIX}/{input_hex}/*/sign/{signer_hex}")
    }
}

/// Page-first content addressing namespace.
///
/// Path: `harmony/page/<addr_00>/<addr_01>/<addr_10>/<addr_11>/<book_cid>/<page_num>`
///
/// Each page has 4 algorithm-variant PageAddr values (32-bit each, 8 hex chars):
/// - `addr_00`: MSB SHA-256 (mode bits = 00)
/// - `addr_01`: LSB SHA-256 (mode bits = 01)
/// - `addr_10`: SHA-224 MSB (mode bits = 10)
/// - `addr_11`: SHA-224 LSB (mode bits = 11)
///
/// The `book_cid` is the 256-bit ContentId of the containing book (64 hex chars).
/// The `page_num` is the page's position within the book (0-255, decimal).
///
/// Coexists with `harmony/book/{cid}/page/{page_addr}` (book-first access in keyspace.rs).
/// This namespace enables page-first discovery: "who has this page address?"
///
/// All addresses are canonical lowercase hex.
pub mod page {
    use alloc::{format, string::String};

    pub const PREFIX: &str = "harmony/page";
    pub const SUB: &str = "harmony/page/**";

    /// Full page key: `harmony/page/{00}/{01}/{10}/{11}/{book_cid}/{page_num}`
    pub fn page_key(
        addr_00: &str,
        addr_01: &str,
        addr_10: &str,
        addr_11: &str,
        book_cid: &str,
        page_num: u8,
    ) -> String {
        format!("{PREFIX}/{addr_00}/{addr_01}/{addr_10}/{addr_11}/{book_cid}/{page_num}")
    }

    /// Query by mode-00 address only (MSB SHA-256): `harmony/page/{addr}/*/*/*/*/*`
    pub fn query_by_addr00(addr_00: &str) -> String {
        format!("{PREFIX}/{addr_00}/*/*/*/*/*")
    }

    /// Query by mode-00 + mode-01: `harmony/page/{00}/{01}/*/*/*/*`
    pub fn query_by_addr00_01(addr_00: &str, addr_01: &str) -> String {
        format!("{PREFIX}/{addr_00}/{addr_01}/*/*/*/*")
    }

    /// Query by all 4 address variants: `harmony/page/{00}/{01}/{10}/{11}/*/*`
    pub fn query_by_all_addrs(
        addr_00: &str,
        addr_01: &str,
        addr_10: &str,
        addr_11: &str,
    ) -> String {
        format!("{PREFIX}/{addr_00}/{addr_01}/{addr_10}/{addr_11}/*/*")
    }

    /// Query by all 4 addresses + book: `harmony/page/{00}/{01}/{10}/{11}/{book_cid}/*`
    pub fn query_by_all_addrs_and_book(
        addr_00: &str,
        addr_01: &str,
        addr_10: &str,
        addr_11: &str,
        book_cid: &str,
    ) -> String {
        format!("{PREFIX}/{addr_00}/{addr_01}/{addr_10}/{addr_11}/{book_cid}/*")
    }

    /// Query all pages of a book: `harmony/page/*/*/*/*/{book_cid}/*`
    ///
    /// Uses `*` in non-terminal positions — works for Zenoh `session.get()`
    /// queries but may not be registerable as a subscriber in Zenoh 1.x.
    pub fn query_by_book(book_cid: &str) -> String {
        format!("{PREFIX}/*/*/*/*/{book_cid}/*")
    }

    /// Query specific page by book + position: `harmony/page/*/*/*/*/{book_cid}/{page_num}`
    ///
    /// Uses `*` in non-terminal positions — works for Zenoh `session.get()`
    /// queries but may not be registerable as a subscriber in Zenoh 1.x.
    pub fn query_by_book_and_pos(book_cid: &str, page_num: u8) -> String {
        format!("{PREFIX}/*/*/*/*/{book_cid}/{page_num}")
    }
}

/// Engram conditional memory table key expressions.
///
/// Namespace: `harmony/engram/{version}/`
///
/// Engram tables are sharded embedding tables stored as CAS books.
/// Nodes hosting shards declare queryables on `harmony/engram/{version}/shard/**`.
/// Any node holding a cached shard can respond.
pub mod engram {
    use alloc::{format, string::String};

    /// Base prefix: `harmony/engram`
    pub const PREFIX: &str = "harmony/engram";

    /// Subscribe to all engram traffic: `harmony/engram/**`
    pub const SUB: &str = "harmony/engram/**";

    // ── Builders ────────────────────────────────────────────────

    /// Manifest key: `harmony/engram/{version}/manifest`
    pub fn manifest_key(version: &str) -> String {
        format!("{PREFIX}/{version}/manifest")
    }

    /// Individual shard key: `harmony/engram/{version}/shard/{index}`
    pub fn shard_key(version: &str, index: u64) -> String {
        format!("{PREFIX}/{version}/shard/{index}")
    }

    /// Shard queryable pattern: `harmony/engram/{version}/shard/**`
    ///
    /// Nodes hosting Engram shards register this pattern.
    pub fn shard_queryable(version: &str) -> String {
        format!("{PREFIX}/{version}/shard/**")
    }
}

// ── Agent Protocol ──────────────────────────────────────────────────

/// Agent-to-agent messaging key expressions.
pub mod agent {
    use alloc::{format, string::String};

    /// Base prefix: `harmony/agent`
    pub const PREFIX: &str = "harmony/agent";

    // ── Subscription patterns ───────────────────────────────────

    /// Subscribe to all agent capacity advertisements: `harmony/agent/*/capacity`
    pub fn capacity_sub_all() -> String {
        format!("{PREFIX}/*/capacity")
    }

    // ── Builders ────────────────────────────────────────────────

    /// Capacity key: `harmony/agent/{agent_id}/capacity`
    pub fn capacity_key(agent_id: &str) -> String {
        format!("{PREFIX}/{agent_id}/capacity")
    }

    /// Task submission endpoint: `harmony/agent/{agent_id}/task`
    pub fn task_key(agent_id: &str) -> String {
        format!("{PREFIX}/{agent_id}/task")
    }

    /// Per-task stream key: `harmony/agent/{agent_id}/stream/{task_id}`
    ///
    /// Agents publish `StreamChunk` messages here for long-running tasks.
    /// Requesters subscribe before sending the task query.
    pub fn stream_key(agent_id: &str, task_id: &str) -> String {
        format!("{PREFIX}/{agent_id}/stream/{task_id}")
    }

    /// Subscribe to all streams from one agent: `harmony/agent/{agent_id}/stream/*`
    ///
    /// Agent-scoped subscription — takes `agent_id` as parameter (unlike
    /// `capacity_sub_all()` which is cross-agent). Stream subscriptions are
    /// always scoped to a specific agent.
    pub fn stream_sub(agent_id: &str) -> String {
        format!("{PREFIX}/{agent_id}/stream/*")
    }
}

// ── Telemetry ──────────────────────────────────────────────────────

/// Telemetry event key expressions.
///
/// Nodes publish structured `TelemetryEvent` messages to these topics.
/// Intents are free-form strings (e.g. `"health"`, `"anomaly"`,
/// `"object_detected"`). Node addresses are lowercase hex.
pub mod telemetry {
    use alloc::{format, string::String};

    /// Base prefix: `harmony/telemetry`
    pub const PREFIX: &str = "harmony/telemetry";

    /// Per-node, per-intent telemetry key: `harmony/telemetry/{node_addr}/{intent}`
    pub fn key(node_addr: &str, intent: &str) -> String {
        format!("{PREFIX}/{node_addr}/{intent}")
    }

    /// Subscribe to all intents from one node: `harmony/telemetry/{node_addr}/*`
    pub fn sub_node(node_addr: &str) -> String {
        format!("{PREFIX}/{node_addr}/*")
    }

    /// Subscribe to one intent across all nodes: `harmony/telemetry/*/{intent}`
    pub fn sub_intent(intent: &str) -> String {
        format!("{PREFIX}/*/{intent}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Locality ────────────────────────────────────────────────

    #[test]
    fn locality_ordering() {
        assert!(Locality::Local < Locality::Lan);
        assert!(Locality::Lan < Locality::Remote);

        // min() picks closest source
        let sources = [Locality::Remote, Locality::Local, Locality::Lan];
        assert_eq!(sources.iter().min(), Some(&Locality::Local));
    }

    #[test]
    fn locality_is_local() {
        assert!(Locality::Local.is_local());
        assert!(!Locality::Lan.is_local());
        assert!(!Locality::Remote.is_local());
    }

    // ── Tier 1: Reticulum ───────────────────────────────────────

    #[test]
    fn reticulum_announce_key() {
        assert_eq!(
            reticulum::announce_key("deadbeef01234567"),
            "harmony/reticulum/announce/deadbeef01234567"
        );
    }

    #[test]
    fn reticulum_link_key() {
        assert_eq!(
            reticulum::link_key("link42"),
            "harmony/reticulum/link/link42"
        );
    }

    #[test]
    fn reticulum_diagnostics_key() {
        assert_eq!(
            reticulum::diagnostics_key("abcd1234"),
            "harmony/reticulum/diagnostics/abcd1234"
        );
    }

    #[test]
    fn reticulum_subscription_patterns() {
        assert_eq!(reticulum::ANNOUNCE_SUB, "harmony/reticulum/announce/*");
        assert_eq!(reticulum::LINK_SUB, "harmony/reticulum/link/*");
    }

    // ── Tier 2: Content ─────────────────────────────────────────

    #[test]
    fn content_shard_pattern() {
        assert_eq!(content::shard_pattern('a'), "harmony/content/a/**");
        assert_eq!(content::shard_pattern('f'), "harmony/content/f/**");
    }

    #[test]
    fn content_all_shard_patterns_has_16() {
        let patterns = content::all_shard_patterns();
        assert_eq!(patterns.len(), 16);
        assert_eq!(patterns[0], "harmony/content/0/**");
        assert_eq!(patterns[10], "harmony/content/a/**");
        assert_eq!(patterns[15], "harmony/content/f/**");
    }

    #[test]
    fn content_fetch_key_uses_second_char_as_shard() {
        // Second hex char (low nibble of byte 0) is pure hash bits,
        // uniformly distributed regardless of ContentFlags.
        let key = content::fetch_key("abc123def456");
        assert_eq!(key, "harmony/content/b/abc123def456");
    }

    #[test]
    fn content_fetch_key_normalizes_to_lowercase() {
        // Uppercase hex must be fully lowercased to match hex::encode convention.
        let key = content::fetch_key("ABC123DEF456");
        assert_eq!(key, "harmony/content/b/abc123def456");
    }

    #[test]
    fn content_fetch_key_empty_does_not_panic() {
        // Empty CID hex is a caller error, but must not panic.
        let key = content::fetch_key("");
        assert_eq!(key, "harmony/content//");
    }

    #[test]
    fn content_publish_key() {
        assert_eq!(
            content::publish_key("abc123"),
            "harmony/content/publish/abc123"
        );
    }

    #[test]
    fn content_transit_key() {
        assert_eq!(
            content::transit_key("abc123"),
            "harmony/content/transit/abc123"
        );
    }

    #[test]
    fn content_stats_key() {
        assert_eq!(content::stats_key("node42"), "harmony/content/stats/node42");
    }

    #[test]
    fn content_subscription_patterns() {
        assert_eq!(content::TRANSIT_SUB, "harmony/content/transit/**");
        assert_eq!(content::PUBLISH_SUB, "harmony/content/publish/*");
    }

    // ── Tier 2: Announce ────────────────────────────────────────

    #[test]
    fn announce_key() {
        assert_eq!(announce::key("abc123"), "harmony/announce/abc123");
    }

    #[test]
    fn announce_subscription_pattern() {
        assert_eq!(announce::SUB, "harmony/announce/*");
    }

    // ── Tier 2: Filters ──────────────────────────────────────────

    #[test]
    fn filters_content_key() {
        assert_eq!(
            filters::content_key("abc123"),
            "harmony/filters/content/abc123"
        );
    }

    #[test]
    fn filters_subscription_pattern() {
        assert_eq!(filters::CONTENT_SUB, "harmony/filters/content/**");
    }

    // ── Tier 2: Flatpack Filters ─────────────────────────────────────

    #[test]
    fn filters_flatpack_key() {
        assert_eq!(
            filters::flatpack_key("abc123"),
            "harmony/filters/flatpack/abc123"
        );
    }

    #[test]
    fn filters_flatpack_subscription_pattern() {
        assert_eq!(filters::FLATPACK_SUB, "harmony/filters/flatpack/**");
    }

    // ── Tier 2: Flatpack Queries ──────────────────────────────────────

    #[test]
    fn flatpack_query_key() {
        assert_eq!(
            flatpack::query_key("abcdef1234567890"),
            "harmony/flatpack/abcdef1234567890"
        );
    }

    #[test]
    fn flatpack_subscription_pattern() {
        assert_eq!(flatpack::SUB, "harmony/flatpack/**");
    }

    // ── Tier 3: Compute ─────────────────────────────────────────

    #[test]
    fn compute_activity_key() {
        assert_eq!(
            compute::activity_key("thumbnail"),
            "harmony/compute/activity/thumbnail"
        );
    }

    #[test]
    fn compute_result_key() {
        assert_eq!(
            compute::result_key("wf-001"),
            "harmony/compute/result/wf-001"
        );
    }

    #[test]
    fn compute_capacity_key() {
        assert_eq!(
            compute::capacity_key("node42"),
            "harmony/compute/capacity/node42"
        );
    }

    #[test]
    fn compute_subscription_patterns() {
        assert_eq!(compute::ACTIVITY_SUB, "harmony/compute/activity/*");
        assert_eq!(compute::RESULT_SUB, "harmony/compute/result/*");
    }

    #[test]
    fn compute_speculative_constants() {
        assert_eq!(
            compute::VERIFY_ACTIVITY,
            "harmony/compute/activity/verify"
        );
        assert_eq!(
            compute::SPECULATIVE_ACTIVITY,
            "harmony/compute/activity/speculative"
        );
        assert_eq!(
            compute::CAPACITY_SUB,
            "harmony/compute/capacity/*"
        );
    }

    // ── Tier 3: Workflow ────────────────────────────────────────

    #[test]
    fn workflow_checkpoint_key() {
        assert_eq!(
            workflow::checkpoint_key("wf-001"),
            "harmony/workflow/wf-001/checkpoint"
        );
    }

    #[test]
    fn workflow_history_key() {
        assert_eq!(
            workflow::history_key("wf-001"),
            "harmony/workflow/wf-001/history"
        );
    }

    #[test]
    fn workflow_history_sub() {
        assert_eq!(
            workflow::history_sub("wf-001"),
            "harmony/workflow/wf-001/history/**"
        );
    }

    // ── Cross-tier: Presence ────────────────────────────────────

    #[test]
    fn presence_token_key() {
        assert_eq!(
            presence::token_key("hub1", "peer42"),
            "harmony/presence/hub1/peer42"
        );
    }

    #[test]
    fn presence_community_sub() {
        assert_eq!(presence::community_sub("hub1"), "harmony/presence/hub1/*");
    }

    #[test]
    fn presence_global_sub() {
        assert_eq!(presence::SUB, "harmony/presence/**");
    }

    // ── Application layer ───────────────────────────────────────

    #[test]
    fn community_channel_key() {
        assert_eq!(
            community::channel_key("hub1", "general"),
            "harmony/community/hub1/channels/general"
        );
    }

    #[test]
    fn messages_inbox_key() {
        assert_eq!(
            messages::inbox_key("peer42"),
            "harmony/messages/peer42/inbox"
        );
    }

    #[test]
    fn profile_key() {
        assert_eq!(profile::key("peer42"), "harmony/profile/peer42");
    }

    #[test]
    fn profile_subscription_pattern() {
        assert_eq!(profile::SUB, "harmony/profile/*");
    }

    // ── Identity ────────────────────────────────────────────────

    #[test]
    fn identity_announce_key_format() {
        let key = identity::announce_key("aa00bb11cc22dd33ee44ff5566778899");
        assert_eq!(
            key,
            "harmony/identity/aa00bb11cc22dd33ee44ff5566778899/announce"
        );
    }

    #[test]
    fn identity_resolve_key_format() {
        let key = identity::resolve_key("aa00bb11cc22dd33ee44ff5566778899");
        assert_eq!(
            key,
            "harmony/identity/aa00bb11cc22dd33ee44ff5566778899/resolve"
        );
    }

    #[test]
    fn identity_alive_key_format() {
        let key = identity::alive_key("aa00bb11cc22dd33ee44ff5566778899");
        assert_eq!(
            key,
            "harmony/identity/aa00bb11cc22dd33ee44ff5566778899/alive"
        );
    }

    #[test]
    fn identity_wildcard_constants() {
        assert_eq!(identity::ALL_ANNOUNCES, "harmony/identity/*/announce");
        assert_eq!(identity::ALL_RESOLVES, "harmony/identity/*/resolve");
        assert_eq!(identity::ALL_ALIVE, "harmony/identity/*/alive");
    }

    #[test]
    fn identity_web_key_root() {
        assert_eq!(
            identity::web::key("example.com", ""),
            "harmony/identity/web/example.com"
        );
    }

    #[test]
    fn identity_web_key_with_path() {
        assert_eq!(
            identity::web::key("example.com", "issuers/1"),
            "harmony/identity/web/example.com/issuers/1"
        );
    }

    // ── Discover ────────────────────────────────────────────────

    #[test]
    fn discover_key() {
        assert_eq!(discover::key("aabbccdd"), "harmony/discover/aabbccdd");
    }

    #[test]
    fn discover_subscription_pattern() {
        assert_eq!(discover::SUB, "harmony/discover/**");
    }

    // ── Endorsement ──

    #[test]
    fn endorsement_key_format() {
        let key = endorsement::key(
            "aa00bb11cc22dd33ee44ff5566778899",
            "1122334455667788aabbccddeeff0011",
        );
        assert_eq!(
            key,
            "harmony/endorsement/aa00bb11cc22dd33ee44ff5566778899/1122334455667788aabbccddeeff0011"
        );
    }

    #[test]
    fn endorsement_by_endorser_format() {
        let pattern = endorsement::by_endorser("aa00bb11cc22dd33ee44ff5566778899");
        assert_eq!(
            pattern,
            "harmony/endorsement/aa00bb11cc22dd33ee44ff5566778899/*"
        );
    }

    #[test]
    fn endorsement_of_endorsee_format() {
        let pattern = endorsement::of_endorsee("aa00bb11cc22dd33ee44ff5566778899");
        assert_eq!(
            pattern,
            "harmony/endorsement/*/aa00bb11cc22dd33ee44ff5566778899"
        );
    }

    // ── Memo ──────────────────────────────────────────────────────

    #[test]
    fn memo_sign_key() {
        assert_eq!(
            memo::sign_key("aabbccdd", "11223344", "deadbeef"),
            "harmony/memo/aabbccdd/11223344/sign/deadbeef"
        );
    }

    #[test]
    fn memo_input_query() {
        assert_eq!(memo::input_query("aabbccdd"), "harmony/memo/aabbccdd/**");
    }

    #[test]
    fn memo_output_query() {
        assert_eq!(
            memo::output_query("aabbccdd", "11223344"),
            "harmony/memo/aabbccdd/11223344/sign/**"
        );
    }

    #[test]
    fn memo_signer_query() {
        assert_eq!(
            memo::signer_query("aabbccdd", "deadbeef"),
            "harmony/memo/aabbccdd/*/sign/deadbeef"
        );
    }

    #[test]
    fn memo_subscription_pattern() {
        assert_eq!(memo::SUB, "harmony/memo/**");
    }

    #[test]
    fn filters_memo_key() {
        assert_eq!(filters::memo_key("node42"), "harmony/filters/memo/node42");
    }

    #[test]
    fn filters_memo_subscription_pattern() {
        assert_eq!(filters::MEMO_SUB, "harmony/filters/memo/**");
    }

    // ── Page namespace ──

    #[test]
    fn page_key_format() {
        let key = page::page_key(
            "aabb0011",
            "ccdd2233",
            "eeff4455",
            "00116677",
            &"ff".repeat(32),
            42,
        );
        assert_eq!(
            key,
            format!(
                "harmony/page/aabb0011/ccdd2233/eeff4455/00116677/{}/42",
                "ff".repeat(32)
            )
        );
    }

    #[test]
    fn page_key_page_zero() {
        let key = page::page_key(
            "00000000",
            "11111111",
            "22222222",
            "33333333",
            &"aa".repeat(32),
            0,
        );
        assert!(key.ends_with("/0"));
    }

    #[test]
    fn page_key_page_255() {
        let key = page::page_key(
            "00000000",
            "11111111",
            "22222222",
            "33333333",
            &"aa".repeat(32),
            255,
        );
        assert!(key.ends_with("/255"));
    }

    #[test]
    fn page_query_by_addr00() {
        let q = page::query_by_addr00("aabb0011");
        assert_eq!(q, "harmony/page/aabb0011/*/*/*/*/*");
        assert_eq!(q.matches('*').count(), 5);
    }

    #[test]
    fn page_query_by_addr00_01() {
        let q = page::query_by_addr00_01("aabb0011", "ccdd2233");
        assert_eq!(q, "harmony/page/aabb0011/ccdd2233/*/*/*/*");
        assert_eq!(q.matches('*').count(), 4);
    }

    #[test]
    fn page_query_by_all_addrs() {
        let q = page::query_by_all_addrs("aabb0011", "ccdd2233", "eeff4455", "00116677");
        assert_eq!(q, "harmony/page/aabb0011/ccdd2233/eeff4455/00116677/*/*");
        assert_eq!(q.matches('*').count(), 2);
    }

    #[test]
    fn page_query_by_all_addrs_and_book() {
        let cid = "dd".repeat(32);
        let q =
            page::query_by_all_addrs_and_book("aabb0011", "ccdd2233", "eeff4455", "00116677", &cid);
        assert_eq!(
            q,
            format!("harmony/page/aabb0011/ccdd2233/eeff4455/00116677/{cid}/*")
        );
        assert_eq!(q.matches('*').count(), 1);
    }

    #[test]
    fn page_query_by_book() {
        let cid = "bb".repeat(32);
        let q = page::query_by_book(&cid);
        assert_eq!(q, format!("harmony/page/*/*/*/*/{cid}/*"));
        assert_eq!(q.matches('*').count(), 5);
    }

    #[test]
    fn page_query_by_book_and_pos() {
        let cid = "cc".repeat(32);
        let q = page::query_by_book_and_pos(&cid, 100);
        assert_eq!(q, format!("harmony/page/*/*/*/*/{cid}/100"));
        assert_eq!(q.matches('*').count(), 4);
    }

    #[test]
    fn page_subscription_pattern() {
        assert_eq!(page::SUB, "harmony/page/**");
    }

    // ── Cross-tier consistency ───────────────────────────────────

    #[test]
    fn all_prefixes_start_with_root() {
        let prefixes = [
            reticulum::PREFIX,
            content::PREFIX,
            announce::PREFIX,
            filters::PREFIX,
            flatpack::PREFIX,
            compute::PREFIX,
            workflow::PREFIX,
            presence::PREFIX,
            community::PREFIX,
            messages::PREFIX,
            profile::PREFIX,
            identity::PREFIX,
            endorsement::PREFIX,
            memo::PREFIX,
            page::PREFIX,
            engram::PREFIX,
            telemetry::PREFIX,
        ];
        for prefix in prefixes {
            assert!(
                prefix.starts_with(ROOT),
                "{prefix} should start with {ROOT}"
            );
        }
    }

    #[test]
    fn shard_fetch_key_matches_shard_pattern() {
        // A fetch key's shard prefix must fall within one of the 16 shard patterns.
        let key = content::fetch_key("abc123def456");
        let patterns = content::all_shard_patterns();
        // Second hex char 'b' determines the shard.
        let shard_prefix = "harmony/content/b/";
        assert!(
            patterns
                .iter()
                .any(|p| p.starts_with(&shard_prefix[..shard_prefix.len() - 1])),
            "fetch key shard should match a registered pattern"
        );
        assert!(key.starts_with(shard_prefix));
    }

    // ── Engram ───────────────────────────────────────────────────

    #[test]
    fn engram_manifest_key() {
        assert_eq!(engram::manifest_key("v1"), "harmony/engram/v1/manifest");
    }

    #[test]
    fn engram_shard_key() {
        assert_eq!(engram::shard_key("v1", 42), "harmony/engram/v1/shard/42");
    }

    #[test]
    fn engram_shard_key_zero() {
        assert_eq!(engram::shard_key("v1", 0), "harmony/engram/v1/shard/0");
    }

    #[test]
    fn engram_shard_queryable() {
        assert_eq!(engram::shard_queryable("v1"), "harmony/engram/v1/shard/**");
    }

    #[test]
    fn engram_subscription_pattern() {
        assert_eq!(engram::SUB, "harmony/engram/**");
    }

    // ── Agent ────────────────────────────────────────────────────

    #[test]
    fn agent_namespace_keys() {
        let cap = agent::capacity_key("deadbeef01020304");
        assert_eq!(cap, "harmony/agent/deadbeef01020304/capacity");

        let task = agent::task_key("deadbeef01020304");
        assert_eq!(task, "harmony/agent/deadbeef01020304/task");

        let sub = agent::capacity_sub_all();
        assert_eq!(sub, "harmony/agent/*/capacity");

        let stream = agent::stream_key("deadbeef01020304", "task-abc");
        assert_eq!(stream, "harmony/agent/deadbeef01020304/stream/task-abc");

        let stream_sub = agent::stream_sub("deadbeef01020304");
        assert_eq!(stream_sub, "harmony/agent/deadbeef01020304/stream/*");
    }

    #[test]
    fn telemetry_namespace_keys() {
        let key = telemetry::key("node01", "anomaly");
        assert_eq!(key, "harmony/telemetry/node01/anomaly");

        let sub_node = telemetry::sub_node("node01");
        assert_eq!(sub_node, "harmony/telemetry/node01/*");

        let sub_intent = telemetry::sub_intent("health");
        assert_eq!(sub_intent, "harmony/telemetry/*/health");
    }
}
