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
    /// The entire `cid_hex` is lowercased to match the convention used by
    /// `hex::encode` (always lowercase). This ensures keys are consistent
    /// regardless of the caller's hex casing.
    /// If `cid_hex` is empty, returns a key with no shard segment (caller error).
    pub fn fetch_key(cid_hex: &str) -> String {
        let lower = cid_hex.to_ascii_lowercase();
        let prefix = lower.get(..1).unwrap_or("");
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
    /// Base prefix: `harmony/announce`
    pub const PREFIX: &str = "harmony/announce";

    /// Subscribe to all announcements: `harmony/announce/*`
    pub const SUB: &str = "harmony/announce/*";

    /// Announcement key: `harmony/announce/{cid_hex}`
    pub fn key(cid_hex: &str) -> String {
        format!("{PREFIX}/{cid_hex}")
    }
}

// ── Tier 3: Compute ─────────────────────────────────────────────────

/// WASM compute key expressions.
pub mod compute {
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
    /// Base prefix: `harmony/community`
    pub const PREFIX: &str = "harmony/community";

    /// Channel key: `harmony/community/{hub_id}/channels/{channel_name}`
    pub fn channel_key(hub_id: &str, channel_name: &str) -> String {
        format!("{PREFIX}/{hub_id}/channels/{channel_name}")
    }
}

/// Direct message inbox key expressions.
pub mod messages {
    /// Base prefix: `harmony/messages`
    pub const PREFIX: &str = "harmony/messages";

    /// Inbox key: `harmony/messages/{peer_addr}/inbox`
    pub fn inbox_key(peer_addr: &str) -> String {
        format!("{PREFIX}/{peer_addr}/inbox")
    }
}

/// Peer profile key expressions.
pub mod profile {
    /// Base prefix: `harmony/profile`
    pub const PREFIX: &str = "harmony/profile";

    /// Subscribe to all profiles: `harmony/profile/*`
    pub const SUB: &str = "harmony/profile/*";

    /// Profile key: `harmony/profile/{peer_addr}`
    pub fn key(peer_addr: &str) -> String {
        format!("{PREFIX}/{peer_addr}")
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
    fn content_fetch_key_uses_first_char_as_shard() {
        let key = content::fetch_key("abc123def456");
        assert_eq!(key, "harmony/content/a/abc123def456");
    }

    #[test]
    fn content_fetch_key_normalizes_to_lowercase() {
        // Uppercase hex must be fully lowercased to match hex::encode convention.
        let key = content::fetch_key("ABC123DEF456");
        assert_eq!(key, "harmony/content/a/abc123def456");
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

    // ── Cross-tier consistency ───────────────────────────────────

    #[test]
    fn all_prefixes_start_with_root() {
        let prefixes = [
            reticulum::PREFIX,
            content::PREFIX,
            announce::PREFIX,
            compute::PREFIX,
            workflow::PREFIX,
            presence::PREFIX,
            community::PREFIX,
            messages::PREFIX,
            profile::PREFIX,
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
        let shard_prefix = "harmony/content/a/";
        assert!(
            patterns
                .iter()
                .any(|p| p.starts_with(&shard_prefix[..shard_prefix.len() - 1])),
            "fetch key shard should match a registered pattern"
        );
        assert!(key.starts_with(shard_prefix));
    }
}
