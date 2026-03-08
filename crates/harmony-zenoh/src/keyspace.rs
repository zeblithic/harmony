//! Harmony key expression hierarchy and builders.
//!
//! Defines the standard key expression patterns used throughout Harmony:
//!
//! | Pattern | Purpose |
//! |---|---|
//! | `harmony/server/{server_id}/channel/{channel_id}/msg` | Channel messages |
//! | `harmony/dm/{user_a}/{user_b}/msg` | Direct messages |
//! | `harmony/presence/{server_id}/{user_id}` | Online presence |
//! | `geo/s2/{l4}/{l8}/{l12}/{l16}/{rns_hash}` | Geospatial routing |
//! | `lookup/rns/{node_hash}` | Distributed node lookup |
//!
//! Each pattern is exposed both as a [`KeFormat`] for building/parsing key
//! expressions and as convenience builder functions.

use alloc::{format, string::{String, ToString}};
use zenoh_keyexpr::key_expr::format::KeFormat;
use zenoh_keyexpr::key_expr::OwnedKeyExpr;

use crate::error::ZenohError;

// ── Format strings ───────────────────────────────────────────────────

/// Channel message: `harmony/server/{server_id}/channel/{channel_id}/msg`
const CHANNEL_MSG_FMT: &str = "harmony/server/${server_id:*}/channel/${channel_id:*}/msg";

/// Direct message: `harmony/dm/{user_a}/{user_b}/msg`
const DM_FMT: &str = "harmony/dm/${user_a:*}/${user_b:*}/msg";

/// Presence: `harmony/presence/{server_id}/{user_id}`
const PRESENCE_FMT: &str = "harmony/presence/${server_id:*}/${user_id:*}";

/// Geospatial: `geo/s2/{l4}/{l8}/{l12}/{l16}/{rns_hash}`
const GEO_S2_FMT: &str = "geo/s2/${l4:*}/${l8:*}/${l12:*}/${l16:*}/${rns_hash:*}";

/// Node lookup: `lookup/rns/{node_hash}`
const NODE_LOOKUP_FMT: &str = "lookup/rns/${node_hash:*}";

// ── Wildcard subscription patterns ───────────────────────────────────

/// Subscribe to all messages on a specific server's channel.
pub fn channel_msg_sub(server_id: &str) -> Result<OwnedKeyExpr, ZenohError> {
    reject_slashes(server_id)?;
    ke(&format!("harmony/server/{server_id}/channel/*/msg"))
}

/// Subscribe to all messages on a specific channel.
pub fn channel_msg_sub_exact(
    server_id: &str,
    channel_id: &str,
) -> Result<OwnedKeyExpr, ZenohError> {
    reject_slashes(server_id)?;
    reject_slashes(channel_id)?;
    ke(&format!(
        "harmony/server/{server_id}/channel/{channel_id}/msg"
    ))
}

/// Subscribe to all messages on all channels across all servers.
pub fn channel_msg_sub_all() -> Result<OwnedKeyExpr, ZenohError> {
    ke("harmony/server/*/channel/*/msg")
}

/// Subscribe to all direct messages involving a user (either side).
/// Returns two subscription patterns since DMs are keyed by `{user_a}/{user_b}`.
pub fn dm_sub_all_for_user(user_id: &str) -> Result<[OwnedKeyExpr; 2], ZenohError> {
    reject_slashes(user_id)?;
    Ok([
        ke(&format!("harmony/dm/{user_id}/*/msg"))?,
        ke(&format!("harmony/dm/*/{user_id}/msg"))?,
    ])
}

/// Subscribe to presence on a specific server.
pub fn presence_sub(server_id: &str) -> Result<OwnedKeyExpr, ZenohError> {
    reject_slashes(server_id)?;
    ke(&format!("harmony/presence/{server_id}/*"))
}

/// Subscribe to all presence across all servers.
pub fn presence_sub_all() -> Result<OwnedKeyExpr, ZenohError> {
    ke("harmony/presence/*/*")
}

/// Subscribe to geospatial announcements within an S2 level-4 cell.
pub fn geo_s2_sub_l4(l4: &str) -> Result<OwnedKeyExpr, ZenohError> {
    reject_slashes(l4)?;
    ke(&format!("geo/s2/{l4}/**"))
}

/// Subscribe to geospatial announcements within an S2 level-8 cell.
pub fn geo_s2_sub_l8(l4: &str, l8: &str) -> Result<OwnedKeyExpr, ZenohError> {
    reject_slashes(l4)?;
    reject_slashes(l8)?;
    ke(&format!("geo/s2/{l4}/{l8}/**"))
}

// ── Builder functions ────────────────────────────────────────────────

/// Build a channel message key expression.
pub fn channel_msg_key(server_id: &str, channel_id: &str) -> Result<OwnedKeyExpr, ZenohError> {
    let fmt = KeFormat::new(CHANNEL_MSG_FMT).map_err(|e| ze(e.to_string()))?;
    let mut f = fmt.formatter();
    f.set("server_id", server_id)
        .map_err(|e| ze(e.to_string()))?;
    f.set("channel_id", channel_id)
        .map_err(|e| ze(e.to_string()))?;
    f.build().map_err(|e| ze(e.to_string()))
}

/// Build a direct message key expression.
///
/// The two user IDs are sorted lexicographically so that the same DM
/// conversation always maps to the same key regardless of who initiates.
pub fn dm_key(user_a: &str, user_b: &str) -> Result<OwnedKeyExpr, ZenohError> {
    let (lo, hi) = if user_a <= user_b {
        (user_a, user_b)
    } else {
        (user_b, user_a)
    };
    let fmt = KeFormat::new(DM_FMT).map_err(|e| ze(e.to_string()))?;
    let mut f = fmt.formatter();
    f.set("user_a", lo).map_err(|e| ze(e.to_string()))?;
    f.set("user_b", hi).map_err(|e| ze(e.to_string()))?;
    f.build().map_err(|e| ze(e.to_string()))
}

/// Build a presence key expression.
pub fn presence_key(server_id: &str, user_id: &str) -> Result<OwnedKeyExpr, ZenohError> {
    let fmt = KeFormat::new(PRESENCE_FMT).map_err(|e| ze(e.to_string()))?;
    let mut f = fmt.formatter();
    f.set("server_id", server_id)
        .map_err(|e| ze(e.to_string()))?;
    f.set("user_id", user_id).map_err(|e| ze(e.to_string()))?;
    f.build().map_err(|e| ze(e.to_string()))
}

/// Build a geospatial key expression.
pub fn geo_s2_key(
    l4: &str,
    l8: &str,
    l12: &str,
    l16: &str,
    rns_hash: &str,
) -> Result<OwnedKeyExpr, ZenohError> {
    let fmt = KeFormat::new(GEO_S2_FMT).map_err(|e| ze(e.to_string()))?;
    let mut f = fmt.formatter();
    f.set("l4", l4).map_err(|e| ze(e.to_string()))?;
    f.set("l8", l8).map_err(|e| ze(e.to_string()))?;
    f.set("l12", l12).map_err(|e| ze(e.to_string()))?;
    f.set("l16", l16).map_err(|e| ze(e.to_string()))?;
    f.set("rns_hash", rns_hash).map_err(|e| ze(e.to_string()))?;
    f.build().map_err(|e| ze(e.to_string()))
}

/// Build a node lookup key expression.
pub fn node_lookup_key(node_hash: &str) -> Result<OwnedKeyExpr, ZenohError> {
    let fmt = KeFormat::new(NODE_LOOKUP_FMT).map_err(|e| ze(e.to_string()))?;
    let mut f = fmt.formatter();
    f.set("node_hash", node_hash)
        .map_err(|e| ze(e.to_string()))?;
    f.build().map_err(|e| ze(e.to_string()))
}

// ── Vine key expressions ─────────────────────────────────────────────

/// Build a vine announcement key expression.
///
/// Pattern: `harmony/vines/{address_hex}/announce/{bundle_cid_hex}`
pub fn vine_announce_key(
    creator_addr_hex: &str,
    bundle_cid_hex: &str,
) -> Result<OwnedKeyExpr, ZenohError> {
    reject_slashes(creator_addr_hex)?;
    reject_slashes(bundle_cid_hex)?;
    ke(&format!(
        "harmony/vines/{creator_addr_hex}/announce/{bundle_cid_hex}"
    ))
}

/// Build a vine announcement subscription pattern.
///
/// Pattern: `harmony/vines/{address_hex}/announce/**`
pub fn vine_announce_sub(creator_addr_hex: &str) -> Result<OwnedKeyExpr, ZenohError> {
    reject_slashes(creator_addr_hex)?;
    ke(&format!(
        "harmony/vines/{creator_addr_hex}/announce/**"
    ))
}

/// Build a vine reaction subscription pattern.
///
/// Pattern: `harmony/vines/{target_creator_hex}/reactions/**`
pub fn vine_reaction_sub(target_creator_hex: &str) -> Result<OwnedKeyExpr, ZenohError> {
    reject_slashes(target_creator_hex)?;
    ke(&format!(
        "harmony/vines/{target_creator_hex}/reactions/**"
    ))
}

/// Build a vine reaction key expression.
///
/// Pattern: `harmony/vines/{target_creator_hex}/reactions/{bundle_cid_hex}/{reactor_hex}`
pub fn vine_reaction_key(
    target_creator_hex: &str,
    bundle_cid_hex: &str,
    reactor_hex: &str,
) -> Result<OwnedKeyExpr, ZenohError> {
    reject_slashes(target_creator_hex)?;
    reject_slashes(bundle_cid_hex)?;
    reject_slashes(reactor_hex)?;
    ke(&format!(
        "harmony/vines/{target_creator_hex}/reactions/{bundle_cid_hex}/{reactor_hex}"
    ))
}

/// Build a vine compilation subscription pattern.
///
/// Pattern: `harmony/vines/{address_hex}/compilations/**`
pub fn vine_compilation_sub(creator_addr_hex: &str) -> Result<OwnedKeyExpr, ZenohError> {
    reject_slashes(creator_addr_hex)?;
    ke(&format!(
        "harmony/vines/{creator_addr_hex}/compilations/**"
    ))
}

/// Build a vine compilation key expression.
///
/// Pattern: `harmony/vines/{address_hex}/compilations/{cid_hex}`
pub fn vine_compilation_key(
    creator_addr_hex: &str,
    compilation_cid_hex: &str,
) -> Result<OwnedKeyExpr, ZenohError> {
    reject_slashes(creator_addr_hex)?;
    reject_slashes(compilation_cid_hex)?;
    ke(&format!(
        "harmony/vines/{creator_addr_hex}/compilations/{compilation_cid_hex}"
    ))
}

// ── Parsing helpers ──────────────────────────────────────────────────

/// Parse a channel message key expression, returning (server_id, channel_id).
pub fn parse_channel_msg(ke: &keyexpr) -> Result<(String, String), ZenohError> {
    let fmt = KeFormat::new(CHANNEL_MSG_FMT).map_err(|e| ze(e.to_string()))?;
    let parsed = fmt.parse(ke).map_err(|e| ze(e.to_string()))?;
    let server_id = parsed
        .get("server_id")
        .map_err(|e| ze(e.to_string()))?
        .to_string();
    let channel_id = parsed
        .get("channel_id")
        .map_err(|e| ze(e.to_string()))?
        .to_string();
    Ok((server_id, channel_id))
}

/// Parse a presence key expression, returning (server_id, user_id).
pub fn parse_presence(ke: &keyexpr) -> Result<(String, String), ZenohError> {
    let fmt = KeFormat::new(PRESENCE_FMT).map_err(|e| ze(e.to_string()))?;
    let parsed = fmt.parse(ke).map_err(|e| ze(e.to_string()))?;
    let server_id = parsed
        .get("server_id")
        .map_err(|e| ze(e.to_string()))?
        .to_string();
    let user_id = parsed
        .get("user_id")
        .map_err(|e| ze(e.to_string()))?
        .to_string();
    Ok((server_id, user_id))
}

/// Parse a node lookup key expression, returning the node_hash.
pub fn parse_node_lookup(ke: &keyexpr) -> Result<String, ZenohError> {
    let fmt = KeFormat::new(NODE_LOOKUP_FMT).map_err(|e| ze(e.to_string()))?;
    let parsed = fmt.parse(ke).map_err(|e| ze(e.to_string()))?;
    Ok(parsed
        .get("node_hash")
        .map_err(|e| ze(e.to_string()))?
        .to_string())
}

// ── Re-export the keyexpr type for consumers ─────────────────────────

pub use zenoh_keyexpr::key_expr::keyexpr;

// ── Internal helpers ─────────────────────────────────────────────────

/// Shorthand to construct a validated OwnedKeyExpr.
fn ke(s: &str) -> Result<OwnedKeyExpr, ZenohError> {
    OwnedKeyExpr::autocanonize(s.to_string()).map_err(|e| ze(e.to_string()))
}

/// Shorthand to construct a ZenohError::InvalidKeyExpr.
fn ze(msg: String) -> ZenohError {
    ZenohError::InvalidKeyExpr(msg)
}

/// Reject field values containing `/` which would silently create extra path
/// segments in format!-based subscription patterns.
fn reject_slashes(field: &str) -> Result<(), ZenohError> {
    if field.contains('/') {
        return Err(ZenohError::InvalidKeyExpr(format!(
            "field value must not contain '/': {field:?}"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Builder tests ────────────────────────────────────────────────

    #[test]
    fn channel_msg_key_builds_correctly() {
        let ke = channel_msg_key("srv1", "general").unwrap();
        assert_eq!(ke.as_str(), "harmony/server/srv1/channel/general/msg");
    }

    #[test]
    fn dm_key_sorts_users_lexicographically() {
        let ke1 = dm_key("alice", "bob").unwrap();
        let ke2 = dm_key("bob", "alice").unwrap();
        assert_eq!(ke1, ke2, "DM key should be the same regardless of order");
        assert_eq!(ke1.as_str(), "harmony/dm/alice/bob/msg");
    }

    #[test]
    fn presence_key_builds_correctly() {
        let ke = presence_key("srv1", "alice").unwrap();
        assert_eq!(ke.as_str(), "harmony/presence/srv1/alice");
    }

    #[test]
    fn geo_s2_key_builds_correctly() {
        let ke = geo_s2_key("3", "94", "1234", "5678", "abc123").unwrap();
        assert_eq!(ke.as_str(), "geo/s2/3/94/1234/5678/abc123");
    }

    #[test]
    fn node_lookup_key_builds_correctly() {
        let ke = node_lookup_key("deadbeef").unwrap();
        assert_eq!(ke.as_str(), "lookup/rns/deadbeef");
    }

    // ── Wildcard subscription tests ──────────────────────────────────

    #[test]
    fn channel_msg_sub_matches_all_channels_on_server() {
        let sub = channel_msg_sub("srv1").unwrap();
        let msg = channel_msg_key("srv1", "general").unwrap();
        assert!(sub.intersects(&msg));

        let msg2 = channel_msg_key("srv1", "random").unwrap();
        assert!(sub.intersects(&msg2));

        // Different server should NOT match
        let msg3 = channel_msg_key("srv2", "general").unwrap();
        assert!(!sub.intersects(&msg3));
    }

    #[test]
    fn channel_msg_sub_all_matches_any_server_and_channel() {
        let sub = channel_msg_sub_all().unwrap();
        let msg1 = channel_msg_key("srv1", "general").unwrap();
        let msg2 = channel_msg_key("srv2", "random").unwrap();
        assert!(sub.intersects(&msg1));
        assert!(sub.intersects(&msg2));
    }

    #[test]
    fn dm_sub_all_for_user_matches_both_sides() {
        // subs[0] = harmony/dm/alice/*/msg  (alice as user_a)
        // subs[1] = harmony/dm/*/alice/msg  (alice as user_b)
        let subs = dm_sub_all_for_user("alice").unwrap();

        // alice < bob, so dm_key sorts to ("alice", "bob") → alice is user_a
        let msg_as_a = dm_key("alice", "bob").unwrap();
        // "aaa" < "alice", so dm_key sorts to ("aaa", "alice") → alice is user_b
        let msg_as_b = dm_key("aaa", "alice").unwrap();

        // At least one subscription should match each message
        assert!(
            subs[0].intersects(&msg_as_a) || subs[1].intersects(&msg_as_a),
            "should match when alice is user_a"
        );
        assert!(
            subs[0].intersects(&msg_as_b) || subs[1].intersects(&msg_as_b),
            "should match when alice is user_b"
        );
    }

    #[test]
    fn presence_sub_matches_all_users_on_server() {
        let sub = presence_sub("srv1").unwrap();
        let p1 = presence_key("srv1", "alice").unwrap();
        let p2 = presence_key("srv1", "bob").unwrap();
        assert!(sub.intersects(&p1));
        assert!(sub.intersects(&p2));

        let p3 = presence_key("srv2", "alice").unwrap();
        assert!(!sub.intersects(&p3));
    }

    #[test]
    fn geo_s2_sub_l4_matches_all_within_cell() {
        let sub = geo_s2_sub_l4("3").unwrap();
        let g1 = geo_s2_key("3", "94", "1234", "5678", "abc").unwrap();
        let g2 = geo_s2_key("3", "10", "5555", "9999", "def").unwrap();
        assert!(sub.intersects(&g1));
        assert!(sub.intersects(&g2));

        // Different L4 cell should NOT match
        let g3 = geo_s2_key("4", "94", "1234", "5678", "abc").unwrap();
        assert!(!sub.intersects(&g3));
    }

    #[test]
    fn geo_s2_sub_l8_narrows_within_cell() {
        let sub = geo_s2_sub_l8("3", "94").unwrap();
        let g1 = geo_s2_key("3", "94", "1234", "5678", "abc").unwrap();
        assert!(sub.intersects(&g1));

        let g2 = geo_s2_key("3", "10", "1234", "5678", "abc").unwrap();
        assert!(!sub.intersects(&g2));
    }

    // ── Parsing tests ────────────────────────────────────────────────

    #[test]
    fn parse_channel_msg_extracts_fields() {
        let ke_str = "harmony/server/srv1/channel/general/msg";
        let ke = keyexpr::new(ke_str).unwrap();
        let (server, channel) = parse_channel_msg(ke).unwrap();
        assert_eq!(&server, "srv1");
        assert_eq!(&channel, "general");
    }

    #[test]
    fn parse_presence_extracts_fields() {
        let ke_str = "harmony/presence/srv1/alice";
        let ke = keyexpr::new(ke_str).unwrap();
        let (server, user) = parse_presence(ke).unwrap();
        assert_eq!(&server, "srv1");
        assert_eq!(&user, "alice");
    }

    #[test]
    fn parse_node_lookup_extracts_hash() {
        let ke_str = "lookup/rns/deadbeef01234567";
        let ke = keyexpr::new(ke_str).unwrap();
        let hash = parse_node_lookup(ke).unwrap();
        assert_eq!(&hash, "deadbeef01234567");
    }

    // ── Roundtrip tests ──────────────────────────────────────────────

    #[test]
    fn channel_msg_build_then_parse_roundtrips() {
        let built = channel_msg_key("myserver", "mychannel").unwrap();
        let (server, channel) = parse_channel_msg(&built).unwrap();
        assert_eq!(&server, "myserver");
        assert_eq!(&channel, "mychannel");
    }

    #[test]
    fn presence_build_then_parse_roundtrips() {
        let built = presence_key("srv42", "user99").unwrap();
        let (server, user) = parse_presence(&built).unwrap();
        assert_eq!(&server, "srv42");
        assert_eq!(&user, "user99");
    }

    #[test]
    fn node_lookup_build_then_parse_roundtrips() {
        let built = node_lookup_key("abcd1234").unwrap();
        let hash = parse_node_lookup(&built).unwrap();
        assert_eq!(&hash, "abcd1234");
    }

    // ── Invalid input tests ──────────────────────────────────────────

    #[test]
    fn empty_server_id_rejected() {
        // Empty fields are only valid for ** patterns, not * patterns
        assert!(channel_msg_key("", "general").is_err());
    }

    #[test]
    fn slash_in_field_rejected() {
        // Slashes in single-chunk fields are invalid
        assert!(channel_msg_key("srv/1", "general").is_err());
    }

    #[test]
    fn slash_in_subscription_field_rejected() {
        // Subscription builders must also reject slashes to stay consistent
        // with key builders — otherwise a silent pub/sub mismatch occurs.
        assert!(channel_msg_sub("srv/1").is_err());
        assert!(channel_msg_sub_exact("srv1", "chan/1").is_err());
        assert!(dm_sub_all_for_user("user/1").is_err());
        assert!(presence_sub("srv/1").is_err());
        assert!(geo_s2_sub_l4("3/4").is_err());
        assert!(geo_s2_sub_l8("3", "9/4").is_err());
    }

    #[test]
    fn wildcard_in_field_produces_wildcard_key() {
        // KeFormat accepts wildcards as valid key expressions — this is
        // correct Zenoh behavior. The result is a wildcard KE, not a
        // concrete one.
        let ke = channel_msg_key("*", "general").unwrap();
        assert!(ke.as_str().contains('*'));
    }

    // ── Vine key expression tests ───────────────────────────────────

    #[test]
    fn vine_announce_key_valid() {
        let k = vine_announce_key("aa11bb22", "cid0099").unwrap();
        assert_eq!(k.as_str(), "harmony/vines/aa11bb22/announce/cid0099");
    }

    #[test]
    fn vine_announce_sub_valid() {
        let k = vine_announce_sub("aa11bb22").unwrap();
        assert_eq!(k.as_str(), "harmony/vines/aa11bb22/announce/**");
    }

    #[test]
    fn vine_announce_sub_matches_key() {
        let sub = vine_announce_sub("aa11bb22").unwrap();
        let key = vine_announce_key("aa11bb22", "cid0099").unwrap();
        assert!(sub.intersects(&key));

        let other = vine_announce_key("cc33dd44", "cid0099").unwrap();
        assert!(!sub.intersects(&other));
    }

    #[test]
    fn vine_announce_key_rejects_slash() {
        assert!(vine_announce_key("aa/bb", "cid01").is_err());
        assert!(vine_announce_key("aabb", "cid/01").is_err());
    }

    #[test]
    fn vine_reaction_sub_valid() {
        let k = vine_reaction_sub("creator01").unwrap();
        assert_eq!(k.as_str(), "harmony/vines/creator01/reactions/**");
    }

    #[test]
    fn vine_reaction_sub_matches_key() {
        let sub = vine_reaction_sub("creator01").unwrap();
        let key = vine_reaction_key("creator01", "bundle99", "reactor02").unwrap();
        assert!(sub.intersects(&key));

        let other = vine_reaction_key("creator02", "bundle99", "reactor02").unwrap();
        assert!(!sub.intersects(&other));
    }

    #[test]
    fn vine_reaction_key_valid() {
        let k = vine_reaction_key("creator01", "bundle99", "reactor02").unwrap();
        assert_eq!(
            k.as_str(),
            "harmony/vines/creator01/reactions/bundle99/reactor02"
        );
    }

    #[test]
    fn vine_compilation_sub_valid() {
        let k = vine_compilation_sub("creator01").unwrap();
        assert_eq!(k.as_str(), "harmony/vines/creator01/compilations/**");
    }

    #[test]
    fn vine_compilation_sub_matches_key() {
        let sub = vine_compilation_sub("creator01").unwrap();
        let key = vine_compilation_key("creator01", "compilcid").unwrap();
        assert!(sub.intersects(&key));

        let other = vine_compilation_key("creator02", "compilcid").unwrap();
        assert!(!sub.intersects(&other));
    }

    #[test]
    fn vine_compilation_key_valid() {
        let k = vine_compilation_key("creator01", "compilcid").unwrap();
        assert_eq!(k.as_str(), "harmony/vines/creator01/compilations/compilcid");
    }

    #[test]
    fn vine_keys_reject_slashes() {
        assert!(vine_announce_sub("aa/bb").is_err());
        assert!(vine_reaction_sub("cre/ator").is_err());
        assert!(vine_reaction_key("cre/ator", "bundle", "reactor").is_err());
        assert!(vine_reaction_key("creator", "bun/dle", "reactor").is_err());
        assert!(vine_reaction_key("creator", "bundle", "rea/ctor").is_err());
        assert!(vine_compilation_sub("cre/ator").is_err());
        assert!(vine_compilation_key("cre/ator", "cid").is_err());
        assert!(vine_compilation_key("creator", "ci/d").is_err());
    }
}
