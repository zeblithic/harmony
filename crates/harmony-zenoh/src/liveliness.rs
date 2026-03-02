//! Sans-I/O liveliness token presence tracking.
//!
//! [`LivelinessRouter`] manages liveliness tokens and subscribers for
//! online presence detection without polling. It composes with
//! [`Session`] (peer lifecycle) via caller-driven events — the router
//! has no internal timers.

use std::collections::HashMap;

use zenoh_keyexpr::key_expr::OwnedKeyExpr;

use crate::error::ZenohError;
use crate::subscription::{SubscriptionId, SubscriptionTable};

/// Opaque liveliness token identifier.
pub type TokenId = u64;

/// Opaque liveliness subscriber identifier.
pub type LivelinessSubscriberId = u64;

/// Inbound events the caller feeds into the router.
#[derive(Debug, Clone)]
pub enum LivelinessEvent {
    /// Remote peer declared a liveliness token.
    TokenDeclared { key_expr: String },
    /// Remote peer undeclared a liveliness token.
    TokenUndeclared { key_expr: String },
    /// Remote peer's session died — bulk-undeclare all their tokens.
    PeerLost,
}

/// Outbound actions the router returns for the caller to execute.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LivelinessAction {
    /// Tell the peer we declared a liveliness token.
    SendTokenDeclare { key_expr: String },
    /// Tell the peer we undeclared a liveliness token.
    SendTokenUndeclare { key_expr: String },
    /// Notify a local subscriber that a token appeared.
    TokenAppeared {
        subscriber_id: LivelinessSubscriberId,
        key_expr: String,
    },
    /// Notify a local subscriber that a token disappeared.
    TokenDisappeared {
        subscriber_id: LivelinessSubscriberId,
        key_expr: String,
    },
}

/// A sans-I/O liveliness router managing token declarations, subscriber
/// notifications, and presence queries for a single peer connection.
pub struct LivelinessRouter {
    /// Remote tokens stored for matching against subscribers and queries.
    remote_tokens: SubscriptionTable,
    /// Reverse map: SubscriptionId in remote_tokens → canonical key expression.
    remote_token_keys: HashMap<SubscriptionId, String>,

    /// Local subscribers for token appear/disappear event fan-out.
    subscribers: SubscriptionTable,
    /// Map: LivelinessSubscriberId → SubscriptionId in subscribers table.
    subscriber_keys: HashMap<LivelinessSubscriberId, SubscriptionId>,

    /// Local tokens declared by this peer.
    local_tokens: HashMap<TokenId, String>,
    next_token_id: TokenId,
    next_subscriber_id: LivelinessSubscriberId,
}

impl LivelinessRouter {
    /// Create a new empty liveliness router.
    pub fn new() -> Self {
        Self {
            remote_tokens: SubscriptionTable::new(),
            remote_token_keys: HashMap::new(),
            subscribers: SubscriptionTable::new(),
            subscriber_keys: HashMap::new(),
            local_tokens: HashMap::new(),
            next_token_id: 1,
            next_subscriber_id: 1,
        }
    }
}

impl Default for LivelinessRouter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_router_is_empty() {
        let router = LivelinessRouter::new();
        assert!(router.local_tokens.is_empty());
        assert!(router.remote_token_keys.is_empty());
        assert!(router.subscriber_keys.is_empty());
    }
}
