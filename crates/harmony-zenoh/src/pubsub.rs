//! Sans-I/O pub/sub routing layer for peer-to-peer message dispatch.
//!
//! [`PubSubRouter`] manages publisher and subscriber declarations,
//! interest-based write-side filtering, and inbound message dispatch.
//! It composes with [`Session`] (peer lifecycle) and [`SubscriptionTable`]
//! (key expression matching) without owning either.

use std::collections::HashMap;

use crate::session::ExprId;
use crate::subscription::{SubscriptionId, SubscriptionTable};

/// Opaque publisher identifier.
pub type PublisherId = u64;

/// Inbound events the caller feeds into the router.
#[derive(Debug, Clone)]
pub enum PubSubEvent {
    /// Peer declared subscriber interest on a key expression.
    SubscriberDeclared { key_expr: String },
    /// Peer undeclared subscriber interest.
    SubscriberUndeclared { key_expr: String },
    /// Inbound message from peer (already decrypted).
    MessageReceived { expr_id: ExprId, payload: Vec<u8> },
}

/// Outbound actions the router returns for the caller to execute.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PubSubAction {
    /// Tell the peer we're subscribing to a key expression.
    SendSubscriberDeclare { key_expr: String },
    /// Tell the peer we're unsubscribing.
    SendSubscriberUndeclare { key_expr: String },
    /// Send a message to the peer (caller encrypts via HarmonyEnvelope).
    SendMessage { expr_id: ExprId, payload: Vec<u8> },
    /// Deliver a received message to a local subscriber.
    Deliver {
        subscription_id: SubscriptionId,
        key_expr: String,
        payload: Vec<u8>,
    },
}

/// A sans-I/O pub/sub router managing publisher/subscriber declarations
/// and message dispatch for a single peer-to-peer connection.
pub struct PubSubRouter {
    subscriptions: SubscriptionTable,
    sub_key_exprs: HashMap<SubscriptionId, String>,
    remote_interest: SubscriptionTable,
    remote_interest_ids: HashMap<String, SubscriptionId>,
    publishers: HashMap<PublisherId, ExprId>,
    next_publisher_id: PublisherId,
}

impl PubSubRouter {
    /// Create a new empty router.
    pub fn new() -> Self {
        Self {
            subscriptions: SubscriptionTable::new(),
            sub_key_exprs: HashMap::new(),
            remote_interest: SubscriptionTable::new(),
            remote_interest_ids: HashMap::new(),
            publishers: HashMap::new(),
            next_publisher_id: 1,
        }
    }
}

impl Default for PubSubRouter {
    fn default() -> Self {
        Self::new()
    }
}
