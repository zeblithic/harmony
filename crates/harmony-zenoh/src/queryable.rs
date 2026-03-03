//! Sans-I/O queryable router for request/reply content retrieval.
//!
//! [`QueryableRouter`] manages queryable declarations and incoming query
//! dispatch. It composes with [`SubscriptionTable`] for key expression
//! matching — the caller drives all I/O.

use std::collections::HashMap;

use zenoh_keyexpr::key_expr::{keyexpr, OwnedKeyExpr};

use crate::error::ZenohError;
use crate::subscription::{SubscriptionId, SubscriptionTable};

/// Opaque queryable identifier.
pub type QueryableId = u64;

/// Opaque query identifier.
pub type QueryId = u64;

/// Inbound events the caller feeds into the router.
#[derive(Debug, Clone)]
pub enum QueryableEvent {
    /// A query was received from a remote peer.
    QueryReceived {
        query_id: QueryId,
        key_expr: String,
        payload: Vec<u8>,
    },
    /// A remote peer declared a queryable (informational).
    QueryableDeclared { key_expr: String },
    /// A remote peer undeclared a queryable (informational).
    QueryableUndeclared { key_expr: String },
}

/// Outbound actions the router returns for the caller to execute.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QueryableAction {
    /// Tell the peer we declared a queryable on a key expression.
    SendQueryableDeclare { key_expr: String },
    /// Tell the peer we undeclared a queryable.
    SendQueryableUndeclare { key_expr: String },
    /// Deliver a query to a local queryable.
    DeliverQuery {
        queryable_id: QueryableId,
        query_id: QueryId,
        key_expr: String,
        payload: Vec<u8>,
    },
    /// Send a reply to a query.
    SendReply { query_id: QueryId, payload: Vec<u8> },
}

/// A sans-I/O queryable router managing queryable declarations and
/// incoming query dispatch for request/reply content retrieval.
pub struct QueryableRouter {
    /// Local queryable registrations: QueryableId → canonical key expression.
    local_queryables: HashMap<QueryableId, OwnedKeyExpr>,
    /// Matching table for incoming queries.
    local_table: SubscriptionTable,
    /// Maps QueryableId → SubscriptionId in the local_table.
    local_sub_ids: HashMap<QueryableId, SubscriptionId>,
    /// Reverse map: SubscriptionId → QueryableId.
    sub_to_queryable: HashMap<SubscriptionId, QueryableId>,
    /// Next queryable ID to allocate.
    next_queryable_id: QueryableId,
}

impl QueryableRouter {
    /// Create a new empty queryable router.
    pub fn new() -> Self {
        Self {
            local_queryables: HashMap::new(),
            local_table: SubscriptionTable::new(),
            local_sub_ids: HashMap::new(),
            sub_to_queryable: HashMap::new(),
            next_queryable_id: 1,
        }
    }

    /// Declare a local queryable on the given key expression.
    ///
    /// Validates the key expression, allocates an ID, registers in the
    /// subscription table, and returns a `SendQueryableDeclare` action.
    pub fn declare(
        &mut self,
        key_expr: &str,
    ) -> Result<(QueryableId, Vec<QueryableAction>), ZenohError> {
        let owned = OwnedKeyExpr::autocanonize(key_expr.to_string())
            .map_err(|e| ZenohError::InvalidKeyExpr(e.to_string()))?;
        let canonical = owned.to_string();

        let qid = self.next_queryable_id;
        self.next_queryable_id += 1;

        let sub_id = self.local_table.subscribe(&owned);
        self.local_queryables.insert(qid, owned);
        self.local_sub_ids.insert(qid, sub_id);
        self.sub_to_queryable.insert(sub_id, qid);

        Ok((
            qid,
            vec![QueryableAction::SendQueryableDeclare {
                key_expr: canonical,
            }],
        ))
    }

    /// Undeclare a previously declared queryable.
    ///
    /// Removes from all maps and the subscription table, returns
    /// `SendQueryableUndeclare`. Errors with `UnknownQueryableId` if not found.
    pub fn undeclare(&mut self, id: QueryableId) -> Result<Vec<QueryableAction>, ZenohError> {
        let owned = self
            .local_queryables
            .remove(&id)
            .ok_or(ZenohError::UnknownQueryableId(id))?;
        let canonical = owned.to_string();

        if let Some(sub_id) = self.local_sub_ids.remove(&id) {
            self.sub_to_queryable.remove(&sub_id);
            let _ = self.local_table.unsubscribe(sub_id);
        }

        Ok(vec![QueryableAction::SendQueryableUndeclare {
            key_expr: canonical,
        }])
    }

    /// Build a reply action for a query.
    pub fn reply(&self, query_id: QueryId, payload: Vec<u8>) -> Vec<QueryableAction> {
        vec![QueryableAction::SendReply { query_id, payload }]
    }

    /// Process an inbound queryable event and return actions for the caller.
    pub fn handle_event(
        &mut self,
        event: QueryableEvent,
    ) -> Result<Vec<QueryableAction>, ZenohError> {
        match event {
            QueryableEvent::QueryReceived {
                query_id,
                key_expr,
                payload,
            } => self.handle_query_received(query_id, &key_expr, payload),
            // Informational events — no local actions needed for now.
            QueryableEvent::QueryableDeclared { .. }
            | QueryableEvent::QueryableUndeclared { .. } => Ok(vec![]),
        }
    }

    /// Match an incoming query against local queryables and return
    /// `DeliverQuery` actions for each match.
    fn handle_query_received(
        &self,
        query_id: QueryId,
        key_expr: &str,
        payload: Vec<u8>,
    ) -> Result<Vec<QueryableAction>, ZenohError> {
        let ke = keyexpr::new(key_expr)
            .map_err(|e| ZenohError::InvalidKeyExpr(e.to_string()))?;

        let matches = self.local_table.matches(ke);
        Ok(matches
            .into_iter()
            .filter_map(|sub_id| {
                self.sub_to_queryable
                    .get(&sub_id)
                    .map(|&qid| QueryableAction::DeliverQuery {
                        queryable_id: qid,
                        query_id,
                        key_expr: key_expr.to_string(),
                        payload: payload.clone(),
                    })
            })
            .collect())
    }
}

impl Default for QueryableRouter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn declare_and_receive_query() {
        let mut router = QueryableRouter::new();

        // Declare a queryable with multi-level wildcard
        let (qid, actions) = router.declare("harmony/content/a/**").unwrap();
        assert_eq!(actions.len(), 1);
        assert_eq!(
            actions[0],
            QueryableAction::SendQueryableDeclare {
                key_expr: "harmony/content/a/**".into(),
            }
        );

        // Feed a query that matches
        let actions = router.handle_event(QueryableEvent::QueryReceived {
            query_id: 1,
            key_expr: "harmony/content/a/abcd1234".into(),
            payload: b"fetch".to_vec(),
        })
        .unwrap();

        assert_eq!(actions.len(), 1);
        assert_eq!(
            actions[0],
            QueryableAction::DeliverQuery {
                queryable_id: qid,
                query_id: 1,
                key_expr: "harmony/content/a/abcd1234".into(),
                payload: b"fetch".to_vec(),
            }
        );
    }

    #[test]
    fn reply_sends_to_querier() {
        let router = QueryableRouter::new();
        let actions = router.reply(42, b"response-data".to_vec());
        assert_eq!(actions.len(), 1);
        assert_eq!(
            actions[0],
            QueryableAction::SendReply {
                query_id: 42,
                payload: b"response-data".to_vec(),
            }
        );
    }

    #[test]
    fn query_dispatches_to_matching_queryable() {
        let mut router = QueryableRouter::new();

        // Two queryables on different prefixes
        let (qid_a, _) = router.declare("harmony/content/a/**").unwrap();
        let (qid_b, _) = router.declare("harmony/content/b/**").unwrap();

        // Query matching only the "a" prefix
        let actions = router.handle_event(QueryableEvent::QueryReceived {
            query_id: 10,
            key_expr: "harmony/content/a/abcd".into(),
            payload: b"get-a".to_vec(),
        })
        .unwrap();
        assert_eq!(actions.len(), 1);
        assert_eq!(
            actions[0],
            QueryableAction::DeliverQuery {
                queryable_id: qid_a,
                query_id: 10,
                key_expr: "harmony/content/a/abcd".into(),
                payload: b"get-a".to_vec(),
            }
        );

        // Query matching only the "b" prefix
        let actions = router.handle_event(QueryableEvent::QueryReceived {
            query_id: 11,
            key_expr: "harmony/content/b/bcde".into(),
            payload: b"get-b".to_vec(),
        })
        .unwrap();
        assert_eq!(actions.len(), 1);
        assert_eq!(
            actions[0],
            QueryableAction::DeliverQuery {
                queryable_id: qid_b,
                query_id: 11,
                key_expr: "harmony/content/b/bcde".into(),
                payload: b"get-b".to_vec(),
            }
        );
    }

    #[test]
    fn undeclare_stops_delivery() {
        let mut router = QueryableRouter::new();

        let (qid, _) = router.declare("harmony/content/a/**").unwrap();

        // Undeclare returns SendQueryableUndeclare
        let actions = router.undeclare(qid).unwrap();
        assert_eq!(actions.len(), 1);
        assert_eq!(
            actions[0],
            QueryableAction::SendQueryableUndeclare {
                key_expr: "harmony/content/a/**".into(),
            }
        );

        // Subsequent query produces no actions
        let actions = router.handle_event(QueryableEvent::QueryReceived {
            query_id: 99,
            key_expr: "harmony/content/a/abcd".into(),
            payload: b"fetch".to_vec(),
        })
        .unwrap();
        assert!(actions.is_empty());

        // Undeclaring again fails with UnknownQueryableId
        let result = router.undeclare(qid);
        assert!(matches!(result, Err(ZenohError::UnknownQueryableId(_))));
    }

    #[test]
    fn declare_rejects_invalid_key_expr() {
        let mut router = QueryableRouter::new();
        let result = router.declare("");
        assert!(matches!(result, Err(ZenohError::InvalidKeyExpr(_))));
    }

    #[test]
    fn query_no_match_returns_empty() {
        let mut router = QueryableRouter::new();
        router.declare("harmony/content/a/**").unwrap();

        // Query on a completely disjoint key expression.
        let actions = router
            .handle_event(QueryableEvent::QueryReceived {
                query_id: 1,
                key_expr: "harmony/other/xyz".to_string(),
                payload: vec![],
            })
            .unwrap();
        assert!(actions.is_empty());
    }
}
