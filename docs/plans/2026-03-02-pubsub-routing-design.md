# Pub/Sub Routing Design

**Bead:** harmony-0p6.4
**Date:** 2026-03-02
**Status:** Approved

## Summary

Add a sans-I/O pub/sub routing layer to harmony-zenoh that manages publisher
and subscriber declarations, interest-based write-side filtering, and inbound
message dispatch. The router composes with the existing Session (peer lifecycle)
and SubscriptionTable (key expression matching) without owning either.

## Decision Record

- **Thin coordinator** over monolithic stack: PubSubRouter takes `&mut Session`
  references rather than owning the session. Consistent with the session design
  pattern — single responsibility, testable independently.

- **Declare-based interest propagation** over flood-then-filter: When a local
  subscriber is added, the router emits `SendSubscriberDeclare` to the peer. The
  peer uses this to know which messages to send. Avoids wasting bandwidth on
  unsubscribed topics.

- **ExprId references** over inline key expressions: Messages carry the ExprId
  (u64) declared via the session resource protocol. Compact on the wire, requires
  resources to be declared before publishing.

- **Wire-only routing** over local+remote delivery: PubSubRouter handles
  inter-peer routing. Local dispatch within a process is the caller's
  responsibility. The router emits `Deliver` actions that report subscription
  matches — the caller decides how to handle them.

- **Side map for key expression recovery** over extending SubscriptionTable:
  PubSubRouter keeps `HashMap<SubscriptionId, String>` and
  `HashMap<String, SubscriptionId>` to recover key expressions on unsubscribe.
  Keeps SubscriptionTable unchanged.

## Core Types

```rust
pub type PublisherId = u64;

pub struct PubSubRouter {
    subscriptions: SubscriptionTable,
    sub_key_exprs: HashMap<SubscriptionId, String>,
    remote_interest: SubscriptionTable,
    remote_interest_ids: HashMap<String, SubscriptionId>,
    publishers: HashMap<PublisherId, ExprId>,
    next_publisher_id: PublisherId,
}

pub enum PubSubEvent {
    SubscriberDeclared { key_expr: String },
    SubscriberUndeclared { key_expr: String },
    MessageReceived { expr_id: ExprId, payload: Vec<u8> },
}

pub enum PubSubAction {
    SendSubscriberDeclare { key_expr: String },
    SendSubscriberUndeclare { key_expr: String },
    SendMessage { expr_id: ExprId, payload: Vec<u8> },
    Deliver { subscription_id: SubscriptionId, key_expr: String, payload: Vec<u8> },
}
```

Entry point: `router.handle_event(event, &session) -> Result<Vec<PubSubAction>, ZenohError>`

## Publisher Lifecycle

1. `declare_publisher(key_expr, &mut session)` → calls `session.declare_resource()`,
   stores `PublisherId → ExprId` mapping, returns `PublisherId`.
2. `publish(pub_id, payload, &session)` → resolves ExprId via publisher map,
   checks `remote_interest.matches()` for write-side filtering. If peer has
   interest, emits `SendMessage { expr_id, payload }`. If no interest, returns
   empty actions (message dropped silently).
3. `undeclare_publisher(pub_id, &mut session)` → calls `session.undeclare_resource()`,
   removes from publisher map.

New error: `UnknownPublisherId(u64)` for operations on undeclared publishers.

## Subscriber Lifecycle & Interest Propagation

**Local subscription:**

1. `subscribe(key_expr)` → validates key expression, registers in local
   `SubscriptionTable`, stores `sub_id → key_expr` in side map, emits
   `SendSubscriberDeclare { key_expr }`.
2. `unsubscribe(sub_id)` → recovers key expression from side map, removes from
   `SubscriptionTable`, emits `SendSubscriberUndeclare { key_expr }`.

**Remote interest (from peer):**

1. `SubscriberDeclared { key_expr }` → registers in `remote_interest` table,
   stores `key_expr → sub_id` in reverse map. No action emitted.
2. `SubscriberUndeclared { key_expr }` → recovers sub_id from reverse map,
   removes from `remote_interest` table.

New error: `UnknownSubscriptionId(u64)` for invalid unsubscribe.

## Inbound Message Dispatch

On `MessageReceived { expr_id, payload }`:

1. Resolve `expr_id` → key expression via `session.resolve_remote(expr_id)`.
2. Match key expression against local `subscriptions.matches()`.
3. Emit `Deliver { subscription_id, key_expr, payload }` per matching subscriber.
4. If no subscribers match, message is silently dropped (normal when interest
   changes mid-flight).

`handle_event` takes `&Session` (read-only) for message dispatch — it resolves
ExprId but doesn't modify session state.

## Error Variants

New additions to `ZenohError`:

- `UnknownPublisherId(u64)` — operation on undeclared publisher
- `UnknownSubscriptionId(u64)` — unsubscribe with invalid ID

Existing errors that propagate: `UnknownExprId`, `SessionNotActive`.

## Module Structure

```
crates/harmony-zenoh/src/
  lib.rs            — add pub mod pubsub, re-export public types
  pubsub.rs         — PubSubRouter, PubSubEvent, PubSubAction, PublisherId
  session.rs        — unchanged
  subscription.rs   — unchanged
  envelope.rs       — unchanged
  keyspace.rs       — unchanged
  error.rs          — add 2 new variants
```

No new external dependencies.

## Tests

1. Declare publisher — session resource declared, PublisherId returned
2. Undeclare publisher — session resource undeclared
3. Publish with remote interest — emits SendMessage
4. Publish without remote interest — no action (write-side filtering)
5. Publish on unknown publisher — error
6. Subscribe — emits SendSubscriberDeclare, SubscriptionId returned
7. Unsubscribe — emits SendSubscriberUndeclare
8. Unsubscribe unknown ID — error
9. Inbound message with matching subscriber — emits Deliver
10. Inbound message with no matching subscriber — no action
11. Inbound message with multiple matching subscribers — multiple Deliver actions
12. Remote interest declare/undeclare lifecycle
