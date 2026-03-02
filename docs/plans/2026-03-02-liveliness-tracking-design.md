# Liveliness Tracking Design

**Goal:** Implement presence detection without polling via liveliness tokens, subscribers, and queries.

**Parent bead:** harmony-0p6.5 (child of harmony-0p6: Zenoh integration)

## Architecture

Standalone `LivelinessRouter` sans-I/O state machine in `harmony-zenoh`, following the same event/action pattern as `Session` and `PubSubRouter`. Two `SubscriptionTable` instances provide efficient key expression matching: one for remote tokens (queried + matched against subscribers), one for local subscribers (matched when tokens appear/disappear).

No internal timers — disconnect detection delegates to Session's existing `PeerStale`/`SessionClosed` mechanism via a `PeerLost` event.

## Types

- `TokenId = u64` — opaque local token identifier
- `LivelinessSubscriberId = u64` — opaque subscriber identifier

## Events (inbound)

| Event | Trigger |
|---|---|
| `TokenDeclared { key_expr }` | Remote peer declared a token |
| `TokenUndeclared { key_expr }` | Remote peer undeclared a token |
| `PeerLost` | Caller detected session death (PeerStale/SessionClosed) |

## Actions (outbound)

| Action | Purpose |
|---|---|
| `SendTokenDeclare { key_expr }` | Wire: tell peer about local token |
| `SendTokenUndeclare { key_expr }` | Wire: tell peer token removed |
| `TokenAppeared { subscriber_id, key_expr }` | Notify local subscriber of new token |
| `TokenDisappeared { subscriber_id, key_expr }` | Notify local subscriber of removed token |

## Public API

- `declare_token(key_expr) -> (TokenId, Vec<LivelinessAction>)` — register local token, emit SendTokenDeclare
- `undeclare_token(token_id) -> Vec<LivelinessAction>` — remove local token, emit SendTokenUndeclare
- `subscribe(key_expr) -> (LivelinessSubscriberId, Vec<LivelinessAction>)` — register subscriber, emit TokenAppeared for existing remote tokens
- `unsubscribe(subscriber_id) -> Result<(), ZenohError>` — remove subscriber
- `query(key_expr) -> Vec<String>` — snapshot of alive remote tokens matching pattern
- `handle_event(event) -> Vec<LivelinessAction>` — process inbound events

## Internal State

```
LivelinessRouter {
    remote_tokens: SubscriptionTable,       // remote token key exprs (for matching + queries)
    remote_token_keys: HashMap<SubscriptionId, String>,  // reverse lookup
    subscribers: SubscriptionTable,         // local subscribers (for event fan-out)
    subscriber_keys: HashMap<LivelinessSubscriberId, SubscriptionId>,
    local_tokens: HashMap<TokenId, String>, // local tokens (for declare/undeclare)
    next_token_id: TokenId,
    next_subscriber_id: LivelinessSubscriberId,
}
```

## Edge Cases

- **Duplicate remote token on same key expr:** Idempotent replace, no spurious events
- **PeerLost:** Bulk-undeclare all remote tokens, emit TokenDisappeared per matching subscriber
- **Subscribe with no existing tokens:** Empty actions
- **Fan-out:** Multiple subscribers matching same token each get their own event

## Error Handling

- Invalid key expression: `ZenohError::InvalidKeyExpr`
- Unknown token ID: new `ZenohError::UnknownTokenId(u64)`
- Unknown subscriber ID: existing `ZenohError::UnknownSubscriptionId`

## Not Building (YAGNI)

- Per-token TTL/expiry timers
- history(false) option — always deliver existing tokens on subscribe
- Payload on tokens
- Remote subscriber tracking / write-side filtering

## Integration

New file `liveliness.rs`, new error variant, new lib.rs exports. No changes to existing Session, PubSubRouter, or SubscriptionTable code.
