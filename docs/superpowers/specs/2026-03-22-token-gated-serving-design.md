# Token-Gated Encrypted Content Serving from Replication Delegates

**Date:** 2026-03-22
**Status:** Draft
**Scope:** `harmony-tunnel` (new replication op), `harmony-node` (validation + public key cache)
**Bead:** harmony-5zt

## Overview

When Bob stores Alice's encrypted books via the replication policy, authorized third parties can retrieve them by presenting a self-certifying PQ UCAN bearer token signed by Alice. Bob validates the token cryptographically without needing prior coordination with Alice — the token carries its own proof of authorization.

This builds on the existing `PqUcanToken` infrastructure (`CapabilityType::Content`) and the replication protocol (`FrameTag::Replication`).

## Token Format

Uses the existing `PqUcanToken` from `harmony-identity/src/ucan.rs`:

| Field | Value for content access |
|-------|------------------------|
| `issuer` | Alice's `identity_hash` (content owner) |
| `audience` | Carol's `identity_hash` (authorized reader) |
| `capability` | `CapabilityType::Content` (value 5) |
| `resource` | 32-byte CID of the requested book |
| `not_before` | Earliest valid timestamp |
| `expires_at` | Token expiry (Unix epoch seconds) |
| `nonce` | Random 16 bytes for uniqueness |
| `proof` | `None` for root tokens (delegation chains deferred) |
| `signature` | ML-DSA-65 signature over the signable payload |

## Protocol

### New Replication Op

Add `PullWithToken = 0x06` to `ReplicationOp`:

```
[1 byte op=0x06][32 bytes CID][token_bytes]
```

Carol sends this to Bob. `token_bytes` is the serialized `PqUcanToken`.

### Validation Flow (Bob's side)

When Bob receives `PullWithToken(cid, token_bytes)`:

1. **Deserialize** the token via `PqUcanToken::from_bytes()`
2. **Check capability** — `token.capability == Content`
3. **Check resource** — `token.resource == requested_cid`
4. **Check expiry** — `token.expires_at > now_unix_seconds`
5. **Check not-before** — `token.not_before <= now_unix_seconds`
6. **Check owner** — look up the replicated CID in `ReplicaStore`, verify `entry.owner_hash == token.issuer` (the token was issued by the content owner, not someone else)
7. **Verify signature** — look up issuer's ML-DSA public key from the public key cache, call `verify_pq_token()` (existing function in harmony-identity)
8. **Serve or reject** — if all pass, respond with `PullResponse(cid, data)`. If any fail, respond with a new `Error` op or simply drop the request.

### Public Key Cache

Bob needs the issuer's ML-DSA public key to verify the token signature. A `HashMap<IdentityHash, Vec<u8>>` in `NodeRuntime`:

- **Populated from:** `AnnounceRecord.public_key` (already received via discovery), `HandshakeComplete.peer_dsa_pubkey` (from tunnel handshake)
- **Lookup on validation:** `pubkey_cache.get(&token.issuer)`
- **Cache miss:** Reject the request. Carol can retry after Bob discovers Alice via announce.

This cache is ephemeral (rebuilt on restart from announces). No persistence needed — if Bob doesn't know Alice's key, he can't serve her content anyway.

## Changes Summary

| Component | Change |
|-----------|--------|
| `harmony-tunnel/src/replication.rs` | Add `PullWithToken = 0x06` op |
| `harmony-node/src/runtime.rs` | Public key cache, token validation logic, handle PullWithToken in ReplicaReceived |
| `harmony-node/src/event_loop.rs` | Populate pubkey cache from HandshakeComplete and discovery announces |

## What we're NOT building

- **Token issuance CLI** — Alice creates tokens programmatically (future bead for UI)
- **Token distribution** — out-of-band for v1 (Zenoh queryable for refresh is future work)
- **Delegation chains** — UCAN `proof` field supports this, but v1 validates root tokens only
- **Audience verification** — v1 doesn't check `token.audience == carol_identity_hash` because Bob doesn't know Carol's identity from the tunnel connection. The token is a bearer token — possession is proof of authorization.
