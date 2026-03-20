# Social Content Routing Design

**Date:** 2026-03-20
**Status:** Approved
**Scope:** Design doc + integration tests (no new crate)
**Bead:** harmony-6cr

## Overview

Social content routing in Harmony is not a separate system — it's an
emergent property of three existing layers composing:

1. **Social topology** — your tunnel peer connections define your
   social graph
2. **Interest declaration** — Zenoh subscriptions propagate what you
   want to your connected peers
3. **Content availability** — Bloom filters broadcast what your peers
   have

This document formalizes how these layers compose to produce
interest-driven data flow along the social graph, and defines
integration tests that prove the composition works.

## Design Philosophy

### Friends Tell You About It, the Library Gives You the Copy

Your friends are how you *discover* content — they share similar
interests and tell you about things you'll care about. But you don't
need to get the content *from* them. Any node with the CID can serve
it (WORM — Write Once Read Many — means every copy is identical).

The social graph is a **discovery mechanism**, not a delivery
mechanism. A Zenoh subscription to a friend's vine goes to your
tunnel peer, who publishes matching content. But a Zenoh `get()` for
a specific CID can be served by any node in the network.

### Emergent, Not Engineered

Social routing doesn't require a routing table, interest taxonomy,
or bandwidth budget system. It emerges naturally from:

- **Who you peer with** → `ContactStore` + `PeerManager`
- **What you subscribe to** → Zenoh `PubSubRouter`
- **What your peers have** → `StorageTier` Bloom filters

No new crate is needed. The existing infrastructure already
implements social routing. This bead formalizes the understanding
and proves it works.

## Layer 1: Social Topology

Your `ContactStore` defines intentional relationships — people you
chose to connect with. `PeerManager` maintains persistent tunnel
connections to contacts with `peering.enabled = true`. These tunnel
peers are your first-hop social graph.

```
Alice ←──tunnel──→ Bob ←──tunnel──→ Carol
         (trust)           (trust)
```

Adjacent peers (Reticulum transport neighbors) are transient and
untrusted. Social routing operates exclusively on tunnel peers —
identities you've chosen to maintain persistent connections with.

The `PeeringPriority` (Low, Normal, High) controls reconnection
aggressiveness but does not affect content routing priority. All
tunnel peers are equal for content routing — the act of peering
is the trust signal.

## Layer 2: Interest Declaration

When a user subscribes to a Zenoh key expression, the interest
propagates to all connected peers:

```
1. Alice subscribes to harmony/vines/{bob_hex}/announce/**
2. Alice's PubSubRouter emits SendSubscriberDeclare
3. Bob's session receives the declaration
4. Bob registers Alice's interest in his remote_interest table
5. When Bob publishes a matching vine, he checks remote_interest
6. Alice is interested → Bob emits SendMessage to Alice
7. Alice receives the vine content
```

This is standard Zenoh write-side filtering. The social routing
aspect is that Alice's subscription goes to Bob (her tunnel peer)
rather than to a global broker. The topology IS the routing.

### Multi-Hop Interest Relay (Future)

Transitive interest propagation (Alice → Bob → Carol) is a natural
extension but requires a relay coordinator that doesn't exist yet.
Currently `PubSubRouter` manages a single peer-to-peer session and
does not forward remote interests to other sessions. Multi-hop
relay is filed as a future bead.

For now, social routing operates on direct tunnel peer connections
only. If Alice wants Carol's content and isn't directly peered with
Carol, Alice peers with Carol (or finds the content via CID from
any node that has it).

### What You Subscribe To Defines Your Social Feed

The key expressions you subscribe to determine what content flows
to you through the social graph:

- `harmony/vines/{friend_hex}/announce/**` — a specific friend's posts
- `harmony/community/{hub_hex}/**` — a community's feed
- `harmony/vines/**/announce/**` — all vine announcements (broad)
- `harmony/profile/*` — all profile updates

The more specific your subscriptions, the more targeted your social
content flow.

## Layer 3: Content Availability

Peers broadcast availability filters advertising what they store:

```
harmony/filters/content/{node_addr}  — Bloom filter: "I probably have these CIDs"
harmony/filters/flatpack/{node_addr} — Cuckoo filter: "I have these content bundles"
```

Before issuing a Zenoh `get()` for a specific CID, a node can check
which peers likely have it. Tunnel peers' filters are the most
useful because:

1. **They're already connected** — no connection setup cost
2. **They share interests** — social graph correlation means friends
   are likely to have content you want
3. **They're trustworthy** — you chose to peer with them

But the key insight is that you don't *need* the content from a
friend. Any node with the CID serves an identical copy (WORM). The
Bloom filter check is an optimization for locality, not a
requirement for correctness.

## What This Bead Delivers

### Design Document (this file)

Captures the architectural narrative: how social routing works in
Harmony as an emergent property of existing systems.

### Integration Tests

Integration tests in `crates/harmony-node/tests/social_routing.rs`
that prove the composition works by exercising the sans-I/O state
machines together:

**Test 1: Interest propagation through tunnel peers**

Alice subscribes to a key expression. Bob (tunnel peer) receives the
interest declaration. When Bob publishes matching content, Alice
receives it. Proves Zenoh subscriptions flow through tunnel peer
connections.

**Test 2: Content discovery via Bloom filters**

Alice and Bob are tunnel peers. Bob has content CID X in his
StorageTier. Alice checks Bob's Bloom filter and confirms he likely
has it. Proves Bloom filter broadcasts from tunnel peers enable
socially-informed content discovery.

### No New Crate

The existing infrastructure (`harmony-zenoh`, `harmony-content`,
`harmony-peers`, `harmony-contacts`, `harmony-trust`) already
implements social routing. This bead formalizes and verifies.

## Integration Test Dependencies

```toml
[dev-dependencies]
harmony-zenoh = { path = "crates/harmony-zenoh" }
harmony-content = { path = "crates/harmony-content" }
harmony-contacts = { path = "crates/harmony-contacts" }
harmony-peers = { path = "crates/harmony-peers" }
harmony-trust = { path = "crates/harmony-trust" }
harmony-identity = { path = "crates/harmony-identity" }
harmony-crypto = { path = "crates/harmony-crypto" }
```

Tests compose the sans-I/O state machines directly — no live Zenoh
runtime needed. Events are injected, actions are inspected.

The workspace root currently has no `tests/` directory. The
implementation will create `tests/social_routing.rs` and add the
dev-dependencies to the workspace root `Cargo.toml`.

## Future Beads

- **Friends-first query preference** — when multiple nodes can serve
  a CID, prefer tunnel peers. Currently all Zenoh respondents are
  equal; a locality-aware ranking could prioritize social connections.

- **Interest-based precaching** — tunnel peers proactively cache
  content they know you'll want (based on your subscription
  patterns), before you request it. Turns "pull" into "push."

- **Bandwidth budgets** — per-peer limits on how much content is
  proactively shared. Prevents a single high-volume peer from
  saturating your connection.

- **Multi-hop interest relay** — relay coordinator that forwards
  remote interests across tunnel peer sessions, enabling transitive
  interest propagation (Alice → Bob → Carol).

- **Interest taxonomy** — structured interest categories beyond raw
  key expressions. "I'm interested in music" rather than subscribing
  to 50 individual vine authors.
