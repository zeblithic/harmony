# Harmony Client Feature Roadmap

Captures four planned client feature areas beyond the current MVP (messaging,
voice, presence, media trust, network visualization). These features share
infrastructure (cryptographic identity, network transport, UI patterns) but
are otherwise independent. The trust network is foundational -- the others
build on it.

---

## 1. Trust Network (foundation)

**Status:** Not started. Current client has a simple 3-level media trust
system (`untrusted`/`preview`/`trusted`) for gating external media. This
feature replaces that with a rich peer-to-peer trust model.

### Core concept

Each user expresses trust in a peer as a single byte (8 bits) encoding four
dimensions, each with four levels (2 bits, `00` low to `11` high). Bits are
ordered most-significant to least-significant:

| Bits | Dimension | Question | Nature |
|------|-----------|----------|--------|
| 0-1 | **Identity** | Do you believe this person is who they say they are? | Verifiable (PGP key-signing party style) |
| 2-3 | **Compliance** | Do you believe this person follows the rules / do you have evidence they wouldn't? | Semi-verifiable (may have cryptographic evidence of wrongdoing) |
| 4-5 | **Association** | Would you vouch for / associate with this person in this group? | Subjective ("vibes") |
| 6-7 | **Endorsement** | Would you stake your social capital on this person? Would you take a reputation hit if they misbehaved? | Social capital exchange (enables delegation/voting) |

### Key properties

- First two dimensions (identity, compliance) can be backed by cryptographic
  evidence -- key-signing ceremonies, signed attestations of rule violations.
- Last two dimensions (association, endorsement) are purely social signals.
- Trust is directional: Alice's trust in Bob != Bob's trust in Alice.
- Trust is contextual: trust scores may differ per community/group.
- **Second-degree derivation:** By traversing the trust graph, users can
  compute transitive trust for peers they haven't directly scored. A peer
  trusted by many of your trusted peers inherits a derived score.
- The existing 3-level media trust system would be subsumed -- media gating
  becomes a function of the identity dimension score.

### Client UI needs

- View/set trust scores for any peer (4 sliders or 4 dropdown pairs).
- Visualize trust graph (reuse network viz infrastructure).
- View derived/transitive trust for peers you haven't directly scored.
- Trust-weighted peer lists and content feeds.

---

## 2. Voting & Polls

**Status:** Not started.

### Core concept

General-purpose voting/polling for communities and groups. Polls can be
simple (yes/no/abstain) or multi-option. Trust scores from the trust network
can optionally weight votes (endorsement dimension enables delegation).

### Key properties

- Polls are cryptographically signed by creator, votes signed by voters.
- Optional trust-weighted tallying: votes from highly-endorsed peers carry
  more weight (configurable per poll).
- Delegation: if Alice endorses Bob (endorsement bits = `11`), Bob can
  optionally vote on Alice's behalf for polls she hasn't voted on (explicit
  opt-in per poll or per community).
- Results are verifiable: anyone can recount from the signed vote records.
- Anonymous voting mode: ZK proofs that a vote came from an eligible voter
  without revealing which one (future work, not MVP).

### Client UI needs

- Create poll (question, options, duration, trust-weighting toggle).
- Vote interface.
- Results display (raw count + trust-weighted count).
- Delegation management.

---

## 3. WTF Search / Knowledge Base

**Status:** Not started.

### Core concept

Inspired by Google's internal "wtf" search -- a distributed,
community-maintained wiki/encyclopedia/lookup. Any peer can publish entries
under a hierarchical key space. Entries are content-addressed and versioned.

### Key properties

- Key space: `/wtf/<topic>` for global, `/wtf/<community>/<topic>` for
  community-scoped.
- Entries are Harmony content blobs (Merkle DAG, content-addressed).
- Version history via blob chain (each version references its predecessor).
- Trust-weighted search ranking: entries authored by highly-trusted peers
  surface first.
- Community moderation via trust network -- low-trust authors' entries are
  deprioritized, not censored.

### Client UI needs

- Search bar with typeahead.
- Entry viewer (markdown rendering).
- Entry editor (create/edit with version history).
- Trust-weighted result ranking indicators.

---

## 4. Payments & Financial Transactions

**Status:** Not started.

### Core concept

The network's cryptographic identity and durable execution (via Temporal
workflows) provide infrastructure for financial transactions. Currency is
information that people want to protect from replication/counterfeiting --
no different from any other private data on the network.

### Key properties

- **Custom currencies:** Anyone can issue a currency under `/$/$NAME/*` or
  `/$/$SHA256/*`. Value is backed entirely by trust in the issuer (trust
  network scores inform counterparty risk).
- **Fiat bridges:** `/$/USD/*` maps 1:1 to dollars, managed by an
  automation service handling ACH transfers in/out of the network.
- **Durable execution:** Temporal workflows guarantee single execution of
  payment pathways -- no double-spending by design.
- **Atomic transfers:** Two-party or multi-party atomic swaps using the
  existing cryptographic primitives (ECDH, signed commitments).
- Regulatory compliance is a system design problem, not an afterthought --
  a well-designed system already satisfies most regulatory requirements.

### Client UI needs

- Wallet view (balances across currencies).
- Send/receive interface.
- Transaction history.
- Currency browser (discover/inspect available currencies and their issuers'
  trust scores).

---

## Dependency Graph

```
Trust Network -----> Voting/Polls (trust-weighted votes, delegation)
              \----> WTF Wiki (trust-weighted ranking, moderation)
              \----> Payments (counterparty trust, currency issuer trust)
```

Trust is the foundation. The other three features can be built in any order
once trust is in place, and each works without the others.

---

## Implementation Order (recommended)

1. **Trust Network** -- foundational, unblocks everything else.
2. **Voting/Polls** -- simplest consumer of trust, validates the model.
3. **WTF Wiki** -- content system, exercises trust-weighted ranking.
4. **Payments** -- most complex, benefits from all prior infrastructure.
