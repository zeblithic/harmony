# Self-Sovereign Cloud Applications: North-Star Architecture

## Purpose

This document defines Harmony's north-star architecture for self-sovereign cloud
applications. It maps out how Harmony's existing primitives (mesh networking,
content-addressed storage, cryptographic identity) compose into a platform where
users own their data, choose their software, and share on their terms — without
trusting any server operator.

Email serves as the running illustration throughout, but the architecture is
general. Any application that stores private data, processes incoming messages,
and shares selectively with others follows the same pattern.

## Framing: What "Self-Sovereign Cloud" Means

A self-sovereign cloud application has three properties:

1. **Data sovereignty.** The user's data is encrypted at rest with keys only
   they control. No infrastructure operator can read it.
2. **Compute sovereignty.** Code that processes user data runs in a verifiable
   sandbox. The user knows exactly what code touched their data and what
   permissions it had.
3. **Relationship sovereignty.** Sharing is peer-to-peer via cryptographic
   delegation, not mediated by a platform that can revoke access or surveil
   the relationship.

Traditional cloud gives you none of these. End-to-end encrypted messaging gives
you (1) but not (2) or (3). Harmony aims to provide all three.

## Architecture: Five Layers

Each layer enables the one above it. The bottom two exist today; the top three
are what this design specifies.

```
┌─────────────────────────────────────────────────┐
│  Layer 5: Applications                          │
│  (email, chat, documents, …)                    │
├─────────────────────────────────────────────────┤
│  Layer 4: Self-Sovereign App Runtime            │
│  (WASM sandbox + UCANs + PRE)                   │
├─────────────────────────────────────────────────┤
│  Layer 3: Encrypted Personal Namespaces         │
│  (HAMT state trees, delta-sync)                 │
├─────────────────────────────────────────────────┤
│  Layer 2: Content-Addressed Storage             │
│  (CAS, HAMT, W-TinyLFU cache)                  │
├─────────────────────────────────────────────────┤
│  Layer 1: Mesh Network                          │
│  (Reticulum transport, Zenoh pub/sub routing)   │
└─────────────────────────────────────────────────┘
```

---

## Layer 1: Mesh Network (Exists)

**Crates:** `harmony-reticulum`, `harmony-zenoh`

Reticulum provides the transport: cryptographic addressing, opportunistic
routing, link establishment over any physical medium (LoRa, TCP, UDP, serial).
Every node is addressable by `SHA256(X25519_pub || Ed25519_pub)[:16]`.

Zenoh provides the pub/sub routing layer above transport. Key expressions like
`harmony/personal/{address}/email/**` define subscription scopes. The Zenoh
subscription table routes messages to the correct handler without central
coordination.

**What this gives upper layers:** Any node can reach any other node by
cryptographic address. Messages are routed by topic, not by server hostname.

## Layer 2: Content-Addressed Storage (Exists)

**Crates:** `harmony-content` (CAS, blob store, storage tiers, W-TinyLFU cache)

Content is stored by its cryptographic hash (CID). The W-TinyLFU admission
policy keeps frequently-accessed content in cache while evicting one-hit
wonders. Storage tiers separate hot (cache), warm (pinned), and cold (archival)
content.

**What this gives upper layers:** Immutable, deduplicated, verifiable storage.
If you have a CID, you can fetch the content from any node that has it and
verify it hasn't been tampered with.

---

## Layer 3: Encrypted Personal Namespaces (To Build)

Every user gets a namespace rooted at `harmony/personal/{address}/*`. This is
their sovereign territory — encrypted, delta-syncable, and structured as a
Hash Array Mapped Trie (HAMT).

### Structure

```
harmony/personal/{address}/
├── .meta              # Schema version, last-modified, public key hints
├── email/
│   ├── inbox/         # HAMT leaves: encrypted message bundles
│   ├── sent/
│   └── drafts/
├── documents/
│   ├── personal/
│   └── shared/
└── apps/
    └── preferences/
```

Each HAMT leaf is an encrypted blob stored in the CAS. The HAMT root is a
single CID that represents the entire namespace state at a point in time.

### Public vs. Private Content

| | Public | Private |
|---|---|---|
| **Examples** | App binaries, shared schemas | User messages, documents, keys |
| **Encryption** | None (plaintext CID) | Encrypted with user's symmetric key |
| **Deduplication** | Global (same CID everywhere) | Per-user (same plaintext → different ciphertext) |
| **Storage cost** | Free (globally shared) | Counts against personal quota |
| **Access** | Anyone with CID | Only key holder (or PRE delegate) |

### Delta-Sync via HAMT

When a user modifies one file, only the changed HAMT leaf produces a new
encrypted blob. The new root references unchanged subtrees by their existing
CIDs. Syncing between devices means exchanging only the changed path from root
to leaf — logarithmic in namespace size, not linear.

```
Root v1: [A, B, C]       Root v2: [A, B', C]
              │                        │
              B: [d, e, f]             B': [d, e', f]
                    │                         │
                    e: blob_1                 e': blob_2  ← only new blob
```

### Zenoh Integration

The node publishes namespace mutations on
`harmony/personal/{address}/sync/**`. Other devices belonging to the same
identity subscribe to this key expression and apply deltas. Conflict resolution
follows last-writer-wins per HAMT path, with vector clocks for causal ordering.

---

## Layer 4: Self-Sovereign App Runtime (To Build)

Three trust mechanisms compose to form the runtime. Each solves a different part
of the trust problem:

| Mechanism | Trust Problem | Guarantee |
|---|---|---|
| **Verified Compute** (WASM) | "What code touched my data?" | Content-addressed modules, deterministic execution |
| **Authorized Access** (UCANs) | "Who allowed this?" | Self-certified capability chains, offline-verifiable |
| **Blind Data Routing** (PRE) | "Can I share without exposing?" | Ciphertext transformation without decryption |

### 4a: Verified Compute — Content-Addressed WASM Modules

**Crate:** `harmony-compute` (exists: fuel budgeting), `harmony-workflow`
(exists: durable execution)

Every application is a WASM module identified by its CID:
`CID = BLAKE3(module_bytes)`. Because the CID is a hash of the code, you can
verify exactly what logic will process your data before granting it access.

Durable execution means workflows survive restarts. The workflow state (fuel
consumed, pending timers, execution history) is checkpointed to the CAS. If a
node reboots, workflows resume from their last checkpoint — no lost messages,
no orphaned state.

**WorkflowId** = `BLAKE3(module_hash || input)` — a cryptographic proof of
"this specific code processed this specific input."

The WASI sandbox restricts what a module can do:
- No raw filesystem access (only granted namespace paths)
- No arbitrary network access (only declared Zenoh key expressions)
- Fuel-budgeted execution (can't monopolize compute)
- Deterministic replay (same input → same output → auditable)

### 4b: Authorized Access — UCANs

UCANs (User Controlled Authorization Networks) are self-contained capability
tokens. They replace OAuth/ACLs with cryptographically-signed permission chains
that work offline and require no central authority.

**Structure of a UCAN:**
```
{
  issuer:     <user's public key>,
  audience:   <module CID or delegatee key>,
  expiration: <timestamp>,
  capabilities: [
    { resource: "harmony/personal/{addr}/email/*", action: "read,write" },
    { resource: "harmony/network/zenoh/email/inbound/**", action: "subscribe" }
  ],
  proof:      [<parent UCAN if delegated>],
  signature:  <Ed25519 signature over above fields>
}
```

**Key properties:**
- **Self-certified:** Any node can verify the chain without contacting the
  issuer. Just check signatures up the chain to a trusted root key.
- **Attenuated delegation:** Alice grants Bob `read,write` on `/email/*`. Bob
  can delegate to Carol, but only `read` on `/email/inbox/*` — permissions can
  only narrow, never widen.
- **Module-CID-scoped:** Permissions can be granted to a specific WASM module
  by CID. "App `abc123` may read my email" means exactly that binary, not any
  future version.
- **Time-bounded:** Every UCAN has an expiration. Revocation is distributed via
  a revocation list published to the user's namespace.

**Signing scheme:** The UCAN header includes the signing algorithm identifier.
Verifiers use this to select the correct verification path (see crypto-agility
below).

### 4c: Blind Data Routing — Proxy Re-Encryption

When Alice shares encrypted data with Bob, no intermediary should see the
plaintext. Proxy Re-Encryption (PRE) enables this: a re-encryption key
transforms ciphertext encrypted for Alice into ciphertext decryptable by Bob,
without any intermediate node performing decryption.

**Starting point:** AFGH-style unidirectional PRE over Curve25519, chosen for
Reticulum compatibility and proven security properties. Ed25519 for signatures,
X25519 for key agreement.

**Post-quantum transition (required before v1.0):**

Ed25519/Curve25519 proves the concept, but elliptic-curve cryptography has a
known expiration date. Data encrypted today could be harvested and decrypted by
quantum computers in 10-15 years ("harvest now, decrypt later"). For a platform
where people build their digital lives, this is unacceptable.

The crypto layer must be **trait-abstracted** from day one:

| Trait | Curve25519 (v0.x) | Post-Quantum (v1.0) |
|---|---|---|
| `SigningScheme` | Ed25519 | Lattice-based (e.g., CRYSTALS-Dilithium) |
| `KemScheme` | X25519 ECDH | Lattice-based KEM (e.g., CRYSTALS-Kyber) |
| `PreScheme` | AFGH over Curve25519 | Lattice-based PRE (active research) |
| `AddressDerivation` | SHA256(X25519 ∥ Ed25519)[:16] | SHA256(pq_enc ∥ pq_sign)[:16] |

**Design implications for all layers:**
- All crypto operations go through trait interfaces, never call primitives
  directly. `harmony-crypto` and `harmony-identity` must expose trait-based
  abstractions even in v0.x.
- Identity keys are **scheme-tagged**: the first bytes identify which scheme
  generated them, so parsers know how to handle them.
- UCAN tokens include the signing scheme in their header, so verifiers select
  the correct algorithm.
- The network supports **hybrid mode** during transition: nodes can hold both
  Curve25519 and post-quantum keys simultaneously, advertising both in
  announces.
- The HAMT namespace structure is crypto-agnostic — it stores opaque encrypted
  blobs regardless of the encryption scheme underneath.
- The PRE re-encryption key format is scheme-dependent, but the delegation
  protocol (request → issue → transform) is scheme-independent.

### 4d: How They Compose — The App Sandbox

The three mechanisms combine into a single interaction pattern:

```
1. User fetches app by CID           → verified compute (know what code runs)
2. User issues scoped UCAN to app    → authorized access (know what it can do)
3. App runs as durable workflow       → sandboxed execution (can't escape)
4. App reads/writes encrypted HAMT   → data sovereignty (user owns state)
5. Sharing uses PRE delegation        → relationship sovereignty (no mediator)
```

**Concretely:** Alice installs an email app (CID `abc123`). She issues a UCAN
granting it `read,write` on `harmony/personal/{alice}/email/*` and `subscribe`
on `harmony/network/zenoh/email/inbound/**`. The app runs as a durable workflow.
Incoming messages arrive via Zenoh subscription, get processed by the verified
WASM module, and are stored as encrypted blobs in Alice's HAMT namespace. When
Alice forwards an email to Bob, the app requests a PRE re-encryption key (via
UCAN delegation) that transforms the ciphertext from Alice's key to Bob's key.
No intermediate node sees the plaintext.

---

## Layer 5: Applications

The architecture above is general. Any application that stores private data,
processes incoming messages, and shares selectively follows the same pattern.
Email illustrates how the layers compose for a concrete use case.

### The General Self-Sovereign App Pattern

1. **App = content-addressed WASM module.** Published to the network by CID.
   Public, deduplicated, freely fetchable.

2. **User discovers app** (out of band: app directory, link, recommendation).
   Fetches the module by CID. Inspects declared capabilities.

3. **User issues a scoped UCAN** granting the app specific namespace
   permissions. Permissions are time-bounded and attenuated.

4. **App runs as a durable workflow** in the user's compute sandbox. WASI host
   enforces namespace boundaries, network restrictions, and fuel budgets.

5. **Incoming data arrives via Zenoh subscriptions.** The node runtime routes
   messages to the correct workflow by key expression match.

6. **Sharing uses PRE.** When the app needs to share data with another user, it
   requests a re-encryption key via UCAN delegation.

### Email as Concrete Illustration

| Step | General Pattern | Email Specific |
|---|---|---|
| App module | WASM binary by CID | `harmony-email.wasm` — SMTP bridge + local storage |
| Namespace | `harmony/personal/{addr}/<app>/*` | `.../email/inbox/*`, `.../sent/*`, `.../drafts/*` |
| Inbound | Zenoh subscription | `harmony/personal/{addr}/email/inbound/**` |
| Processing | Workflow processes messages | Parse MIME, extract attachments, index, encrypt, store |
| Sharing | PRE re-encryption | Forward email = PRE transform to recipient's key |
| Delegation | UCAN with attenuated perms | "Client X may read inbox but not delete" |

### What This Gives Users

- **No server operator sees your email.** Encrypted in your namespace,
  processed by verified WASM.
- **Switch clients freely.** Any email app that speaks the namespace schema
  works. Your data stays yours.
- **Offline-first.** HAMT delta-sync catches up on reconnection. Durable
  workflows resume from checkpoint.
- **Granular sharing.** PRE lets you share specific folders with specific
  people without giving them your master key.
- **Auditable.** Every app is a CID, every permission is a UCAN chain, every
  state mutation is a workflow step.

---

## Implementation Roadmap

The layers build bottom-up. Layers 1-2 exist. The roadmap for 3-5:

### Phase 1: Encrypted Personal Namespaces (Layer 3)
- HAMT implementation over CAS
- Namespace schema and path conventions
- Delta-sync protocol over Zenoh
- Symmetric encryption of leaf blobs

### Phase 2: Crypto-Agile Foundation
- Trait abstractions in `harmony-crypto` for signing, KEM, and PRE
- Scheme-tagged identity keys in `harmony-identity`
- Curve25519 implementations behind traits (preserving Reticulum compat)
- PRE prototype (AFGH over Curve25519)

### Phase 3: App Runtime (Layer 4)
- UCAN token format, signing, and verification
- WASI host integration with UCAN-based permission enforcement
- Durable workflow ↔ namespace integration
- PRE delegation protocol

### Phase 4: Applications (Layer 5)
- Reference email app as proof-of-concept
- App discovery and CID-based distribution
- Client-agnostic namespace schemas

### Phase 5: Post-Quantum Transition
- Lattice-based implementations behind existing traits
- Hybrid key support in identity and announces
- Migration tooling for existing namespaces
- Network-wide transition protocol

---

## Open Questions

1. **HAMT branching factor.** 16-way? 32-way? Affects tree depth vs. node size
   for delta-sync efficiency.
2. **Conflict resolution.** Last-writer-wins is simple but lossy. Do we need
   CRDTs for specific namespace paths (e.g., shared documents)?
3. **Revocation propagation.** UCAN revocation lists need to reach all holders.
   Zenoh pub/sub is natural, but what's the consistency guarantee?
4. **PRE scheme selection for post-quantum.** Lattice-based PRE is an active
   research area. Which construction is mature enough for production?
5. **Storage quotas.** How are personal namespace quotas enforced in a
   decentralized network? Reputation-based? Token-based? Honor system with
   peer pressure?
