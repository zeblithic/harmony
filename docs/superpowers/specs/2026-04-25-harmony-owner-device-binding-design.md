# Owner→Device Identity Binding

**Date:** 2026-04-25
**Status:** Draft
**Scope:** New crate `harmony-owner` (working name); integrations with `harmony-identity`, `harmony-zenoh`, `harmony-discovery`, `harmony-client`.
**Resolves:** ZEB-173 (Track A umbrella ZEB-169).

## Overview

`harmony-identity` today produces atomic per-device identities: each device generates an Ed25519 + X25519 keypair (plus optional ML-DSA + ML-KEM post-quantum variant) and a 128-bit `IdentityHash` derived from the public-key concatenation. There is no protocol-level concept of "this device belongs to that human." Two devices owned by the same person look exactly like two unrelated peers.

The user's vision requires recognizing roughly a dozen devices as belonging to one owner (`zeblith`), so they can:

1. Discover each other and prefer to stay in touch
2. Share unified identity for cross-device features (mail, inference RPC, content sync)
3. Degrade gracefully when one is lost or compromised
4. Have ownership recoverable from a portable artifact in the worst case

This design introduces a two-tier binding primitive that achieves these goals without depending on a central server, blockchain, or any third party (including friends/family in social-recovery setups).

## Goals

1. A device can prove "I belong to owner `M`" to any peer using only cryptographic evidence the peer can verify locally.
2. Adding a new device to an existing owner's family is a normal user operation, not an out-of-band ceremony every time.
3. Compromise of any single component (one device, the master key, the recovery artifact) does not silently grant the attacker full identity control.
4. Total loss of all devices is recoverable from a portable offline artifact; loss of the artifact is acceptable as a graceful failure (fresh identity).
5. The model interoperates across all harmony clients (harmony-client, harmony-arch, harmony-browser, harmony-glitch, harmony-os, harmony-stq8) — the wire format is the contract.

## Non-Goals

1. Social recovery — N-of-K shares held by friends/family. Explicit user preference for self-sovereignty over recoverability-by-others.
2. Cloud / third-party escrow.
3. Multi-owner shared devices (e.g., a family tablet under both partners' identities). Future work; this design assumes one owner per device.
4. Sharded inference, capability advertisement, network bring-up — those are siblings under Tracks B/C/D.
5. Specific UI/UX for pairing or device management — Track B (ZEB-170 and children).

## Why two tiers

Most multi-device identity systems conflate two distinct questions:

1. **Admission:** Is this device authorized to act under the owner's identity at all?
2. **Ratification:** Do the owner's other devices personally trust this device for their interactions?

Apple's iCloud Keychain "circle" handles only admission. Matrix cross-signing has hints of both via TOFU but treats enrollment as trust by default. Keybase sigchain is admission-only, with reputation accruing implicitly through social proofs.

Treating the two as orthogonal axes — a device is either *admitted* or not, and *separately* ratified by sibling devices on a per-pair basis — yields graceful degradation under partial compromise. A stolen master key gets the attacker an admitted device that no real sibling has ratified; the unratified state is what the network refuses to extend full trust to.

## Primitives

### Owner identity `M`

Master keypair, defining the owner. Ed25519 + X25519 (matching `harmony-identity`'s existing scheme), with optional ML-DSA + ML-KEM PQ variant.

`M` is **cold by default**. It exists in:
1. The user's recovery artifact (BIP39 mnemonic, optional encrypted-file export — see ZEB-175)
2. Transiently in RAM on a trusted device during enrollment ceremonies (entered from the artifact, used to sign one cert, wiped)

`M` is **wiped from device #1's persistent storage** after the initial mint, so even the device that originally generated `M` does not retain it. The recovery artifact is the only persistent home of `M`.

### Device identity `D`

Per-device keypair, generated on the device and persisting locally. `D.id` is the 128-bit `IdentityHash` derived from `D`'s public keys.

A device's identity material is encrypted at rest via OS keychain (ZEB-174).

### Enrollment Cert

Signed statement authorizing a device under owner identity `M`:

```rust
struct EnrollmentCert {
    owner_id: IdentityHash,        // M's hash
    device_id: IdentityHash,       // D.id
    device_pubkeys: PubKeyBundle,  // see below
    issued_at: u64,                // wall-clock seconds
    expires_at: Option<u64>,       // None = no expiry; siblings can revoke
    issuer: EnrollmentIssuer,      // see below
    signature: Vec<u8>,
}

struct PubKeyBundle {
    classical: ClassicalKeys,             // Ed25519 + X25519 (always present)
    post_quantum: Option<PqKeys>,         // ML-DSA + ML-KEM; optional in v1, required post-PQ-migration
}

struct ClassicalKeys { ed25519_verify: [u8; 32], x25519_pub: [u8; 32] }
struct PqKeys { ml_dsa_verify: Vec<u8>, ml_kem_pub: Vec<u8> }

enum EnrollmentIssuer {
    Master,
    Quorum {
        signers: Vec<IdentityHash>,    // K=2 sibling device IDs
        signatures: Vec<Vec<u8>>,
    },
}
```

Verification:
1. `EnrollmentIssuer::Master` — verify against `M`'s public key (fetched from owner-identity announce or the cert itself if self-contained).
2. `EnrollmentIssuer::Quorum` — verify each signer is itself enrolled (recursive walk to a Master-signed cert) and currently active (not archived); verify each signature; require `signers.len() >= K_v1` (2).

### Vouching Cert

Per-(signer, target) attestation:

```rust
struct VouchingCert {
    owner_id: IdentityHash,
    signer: IdentityHash,        // sibling A
    target: IdentityHash,        // sibling D
    stance: Stance,
    issued_at: u64,
    signature: Vec<u8>,           // signed by A's device key
}

enum Stance { Vouch, Challenge }
```

The trust state is two CRDTs at different layers:

1. **VouchingCerts — LWW per `(signer, target)` cell.** Each device can publish at most one current `VouchingCert` per target; newer entries from the same signer supersede older ones. Correct under partition healing because no other signer can override a given signer's cell — flip-flopping reflects the human signer changing their mind, which is intended behavior.
2. **Revocation set — strict Remove-Wins / monotonic add-only.** Once a `RevocationCert` for `(owner, target)` is in network history, no subsequent enrollment of the same `device_id` is honored, and no vouch can resurrect it. This protects against the partition-race failure mode documented in production "Last-Writer-Wins" / "Add-Wins" CRDT designs — including Apple iCloud Keychain's "one or the other is chosen" merge — where a compromised offline node could re-enable a revoked key by publishing a newer-timestamped delegation. We adopt the strict Remove-Wins pattern recommended by UCAN's Policy-CRDT literature.

### Liveness Cert

Periodic signed timestamp:

```rust
struct LivenessCert {
    signer: IdentityHash,         // device D
    timestamp: u64,
    signature: Vec<u8>,
}
```

Used for three purposes:
1. **Heartbeat** — published every few minutes (~1–15 min) to keep the device in the active set.
2. **Revocation refutation** — if a master-signed revocation targets `D` while `D` is in fact still operating, `D` signs a fresh `LivenessCert` with `timestamp > revocation.issued_at`. Verifiers see contested state.
3. **Reclamation refutation** — same mechanism at the owner level (see Reclamation below).

### Reclamation Cert

Published once per fresh-identity event after total loss:

```rust
struct ReclamationCert {
    new_owner_id: IdentityHash,   // M2
    claimed_predecessor: IdentityHash,  // M1
    issued_at: u64,
    challenge_window_end: u64,    // issued_at + e.g. 30 days
    note: String,                  // user-visible explanation
    signature: Vec<u8>,            // by M2
}
```

During the challenge window, `M1` (if alive) can publish a `LivenessCert` signed by any device under `M1`'s identity with `timestamp > issued_at`. The presence of such a `LivenessCert` invalidates the reclamation: peers refuse to honor the predecessor relationship.

If the window expires with no refutation, peers MAY honor the predecessor relationship at reduced initial trust. This is a soft signal — there is no cryptographic proof of identity continuity — but it provides a public, time-bounded forum for the dead identity to refute false claims.

## Lifecycle

### Mint (device #1)

1. User generates `M` and `A` (device #1's keys) on a clean device.
2. Sign `EnrollmentCert { owner: M, device: A, issuer: Master, ... }`.
3. Serialize `M` into the recovery artifact (passphrase-encrypted, BIP39-style; see ZEB-175 for format).
4. **Wipe `M` from device #1's persistent storage.** Device #1 retains only `A` and the Enrollment Cert.
5. Device #1 publishes its Enrollment Cert on the owner-identity gossip topic (`harmony/identity/{owner_hash}/state`) and registers on the liveliness topic.

At this point the entire identity lives on device #1, with the master snapshot only in the artifact.

### Enroll device #2

The user has device #1 (existing, trusted) and wants to enroll device #2.

1. On device #2: user initiates pairing. Device #2 generates `B`, displays a pairing payload (its public keys + a transient nonce) — over QR, paste-token, or NFC depending on Track B's UI choice.
2. On device #1: user enters the recovery seed from the artifact. `M` is reconstructed in RAM.
3. Device #1's pairing flow signs `EnrollmentCert { owner: M, device: B, issuer: Master, ... }` over the pairing payload, then **wipes `M` from RAM**.
4. Device #2 receives the Enrollment Cert (over the same out-of-band channel or via Zenoh once it has discovered device #1).
5. Device #2 publishes its Enrollment Cert and auto-publishes a `VouchingCert { signer: B, target: A, stance: Vouch }` for device #1.
6. Device #1, on next user interaction, prompts: "New sibling B joined at T. Accept?" User accepts → device #1 publishes `VouchingCert { signer: A, target: B, stance: Vouch }`.

Device #2 now has `Enrollment Cert + ≥ 1 sibling vouch`, satisfying the v1 threshold for full trust.

### Enroll device #3+

Same flow as device #2 with two paths available:

1. **Master path:** identical to device #2 — recovery artifact required.
2. **Quorum path:** two existing devices each sign the enrollment cert (`EnrollmentIssuer::Quorum`), no recovery artifact needed. The new device auto-vouches for both signers; each signer's user prompts to ratify the new device.

The user chooses path per pairing event.

### Steady state

1. Each active device publishes `LivenessCert` on a regular cadence (default 5 minutes; configurable 1–15 min) via Zenoh liveliness on `harmony/identity/{owner_hash}/liveness/{device_hash}`.
2. Vouches and challenges propagate via gossip topic `harmony/identity/{owner_hash}/state`.
3. A queryable on `harmony/identity/{owner_hash}/state/snapshot` returns the current trust set for cold-start peers.

### Archival

A device is considered **active** if it has published a `LivenessCert` within the active window (default: 90 days, configurable per identity). Devices outside the window are **archived**:

1. Their existing Enrollment Cert remains historically valid for verifying past actions.
2. Their `VouchingCert`s stop counting toward the N-vouch threshold for new sibling admission.
3. They do not count toward `K`-quorum enrollment.

Archive is reversible: if an archived device returns and publishes a fresh `LivenessCert`, it un-archives in the next gossip round.

The 90-day default is loose enough for infrequently-used devices (an old laptop, a backup phone) and tight enough that genuinely abandoned devices stop influencing trust within a quarter.

### Revocation

Two paths:

1. **Self-revocation:** the device's owner signs a `RevocationCert` from the device itself. Idiomatic for "I'm decommissioning this laptop."

```rust
struct RevocationCert {
    owner_id: IdentityHash,
    target: IdentityHash,
    issued_at: u64,
    issuer: RevocationIssuer,
    reason: RevocationReason,
    signature: Vec<u8>,
}

enum RevocationIssuer { SelfDevice, Master, Quorum { signers, signatures } }
enum RevocationReason { Decommissioned, Lost, Compromised, Other(String) }
```

2. **Master / quorum revocation:** for stolen / compromised devices that may still be online and refusing to self-revoke. Same authority rules as enrollment.

**Compromised-master detection:** if `M` issues a flurry of revocations against currently-active devices, those devices counter-sign `LivenessCert`s with timestamp > revocation. Verifiers see a contested state at the whole-identity level and degrade trust pending out-of-band resolution. The compromise is visibly suspicious rather than a silent takeover.

### Reclamation

When all devices are lost AND the recovery artifact is also lost, the user mints a fresh `M2` on a new device:

1. `M2` publishes a `ReclamationCert` claiming continuity from `M1`, with a 30-day challenge window.
2. During the window, peers honor `M2` at *reduced trust* (it is operationally a new identity that claims to be the same person).
3. If `M1` is silently still alive (e.g., one device offline that the user thought was destroyed), it publishes a `LivenessCert` after `M2`'s issuance time. This invalidates the reclamation.
4. If multiple competing claimants publish reclamations and `M1` is silent, peers see contested state. Resolution is social/off-system (out-of-band communication between humans, not the protocol's job).

After a successful reclamation (window expires with no refutation), peers MAY transfer prior trust associations from `M1` to `M2`. This is policy-level, per-peer; no protocol enforcement.

## Trust evaluation

When peer P encounters device `D` claiming to act under owner `M`:

```text
1. Fetch trust state for M (gossip cache; fall back to queryable).
2. Verify Enrollment Cert chain to a Master-signed root.
3. Build active sibling set: enrolled devices with LivenessCert within active window.
4. Count active vouches FOR D from active siblings (excluding D itself).
5. Count active challenges AGAINST D from active siblings.
6. Check for whole-identity contested state (master-vs-sibling Liveness conflicts).
7. Check trust-state freshness: reject the snapshot if its newest signed entry is older than the freshness window (default: 30 days). Trust state must be re-published periodically; this prevents network-suppression attacks where an adversary isolates P from the gossip topic and feeds P a stale view that omits a recent revocation.
8. Decide:
    - identity contested                   → refuse, surface contested UI
    - trust state stale (>30d)             → refuse, surface stale-state UI; encourage P to re-fetch via queryable
    - challenges_against_D > 0             → refuse high-stakes; allow read-only
    - active_siblings == 1 AND target == sibling
                                           → full trust (single-device case)
    - vouches_for_D >= N_v1 (1)            → full trust
    - else                                 → provisional (low-stakes only)
```

The threshold `N` and the high-stakes/low-stakes split are policy knobs in v1: configurable by application, with documented defaults. A future high-security mode raises `N` to a function of the active-sibling count (e.g., majority).

### Single-device case

When only one device is active (the most common state for users new to harmony, or for users who never expand beyond their primary device), the active-sibling set has size 1. The trust algorithm short-circuits: a single-device family has no siblings to vouch and the entire identity is implicit in that device. This avoids the bootstrapping awkwardness of "no vouches, never trusted."

## Wire formats

All certs serialize as **RFC 8949 §4.2 canonical CBOR** and sign over the canonical bytes. Map entries are sorted by length-then-bytewise-lex on the canonically-encoded key bytes; integers use shortest form; containers are definite-length. The implementation in `harmony-owner::cbor::to_canonical` enforces §4.2 by serializing through `ciborium::value::Value`, recursively sorting map entries, and re-encoding via `ciborium::ser::into_writer`. This guarantees byte-for-byte interop with any RFC 8949 §4.2-compliant CBOR library — second implementations of this protocol are not coupled to ciborium. CBOR rather than JCS-JSON because:

1. Smaller wire size for binary-heavy payloads (signatures, public keys).
2. Type-distinguishable encoding avoids ambiguity (no "is this string or bytes" guessing).
3. Existing `harmony-mailbox` and related crates already use CBOR.

A `version: u8` field is part of every cert payload (initial value `0x01`) and is included in the canonical CBOR object covered by the signature. Per the mailbox wire-format policy precedent: bumps are breaking; readers reject unknown versions; writers pin to current.

### Domain-separated signatures

Every signature is computed over `tag || canonical_cbor(payload)` where `tag` is a fixed byte string identifying the cert type:

- `b"harmony-owner/v1/Enrollment"`
- `b"harmony-owner/v1/Vouching"`
- `b"harmony-owner/v1/Liveness"`
- `b"harmony-owner/v1/Revocation"`
- `b"harmony-owner/v1/Reclamation"`

A signature valid in one context is structurally invalid in any other. This prevents the cert-confusion attack class documented in Matrix's cross-signing implementation, where insufficient domain separation between master cross-signing keys and standard device identifiers let an attacker coerce a verifier into accepting a malformed device cert as a master cross-signing key (Black Hat EU 2022 "Practically-exploitable Cryptographic Vulnerabilities in Matrix"). Cheap to enforce; eliminates an entire failure class.

Concrete CBOR field tags will be specified in the implementation plan, not pinned in this design.

## Crate placement

Working name: **`harmony-owner`**. Final name TBD; alternatives considered: `harmony-family`, `harmony-multidev`, `harmony-cluster`. The crate owns:

1. Cert types (`EnrollmentCert`, `VouchingCert`, `LivenessCert`, `RevocationCert`, `ReclamationCert`)
2. Trust state CRDT (vouching/challenge merge logic)
3. Verification algorithm
4. Gossip topic publish/subscribe glue (depends on `harmony-zenoh`)
5. Active-window archival logic

`harmony-identity` continues to own the per-device key primitives and `IdentityHash` derivation. `harmony-owner` depends on `harmony-identity` for crypto primitives, on `harmony-zenoh` for transport, and on `harmony-crypto` for hashing/canonicalization.

`harmony-groups` is **not** repurposed — its Founder/Officer/Member taxonomy is a different model (multi-owner shared membership) and the access patterns don't overlap cleanly. A future shared-device feature might use `harmony-groups`; this device-binding does not.

## Threat model coverage

| Scenario | Outcome |
|---|---|
| Recovery artifact stolen alone | Attacker enrolls a device under master authority. Reaches *provisional* trust but never *full* — real siblings see the prompt and challenge. Owner rotates master (post-v1 flow). |
| Artifact + revocation attempt against real devices | Real devices counter-sign `LivenessCert`s. Network sees contested state; whole-identity trust degrades. Visibly suspicious. |
| Single sibling device theft | Attacker has one vote in CRDT and cannot solo-enroll new devices (K=2). Owner revokes from siblings; trust on the stolen device collapses. |
| All devices + artifact lost | Mint M2, publish ReclamationCert. Window passes silent → M2 honored at reduced initial trust. Fresh start, gracefully. |
| Reclamation imposter (someone else publishes ReclamationCert claiming M1 lineage) | If M1 alive: signs LivenessCert > claim, invalidates. If M1 truly dead and competing claimants: contested state, social resolution. |
| Stale-device bloat under future high-security N | Opt-in cost of the high-security mode. Acceptable tradeoff per design. |
| Device key exfiltrated via OS malware (keychain bypassed) | Owner signs `RevocationCert` from another device or via master + recovery artifact. Verifiers reject signatures from revoked device after the cert propagates. Recoverable — in contrast to WebAuthn synced-passkey case where local credential deletion does not propagate to relying parties (DR section 2.6) and an attacker with exfiltrated material has indefinite access. |
| Trust-state suppression (network adversary feeds stale view) | Mitigated by 30-day freshness window in trust evaluation: stale snapshots are rejected, P is forced to re-fetch via queryable. Combined with 5-min liveness heartbeat, hard upper bound on view staleness. |
| Hybrid PQ-cutover downgrade attack (post-migration) | Mitigated by strict per-identity cutover (no perpetual hybrid acceptance) — see Future work item 2. Once an identity's `pq_required` flag is set, classical-only signatures from it are rejected. Avoids the documented WebAuthn / Apple CryptoKit / AT Protocol `xDSA` hybrid surface. |
| Pre-migration quantum cryptanalysis breaks classical signatures | Same risk as `harmony-identity` baseline. PQ migration ceremony is the mitigation; v1 ships classical-only consciously. |

## Comparison to prior art

This design occupies a niche not directly covered by any single production system: pure-decentralized, no central directory, no blockchain anchor, no third-party recovery, with both admission and ratification as separate CRDTs. The closest pattern is UCAN/CACAO's Policy-CRDT model, with reclamation as a novel addition. Brief comparisons follow.

### Apple iCloud Keychain (syncing circle)

*Mechanism:* P-384 syncing identity + P-256 password-derived key dual-sign a "circle" CloudKit-replicated to all devices. New devices are sponsor-mediated (existing device approves applicant).

*What we borrow:* dual-tier authority concept (sponsor + identity), and now (via DR) ML-DSA dual-signing precedent for our future PQ ceremony.

*What we don't:* central CloudKit directory, vendor-controlled recovery, "one or the other is chosen" non-deterministic merge (our Remove-Wins for revocations explicitly avoids this), OTR-for-transit (CVE-2017-2448 demonstrated this brittle).

### Matrix cross-signing

*Mechanism:* master cross-signing key signs all device keys; users do single out-of-band master verification (7-emoji SAS); subsequent device additions inherit master trust.

*What we borrow:* master-cross-signing-as-rotation-authority pattern (M signs Enrollment Certs).

*What we don't:* trust-by-default-on-enrollment (our explicit per-device ratification is the gap that catches stolen-master attacks before damage), homeserver-mediated device-list synchronization (we use Zenoh gossip directly), insufficient signature domain separation — Matrix's specific exploit class (Black Hat EU 2022) is structurally prevented by our tagged signing contexts.

### Keybase sigchain (Per-User Keys)

*Mechanism:* append-only sigchain per user; each device-add or revocation is a signed link; revocation rotates a Per-User Key encrypted to remaining trusted devices; sigchain root periodically anchored to Bitcoin.

*What we borrow:* master-key-as-rotation-authority pattern.

*What we don't:* append-only sigchain (we use a CRDT instead — simpler, partition-tolerant, no global ordering required), Bitcoin anchor (no blockchain dependency), vendor-server provisioning (we go fully P2P), historical client validation flaws — NCC-KB2018-001's hash-of-signature-instead-of-payload confusion is structurally avoided by our explicit canonical-bytes-then-tag-then-sign discipline.

### AT Protocol (`did:plc`)

*Mechanism:* directed acyclic graph of operations signed by offline rotation keys; identity URI is hash of genesis operation; central PLC directory aggregates operation logs; Read Replicas independently verify against equivocation.

*What we borrow:* cold-rotation-key-vs-warm-operational-key separation (M cold, devices warm), identity-defined-by-cryptographic-hash genesis pattern.

*What we don't:* central PLC directory (no vendor index — we don't have a single point of equivocation to defend against), DID Document mutation as the state model (we use individually-published certs forming a CRDT), reliance on Read Replica architecture as equivocation defense (we rely on per-cert signatures + freshness windows, which scales without coordination).

### Signal linked devices

*Mechanism:* primary device signs pre-keys for linked devices; server-mediated registration; revocation rotates symmetric ratchet via DeviceRecord deletion.

*What we borrow:* multi-device-under-single-account model, post-compromise security via ratcheting.

*What we don't:* server-mediated registration (no Signal-server analog), Sealed Sender (different threat model — we don't try to hide which identity a peer acts under, and the Statistical Disclosure Attack from DR section 2.5 isn't in our threat model), single-primary-device hierarchy (our model is more horizontal once enrolled — any sibling can vouch, K=2 quorum eliminates a single primary).

### WebAuthn / synced passkeys

*Mechanism:* per-device credentials in OS secure enclave; cross-device sync via vendor cloud; per-relying-party public-key registration.

*What we borrow:* per-device key generation in OS-protected storage (ZEB-174 implements this for our local key material).

*What we don't:* relying-party-centric authorization model (we authenticate identity, not service-specific authorizations), lack of decentralized revocation — the "WebAuthn deletion doesn't propagate to relying parties" failure mode (DR section 2.6) is exactly what our explicit `RevocationCert` solves, substring/regex origin validation flaws (we sign full canonical bytes with domain-separated tags, no permissive matching).

### UCAN / CACAO (capability tokens)

*Mechanism:* signed JWT/IPLD delegation tokens form verifiable chains; revocation propagates via Policy-CRDT (2P-Set, Remove-Wins); peers gossip CRDT state.

*What we borrow heavily:* Policy-CRDT pattern with strict Remove-Wins semantics for revocation, gossip-based propagation. DR section 1.2 + 4.1 explicitly recommends this pattern as the SOTA for decentralized revocation.

*What we don't:* capability-token-based authorization (we authenticate identity; per-action authorization is potentially Track C work), short-token-lifespan + replay-window mitigation strategy (our certs are durable enrollments, not ephemeral capabilities), reliance on FN-DSA/Falcon for token-bloat avoidance — our publication rates accommodate ML-DSA-44's 2.4 KB signatures directly without needing Falcon's smaller-but-side-channel-risky output.

### What's novel here

1. **Two-tier model treating admission and ratification as orthogonal CRDTs.** Closest precedent is UCAN's Policy-CRDT, but UCAN's "ratification" is implicit in delegation chains rather than an explicit per-pair attestation. The asymmetric default (new auto-vouches for old; old must explicitly ratify new) mirrors human intuition that the new arrival should be the one demonstrating trust.
2. **Time-bounded reclamation with public liveness-challenge.** No surveyed system has a "lost everything, mint fresh, claim continuity, time-window for the dead identity to refute" mechanism. Most systems treat lost-master as terminal (Keybase) or rely on vendor-mediated recovery (Apple, Signal). Our model gives a structured, decentralized handling of identity death — without claiming cryptographic proof of continuity (it's a soft signal, time-bounded for refutation).
3. **No vendor or third-party in the trust path at all.** Apple, Matrix, Signal, AT Protocol all have at least one centralized entity (Apple servers, homeservers, Signal's TLS terminator, the PLC directory). UCAN/CACAO comes closest but typically deployed atop IPFS/IPLD with at least pinning services. Ours has zero — Zenoh + the harmony stack carries everything.

### What we don't have that prior art does

1. **Vendor-mediated recovery** — by explicit choice. A user who loses both the recovery artifact AND all devices has no path back to their old identity (only fresh-via-reclamation).
2. **Hardware-anchored master** — Future Work item 4. Apple's Secure Enclave + cloud sync is more usable for end-users; we trade usability for self-sovereignty.
3. **Cross-implementation maturity** — protocols like WebAuthn have many independent implementations and years of interop testing. We will be the only implementation initially. Cross-implementation interop tests (testing item 12) are partial mitigation, but real interop maturity comes from a second-implementation effort.
4. **Established PQ migration shipping in production.** Signal's PQXDH/SPQR and Apple CryptoKit are live; we have a designed-but-deferred path. The risk window is the gap between v1 ship and PQ ceremony.

## Out of scope for v1

1. Configurable `K` and `N` per identity. v1 ships fixed K=2, N=1.
2. Explicit master-rotation ceremony (M → M' migration). Compromise detection works via Liveness counter-signatures; full re-keying flow deferred.
3. Hardware-token (YubiKey/TPM) integration for the master key.
4. Cloud / file-system-shared backup destinations beyond local artifact (covered minimally by ZEB-175).
5. Multi-owner shared devices.
6. Cross-identity reclamation predecessor chains (M2 reclaims M1, M3 reclaims M2 — "did you mean to keep going?"). v1 supports a single hop.
7. PQ migration ceremony. The cert types include `device_pubkeys: PubKeyBundle` which can be extended; the migration ceremony itself is future work.

## Testing

Required test coverage for the v1 implementation:

1. **Mint:** generate M + A, sign A's enrollment, wipe M, recovery artifact reconstitutes M deterministically.
2. **Enroll #2 via master:** existing device + recovery artifact → new device enrolled, both vouch, full trust.
3. **Enroll #3 via quorum:** two existing siblings co-sign → new device enrolled without recovery artifact.
4. **Trust evaluation:** enrolled device with no vouches = provisional; with N=1 vouch = full; with active challenge = refused.
5. **Archival:** device silent past 90-day window stops counting in trust evaluation; returning device un-archives.
6. **Self-revocation:** device signs its own revocation; subsequent verifications refuse.
7. **Compromised-master detection:** master-signed revocation against active device, target counter-signs Liveness, network state is contested.
8. **Reclamation honored:** M2 publishes claim, window passes silent, peers honor predecessor relationship.
9. **Reclamation refuted:** M2 publishes claim, M1's device publishes Liveness within window, claim invalidated.
10. **CRDT merge:** two siblings' independently-issued Vouching Certs for the same target converge under merge; later supersedes earlier from the same signer.
11. **Wire-format roundtrip:** CBOR encode/decode for every cert type; verify deterministic encoding.
12. **Cross-implementation interop:** dummy second implementation (test fixture in another language or hand-rolled bytes) verifies the wire format is unambiguous.

## Future work (post-v1)

1. Master rotation ceremony (M → M' with quorum signature confirming the rotation, all devices re-enrolled under M').
2. **PQ migration to ML-DSA-44 (FIPS 204) using key-rotation-with-quarantine, NOT perpetual hybrid acceptance.** Post-migration, the identity carries a `pq_required: bool` flag; verifiers reject classical-only signatures from devices under that identity. Cutover is per-identity (different identities can be at different migration stages, but each identity's verifiers know the status). This avoids the downgrade-attack surface industry-wide in WebAuthn / Apple CryptoKit / AT Protocol `xDSA` hybrid composite designs (DR section 4.1). SLH-DSA's 49 KB signatures are unnecessary at our cert publication rates; FN-DSA/Falcon's floating-point side-channel risk is irrelevant since 2.4 KB ML-DSA-44 signatures are well within our wire bandwidth.
3. High-security mode: configurable N as a function of active-sibling count.
4. Hardware-token integration for the master key (YubiKey storage of M instead of mnemonic).
5. Multi-owner shared devices.
6. Cross-identity reclamation chains.
