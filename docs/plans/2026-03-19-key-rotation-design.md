# Key Rotation Design: KERI-Inspired Key Event Log with Pre-Rotation

**Date:** 2026-03-19
**Status:** Approved
**Scope:** `MlDsa65Rotatable` variant in `harmony-identity` + new `harmony-kel` crate
**Bead:** harmony-eo2

## Overview

Harmony's current identity model permanently binds each identity to a
single keypair. If a key is compromised, the identity is lost — there is
no recovery path. This design introduces KERI-inspired key rotation via:

1. **Pre-rotation commitments** — hash of the next keypair stored in the
   current event, proving the rotation was planned before any compromise
2. **Key Event Log (KEL)** — an append-only, BLAKE3-hash-chained sequence
   of events forming the auditable lifecycle of a rotatable identity
3. **Inception-derived addresses** — permanent addresses derived from the
   inception event payload, not from current key material

## Design Philosophy

### Pre-Rotation is Post-Quantum Secure

An attacker who compromises the current signing key cannot forge a rotation
because they don't know which keys hash to the `next_key_commitment`.
BLAKE3 is collision-resistant — even a quantum computer cannot reverse it
(Grover gives only a square-root speedup on preimage, so 256-bit BLAKE3
retains ~128-bit quantum security).

### Addresses Must Be Permanent

The current address derivation (`SHA256(pub_keys)[:16]`) breaks on key
rotation. Rotatable identities derive their address from the inception
event's **unsigned payload** instead: `SHA256(inception_payload)[:16]`.
The signature is excluded from address derivation so the address is
deterministic and stable (re-signing doesn't change the address).
Static identities (`Identity`, `PqIdentity`) keep their key-derived
addresses for Reticulum compatibility.

### Rotation is Explicit, Not Automatic

Key rotation is a deliberate action by the identity owner. There is no
automatic rotation schedule or forced rotation. The owner generates their
next keypair, commits its hash, and stores the keypair in cold storage
until they choose to rotate.

### Dual Signatures for Defense in Depth

Rotation events carry two signatures: one from the **old** (outgoing) key
and one from the **new** (incoming) key. The old-key signature provides
explicit cryptographic consent from the current key holder. The new-key
signature proves possession of the pre-committed keys. Combined with the
pre-rotation commitment hash, this gives three layers of protection:
pre-commitment + old-key authorization + new-key possession.

## Architecture

```
harmony-identity
    ├── CryptoSuite::MlDsa65Rotatable = 0x02  (new variant)
    └── (existing types unchanged)

harmony-crypto
    └── (existing BLAKE3, ML-DSA-65, ML-KEM-768 — all consumed)

harmony-kel  (new crate)
    ├── KeyEvent enum (Inception, Rotation, Interaction)
    ├── InceptionEvent, RotationEvent, InteractionEvent
    └── KeyEventLog (append-only, validated, hash-chained)
```

### Dependencies

```
harmony-crypto
harmony-identity
    └── harmony-kel
```

`harmony-kel` is a pure-data, `no_std`-compatible crate. No I/O, no
networking. Depends on `harmony-identity` for `IdentityHash`, `CryptoSuite`,
`IdentityRef`, and `harmony-crypto` for BLAKE3, ML-DSA-65 verification,
and ML-KEM-768 public key types.

## Change 1: `CryptoSuite::MlDsa65Rotatable`

Add a new variant to `CryptoSuite` in `harmony-identity`:

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(try_from = "u8", into = "u8")]
#[repr(u8)]
pub enum CryptoSuite {
    Ed25519 = 0x00,
    MlDsa65 = 0x01,
    /// ML-DSA-65/ML-KEM-768 with KERI-style key rotation.
    /// Address derived from inception event payload, not current keys.
    MlDsa65Rotatable = 0x02,
}
```

Updates needed:
- `is_post_quantum()`: returns `true` for both `MlDsa65` and `MlDsa65Rotatable`
- `signing_multicodec()`: returns `0x1211` for both (same algorithm)
- `encryption_multicodec()`: returns `0x120c` for both
- `from_byte()` / `TryFrom<u8>`: handle `0x02`
- `from_signing_multicodec()` / `from_encryption_multicodec()`: these return
  `MlDsa65` for the PQ codes (not `MlDsa65Rotatable`, since multicodec
  identifies the algorithm, not the key management strategy). **Note:**
  This means multicodec round-trips are lossy for `MlDsa65Rotatable` —
  `MlDsa65Rotatable.signing_multicodec()` returns `0x1211` but
  `from_signing_multicodec(0x1211)` returns `MlDsa65`. This is intentional
  and implementers should not rely on multicodec round-trips preserving
  the `Rotatable` variant.

### IdentityRef doc comment update

The `IdentityRef.hash` field doc comment currently says
"SHA256(pub_keys)[:16]". This should be updated to note that the
derivation depends on the suite — key-derived for `Ed25519`/`MlDsa65`,
inception-derived for `MlDsa65Rotatable`.

## Change 2: Key Event Types

All event structs derive `Debug, Clone, Serialize, Deserialize`.
`PartialEq` and `Eq` are omitted because `MlDsaPublicKey` and
`MlDsaSignature` do not implement them — equality testing in tests
should compare individual fields or use custom helpers.

### Prerequisite: serde/Debug on crypto types

`MlDsaPublicKey`, `MlDsaSignature`, and `MlKemPublicKey` in
`harmony-crypto` currently derive only `Clone` (or nothing). For
event struct serialization to compile, these types need `Serialize`,
`Deserialize`, and `Debug` implementations. These should be added
as manual byte-based impls in `harmony-crypto` (serializing as raw
byte arrays), gated behind a `serde` feature flag. This is a
prerequisite task before the KEL event types can be implemented.

### Event Payload vs Signed Event

Each event has a **payload** (all fields except signatures) and
**signatures**. The payload is what gets signed and what gets hashed
for address derivation and hash chaining. Signatures are appended
after payload serialization.

### Sequence Numbering

- `InceptionEvent` has an implicit sequence of **0**
- All subsequent events (`RotationEvent`, `InteractionEvent`) share a
  single monotonically increasing counter starting at **1**
- Events can interleave freely: Inception (0) → Interaction (1) →
  Rotation (2) → Interaction (3) → Rotation (4) is valid

### InceptionEvent

Creates a rotatable identity. The identity's permanent address is
derived from the **unsigned payload** of this event.

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InceptionEvent {
    /// Initial ML-DSA-65 signing public key (1,952 bytes).
    pub signing_key: MlDsaPublicKey,
    /// Initial ML-KEM-768 encryption public key (1,184 bytes).
    pub encryption_key: MlKemPublicKey,
    /// BLAKE3(next_signing_pub || next_encryption_pub).
    /// Pre-rotation commitment — proves the next rotation was
    /// planned before the current key could be compromised.
    pub next_key_commitment: [u8; 32],
    /// When this identity was created (unix timestamp seconds).
    /// Advisory only — no monotonicity enforced across events.
    pub created_at: u64,
    /// ML-DSA-65 signature over the payload (all fields above).
    pub signature: MlDsaSignature,
}
```

### RotationEvent

Retires current keys, activates pre-committed keys, commits to the
next set. **Dual-signed**: old key authorizes the rotation, new key
proves possession.

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RotationEvent {
    /// Monotonically increasing (starts at 1 for first rotation).
    pub sequence: u64,
    /// BLAKE3 hash of the serialized previous event payload.
    pub previous_hash: [u8; 32],
    /// New ML-DSA-65 signing key — must match previous event's
    /// next_key_commitment when hashed with encryption_key.
    pub signing_key: MlDsaPublicKey,
    /// New ML-KEM-768 encryption key.
    pub encryption_key: MlKemPublicKey,
    /// BLAKE3 commitment to the NEXT key pair after this one.
    pub next_key_commitment: [u8; 32],
    /// Timestamp (unix seconds). Advisory only.
    pub created_at: u64,
    /// Signed by the OLD (outgoing) signing key — authorizes rotation.
    pub old_signature: MlDsaSignature,
    /// Signed by the NEW (incoming) signing key — proves possession.
    pub new_signature: MlDsaSignature,
}
```

### InteractionEvent

Anchors arbitrary data to the log without rotating keys. Useful for
signed statements, document attestations, trust score publications.

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InteractionEvent {
    /// Sequence number (shared counter with rotation events).
    pub sequence: u64,
    /// BLAKE3 hash of the serialized previous event payload.
    pub previous_hash: [u8; 32],
    /// BLAKE3 hash of the anchored data (data itself is external).
    pub data_hash: [u8; 32],
    /// Timestamp (unix seconds). Advisory only.
    pub created_at: u64,
    /// Signed by the current active signing key.
    pub signature: MlDsaSignature,
}
```

### Unified enum

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum KeyEvent {
    Inception(InceptionEvent),
    Rotation(RotationEvent),
    Interaction(InteractionEvent),
}
```

## Change 3: Pre-Rotation Commitment

### Computing the commitment

```rust
/// Compute the pre-rotation commitment hash.
/// Concatenates the raw key bytes and hashes with BLAKE3.
fn compute_commitment(
    next_signing_key: &MlDsaPublicKey,
    next_encryption_key: &MlKemPublicKey,
) -> [u8; 32] {
    // Concatenate key bytes, then hash via harmony-crypto's blake3_hash
    let mut buf = alloc::vec::Vec::new();
    buf.extend_from_slice(next_signing_key.as_bytes());
    buf.extend_from_slice(next_encryption_key.as_bytes());
    harmony_crypto::hash::blake3_hash(&buf)
}
```

Uses `harmony_crypto::hash::blake3_hash()` (single-slice API) with
concatenation, rather than the raw `blake3` crate's incremental hasher,
to stay within the harmony-crypto abstraction layer.

### Verifying on rotation

When a RotationEvent arrives, the verifier checks:

1. **Pre-rotation commitment**: `BLAKE3(rotation.signing_key || rotation.encryption_key) == previous_event.next_key_commitment`
2. **Old-key authorization**: `ml_dsa_verify(current_signing_key, rotation_payload, rotation.old_signature)` — the outgoing key authorized this rotation
3. **New-key possession**: `ml_dsa_verify(rotation.signing_key, rotation_payload, rotation.new_signature)` — the incoming key proves possession
4. **Hash chain**: `rotation.previous_hash == BLAKE3(serialize(previous_event_payload))`
5. **Sequence monotonicity**: `rotation.sequence == latest_sequence + 1`

All five must pass for the rotation to be accepted.

### Security under key compromise

The attacker has the current signing key but NOT the pre-generated next
keypair. They can:
- Sign Interaction Events — yes, detectable after legitimate rotation
  (conflicting sequence numbers). **Note:** fork detection requires
  external coordination (future Watcher/Witness bead); this crate alone
  validates a single linear log.
- Forge a Rotation Event — **no**, because:
  1. They cannot produce keys matching `next_key_commitment` (BLAKE3 preimage)
  2. Even if they could, they'd need the new private key to produce `new_signature`

The legitimate owner rotates using the pre-committed keys, invalidating
the compromised key.

## Change 4: Key Event Log (KEL)

### Structure

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KeyEventLog {
    /// Permanent address (SHA256(inception_payload)[:16]).
    address: IdentityHash,
    /// Ordered events. events[0] is always Inception.
    events: Vec<KeyEvent>,
}
```

### Invariants

1. `events[0]` is always an `InceptionEvent`
2. `InceptionEvent` has implicit sequence 0; all subsequent events have
   `sequence == previous_event_effective_sequence + 1`
3. For `i > 0`: `events[i].previous_hash == BLAKE3(serialize_payload(events[i-1]))`
4. Each `RotationEvent`'s keys match the previous event's `next_key_commitment`
5. `InceptionEvent` and `InteractionEvent` signatures valid under current active signing key
6. `RotationEvent` dual-signed: `old_signature` valid under current active key,
   `new_signature` valid under the new signing key
7. No event after `events[0]` may be an `InceptionEvent`

### API

```rust
impl KeyEventLog {
    /// Create a new KEL from an inception event.
    /// Validates signature. Derives the permanent address from the
    /// unsigned payload (excludes signature from hash input).
    pub fn from_inception(event: InceptionEvent) -> Result<Self, KelError>;

    /// Append a rotation event.
    /// Validates: hash chain, pre-rotation commitment, sequence,
    /// old-key authorization signature, new-key possession signature.
    pub fn apply_rotation(&mut self, event: RotationEvent) -> Result<(), KelError>;

    /// Append an interaction event.
    /// Validates: hash chain, sequence, signature under current active key.
    pub fn apply_interaction(&mut self, event: InteractionEvent) -> Result<(), KelError>;

    /// The identity's permanent address (from inception).
    pub fn address(&self) -> &IdentityHash;

    /// The current active signing public key.
    pub fn current_signing_key(&self) -> &MlDsaPublicKey;

    /// The current active encryption public key.
    pub fn current_encryption_key(&self) -> &MlKemPublicKey;

    /// The current pre-rotation commitment hash.
    pub fn next_key_commitment(&self) -> &[u8; 32];

    /// Effective sequence number of the latest event.
    /// InceptionEvent = 0, all others use their explicit sequence field.
    pub fn latest_sequence(&self) -> u64;

    /// Number of events in the log.
    pub fn len(&self) -> usize;

    /// The full event history.
    pub fn events(&self) -> &[KeyEvent];

    /// Build an IdentityRef for this rotatable identity.
    pub fn identity_ref(&self) -> IdentityRef;

    /// Serialize the entire log for persistence.
    /// Format: version byte prefix + postcard-encoded log.
    pub fn serialize(&self) -> Result<Vec<u8>, KelError>;

    /// Deserialize a log. Rejects logs with duplicate inception events.
    pub fn deserialize(data: &[u8]) -> Result<Self, KelError>;
}
```

### Address derivation

```rust
// Permanent address = SHA256(inception_payload_bytes)[:16]
// Payload = all InceptionEvent fields EXCEPT signature.
let payload = serialize_inception_payload(&inception_event);
let address = harmony_crypto::hash::truncated_hash(&payload);
```

The `truncated_hash` function (existing in `harmony-crypto`) computes
`SHA256(data)[:16]`, matching the existing address derivation pattern.
The signature is excluded so the address is deterministic regardless
of signing randomness.

### Relationship to `IdentityRef`

```rust
pub fn identity_ref(&self) -> IdentityRef {
    IdentityRef::new(self.address, CryptoSuite::MlDsa65Rotatable)
}
```

Consumers that check `is_post_quantum()` get `true`. Consumers that
specifically match `MlDsa65Rotatable` know to look up the KEL for
current key material rather than using a static public key.

## Error Types

```rust
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KelError {
    /// Inception event has an invalid signature.
    InvalidInceptionSignature,
    /// Rotation event's keys don't match the pre-rotation commitment.
    PreRotationMismatch,
    /// Event's previous_hash doesn't match BLAKE3 of the prior event payload.
    HashChainBroken,
    /// Sequence number is not monotonically increasing.
    SequenceViolation,
    /// Event signature is invalid (Interaction or Rotation old/new).
    InvalidSignature,
    /// Deserialized log contains a second InceptionEvent.
    DuplicateInception,
    /// KEL is empty (no inception event).
    EmptyLog,
    /// Serialization failed.
    SerializeError(&'static str),
    /// Deserialization failed.
    DeserializeError(&'static str),
}
```

`Display` impl. `std::error::Error` under `std` feature.

## `no_std` Compatibility

Same pattern as other Harmony crates:
- `#![cfg_attr(not(feature = "std"), no_std)]` with `extern crate alloc`
- `postcard` for serialization with format version byte prefix
- `serde` with `derive` + `alloc` features
- No `hashbrown` needed (KEL uses `Vec`, not maps)

## What This Crate Does NOT Do

- No network distribution of KEL events (future: Zenoh pub/sub)
- No Watcher/Witness consensus for event ordering (future bead)
- No fork detection across distributed copies (this crate validates
  a single linear log)
- No root/leaf identity hierarchy (future bead)
- No threshold signatures (future bead)
- No interaction event data storage (only the hash; data is external)
- No automatic rotation scheduling
- No timestamp validation (timestamps are advisory)

## Testing Strategy

### CryptoSuite update (harmony-identity)
- `MlDsa65Rotatable` is post-quantum
- `from_byte(0x02)` returns `MlDsa65Rotatable`
- `TryFrom<u8>` for `0x02`
- Multicodec: same signing/encryption codes as `MlDsa65`
- `from_signing_multicodec` still returns `MlDsa65` (not rotatable)
- Serde round-trip for new variant
- Existing tests unchanged

### InceptionEvent
- Create inception, verify signature
- Reject invalid signature
- Derive address from unsigned payload (not including signature)
- Address is deterministic (same payload → same address)

### RotationEvent
- Apply valid rotation with dual signatures, verify keys update
- Reject mismatched pre-rotation commitment
- Reject broken hash chain
- Reject wrong sequence number
- Reject invalid old-key signature
- Reject invalid new-key signature

### InteractionEvent
- Append interaction, verify hash chain
- Reject broken hash chain
- Reject wrong sequence number
- Reject invalid signature (must be current active key)

### KeyEventLog
- `from_inception` creates valid KEL with sequence 0
- `current_signing_key` returns inception key initially
- `current_signing_key` returns rotation key after rotation
- Address is permanent through rotations
- `identity_ref()` returns `MlDsa65Rotatable` suite
- Double rotation (rotate twice in sequence)
- Interaction after rotation uses new key
- Inception-only KEL has sequence 0
- Mixed event sequence (inception → interaction → rotation → interaction)
- Sequence interleaving: inception(0) → interaction(1) → rotation(2) is valid
- Persistence: serialize/deserialize round-trip
- Deserialize rejects log with duplicate inception
- Corrupt data handling
