# ContentId Signal Flags — Storage-Class Bits in the 256-bit Address Space

**Date:** 2026-03-07
**Status:** Approved
**Scope:** harmony-content (primary), harmony-crypto (additive)

## Problem

ContentId is a 32-byte content-addressed identifier. Currently it encodes a 224-bit truncated SHA-256 hash, a 20-bit payload size, and a 12-bit type tag with checksum. But the network has no way to know — from the address alone — how a blob should be stored, replicated, cached, or cleaned up. Storage policy decisions require inspecting the blob or relying on out-of-band metadata.

The network needs three signals baked into every ContentId:

1. **Encrypted or cleartext?** Determines scannability (AI training, search indexing), quota attribution (network-shared vs owner-charged), and at-rest storage requirements.
2. **Durable or ephemeral?** Determines caching priority (LFU weight), cold-storage eligibility, and replication behavior.
3. **Hash algorithm?** Provides "power of choice" collision resistance (SHA-256 vs SHA-224) and enables double-verification of critical content.

## Design

### Bit Layout

Cannibalize the top 3 bits of `hash[0]`. The hash field remains 28 bytes physically, but effective hash length drops from 224 to 221 bits.

```
ContentId (32 bytes):

  byte 0:     [E:1][D:1][A:1][hash_bits:5]
  bytes 1-27: hash_bits (216 bits)
  bytes 28-31: size_and_tag (unchanged)

  E = encrypted:  0 = cleartext, 1 = encrypted
  D = durability: 0 = durable,   1 = ephemeral
  A = algorithm:  0 = SHA-256,   1 = SHA-224
```

**Collision resistance:** Birthday bound drops from 2^112 to 2^110.5 — negligible. With power-of-choice (both algorithms), an attacker must collide in SHA-256 AND SHA-224 simultaneously.

### Hash Computation

1. Compute full digest based on algorithm flag:
   - A=0: `SHA-256(blob)` → 32 bytes, truncate to 28
   - A=1: `SHA-224(blob)` → 28 bytes (native output length)
2. Mask top 3 bits: `hash[0] &= 0x1F`
3. Set flags: `hash[0] |= (E << 7) | (D << 6) | (A << 5)`

### Identity Semantics

Same bytes + different flags = **different ContentId**. The flags are part of the content's identity because they prescribe how the network handles the data. An ephemeral copy must not dedup against a durable copy.

### Storage Policy Matrix

| E | D | Class | At-rest | Caching | Replication | Scanning |
|---|---|-------|---------|---------|-------------|----------|
| 0 | 0 | **Public Durable** | Disk + cold storage | Aggressive LFU, highly copyable | Eagerly replicate to libraries | Yes — AI training, search, semantic indexing |
| 0 | 1 | **Public Ephemeral** | Memory/hot cache only | Penalized LFU, short TTL | No cold-storage migration | Allowed while live |
| 1 | 0 | **Private Durable** | Encrypted-at-rest, disk OK | Best-effort cache for owner convenience | Optimistic — owner hosts primary | No — only useful to key holders |
| 1 | 1 | **Private Ephemeral** | **Never write to disk** | No cache, deliver-and-delete | None — single delivery | No |

### Power of Choice (Algorithm Bit)

The algorithm bit serves three purposes:

1. **Collision resolution.** If two different blobs produce the same 221-bit truncated SHA-256, the colliding blob can be re-addressed with A=1 (SHA-224). The probability of colliding in both is ~2^-221.

2. **Double-verification for critical content.** High-value blobs (kernel images, signed binaries) can be registered under both algorithms. Verification checks both CIDs — an attacker needs a simultaneous preimage in both hash families.

3. **Defense in depth with Athenaeum.** A blob ingested through Athenaeum has a Book/blueprint where every 4KB chunk has its own 32-bit ChunkAddr with independent hash verification across 4 algorithm variants (SHA-256 MSB/LSB, SHA-224 MSB/LSB). For a 1MB blob, that's 256 independent chunk-level verifications on top of the blob-level CID.

## API Changes

### harmony-crypto (additive, non-breaking)

```rust
/// SHA-224 digest (28 bytes, native output length).
pub fn sha224_hash(data: &[u8]) -> [u8; 28];
```

### harmony-content (breaking — signature change)

**New type:**

```rust
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ContentFlags {
    pub encrypted: bool,
    pub ephemeral: bool,
    pub alt_hash: bool,  // false = SHA-256, true = SHA-224
}
```

**Modified constructors:**

```rust
ContentId::for_blob(data: &[u8], flags: ContentFlags) -> Result<Self, ContentError>
ContentId::for_bundle(bundle_bytes: &[u8], children: &[ContentId], flags: ContentFlags) -> Result<Self, ContentError>
```

**New accessor:**

```rust
impl ContentId {
    pub fn flags(&self) -> ContentFlags;
}
```

**Modified verification:**

```rust
impl ContentId {
    pub fn verify_hash(&self, data: &[u8]) -> bool;
    // Now checks A flag to select SHA-256 or SHA-224,
    // masks top 3 bits before comparing.
}
```

**Checksum:** No changes. Flags are embedded in `hash[0]`, which is already part of the checksum input. Different flags → different checksum automatically.

### Unchanged crates

| Crate | Why unchanged |
|-------|---------------|
| harmony-athenaeum | Uses raw `[u8; 32]` SHA-256 CIDs, not ContentId. ChunkAddr has its own algorithm field. |
| harmony-os ContentServer | Uses Athenaeum's raw CIDs. Will inherit flag support when/if migrated to ContentId. |
| CidType / tag encoding | Completely untouched — flags are orthogonal to content type. |

### Migration

All existing call sites pass `ContentFlags::default()` (cleartext, durable, SHA-256) to preserve current behavior. Canonical test vectors for the default flags don't change. New canonical vectors are needed for each flag combination.

## Testing

- **Flag round-trip:** Set each flag combination, verify `flags()` returns correct values
- **Hash verification with flags:** `verify_hash` passes for matching data+algorithm, fails for wrong data or wrong algorithm
- **Identity distinctness:** Same data with different flags produces different ContentId
- **Checksum covers flags:** Corrupting a flag bit causes checksum verification to fail
- **SHA-224 produces valid CIDs:** Blobs addressed with A=1 round-trip correctly
- **Canonical vectors:** One vector per flag combination (8 total: 2^3) for cross-language interop
- **Power of choice:** Same blob with A=0 and A=1 produces different hashes (SHA-256 ≠ SHA-224)
