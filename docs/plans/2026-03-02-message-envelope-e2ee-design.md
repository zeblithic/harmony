# Message Envelope + E2EE Design

**Bead:** harmony-0p6.2
**Date:** 2026-03-02
**Status:** Approved

## Summary

Add a binary message envelope to harmony-zenoh that wraps E2EE payloads for
transmission over Zenoh key expressions. The envelope uses ChaCha20-Poly1305
(from harmony-crypto) with ECDH key agreement (from harmony-identity) and a
fixed 33-byte header whose bytes serve as AAD.

## Decision Record

- **Own binary format** over zenoh-protocol/zenoh-codec: zenoh-protocol's
  Put/Del types are tightly coupled to Zenoh's internal framing (ZBuf,
  WireExpr, macro-generated extensions, uhlc). They assume cleartext payloads
  and bring ~40 transitive deps. Our envelope is an E2EE wrapper — different
  abstraction level.

- **Crypto-only envelope** over full routing metadata: key expressions are
  already the Zenoh topic the message is published on. Duplicating them inside
  the envelope is redundant. The envelope carries only crypto metadata (nonce,
  sender address, sequence) plus the encrypted payload.

## Wire Format

```
 0                   1                   2                   3
 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|  Ver  | Type  |                                               |
+-+-+-+-+-+-+-+-+         sender_address (16 bytes)             |
|                                                               |
|                                                               |
|                                                               |
|               +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|               |                                               |
+-+-+-+-+-+-+-+-+            nonce (12 bytes)                   |
|                                                               |
|               +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|               |                  sequence                     |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                    ciphertext + poly1305 tag                  |
|                          (variable)                           |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
```

### Fields

| Field | Offset | Size | Description |
|-------|--------|------|-------------|
| version | 0 | 4 bits | Format version, initially `0x01` |
| type | 0 | 4 bits | `0x00` = Put (data), `0x01` = Del (tombstone) |
| sender_address | 1 | 16 bytes | `Identity.address_hash()` for sender lookup |
| nonce | 17 | 12 bytes | Random ChaCha20-Poly1305 nonce |
| sequence | 29 | 4 bytes | Big-endian u32, monotonic per sender |
| ciphertext | 33 | variable | Encrypted payload + 16-byte Poly1305 tag |

- **Header size:** 33 bytes fixed.
- **Minimum envelope:** 49 bytes (33 header + 16 byte tag for empty plaintext).
- **AAD:** The 33-byte header is passed as associated data to encrypt/decrypt,
  cryptographically binding routing metadata to the ciphertext.

### Key Derivation

```
shared_secret = ECDH(sender_x25519, recipient_x25519)
shared_key    = HKDF-SHA256(ikm=shared_secret, salt=sender_addr||recipient_addr, info="harmony-envelope-v1", len=32)
```

The salt includes both addresses to ensure directionality: A-to-B and B-to-A
derive different keys. The info string is versioned with the envelope format.

## Module Structure

Lives in the existing `harmony-zenoh` crate:

```
crates/harmony-zenoh/src/
  lib.rs            -- add pub mod envelope
  envelope.rs       -- HarmonyEnvelope, MessageType, seal/open, encode/decode
  error.rs          -- add EnvelopeTooShort, UnsupportedVersion, InvalidMessageType
  keyspace.rs       -- unchanged
  subscription.rs   -- unchanged
```

New workspace dependencies for harmony-zenoh:

- `harmony-crypto` -- aead::encrypt/decrypt, aead::generate_nonce, hkdf
- `harmony-identity` -- Identity, PrivateIdentity, address hashes, ECDH
- `rand_core` -- CryptoRngCore for seal()

No external crates added.

## Public API

```rust
/// Message type carried in the envelope.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MessageType {
    Put = 0,
    Del = 1,
}

/// A decoded Harmony message envelope.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HarmonyEnvelope {
    pub version: u8,
    pub msg_type: MessageType,
    pub sender_address: [u8; 16],
    pub sequence: u32,
    pub plaintext: Vec<u8>,
}

/// Header size in bytes.
pub const HEADER_SIZE: usize = 33;

/// Minimum envelope size (header + poly1305 tag).
pub const MIN_ENVELOPE_SIZE: usize = HEADER_SIZE + 16;

impl HarmonyEnvelope {
    /// Encrypt plaintext and produce a sealed envelope.
    pub fn seal(
        rng: &mut impl CryptoRngCore,
        msg_type: MessageType,
        sender: &PrivateIdentity,
        recipient: &Identity,
        sequence: u32,
        plaintext: &[u8],
    ) -> Result<Vec<u8>, ZenohError>;

    /// Parse and decrypt a sealed envelope.
    pub fn open(
        recipient: &PrivateIdentity,
        sender: &Identity,
        data: &[u8],
    ) -> Result<HarmonyEnvelope, ZenohError>;
}
```

### seal() flow

1. Build 33-byte header (version|type, sender address, random nonce, sequence)
2. ECDH(sender, recipient) -> shared secret
3. HKDF(shared_secret, salt, info) -> 32-byte symmetric key
4. aead::encrypt(key, nonce, plaintext, aad=header)
5. Return header || ciphertext

### open() flow

1. Validate data.len() >= MIN_ENVELOPE_SIZE
2. Split header (33 bytes) and ciphertext
3. Parse version, type, sender_address, nonce, sequence from header
4. ECDH(recipient, sender) -> shared secret
5. HKDF(shared_secret, salt, info) -> 32-byte symmetric key
6. aead::decrypt(key, nonce, ciphertext, aad=header)
7. Return HarmonyEnvelope with decrypted plaintext

## Error Variants

Add to ZenohError:

- `EnvelopeTooShort(usize)` -- data shorter than MIN_ENVELOPE_SIZE
- `UnsupportedVersion(u8)` -- version != 1
- `InvalidMessageType(u8)` -- type not 0 or 1
- `SealFailed` -- encryption failure (delegates to CryptoError)
- `OpenFailed` -- decryption failure (wrong key, tampered, etc.)

## Tests

1. **Roundtrip** -- seal then open recovers plaintext, message type, sender, sequence
2. **Tampered header** -- flip byte in sender/sequence/type, open fails (AAD)
3. **Tampered ciphertext** -- flip byte in encrypted portion, open fails
4. **Wrong recipient** -- different identity cannot decrypt
5. **Put vs Del** -- both types roundtrip correctly
6. **Empty payload** -- 0-byte plaintext works (16-byte ciphertext from tag)
7. **Large payload** -- multi-KB plaintext roundtrips
8. **Sequence preserved** -- encode/decode preserves u32 sequence number
9. **Cross-crate integration** -- PrivateIdentity::generate() + aead confirms dep chain
10. **Version/type encoding** -- verify version|type byte packs/unpacks correctly
11. **Directionality** -- A-to-B and B-to-A produce different ciphertexts (different derived keys)
