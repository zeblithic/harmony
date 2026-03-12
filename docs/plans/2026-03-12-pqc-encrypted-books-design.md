# Post-Quantum Cryptography & Encrypted Books Design

## Problem

Harmony's cryptographic stack is built on Curve25519 (X25519 key exchange,
Ed25519 signatures). These algorithms will be broken by quantum computers
running Shor's algorithm. The "harvest now, decrypt later" threat means
data encrypted today with Curve25519 is already compromised in terms of
long-term confidentiality. NSA's CNSA 2.0 mandates exclusive use of
post-quantum algorithms by 2030-2033.

Separately, the self-indexing books feature (PR #64) established a sentinel
system for embedding metadata at page 0 of a Book. Public books use the
`00` sentinel for a table-of-contents index. Encrypted books need a
different mechanism: a metadata header carrying encryption keys, owner
identity, expiry, and handling instructions — signaled by the `11` sentinel.

## Solution

Three-phase implementation delivered as one cohesive unit:

1. PQC primitives in `harmony-crypto` (ML-KEM-768, ML-DSA-65)
2. PQC identity types in `harmony-identity`
3. Encrypted books in `harmony-athenaeum`

## Design Decisions

### Clean Break from Reticulum Crypto

Reticulum is treated as a public internet transport. Harmony speaks its
wire format for interop but treats everything over Reticulum as
plaintext-equivalent. The existing X25519/Ed25519/Fernet path stays for
that public interface but never touches encrypted content.

A Reticulum node can "talk" Curve25519 to Harmony, but there is no
protocol negotiation or downgrade — if a peer can't speak PQC, the
connection is treated as public/unencrypted. This prevents downgrade
attacks entirely.

### 128-bit Addresses

Address format unchanged: `SHA256(encapsulation_pub || signing_pub)[:16]`.
128 bits gives 2^64 birthday bound (~18 quintillion), sufficient for every
conceivable agent on the internet for decades. The input keys change
(ML-KEM-768 + ML-DSA-65 instead of X25519 + Ed25519) but the address
format is identical.

### Security Level

ML-KEM-768 + ML-DSA-65 (NIST Level 3, ~AES-192 equivalent). This is the
industry-consensus recommendation (Chrome, Cloudflare, Signal). The type
system is designed so Level 5 (ML-KEM-1024 + ML-DSA-87) can slot in later
without breaking changes.

### Rust Crate Selection

RustCrypto ecosystem: `ml-kem` and `ml-dsa` crates. Pure Rust, no_std,
pass all NIST KAT vectors, integrate with the existing RustCrypto stack
(`sha2`, `chacha20poly1305`, `hkdf`, `ed25519-dalek`). Same `zeroize` and
`rand_core` patterns already used throughout Harmony.

---

## Phase 1: PQC Crypto Primitives (`harmony-crypto`)

### New Modules

**`ml_kem.rs`** — ML-KEM-768 key encapsulation:

```
MlKemPublicKey    (1,184 bytes, Clone)
MlKemSecretKey    (2,400 bytes, Zeroize on drop)
MlKemCiphertext   (1,088 bytes)
MlKemSharedSecret (32 bytes, Zeroize on drop)

generate(rng) -> (MlKemPublicKey, MlKemSecretKey)
encapsulate(rng, pub_key) -> (MlKemCiphertext, MlKemSharedSecret)
decapsulate(secret_key, ciphertext) -> MlKemSharedSecret
```

**`ml_dsa.rs`** — ML-DSA-65 digital signatures:

```
MlDsaPublicKey   (1,952 bytes, Clone)
MlDsaSecretKey   (4,032 bytes, Zeroize on drop)
MlDsaSignature   (3,309 bytes)

generate(rng) -> (MlDsaPublicKey, MlDsaSecretKey)
sign(secret_key, message) -> MlDsaSignature
verify(pub_key, message, signature) -> Result<(), CryptoError>
```

**`hybrid_kem.rs`** — X25519 + ML-KEM-768 combined key establishment
(defense-in-depth for the transition period):

```
Combined shared secret = HKDF-SHA256(
    ikm = X25519_shared || ML-KEM_shared,
    salt = context-dependent,
    info = "harmony-hybrid-kem-v1",
    len = 32
) -> 32-byte symmetric key for ChaCha20-Poly1305
```

### Unchanged

- `hash.rs`, `hkdf.rs`, `aead.rs` (ChaCha20-Poly1305) — untouched
- `fernet.rs` — stays for Reticulum public-channel compat

### New Error Variants

```rust
CryptoError::MlKemEncapsulationFailed
CryptoError::MlKemDecapsulationFailed
CryptoError::MlDsaSignFailed
CryptoError::MlDsaVerifyFailed
```

---

## Phase 2: PQC Identity (`harmony-identity`)

### New Types

**`PqIdentity`** (public):

```rust
pub struct PqIdentity {
    pub encryption_key: MlKemPublicKey,      // 1,184 bytes
    pub verifying_key: MlDsaPublicKey,       // 1,952 bytes
    pub address_hash: [u8; 16],              // SHA256(enc_key || ver_key)[:16]
}
```

**`PqPrivateIdentity`** (full keypairs):

```rust
pub struct PqPrivateIdentity {
    pub identity: PqIdentity,
    encryption_secret: MlKemSecretKey,       // 2,400 bytes, Zeroize
    signing_key: MlDsaSecretKey,             // 4,032 bytes, Zeroize
}
```

Same API surface as the classical pair: `generate`, `sign`, `verify`,
`encrypt`, `decrypt`, `to_public_bytes`, `from_public_bytes`,
`to_private_bytes`, `from_private_bytes`.

### PQC Encryption Flow

```
1. ML-KEM encapsulate(recipient.encryption_key) -> (ciphertext, shared_secret)
2. HKDF-SHA256(shared_secret, salt=recipient.address_hash,
               info="harmony-pq-encrypt-v1") -> 32B key
3. ChaCha20-Poly1305(key, nonce, plaintext) -> encrypted_payload
4. Wire: [1,088B ML-KEM ciphertext][12B nonce][encrypted_payload + 16B tag]
```

### Classical Types Unchanged

`Identity` and `PrivateIdentity` stay exactly as they are for Reticulum
public-channel interop. No changes, no deprecation.

### UCAN Updates

New `crypto_suite` byte at the start of the wire format:

```
[1B crypto_suite][16B issuer][16B audience][1B capability]
[2B resource_len][NB resource][8B not_before][8B expires_at]
[16B nonce][1B has_proof][32B proof?]
[variable signature: 64B Ed25519 or 3,309B ML-DSA-65]
```

- `0x00` = Ed25519 (classical, public-channel UCANs)
- `0x01` = ML-DSA-65 (PQC, Harmony-native UCANs)

---

## Phase 3: Encrypted Books (`harmony-athenaeum`)

### Book Type Detection

Three book types, detected by the first 4 bytes of page 0:

| First 4 bytes | Type | Page 0 meaning |
|---|---|---|
| `0x3FFFFFFF` (`00` sentinel) | Self-indexing public | ToC — index of PageAddrs |
| `0xFFFFFFFC` (`11` sentinel) | Encrypted | Metadata — keys, expiry, handling |
| Anything else | Raw | No embedded metadata |

### Encrypted Book Metadata Format

Each metadata page starts with the `11` sentinel (`0xFFFFFFFC`) in its
first 4 bytes. Remaining 4,092 bytes are payload.

Reading: consume pages while first 4 bytes = `0xFFFFFFFC`. Concatenate
payload portions (bytes 4..4096 of each). First non-`11`-sentinel page
is the start of encrypted data.

**Metadata wire format** (concatenated payload):

```
Offset     Size    Field
0          2       version (u16 LE)
2          1       flags (bit 0 = has_expiry, bit 1 = has_tags)
3          1       encryption_algo (0x00 = ChaCha20-Poly1305)
4          2       owner_key_len (u16 LE) — 1184 for ML-KEM-768
6          K       owner_public_key
6+K        2       encapsulated_key_len (u16 LE) — 1088 for ML-KEM-768
8+K        C       encapsulated_key
8+K+C      2       signature_len (u16 LE) — 3309 for ML-DSA-65
10+K+C     S       signature
10+K+C+S   8       expiry (u64 LE, unix timestamp) — if has_expiry
...        2       tags_len (u16 LE) — if has_tags
...        T       tags (opaque bytes) — if has_tags
```

Minimum size with ML-KEM-768 + ML-DSA-65:
10 + 1,184 + 1,088 + 3,309 = **5,591 bytes → 2 metadata pages**.

### BookType Enum

```rust
pub enum BookType {
    Raw,                         // No embedded metadata
    SelfIndexing,                // Page 0 = ToC (00 sentinel)
    Encrypted { metadata_pages: u8 },  // Page 0+ = metadata (11 sentinel)
}
```

Replaces the `self_indexing: bool` field on `Book`.

### Data Page Encryption

Data pages are encrypted with the symmetric key encapsulated in the
metadata. Each page independently encrypted with ChaCha20-Poly1305 using
a page-index-derived nonce:

```
nonce = HKDF-SHA256(shared_secret,
                    salt=page_index_as_u8,
                    info="harmony-page-nonce")[:12]
```

This allows random-access decryption of individual pages.

### Volume Serialization

The flags byte at offset 38 in the book wire format uses 2 bits for
`BookType`:

```
bits 0-1: book_type (00=Raw, 01=SelfIndexing, 10=Encrypted, 11=reserved)
bits 2-7: reserved
```

Backward compatible: `0x00` → Raw, `0x01` → SelfIndexing (matches the
boolean encoding from PR #64).

### API

```rust
Book::from_blob_encrypted(cid, data, metadata) -> Result<Self, BookError>
Book::metadata_pages(&self) -> Option<&[u8]>
Book::data_page_count(&self) -> usize
Book::data_pages() -> skips metadata pages
Book::reassemble() -> skips metadata pages
Book::is_encrypted() -> bool
```

---

## Scope Boundaries

### In scope

- ML-KEM-768 and ML-DSA-65 in `harmony-crypto`
- `PqIdentity` / `PqPrivateIdentity` in `harmony-identity`
- UCAN crypto_suite byte
- Encrypted book type with `11` sentinel metadata pages
- Volume serialization `BookType` refactor
- Comprehensive tests (KATs, round-trip, backward compat)

### Out of scope

| Item | Reason |
|---|---|
| Mnemonic key derivation (BIP-39/32/85) | Separate wallet/UX concern |
| Zero-knowledge proofs | Future layer on top of PQC identity |
| ML-KEM-1024 / ML-DSA-87 (Level 5) | Type system accepts them; only Level 3 implemented |
| Hardware acceleration | Performance optimization, not correctness |
| Message envelope updates | Consumes new crypto; separate follow-on |
| QoS / bandwidth prioritization | Networking-layer concern |
| Encyclopedia / Volume structural changes | Encrypted book logic encapsulated in `Book` |

---

## Testing Strategy

- **NIST KATs** for ML-KEM-768 and ML-DSA-65 via RustCrypto test vectors
- **Round-trip**: generate → encapsulate → decapsulate; generate → sign → verify
- **PQC identity**: encrypt → decrypt; sign → verify; address derivation
- **Encrypted book**: construct → serialize metadata → parse → decrypt pages → verify original data
- **Volume backward compat**: Raw and SelfIndexing books deserialize correctly after BookType refactor
- **Reticulum interop**: existing 14 cross-language tests remain green (classical crypto untouched)
