# Roxy Content Licensing Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement the `harmony-roxy` library crate — sans-I/O content licensing primitives including license manifests, per-consumer key wrapping, cache lifecycle management, and Zenoh catalog discovery patterns.

**Architecture:** A new workspace member crate (`crates/harmony-roxy/`) following the established harmony pattern: `no_std` by default with `std` feature, postcard for binary serialization, inline `#[cfg(test)] mod tests`, and `thiserror` for error types. Key wrapping uses the existing `Identity::encrypt()` / `PrivateIdentity::decrypt()` ECDH APIs from harmony-identity (Fernet-based internally; a ChaCha20-native path can be added to harmony-identity later).

**Tech Stack:** Rust, serde + postcard, harmony-crypto (HKDF, ChaCha20, AeadKey), harmony-identity (Identity, PrivateIdentity, UcanToken, CapabilityType), harmony-content (ContentId, ContentFlags), bitflags crate

**Design Doc:** `docs/plans/2026-03-08-roxy-content-licensing-design.md`

---

### Task 1: Scaffold the Crate

**Files:**
- Create: `crates/harmony-roxy/Cargo.toml`
- Create: `crates/harmony-roxy/src/lib.rs`
- Modify: `Cargo.toml` (workspace root — add member and workspace dependency)

**Step 1: Add harmony-roxy to workspace members and dependencies**

In the root `Cargo.toml`, add `"crates/harmony-roxy"` to `[workspace] members` (alphabetical order) and add to `[workspace.dependencies]`:

```toml
harmony-roxy = { path = "crates/harmony-roxy", default-features = false }
```

**Step 2: Create `crates/harmony-roxy/Cargo.toml`**

```toml
[package]
name = "harmony-roxy"
version.workspace = true
edition.workspace = true
license.workspace = true
repository.workspace = true
rust-version.workspace = true
description = "Self-sovereign content licensing primitives for the Harmony decentralized stack"

[features]
default = ["std"]
std = [
    "harmony-crypto/std",
    "harmony-identity/std",
    "harmony-content/std",
    "thiserror/std",
]

[dependencies]
harmony-crypto.workspace = true
harmony-identity.workspace = true
harmony-content.workspace = true
thiserror.workspace = true
serde = { workspace = true, default-features = false, features = ["derive", "alloc"] }
postcard = { workspace = true }
bitflags = { workspace = true }
```

**Step 3: Create `crates/harmony-roxy/src/lib.rs`**

```rust
#![cfg_attr(not(feature = "std"), no_std)]
extern crate alloc;

pub mod cache;
pub mod catalog;
pub mod keywrap;
pub mod manifest;
pub mod types;

mod error;
pub use error::RoxyError;
```

**Step 4: Create `crates/harmony-roxy/src/error.rs`**

```rust
use core::fmt;

#[derive(Debug)]
pub enum RoxyError {
    /// Serialization/deserialization failed.
    Serialization(postcard::Error),
    /// Cryptographic operation failed.
    Crypto(harmony_crypto::CryptoError),
    /// Identity operation failed (ECDH, signature).
    Identity(harmony_identity::IdentityError),
    /// Content addressing error.
    Content(harmony_content::ContentError),
    /// Manifest signature verification failed.
    InvalidSignature,
    /// Manifest creator does not match signer.
    CreatorMismatch,
    /// UCAN resource bytes are malformed.
    InvalidResource,
    /// License has expired.
    LicenseExpired,
}

impl fmt::Display for RoxyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Serialization(e) => write!(f, "serialization error: {e}"),
            Self::Crypto(e) => write!(f, "crypto error: {e}"),
            Self::Identity(e) => write!(f, "identity error: {e}"),
            Self::Content(e) => write!(f, "content error: {e}"),
            Self::InvalidSignature => write!(f, "manifest signature verification failed"),
            Self::CreatorMismatch => write!(f, "manifest creator does not match signer"),
            Self::InvalidResource => write!(f, "malformed UCAN resource bytes"),
            Self::LicenseExpired => write!(f, "license has expired"),
        }
    }
}

impl From<postcard::Error> for RoxyError {
    fn from(e: postcard::Error) -> Self {
        Self::Serialization(e)
    }
}

impl From<harmony_crypto::CryptoError> for RoxyError {
    fn from(e: harmony_crypto::CryptoError) -> Self {
        Self::Crypto(e)
    }
}

impl From<harmony_identity::IdentityError> for RoxyError {
    fn from(e: harmony_identity::IdentityError) -> Self {
        Self::Identity(e)
    }
}

impl From<harmony_content::ContentError> for RoxyError {
    fn from(e: harmony_content::ContentError) -> Self {
        Self::Content(e)
    }
}
```

**Step 5: Create stub modules**

Create empty files with just a comment for each module:

- `crates/harmony-roxy/src/manifest.rs` — `//! License manifest types and signing.`
- `crates/harmony-roxy/src/keywrap.rs` — `//! Per-consumer key wrapping via ECDH.`
- `crates/harmony-roxy/src/cache.rs` — `//! Cache lifecycle state machine.`
- `crates/harmony-roxy/src/catalog.rs` — `//! Zenoh key expression patterns for catalog discovery.`
- `crates/harmony-roxy/src/types.rs` — `//! Shared types: artist profile, price, etc.`

**Step 6: Verify it compiles**

Run: `cargo check -p harmony-roxy`
Expected: compiles with no errors

**Step 7: Commit**

```bash
git add crates/harmony-roxy/ Cargo.toml
git commit -m "feat(roxy): scaffold harmony-roxy crate"
```

---

### Task 2: Core Types — LicenseType, UsageRights, Price

**Files:**
- Modify: `crates/harmony-roxy/src/types.rs`

**Step 1: Write failing tests**

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn usage_rights_bitflags() {
        let rights = UsageRights::STREAM | UsageRights::DOWNLOAD;
        assert!(rights.contains(UsageRights::STREAM));
        assert!(rights.contains(UsageRights::DOWNLOAD));
        assert!(!rights.contains(UsageRights::REMIX));
        assert!(!rights.contains(UsageRights::RESHARE));
    }

    #[test]
    fn usage_rights_serialization_round_trip() {
        let rights = UsageRights::STREAM | UsageRights::REMIX;
        let bytes = postcard::to_allocvec(&rights).unwrap();
        let decoded: UsageRights = postcard::from_bytes(&bytes).unwrap();
        assert_eq!(rights, decoded);
    }

    #[test]
    fn license_type_serialization_round_trip() {
        for lt in [
            LicenseType::Free,
            LicenseType::OneTime,
            LicenseType::Subscription,
            LicenseType::Custom,
        ] {
            let bytes = postcard::to_allocvec(&lt).unwrap();
            let decoded: LicenseType = postcard::from_bytes(&bytes).unwrap();
            assert_eq!(lt, decoded);
        }
    }

    #[test]
    fn price_serialization_round_trip() {
        let price = Price {
            amount: 500,
            currency: alloc::string::String::from("USD"),
            per: PricePer::Month,
        };
        let bytes = postcard::to_allocvec(&price).unwrap();
        let decoded: Price = postcard::from_bytes(&bytes).unwrap();
        assert_eq!(price.amount, decoded.amount);
        assert_eq!(price.currency, decoded.currency);
        assert_eq!(price.per, decoded.per);
    }
}
```

**Step 2: Run tests to verify they fail**

Run: `cargo test -p harmony-roxy`
Expected: FAIL — types not defined

**Step 3: Implement types**

```rust
//! Shared types: artist profile, price, licensing terms.

use alloc::string::String;
use bitflags::bitflags;
use serde::{Deserialize, Serialize};

/// Type of license an artist offers for content.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u8)]
pub enum LicenseType {
    /// No token needed. Content key is public.
    Free = 0,
    /// Pay once, access forever (UCAN with no expiry).
    OneTime = 1,
    /// Recurring access windows with expiry.
    Subscription = 2,
    /// Opaque terms referencing an external contract.
    Custom = 3,
}

/// How a price is charged.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u8)]
pub enum PricePer {
    /// One-time payment.
    Once = 0,
    /// Per calendar month.
    Month = 1,
    /// Per calendar year.
    Year = 2,
    /// Per access (each stream/download).
    Access = 3,
}

/// Price for a license.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Price {
    /// Amount in smallest currency unit (e.g., cents for USD).
    pub amount: u64,
    /// Currency identifier (e.g., "USD", "BTC", "HAR").
    pub currency: String,
    /// Billing period.
    pub per: PricePer,
}

bitflags! {
    /// What the consumer is allowed to do with licensed content.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
    pub struct UsageRights: u8 {
        /// Decrypt and play in real-time.
        const STREAM   = 0b0001;
        /// Persist decrypted content locally.
        const DOWNLOAD = 0b0010;
        /// Derive new content from this content.
        const REMIX    = 0b0100;
        /// Delegate access to others via UCAN chain.
        const RESHARE  = 0b1000;
    }
}
```

**Step 4: Run tests to verify they pass**

Run: `cargo test -p harmony-roxy`
Expected: PASS (3 tests)

**Step 5: Commit**

```bash
git add crates/harmony-roxy/src/types.rs
git commit -m "feat(roxy): core types — LicenseType, UsageRights, Price"
```

---

### Task 3: License Manifest — Struct, Serialization, Signing

**Files:**
- Modify: `crates/harmony-roxy/src/manifest.rs`

**Step 1: Write failing tests**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{LicenseType, PricePer, Price, UsageRights};
    use harmony_content::ContentFlags;
    use harmony_identity::PrivateIdentity;
    use rand::rngs::OsRng;

    fn make_test_manifest(artist: &PrivateIdentity) -> LicenseManifest {
        let content_cid = harmony_content::ContentId::for_blob(
            b"test content",
            ContentFlags::default(),
        )
        .unwrap();
        let key_cid = harmony_content::ContentId::for_blob(
            b"encrypted key blob",
            ContentFlags { encrypted: true, ..Default::default() },
        )
        .unwrap();

        LicenseManifest {
            creator: artist.public_identity().address_hash,
            content_cid,
            manifest_version: 1,
            license_type: LicenseType::Subscription,
            price: Some(Price {
                amount: 500,
                currency: alloc::string::String::from("USD"),
                per: PricePer::Month,
            }),
            duration_secs: Some(30 * 24 * 3600), // 30 days
            usage_rights: UsageRights::STREAM | UsageRights::DOWNLOAD,
            expiry_notice_secs: 3 * 24 * 3600, // 3 days
            content_key_cid: key_cid,
            signature: [0u8; 64], // placeholder, will be signed
        }
    }

    #[test]
    fn manifest_serialization_round_trip() {
        let artist = PrivateIdentity::generate(&mut OsRng);
        let manifest = make_test_manifest(&artist);
        let bytes = manifest.to_bytes().unwrap();
        let decoded = LicenseManifest::from_bytes(&bytes).unwrap();
        assert_eq!(manifest.creator, decoded.creator);
        assert_eq!(manifest.license_type, decoded.license_type);
        assert_eq!(manifest.usage_rights, decoded.usage_rights);
        assert_eq!(manifest.expiry_notice_secs, decoded.expiry_notice_secs);
    }

    #[test]
    fn manifest_sign_and_verify() {
        let artist = PrivateIdentity::generate(&mut OsRng);
        let mut manifest = make_test_manifest(&artist);
        manifest.sign(&artist);

        assert!(manifest.verify(&artist.public_identity()).is_ok());
    }

    #[test]
    fn manifest_verify_rejects_wrong_signer() {
        let artist = PrivateIdentity::generate(&mut OsRng);
        let imposter = PrivateIdentity::generate(&mut OsRng);
        let mut manifest = make_test_manifest(&artist);
        manifest.sign(&artist);

        let result = manifest.verify(&imposter.public_identity());
        assert!(result.is_err());
    }

    #[test]
    fn manifest_verify_rejects_tampered_data() {
        let artist = PrivateIdentity::generate(&mut OsRng);
        let mut manifest = make_test_manifest(&artist);
        manifest.sign(&artist);

        // Tamper with the price
        manifest.price = Some(Price {
            amount: 0,
            currency: alloc::string::String::from("USD"),
            per: PricePer::Month,
        });

        let result = manifest.verify(&artist.public_identity());
        assert!(result.is_err());
    }

    #[test]
    fn free_manifest_has_no_price_or_duration() {
        let artist = PrivateIdentity::generate(&mut OsRng);
        let content_cid = harmony_content::ContentId::for_blob(
            b"free content",
            ContentFlags::default(),
        )
        .unwrap();
        let key_cid = harmony_content::ContentId::for_blob(
            b"public key",
            ContentFlags::default(),
        )
        .unwrap();

        let mut manifest = LicenseManifest {
            creator: artist.public_identity().address_hash,
            content_cid,
            manifest_version: 1,
            license_type: LicenseType::Free,
            price: None,
            duration_secs: None,
            usage_rights: UsageRights::STREAM | UsageRights::DOWNLOAD,
            expiry_notice_secs: 0,
            content_key_cid: key_cid,
            signature: [0u8; 64],
        };
        manifest.sign(&artist);
        assert!(manifest.verify(&artist.public_identity()).is_ok());
    }
}
```

**Step 2: Run tests to verify they fail**

Run: `cargo test -p harmony-roxy`
Expected: FAIL — LicenseManifest not defined

**Step 3: Implement LicenseManifest**

```rust
//! License manifest types and signing.

use alloc::vec::Vec;
use harmony_content::ContentId;
use harmony_identity::{Identity, PrivateIdentity};
use serde::{Deserialize, Serialize};

use crate::error::RoxyError;
use crate::types::{LicenseType, Price, UsageRights};

/// A signed, content-addressed document describing licensing terms for content.
///
/// Serialized as postcard, content-addressed (gets its own CID), signed by the
/// artist. Immutable — updated terms = new manifest CID. Old grants reference
/// old terms, so changing terms never breaks existing licenses.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LicenseManifest {
    /// Artist's address hash (16 bytes).
    pub creator: [u8; 16],
    /// CID of the content (single piece or bundle).
    pub content_cid: ContentId,
    /// Schema version for forward compatibility.
    pub manifest_version: u8,
    /// Type of license offered.
    pub license_type: LicenseType,
    /// Price terms (None for Free).
    pub price: Option<Price>,
    /// Access window in seconds per grant (None for perpetual).
    pub duration_secs: Option<u64>,
    /// What the consumer may do with the content.
    pub usage_rights: UsageRights,
    /// Seconds before expiry to notify the consumer.
    pub expiry_notice_secs: u32,
    /// CID of the encrypted symmetric key blob.
    pub content_key_cid: ContentId,
    /// Ed25519 signature over all fields above.
    #[serde(with = "serde_sig")]
    pub signature: [u8; 64],
}

/// Serde helper for fixed-size signature arrays.
mod serde_sig {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    pub fn serialize<S: Serializer>(sig: &[u8; 64], s: S) -> Result<S::Ok, S::Error> {
        sig.as_slice().serialize(s)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<[u8; 64], D::Error> {
        let v: alloc::vec::Vec<u8> = Deserialize::deserialize(d)?;
        v.try_into()
            .map_err(|_| serde::de::Error::custom("expected 64-byte signature"))
    }
}

impl LicenseManifest {
    /// Serialize the manifest to postcard bytes.
    pub fn to_bytes(&self) -> Result<Vec<u8>, RoxyError> {
        Ok(postcard::to_allocvec(self)?)
    }

    /// Deserialize a manifest from postcard bytes.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, RoxyError> {
        Ok(postcard::from_bytes(bytes)?)
    }

    /// Compute the signable bytes (all fields except signature).
    fn signable_bytes(&self) -> Result<Vec<u8>, RoxyError> {
        // Serialize a copy with zeroed signature to get deterministic bytes.
        let mut copy = self.clone();
        copy.signature = [0u8; 64];
        copy.to_bytes()
    }

    /// Sign this manifest with the artist's private identity.
    /// Sets the `signature` field in place.
    pub fn sign(&mut self, artist: &PrivateIdentity) {
        let bytes = self.signable_bytes().expect("serialization of own manifest");
        self.signature = artist.sign(&bytes);
    }

    /// Verify the manifest signature against a public identity.
    /// Returns `Ok(())` if valid, `Err` if signature is invalid or
    /// the identity's address hash doesn't match `creator`.
    pub fn verify(&self, identity: &Identity) -> Result<(), RoxyError> {
        if identity.address_hash != self.creator {
            return Err(RoxyError::CreatorMismatch);
        }
        let bytes = self.signable_bytes()?;
        identity
            .verify(&bytes, &self.signature)
            .map_err(|_| RoxyError::InvalidSignature)
    }
}
```

**Step 4: Run tests to verify they pass**

Run: `cargo test -p harmony-roxy`
Expected: PASS (all tests)

**Step 5: Commit**

```bash
git add crates/harmony-roxy/src/manifest.rs
git commit -m "feat(roxy): license manifest with signing and verification"
```

---

### Task 4: Key Wrapping — Per-Consumer Content Key Distribution

**Files:**
- Modify: `crates/harmony-roxy/src/keywrap.rs`

**Context:** Uses `Identity::encrypt()` and `PrivateIdentity::decrypt()` from harmony-identity, which internally performs ECDH + HKDF + Fernet encryption. This wraps the 32-byte ChaCha20 content key per-consumer. The consumer unwraps with their private identity to get the raw key.

**Step 1: Write failing tests**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use harmony_crypto::aead::AeadKey;
    use harmony_identity::PrivateIdentity;
    use rand::rngs::OsRng;

    #[test]
    fn wrap_and_unwrap_round_trip() {
        let artist = PrivateIdentity::generate(&mut OsRng);
        let consumer = PrivateIdentity::generate(&mut OsRng);
        let content_key = AeadKey::generate(&mut OsRng);

        let wrapped = wrap_key(&mut OsRng, &content_key, consumer.public_identity()).unwrap();
        let unwrapped = unwrap_key(&consumer, &wrapped).unwrap();

        assert_eq!(content_key.as_bytes(), unwrapped.as_bytes());
    }

    #[test]
    fn wrong_consumer_cannot_unwrap() {
        let consumer = PrivateIdentity::generate(&mut OsRng);
        let wrong_consumer = PrivateIdentity::generate(&mut OsRng);
        let content_key = AeadKey::generate(&mut OsRng);

        let wrapped = wrap_key(&mut OsRng, &content_key, consumer.public_identity()).unwrap();
        let result = unwrap_key(&wrong_consumer, &wrapped);

        assert!(result.is_err());
    }

    #[test]
    fn wrapped_key_is_different_each_time() {
        let consumer = PrivateIdentity::generate(&mut OsRng);
        let content_key = AeadKey::generate(&mut OsRng);

        let wrapped1 = wrap_key(&mut OsRng, &content_key, consumer.public_identity()).unwrap();
        let wrapped2 = wrap_key(&mut OsRng, &content_key, consumer.public_identity()).unwrap();

        // Ephemeral key differs each time, so wrapped output differs.
        assert_ne!(wrapped1, wrapped2);

        // But both unwrap to the same content key.
        let k1 = unwrap_key(&consumer, &wrapped1).unwrap();
        let k2 = unwrap_key(&consumer, &wrapped2).unwrap();
        assert_eq!(k1.as_bytes(), k2.as_bytes());
    }
}
```

**Step 2: Run tests to verify they fail**

Run: `cargo test -p harmony-roxy`
Expected: FAIL — wrap_key/unwrap_key not defined

**Step 3: Implement key wrapping**

```rust
//! Per-consumer key wrapping via ECDH.
//!
//! Artists generate a symmetric content key (ChaCha20-Poly1305 `AeadKey`)
//! and encrypt their content with it. To grant access, the content key is
//! "wrapped" (encrypted) for a specific consumer using their public identity.
//! The consumer unwraps with their private identity to recover the key.

use alloc::vec::Vec;
use harmony_crypto::aead::AeadKey;
use harmony_identity::{Identity, PrivateIdentity};
use rand_core::CryptoRngCore;

use crate::error::RoxyError;

/// Wrap a content key for a specific consumer.
///
/// Uses ECDH (ephemeral X25519 → consumer's public key) + HKDF + Fernet
/// internally via `Identity::encrypt()`. The wrapped output can only be
/// unwrapped by the consumer's private identity.
///
/// Returns opaque wrapped bytes. Different on each call (ephemeral key).
pub fn wrap_key(
    rng: &mut impl CryptoRngCore,
    content_key: &AeadKey,
    recipient: &Identity,
) -> Result<Vec<u8>, RoxyError> {
    let wrapped = recipient.encrypt(rng, content_key.as_bytes())?;
    Ok(wrapped)
}

/// Unwrap a content key using the consumer's private identity.
///
/// Returns the recovered `AeadKey` for decrypting content.
pub fn unwrap_key(
    consumer: &PrivateIdentity,
    wrapped: &[u8],
) -> Result<AeadKey, RoxyError> {
    let plaintext = consumer.decrypt(wrapped)?;
    if plaintext.len() != 32 {
        return Err(RoxyError::Crypto(harmony_crypto::CryptoError::InvalidKeyLength {
            expected: 32,
            got: plaintext.len(),
        }));
    }
    let mut key_bytes = [0u8; 32];
    key_bytes.copy_from_slice(&plaintext);
    Ok(AeadKey::from_bytes(key_bytes))
}
```

**Step 4: Run tests to verify they pass**

Run: `cargo test -p harmony-roxy`
Expected: PASS (all tests)

**Step 5: Commit**

```bash
git add crates/harmony-roxy/src/keywrap.rs
git commit -m "feat(roxy): per-consumer key wrapping via ECDH"
```

---

### Task 5: Cache Lifecycle State Machine

**Files:**
- Modify: `crates/harmony-roxy/src/cache.rs`

**Step 1: Write failing tests**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use harmony_content::ContentFlags;

    fn make_cid(data: &[u8]) -> harmony_content::ContentId {
        harmony_content::ContentId::for_blob(data, ContentFlags::default()).unwrap()
    }

    #[test]
    fn active_entry_transitions_to_expiring() {
        let mut mgr = CacheManager::new();
        let manifest_cid = make_cid(b"manifest");
        let content_cid = make_cid(b"content");

        mgr.add_entry(CacheEntry {
            manifest_cid,
            content_cid,
            wrapped_key: alloc::vec![1, 2, 3],
            ucan_not_after: Some(100.0),
            expiry_notice_secs: 10,
            state: CacheState::Active,
            auto_renew: false,
        });

        // At t=89, still 11s before expiry — no action.
        let actions = mgr.tick(89.0);
        assert!(actions.is_empty());

        // At t=91, only 9s left — should transition to Expiring.
        let actions = mgr.tick(91.0);
        assert_eq!(actions.len(), 1);
        assert!(matches!(
            &actions[0],
            CacheAction::NotifyExpiring { seconds_remaining, .. } if *seconds_remaining == 9
        ));
    }

    #[test]
    fn expiring_entry_evicts_on_deadline() {
        let mut mgr = CacheManager::new();
        let manifest_cid = make_cid(b"manifest");
        let content_cid = make_cid(b"content");

        mgr.add_entry(CacheEntry {
            manifest_cid,
            content_cid,
            wrapped_key: alloc::vec![1, 2, 3],
            ucan_not_after: Some(100.0),
            expiry_notice_secs: 10,
            state: CacheState::Active,
            auto_renew: false,
        });

        // Advance past expiry.
        let _ = mgr.tick(91.0); // transitions to Expiring
        let actions = mgr.tick(101.0); // past not_after

        assert_eq!(actions.len(), 2);
        assert!(matches!(&actions[0], CacheAction::WipeKey { .. }));
        assert!(matches!(&actions[1], CacheAction::EvictContent { .. }));

        // Entry should be removed.
        assert_eq!(mgr.entry_count(), 0);
    }

    #[test]
    fn auto_renew_requests_renewal() {
        let mut mgr = CacheManager::new();
        let manifest_cid = make_cid(b"manifest");
        let content_cid = make_cid(b"content");

        mgr.add_entry(CacheEntry {
            manifest_cid,
            content_cid,
            wrapped_key: alloc::vec![1, 2, 3],
            ucan_not_after: Some(100.0),
            expiry_notice_secs: 10,
            state: CacheState::Active,
            auto_renew: true,
        });

        let actions = mgr.tick(91.0);
        assert_eq!(actions.len(), 2);
        assert!(matches!(&actions[0], CacheAction::NotifyExpiring { .. }));
        assert!(matches!(&actions[1], CacheAction::RequestRenewal { .. }));
    }

    #[test]
    fn renewal_success_resets_to_active() {
        let mut mgr = CacheManager::new();
        let manifest_cid = make_cid(b"manifest");
        let content_cid = make_cid(b"content");

        mgr.add_entry(CacheEntry {
            manifest_cid,
            content_cid,
            wrapped_key: alloc::vec![1, 2, 3],
            ucan_not_after: Some(100.0),
            expiry_notice_secs: 10,
            state: CacheState::Active,
            auto_renew: true,
        });

        let _ = mgr.tick(91.0); // transitions to Expiring
        mgr.handle_renewal_success(&manifest_cid, 200.0);

        // Should be back to Active with new deadline.
        let actions = mgr.tick(95.0);
        assert!(actions.is_empty()); // Still well before new expiry
    }

    #[test]
    fn perpetual_license_never_expires() {
        let mut mgr = CacheManager::new();
        let manifest_cid = make_cid(b"manifest");
        let content_cid = make_cid(b"content");

        mgr.add_entry(CacheEntry {
            manifest_cid,
            content_cid,
            wrapped_key: alloc::vec![1, 2, 3],
            ucan_not_after: None, // perpetual
            expiry_notice_secs: 0,
            state: CacheState::Active,
            auto_renew: false,
        });

        // Even at a very large time, no actions.
        let actions = mgr.tick(999_999_999.0);
        assert!(actions.is_empty());
        assert_eq!(mgr.entry_count(), 1);
    }

    #[test]
    fn revoke_immediately_evicts() {
        let mut mgr = CacheManager::new();
        let manifest_cid = make_cid(b"manifest");
        let content_cid = make_cid(b"content");

        mgr.add_entry(CacheEntry {
            manifest_cid,
            content_cid,
            wrapped_key: alloc::vec![1, 2, 3],
            ucan_not_after: Some(100.0),
            expiry_notice_secs: 10,
            state: CacheState::Active,
            auto_renew: false,
        });

        let actions = mgr.revoke(&manifest_cid);
        assert_eq!(actions.len(), 2);
        assert!(matches!(&actions[0], CacheAction::WipeKey { .. }));
        assert!(matches!(&actions[1], CacheAction::EvictContent { .. }));
        assert_eq!(mgr.entry_count(), 0);
    }
}
```

**Step 2: Run tests to verify they fail**

Run: `cargo test -p harmony-roxy`
Expected: FAIL — CacheManager not defined

**Step 3: Implement cache state machine**

```rust
//! Cache lifecycle state machine for licensed content.
//!
//! Green-managed, sans-I/O. The `CacheManager` tracks licensed content
//! cached on a consumer's node and emits `CacheAction`s for the caller
//! to execute (file deletion, notifications, renewal requests).

use alloc::vec;
use alloc::vec::Vec;
use harmony_content::ContentId;
use hashbrown::HashMap;

/// State of a cached license entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheState {
    /// License is valid, content is accessible.
    Active,
    /// Approaching expiry — notification sent, awaiting renewal.
    Expiring,
}

/// A cached content entry with its license state.
#[derive(Debug, Clone)]
pub struct CacheEntry {
    /// CID of the license manifest.
    pub manifest_cid: ContentId,
    /// CID of the encrypted content blob.
    pub content_cid: ContentId,
    /// Consumer-specific wrapped symmetric key.
    pub wrapped_key: Vec<u8>,
    /// UCAN expiry timestamp (None = perpetual).
    pub ucan_not_after: Option<f64>,
    /// Seconds before expiry to notify.
    pub expiry_notice_secs: u32,
    /// Current lifecycle state.
    pub state: CacheState,
    /// Whether to auto-request renewal.
    pub auto_renew: bool,
}

/// Actions emitted by the cache manager for the caller to execute.
#[derive(Debug, Clone)]
pub enum CacheAction {
    /// License is approaching expiry.
    NotifyExpiring {
        manifest_cid: ContentId,
        seconds_remaining: u32,
    },
    /// Auto-renew: request a new UCAN from the artist/delegate.
    RequestRenewal {
        manifest_cid: ContentId,
    },
    /// Zero the cached symmetric key (license expired or revoked).
    WipeKey {
        manifest_cid: ContentId,
    },
    /// Delete the encrypted content blob from local storage.
    EvictContent {
        content_cid: ContentId,
    },
}

/// Sans-I/O cache lifecycle manager.
///
/// Call `tick(now)` each frame/interval. Returns actions for the caller.
pub struct CacheManager {
    entries: HashMap<ContentId, CacheEntry>,
}

impl CacheManager {
    pub fn new() -> Self {
        Self {
            entries: HashMap::new(),
        }
    }

    /// Add a new cache entry (after acquiring a UCAN and wrapped key).
    pub fn add_entry(&mut self, entry: CacheEntry) {
        self.entries.insert(entry.manifest_cid, entry);
    }

    /// Number of tracked entries.
    pub fn entry_count(&self) -> usize {
        self.entries.len()
    }

    /// Handle a successful renewal — reset to Active with new expiry.
    pub fn handle_renewal_success(&mut self, manifest_cid: &ContentId, new_not_after: f64) {
        if let Some(entry) = self.entries.get_mut(manifest_cid) {
            entry.ucan_not_after = Some(new_not_after);
            entry.state = CacheState::Active;
        }
    }

    /// Immediately revoke a license — evict content and wipe key.
    pub fn revoke(&mut self, manifest_cid: &ContentId) -> Vec<CacheAction> {
        if let Some(entry) = self.entries.remove(manifest_cid) {
            vec![
                CacheAction::WipeKey {
                    manifest_cid: entry.manifest_cid,
                },
                CacheAction::EvictContent {
                    content_cid: entry.content_cid,
                },
            ]
        } else {
            Vec::new()
        }
    }

    /// Tick the cache manager. Checks all entries for expiry transitions.
    pub fn tick(&mut self, now: f64) -> Vec<CacheAction> {
        let mut actions = Vec::new();
        let mut to_remove = Vec::new();

        for (cid, entry) in self.entries.iter_mut() {
            let not_after = match entry.ucan_not_after {
                Some(t) => t,
                None => continue, // Perpetual — never expires.
            };

            match entry.state {
                CacheState::Active => {
                    let remaining = not_after - now;
                    if remaining <= entry.expiry_notice_secs as f64 {
                        entry.state = CacheState::Expiring;
                        actions.push(CacheAction::NotifyExpiring {
                            manifest_cid: *cid,
                            seconds_remaining: remaining.max(0.0) as u32,
                        });
                        if entry.auto_renew {
                            actions.push(CacheAction::RequestRenewal {
                                manifest_cid: *cid,
                            });
                        }
                    }
                }
                CacheState::Expiring => {
                    if now >= not_after {
                        actions.push(CacheAction::WipeKey {
                            manifest_cid: *cid,
                        });
                        actions.push(CacheAction::EvictContent {
                            content_cid: entry.content_cid,
                        });
                        to_remove.push(*cid);
                    }
                }
            }
        }

        for cid in to_remove {
            self.entries.remove(&cid);
        }

        actions
    }
}
```

**Step 4: Run tests to verify they pass**

Run: `cargo test -p harmony-roxy`
Expected: PASS (all tests)

**Step 5: Commit**

```bash
git add crates/harmony-roxy/src/cache.rs
git commit -m "feat(roxy): cache lifecycle state machine with expiry and revocation"
```

---

### Task 6: Catalog — Zenoh Key Expressions & Artist Profile

**Files:**
- Modify: `crates/harmony-roxy/src/catalog.rs`

**Step 1: Write failing tests**

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn catalog_key_for_content() {
        let artist_hash = [0xABu8; 16];
        let key = catalog_key(&artist_hash, ContentCategory::Music, "deadbeef");
        assert_eq!(key, "roxy/catalog/abababababababababababababababab/music/deadbeef");
    }

    #[test]
    fn catalog_key_for_meta() {
        let artist_hash = [0x01u8; 16];
        let key = meta_key(&artist_hash);
        assert_eq!(key, "roxy/catalog/01010101010101010101010101010101/meta");
    }

    #[test]
    fn license_key_for_consumer() {
        let consumer_hash = [0xCDu8; 16];
        let key = license_key(&consumer_hash, "aabbccdd");
        assert_eq!(key, "roxy/license/cdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcd/aabbccdd");
    }

    #[test]
    fn revocation_key() {
        let artist_hash = [0xEFu8; 16];
        let key = revocation_key(&artist_hash, "11223344");
        assert_eq!(key, "roxy/revocation/efefefefefefefefefefefefefefefef/11223344");
    }

    #[test]
    fn subscription_pattern_all_music_from_artist() {
        let artist_hash = [0xABu8; 16];
        let pattern = artist_content_pattern(&artist_hash, Some(ContentCategory::Music));
        assert_eq!(
            pattern,
            "roxy/catalog/abababababababababababababababab/music/**"
        );
    }

    #[test]
    fn subscription_pattern_all_from_artist() {
        let artist_hash = [0xABu8; 16];
        let pattern = artist_content_pattern(&artist_hash, None);
        assert_eq!(pattern, "roxy/catalog/abababababababababababababababab/**");
    }

    #[test]
    fn subscription_pattern_all_music() {
        let pattern = global_content_pattern(ContentCategory::Music);
        assert_eq!(pattern, "roxy/catalog/*/music/**");
    }

    #[test]
    fn content_category_serialization_round_trip() {
        for cat in [
            ContentCategory::Music,
            ContentCategory::Video,
            ContentCategory::Text,
            ContentCategory::Image,
            ContentCategory::Software,
            ContentCategory::Dataset,
            ContentCategory::Bundle,
        ] {
            let bytes = postcard::to_allocvec(&cat).unwrap();
            let decoded: ContentCategory = postcard::from_bytes(&bytes).unwrap();
            assert_eq!(cat, decoded);
        }
    }

    #[test]
    fn artist_profile_serialization_round_trip() {
        let profile = ArtistProfile {
            address_hash: [0xABu8; 16],
            display_name: alloc::string::String::from("Test Artist"),
            bio: alloc::string::String::from("A test artist"),
            avatar_cid: None,
            manifest_cids: alloc::vec![],
        };
        let bytes = postcard::to_allocvec(&profile).unwrap();
        let decoded: ArtistProfile = postcard::from_bytes(&bytes).unwrap();
        assert_eq!(profile.display_name, decoded.display_name);
        assert_eq!(profile.address_hash, decoded.address_hash);
    }
}
```

**Step 2: Run tests to verify they fail**

Run: `cargo test -p harmony-roxy`
Expected: FAIL — types not defined

**Step 3: Implement catalog module**

```rust
//! Zenoh key expression patterns for catalog discovery.
//!
//! Artists publish license manifests to structured Zenoh key expressions.
//! Consumers subscribe to patterns to discover content. Yellow indexes
//! manifests for semantic search.

use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;
use harmony_content::ContentId;
use serde::{Deserialize, Serialize};

/// Category of content for Zenoh key expression routing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u8)]
pub enum ContentCategory {
    Music = 0,
    Video = 1,
    Text = 2,
    Image = 3,
    Software = 4,
    Dataset = 5,
    Bundle = 6,
}

impl ContentCategory {
    /// Zenoh path segment for this category.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Music => "music",
            Self::Video => "video",
            Self::Text => "text",
            Self::Image => "image",
            Self::Software => "software",
            Self::Dataset => "dataset",
            Self::Bundle => "bundle",
        }
    }
}

/// Artist profile published at `roxy/catalog/{artist_hash}/meta`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArtistProfile {
    /// Artist's address hash (16 bytes).
    pub address_hash: [u8; 16],
    /// Display name.
    pub display_name: String,
    /// Short biography.
    pub bio: String,
    /// CID of avatar image (optional).
    pub avatar_cid: Option<ContentId>,
    /// CIDs of published license manifests.
    pub manifest_cids: Vec<ContentId>,
}

/// Build a Zenoh key expression for a specific content manifest.
///
/// Pattern: `roxy/catalog/{artist_hash}/{category}/{manifest_id}`
pub fn catalog_key(artist_hash: &[u8; 16], category: ContentCategory, manifest_id: &str) -> String {
    format!(
        "roxy/catalog/{}/{}/{}",
        hex::encode(artist_hash),
        category.as_str(),
        manifest_id,
    )
}

/// Build a Zenoh key expression for an artist's profile.
///
/// Pattern: `roxy/catalog/{artist_hash}/meta`
pub fn meta_key(artist_hash: &[u8; 16]) -> String {
    format!("roxy/catalog/{}/meta", hex::encode(artist_hash))
}

/// Build a Zenoh key expression for a consumer's active license.
///
/// Pattern: `roxy/license/{consumer_hash}/{manifest_id}`
pub fn license_key(consumer_hash: &[u8; 16], manifest_id: &str) -> String {
    format!(
        "roxy/license/{}/{}",
        hex::encode(consumer_hash),
        manifest_id,
    )
}

/// Build a Zenoh key expression for a license revocation.
///
/// Pattern: `roxy/revocation/{artist_hash}/{ucan_hash}`
pub fn revocation_key(artist_hash: &[u8; 16], ucan_hash: &str) -> String {
    format!(
        "roxy/revocation/{}/{}",
        hex::encode(artist_hash),
        ucan_hash,
    )
}

/// Build a Zenoh subscription pattern for a specific artist's content.
///
/// If `category` is `Some`, matches only that category.
/// If `None`, matches all categories.
///
/// Examples:
/// - `roxy/catalog/{hash}/music/**` — all music from one artist
/// - `roxy/catalog/{hash}/**` — all content from one artist
pub fn artist_content_pattern(
    artist_hash: &[u8; 16],
    category: Option<ContentCategory>,
) -> String {
    match category {
        Some(cat) => format!(
            "roxy/catalog/{}/{}/**",
            hex::encode(artist_hash),
            cat.as_str(),
        ),
        None => format!("roxy/catalog/{}/**", hex::encode(artist_hash)),
    }
}

/// Build a Zenoh subscription pattern for a content category from any artist.
///
/// Example: `roxy/catalog/*/music/**` — all music from anyone
pub fn global_content_pattern(category: ContentCategory) -> String {
    format!("roxy/catalog/*/{}/**", category.as_str())
}
```

**Step 4: Add `hex` dependency to `Cargo.toml`**

Add to `crates/harmony-roxy/Cargo.toml` under `[dependencies]`:

```toml
hex = { workspace = true }
```

**Step 5: Run tests to verify they pass**

Run: `cargo test -p harmony-roxy`
Expected: PASS (all tests)

**Step 6: Commit**

```bash
git add crates/harmony-roxy/
git commit -m "feat(roxy): Zenoh catalog key expressions and artist profile"
```

---

### Task 7: UCAN Resource Encoding

**Files:**
- Create: `crates/harmony-roxy/src/resource.rs`
- Modify: `crates/harmony-roxy/src/lib.rs` (add `pub mod resource;`)

**Context:** The UCAN token's `resource` field is opaque `Vec<u8>` (max 256 bytes). Roxy defines a format for encoding the manifest CID reference in this field so UCAN tokens can be verified against specific license manifests.

**Step 1: Write failing tests**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use harmony_content::ContentFlags;

    fn make_cid(data: &[u8]) -> harmony_content::ContentId {
        harmony_content::ContentId::for_blob(data, ContentFlags::default()).unwrap()
    }

    #[test]
    fn encode_decode_round_trip() {
        let manifest_cid = make_cid(b"manifest");
        let encoded = encode_resource(&manifest_cid);
        let decoded = decode_resource(&encoded).unwrap();
        assert_eq!(manifest_cid, decoded);
    }

    #[test]
    fn encoded_resource_fits_ucan() {
        let manifest_cid = make_cid(b"manifest");
        let encoded = encode_resource(&manifest_cid);
        // ContentId is 32 bytes + 1 byte version prefix = 33 bytes.
        // UCAN max resource is 256 bytes. Plenty of room.
        assert!(encoded.len() <= 256);
    }

    #[test]
    fn decode_rejects_wrong_version() {
        let mut encoded = alloc::vec![99u8]; // wrong version
        encoded.extend_from_slice(&[0u8; 32]);
        let result = decode_resource(&encoded);
        assert!(result.is_err());
    }

    #[test]
    fn decode_rejects_short_input() {
        let result = decode_resource(&[1u8; 5]);
        assert!(result.is_err());
    }
}
```

**Step 2: Run tests to verify they fail**

Run: `cargo test -p harmony-roxy`
Expected: FAIL — encode_resource/decode_resource not defined

**Step 3: Implement resource encoding**

```rust
//! UCAN resource field encoding for Roxy license grants.
//!
//! The UCAN token's `resource` field is opaque bytes (max 256).
//! Roxy encodes the manifest CID so grants can be verified against
//! specific license manifests.

use alloc::vec::Vec;
use harmony_content::ContentId;

use crate::error::RoxyError;

/// Current resource encoding version.
const RESOURCE_VERSION: u8 = 1;

/// Encode a manifest CID into UCAN resource bytes.
///
/// Format: `[version: 1B][manifest_cid: 32B]` = 33 bytes total.
pub fn encode_resource(manifest_cid: &ContentId) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(33);
    bytes.push(RESOURCE_VERSION);
    bytes.extend_from_slice(&manifest_cid.to_bytes());
    bytes
}

/// Decode a manifest CID from UCAN resource bytes.
pub fn decode_resource(bytes: &[u8]) -> Result<ContentId, RoxyError> {
    if bytes.len() < 33 {
        return Err(RoxyError::InvalidResource);
    }
    if bytes[0] != RESOURCE_VERSION {
        return Err(RoxyError::InvalidResource);
    }
    let mut cid_bytes = [0u8; 32];
    cid_bytes.copy_from_slice(&bytes[1..33]);
    Ok(ContentId::from_bytes(cid_bytes))
}
```

**Step 4: Add `pub mod resource;` to `lib.rs`**

Add after the existing module declarations:

```rust
pub mod resource;
```

**Step 5: Run tests to verify they pass**

Run: `cargo test -p harmony-roxy`
Expected: PASS (all tests)

**Step 6: Run full workspace tests and clippy**

Run: `cargo test --workspace`
Expected: All tests pass (365+ existing + new roxy tests)

Run: `cargo clippy --workspace`
Expected: Zero warnings

**Step 7: Commit**

```bash
git add crates/harmony-roxy/
git commit -m "feat(roxy): UCAN resource encoding for manifest CID references"
```

---

### Summary

| Task | What it builds | Tests |
|---|---|---|
| 1 | Crate scaffold, error types | compiles |
| 2 | LicenseType, UsageRights, Price | 3 tests |
| 3 | LicenseManifest with signing/verification | 5 tests |
| 4 | Key wrapping (wrap_key/unwrap_key) | 3 tests |
| 5 | Cache lifecycle state machine | 6 tests |
| 6 | Zenoh catalog key expressions, ArtistProfile | 9 tests |
| 7 | UCAN resource encoding | 4 tests |

**Total: 7 tasks, ~30 tests, one new crate.**

After all tasks complete, the `harmony-roxy` crate provides everything a Harmony app needs to implement content licensing: manifest authoring, key distribution, cache management, catalog discovery, and UCAN integration. No I/O, no runtime coupling — pure state machines and data types.
