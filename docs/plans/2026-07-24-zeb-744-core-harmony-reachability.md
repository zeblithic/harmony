# harmony-reachability core crate (ZEB-744 PR 1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a new core crate `harmony-reachability` housing the reachability announce record (byte-stable CBOR, migrated verbatim from harmony-client) plus a generic multi-device LWW kernel (`ReachabilityRecord` trait, `lww_newer` comparator, `MultiDeviceMap`, `ReachabilityFallback` trait). This is PR 1 of ZEB-744; PR 2 rewires harmony-client onto it.

**Architecture:** The record is a byte-for-byte move — same field order, same `#[serde(rename)]` keys, same custom byte-string encoding — validated by the migrated golden-hex wire vector. The kernel is the ~150 lines of the client's `ReachabilityResolver` that are genuinely generic; the Harmony-specific resolver policy stays in the client (rewired onto this kernel in PR 2). The crate takes **no** iroh/pkarr/identity/crypto dependency and no client-only types, so it is pin-trap-immune.

**Tech Stack:** Rust, `serde` (derive + std), `ciborium` (CBOR), `async-trait` (fallback trait). Dev: `hex`. Std crate (uses `std::net::SocketAddr`).

## Global Constraints

- **Byte-identity is the acceptance gate.** `canonical_cbor_encode(ReachabilityAnnouncePayload)` MUST equal the pinned legacy hex `EXPECTED_LEGACY_HEX` (Task 2). Any field reorder, `#[serde(rename)]` change, byte-string→array flip, or dropped `skip_serializing_if` is a network-compat break — forbidden.
- The `[u8;N]` fields MUST serialize via `serialize_bytes_as_bstr` (CBOR major-type-2 byte string), never the serde default array path.
- Field declaration order on `ReachabilityAnnouncePayload` is exactly `nd, rl, da, ts, sg, bs, ba`. On `DelegateEndpoint` exactly `d, ep, vk, hr, pn`.
- `bs`/`ba` carry `#[serde(default, skip_serializing_if = ...)]` so a butler-less record stays byte-identical to the legacy encoding.
- No `iroh`, `pkarr`, `harmony-owner`, `harmony-identity`, `harmony-crypto`, or `tokio` dependency. Crate stays crypto-free.
- `serde` MUST enable the `std` feature (SocketAddr `Serialize`/`Deserialize` live behind it).
- Rename only: `ButlerSetEntry` → `DelegateEndpoint`. The `#[serde(rename)]` wire keys (`d`/`ep`/`vk`/`hr`/`pn`) are unchanged, so bytes are identical.
- Final gate both: `cargo fmt --all -- --check`, `cargo clippy -p harmony-reachability --all-targets -- -D warnings`, `cargo test -p harmony-reachability`.

---

## File Structure

- `crates/harmony-reachability/Cargo.toml` — manifest (create)
- `crates/harmony-reachability/src/lib.rs` — crate doc + module decls + re-exports (create)
- `crates/harmony-reachability/src/canonical.rs` — CBOR encode/decode + serde byte-string helpers + `is_zero_u64` + error type (create)
- `crates/harmony-reachability/src/record.rs` — `DelegateEndpoint`, `ReachabilityAnnouncePayload`, `canonical_payload_bytes`, wire tests (create)
- `crates/harmony-reachability/src/kernel.rs` — `ReachabilityRecord`, `lww_newer`, `MultiDeviceMap`, `ReachabilityFallback` + kernel tests (create)
- `Cargo.toml` (workspace root) — add `crates/harmony-reachability` to `members` (modify)

---

### Task 1: Crate scaffold + canonical CBOR + serde byte-string helpers

**Files:**
- Create: `crates/harmony-reachability/Cargo.toml`
- Create: `crates/harmony-reachability/src/lib.rs`
- Create: `crates/harmony-reachability/src/canonical.rs`
- Modify: `Cargo.toml` (workspace root, `members` array)

**Interfaces:**
- Produces: `canonical::canonical_cbor_encode<T: Serialize>(&T) -> Result<Vec<u8>, CborError>`, `canonical::canonical_cbor_decode<T: DeserializeOwned>(&[u8]) -> Result<T, CborError>`, `canonical::serialize_bytes_as_bstr<const N: usize, S>(&[u8;N], S)`, `canonical::deserialize_bytes_from_bstr<'de, const N: usize, D>(D) -> Result<[u8;N], _>`, `canonical::is_zero_u64(&u64) -> bool`, `pub enum CborError`.

- [ ] **Step 1: Create the manifest**

`crates/harmony-reachability/Cargo.toml`:
```toml
[package]
name = "harmony-reachability"
description = "Reachability announce record (byte-stable CBOR) + multi-device LWW kernel"
version.workspace = true
edition.workspace = true
license.workspace = true
repository.workspace = true
rust-version.workspace = true

[dependencies]
serde = { workspace = true, features = ["derive", "std"] }
ciborium = { workspace = true }
async-trait = { workspace = true }
thiserror = { workspace = true }

[dev-dependencies]
hex = { workspace = true }
```
NOTE: `serde` needs `std` for `SocketAddr`. If `cargo build -p harmony-reachability` reports a missing `ciborium` writer impl, add `features = ["std"]` to the `ciborium` line (ciborium's `Write` for `Vec<u8>` is behind its default/std feature and the workspace pins `default-features = false`).

- [ ] **Step 2: Add to workspace members**

In the workspace-root `Cargo.toml`, add `"crates/harmony-reachability",` to the `members` array (keep the array's existing ordering/formatting).

- [ ] **Step 3: Write `canonical.rs`** (verbatim move of the byte-critical helpers, crate-local error)

```rust
//! Byte-stable CBOR encode/decode + the byte-string serde helpers used by the
//! reachability record. Moved verbatim from harmony-client
//! (`owner_state_crypto::canonical_cbor_encode` + `owner_state_types` bstr
//! helpers) so the on-wire encoding is byte-identical; see the record's golden
//! vector for the compat lock.

use serde::{de::DeserializeOwned, Deserializer, Serialize, Serializer};

#[derive(Debug, thiserror::Error)]
pub enum CborError {
    #[error("CBOR encode failed: {0}")]
    Encode(String),
    #[error("CBOR decode failed: {0}")]
    Decode(String),
}
```
(`thiserror` is declared in the manifest, Task 1 Step 1.)

```rust
/// Deterministic CBOR encode: a thin `ciborium::into_writer` wrapper. Byte-stable
/// across instances for a serde tree with same-length map keys at each level, no
/// f32/f64, no HashMap, no CBOR tags (the reachability record satisfies this).
pub fn canonical_cbor_encode<T: Serialize>(value: &T) -> Result<Vec<u8>, CborError> {
    let mut buf = Vec::new();
    ciborium::into_writer(value, &mut buf).map_err(|e| CborError::Encode(format!("{e}")))?;
    Ok(buf)
}

/// Decoder paired with [`canonical_cbor_encode`]; rejects trailing bytes so a
/// fingerprinted/signed encoding can't be extended to a distinct byte sequence
/// that decodes to the same value.
pub fn canonical_cbor_decode<T: DeserializeOwned>(bytes: &[u8]) -> Result<T, CborError> {
    let mut cursor = std::io::Cursor::new(bytes);
    let value = ciborium::from_reader(&mut cursor).map_err(|e| CborError::Decode(format!("{e}")))?;
    if cursor.position() as usize != bytes.len() {
        return Err(CborError::Decode(format!(
            "trailing bytes after canonical value: consumed {} of {}",
            cursor.position(),
            bytes.len()
        )));
    }
    Ok(value)
}

/// Serialize a byte array as a CBOR byte string (major type 2), NOT an array of
/// u8. Load-bearing for wire compat — see the record's golden vector.
pub fn serialize_bytes_as_bstr<const N: usize, S>(b: &[u8; N], s: S) -> Result<S::Ok, S::Error>
where
    S: Serializer,
{
    s.serialize_bytes(b)
}

/// Deserialize a CBOR byte string into a fixed-size array.
pub fn deserialize_bytes_from_bstr<'de, const N: usize, D>(d: D) -> Result<[u8; N], D::Error>
where
    D: Deserializer<'de>,
{
    use serde::de::Visitor;
    use std::fmt;

    struct BytesVisitor<const N: usize>;

    impl<'de, const N: usize> Visitor<'de> for BytesVisitor<N> {
        type Value = [u8; N];

        fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
            write!(formatter, "a byte array of length {}", N)
        }

        fn visit_bytes<E>(self, value: &[u8]) -> Result<[u8; N], E>
        where
            E: serde::de::Error,
        {
            if value.len() != N {
                return Err(E::custom(format!("expected {} bytes, got {}", N, value.len())));
            }
            let mut arr = [0u8; N];
            arr.copy_from_slice(value);
            Ok(arr)
        }

        fn visit_byte_buf<E>(self, v: Vec<u8>) -> Result<[u8; N], E>
        where
            E: serde::de::Error,
        {
            self.visit_bytes(&v)
        }
    }

    d.deserialize_bytes(BytesVisitor::<N>)
}

/// `skip_serializing_if` predicate: a zero stamp means "no butler set" and is
/// elided so legacy blobs stay byte-identical.
pub fn is_zero_u64(v: &u64) -> bool {
    *v == 0
}
```

- [ ] **Step 4: Write `lib.rs`**

```rust
//! Reachability announce record + a generic multi-device LWW kernel.
//!
//! Extracted from harmony-client (ZEB-744 / ZEB-571 item 7). The record is a
//! byte-stable CBOR value published to the pkarr DHT; the kernel is the generic
//! `(owner, node_id)`-keyed last-writer-wins substrate. App-specific reachability
//! policy (source arbitration, reconnect/liveness integration, pkarr refresh)
//! stays in the consuming app.
//!
//! This crate is transport- and identity-agnostic: no iroh, no pkarr, no
//! identity/crypto dependency. Signing over the record is the caller's concern.

pub mod canonical;
pub mod kernel;
pub mod record;

pub use canonical::{canonical_cbor_decode, canonical_cbor_encode, CborError};
pub use record::{DelegateEndpoint, ReachabilityAnnouncePayload};
```
NOTE: `kernel` is added in Task 3 — until then, either stub `pub mod kernel {}` or omit the `mod kernel;`/kernel re-exports and add them in Task 3. Prefer omitting kernel lines here and adding them in Task 3 Step 5 so each task compiles standalone.

- [ ] **Step 5: Add a bstr round-trip test in `canonical.rs`**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use serde::Deserialize;

    #[derive(Serialize, Deserialize, PartialEq, Debug)]
    struct Probe {
        #[serde(serialize_with = "serialize_bytes_as_bstr", deserialize_with = "deserialize_bytes_from_bstr")]
        b: [u8; 4],
    }

    #[test]
    fn bstr_is_major_type_2_and_round_trips() {
        let p = Probe { b: [1, 2, 3, 4] };
        let bytes = canonical_cbor_encode(&p).expect("encode");
        // map(1) "b" -> bstr(4) 01020304. The bstr header for a 4-byte string is
        // 0x44 (major type 2, length 4) — NOT 0x84 (major type 4, array len 4).
        assert!(bytes.windows(1).any(|w| w[0] == 0x44), "value must encode as a CBOR byte string");
        let back: Probe = canonical_cbor_decode(&bytes).expect("decode");
        assert_eq!(back, p);
    }

    #[test]
    fn decode_rejects_trailing_bytes() {
        let p = Probe { b: [9; 4] };
        let mut bytes = canonical_cbor_encode(&p).expect("encode");
        bytes.push(0x00);
        assert!(canonical_cbor_decode::<Probe>(&bytes).is_err());
    }
}
```

- [ ] **Step 6: Run tests**

Run: `cargo test -p harmony-reachability`
Expected: PASS (2 tests). If `ciborium`/`serde` features are wrong, fix per the Task-1 NOTEs.

- [ ] **Step 7: Commit**

```bash
git add crates/harmony-reachability/Cargo.toml crates/harmony-reachability/src/lib.rs crates/harmony-reachability/src/canonical.rs Cargo.toml
git commit -m "harmony-reachability: crate scaffold + canonical CBOR + bstr helpers (ZEB-744)"
```

---

### Task 2: The reachability record (byte-preserving move)

**Files:**
- Create: `crates/harmony-reachability/src/record.rs`
- Modify: `crates/harmony-reachability/src/lib.rs` (already declares `mod record` + re-exports from Task 1)

**Interfaces:**
- Consumes: `canonical::{canonical_cbor_encode, serialize_bytes_as_bstr, deserialize_bytes_from_bstr, is_zero_u64, CborError}`.
- Produces: `pub struct DelegateEndpoint`, `pub struct ReachabilityAnnouncePayload`, `pub fn canonical_payload_bytes(&ReachabilityAnnouncePayload) -> Result<Vec<u8>, CborError>`.

- [ ] **Step 1: Write the failing golden-vector test first** (put it in `record.rs`'s test module; it will not compile until the struct exists — that is the intended red)

```rust
#[cfg(test)]
mod tests {
    use super::*;

    fn fixture_payload() -> ReachabilityAnnouncePayload {
        ReachabilityAnnouncePayload {
            iroh_node_id: [0xAB; 32],
            home_relay_url: "https://derp.example/".into(),
            direct_addresses: vec![],
            announced_at_ms: 1_700_000_000_000,
            identity_signature: [0xCD; 64],
            butler_set: Vec::new(),
            bs_at: 0,
        }
    }

    /// Byte-compat lock (migrated from harmony-client, DO NOT REGENERATE): a
    /// record WITHOUT a butler set must encode byte-identically to the legacy
    /// (pre-butler-set) wire encoding, so already-deployed peers keep decoding
    /// published pkarr routing blobs unchanged.
    #[test]
    fn routing_blob_without_butler_set_is_wire_identical_to_legacy() {
        const EXPECTED_LEGACY_HEX: &str = "a5626e645820abababababababababababababababababababababababababababababababab62726c7568747470733a2f2f646572702e6578616d706c652f626461806274731b0000018bcfe568006273675840cdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcd";
        let p = fixture_payload();
        let bytes = canonical_payload_bytes(&p).expect("encode");
        assert_eq!(hex::encode(&bytes), EXPECTED_LEGACY_HEX, "legacy wire encoding drifted");
    }
}
```

- [ ] **Step 2: Run it to confirm it fails to compile**

Run: `cargo test -p harmony-reachability record::`
Expected: FAIL (cannot find `ReachabilityAnnouncePayload` / `canonical_payload_bytes`).

- [ ] **Step 3: Write `record.rs`** (verbatim struct move; `ButlerSetEntry` → `DelegateEndpoint`; wire keys unchanged)

```rust
//! Reachability announce record: the node-id / relay / direct-addrs / identity-sig
//! quad (+ an optional delegate-endpoint advertisement) published to the pkarr
//! DHT. Byte-stable CBOR — see the golden vector in tests.
//!
//! Moved from harmony-client `reachability_record.rs` (ZEB-744). The inner
//! identity signature and delegate/butler *policy* stay in the app; this crate
//! carries only the record shape + its byte-stable encoding.

use serde::{Deserialize, Serialize};
use std::net::SocketAddr;

use crate::canonical::{
    canonical_cbor_encode, deserialize_bytes_from_bstr, is_zero_u64, serialize_bytes_as_bstr,
    CborError,
};

/// One delegate endpoint advertised in a reachability record: another device of
/// the same owner that can be reached (and, in the app, can accept sealed
/// deposits) on the owner's behalf. Byte-identical to harmony-client's
/// `ButlerSetEntry` (renamed; wire keys `d`/`ep`/`vk`/`hr`/`pn` unchanged).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DelegateEndpoint {
    /// 16-byte identity hash of the device.
    #[serde(
        rename = "d",
        serialize_with = "serialize_bytes_as_bstr",
        deserialize_with = "deserialize_bytes_from_bstr"
    )]
    pub device_id: [u8; 16],

    /// Iroh EndpointId / NodeId (32-byte transport key).
    #[serde(
        rename = "ep",
        serialize_with = "serialize_bytes_as_bstr",
        deserialize_with = "deserialize_bytes_from_bstr"
    )]
    pub iroh_endpoint_id: [u8; 32],

    /// The device's Ed25519 verify key.
    #[serde(
        rename = "vk",
        serialize_with = "serialize_bytes_as_bstr",
        deserialize_with = "deserialize_bytes_from_bstr"
    )]
    pub device_ed25519_verify: [u8; 32],

    /// Home relay URL for dialing the device.
    #[serde(rename = "hr")]
    pub home_relay: String,

    /// Pinned always-on device.
    #[serde(rename = "pn")]
    pub pinned: bool,
}

/// A reachability announce record. Field order and `#[serde(rename)]` keys are
/// the byte-compat surface — do not reorder or rename. All keys are 2 chars to
/// keep the same-length-keys CBOR determinism invariant.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReachabilityAnnouncePayload {
    /// Iroh NodeId (Ed25519 public key, 32 bytes).
    #[serde(
        rename = "nd",
        serialize_with = "serialize_bytes_as_bstr",
        deserialize_with = "deserialize_bytes_from_bstr"
    )]
    pub iroh_node_id: [u8; 32],

    /// Home relay URL.
    #[serde(rename = "rl")]
    pub home_relay_url: String,

    /// Direct-traversal hint addresses (may be empty).
    #[serde(rename = "da")]
    pub direct_addresses: Vec<SocketAddr>,

    /// Wall-clock milliseconds when this record was authored.
    #[serde(rename = "ts")]
    pub announced_at_ms: u64,

    /// Inner Ed25519 signature by the author's identity key. Computed and
    /// verified by the app (the preimage binds app-side envelope fields);
    /// zero-filled on the pkarr-published path. 64 bytes.
    #[serde(
        rename = "sg",
        serialize_with = "serialize_bytes_as_bstr",
        deserialize_with = "deserialize_bytes_from_bstr"
    )]
    pub identity_signature: [u8; 64],

    /// Ordered delegate-endpoint advertisement. Elided when empty so records
    /// without it stay byte-identical to the legacy encoding.
    #[serde(rename = "bs", default, skip_serializing_if = "Vec::is_empty")]
    pub butler_set: Vec<DelegateEndpoint>,

    /// Wall-clock ms freshness stamp for the delegate set. Zero (elided) means
    /// "no delegate set".
    #[serde(rename = "ba", default, skip_serializing_if = "is_zero_u64")]
    pub bs_at: u64,
}

/// Canonical-encode for hashing / signing / publishing.
pub fn canonical_payload_bytes(p: &ReachabilityAnnouncePayload) -> Result<Vec<u8>, CborError> {
    canonical_cbor_encode(p)
}
```

- [ ] **Step 4: Run the golden vector**

Run: `cargo test -p harmony-reachability record::`
Expected: PASS. If the hex differs, STOP — a byte-affecting property drifted (field order / rename / bstr encoding). Do not edit the hex; fix the struct.

- [ ] **Step 5: Add the remaining migrated wire tests** (append to the `record.rs` test module)

```rust
    fn fixture_delegate(seed: u8) -> DelegateEndpoint {
        DelegateEndpoint {
            device_id: [seed; 16],
            iroh_endpoint_id: [seed.wrapping_add(1); 32],
            device_ed25519_verify: [seed.wrapping_add(2); 32],
            home_relay: "https://use1-1.relay.iroh.network./".into(),
            pinned: false,
        }
    }

    #[test]
    fn legacy_routing_blob_decodes_with_empty_butler_set() {
        let legacy = canonical_payload_bytes(&fixture_payload()).expect("encode");
        let decoded: ReachabilityAnnouncePayload =
            ciborium::de::from_reader(&legacy[..]).expect("decode legacy");
        assert!(decoded.butler_set.is_empty());
        assert_eq!(decoded.bs_at, 0);
    }

    #[test]
    fn routing_blob_with_butler_set_round_trips() {
        let mut p = fixture_payload();
        p.butler_set = vec![fixture_delegate(0x10), fixture_delegate(0x20)];
        p.bs_at = 1_700_000_000_000;
        let bytes = canonical_payload_bytes(&p).expect("encode");
        let decoded: ReachabilityAnnouncePayload =
            ciborium::de::from_reader(&bytes[..]).expect("decode");
        assert_eq!(decoded, p);
    }

    #[test]
    fn roundtrip_cbor() {
        let p = fixture_payload();
        let bytes = canonical_payload_bytes(&p).expect("encode");
        let decoded: ReachabilityAnnouncePayload =
            ciborium::de::from_reader(&bytes[..]).expect("decode");
        assert_eq!(decoded, p);
    }

    #[test]
    fn payload_keys_are_2_chars() {
        let bytes = canonical_payload_bytes(&fixture_payload()).expect("encode");
        let val: ciborium::Value = ciborium::de::from_reader(&bytes[..]).expect("decode");
        for (k, _) in val.as_map().expect("payload is map") {
            assert_eq!(k.as_text().expect("key is text").chars().count(), 2);
        }
    }

    #[test]
    fn encoded_size_with_two_entries_under_bep44_budget() {
        let p = ReachabilityAnnouncePayload {
            iroh_node_id: [0xAB; 32],
            home_relay_url: "https://use1-1.relay.iroh.network./".into(),
            direct_addresses: vec![
                "203.0.113.7:62103".parse().expect("v4"),
                "[2001:db8::1234:5678]:62103".parse().expect("v6"),
            ],
            announced_at_ms: 1_700_000_000_000,
            identity_signature: [0xCD; 64],
            butler_set: vec![fixture_delegate(0x10), fixture_delegate(0x20)],
            bs_at: 1_700_000_000_000,
        };
        assert!(canonical_payload_bytes(&p).expect("encode").len() < 900);
    }
```
NOTE: the client's `butler_set_capped_at_two` / `stale_bs_at_is_filtered_by_reader` tests are NOT migrated — they exercise `fresh_butler_set`/`durable_butler_set`, which reference client `butler_deposit` constants and stay client-side.

- [ ] **Step 6: Run the record tests**

Run: `cargo test -p harmony-reachability record::`
Expected: PASS (6 tests).

- [ ] **Step 7: Commit**

```bash
git add crates/harmony-reachability/src/record.rs
git commit -m "harmony-reachability: reachability record (byte-preserving move) + wire vectors (ZEB-744)"
```

---

### Task 3: The multi-device LWW kernel + fallback trait

**Files:**
- Create: `crates/harmony-reachability/src/kernel.rs`
- Modify: `crates/harmony-reachability/src/lib.rs` (add `pub mod kernel;` + kernel re-exports)

**Interfaces:**
- Consumes: `record::ReachabilityAnnouncePayload`.
- Produces: `trait ReachabilityRecord`, `fn lww_newer`, `struct MultiDeviceMap<Owner, V>`, `trait ReachabilityFallback<Owner>`.

- [ ] **Step 1: Write failing kernel tests** (in `kernel.rs`)

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone)]
    struct Rec { node: [u8; 32], at: u64 }
    impl ReachabilityRecord for Rec {
        fn node_id(&self) -> [u8; 32] { self.node }
        fn announced_at_ms(&self) -> u64 { self.at }
    }
    fn rec(n: u8, at: u64) -> Rec { Rec { node: [n; 32], at } }

    #[test]
    fn lww_newer_orders_by_clock_then_announced_then_node() {
        // Higher clock wins regardless of announce.
        assert!(lww_newer(&1u64, &rec(1, 9), &2u64, &rec(1, 0)));
        assert!(!lww_newer(&2u64, &rec(1, 0), &1u64, &rec(1, 9)));
        // Equal clock: higher announced_at wins.
        assert!(lww_newer(&5u64, &rec(1, 100), &5u64, &rec(1, 200)));
        // Equal clock + announced: greater node_id wins.
        assert!(lww_newer(&5u64, &rec(1, 100), &5u64, &rec(2, 100)));
        // Full equality is NOT newer (byte-identical replay is a no-op).
        assert!(!lww_newer(&5u64, &rec(1, 100), &5u64, &rec(1, 100)));
    }

    #[test]
    fn multi_device_map_range_and_reverse() {
        let mut m: MultiDeviceMap<[u8; 16], u32> = MultiDeviceMap::new();
        let owner_a = [0xAA; 16];
        let owner_b = [0xBB; 16];
        m.insert((owner_a, [1; 32]), 10);
        m.insert((owner_a, [2; 32]), 11);
        m.insert((owner_b, [1; 32]), 20);
        // Owner-prefix range returns only that owner's devices.
        let a: Vec<u32> = m.range_owner(&owner_a).map(|(_, v)| *v).collect();
        assert_eq!(a.len(), 2);
        assert!(a.contains(&10) && a.contains(&11));
        // Reverse-by-node finds across owners (both owners have node [1;32]).
        let hits: Vec<[u8; 16]> = m.find_by_node_id(&[1; 32]).map(|((o, _), _)| *o).collect();
        assert_eq!(hits.len(), 2);
    }
}
```

- [ ] **Step 2: Run to confirm failure**

Run: `cargo test -p harmony-reachability kernel::`
Expected: FAIL (kernel items not defined).

- [ ] **Step 3: Write `kernel.rs`**

```rust
//! Generic multi-device last-writer-wins reachability substrate: the
//! `(owner, node_id)` keying + LWW comparator + async fallback trait. App-specific
//! source arbitration / reconnect / liveness / refresh policy is layered on top
//! by the consumer (see harmony-client's `ReachabilityResolver`).

use async_trait::async_trait;
use std::collections::BTreeMap;

use crate::record::ReachabilityAnnouncePayload;

/// A record the kernel can order and index: it exposes a 32-byte node id and an
/// author-stamped announce time. Implemented by the reachability record and by
/// any app record that wants the LWW/multi-device machinery.
pub trait ReachabilityRecord {
    fn node_id(&self) -> [u8; 32];
    fn announced_at_ms(&self) -> u64;
}

impl ReachabilityRecord for ReachabilityAnnouncePayload {
    fn node_id(&self) -> [u8; 32] {
        self.iroh_node_id
    }
    fn announced_at_ms(&self) -> u64 {
        self.announced_at_ms
    }
}

/// Same-source LWW comparator: is `next` strictly newer than `prev`? Ordering:
/// primary by `clock` (`Ord`), ties by greater `announced_at_ms()`, remaining
/// ties by lexicographically greater `node_id()`. Full equality returns `false`
/// (a byte-identical replay is not a change). The caller supplies the clock's
/// ordering (e.g. an HLC compared as `(wall_ms, logical, device_id)`).
pub fn lww_newer<C, R>(prev_clock: &C, prev_rec: &R, next_clock: &C, next_rec: &R) -> bool
where
    C: Ord,
    R: ReachabilityRecord,
{
    use std::cmp::Ordering;
    match next_clock.cmp(prev_clock) {
        Ordering::Greater => true,
        Ordering::Less => false,
        Ordering::Equal => match next_rec.announced_at_ms().cmp(&prev_rec.announced_at_ms()) {
            Ordering::Greater => true,
            Ordering::Less => false,
            Ordering::Equal => next_rec.node_id() > prev_rec.node_id(),
        },
    }
}

/// A `BTreeMap` keyed by `(owner, node_id)` so a peer's multiple devices coexist
/// under one owner. Wraps the two non-trivial access patterns the reachability
/// resolver relies on: an owner-prefix range and a reverse-by-node-id scan.
#[derive(Debug, Clone)]
pub struct MultiDeviceMap<Owner: Ord + Copy, V> {
    inner: BTreeMap<(Owner, [u8; 32]), V>,
}

impl<Owner: Ord + Copy, V> Default for MultiDeviceMap<Owner, V> {
    fn default() -> Self {
        Self { inner: BTreeMap::new() }
    }
}

impl<Owner: Ord + Copy, V> MultiDeviceMap<Owner, V> {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert(&mut self, key: (Owner, [u8; 32]), value: V) -> Option<V> {
        self.inner.insert(key, value)
    }

    pub fn get(&self, key: &(Owner, [u8; 32])) -> Option<&V> {
        self.inner.get(key)
    }

    pub fn get_mut(&mut self, key: &(Owner, [u8; 32])) -> Option<&mut V> {
        self.inner.get_mut(key)
    }

    pub fn entry(&mut self, key: (Owner, [u8; 32])) -> std::collections::btree_map::Entry<'_, (Owner, [u8; 32]), V> {
        self.inner.entry(key)
    }

    pub fn remove(&mut self, key: &(Owner, [u8; 32])) -> Option<V> {
        self.inner.remove(key)
    }

    pub fn iter(&self) -> impl Iterator<Item = (&(Owner, [u8; 32]), &V)> {
        self.inner.iter()
    }

    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// All `(key, value)` entries for one owner (every device), via the
    /// `(owner, 0..)..=(owner, 0xFF..)` prefix range.
    pub fn range_owner(&self, owner: &Owner) -> impl Iterator<Item = (&(Owner, [u8; 32]), &V)> {
        self.inner
            .range((*owner, [0u8; 32])..=(*owner, [0xFFu8; 32]))
    }

    /// Owner keys of this owner's devices (helper for bulk removal).
    pub fn owner_keys(&self, owner: &Owner) -> Vec<(Owner, [u8; 32])> {
        self.range_owner(owner).map(|(k, _)| *k).collect()
    }

    /// All entries whose node-id half matches, across owners (reverse lookup).
    pub fn find_by_node_id<'a>(
        &'a self,
        node_id: &'a [u8; 32],
    ) -> impl Iterator<Item = (&'a (Owner, [u8; 32]), &'a V)> {
        self.inner.iter().filter(move |((_, n), _)| n == node_id)
    }
}

/// Async fallback consulted on a cache miss (in the app: the pkarr resolver
/// adapter). Kept behind this trait so the kernel takes no transport dependency.
#[async_trait]
pub trait ReachabilityFallback<Owner>: Send + Sync {
    async fn resolve(&self, owner: &Owner) -> Vec<ReachabilityAnnouncePayload>;
}
```
NOTE (clippy): `MultiDeviceMap` has `is_empty` + `len`, so `#[must_use]`/len-without-is_empty lints are satisfied. If clippy flags `new_without_default`, `Default` is already derived-equivalent above.

- [ ] **Step 4: Run kernel tests**

Run: `cargo test -p harmony-reachability kernel::`
Expected: PASS (2 tests).

- [ ] **Step 5: Wire `lib.rs` re-exports**

Add to `lib.rs`:
```rust
pub mod kernel;
pub use kernel::{lww_newer, MultiDeviceMap, ReachabilityFallback, ReachabilityRecord};
```

- [ ] **Step 6: Full crate gate**

Run: `cargo fmt --all -- --check` → PASS
Run: `cargo clippy -p harmony-reachability --all-targets -- -D warnings` → PASS (fix any lints; do not `#[allow]` without cause)
Run: `cargo test -p harmony-reachability` → PASS (all tests)

- [ ] **Step 7: Commit**

```bash
git add crates/harmony-reachability/src/kernel.rs crates/harmony-reachability/src/lib.rs
git commit -m "harmony-reachability: multi-device LWW kernel + fallback trait (ZEB-744)"
```

---

## Self-Review notes (planner)

- **Spec coverage:** record byte-preserving move (Task 2 + golden vector) ✓; kernel = `ReachabilityRecord`/`lww_newer`/`MultiDeviceMap`/`ReachabilityFallback` (Task 3) ✓; no iroh/pkarr/tokio/crypto deps (Task 1 manifest) ✓; crypto-free / signing stays client ✓.
- **Deferred to PR 2 (not this plan):** the workspace-root `Cargo.toml` client-side lockstep pin bump, deleting `reachability_record.rs`, slimming `reachability_resolver.rs` onto the kernel, the ~consumer import repointing, and keeping the client wire-fixture + resolver test suites green.
- **Type consistency:** `lww_newer(prev_clock, prev_rec, next_clock, next_rec)` signature is used identically in Task 3 Step 1 test and Step 3 impl. `MultiDeviceMap::range_owner`/`find_by_node_id` names match between test and impl.
- **Open checkpoint for the implementer:** `ciborium` feature set — if `ciborium::into_writer` fails to resolve a `Write` impl for `Vec<u8>` at first compile, add `features = ["std"]` to the `ciborium` manifest line (workspace pins `default-features = false`). `serde`/`std` and `thiserror` are already pinned.
