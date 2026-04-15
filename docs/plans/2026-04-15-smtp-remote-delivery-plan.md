# SMTP Remote Delivery (ZEB-113 PR A) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** When the SMTP handler lands an RFC 5322 message whose recipient is NOT homed on this gateway, seal the translated `HarmonyMessage` bytes to the recipient's public identity and publish the sealed envelope over Zenoh, so the recipient's gateway (or client) can receive it on the `harmony/msg/v1/unicast/{recipient_hash_hex}` keyspace.

**Architecture:** Extend `process_async_actions::DeliverToHarmony` in `crates/harmony-mail/src/server.rs`: the `Ok(None)` branch of `imap_store.get_user_by_address` — which today just logs "no local user" and drops — now calls a new sans-I/O helper `remote_delivery::seal_for_recipient(gateway_priv, recipient_id, msg_bytes, rng)` that returns the sealed envelope bytes, then fire-and-forget-publishes via a new `ZenohPublisher::publish_sealed_unicast(hash_hex, bytes)` method. The recipient's `Identity` is supplied by a new `RecipientResolver` trait threaded through server startup alongside the existing `mailbox_publisher`. No `OfflineResolver`/store-and-forward in this PR — `None` from the resolver means "not announced; warn + skip" (per-recipient MX behavior). Keyspace `harmony/msg/v1/unicast/` is deliberately separate from `harmony/mail/v1/` — the latter is the gateway-authoritative mailbox tree, the former is a sender-sealed unicast drop-off.

**Tech Stack:** Rust 1.75, Zenoh 1.x, harmony-zenoh (`HarmonyEnvelope::seal`), harmony-identity (X25519 + Ed25519 classical `Identity`), harmony-discovery (`AnnounceRecord`), tokio (fire-and-forget spawn), tracing.

---

## File structure

**New files:**
- `crates/harmony-zenoh/src/namespace.rs` — add a `msg` module (in-file; namespace.rs is the canonical home for cross-crate keyspace constants)
- `crates/harmony-mail/src/remote_delivery.rs` — sans-I/O helpers: `seal_for_recipient`, `identity_from_announce_record`, `RecipientResolver` trait, `RemoteDeliveryError`
- `crates/harmony-mail/tests/smtp_remote_delivery_integration.rs` — NEW integration test file (keeps server.rs tests module from growing past the existing ~1100 lines of tests). Follows the pattern at `server.rs:3053`.

**Modified files:**
- `crates/harmony-mail/src/lib.rs` — `pub mod remote_delivery;`
- `crates/harmony-mail/src/mailbox_manager.rs:393-445` — add `publish_sealed_unicast` method to `ZenohPublisher` (sibling to `publish_raw_mail`; same permits/timeout pattern)
- `crates/harmony-mail/src/server.rs` —
  - Top-of-file: import `RecipientResolver` and `remote_delivery`
  - `run_with_config` startup at `server.rs:197+` — accept + stash `gateway_identity` and `recipient_resolver` optionals
  - `handle_connection` / `handle_connection_generic` signatures — thread the two new optionals through
  - `process_async_actions` signature — add `gateway_identity: &Option<Arc<PrivateIdentity>>` and `recipient_resolver: &Option<Arc<dyn RecipientResolver>>`
  - `DeliverToHarmony` `Ok(None)` branch at `server.rs:1251-1256` — replace the silent log with remote-delivery attempt

**Why a new file for `remote_delivery.rs`:** `server.rs` is 4,153 lines. The sans-I/O seal/resolve logic is meaningfully distinct from the SMTP state machine and needs its own unit-test module. Splitting it keeps the I/O surface (server.rs) free of cryptographic plumbing details.

---

## Design decisions (locked in before task 1)

1. **Keyspace:** `harmony/msg/v1/unicast/{recipient_hash_hex}` — 32-char lowercase hex of the 16-byte address hash. The `/v1/` segment mirrors `harmony/mail/v1/` for symmetric versioning. Declared in `crates/harmony-zenoh/src/namespace.rs` as `msg::unicast_key(&hash_hex)`.

2. **`MessageType` for seal():** Use the existing `MessageType::Put` variant at `crates/harmony-zenoh/src/envelope.rs:36`. The envelope's type field is 4 bits and currently only encodes `Put` vs `Del`; introducing a new `Mail` variant would be premature — mail is semantically a Put of data and doesn't need a dedicated wire code.

3. **`sequence: u32` source:** Fresh `rng.next_u32()` per sealed envelope. Rationale: (a) `seal()` already takes `&mut impl CryptoRngCore` for the nonce, so reusing it is free; (b) `open()` does not enforce monotonicity, so a counter would be dead weight for this PR; (c) per-recipient counters would require persistent state that doesn't exist yet. Random sequence preserves the AAD entropy contract without adding plumbing.

4. **Recipient `Identity` construction:** Free function `identity_from_announce_record(rec: &AnnounceRecord) -> Result<Identity, RemoteDeliveryError>` in `remote_delivery.rs`. Converts the record's `public_key: Vec<u8>` (32B Ed25519) + `encryption_key: Vec<u8>` (32B X25519) via `Identity::from_public_keys(&x25519_pub, &ed25519_pub)`. Lives in `harmony-mail` rather than `harmony-discovery` because it couples discovery to identity's wire format, and `harmony-discovery` deliberately stays type-parameter-free on that boundary.

5. **Gateway `PrivateIdentity` plumbing:** Added as `gateway_identity: Option<Arc<PrivateIdentity>>` alongside the existing `mailbox_publisher: Option<Arc<ZenohPublisher>>` parameter. `Option` because the harmony-mail crate has no-identity test fixtures (`SmtpServer::run(..)` without a gateway identity is still a valid configuration for the IMAP-only code paths and the existing local-delivery tests). Signature additions (exactly two new params, at the END of the existing parameter list to minimize churn in existing call sites):

```rust
async fn process_async_actions<W: AsyncWrite + Unpin>(
    // ... existing 11 parameters unchanged ...
    mailbox_publisher: &Option<Arc<crate::mailbox_manager::ZenohPublisher>>,
    // NEW:
    gateway_identity: &Option<Arc<harmony_identity::PrivateIdentity>>,
    recipient_resolver: &Option<Arc<dyn crate::remote_delivery::RecipientResolver>>,
) -> Result<bool, Box<dyn std::error::Error + Send + Sync>>
```

Same two additions appear on `handle_connection`, `handle_connection_generic`, and `run_with_config`.

---

## Prerequisite verification steps (run BEFORE Task 1)

These were checked during plan authoring but should be re-verified in the implementer's environment:

- [ ] **Verify build is green on branch base:**

Run: `cd $REPO_ROOT && cargo build -p harmony-mail -p harmony-zenoh`
Expected: clean build, warnings OK.

- [ ] **Verify `Identity::from_public_keys` signature hasn't drifted:**

Run: `grep -n "pub fn from_public_keys" crates/harmony-identity/src/identity.rs`
Expected: `pub fn from_public_keys(x25519_pub: &[u8; 32], ed25519_pub: &[u8; 32]) -> Result<Self, IdentityError>` at/near line 53.

- [ ] **Verify `AnnounceRecord` has `public_key` and `encryption_key` as `Vec<u8>`:**

Run: `grep -n "pub.*public_key\|pub.*encryption_key" crates/harmony-discovery/src/record.rs`
Expected: both `Vec<u8>` on the `AnnounceRecord` struct.

---

## Task 1: `msg` namespace constants + `unicast_key` builder

**Files:**
- Modify: `crates/harmony-zenoh/src/namespace.rs` (append new module at end of file, before the final `#[cfg(test)]` block if one exists; otherwise at end)

- [ ] **Step 1: Write the failing test**

Append to `crates/harmony-zenoh/src/namespace.rs` inside the existing `#[cfg(test)] mod tests { ... }` block (or add one at end of file if none exists):

```rust
#[cfg(test)]
mod msg_namespace_tests {
    use super::msg;

    #[test]
    fn unicast_key_format_is_v1_prefixed_hex() {
        let hash_hex = "00112233445566778899aabbccddeeff";
        assert_eq!(
            msg::unicast_key(hash_hex),
            "harmony/msg/v1/unicast/00112233445566778899aabbccddeeff"
        );
    }

    #[test]
    fn unicast_sub_pattern_wildcards_the_hash() {
        assert_eq!(msg::UNICAST_SUB, "harmony/msg/v1/unicast/*");
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p harmony-zenoh --lib msg_namespace_tests 2>&1 | tail -20`
Expected: FAIL with "unresolved import `super::msg`" (or similar — module doesn't exist yet).

- [ ] **Step 3: Write minimal implementation**

Append this module to `crates/harmony-zenoh/src/namespace.rs` at end of file (after all existing tier modules, before any `#[cfg(test)]` block):

```rust
/// Tier 3 / application namespace for unicast sealed messages.
///
/// Deliberately separate from `harmony/mail/v1/` — that keyspace carries
/// gateway-authoritative mailbox tree publications. `harmony/msg/v1/unicast/`
/// carries sender-sealed `HarmonyEnvelope` payloads addressed to a specific
/// recipient's address hash. A recipient's gateway (or client) subscribes
/// on this prefix and calls `HarmonyEnvelope::open` with its
/// `PrivateIdentity` to recover the plaintext.
pub mod msg {
    use alloc::string::String;

    /// Base prefix: `harmony/msg/v1`
    pub const PREFIX: &str = "harmony/msg/v1";

    /// Unicast prefix: `harmony/msg/v1/unicast`
    pub const UNICAST_PREFIX: &str = "harmony/msg/v1/unicast";

    /// Subscribe pattern for all unicast messages to a single recipient's
    /// inbox (caller appends the hash segment themselves if single-recipient
    /// subscribe is preferred) or the wildcard for gateway observers.
    pub const UNICAST_SUB: &str = "harmony/msg/v1/unicast/*";

    /// Publish key: `harmony/msg/v1/unicast/{recipient_hash_hex}`.
    ///
    /// `recipient_hash_hex` is the lowercase 32-char hex encoding of the
    /// 16-byte truncated SHA-256 address hash (see
    /// `harmony_identity::ADDRESS_HASH_LENGTH`).
    pub fn unicast_key(recipient_hash_hex: &str) -> String {
        alloc::format!("{UNICAST_PREFIX}/{recipient_hash_hex}")
    }
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p harmony-zenoh --lib msg_namespace_tests 2>&1 | tail -10`
Expected: `test result: ok. 2 passed`.

- [ ] **Step 5: Commit**

```bash
cd $REPO_ROOT
git add crates/harmony-zenoh/src/namespace.rs
git commit -m "feat(zenoh): add msg/v1/unicast namespace for sealed sender-to-recipient messages (ZEB-113)"
```

---

## Task 2: `ZenohPublisher::publish_sealed_unicast` method

**Files:**
- Modify: `crates/harmony-mail/src/mailbox_manager.rs` — add new method after `publish_raw_mail` (around line 445)

- [ ] **Step 1: Write the failing test**

Append this test to the existing `#[cfg(test)] mod tests { ... }` block in `crates/harmony-mail/src/mailbox_manager.rs`:

```rust
#[tokio::test]
async fn publish_sealed_unicast_records_notification_on_test_publisher() {
    let (publisher, handles) = ZenohPublisher::inert_for_test();
    let recipient_hash_hex = "aabbccddeeff00112233445566778899";
    let envelope = alloc::sync::Arc::new(vec![0xDE, 0xAD, 0xBE, 0xEF]);

    publisher.publish_sealed_unicast(
        recipient_hash_hex.to_string(),
        alloc::sync::Arc::clone(&envelope),
    );

    let note = handles
        .recv_sealed_unicast(std::time::Duration::from_secs(2))
        .await
        .expect("ZenohPublisher did not record a sealed-unicast notification");
    assert_eq!(note.key, "harmony/msg/v1/unicast/aabbccddeeff00112233445566778899");
    assert_eq!(note.payload.as_slice(), envelope.as_slice());
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p harmony-mail --lib publish_sealed_unicast_records_notification_on_test_publisher 2>&1 | tail -20`
Expected: FAIL — `publish_sealed_unicast` and `recv_sealed_unicast` don't exist.

- [ ] **Step 3: Extend `inert_for_test` handles to capture sealed-unicast notifications**

Locate the `TestPublisherHandles` struct (grep `inert_for_test` in `mailbox_manager.rs` — it should be around line 96-100 per the plan's survey) and its impl. It currently exposes a channel for raw-mail notifications. Mirror that channel for sealed unicast.

If the struct currently looks like:

```rust
pub struct TestPublisherHandles {
    raw_rx: tokio::sync::mpsc::Receiver<RawMailNotification>,
}
```

(confirm exact field names with `grep -n "pub struct TestPublisherHandles\|raw_rx\|recv_raw_mail" crates/harmony-mail/src/mailbox_manager.rs`)

Add a sibling field + method. Add the struct definition `SealedUnicastNotification`:

```rust
/// Notification recorded by the in-memory test publisher when
/// `publish_sealed_unicast` is called. Mirrors `RawMailNotification`.
#[derive(Debug, Clone)]
pub struct SealedUnicastNotification {
    pub key: String,
    pub payload: Arc<Vec<u8>>,
}
```

Extend `TestPublisherHandles`:

```rust
pub struct TestPublisherHandles {
    raw_rx: tokio::sync::mpsc::Receiver<RawMailNotification>,
    // NEW:
    sealed_rx: tokio::sync::mpsc::Receiver<SealedUnicastNotification>,
}

impl TestPublisherHandles {
    // ... existing recv_raw_mail method ...

    pub async fn recv_sealed_unicast(
        &mut self,
        timeout: std::time::Duration,
    ) -> Option<SealedUnicastNotification> {
        tokio::time::timeout(timeout, self.sealed_rx.recv())
            .await
            .ok()
            .flatten()
    }
}
```

- [ ] **Step 4: Extend `ZenohPublisher` sink enum + inert_for_test constructor**

Locate `enum RawSink` (grep `enum RawSink` in the same file). It likely has `Session { ... }` and `Test { tx: mpsc::Sender<RawMailNotification> }` variants. The sealed-unicast path reuses the same session (same Zenoh handle) but in the test variant needs its own tx.

Modify:

```rust
enum RawSink {
    Session {
        session: zenoh::Session,
        cancel: tokio_util::sync::CancellationToken,
        permits: Arc<tokio::sync::Semaphore>,
    },
    Test {
        raw_tx: tokio::sync::mpsc::Sender<RawMailNotification>,
        // NEW:
        sealed_tx: tokio::sync::mpsc::Sender<SealedUnicastNotification>,
    },
}
```

In the existing `impl ZenohPublisher { pub fn inert_for_test() -> (Arc<Self>, TestPublisherHandles) { ... } }`, add a second channel:

```rust
pub fn inert_for_test() -> (Arc<Self>, TestPublisherHandles) {
    let (raw_tx, raw_rx) = tokio::sync::mpsc::channel(16);
    let (sealed_tx, sealed_rx) = tokio::sync::mpsc::channel(16);
    let publisher = Arc::new(Self {
        raw_sink: RawSink::Test { raw_tx, sealed_tx },
        // ... whatever other fields ZenohPublisher has, populated same as before ...
    });
    let handles = TestPublisherHandles { raw_rx, sealed_rx };
    (publisher, handles)
}
```

(Rename the pre-existing `tx` field to `raw_tx` where the Test variant is constructed inside `inert_for_test`, and update the `publish_raw_mail` match arm to destructure `raw_tx` — any existing references break with clear errors, fix them mechanically.)

- [ ] **Step 5: Implement `publish_sealed_unicast`**

Add this method to `impl ZenohPublisher`, immediately after `publish_raw_mail`:

```rust
/// Publish a sealed envelope on the per-recipient unicast keyspace.
///
/// Fire-and-forget, same backpressure/cancellation semantics as
/// `publish_raw_mail`. Caller is responsible for having sealed the
/// payload via `HarmonyEnvelope::seal` — this method is transport
/// only, no cryptographic work.
///
/// Short-circuits when the cancel token has already fired — prevents
/// post-shutdown spawns from keeping the runtime alive.
pub fn publish_sealed_unicast(
    &self,
    recipient_hash_hex: String,
    envelope: Arc<Vec<u8>>,
) {
    match &self.raw_sink {
        RawSink::Session {
            session,
            cancel,
            permits,
        } => {
            if cancel.is_cancelled() {
                return;
            }
            let permit = match Arc::clone(permits).try_acquire_owned() {
                Ok(p) => p,
                Err(tokio::sync::TryAcquireError::NoPermits) => {
                    tracing::warn!(
                        recipient = %recipient_hash_hex,
                        "sealed-unicast publish dropped: in-flight limit reached",
                    );
                    return;
                }
                Err(tokio::sync::TryAcquireError::Closed) => return,
            };
            let session = session.clone();
            let cancel = cancel.clone();
            tokio::spawn(async move {
                let _permit = permit;
                let key = harmony_zenoh::namespace::msg::unicast_key(&recipient_hash_hex);
                let put_fut = session.put(&key, envelope.as_slice());
                let result = tokio::select! {
                    _ = cancel.cancelled() => {
                        tracing::debug!(recipient = %recipient_hash_hex, "sealed-unicast publish cancelled");
                        return;
                    }
                    r = tokio::time::timeout(RAW_PUBLISH_TIMEOUT, put_fut) => r,
                };
                match result {
                    Ok(Ok(())) => {}
                    Ok(Err(e)) => tracing::warn!(
                        recipient = %recipient_hash_hex,
                        error = %e,
                        "sealed-unicast publish failed",
                    ),
                    Err(_) => tracing::warn!(
                        recipient = %recipient_hash_hex,
                        timeout_ms = RAW_PUBLISH_TIMEOUT.as_millis() as u64,
                        "sealed-unicast publish timed out",
                    ),
                }
            });
        }
        RawSink::Test { sealed_tx, .. } => {
            let key = harmony_zenoh::namespace::msg::unicast_key(&recipient_hash_hex);
            let _ = sealed_tx.try_send(SealedUnicastNotification { key, payload: envelope });
        }
    }
}
```

Add `use harmony_zenoh::namespace::msg;` near the top-of-file use-block if not already present (or use the fully-qualified path inside the method body as shown).

- [ ] **Step 6: Run test to verify it passes**

Run: `cargo test -p harmony-mail --lib publish_sealed_unicast_records_notification_on_test_publisher 2>&1 | tail -10`
Expected: `test result: ok. 1 passed`.

Also run the full mailbox_manager test module to confirm no regressions: `cargo test -p harmony-mail --lib mailbox_manager 2>&1 | tail -10`.
Expected: all pre-existing mailbox_manager tests still pass.

- [ ] **Step 7: Commit**

```bash
git add crates/harmony-mail/src/mailbox_manager.rs
git commit -m "feat(mail): ZenohPublisher::publish_sealed_unicast for ZEB-113 remote delivery"
```

---

## Task 3: `remote_delivery.rs` — sans-I/O helpers

**Files:**
- Create: `crates/harmony-mail/src/remote_delivery.rs`
- Modify: `crates/harmony-mail/src/lib.rs` (add `pub mod remote_delivery;`)

- [ ] **Step 1: Write the failing unit tests**

Create `crates/harmony-mail/src/remote_delivery.rs` with ONLY this test module (no other code yet):

```rust
#![cfg(test)]
//! STAGE 1 — tests written before any implementation. Delete this cfg-gate
//! on the whole file once step 3 moves the real code in.

use super::*; // will fail until impl exists

#[test]
fn seal_for_recipient_round_trips_via_envelope_open() {
    use harmony_identity::PrivateIdentity;
    use harmony_zenoh::envelope::HarmonyEnvelope;
    use rand_core::OsRng;

    let mut rng = OsRng;
    let gateway_priv = PrivateIdentity::generate(&mut rng);
    let recipient_priv = PrivateIdentity::generate(&mut rng);

    let plaintext = b"harmony-message-bytes-go-here";
    let sealed = seal_for_recipient(
        &mut rng,
        &gateway_priv,
        recipient_priv.public_identity(),
        plaintext,
    )
    .expect("seal should succeed");

    let opened = HarmonyEnvelope::open(
        &recipient_priv,
        gateway_priv.public_identity(),
        &sealed,
    )
    .expect("open should succeed");
    assert_eq!(opened.plaintext, plaintext);
    assert_eq!(opened.sender_address, gateway_priv.public_identity().address_hash);
}

#[test]
fn identity_from_announce_record_extracts_classical_keys() {
    use harmony_identity::PrivateIdentity;
    use rand_core::OsRng;

    let mut rng = OsRng;
    let priv_id = PrivateIdentity::generate(&mut rng);
    let pub_id = priv_id.public_identity();

    // Fake AnnounceRecord with just the fields we read. We build from raw
    // keys rather than constructing a full signed record to keep this test
    // a pure unit test with no harmony-discovery signing dependency.
    let pub_bytes = pub_id.to_public_bytes();  // [32B X25519][32B Ed25519]
    let rec = harmony_discovery::AnnounceRecord {
        identity_ref: harmony_identity::IdentityRef::default(),
        public_key: pub_bytes[32..].to_vec(),     // Ed25519 (verifying key)
        encryption_key: pub_bytes[..32].to_vec(), // X25519 (encryption key)
        routing_hints: alloc::vec![],
        published_at: 0,
        expires_at: 0,
        nonce: [0u8; 16],
        signature: alloc::vec![],
    };

    let derived = identity_from_announce_record(&rec).expect("conversion should succeed");
    assert_eq!(derived.address_hash, pub_id.address_hash);
}

#[test]
fn identity_from_announce_record_rejects_wrong_length() {
    let rec = harmony_discovery::AnnounceRecord {
        identity_ref: harmony_identity::IdentityRef::default(),
        public_key: vec![0u8; 10],           // wrong length
        encryption_key: vec![0u8; 32],
        routing_hints: alloc::vec![],
        published_at: 0,
        expires_at: 0,
        nonce: [0u8; 16],
        signature: alloc::vec![],
    };
    let err = identity_from_announce_record(&rec).expect_err("should reject");
    assert!(
        matches!(err, RemoteDeliveryError::InvalidAnnounceKey(_)),
        "unexpected error: {err:?}"
    );
}
```

(Note: `harmony_identity::IdentityRef::default()` — confirm default impl exists with `grep -n "impl Default for IdentityRef" crates/harmony-identity/src`. If not present, construct via whatever the idiomatic default is — e.g., `IdentityRef { type_code: 0, hash: [0; 16] }` or similar — and update the test.)

Also add `pub mod remote_delivery;` to `crates/harmony-mail/src/lib.rs` — find the block of `pub mod ...;` declarations (near the top) and add `pub mod remote_delivery;` in alphabetical order.

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test -p harmony-mail --lib remote_delivery 2>&1 | tail -20`
Expected: FAIL with "cannot find function `seal_for_recipient`" / "cannot find function `identity_from_announce_record`" / "cannot find type `RemoteDeliveryError`".

- [ ] **Step 3: Write the implementation**

Replace the entire contents of `crates/harmony-mail/src/remote_delivery.rs` with:

```rust
//! Sans-I/O helpers for ZEB-113 online-recipient remote mail delivery.
//!
//! The SMTP handler calls into this module from the `DeliverToHarmony`
//! action when a recipient is not homed on the local gateway. The helpers
//! here perform no I/O: they seal plaintext via `HarmonyEnvelope` and
//! construct an `Identity` from an `AnnounceRecord`. The caller is
//! responsible for (a) resolving the `AnnounceRecord` via the
//! `RecipientResolver` trait and (b) publishing the sealed bytes via
//! `ZenohPublisher::publish_sealed_unicast`.

use alloc::vec::Vec;
use core::fmt;
use harmony_discovery::AnnounceRecord;
use harmony_identity::{Identity, IdentityError, IdentityHash, PrivateIdentity};
use harmony_zenoh::envelope::{HarmonyEnvelope, MessageType};
use harmony_zenoh::ZenohError;
use rand_core::CryptoRngCore;

/// Errors surfaced by the remote delivery helpers.
#[derive(Debug)]
pub enum RemoteDeliveryError {
    /// An `AnnounceRecord` carried a public or encryption key whose byte
    /// length did not match the classical (X25519 + Ed25519) identity
    /// format.
    InvalidAnnounceKey(IdentityError),
    /// `HarmonyEnvelope::seal` failed (ECDH / AEAD / serialization).
    Seal(ZenohError),
}

impl fmt::Display for RemoteDeliveryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidAnnounceKey(e) => write!(f, "announce record keys invalid: {e:?}"),
            Self::Seal(e) => write!(f, "seal failed: {e:?}"),
        }
    }
}

impl std::error::Error for RemoteDeliveryError {}

/// Seal `plaintext` addressed from `sender` (gateway's own identity) to
/// `recipient` (the remote user's `Identity`). Returns the fully-framed
/// envelope bytes ready for `session.put(...)`.
///
/// The sequence number is fresh-random per call — `HarmonyEnvelope::open`
/// does not enforce monotonicity on the receiver side, and no persistent
/// per-recipient counter exists at the gateway today. Random sequence
/// preserves the AAD entropy contract without new plumbing.
pub fn seal_for_recipient(
    rng: &mut impl CryptoRngCore,
    sender: &PrivateIdentity,
    recipient: &Identity,
    plaintext: &[u8],
) -> Result<Vec<u8>, RemoteDeliveryError> {
    let sequence = rng.next_u32();
    HarmonyEnvelope::seal(
        rng,
        MessageType::Put,
        sender,
        recipient,
        sequence,
        plaintext,
    )
    .map_err(RemoteDeliveryError::Seal)
}

/// Build a classical `Identity` from the `public_key` (Ed25519) and
/// `encryption_key` (X25519) fields of an `AnnounceRecord`.
///
/// `AnnounceRecord.public_key` stores the Ed25519 verifying key as a
/// `Vec<u8>`; `AnnounceRecord.encryption_key` stores the X25519 public
/// key as a `Vec<u8>`. Both must be exactly 32 bytes for the classical
/// identity path.
pub fn identity_from_announce_record(
    rec: &AnnounceRecord,
) -> Result<Identity, RemoteDeliveryError> {
    let x25519_pub: &[u8; 32] = rec
        .encryption_key
        .as_slice()
        .try_into()
        .map_err(|_| RemoteDeliveryError::InvalidAnnounceKey(IdentityError::InvalidPublicKeyLength(rec.encryption_key.len())))?;
    let ed25519_pub: &[u8; 32] = rec
        .public_key
        .as_slice()
        .try_into()
        .map_err(|_| RemoteDeliveryError::InvalidAnnounceKey(IdentityError::InvalidPublicKeyLength(rec.public_key.len())))?;
    Identity::from_public_keys(x25519_pub, ed25519_pub)
        .map_err(RemoteDeliveryError::InvalidAnnounceKey)
}

/// Runtime contract for looking up a recipient's `Identity` by their
/// 16-byte address hash. Plumbed through the SMTP handler and consulted
/// once per non-local recipient in `DeliverToHarmony`.
///
/// `None` means "no announce record cached / not announced" — the
/// caller treats this as a warn-and-skip (per-recipient MX behavior);
/// the SMTP transaction still succeeds overall if other recipients
/// succeeded. A future `OfflineResolver` integration (ZEB-113 PR B) will
/// be wired behind this same trait, so the SMTP-handler side does not
/// need to change when store-and-forward lands.
pub trait RecipientResolver: Send + Sync {
    fn resolve(&self, address_hash: &IdentityHash) -> Option<Identity>;
}

#[cfg(test)]
mod tests {
    // [Replace the STAGE 1 test block from step 1 with the real tests here.]
    // Copy the three tests from step 1 verbatim into this module.
}
```

Then: delete the `#![cfg(test)]` outer gate at the top of the file from step 1, and move the three tests inside the new `#[cfg(test)] mod tests { ... }` block. The final file should have the module-level code NOT gated, and the test module cfg-gated.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test -p harmony-mail --lib remote_delivery 2>&1 | tail -15`
Expected: `test result: ok. 3 passed`.

- [ ] **Step 5: Commit**

```bash
git add crates/harmony-mail/src/remote_delivery.rs crates/harmony-mail/src/lib.rs
git commit -m "feat(mail): remote_delivery helpers + RecipientResolver trait (ZEB-113)"
```

---

## Task 4: Thread `gateway_identity` + `recipient_resolver` through SMTP server

**Files:**
- Modify: `crates/harmony-mail/src/server.rs` — `process_async_actions`, `handle_connection`, `handle_connection_generic`, and `run_with_config` (or whatever the top-level server entry function is named — confirm with `grep -n "pub.*async fn run\|pub fn run" crates/harmony-mail/src/server.rs | head -5` before editing).

- [ ] **Step 1: Write a compile-only checkpoint test**

This task is a pure plumbing refactor; no behavior change. The success signal is that the whole crate still compiles after threading two new `Option<Arc<…>>` parameters end-to-end, and all existing tests still pass (they pass `None` for both new parameters).

Add the following smoke test to the existing `tests` module at the bottom of `server.rs` (around `server.rs:2810` — confirm with `grep -n "mod tests\|fn smtp_delivers_to_local_imap" crates/harmony-mail/src/server.rs`):

```rust
#[tokio::test]
async fn run_with_config_accepts_none_gateway_identity_and_none_resolver() {
    // Compile-time checkpoint: the two new optional parameters must accept
    // `None` without forcing callers to construct real cryptographic state.
    // Production callers that want remote delivery pass Some(...) for both.
    let gateway_identity: Option<std::sync::Arc<harmony_identity::PrivateIdentity>> = None;
    let recipient_resolver: Option<
        std::sync::Arc<dyn crate::remote_delivery::RecipientResolver>,
    > = None;

    // Both variables must have inferrable types that match the signatures
    // threaded through run_with_config / handle_connection / process_async_actions.
    let _ = (gateway_identity, recipient_resolver);
}
```

- [ ] **Step 2: Run the test to verify it compiles and passes trivially**

Run: `cargo build -p harmony-mail 2>&1 | tail -5` — should be clean on the unchanged base branch.
Run: `cargo test -p harmony-mail --lib run_with_config_accepts_none_gateway_identity_and_none_resolver 2>&1 | tail -10`
Expected: PASS (1 test).

- [ ] **Step 3: Add the two new parameters to `process_async_actions`**

Modify the signature at `server.rs:1007` (confirm exact line with `grep -n "async fn process_async_actions" crates/harmony-mail/src/server.rs`):

```rust
#[allow(clippy::too_many_arguments)]
async fn process_async_actions<W: AsyncWrite + Unpin>(
    actions: &[SmtpAction],
    session: &mut SmtpSession,
    writer: &mut W,
    imap_store: &Arc<ImapStore>,
    authenticator: &Option<Arc<mail_auth::MessageAuthenticator>>,
    local_domain: &str,
    content_store_path: &Path,
    spf_result: &mut crate::spam::SpfResult,
    reject_threshold: i32,
    mailbox_manager: &Option<Arc<std::sync::Mutex<crate::mailbox_manager::MailboxManager>>>,
    mailbox_publisher: &Option<Arc<crate::mailbox_manager::ZenohPublisher>>,
    // NEW:
    gateway_identity: &Option<Arc<harmony_identity::PrivateIdentity>>,
    recipient_resolver: &Option<Arc<dyn crate::remote_delivery::RecipientResolver>>,
) -> Result<bool, Box<dyn std::error::Error + Send + Sync>> {
```

The function body does NOT use the new parameters yet — Task 5 wires them in. Underscore-prefix them inside the function to suppress the `unused` warning: at the top of the function body add:

```rust
let _ = (gateway_identity, recipient_resolver);
```

Remove this silencer in Task 5 when the real usage lands.

- [ ] **Step 4: Update both `handle_connection` and `handle_connection_generic` signatures + call sites**

For each of these two functions (grep `async fn handle_connection\|async fn handle_connection_generic` to find both):

(a) Add the same two new parameters to the end of the signature:

```rust
    mailbox_publisher: Option<Arc<crate::mailbox_manager::ZenohPublisher>>,
    // NEW:
    gateway_identity: Option<Arc<harmony_identity::PrivateIdentity>>,
    recipient_resolver: Option<Arc<dyn crate::remote_delivery::RecipientResolver>>,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>>
```

(b) Update every `process_async_actions(...)` call inside each function to pass `&gateway_identity, &recipient_resolver` at the end. There are TWO such call sites per survey (`server.rs:741` and `server.rs:914`).

(c) When `handle_connection` calls `handle_connection_generic` (if it does — check with `grep -n "handle_connection_generic(" crates/harmony-mail/src/server.rs`), forward both parameters.

- [ ] **Step 5: Update the top-level server entry point and the TLS port dispatchers**

(a) Top-level: Find the public entrypoint that starts the server. Likely `run_with_config` or `SmtpServer::run`. Confirm:

Run: `grep -n "pub.*fn run\|pub async fn run\|pub struct SmtpServer" crates/harmony-mail/src/server.rs | head -10`

Whatever it is, add the two new parameters to its signature at the end:

```rust
    gateway_identity: Option<Arc<harmony_identity::PrivateIdentity>>,
    recipient_resolver: Option<Arc<dyn crate::remote_delivery::RecipientResolver>>,
```

(b) Inside that function, the port dispatcher loops spawn one task per accepted connection (grep `tokio::spawn` in server.rs to find them — per survey there are loops for ports 465, 587, and the default). For each spawn, clone the two new Arcs and pass them to `handle_connection` / `handle_connection_generic`. Pattern matches the existing `let imap_store_465 = Arc::clone(&imap_store);` lines at `server.rs:394`:

```rust
let gateway_identity_465 = gateway_identity.clone();
let recipient_resolver_465 = recipient_resolver.clone();
// ... inside the spawn:
let gateway_identity = gateway_identity_465.clone();
let recipient_resolver = recipient_resolver_465.clone();
// ... passed into handle_connection_generic(..., gateway_identity, recipient_resolver)
```

Repeat for each numbered-port dispatcher.

- [ ] **Step 6: Update ALL existing test and external call sites to pass `None`**

Run: `grep -n "run_with_config\|\.run(" crates/harmony-mail/src/server.rs | head -20` and find every in-crate call. Run: `grep -rn "harmony_mail::server\|harmony_mail::.*::run\b" crates/ --include='*.rs'` to find external call sites (harmony-node likely calls this).

For each call site, add `None, None` at the end of the argument list.

Same for every test-internal call to `process_async_actions`, `handle_connection`, or `handle_connection_generic` — append `&None, &None`.

- [ ] **Step 7: Run the full test suite to verify no regression**

Run: `cargo build -p harmony-mail 2>&1 | tail -10`
Expected: clean build (warnings OK).

Run: `cargo test -p harmony-mail 2>&1 | tail -20`
Expected: same green count as before this task (count it beforehand with `cargo test -p harmony-mail 2>&1 | grep "test result:" | tail -5` on the base branch).

Run: `cargo build --workspace 2>&1 | tail -10`
Expected: clean build. If harmony-node failed to compile because its `SmtpServer::run` call lacks the two new arguments, fix the harmony-node call site by passing `None, None` (Task 4's scope includes upstream consumers — they opt into remote delivery in a future wiring PR).

- [ ] **Step 8: Commit**

```bash
git add crates/harmony-mail/src/server.rs
# Plus any harmony-node file you had to update:
git status
git add crates/harmony-node/src/...  # only if grep turned up changes
git commit -m "refactor(mail): thread gateway_identity + recipient_resolver through SMTP pipeline (ZEB-113)"
```

---

## Task 5: Wire remote delivery into `DeliverToHarmony::Ok(None)` branch

**Files:**
- Modify: `crates/harmony-mail/src/server.rs` — lines 1224-1265 (the recipient iteration loop inside `DeliverToHarmony`)

- [ ] **Step 1: Write the failing unit test — negative path (no announce record)**

Append this test to the tests module at bottom of `server.rs`:

```rust
#[tokio::test]
async fn delivers_remote_warn_and_skip_when_resolver_returns_none() {
    use crate::remote_delivery::RecipientResolver;
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct NullResolver(Arc<AtomicUsize>);
    impl RecipientResolver for NullResolver {
        fn resolve(&self, _addr: &harmony_identity::IdentityHash) -> Option<harmony_identity::Identity> {
            self.0.fetch_add(1, Ordering::SeqCst);
            None
        }
    }

    let smtp_test_dir = tempfile::tempdir().unwrap();
    let store = Arc::new(
        crate::imap_store::ImapStore::open(&smtp_test_dir.path().join("smtp-test.db")).unwrap(),
    );

    let (publisher, mut handles) = crate::mailbox_manager::ZenohPublisher::inert_for_test();
    let mut rng = rand::rngs::OsRng;
    let gateway_id = Arc::new(harmony_identity::PrivateIdentity::generate(&mut rng));
    let calls = Arc::new(AtomicUsize::new(0));
    let resolver: Arc<dyn RecipientResolver> = Arc::new(NullResolver(Arc::clone(&calls)));

    // Synthesize a DeliverToHarmony action with a recipient hash that is
    // NOT in the imap_store (so the local branch skips) and whose resolver
    // returns None (so the remote branch warns + skips). Use a fake
    // RFC 5322 blob that translate_inbound can parse successfully.
    let rfc822 = b"From: alice@local\r\nTo: bob@remote\r\nSubject: hi\r\n\r\nhello\r\n";
    let recipient_hash = [0xAA; crate::message::ADDRESS_HASH_LEN];

    let actions = vec![crate::SmtpAction::DeliverToHarmony {
        recipients: vec![recipient_hash],
        data: rfc822.to_vec(),
    }];

    // Minimal harness: build a throwaway session + writer + call process_async_actions.
    let mut session = crate::SmtpSession::new_for_test(); // add this helper if it doesn't exist
    let mut writer = Vec::<u8>::new();
    let mut spf_result = crate::spam::SpfResult::None;

    // Disable the spam rejector: zero out the threshold so the message passes.
    let reject_threshold = 1000;

    crate::process_async_actions(
        &actions,
        &mut session,
        &mut writer,
        &store,
        &None,
        "local",
        smtp_test_dir.path(),
        &mut spf_result,
        reject_threshold,
        &None,
        &Some(Arc::clone(&publisher)),
        &Some(Arc::clone(&gateway_id)),
        &Some(resolver),
    )
    .await
    .expect("process_async_actions should succeed even with all remote skips");

    // Resolver was consulted exactly once (for bob).
    assert_eq!(calls.load(Ordering::SeqCst), 1);

    // Publisher recorded NO sealed-unicast publish.
    let note = handles
        .recv_sealed_unicast(std::time::Duration::from_millis(100))
        .await;
    assert!(
        note.is_none(),
        "no publish should occur when resolver returns None; got {note:?}"
    );
}
```

Note: `SmtpSession::new_for_test()` may not exist. If it doesn't (check with `grep -n "fn new_for_test\|pub fn new" crates/harmony-mail/src/session.rs`), use whatever constructor the surrounding tests use — see `smtp_delivers_to_local_imap` at `server.rs:3053` for the pattern. Copy the session-bootstrap scaffold from there.

Also note: `process_async_actions` is currently private. If this test lives OUTSIDE the `crate` boundary, either (a) make it `pub(crate)` or (b) put the test inside the same module (the `#[cfg(test)] mod tests {` at the bottom of server.rs already is in-crate, so it can reach private items — this is fine).

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p harmony-mail --lib delivers_remote_warn_and_skip_when_resolver_returns_none 2>&1 | tail -25`
Expected: FAIL — the current `Ok(None)` branch silently drops without consulting the resolver, so `calls.load()` will be 0, tripping the `assert_eq!(..., 1)` assertion.

- [ ] **Step 3: Write the implementation**

At `server.rs:1224-1265`, replace the recipient loop with the expanded version. The key changes:

1. `Ok(None)` branch no longer just logs — it calls the resolver and, if it returns `Some(identity)`, seals and publishes.
2. All remote-delivery wiring is gated behind `gateway_identity.is_some() && recipient_resolver.is_some() && msg_bytes.is_some() && mailbox_publisher.is_some()` — any missing piece means "remote delivery disabled; silently skip" (backward-compat default).

Also remove the `let _ = (gateway_identity, recipient_resolver);` silencer added in Task 4 step 3.

```rust
// Drop the silencer added in Task 4:
// let _ = (gateway_identity, recipient_resolver);

// Inside DeliverToHarmony, just before the `for recipient_hash in recipients` loop:
let remote_ctx = match (
    gateway_identity.as_ref(),
    recipient_resolver.as_ref(),
    mailbox_publisher.as_ref(),
    msg_bytes.as_ref(),
) {
    (Some(gw), Some(res), Some(pub_), Some(bytes)) => {
        Some((gw.clone(), Arc::clone(res), Arc::clone(pub_), bytes.clone()))
    }
    _ => None,
};

for recipient_hash in recipients {
    match imap_store.get_user_by_address(recipient_hash) {
        Ok(Some(_user)) => {
            // ... existing local-delivery branch unchanged ...
        }
        Ok(None) => {
            // Non-local recipient: attempt remote delivery if the gateway
            // is configured for it. Per-recipient success/failure; overall
            // SMTP transaction succeeds regardless.
            if let Some((gw_id, resolver, publisher, bytes)) = remote_ctx.as_ref() {
                match resolver.resolve(recipient_hash) {
                    Some(recipient_identity) => {
                        let mut rng = rand::rngs::OsRng;
                        match crate::remote_delivery::seal_for_recipient(
                            &mut rng,
                            gw_id,
                            &recipient_identity,
                            bytes,
                        ) {
                            Ok(sealed) => {
                                let hash_hex = hex::encode(recipient_hash);
                                publisher.publish_sealed_unicast(
                                    hash_hex,
                                    Arc::new(sealed),
                                );
                                tracing::debug!(
                                    recipient = %hex::encode(recipient_hash),
                                    "remote recipient sealed + published",
                                );
                            }
                            Err(e) => tracing::warn!(
                                recipient = %hex::encode(recipient_hash),
                                error = %e,
                                "seal failed for remote recipient",
                            ),
                        }
                    }
                    None => tracing::warn!(
                        recipient = %hex::encode(recipient_hash),
                        "no announce record for remote recipient; skipping (offline store-and-forward in ZEB-113 PR B)",
                    ),
                }
            } else {
                tracing::debug!(
                    recipient = %hex::encode(recipient_hash),
                    "no local user and remote delivery not configured; dropping",
                );
            }
        }
        Err(e) => {
            tracing::warn!(
                recipient = %hex::encode(recipient_hash),
                error = %e,
                "user lookup failed"
            );
        }
    }
}
```

(The `hex` and `rand` crates — confirm they're already listed in `crates/harmony-mail/Cargo.toml`. If not, add `hex = "0.4"` and `rand = "0.8"`.)

Also at the module top of `server.rs`, ensure `use std::sync::Arc;` and the other imports are present — most already are.

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p harmony-mail --lib delivers_remote_warn_and_skip_when_resolver_returns_none 2>&1 | tail -15`
Expected: `test result: ok. 1 passed`.

Run the whole crate's test suite to catch regressions:

Run: `cargo test -p harmony-mail 2>&1 | tail -10`
Expected: all tests pass (same count as after Task 4, plus the new test).

- [ ] **Step 5: Commit**

```bash
git add crates/harmony-mail/src/server.rs
git commit -m "feat(mail): seal + publish to harmony/msg/v1/unicast for remote recipients (ZEB-113)"
```

---

## Task 6: Positive-path integration test — two in-process Zenoh sessions

**Files:**
- Create: `crates/harmony-mail/tests/smtp_remote_delivery_integration.rs`

- [ ] **Step 1: Write the failing integration test**

Create `crates/harmony-mail/tests/smtp_remote_delivery_integration.rs` with:

```rust
//! End-to-end integration test for ZEB-113 online-recipient remote delivery.
//!
//! Harness:
//!   - Gateway A: full SMTP server + ZenohPublisher writing to an in-process
//!     Zenoh session, plus a RecipientResolver that returns Bob's Identity
//!     from a stub cache.
//!   - Gateway B: independent Zenoh session on the same runtime,
//!     subscribed to `harmony/msg/v1/unicast/{bob_hash}`. Holds Bob's
//!     PrivateIdentity.
//!
//! Deliver an SMTP message addressed to bob@remote.example via Gateway A.
//! Gateway B's subscriber should receive the sealed envelope. We
//! `HarmonyEnvelope::open` with Bob's private identity + Alice (gateway A)'s
//! public identity and verify the recovered plaintext matches the
//! serialized HarmonyMessage that Gateway A translated.

use std::sync::Arc;
use std::time::Duration;

use harmony_identity::{Identity, IdentityHash, PrivateIdentity};
use harmony_mail::remote_delivery::RecipientResolver;
use harmony_zenoh::envelope::HarmonyEnvelope;
use harmony_zenoh::namespace::msg;
use rand_core::OsRng;

struct StubResolver {
    bob_hash: IdentityHash,
    bob_identity: Identity,
}
impl RecipientResolver for StubResolver {
    fn resolve(&self, address_hash: &IdentityHash) -> Option<Identity> {
        if address_hash == &self.bob_hash {
            Some(self.bob_identity.clone())
        } else {
            None
        }
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn smtp_remote_delivery_round_trips_through_zenoh_to_recipient() {
    // ── identities ─────────────────────────────────────────────
    let mut rng = OsRng;
    let gateway_a_priv = Arc::new(PrivateIdentity::generate(&mut rng));
    let bob_priv = PrivateIdentity::generate(&mut rng);
    let bob_pub = bob_priv.public_identity().clone();
    let bob_hash = bob_pub.address_hash;

    // ── Zenoh sessions ─────────────────────────────────────────
    // Two independent sessions on the local loopback peer-to-peer
    // network. Zenoh auto-discovers peers on the same machine.
    let session_a = zenoh::open(zenoh::Config::default()).await.unwrap();
    let session_b = zenoh::open(zenoh::Config::default()).await.unwrap();

    // Subscribe Bob's gateway before Gateway A publishes anything.
    let sub_key = msg::unicast_key(&hex::encode(bob_hash));
    let sub = session_b.declare_subscriber(&sub_key).await.unwrap();

    // Readiness probe — same pattern as the client walker integration test.
    // Ensures the subscriber declaration has propagated before we publish.
    tokio::time::sleep(Duration::from_millis(100)).await;

    // ── Publisher wrapping session_a ───────────────────────────
    // Build a real ZenohPublisher against session_a (NOT the test-inert one).
    // Cross-reference the ZenohPublisher constructor — grep
    // `impl ZenohPublisher` in mailbox_manager.rs for the production
    // constructor (likely `new(session, cancel_token)` or similar).
    // Example assuming the production ctor is `ZenohPublisher::new`:
    let cancel = tokio_util::sync::CancellationToken::new();
    let publisher = Arc::new(
        harmony_mail::mailbox_manager::ZenohPublisher::new(session_a.clone(), cancel.clone())
            .await
            .expect("build publisher"),
    );

    // ── IMAP store with NO user for bob ────────────────────────
    let tmp = tempfile::tempdir().unwrap();
    let imap = Arc::new(
        harmony_mail::imap_store::ImapStore::open(&tmp.path().join("a.db")).unwrap(),
    );

    // ── Resolver: maps bob_hash -> bob's Identity ──────────────
    let resolver: Arc<dyn RecipientResolver> = Arc::new(StubResolver {
        bob_hash,
        bob_identity: bob_pub.clone(),
    });

    // ── Drive Gateway A end-to-end ─────────────────────────────
    // Compose an RFC 5322 message. Use the same fixture shape as the
    // existing `smtp_delivers_to_local_imap` test at server.rs:3053 —
    // From/To/Subject/Date/Message-ID headers + a text body.
    let rfc822 = b"From: alice@local.example\r\n\
                   To: bob@remote.example\r\n\
                   Subject: hello from A\r\n\
                   Date: Tue, 15 Apr 2026 12:00:00 +0000\r\n\
                   Message-ID: <test-zeb113@local.example>\r\n\
                   \r\n\
                   Hello Bob, this is Alice.\r\n";

    let actions = vec![harmony_mail::SmtpAction::DeliverToHarmony {
        recipients: vec![bob_hash],
        data: rfc822.to_vec(),
    }];

    let mut session = harmony_mail::SmtpSession::new_for_test();
    let mut writer = Vec::<u8>::new();
    let mut spf_result = harmony_mail::spam::SpfResult::None;

    harmony_mail::process_async_actions(
        &actions,
        &mut session,
        &mut writer,
        &imap,
        &None,
        "local.example",
        tmp.path(),
        &mut spf_result,
        1000,                                // reject_threshold
        &None,
        &Some(Arc::clone(&publisher)),
        &Some(Arc::clone(&gateway_a_priv)),
        &Some(resolver),
    )
    .await
    .expect("process_async_actions");

    // ── Bob's gateway should receive the sealed envelope ───────
    let sample = tokio::time::timeout(Duration::from_secs(3), sub.recv_async())
        .await
        .expect("subscriber timeout")
        .expect("subscriber closed");
    let sealed_bytes = sample.payload().to_bytes().to_vec();

    // ── Open with Bob's private identity + Alice's public identity ──
    let opened = HarmonyEnvelope::open(
        &bob_priv,
        gateway_a_priv.public_identity(),
        &sealed_bytes,
    )
    .expect("open should succeed");

    // The recovered plaintext should be the serialized HarmonyMessage
    // that translate_inbound produced. We don't reproduce translate here —
    // we just assert the plaintext parses back into a valid HarmonyMessage
    // and its subject matches the SMTP Subject header.
    let recovered_msg =
        harmony_mail::message::HarmonyMessage::from_bytes(&opened.plaintext)
            .expect("recovered bytes should parse as HarmonyMessage");
    assert_eq!(recovered_msg.subject, "hello from A");
    assert_eq!(opened.sender_address, gateway_a_priv.public_identity().address_hash);

    // Cleanup: drop publisher + sessions so the Zenoh runtime can shut
    // down cleanly within the test timeout budget.
    cancel.cancel();
    drop(sub);
    drop(session_b);
    drop(session_a);
}
```

Note: several symbols in this test may be gated behind `pub(crate)` today. Promote them to `pub` as the compiler reports them:
- `harmony_mail::SmtpSession` and its `new_for_test`
- `harmony_mail::process_async_actions`
- `harmony_mail::SmtpAction`
- `harmony_mail::spam::SpfResult`
- `harmony_mail::message::HarmonyMessage` (likely already `pub`)
- `harmony_mail::imap_store::ImapStore` (likely already `pub`)
- `harmony_mail::mailbox_manager::ZenohPublisher` (likely already `pub`)

For each, open the defining file and replace `pub(crate)` with `pub`. If a type is currently non-pub at all, make it `pub`. Prefer this over `#[cfg(feature = "test-util")]` — simpler for a small refactor.

- [ ] **Step 2: Run integration test to verify it fails**

Run: `cargo test -p harmony-mail --test smtp_remote_delivery_integration 2>&1 | tail -25`
Expected: either a compile error (missing pub visibility) OR a runtime panic ("subscriber timeout"), depending on which gap bites first. Compile errors are fine — fix each as it surfaces, re-running the command until you reach a real runtime failure.

- [ ] **Step 3: Fix visibility issues iteratively**

For each compile error reported, promote the named symbol's visibility to `pub`. Commit the visibility changes when the test compiles and fails at runtime (not before — you want one clean commit per semantic change).

- [ ] **Step 4: Run the test and verify it passes**

Once compilation is clean:

Run: `cargo test -p harmony-mail --test smtp_remote_delivery_integration 2>&1 | tail -15`
Expected: `test result: ok. 1 passed`.

If the test times out waiting for the subscriber, first check the readiness-probe comment — the Zenoh auto-peer discovery may need longer than 100ms on some runners. Bump to 250ms and re-run. If still timing out, add `tracing::debug!` traces around the publish path in `publish_sealed_unicast` to confirm the Session variant fired.

- [ ] **Step 5: Commit**

```bash
git add crates/harmony-mail/tests/smtp_remote_delivery_integration.rs crates/harmony-mail/src/*.rs
git commit -m "test(mail): integration test for ZEB-113 end-to-end sealed unicast delivery"
```

---

## Task 7: Documentation + `Cargo.toml` housekeeping

**Files:**
- Modify: `crates/harmony-mail/Cargo.toml` — add any newly-needed deps (confirm `hex`, `rand`, `rand_core`, `tokio-util`, `harmony-discovery`, `harmony-identity`, `harmony-zenoh` are all already declared; most will be since the existing server.rs uses them)
- Modify: top-of-file doc comment in `crates/harmony-mail/src/server.rs` — add a short note about the remote-delivery branch.

- [ ] **Step 1: Verify Cargo deps are present**

Run:

```bash
cd $REPO_ROOT
grep -E '^(harmony-discovery|harmony-identity|harmony-zenoh|hex|rand|rand_core|tokio-util)' crates/harmony-mail/Cargo.toml
```

Expected: every name on the left appears at least once in the `[dependencies]` table. If any is missing, add it — for example, if `rand_core` is missing:

```toml
# under [dependencies]
rand_core = { version = "0.6", default-features = false }
```

Add `[dev-dependencies]` `tempfile = "3"` if Task 5 or 6 required it and it wasn't already listed.

- [ ] **Step 2: Update top-of-file doc comment on `server.rs`**

Find the `//!` module doc at the top of `crates/harmony-mail/src/server.rs`. Add a new paragraph after the existing text (near the top, not buried deep):

```rust
//! ## Remote delivery (ZEB-113 PR A)
//!
//! When a recipient is NOT homed on this gateway (`imap_store` has no user
//! for that address hash), the `DeliverToHarmony` action attempts remote
//! delivery via the optional `gateway_identity` + `recipient_resolver`
//! parameters. A hit resolves the recipient's classical `Identity` from
//! their `AnnounceRecord`, seals the serialized `HarmonyMessage` via
//! `HarmonyEnvelope::seal`, and publishes to the per-recipient key
//! `harmony/msg/v1/unicast/{recipient_hash_hex}`. Offline recipients
//! (`resolver.resolve` returns `None`) are warned-and-skipped; store-
//! and-forward is ZEB-113 PR B.
```

- [ ] **Step 3: Run the full workspace build + test once more**

Run: `cargo build --workspace 2>&1 | tail -5`
Expected: clean.

Run: `cargo test -p harmony-mail 2>&1 | tail -10`
Expected: clean. Record total test count for the PR description.

Run: `cargo test --workspace --exclude harmony-deliver 2>&1 | tail -10`
Expected: clean (or equivalent to pre-change baseline — compare test counts).

- [ ] **Step 4: Commit**

```bash
git add -A
git status  # verify only docs + Cargo.toml touched
git commit -m "docs(mail): document ZEB-113 PR A remote delivery branch in server.rs"
```

---

## Final steps: PR preparation

- [ ] **Push branch and open PR against `main`:**

```bash
cd $REPO_ROOT
git push -u origin zeb-113-smtp-remote-delivery
gh pr create --title "feat(mail): SMTP remote delivery over Zenoh (ZEB-113 PR A)" \
  --body "$(cat <<'EOF'
## Summary
- Seal + publish HarmonyMessage bytes for non-local SMTP recipients on `harmony/msg/v1/unicast/{recipient_hash_hex}`
- Thread optional `gateway_identity` + `recipient_resolver` through the SMTP pipeline; `None` = remote delivery off (backward-compat)
- Warn-and-skip when the resolver reports no announce record (per-recipient MX semantics); store-and-forward is ZEB-113 PR B

## Test plan
- [ ] `cargo test -p harmony-mail` green — <N> tests, including new `delivers_remote_warn_and_skip_when_resolver_returns_none` and `smtp_remote_delivery_round_trips_through_zenoh_to_recipient`
- [ ] `cargo test -p harmony-zenoh` green — new `msg_namespace_tests` module
- [ ] `cargo build --workspace` clean
- [ ] Manual QA (out of CI): spin up two gateways on a real mesh, SMTP to an address homed on the other gateway, confirm sealed envelope arrives
EOF
)"
```

---

## Self-review checklist (plan author)

**1. Spec coverage**

| Spec requirement | Task(s) |
|---|---|
| Thread Zenoh publisher + gateway `PrivateIdentity` through SMTP handler | Task 4 |
| For each non-local recipient, seal with recipient's public key from `AnnounceRecord` | Task 3 (helpers) + Task 5 (wire-in) |
| Publish sealed envelope to recipient's keyspace | Task 1 (namespace) + Task 2 (publisher method) + Task 5 (call site) |
| `harmony/msg/unicast/{recipient_hash}` key format → with v1 segment `harmony/msg/v1/unicast/{hash}` | Task 1 |
| Out-of-scope: `OfflineResolver` / store-and-forward | Explicitly not implemented; warn+skip covers PR A's surface |
| Unit test for sealing helper | Task 3 |
| Integration test: two sessions, sealed envelope round-trips through `open` | Task 6 |
| Negative test: no announce record → no publish, warn, overall SMTP success | Task 5 |

**2. Placeholder scan**

Scanned for: "TBD", "TODO", "implement later", "similar to Task N", "add error handling" (generic). None present. A few "confirm the line number with `grep -n …`" prompts remain — those are explicit verification steps, not hand-waving; each has a concrete command.

**3. Type consistency**

- `RecipientResolver::resolve(&self, &IdentityHash) -> Option<Identity>` — same signature in Task 3 definition, Task 5 usage, Task 6 test fixture.
- `seal_for_recipient(rng, sender, recipient, plaintext) -> Result<Vec<u8>, RemoteDeliveryError>` — same signature in Task 3 definition and Task 5 usage.
- `publish_sealed_unicast(recipient_hash_hex: String, envelope: Arc<Vec<u8>>)` — same in Task 2 definition and Task 5 usage.
- `msg::unicast_key(&str) -> String` — same in Task 1 definition, Task 2 usage, Task 6 usage.
- `gateway_identity: Option<Arc<PrivateIdentity>>` and `recipient_resolver: Option<Arc<dyn RecipientResolver>>` — consistent across Task 4 signatures and Task 5 destructure.
- `RemoteDeliveryError` has `InvalidAnnounceKey(IdentityError)` and `Seal(ZenohError)` variants — both referenced in Task 3 tests.

All types match across tasks. Plan ready.
