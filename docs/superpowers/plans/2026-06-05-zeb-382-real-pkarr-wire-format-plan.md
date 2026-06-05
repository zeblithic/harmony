# ZEB-382: Real pkarr wire format (pkarr-as-codec) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `harmony-pkarr` publish/resolve records that `relay.pkarr.org` accepts, by adopting the upstream `pkarr` crate (3.10.0) as a no-network wire-format codec while keeping our own webpki relay HTTP client.

**Architecture:** A new internal `wire.rs` module owns the pkarr codec (record → base64url TXT → `SignedPacket` → relay payload bytes, and back). `publisher.rs`/`resolver.rs` keep their public API and relay I/O but delegate all encoding/decoding to `wire.rs`. The in-house envelope (hex keys, LE seq, raw-CBOR `v`) is deleted. `MockPkarrRelay` gains a strict validating mode so default-CI tests exercise real pkarr bytes; a gated `#[ignore]` test exercises the live relay.

**Tech Stack:** Rust, `pkarr` 3.10.0 (`signed_packet` feature — no reqwest/mainline), `base64` 0.22 (URL_SAFE_NO_PAD), `bytes`, `ciborium`, `ed25519-dalek` 2.x, existing `reqwest`-based `RelayClient`.

**Spec:** `docs/superpowers/specs/2026-06-05-zeb-382-real-pkarr-wire-format-design.md`

**Per-task gate (local; the harmony core repo has no CI):**
```
cargo fmt -p harmony-pkarr -- --check
cargo clippy -p harmony-pkarr --all-targets -- -D warnings
cargo nextest run -p harmony-pkarr
cargo nextest run -p harmony-pkarr --features test-fixtures
```
harmony-pkarr is a small crate; these run in seconds. Commit before each gate; 10-minute wall-clock kill switch; `DONE_WITH_CONCERNS` if a gate is ambiguous.

---

## File Structure

| File | Responsibility | Change |
|---|---|---|
| `Cargo.toml` (workspace) | add `pkarr` + `bytes` to `[workspace.dependencies]` (`base64` already present) | Modify |
| `crates/harmony-pkarr/Cargo.toml` | add `pkarr`, `base64`, `bytes` deps | Modify |
| `crates/harmony-pkarr/src/lib.rs` | `mod wire;` | Modify |
| `crates/harmony-pkarr/src/wire.rs` | **NEW** — the pkarr codec: record↔TXT, build/parse relay payloads, z32 helper | Create |
| `crates/harmony-pkarr/src/error.rs` | add `RecordTooLarge` variant + Display | Modify |
| `crates/harmony-pkarr/src/publisher.rs` | delete `wrap_bep44_envelope` + `z32_encode_public`; call `wire::build_relay_payload` | Modify |
| `crates/harmony-pkarr/src/resolver.rs` | delete `parse_and_verify`; call `wire::parse_relay_payload` + `wire::z32_for_verifying_key` | Modify |
| `crates/harmony-pkarr/src/testing.rs` | add `MockPkarrRelay::start_strict()` (real-format validation on PUT) | Modify |

---

## Task 1: Dependencies + codec round-trip spike (RISK GATE)

Proves the entire pkarr 3.10.0 codec API compiles and round-trips in-memory before any production wiring, and confirms no second pkarr copy. If any pkarr/simple_dns method name differs from this plan, fix it here and propagate to later tasks.

**Files:**
- Modify: `Cargo.toml` (workspace `[workspace.dependencies]`)
- Modify: `crates/harmony-pkarr/Cargo.toml`
- Create: `crates/harmony-pkarr/src/wire.rs`
- Modify: `crates/harmony-pkarr/src/lib.rs`

- [ ] **Step 1: Add workspace deps.** In the root `Cargo.toml` `[workspace.dependencies]` table add (near the other entries; `base64` already exists at `0.22`):

```toml
pkarr = { version = "3.10.0", default-features = false, features = ["signed_packet"] }
bytes = "1"
```

- [ ] **Step 2: Add crate deps.** In `crates/harmony-pkarr/Cargo.toml` `[dependencies]` add:

```toml
pkarr = { workspace = true }
base64 = { workspace = true, default-features = false, features = ["alloc"] }
bytes = { workspace = true }
```

- [ ] **Step 3: Create `wire.rs` with only the spike test.** Create `crates/harmony-pkarr/src/wire.rs`:

```rust
//! pkarr wire-format codec (ZEB-382).
//!
//! Encodes a harmony `PkarrRoutingRecord` as a real pkarr `SignedPacket`
//! relay payload (z-base-32 key + BEP44 big-endian + DNS-packet `v`) and back,
//! while `relay.rs` keeps doing the actual HTTP. The record's canonical CBOR
//! rides inside one `_r` TXT record as base64url.

#[cfg(test)]
mod spike_tests {
    use base64::engine::general_purpose::URL_SAFE_NO_PAD;
    use base64::Engine as _;
    use pkarr::dns::rdata::RData;
    use pkarr::dns::{CharacterString, Name};
    use pkarr::dns::rdata::TXT;
    use pkarr::{Keypair, PublicKey, SignedPacket};

    /// Confirms: Keypair::from_secret_key, z32, builder().txt().sign(),
    /// to_relay_payload / from_relay_payload, multi-char-string TXT, and
    /// reading the TXT back — including a payload that needs > 1 char-string.
    #[test]
    fn pkarr_codec_round_trips_in_memory() {
        let seed = [7u8; 32];
        let kp = Keypair::from_secret_key(&seed);

        // 600-byte base64 payload → requires 3 DNS character-strings (<=255 each).
        let raw = vec![0xABu8; 450];
        let b64 = URL_SAFE_NO_PAD.encode(&raw);
        assert!(b64.len() > 255, "payload must exercise multi-char-string path");

        let mut txt = TXT::new();
        for chunk in b64.as_bytes().chunks(255) {
            let cs = CharacterString::new(chunk).expect("char-string <=255");
            txt.add_char_string(cs).expect("add char-string");
        }

        let signed = SignedPacket::builder()
            .txt(Name::new("_r").expect("name"), txt, 300)
            .sign(&kp)
            .expect("sign");

        let payload = signed.to_relay_payload();

        // Round-trip via the public relay-payload codec.
        let pk = PublicKey::try_from(&kp.public_key().to_bytes()).expect("pk");
        let parsed = SignedPacket::from_relay_payload(&pk, &payload).expect("verify");

        let mut found = None;
        for rr in parsed.resource_records("_r") {
            if let RData::TXT(t) = &rr.rdata {
                found = Some(String::try_from(t.clone()).expect("txt string"));
            }
        }
        assert_eq!(found.as_deref(), Some(b64.as_str()));

        // Key-consistency invariant: pkarr Keypair pubkey == ed25519_dalek's.
        let sk = ed25519_dalek::SigningKey::from_bytes(&seed);
        assert_eq!(kp.public_key().to_bytes(), sk.verifying_key().to_bytes());
    }
}
```

- [ ] **Step 4: Register the module.** In `crates/harmony-pkarr/src/lib.rs`, add after `pub mod relay;` (keep modules alphabetical-ish with the others):

```rust
mod wire;
```

- [ ] **Step 5: Resolve + verify single pkarr copy.** Run:

```
cargo tree -p harmony-pkarr -i pkarr
```
Expected: pkarr appears at **3.10.0** only (one version). If a second pkarr version appears, stop and reconcile the version before proceeding.

- [ ] **Step 6: Run the spike test.**

```
cargo nextest run -p harmony-pkarr wire::spike_tests
```
Expected: PASS. **If `add_char_string`, `CharacterString::new`, `String::try_from(TXT)`, `to_relay_payload`, or `from_relay_payload` have different signatures, adjust this test until it passes, and note the corrected names — Tasks 3-8 must use the same names.**

- [ ] **Step 7: Commit.**

```
git add Cargo.toml Cargo.lock crates/harmony-pkarr/Cargo.toml crates/harmony-pkarr/src/wire.rs crates/harmony-pkarr/src/lib.rs
git commit -m "feat(zeb-382): add pkarr 3.10.0 codec dep + round-trip spike"
```

---

## Task 2: `RecordTooLarge` error variant

**Files:**
- Modify: `crates/harmony-pkarr/src/error.rs`

- [ ] **Step 1: Write the failing test.** In `error.rs`'s `mod tests`, add:

```rust
#[test]
fn record_too_large_displays() {
    let s = format!("{}", PkarrError::RecordTooLarge);
    assert!(s.contains("too large"));
}
```

- [ ] **Step 2: Run it to verify it fails.**

```
cargo nextest run -p harmony-pkarr error::tests::record_too_large_displays
```
Expected: FAIL — `RecordTooLarge` variant does not exist (compile error).

- [ ] **Step 3: Add the variant.** In `enum PkarrError`, after `InvalidRecord`:

```rust
    /// Record exceeds the pkarr SignedPacket size budget (MAX_BYTES = 1104).
    RecordTooLarge,
```
And in the `Display` match, after the `InvalidRecord` arm:

```rust
            Self::RecordTooLarge => write!(f, "record too large for pkarr packet"),
```

- [ ] **Step 4: Run the test to verify it passes.**

```
cargo nextest run -p harmony-pkarr error::tests::record_too_large_displays
```
Expected: PASS.

- [ ] **Step 5: Commit.**

```
git add crates/harmony-pkarr/src/error.rs
git commit -m "feat(zeb-382): add PkarrError::RecordTooLarge"
```

---

## Task 3: `wire.rs` — record ↔ base64url TXT-string helpers

**Files:**
- Modify: `crates/harmony-pkarr/src/wire.rs`

- [ ] **Step 1: Write the failing test.** In `wire.rs`, add a real `#[cfg(test)] mod tests` (separate from `spike_tests`):

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use ed25519_dalek::SigningKey;
    use rand::rngs::OsRng;

    fn sample_record() -> crate::record::PkarrRoutingRecord {
        let sk = SigningKey::generate(&mut OsRng);
        let mut id_pub = [0u8; 64];
        id_pub[32..].copy_from_slice(&sk.verifying_key().to_bytes());
        crate::record::PkarrRoutingRecord::sign_new(
            vec![0xCDu8; 120], // routing_blob big enough to force >255 base64
            id_pub,
            1_000_000,
            &sk,
        )
        .expect("sign record")
    }

    #[test]
    fn txt_string_round_trips() {
        let rec = sample_record();
        let s = routing_record_to_txt_string(&rec).expect("encode");
        assert!(s.len() > 255, "encoded payload should exceed one char-string");
        let back = txt_string_to_routing_record(&s).expect("decode");
        assert_eq!(rec, back);
    }

    #[test]
    fn txt_string_rejects_garbage() {
        assert!(txt_string_to_routing_record("!!!not-base64!!!").is_err());
    }
}
```

- [ ] **Step 2: Run it to verify it fails.**

```
cargo nextest run -p harmony-pkarr wire::tests
```
Expected: FAIL — `routing_record_to_txt_string` / `txt_string_to_routing_record` not defined.

- [ ] **Step 3: Implement the helpers + module imports.** At the top of `wire.rs` (above the test modules), add:

```rust
use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use base64::Engine as _;

use crate::error::PkarrError;
use crate::record::PkarrRoutingRecord;

/// DNS label carrying the routing record's base64url payload.
const RECORD_LABEL: &str = "_r";
/// TTL (seconds) on the TXT record. Not load-bearing — harmony freshness is
/// governed by PkarrRoutingRecord::verify_skew + epoch rotation, not DNS TTL.
const RECORD_TTL: u32 = 300;

/// Encode a routing record's canonical CBOR as a base64url-unpadded string.
pub(crate) fn routing_record_to_txt_string(
    record: &PkarrRoutingRecord,
) -> Result<String, PkarrError> {
    let cbor = record.to_canonical_cbor()?;
    Ok(URL_SAFE_NO_PAD.encode(&cbor))
}

/// Decode a base64url-unpadded string back into a routing record.
pub(crate) fn txt_string_to_routing_record(
    s: &str,
) -> Result<PkarrRoutingRecord, PkarrError> {
    let cbor = URL_SAFE_NO_PAD
        .decode(s.as_bytes())
        .map_err(|_| PkarrError::InvalidRecord)?;
    PkarrRoutingRecord::from_canonical_cbor(&cbor)
}
```

- [ ] **Step 4: Run the test to verify it passes.**

```
cargo nextest run -p harmony-pkarr wire::tests
```
Expected: PASS.

- [ ] **Step 5: Commit.**

```
git add crates/harmony-pkarr/src/wire.rs
git commit -m "feat(zeb-382): wire.rs record<->base64url TXT-string helpers"
```

---

## Task 4: `wire.rs` — `build_relay_payload` (+ RecordTooLarge guard)

**Files:**
- Modify: `crates/harmony-pkarr/src/wire.rs`

- [ ] **Step 1: Write the failing tests.** Add to `wire.rs`'s `mod tests`:

```rust
    #[test]
    fn build_produces_z32_and_payload() {
        let sk = SigningKey::generate(&mut OsRng);
        let rec = sample_record();
        let (z32, payload) = build_relay_payload(&sk, &rec).expect("build");
        assert_eq!(z32.len(), 52, "z-base-32 ed25519 key is 52 chars");
        assert!(!payload.is_empty());
        assert!((payload.len() as u64) < pkarr::SignedPacket::MAX_BYTES);
    }

    #[test]
    fn build_rejects_oversize_record() {
        let sk = SigningKey::generate(&mut OsRng);
        let mut id_pub = [0u8; 64];
        id_pub[32..].copy_from_slice(&sk.verifying_key().to_bytes());
        // routing_blob far over the ~1000-byte v budget once base64'd.
        let big = crate::record::PkarrRoutingRecord::sign_new(
            vec![0u8; 3000],
            id_pub,
            1_000_000,
            &sk,
        )
        .expect("sign");
        assert_eq!(
            build_relay_payload(&sk, &big),
            Err(PkarrError::RecordTooLarge)
        );
    }
```

- [ ] **Step 2: Run them to verify they fail.**

```
cargo nextest run -p harmony-pkarr wire::tests::build_
```
Expected: FAIL — `build_relay_payload` not defined.

- [ ] **Step 3: Implement.** Add to `wire.rs` imports:

```rust
use ed25519_dalek::SigningKey;
use pkarr::dns::rdata::TXT;
use pkarr::dns::{CharacterString, Name};
use pkarr::{Keypair, SignedPacket};
```
And the function:

```rust
/// Build the `(z-base-32 key, relay PUT payload)` for `record` signed under
/// the ephemeral `signing_key`. Returns `RecordTooLarge` if the packet would
/// exceed `SignedPacket::MAX_BYTES`.
pub(crate) fn build_relay_payload(
    signing_key: &SigningKey,
    record: &PkarrRoutingRecord,
) -> Result<(String, Vec<u8>), PkarrError> {
    let keypair = Keypair::from_secret_key(&signing_key.to_bytes());
    let b64 = routing_record_to_txt_string(record)?;

    // Split the base64 payload into <=255-byte DNS character-strings.
    let mut txt = TXT::new();
    for chunk in b64.as_bytes().chunks(255) {
        let cs = CharacterString::new(chunk)
            .map_err(|_| PkarrError::SerializeError("txt char-string"))?;
        txt.add_char_string(cs)
            .map_err(|_| PkarrError::SerializeError("txt push"))?;
    }

    let name = Name::new(RECORD_LABEL).map_err(|_| PkarrError::SerializeError("txt name"))?;
    // The only realistic failure for this fixed, valid single-TXT packet is
    // exceeding SignedPacket::MAX_BYTES — map it to RecordTooLarge.
    let signed = SignedPacket::builder()
        .txt(name, txt, RECORD_TTL)
        .sign(&keypair)
        .map_err(|_| PkarrError::RecordTooLarge)?;

    Ok((keypair.to_z32(), signed.to_relay_payload().to_vec()))
}
```
**Note (from Task 1):** if `add_char_string` returns `()` rather than `Result`, drop its `.map_err(...)?`. If `.sign()` does NOT reject oversize packets, the `build_rejects_oversize_record` test will fail; in that case add, before the `Ok(...)`:
```rust
    if signed.as_bytes().len() as u64 > SignedPacket::MAX_BYTES {
        return Err(PkarrError::RecordTooLarge);
    }
```

- [ ] **Step 4: Run the tests to verify they pass.**

```
cargo nextest run -p harmony-pkarr wire::tests::build_
```
Expected: PASS.

- [ ] **Step 5: Commit.**

```
git add crates/harmony-pkarr/src/wire.rs
git commit -m "feat(zeb-382): wire::build_relay_payload with size guard"
```

---

## Task 5: `wire.rs` — `parse_relay_payload` + `z32_for_verifying_key`

**Files:**
- Modify: `crates/harmony-pkarr/src/wire.rs`

- [ ] **Step 1: Write the failing tests.** Add to `wire.rs`'s `mod tests`:

```rust
    #[test]
    fn build_then_parse_round_trips() {
        let sk = SigningKey::generate(&mut OsRng);
        let rec = sample_record();
        let (_z32, payload) = build_relay_payload(&sk, &rec).expect("build");
        let parsed =
            parse_relay_payload(&sk.verifying_key().to_bytes(), &payload).expect("parse");
        assert_eq!(parsed, rec);
    }

    #[test]
    fn parse_rejects_tampered_payload() {
        let sk = SigningKey::generate(&mut OsRng);
        let rec = sample_record();
        let (_z32, mut payload) = build_relay_payload(&sk, &rec).expect("build");
        payload[0] ^= 0xFF; // corrupt the signature
        assert_eq!(
            parse_relay_payload(&sk.verifying_key().to_bytes(), &payload),
            Err(PkarrError::OuterSignatureInvalid)
        );
    }

    #[test]
    fn parse_rejects_wrong_pubkey() {
        let sk = SigningKey::generate(&mut OsRng);
        let other = SigningKey::generate(&mut OsRng);
        let rec = sample_record();
        let (_z32, payload) = build_relay_payload(&sk, &rec).expect("build");
        assert_eq!(
            parse_relay_payload(&other.verifying_key().to_bytes(), &payload),
            Err(PkarrError::OuterSignatureInvalid)
        );
    }

    #[test]
    fn z32_matches_build() {
        let sk = SigningKey::generate(&mut OsRng);
        let rec = sample_record();
        let (z32, _payload) = build_relay_payload(&sk, &rec).expect("build");
        assert_eq!(z32_for_verifying_key(&sk.verifying_key()).unwrap(), z32);
    }
```

- [ ] **Step 2: Run them to verify they fail.**

```
cargo nextest run -p harmony-pkarr wire::tests
```
Expected: FAIL — `parse_relay_payload` / `z32_for_verifying_key` not defined.

- [ ] **Step 3: Implement.** Add to `wire.rs` imports:

```rust
use ed25519_dalek::VerifyingKey;
use pkarr::dns::rdata::RData;
use pkarr::PublicKey;
```
And the functions:

```rust
/// Parse + outer-verify a relay GET payload back into a routing record.
/// Verifies the BEP44 (ephemeral-key) signature against `expected_pk`; does
/// NOT verify the inner identity signature — the caller does that.
pub(crate) fn parse_relay_payload(
    expected_pk: &[u8; 32],
    payload: &[u8],
) -> Result<PkarrRoutingRecord, PkarrError> {
    let pk = PublicKey::try_from(expected_pk).map_err(|_| PkarrError::InvalidRecord)?;
    let bytes = bytes::Bytes::copy_from_slice(payload);
    let signed = SignedPacket::from_relay_payload(&pk, &bytes)
        .map_err(|_| PkarrError::OuterSignatureInvalid)?;
    let txt = read_record_txt(&signed)?;
    txt_string_to_routing_record(&txt)
}

/// z-base-32 string for an ed25519 verifying key (the relay GET URL key).
pub(crate) fn z32_for_verifying_key(pk: &VerifyingKey) -> Result<String, PkarrError> {
    let pk = PublicKey::try_from(&pk.to_bytes()).map_err(|_| PkarrError::InvalidRecord)?;
    Ok(pk.to_z32())
}

/// Read the concatenated base64 string from the `_r` TXT record.
fn read_record_txt(signed: &SignedPacket) -> Result<String, PkarrError> {
    for rr in signed.resource_records(RECORD_LABEL) {
        if let RData::TXT(txt) = &rr.rdata {
            return String::try_from(txt.clone()).map_err(|_| PkarrError::InvalidRecord);
        }
    }
    Err(PkarrError::InvalidRecord)
}
```

- [ ] **Step 4: Run the tests to verify they pass.**

```
cargo nextest run -p harmony-pkarr wire::tests
```
Expected: PASS (all wire::tests).

- [ ] **Step 5: Commit.**

```
git add crates/harmony-pkarr/src/wire.rs
git commit -m "feat(zeb-382): wire::parse_relay_payload + z32_for_verifying_key"
```

---

## Task 6: `MockPkarrRelay` strict validating mode

Adds a strict constructor that rejects anything that is not a real pkarr relay payload, so the publisher/resolver tests (Tasks 7-8) prove real-format conformance. The existing lax `start()` is retained for the relay-pool/cooldown tests in `relay.rs` that intentionally PUT arbitrary bytes.

**Files:**
- Modify: `crates/harmony-pkarr/src/testing.rs`

- [ ] **Step 1: Write the failing test.** In `testing.rs`'s `mod tests`, add:

```rust
    #[tokio::test]
    async fn strict_relay_rejects_non_pkarr_bytes() {
        let relay = MockPkarrRelay::start_strict().await;
        let client = reqwest::Client::new();
        // Non-z32 key + garbage body → 400.
        let resp = client
            .put(format!("{}/not-a-real-key", relay.base_url))
            .body(b"in-house-bytes".to_vec())
            .send()
            .await
            .expect("put");
        assert_eq!(resp.status(), 400);
    }
```

- [ ] **Step 2: Run it to verify it fails.**

```
cargo nextest run -p harmony-pkarr testing::tests::strict_relay_rejects_non_pkarr_bytes
```
Expected: FAIL — `start_strict` not defined.

- [ ] **Step 3: Implement strict mode.** In `testing.rs`:

Change the `Store` state to carry a strict flag by introducing a small state struct. Replace the `type Store = ...` line and the `start`/route/handlers with:

```rust
/// In-memory store keyed by the URL path key (z-base-32 in strict mode).
type Store = Arc<RwLock<HashMap<String, Vec<u8>>>>;

#[derive(Clone)]
struct MockState {
    store: Store,
    /// When true, PUT validates the z32 key + BEP44 signature and returns 400
    /// for anything that is not a real pkarr relay payload.
    strict: bool,
}
```

Replace `MockPkarrRelay::start` with a lax + strict pair sharing an inner constructor:

```rust
impl MockPkarrRelay {
    /// Start a lax mock (stores whatever is PUT). For relay-pool/cooldown tests.
    pub async fn start() -> Self {
        Self::start_inner(false).await
    }

    /// Start a strict mock that validates the real pkarr relay format on PUT
    /// (z-base-32 key + `from_relay_payload` signature check). Rejects the old
    /// in-house dialect with 400.
    pub async fn start_strict() -> Self {
        Self::start_inner(true).await
    }

    async fn start_inner(strict: bool) -> Self {
        let state = MockState {
            store: Arc::new(RwLock::new(HashMap::new())),
            strict,
        };
        let app = Router::new()
            .route("/{key}", put(put_record).get(get_record))
            .with_state(state);

        let listener = tokio::net::TcpListener::bind(SocketAddr::from(([127, 0, 0, 1], 0)))
            .await
            .expect("bind to 127.0.0.1:0");
        let addr = listener.local_addr().expect("local_addr");
        let base_url = format!("http://{}", addr);

        let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel();
        let server_task = tokio::spawn(async move {
            axum::serve(listener, app)
                .with_graceful_shutdown(async move {
                    let _ = shutdown_rx.await;
                })
                .await
                .expect("axum serve");
        });

        Self {
            base_url,
            _shutdown: shutdown_tx,
            _server_task: server_task,
        }
    }
}
```

Replace the two handlers to take `MockState`:

```rust
async fn put_record(
    Path(key): Path<String>,
    State(state): State<MockState>,
    body: axum::body::Bytes,
) -> impl IntoResponse {
    if state.strict {
        let Ok(pk) = pkarr::PublicKey::try_from(key.as_str()) else {
            return StatusCode::BAD_REQUEST;
        };
        let payload = bytes::Bytes::copy_from_slice(&body);
        if pkarr::SignedPacket::from_relay_payload(&pk, &payload).is_err() {
            return StatusCode::BAD_REQUEST;
        }
    }
    state.store.write().await.insert(key, body.to_vec());
    StatusCode::OK
}

async fn get_record(
    Path(key): Path<String>,
    State(state): State<MockState>,
) -> impl IntoResponse {
    match state.store.read().await.get(&key) {
        Some(bytes) => (StatusCode::OK, bytes.clone()).into_response(),
        None => StatusCode::NOT_FOUND.into_response(),
    }
}
```

- [ ] **Step 4: Run the test + the existing lax tests.**

```
cargo nextest run -p harmony-pkarr testing::tests
cargo nextest run -p harmony-pkarr --features test-fixtures testing::tests
```
Expected: PASS (lax `round_trip_put_then_get`, `get_missing_key_returns_404`, and the new strict test).

- [ ] **Step 5: Commit.**

```
git add crates/harmony-pkarr/src/testing.rs
git commit -m "feat(zeb-382): MockPkarrRelay strict real-format validation mode"
```

---

## Task 7: `publisher.rs` — swap to real wire format

**Files:**
- Modify: `crates/harmony-pkarr/src/publisher.rs`

- [ ] **Step 1: Write/adjust the failing test.** In `publisher.rs`'s `mod tests`, add a test that the PUT body is a real pkarr payload by using the strict mock. Add near the other publisher tests:

```rust
    #[tokio::test]
    async fn publishes_real_pkarr_payload_to_strict_mock() {
        use crate::testing::MockPkarrRelay;
        let relay = MockPkarrRelay::start_strict().await;
        let client = Arc::new(crate::relay::RelayClient::new(
            crate::relay::RelayPool::new(vec![relay.base_url.clone()]),
        ));
        let publisher = Arc::new(PkarrPublisher::new(client));

        let id_sk = ed25519_dalek::SigningKey::from_bytes(&[3u8; 32]);
        let mut id_pub = [0u8; 64];
        id_pub[32..].copy_from_slice(&id_sk.verifying_key().to_bytes());
        let id_sk_for_builder = id_sk.clone();

        publisher
            .register(
                "h1".to_string(),
                Arc::new(|_now| ed25519_dalek::SigningKey::from_bytes(&[9u8; 32])),
                Arc::new(move |now| {
                    crate::record::PkarrRoutingRecord::sign_new(
                        b"routing".to_vec(),
                        id_pub,
                        now,
                        &id_sk_for_builder,
                    )
                    .expect("sign record")
                }),
            )
            .await;

        let handle = Arc::clone(&publisher).spawn();
        // Give the background loop time to publish once.
        tokio::time::sleep(std::time::Duration::from_millis(200)).await;
        handle.abort();

        // Strict mock accepted the PUT → the payload is real pkarr format.
        let ephemeral = ed25519_dalek::SigningKey::from_bytes(&[9u8; 32]);
        let z32 = crate::wire::z32_for_verifying_key(&ephemeral.verifying_key()).unwrap();
        let client2 = reqwest::Client::new();
        let got = client2
            .get(format!("{}/{}", relay.base_url, z32))
            .send()
            .await
            .expect("get");
        assert_eq!(got.status(), 200);
    }
```

- [ ] **Step 2: Run it to verify it fails.**

```
cargo nextest run -p harmony-pkarr publisher::tests::publishes_real_pkarr_payload_to_strict_mock
```
Expected: FAIL — the current `publish_one` writes the in-house envelope + hex key, which the strict mock rejects (the GET returns 404 because the PUT was 400'd).

- [ ] **Step 3: Swap `publish_one` to the wire codec.** In `publisher.rs`, replace the body of `publish_one` from the `let cbor = ...` line through the `self.relay.put(...)` call:

```rust
        let ephemeral_key = (pub_state.ephemeral_key_builder)(now);
        let record = (pub_state.builder)(now);
        let (key_z32, payload) = crate::wire::build_relay_payload(&ephemeral_key, &record)?;
        self.relay.put(&key_z32, &payload).await
```

- [ ] **Step 4: Delete the dead in-house code.** Remove `fn wrap_bep44_envelope(...)` and `fn z32_encode_public(...)` (the two functions, ~lines 265-307) and their doc comments. Remove the now-unused `use crate::record::PkarrRoutingRecord;` import only if it is no longer referenced (it is still referenced by `RecordBuilder` — keep it). Leave `now_ms()` (still used).

- [ ] **Step 5: Run the test + existing publisher tests.**

```
cargo nextest run -p harmony-pkarr publisher::tests
```
Expected: PASS. If an existing publisher test asserted the old envelope bytes, update it to assert via `crate::wire::parse_relay_payload` round-trip instead.

- [ ] **Step 6: Gate + commit.**

```
cargo fmt -p harmony-pkarr
cargo clippy -p harmony-pkarr --all-targets -- -D warnings
git add crates/harmony-pkarr/src/publisher.rs
git commit -m "feat(zeb-382): publisher emits real pkarr relay payloads"
```

---

## Task 8: `resolver.rs` — swap to real wire format

**Files:**
- Modify: `crates/harmony-pkarr/src/resolver.rs`

- [ ] **Step 1: Write/confirm the failing test.** In `resolver.rs`'s `mod tests`, add an end-to-end test through the strict mock (publish a real payload, resolve it):

```rust
    #[tokio::test]
    async fn resolves_real_pkarr_payload() {
        use crate::testing::MockPkarrRelay;
        let relay = MockPkarrRelay::start_strict().await;
        let put_client = Arc::new(crate::relay::RelayClient::new(
            crate::relay::RelayPool::new(vec![relay.base_url.clone()]),
        ));

        let ephemeral = ed25519_dalek::SigningKey::from_bytes(&[5u8; 32]);
        let id_sk = ed25519_dalek::SigningKey::from_bytes(&[6u8; 32]);
        let mut id_pub = [0u8; 64];
        id_pub[32..].copy_from_slice(&id_sk.verifying_key().to_bytes());
        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;
        let rec = crate::record::PkarrRoutingRecord::sign_new(
            b"iroh-routing".to_vec(),
            id_pub,
            now_ms,
            &id_sk,
        )
        .expect("sign");
        let (z32, payload) = crate::wire::build_relay_payload(&ephemeral, &rec).unwrap();
        put_client.put(&z32, &payload).await.expect("publish");

        let resolver = PkarrResolver::new(put_client);
        let got = resolver
            .resolve(&ephemeral.verifying_key())
            .await
            .expect("resolve")
            .expect("present");
        assert_eq!(got.routing_blob, b"iroh-routing".to_vec());
    }
```

- [ ] **Step 2: Run it to verify it fails.**

```
cargo nextest run -p harmony-pkarr resolver::tests::resolves_real_pkarr_payload
```
Expected: FAIL — `resolve` uses `hex::encode` for the GET key + `parse_and_verify` on the in-house envelope, neither of which matches the real payload the strict mock stored.

- [ ] **Step 3: Swap the resolve key + parse.** In `resolver.rs::resolve`, replace:

```rust
        let key_z32 = hex::encode(pk_bytes);
```
with:

```rust
        let key_z32 = crate::wire::z32_for_verifying_key(pk)?;
```
and replace the `parse_and_verify(&envelope, pk)` call (inside the `Some(envelope) =>` arm) with:

```rust
                let record = match crate::wire::parse_relay_payload(&pk_bytes, &envelope) {
```
(keep the surrounding `match { Ok(r) => r, Err(e) => { ...negative-cache, return Ok(None) } }` block unchanged).

- [ ] **Step 4: Delete dead code + fix imports.** Remove `fn parse_and_verify(...)` (~lines 192-219). Remove the now-unused imports `use ed25519_dalek::{Signature, Verifier, VerifyingKey};` → change to `use ed25519_dalek::VerifyingKey;` (only `VerifyingKey` is still used).

- [ ] **Step 5: Run the test + existing resolver tests.**

```
cargo nextest run -p harmony-pkarr resolver::tests
```
Expected: PASS. Existing resolver tests that publish via `PkarrPublisher`/mock now round-trip through the real format. If any test built an in-house envelope by hand, replace it with `crate::wire::build_relay_payload`.

- [ ] **Step 6: Gate + commit.**

```
cargo fmt -p harmony-pkarr
cargo clippy -p harmony-pkarr --all-targets -- -D warnings
git add crates/harmony-pkarr/src/resolver.rs
git commit -m "feat(zeb-382): resolver parses real pkarr relay payloads"
```

---

## Task 9: Gated live-relay integration test

A real publish+resolve against `relay.pkarr.org`, kept out of the default gate (`#[ignore]` + env-gated). This is the "can never silently regress to mock-only" guard and the automated cross-machine smoke test.

**Files:**
- Modify: `crates/harmony-pkarr/src/wire.rs`

- [ ] **Step 1: Add the gated test.** In `wire.rs`'s `mod tests`, add:

```rust
    /// Live round-trip against the public relay. Ignored by default; run with:
    ///   HARMONY_PKARR_LIVE_RELAY=1 cargo nextest run -p harmony-pkarr \
    ///     --run-ignored all wire::tests::live_relay_round_trip
    #[tokio::test]
    #[ignore = "hits relay.pkarr.org; set HARMONY_PKARR_LIVE_RELAY=1 to run"]
    async fn live_relay_round_trip() {
        if std::env::var("HARMONY_PKARR_LIVE_RELAY").is_err() {
            return;
        }
        use crate::relay::{RelayClient, RelayPool};

        let relay = RelayClient::new(RelayPool::new(vec![
            "https://relay.pkarr.org".to_string(),
        ]));
        let ephemeral = SigningKey::generate(&mut OsRng);
        let rec = sample_record();
        let (z32, payload) = build_relay_payload(&ephemeral, &rec).expect("build");

        relay.put(&z32, &payload).await.expect("publish to live relay");
        tokio::time::sleep(std::time::Duration::from_secs(2)).await;

        let got = relay
            .get(&z32)
            .await
            .expect("get from live relay")
            .expect("record present on relay");
        let parsed =
            parse_relay_payload(&ephemeral.verifying_key().to_bytes(), &got).expect("parse");
        assert_eq!(parsed, rec);
    }
```

- [ ] **Step 2: Verify it is skipped by default.**

```
cargo nextest run -p harmony-pkarr wire::tests
```
Expected: PASS, with `live_relay_round_trip` reported as skipped/ignored (not executed).

- [ ] **Step 3: Verify it runs + passes live (manual, networked).**

```
HARMONY_PKARR_LIVE_RELAY=1 cargo nextest run -p harmony-pkarr --run-ignored all wire::tests::live_relay_round_trip
```
Expected: PASS against the live relay. (If the relay is briefly unreachable, this is the one networked test and is allowed to be re-run; it never runs in the default gate.)

- [ ] **Step 4: Commit.**

```
git add crates/harmony-pkarr/src/wire.rs
git commit -m "test(zeb-382): gated live relay.pkarr.org round-trip"
```

---

## Task 10: Final sweep + push + PR

**Files:** none (verification + VCS)

- [ ] **Step 1: Full local gate.**

```
cargo fmt --all -- --check
cargo clippy -p harmony-pkarr --all-targets -- -D warnings
cargo nextest run -p harmony-pkarr
cargo nextest run -p harmony-pkarr --features test-fixtures
```
Expected: all green; live test skipped.

- [ ] **Step 2: Confirm dead code is gone + single pkarr.**

```
grep -rn "wrap_bep44_envelope\|z32_encode_public\|parse_and_verify" crates/harmony-pkarr/src/ || echo "in-house codec removed"
cargo tree -p harmony-pkarr -i pkarr | grep -c "pkarr v3.10.0"
```
Expected: no matches for the deleted fns; exactly one pkarr 3.10.0.

- [ ] **Step 3: Push the branch.**

```
git push -u origin zeb-382-real-pkarr-wire-format
```

- [ ] **Step 4: Open the PR (harmony repo).**

```
gh pr create --repo zeblithic/harmony --title "ZEB-382: real pkarr wire format (pkarr-as-codec)" --body "<summary + spec/plan links + Closes ZEB-382>"
```
PR body references: spec `docs/superpowers/specs/2026-06-05-zeb-382-real-pkarr-wire-format-design.md`, plan `docs/superpowers/plans/2026-06-05-zeb-382-real-pkarr-wire-format-plan.md`, the two-gate relationship to ZEB-381 (#272), relay-only scope (ZEB-380 owns the SPOF), and the testing strategy.

- [ ] **Step 5: Autonomous bot-review loop.** Watch CodeRabbit / Cursor / CodeAnt / Qodo across all three comment buckets; **never** trigger Greptile; bundle fixes into ONE push per round; pushover at ready-to-merge. Do NOT self-merge (Jake's gate).

---

## Follow-up (after the harmony PR merges — not part of this plan's tasks)

- **harmony-client rev-bump PR:** bump `harmony-pkarr` git `rev` (prod dep + `test-fixtures` dev-dep in `src-tauri/Cargo.toml`) to the merged harmony `main` SHA; regenerate `Cargo.lock`; no source change. Mirrors ZEB-381 #193.
- **Cross-machine milestone:** run the gated live test (or a manual Koya-publish / Ildwyn-resolve) to complete the invite-redeem first-contact.

---

## Self-Review

**Spec coverage:** §3 approach → Tasks 1,4,5 (codec via pkarr, our relay kept). §4 module surgery → Tasks 7 (publisher), 8 (resolver), `wire.rs` new (1,3,4,5); derive/record/relay untouched. §4.2 key invariant → Task 1 + Task 5. §5 record→TXT (unified CBOR blob, `_r`, base64url, 255-chunk, RecordTooLarge) → Tasks 2,3,4. §6 two-sig model → Tasks 5 (outer via from_relay_payload), 7/8 keep record.rs inner verifications. §6.1 testing (strict mock + unit + gated live) → Tasks 6,7,8,9. §7 deps → Task 1. §8 hard cutover → no migration tasks (correct). §9 rollout → Task 10 + Follow-up.

**Placeholder scan:** No TBD/TODO; the two conditional notes in Task 4 are explicit fallbacks gated by a named test, not placeholders.

**Type consistency:** `build_relay_payload(&SigningKey, &PkarrRoutingRecord) -> (String, Vec<u8>)`, `parse_relay_payload(&[u8;32], &[u8]) -> PkarrRoutingRecord`, `z32_for_verifying_key(&VerifyingKey) -> String`, `routing_record_to_txt_string`/`txt_string_to_routing_record`, `RECORD_LABEL = "_r"`, `MockPkarrRelay::start`/`start_strict` — names consistent across Tasks 3-9. `PkarrError::RecordTooLarge` defined Task 2, used Task 4. `crate::wire::*` paths consistent in publisher/resolver.
