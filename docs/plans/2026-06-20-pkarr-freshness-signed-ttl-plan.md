# pkarr Freshness: Signed-TTL Trust Model (facet-A) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. **This repo forbids `- [ ]` markdown checkbox TODO tracking** — steps below are plain numbered/bold form. Track progress with TodoWrite during execution, not with checkboxes in this file.

**Goal:** Replace pkarr's hard ±30-minute wall-clock freshness window with a publisher-signed TTL (`valid_until_ms`) as the primary freshness mechanism, so a routing record stays resolvable for its cryptographically-committed validity window instead of only ~30 min after each (rare) republish.

**Architecture:** A one-flag-day wire change adds `valid_until_ms` to the inner signed preimage of `PkarrRoutingRecord`. The resolver swaps the symmetric ±30 min `verify_skew` for a `verify_freshness` (future-strict lower bound + signed-TTL upper bound), adds in-memory per-key BEP44-`seq` anti-rollback, and gains a cache-bypassing "query-all-relays, take-freshest-by-seq" resolve consumed by a dial-fail retry. Hard cutover (no legacy fallback) closes the downgrade-attack surface. Two PRs: **harmony core first** (`crates/harmony-pkarr`), then **harmony-client** (`src-tauri`) once core merges and its git rev is bumped.

**Tech Stack:** Rust; `ed25519-dalek` (inner identity sig), `pkarr 3.10` / `ntimestamp` (BEP44 envelope + `seq`), `ciborium` (canonical CBOR), `lru`, `tokio`, `cargo-nextest`.

**Design spec:** `docs/specs/2026-06-20-pkarr-freshness-signed-ttl-design.md` (commit `21b56bb`, this branch).

---

## Flag-Day Coordination (read first)

Adding `valid_until_ms` to the signed preimage is **wire-incompatible**:
- Old records sign a CBOR **3-tuple** `(routing_blob, harmony_identity_pub, announced_at_ms)`; new records sign a **4-tuple** that also covers `valid_until_ms`.
- The struct gains a required (`vu`) field with **no `#[serde(default)]`** — an old CBOR map lacking `vu` fails to decode and is dropped. There is deliberately **no legacy ±30 min fallback path** (that would be a downgrade-attack surface: an adversary strips `valid_until` to force the weaker guard). This is the hard cutover.
- Acceptable pre-launch: small fleet, frequent republish, the canonical Zeblithic node re-publishes fresh under the new format on upgrade.

**Decisions locked for this plan (spec-review defaults; change here if Jake overrides):**
- Reachability TTL (all four cases: identity / community / friend / invite) = **7 days** (one epoch).
- `FUTURE_TOLERANCE_MS` = **5 minutes**.
- Anti-rollback comparison = **`seq < highwater` rejects** (strictly-older). Equal-`seq` is the same signed bytes and is accepted, so an idempotent re-resolve after cache expiry is not falsely rejected. (Spec §3.3 wrote `<=`; this is the corrected form — see Self-Review.)
- Diverse-relay (§3.4) is realized as **"query all relays, take freshest-by-`seq`"** (cache-bypassing) rather than per-relay provenance tracking. Same security property (a single stale relay can't pin the resolver), simpler seam. See Self-Review.

**Scope knob for Jake:** the headline DoD win — *resolve >30 min after publish, within TTL* — is delivered by **Part A Tasks A1–A2 + A6** and **Part B Tasks B1–B3** alone. The defense-in-depth tier (seq anti-rollback A3; diverse-relay seam A4 + dial retry B4) matches the approved spec's full DoD but is the larger, more optional half. If a tighter first cut is wanted, the natural cut line is: ship signed-TTL + future-strict + fixtures first, fast-follow seq + diverse-relay. Default in this plan = **ship the whole approved spec.**

---

## File Structure

### PR1 — harmony core (`crates/harmony-pkarr/`)
- **`src/record.rs`** *(modify)* — add `valid_until_ms` field (`vu`) to `PkarrRoutingRecord`; 4-tuple `canonical_signed_bytes`; `sign_new` gains `valid_until_ms` param; replace `verify_skew`/`SKEW_TOLERANCE_MS` with `verify_freshness`/`FUTURE_TOLERANCE_MS`.
- **`src/wire.rs`** *(modify)* — `parse_relay_payload` returns `(PkarrRoutingRecord, u64)` surfacing the BEP44 `seq`; add a test-only `build_relay_payload_with_seq` helper (explicit-`seq` payloads for deterministic tests).
- **`src/resolver.rs`** *(modify)* — call `verify_freshness`; thread `seq`; add in-memory per-key seq highwater + check; add `resolve_freshest` / `resolve_window_freshest` (cache-bypassing, all-relay, freshest-by-seq).
- **`src/relay.rs`** *(modify)* — add `RelayClient::get_all(key_z32)` (query all available relays, return every `(relay_url, envelope)` hit) to back `resolve_freshest`.
- **`tests/wire_format_pkarr_routing_record_fixtures.rs`** *(modify)* — regenerate the pinned canonical-CBOR hex (now 4-field map with `vu`).

### PR2 — harmony-client (`src-tauri/`)
- **`Cargo.toml`** *(modify)* — bump the two `harmony-pkarr` git `rev`s (lines 109 + 204) to PR1's merge commit.
- **`src/reachability_record.rs`** *(modify)* — add `pub(crate) const REACHABILITY_RECORD_TTL_MS`.
- **`src/pkarr_identity_publisher.rs` / `pkarr_community_publisher.rs` / `pkarr_friend_publisher.rs` / `pkarr_invite_publisher.rs`** *(modify)* — thread `valid_until = at_ms + REACHABILITY_RECORD_TTL_MS` into each `sign_new`.
- **`src/pkarr_resolver_adapter.rs`** *(modify)* — `verify_skew` → `verify_freshness` (line 82).
- **`src/lib.rs`** *(modify)* — `verify_skew` → `verify_freshness` at the three live sites (39977 / 41546 / 43054); add dial-fail diverse-relay retry in the redeem (`~40042`) and add-friend (`~43083`) paths.
- **`src/network_health.rs`** *(modify)* — `verify_skew` → `verify_freshness` (1341); `sign_new` + TTL (2560).
- **client fixtures** *(modify)* — regenerate any pinned pkarr-record hex under `src-tauri/tests/`.

---

# PART A — harmony core (PR1)

**Branch:** `pkarr-freshness-signed-ttl` (already created; carries the spec at `21b56bb`). All `cargo` commands run from the repo root `/Users/zeblith/work/zeblithic/harmony`. Per-task gate for this crate: `cargo nextest run -p harmony-pkarr --all-targets --features test-fixtures`. Implementer discipline: commit before running the full gate; 10-min wall-clock kill switch on any cargo command; report DONE_WITH_CONCERNS rather than stalling.

---

### Task A1: Signed `valid_until_ms` — field, 4-tuple preimage, `sign_new` param, fixture regen

**Files:**
- Modify: `crates/harmony-pkarr/src/record.rs`
- Modify: `crates/harmony-pkarr/tests/wire_format_pkarr_routing_record_fixtures.rs`

This is an atomic wire change: the struct field, the signing preimage, the `sign_new` signature, and every in-crate caller move together so the crate stays compiling. `verify_skew` is left untouched in this task (it reads only `announced_at_ms`, so it still compiles) — it is replaced in A2.

**Step 1 — Write the failing test.** Append to the `tests` module in `record.rs`:

```rust
    #[test]
    fn valid_until_is_signed_and_tamper_proof() {
        let sk = SigningKey::generate(&mut OsRng);
        let identity_pub = fixture_identity_pubkey(&sk);
        let rec = PkarrRoutingRecord::sign_new(
            b"blob".to_vec(),
            identity_pub,
            1_000_000,
            1_000_000 + 7 * 24 * 60 * 60 * 1000, // valid_until = announced + 7d
            &sk,
        )
        .expect("sign");
        assert_eq!(rec.valid_until_ms, 1_000_000 + 7 * 24 * 60 * 60 * 1000);
        assert!(rec.verify_inner_sig().is_ok());

        // Tampering valid_until must break the inner signature.
        let mut tampered = rec.clone();
        tampered.valid_until_ms += 1;
        assert_eq!(
            tampered.verify_inner_sig(),
            Err(PkarrError::InnerSignatureInvalid)
        );
    }
```

**Step 2 — Run, verify it fails to COMPILE.**
Run: `cargo nextest run -p harmony-pkarr --features test-fixtures -E 'test(valid_until_is_signed)'`
Expected: compile error — `sign_new` takes 4 args, not 5; no field `valid_until_ms`.

**Step 3 — Add the field.** In the `PkarrRoutingRecord` struct (`record.rs:13-33`), add after `announced_at_ms` (note: **no `#[serde(default)]`** — old records must fail to decode):

```rust
    /// Wall-clock publication time, ms since unix epoch.
    #[serde(rename = "at")]
    pub announced_at_ms: u64,

    /// Signed validity horizon, ms since unix epoch. The record is honored
    /// while `now <= valid_until_ms`. Covered by `inner_sig`, so it cannot be
    /// forged or extended by a relay/attacker. No `serde(default)`: a record
    /// without this field (old wire format, or a stripped field) fails to
    /// decode and is dropped — the hard cutover that prevents downgrade.
    #[serde(rename = "vu")]
    pub valid_until_ms: u64,
```

**Step 4 — Make the preimage a 4-tuple.** Replace `canonical_signed_bytes` (`record.rs:130-148`):

```rust
fn canonical_signed_bytes(
    routing_blob: &[u8],
    harmony_identity_pub: &[u8; 64],
    announced_at_ms: u64,
    valid_until_ms: u64,
) -> Result<alloc::vec::Vec<u8>, PkarrError> {
    // Tuple-as-array: deterministic CBOR ordering. ciborium encodes tuples
    // as CBOR arrays.
    let mut out = alloc::vec::Vec::new();
    ciborium::into_writer(
        &(
            serde_bytes::Bytes::new(routing_blob),
            serde_bytes::Bytes::new(harmony_identity_pub.as_ref()),
            announced_at_ms,
            valid_until_ms,
        ),
        &mut out,
    )
    .map_err(|_| PkarrError::SerializeError("canonical_signed_bytes"))?;
    Ok(out)
}
```

**Step 5 — Add the `sign_new` param + set the field.** In `sign_new` (`record.rs:48-72`) add `valid_until_ms: u64` after `announced_at_ms`, pass it to `canonical_signed_bytes`, and set the field:

```rust
    pub fn sign_new(
        routing_blob: alloc::vec::Vec<u8>,
        harmony_identity_pub: [u8; 64],
        announced_at_ms: u64,
        valid_until_ms: u64,
        identity_signing_key: &SigningKey,
    ) -> Result<Self, PkarrError> {
        // ... key-match guard unchanged ...
        let to_sign = canonical_signed_bytes(
            &routing_blob,
            &harmony_identity_pub,
            announced_at_ms,
            valid_until_ms,
        )?;
        let sig = identity_signing_key.sign(&to_sign);
        Ok(Self {
            routing_blob,
            harmony_identity_pub,
            announced_at_ms,
            valid_until_ms,
            inner_sig: sig.to_bytes(),
        })
    }
```

**Step 6 — Update `verify_inner_sig`'s preimage call** (`record.rs:83-87`) to pass `self.valid_until_ms`:

```rust
        let to_verify = canonical_signed_bytes(
            &self.routing_blob,
            &self.harmony_identity_pub,
            self.announced_at_ms,
            self.valid_until_ms,
        )?;
```

**Step 7 — Update every other in-crate `sign_new` caller.** Add a `valid_until_ms` arg to each. Use `announced_at + 7d` form for clarity. Sites:
- `record.rs` tests: `round_trip_canonical_cbor` (line ~168), `verify_inner_sig_accepts_valid` (~184), `verify_inner_sig_rejects_tampered_blob` (~193), `verify_inner_sig_rejects_tampered_at` (~206), the `verify_skew_*` tests (~221/231 — these are removed in A2, but make them compile now), `sign_new_rejects_mismatched_key` (~249), `verify_identity_match_rejects_substitution` (~258). For each, insert the 4th arg, e.g.:
  `PkarrRoutingRecord::sign_new(b"blob".to_vec(), identity_pub, 1_000_000, 1_000_000 + 604_800_000, &sk)`
  (`604_800_000` = 7 days in ms.)
- `resolver.rs` test builders (lines ~224 and ~280): inside `Arc::new(move |now_ms| { PkarrRoutingRecord::sign_new(..., now_ms, now_ms + 604_800_000, &identity_sk_clone) ... })`.
- `resolver.rs` `resolves_real_pkarr_payload` (line ~324): `sign_new(b"iroh-routing".to_vec(), id_pub, now_ms, now_ms + 604_800_000, &id_sk)`.
- `wire.rs` `sample_record` (line ~129): `sign_new(vec![0xCDu8; 120], id_pub, 1_000_000, 1_000_000 + 604_800_000, &sk)`.
- `wire.rs` `build_rejects_oversize_record` (line ~196): `sign_new(vec![0u8; 3000], id_pub, 1_000_000, 1_000_000 + 604_800_000, &sk)`.

**Step 8 — Regenerate the fixture pin.** In `tests/wire_format_pkarr_routing_record_fixtures.rs`, add the `valid_until_ms` arg to the `sign_new` call (use a fixed value for determinism), then regenerate the expected hex:

```rust
    let rec = PkarrRoutingRecord::sign_new(
        b"deterministic-routing-blob".to_vec(),
        identity_pub,
        1_700_000_000_000u64,            // fixed announced_at
        1_700_000_000_000u64 + 604_800_000u64, // fixed valid_until (+7d)
        &sk,
    )
    .expect("sign");
```
Run the test once to get the actual hex from the failure diff, then paste it into `expected`. Keep the "regenerate only with full understanding" comment.

Run: `cargo nextest run -p harmony-pkarr --features test-fixtures -E 'test(canonical_cbor_bytes_pinned)'` — read the `left`/`right` diff, copy the actual (`left`) hex into `expected`, re-run → PASS.

**Step 9 — Run the crate tests.**
Run: `cargo nextest run -p harmony-pkarr --all-targets --features test-fixtures`
Expected: PASS (the A2-doomed `verify_skew_*` tests still pass here — they're unchanged behavior).

**Step 10 — Commit.**
```bash
git add crates/harmony-pkarr/src/record.rs crates/harmony-pkarr/tests/wire_format_pkarr_routing_record_fixtures.rs
git commit -m "feat(pkarr): sign valid_until_ms into PkarrRoutingRecord preimage (4-tuple)"
```

---

### Task A2: `verify_freshness` — future-strict lower bound + signed-TTL upper bound

**Files:**
- Modify: `crates/harmony-pkarr/src/record.rs`
- Modify: `crates/harmony-pkarr/src/resolver.rs` (the two `verify_skew` call sites)

**Step 1 — Write the failing tests.** Replace the `verify_skew_accepts_within_window` and `verify_skew_rejects_outside_window` tests in `record.rs` with:

```rust
    fn signed(at: u64, valid_until: u64) -> PkarrRoutingRecord {
        let sk = SigningKey::generate(&mut OsRng);
        let identity_pub = fixture_identity_pubkey(&sk);
        PkarrRoutingRecord::sign_new(b"blob".to_vec(), identity_pub, at, valid_until, &sk)
            .expect("sign")
    }

    #[test]
    fn verify_freshness_accepts_old_record_within_ttl() {
        // Published 1 hour ago, TTL 7 days → resolvable (impossible under ±30min).
        let now = 10_000_000_000u64;
        let at = now - 60 * 60 * 1000;
        let rec = signed(at, at + 604_800_000);
        assert!(rec.verify_freshness(now).is_ok());
    }

    #[test]
    fn verify_freshness_rejects_expired() {
        let now = 10_000_000_000u64;
        let at = now - 8 * 24 * 60 * 60 * 1000; // 8 days ago
        let rec = signed(at, at + 604_800_000); // expired 1 day ago
        assert_eq!(rec.verify_freshness(now), Err(PkarrError::StaleOrSkewed));
    }

    #[test]
    fn verify_freshness_rejects_forged_future() {
        let now = 10_000_000_000u64;
        let at = now + FUTURE_TOLERANCE_MS + 1; // beyond skew allowance
        let rec = signed(at, at + 604_800_000);
        assert_eq!(rec.verify_freshness(now), Err(PkarrError::StaleOrSkewed));
    }

    #[test]
    fn verify_freshness_allows_small_future_skew() {
        let now = 10_000_000_000u64;
        let at = now + FUTURE_TOLERANCE_MS - 1; // within allowance
        let rec = signed(at, at + 604_800_000);
        assert!(rec.verify_freshness(now).is_ok());
    }
```

**Step 2 — Run, verify it fails.**
Run: `cargo nextest run -p harmony-pkarr --features test-fixtures -E 'test(verify_freshness)'`
Expected: compile error — no `verify_freshness`, no `FUTURE_TOLERANCE_MS`.

**Step 3 — Replace the constant + method.** In `record.rs`: replace `SKEW_TOLERANCE_MS` (line 36) with:

```rust
/// Clock-skew allowance for the future-strict lower bound. A record whose
/// `announced_at_ms` is more than this far in the future is rejected as forged.
pub const FUTURE_TOLERANCE_MS: u64 = 5 * 60 * 1000;
```

Replace `verify_skew` (`record.rs:92-101`) with:

```rust
    /// Freshness check (RPK4): reject a record that is forged-future
    /// (`announced_at_ms > now + FUTURE_TOLERANCE_MS`) or expired
    /// (`now > valid_until_ms`). The upper bound is the publisher's signed
    /// TTL, not a wall-clock window — so a valid record stays resolvable for
    /// its whole committed validity period regardless of republish cadence.
    pub fn verify_freshness(&self, now_ms: u64) -> Result<(), PkarrError> {
        if self.announced_at_ms > now_ms.saturating_add(FUTURE_TOLERANCE_MS) {
            return Err(PkarrError::StaleOrSkewed);
        }
        if now_ms > self.valid_until_ms {
            return Err(PkarrError::StaleOrSkewed);
        }
        Ok(())
    }
```

**Step 4 — Update resolver call sites.** In `resolver.rs`:
- In `resolve()` (~line 97): `record.verify_skew(now_ms)` → `record.verify_freshness(now_ms)`. Update the adjacent `tracing::warn!` message from "outside ±30min skew window — dropping (RPK4)" to "failed freshness (expired or forged-future) — dropping (RPK4)"; add `valid_until_ms = record.valid_until_ms` to the log fields.
- In `cache_get()` (~line 173): `rec.verify_skew(now_ms)` → `rec.verify_freshness(now_ms)`. Update the surrounding comment to describe TTL expiry rather than the +30 min edge.

**Step 5 — Run the crate tests.**
Run: `cargo nextest run -p harmony-pkarr --all-targets --features test-fixtures`
Expected: PASS.

**Step 6 — Commit.**
```bash
git add crates/harmony-pkarr/src/record.rs crates/harmony-pkarr/src/resolver.rs
git commit -m "feat(pkarr): replace ±30min skew with signed-TTL verify_freshness"
```

---

### Task A3: Surface BEP44 `seq` + in-memory anti-rollback highwater

> **As-shipped note (post bot-review, commit `8e8d07f`):** the highwater is a **bounded `LruCache<[u8;32], u64>` (cap 4096)** with `.put()` / `.get()`, NOT the `HashMap` / `.insert()` shown in the Step-4 code block below. Ephemeral keys rotate per epoch, so the keyspace is unbounded over time — an unbounded map would be a slow memory-growth / DoS vector. LRU eviction only drops best-effort rollback protection for stale keys (which already resets on reboot). Treat the LruCache form as authoritative; the `HashMap` blocks below are the original pre-review sketch.

**Files:**
- Modify: `crates/harmony-pkarr/src/wire.rs` (return `seq`; add explicit-`seq` test helper)
- Modify: `crates/harmony-pkarr/src/resolver.rs` (thread `seq`; bounded-LRU highwater + check)

**Step 1 — Write the failing test.** Add to the `tests` module in `resolver.rs`:

```rust
    #[tokio::test]
    async fn rejects_rolled_back_seq() {
        let relay = MockPkarrRelay::start().await;
        let pool = crate::relay::RelayPool::new(vec![relay.base_url.clone()]);
        let client = Arc::new(crate::relay::RelayClient::new(pool));
        let resolver = PkarrResolver::new(Arc::clone(&client));

        let ephemeral = SigningKey::generate(&mut OsRng);
        let vk = ephemeral.verifying_key();
        let identity_sk = SigningKey::generate(&mut OsRng);
        let identity_pub = fixture_identity_pubkey(&identity_sk);
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;
        let rec = PkarrRoutingRecord::sign_new(
            b"r".to_vec(), identity_pub, now, now + 604_800_000, &identity_sk,
        ).expect("sign");

        let key_z32 = crate::wire::z32_for_verifying_key(&vk).unwrap();

        // Publish with a HIGH seq, resolve → accepted, highwater = 200.
        let hi = crate::wire::build_relay_payload_with_seq(&ephemeral, &rec, 200).unwrap();
        client.put(&key_z32, &hi).await.unwrap();
        assert!(resolver.resolve(&vk).await.unwrap().is_some());

        // A relay now serves an OLDER seq for the same key → rolled back → drop.
        let lo = crate::wire::build_relay_payload_with_seq(&ephemeral, &rec, 100).unwrap();
        client.put(&key_z32, &lo).await.unwrap();
        // Bypass the positive cache to force a fresh GET + seq check.
        resolver.invalidate_for_test(&vk.to_bytes());
        assert!(
            resolver.resolve(&vk).await.unwrap().is_none(),
            "older seq must be rejected as rollback"
        );
    }
```

**Step 2 — Run, verify it fails.**
Run: `cargo nextest run -p harmony-pkarr --features test-fixtures -E 'test(rejects_rolled_back_seq)'`
Expected: compile error — no `build_relay_payload_with_seq`, no `invalidate_for_test`.

**Step 3 — Surface `seq` from the wire layer.** In `wire.rs`, change `parse_relay_payload` to return the seq, and add the test helper:

```rust
/// Parse + outer-verify a relay GET payload. Returns the record plus the
/// BEP44 `seq` (the SignedPacket timestamp, µs since epoch — DHT-CAS-monotonic
/// at PUT time), which the resolver uses for in-memory anti-rollback.
pub(crate) fn parse_relay_payload(
    expected_pk: &[u8; 32],
    payload: &[u8],
) -> Result<(PkarrRoutingRecord, u64), PkarrError> {
    let pk = PublicKey::try_from(expected_pk).map_err(|_| PkarrError::InvalidRecord)?;
    let bytes = bytes::Bytes::copy_from_slice(payload);
    let signed = SignedPacket::from_relay_payload(&pk, &bytes)
        .map_err(|_| PkarrError::OuterSignatureInvalid)?;
    let seq = signed.timestamp().as_u64();
    let txt = read_record_txt(&signed)?;
    let record = txt_string_to_routing_record(&txt)?;
    Ok((record, seq))
}

/// Test-only: build a relay payload with an explicit BEP44 `seq`, so
/// anti-rollback tests can author deterministic older/newer records.
#[cfg(any(test, feature = "test-fixtures"))]
pub fn build_relay_payload_with_seq(
    signing_key: &SigningKey,
    record: &PkarrRoutingRecord,
    seq: u64,
) -> Result<Vec<u8>, PkarrError> {
    use pkarr::Timestamp;
    let keypair = Keypair::from_secret_key(&signing_key.to_bytes());
    let b64 = routing_record_to_txt_string(record)?;
    let mut txt = TXT::new();
    for chunk in b64.as_bytes().chunks(255) {
        let cs = CharacterString::new(chunk)
            .map_err(|_| PkarrError::SerializeError("txt char-string"))?;
        txt.add_char_string(cs);
    }
    let name = Name::new(RECORD_LABEL).map_err(|_| PkarrError::SerializeError("txt name"))?;
    let signed = SignedPacket::builder()
        .txt(name, txt, RECORD_TTL)
        .timestamp(Timestamp::from(seq))
        .sign(&keypair)
        .map_err(|_| PkarrError::SerializeError("pkarr signed-packet build"))?;
    Ok(signed.to_relay_payload().to_vec())
}
```

Update the in-`wire.rs` test callers of `parse_relay_payload` (lines ~157, ~209, ~219, ~231, ~273) to destructure: `let (parsed, _seq) = parse_relay_payload(...)?;` and compare `parsed` (the `parse_rejects_*` tests compare against an `Err`, so `assert_eq!(parse_relay_payload(...), Err(...))` becomes a tuple-vs-Err comparison — change those to `.map(|(r, _)| r)` or assert `.is_err()` / match the `Err`). For the `Err` assertions, use:
`assert_eq!(parse_relay_payload(&kp.public_key().to_bytes(), &payload).err(), Some(PkarrError::InvalidRecord));`

**Step 4 — Add the highwater map + check + test seam in the resolver.** In `resolver.rs`:

Add to the struct and `new`:
```rust
use std::collections::HashMap;
// ...
pub struct PkarrResolver {
    relay: Arc<RelayClient>,
    cache: Arc<Mutex<LruCache<[u8; 32], CachedResolution>>>,
    seq_highwater: Arc<Mutex<HashMap<[u8; 32], u64>>>,
}
// in new():
            seq_highwater: Arc::new(Mutex::new(HashMap::new())),
```

In `resolve()`, change the parse destructure and add the seq gate **after** `verify_freshness` passes (so a malformed/expired record never poisons the highwater):
```rust
                let (record, seq) = match crate::wire::parse_relay_payload(&pk_bytes, &envelope) {
                    Ok(r) => r,
                    Err(e) => { /* unchanged RPK1 warn + negative-cache + return Ok(None) */ }
                };
                // ... unchanged verify_inner_sig (RPK2) ...
                // ... unchanged verify_freshness (RPK4) ...
                // RPK5: in-memory anti-rollback. Reject a strictly-older seq than
                // the highest accepted for this key (a relay replaying a stale,
                // still-within-TTL record). Equal seq = same signed bytes → allow
                // (idempotent re-resolve after cache expiry). Update on accept.
                {
                    let mut hw = self.seq_highwater.lock().expect("seq_highwater poisoned");
                    if let Some(&seen) = hw.get(&pk_bytes) {
                        if seq < seen {
                            tracing::warn!(
                                key = %key_z32, seq, seen,
                                "pkarr record seq rolled back — dropping (RPK5)"
                            );
                            self.cache_put(pk_bytes, None, NEGATIVE_CACHE_TTL);
                            return Ok(None);
                        }
                    }
                    hw.insert(pk_bytes, seq);
                }
                self.cache_put(pk_bytes, Some(record.clone()), POSITIVE_CACHE_TTL);
                Ok(Some(record))
```

Add a test-only cache-invalidation seam:
```rust
    /// Test-only: drop any cached resolution for a key, forcing the next
    /// `resolve` to hit the relay (so seq anti-rollback can be exercised).
    #[cfg(any(test, feature = "test-fixtures"))]
    pub fn invalidate_for_test(&self, pk: &[u8; 32]) {
        self.cache.lock().expect("cache poisoned").pop(pk);
    }
```

**Step 5 — Run the crate tests.**
Run: `cargo nextest run -p harmony-pkarr --all-targets --features test-fixtures`
Expected: PASS.

**Step 6 — Commit.**
```bash
git add crates/harmony-pkarr/src/wire.rs crates/harmony-pkarr/src/resolver.rs
git commit -m "feat(pkarr): surface BEP44 seq + in-memory anti-rollback highwater"
```

---

### Task A4: Diverse-relay seam — `get_all` + `resolve_freshest`

**Files:**
- Modify: `crates/harmony-pkarr/src/relay.rs` (add `get_all`)
- Modify: `crates/harmony-pkarr/src/resolver.rs` (add `resolve_freshest` / `resolve_window_freshest`)

This is the core half of §3.4. On a dial failure the client re-resolves with these (Part B Task B4): they bypass the positive cache, query **all** available relays, and return the **freshest** valid record by `seq`. A single stale relay loses to any relay holding a newer record.

**Step 1 — Write the failing test.** Add to the `tests` module in `resolver.rs`:

```rust
    #[tokio::test]
    async fn resolve_freshest_beats_stale_relay() {
        // Two relays: one holds an OLD seq, one holds a NEW seq. resolve_freshest
        // must return the new one regardless of pool order.
        let stale = MockPkarrRelay::start().await;
        let fresh = MockPkarrRelay::start().await;
        let pool = crate::relay::RelayPool::new(vec![
            stale.base_url.clone(), // stale FIRST — first-hit would pick this
            fresh.base_url.clone(),
        ]);
        let client = Arc::new(crate::relay::RelayClient::new(pool));
        let resolver = PkarrResolver::new(Arc::clone(&client));

        let ephemeral = SigningKey::generate(&mut OsRng);
        let vk = ephemeral.verifying_key();
        let id_sk = SigningKey::generate(&mut OsRng);
        let id_pub = fixture_identity_pubkey(&id_sk);
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH).unwrap().as_millis() as u64;
        let old_rec = PkarrRoutingRecord::sign_new(
            b"old".to_vec(), id_pub, now, now + 604_800_000, &id_sk).unwrap();
        let new_rec = PkarrRoutingRecord::sign_new(
            b"new".to_vec(), id_pub, now, now + 604_800_000, &id_sk).unwrap();

        let key_z32 = crate::wire::z32_for_verifying_key(&vk).unwrap();
        // Put directly to each relay via single-relay clients.
        let stale_c = crate::relay::RelayClient::new(
            crate::relay::RelayPool::new(vec![stale.base_url.clone()]));
        let fresh_c = crate::relay::RelayClient::new(
            crate::relay::RelayPool::new(vec![fresh.base_url.clone()]));
        stale_c.put(&key_z32,
            &crate::wire::build_relay_payload_with_seq(&ephemeral, &old_rec, 100).unwrap())
            .await.unwrap();
        fresh_c.put(&key_z32,
            &crate::wire::build_relay_payload_with_seq(&ephemeral, &new_rec, 200).unwrap())
            .await.unwrap();

        let got = resolver.resolve_freshest(&vk).await.unwrap().expect("present");
        assert_eq!(got.routing_blob, b"new", "freshest-by-seq must win");
    }
```

**Step 2 — Run, verify it fails.**
Run: `cargo nextest run -p harmony-pkarr --features test-fixtures -E 'test(resolve_freshest_beats_stale_relay)'`
Expected: compile error — no `resolve_freshest`, no `get_all`.

**Step 3 — Add `get_all` to `RelayClient`.** In `relay.rs`, after `get` (~line 304), add a variant that does NOT stop at the first hit:

```rust
    /// Query every available (non-cooldown) relay and return each
    /// `(relay_url, envelope)` that served a record. Unlike `get` (first-hit),
    /// this surfaces all hits so the caller can pick the freshest — defeating a
    /// single stale relay. Returns `Err(NoRelaysAvailable)` only if every relay
    /// is on cooldown; an empty Vec means "all reachable relays returned 404".
    pub async fn get_all(&self, key_z32: &str) -> Result<Vec<(String, Vec<u8>)>, PkarrError> {
        let (gen, relays) = self.available_relays();
        if relays.is_empty() {
            return Err(PkarrError::NoRelaysAvailable);
        }
        let mut hits = Vec::new();
        for base in relays {
            let url = format!("{}/{}", base, key_z32);
            match self.http.get(&url).send().await {
                Ok(resp) if resp.status().is_success() => {
                    match resp.bytes().await {
                        Ok(b) => {
                            self.record_outcome(gen, &base, RelayOutcome::Success);
                            hits.push((base, b.to_vec()));
                        }
                        Err(_) => {
                            self.mark_cooldown(gen, &base);
                            self.record_outcome(gen, &base, RelayOutcome::Transport);
                        }
                    }
                }
                Ok(resp) if resp.status().as_u16() == 404 => continue,
                Ok(resp) if resp.status().as_u16() == 429 => {
                    self.mark_cooldown(gen, &base);
                    self.record_outcome(gen, &base, RelayOutcome::Http(429));
                }
                Ok(resp) => {
                    self.mark_cooldown(gen, &base);
                    self.record_outcome(gen, &base, RelayOutcome::Http(resp.status().as_u16()));
                }
                Err(e) => {
                    self.mark_cooldown(gen, &base);
                    self.record_outcome(
                        gen, &base,
                        if e.is_timeout() { RelayOutcome::Timeout } else { RelayOutcome::Transport },
                    );
                }
            }
        }
        Ok(hits)
    }
```

**Step 4 — Add `resolve_freshest` / `resolve_window_freshest` to the resolver.** In `resolver.rs`:

```rust
    /// Cache-bypassing resolve that queries ALL relays and returns the freshest
    /// valid record by BEP44 `seq` (ties broken by latest `announced_at` within
    /// TTL). Used on a dial failure to cross-check relays — a single stale relay
    /// cannot pin the resolver to an old-but-within-TTL record. Updates the
    /// positive cache + seq highwater with the winner.
    pub async fn resolve_freshest(
        &self,
        pk: &VerifyingKey,
    ) -> Result<Option<PkarrRoutingRecord>, PkarrError> {
        let pk_bytes = pk.to_bytes();
        let key_z32 = crate::wire::z32_for_verifying_key(pk)?;
        let hits = self.relay.get_all(&key_z32).await?;
        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("system clock < UNIX epoch is unsupported")
            .as_millis() as u64;

        let mut best: Option<(u64, PkarrRoutingRecord)> = None; // (seq, record)
        for (_relay, envelope) in hits {
            let (record, seq) = match crate::wire::parse_relay_payload(&pk_bytes, &envelope) {
                Ok(r) => r,
                Err(_) => continue,
            };
            if record.verify_inner_sig().is_err() {
                continue;
            }
            if record.verify_freshness(now_ms).is_err() {
                continue;
            }
            if best.as_ref().is_none_or(|(bseq, brec)| {
                seq > *bseq || (seq == *bseq && record.announced_at_ms > brec.announced_at_ms)
            }) {
                best = Some((seq, record));
            }
        }

        match best {
            Some((seq, record)) => {
                // Anti-rollback: never accept a winner older than what we've seen.
                {
                    let mut hw = self.seq_highwater.lock().expect("seq_highwater poisoned");
                    if let Some(&seen) = hw.get(&pk_bytes) {
                        if seq < seen {
                            return Ok(None);
                        }
                    }
                    hw.insert(pk_bytes, seq);
                }
                self.cache_put(pk_bytes, Some(record.clone()), POSITIVE_CACHE_TTL);
                Ok(Some(record))
            }
            None => Ok(None),
        }
    }

    /// `resolve_freshest` across the epoch-tolerance key window; returns the
    /// freshest valid record found for any key.
    pub async fn resolve_window_freshest(
        &self,
        keys: &[VerifyingKey],
    ) -> Result<Option<PkarrRoutingRecord>, PkarrError> {
        let futures = keys.iter().map(|pk| self.resolve_freshest(pk));
        let results = join_all(futures).await;
        let mut best: Option<PkarrRoutingRecord> = None;
        let mut any_err: Option<PkarrError> = None;
        for r in results {
            match r {
                Ok(Some(rec)) => {
                    if best.as_ref().is_none_or(|b| rec.announced_at_ms > b.announced_at_ms) {
                        best = Some(rec);
                    }
                }
                Ok(None) => {}
                Err(e) => any_err = Some(e),
            }
        }
        match (best, any_err) {
            (Some(rec), _) => Ok(Some(rec)),
            (None, Some(e)) => Err(e),
            (None, None) => Ok(None),
        }
    }
```

**Step 5 — Run the crate tests.**
Run: `cargo nextest run -p harmony-pkarr --all-targets --features test-fixtures`
Expected: PASS.

**Step 6 — Commit.**
```bash
git add crates/harmony-pkarr/src/relay.rs crates/harmony-pkarr/src/resolver.rs
git commit -m "feat(pkarr): get_all + resolve_freshest for diverse-relay re-resolve"
```

---

### Task A5: Core full gate

**Step 1 — fmt.** Run: `cargo fmt --all -- --check` (expected: clean). If it complains, run `cargo fmt --all` and re-check.

**Step 2 — clippy.** Run: `cargo clippy -p harmony-pkarr --locked --all-targets --features test-fixtures --no-deps -- -D warnings`
Expected: 0 warnings. (Watch for: unused `RelayOutcome` import, `is_none_or` MSRV, dead-code on the test-only helpers — they're feature-gated so should be fine.)

**Step 3 — full crate test.** Run: `cargo nextest run -p harmony-pkarr --locked --all-targets --features test-fixtures`
Expected: PASS.

**Step 4 — workspace sanity (the change is leaf, but confirm nothing else in core depends on the old signatures).** Run: `cargo check --locked --workspace --all-targets --features test-fixtures` (if the workspace doesn't define that feature at the top level, fall back to `cargo check --locked --workspace`). Expected: clean.

**Step 5 — Open PR1.** Push the branch and open the PR (ZEB-free title/body; spec/plan files may reference ZEB freely). Title: `pkarr signed-TTL freshness (core)`. Body summarizes: 4-tuple preimage + `valid_until_ms`; `verify_freshness` (future-strict + TTL); BEP44 seq surfacing + in-memory anti-rollback; `get_all` + `resolve_freshest`; fixture regen; hard cutover (no legacy fallback). Reference the spec + plan paths. **Do NOT self-merge** — Jake's gate; pushover at ready-to-merge; run the bot loop (Qodo+CodeAnt → address → one CodeRabbit final; never Greptile).

---

# PART B — harmony-client (PR2)

**Gated on PR1 merging.** Branch from latest `origin/main` (`git checkout -b pkarr-freshness-signed-ttl-client`). All `cargo` commands run from `src-tauri/`. Per-task gate: `cargo nextest run --locked -p harmony-app --lib --features test-fixtures` for lib-only tasks; reserve `--all-targets` for the final gate (relink cost — see CLAUDE.md). Full gate at the end: `cargo fmt --all -- --check` + `cargo clippy --locked --all-targets --features test-fixtures --no-deps -- -D warnings` + `cargo nextest run --locked --workspace --all-targets --features test-fixtures`.

---

### Task B1: Bump the `harmony-pkarr` dep to PR1's merge commit

**Files:** Modify: `src-tauri/Cargo.toml`

**Step 1.** Get PR1's squash-merge commit SHA on `harmony` `main` (from the merged PR).

**Step 2.** Update both `rev` pins to that SHA:
- Line 109: `harmony-pkarr = { git = "https://github.com/zeblithic/harmony.git", rev = "<NEW_SHA>" }`
- Line 204: `harmony-pkarr = { git = "https://github.com/zeblithic/harmony.git", rev = "<NEW_SHA>", features = ["test-fixtures"] }`

**Step 3.** Run: `cargo update -p harmony-pkarr --precise <NEW_SHA>` is not applicable for git deps — instead run `cargo check -p harmony-app` to re-resolve the git dep and update `Cargo.lock`.
Expected: it FAILS to compile — every `sign_new` now needs 5 args and `verify_skew` is gone. That is the intended signal that B2/B3 are required. (Commit the `Cargo.toml` + `Cargo.lock` change once B2/B3 make it green; do not commit a non-compiling tree.)

---

### Task B2: Thread `valid_until` into the four publisher closures

**Files:**
- Modify: `src-tauri/src/reachability_record.rs` (add the TTL constant)
- Modify: `src-tauri/src/pkarr_identity_publisher.rs` (line 57)
- Modify: `src-tauri/src/pkarr_community_publisher.rs` (line 53)
- Modify: `src-tauri/src/pkarr_friend_publisher.rs` (line 64)
- Modify: `src-tauri/src/pkarr_invite_publisher.rs` (line 185)

**Step 1 — Write the failing test.** Add to `pkarr_identity_publisher.rs` tests a check that the published record carries a `valid_until` one TTL ahead of `announced_at`. Resolve the published record via a `PkarrResolver` over the mock and assert:

```rust
    #[tokio::test]
    async fn published_record_carries_ttl() {
        use harmony_pkarr::PkarrResolver;
        let relay = MockPkarrRelay::start().await;
        let pool = RelayPool::new(vec![relay.base_url.clone()]);
        let client = Arc::new(RelayClient::new(pool));
        let publisher = Arc::new(PkarrPublisher::new(Arc::clone(&client)));
        let _ph = Arc::clone(&publisher).spawn();
        let resolver = PkarrResolver::new(Arc::clone(&client));

        let sk = SigningKey::generate(&mut OsRng);
        let id_pub = build_id_pub(&sk);
        // Derive the same epoch key the publisher will use, so we can resolve it.
        let pubr = PkarrIdentityPublisher::new(
            Arc::clone(&publisher), sk, id_pub, Arc::new(|| b"blob".to_vec()));
        pubr.enable().await;

        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH).unwrap().as_millis() as u64;
        let epoch_id = harmony_pkarr::current_epoch_id(now);
        let vk = harmony_pkarr::derive_ephemeral_key(
            harmony_pkarr::PkarrCase::Identity, &id_pub, &epoch_id.to_be_bytes(),
        ).verifying_key();

        let mut attempts = 0;
        loop {
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
            attempts += 1;
            assert!(attempts < 60, "resolve timed out");
            if let Some(rec) = resolver.resolve(&vk).await.expect("resolve") {
                assert_eq!(
                    rec.valid_until_ms,
                    rec.announced_at_ms + crate::reachability_record::REACHABILITY_RECORD_TTL_MS
                );
                return;
            }
        }
    }
```

**Step 2 — Run, verify it fails.**
Run: `cargo nextest run --locked -p harmony-app --lib --features test-fixtures -E 'test(published_record_carries_ttl)'`
Expected: compile error (no `REACHABILITY_RECORD_TTL_MS`; `sign_new` arity).

**Step 3 — Add the constant.** In `reachability_record.rs` (near the top, with the other consts):

```rust
/// Signed validity window applied to every reachability pkarr record
/// (identity / community / friend / invite). One epoch (7 days): covers the
/// ~3.5-day republish gap with margin and never outlives the per-epoch
/// ephemeral key. The resolver honors a record while `now <= valid_until`.
pub(crate) const REACHABILITY_RECORD_TTL_MS: u64 = 7 * 24 * 60 * 60 * 1000;
```

**Step 4 — Thread it into each closure.** In each of the four publishers, change the `sign_new` call inside the `RecordBuilder` to pass `at_ms + REACHABILITY_RECORD_TTL_MS` as the new `valid_until_ms` arg (4th positional, before the key). Import the constant where needed (`use crate::reachability_record::REACHABILITY_RECORD_TTL_MS;`). Examples:

- `pkarr_identity_publisher.rs:56-59` and `pkarr_invite_publisher.rs:184-186` and `pkarr_community_publisher.rs:52-54`:
```rust
        let builder: RecordBuilder = Arc::new(move |at_ms| {
            PkarrRoutingRecord::sign_new(
                blob_builder(),
                id_pub,
                at_ms,
                at_ms + REACHABILITY_RECORD_TTL_MS,
                &id_sk,
            )
            .expect("sign — fixed-size buffers should not fail")
        });
```
- `pkarr_friend_publisher.rs:57-65` (same, but the blob is `sealed` and key is `cd_key`):
```rust
            PkarrRoutingRecord::sign_new(sealed, id_pub, at_ms, at_ms + REACHABILITY_RECORD_TTL_MS, &cd_key)
```

**Step 5 — Run.** Run: `cargo nextest run --locked -p harmony-app --lib --features test-fixtures -E 'test(published_record_carries_ttl)'` → PASS.

**Step 6 — Commit.**
```bash
git add src-tauri/src/reachability_record.rs src-tauri/src/pkarr_*_publisher.rs
git commit -m "feat: publish reachability records with a 7-day signed TTL"
```

---

### Task B3: `verify_skew` → `verify_freshness` at all client call sites + `network_health` `sign_new`

**Files:**
- Modify: `src-tauri/src/pkarr_resolver_adapter.rs` (line 82)
- Modify: `src-tauri/src/lib.rs` (lines 39977, 41546, 43054)
- Modify: `src-tauri/src/network_health.rs` (line 1341 verify_skew; line 2560 sign_new)

**Step 1 — Enumerate.** Run from `src-tauri/`: `rg -n 'verify_skew|PkarrRoutingRecord::sign_new' src/` and confirm the sites match the list above (plus any test-only `sign_new` calls — update those too with a `+ 604_800_000` valid_until).

**Step 2 — Mechanical replace.** For each `verify_skew(now_ms)` → `verify_freshness(now_ms)` (identical arg; the record now carries the TTL it needs). For `network_health.rs:2560` `sign_new(...)` add the `valid_until` arg (`at_ms + 604_800_000` or the local `now + ...`).

**Step 3 — Build (lib).** Run: `cargo check --locked -p harmony-app --all-targets --features test-fixtures`
Expected: clean (this is where any missed `sign_new`/`verify_skew` site surfaces).

**Step 4 — Run the touched tests.** Run: `cargo nextest run --locked -p harmony-app --lib --features test-fixtures -E 'test(pkarr) + test(reachability) + test(network_health)'`
Expected: PASS.

**Step 5 — Commit.**
```bash
git add src-tauri/src/pkarr_resolver_adapter.rs src-tauri/src/lib.rs src-tauri/src/network_health.rs
git commit -m "feat: resolver honors signed TTL (verify_freshness) at all client sites"
```

---

### Task B4: Diverse-relay re-resolve on dial-fail (redeem + add-friend)

**Files:** Modify: `src-tauri/src/lib.rs` (redeem path `~40042-40073`; add-friend path `~43083-43114`)

The dial paths are one-shot today. Add a single retry that, on the first dial failure, re-resolves via `resolve_window_freshest` (cache-bypassing, all-relay), re-synthesizes the `EndpointAddr`, and dials once more before reporting unreachable/error.

**Step 1 — Write the failing test.** This path is hard to exercise end-to-end (needs an iroh endpoint). Write a **focused unit test** for an extracted helper instead: factor the "synthesize `EndpointAddr` from a `ReachabilityAnnouncePayload`/routing record" block (currently inline at ~40017 and ~43061) into a free function `fn endpoint_addr_from_routing(routing: &ReachabilityAnnouncePayload) -> Result<iroh::EndpointAddr, String>` and test it directly (valid node_id → Ok; malformed relay URL skipped; bad node_id → Err). Then the retry wiring reuses the helper. (Rationale: the retry control-flow is verified by the integration/fleet re-test in the DoD; the unit test pins the reusable synthesis logic. Note this split explicitly in the PR description — no silent coverage gap.)

```rust
    #[test]
    fn endpoint_addr_from_routing_parses_node_id() {
        let mut routing = ReachabilityAnnouncePayload::default();
        routing.iroh_node_id = /* valid 32-byte ed25519 node id */;
        routing.home_relay_url = "https://relay.example/".to_string();
        let addr = endpoint_addr_from_routing(&routing).expect("ok");
        assert_eq!(addr.id, iroh::EndpointId::from_bytes(&routing.iroh_node_id).unwrap());
    }
    #[test]
    fn endpoint_addr_from_routing_skips_malformed_relay() {
        let mut routing = ReachabilityAnnouncePayload::default();
        routing.iroh_node_id = /* valid node id */;
        routing.home_relay_url = "not a url".to_string();
        assert!(endpoint_addr_from_routing(&routing).is_ok()); // malformed relay skipped, not fatal
    }
```
(Fill the `iroh_node_id` with a real verifying-key byte array from a generated `SigningKey`, mirroring the existing tests.)

**Step 2 — Run, verify it fails.** `cargo nextest run --locked -p harmony-app --lib --features test-fixtures -E 'test(endpoint_addr_from_routing)'` → compile error (no such fn).

**Step 3 — Extract the helper.** Lift the inline `EndpointAddr` synthesis (node_id decode + `with_relay_url` on parse-ok + `with_ip_addr` loop) from both sites into `endpoint_addr_from_routing`. Replace both inline blocks with calls to it.

**Step 4 — Add the retry.** In the redeem path, wrap the connect so a failure re-resolves once. Sketch (preserve the existing timeout/`dial_config` + log tags):

```rust
    let conn = match dial_once(&iroh_endpoint, &alice_addr, &dial_config).await {
        Ok(c) => c,
        Err(first) => {
            tracing::warn!(error = %first, "dial failed; re-resolving freshest across relays");
            match resolver.resolve_window_freshest(&verifying_keys).await {
                Ok(Some(rec2)) => {
                    if let Ok(routing2) =
                        ciborium::from_reader::<ReachabilityAnnouncePayload, _>(rec2.routing_blob.as_slice())
                    {
                        if let Ok(addr2) = endpoint_addr_from_routing(&routing2) {
                            match dial_once(&iroh_endpoint, &addr2, &dial_config).await {
                                Ok(c) => c,
                                Err(_) => return Ok(RedemptionOutcome {
                                    status: "inviter_unreachable".to_string(), community_id: None }),
                            }
                        } else { return Ok(RedemptionOutcome {
                            status: "inviter_unreachable".to_string(), community_id: None }); }
                    } else { return Ok(RedemptionOutcome {
                        status: "inviter_unreachable".to_string(), community_id: None }); }
                }
                _ => return Ok(RedemptionOutcome {
                    status: "inviter_unreachable".to_string(), community_id: None }),
            }
        }
    };
```
Where `dial_once` is a small local async closure/fn wrapping the existing `tokio::time::timeout(dial_config.connect_timeout, iroh_endpoint.inner().connect(addr, HARMONY_HANDSHAKE_V1))` and mapping timeout/`Err` to a single `Err`. Apply the analogous retry in the add-friend path (which returns `Err(String)` on failure rather than the `inviter_unreachable` outcome — preserve that path's error contract).

**Step 5 — Build + test.** `cargo check --locked -p harmony-app --all-targets --features test-fixtures` (clean) then the `endpoint_addr_from_routing` tests → PASS.

**Step 6 — Commit.**
```bash
git add src-tauri/src/lib.rs
git commit -m "feat: re-resolve freshest across relays on dial failure"
```

---

### Task B5: Gate fail-loud (§3.7, minor)

**Files:** Modify: `src-tauri/src/pkarr_settings.rs` (`load_or_default`, ~line 193)

**Step 1 — Write the failing test.** Pin that a **parse error** is logged loudly and is distinguishable from an explicit opt-out, while still failing **closed** (never fail-open to discoverable). Since `load_or_default` returns a value (not a Result), add a sibling `load_or_default_logged` (or have `load_or_default` emit a `tracing::error!` on the parse-error branch) and assert behavior via a captured-log test or by splitting parse from default:

```rust
    #[test]
    fn parse_error_fails_closed_not_open() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("connectivity-settings.json");
        std::fs::write(&path, b"{ this is not valid json").unwrap();
        let s = PkarrSettings::load_or_default(&path);
        // Privacy-fail-open is worse than a freeze: a parse error must NOT
        // silently enable discoverability.
        assert!(!s.identity_discoverable);
    }
```

**Step 2 — Run, verify it passes-or-fails.** If `load_or_default` already returns `Default { identity_discoverable: false }` on parse error, this test passes immediately — that confirms fail-closed is intact. The change is to **add the loud log** on the parse-error branch (distinct from missing-file): `tracing::error!(path = %path.display(), error = %e, "connectivity-settings.json failed to parse — failing closed (not discoverable); a prior opt-in will not take effect until fixed");`. Keep missing-file as a quiet default.

**Step 3 — Build + test.** Lib test run → PASS.

**Step 4 — Commit.**
```bash
git add src-tauri/src/pkarr_settings.rs
git commit -m "feat: load_or_default logs loudly on parse error (fail-closed, never fail-open)"
```

---

### Task B6: Client fixture regen

**Files:** Modify: any pinned pkarr-record fixtures under `src-tauri/tests/`

**Step 1 — Locate.** Run from `src-tauri/`: `rg -n 'PkarrRoutingRecord|pkarr.*fixture|expected.*=.*"a[0-9]' tests/` to find any hex-pinned pkarr record fixtures (the client mirrors core wire pins).

**Step 2 — Regenerate.** For each, add the `valid_until` arg to the `sign_new` call (fixed value for determinism), run the test, copy the actual hex from the failure diff into `expected`, re-run → PASS. If no client-side pkarr-record hex pin exists, record that in the commit message and skip.

**Step 3 — Commit.**
```bash
git add src-tauri/tests/
git commit -m "test: regen pkarr-record wire fixtures for signed-TTL format"
```

---

### Task B7: Client full gate + PR

**Step 1 — fmt.** `cargo fmt --all -- --check` (fix with `cargo fmt --all` if needed).

**Step 2 — clippy.** `cargo clippy --locked --all-targets --features test-fixtures --no-deps -- -D warnings` → 0.

**Step 3 — full test.** `cargo nextest run --locked --workspace --all-targets --features test-fixtures` → PASS. (macOS: ensure Developer-Tools terminal entitlement is on, else cold link hangs — see CLAUDE.md.)

**Step 4 — frontend (no TS change expected, but CI runs it).** From repo root: `npx tsc --noEmit && npx vitest run` → PASS.

**Step 5 — Open PR2.** ZEB-free title `pkarr signed-TTL freshness (client)`; body references the spec + plan, notes the dep bump, lists the changes, and explicitly flags the B4 unit-vs-integration coverage split. **Do NOT self-merge**; bot loop; pushover at ready-to-merge.

**Step 6 — Fleet re-test (DoD).** On ZEB-477, coordinate the A-side **delayed** first-contact re-test: publish, wait **>30 min**, then redeem → must resolve (the facet-A win, impossible under the old ±30 min guard). This is the empirical close-out of ZEB-516 facet-A and the async-DM precondition.

---

## Self-Review

**Spec coverage:** §3.1 signed `valid_until` → A1. §3.2 future-strict → A2. §3.3 seq anti-rollback → A3. §3.4 diverse-relay → A4 (core seam) + B4 (dial retry). §3.5 TTL durations → B2 (7d constant). §3.6 hard cutover → A1 (no `serde(default)`; preimage change). §3.7 gate fail-loud → B5. §5 cross-repo split → Part A / Part B + B1 dep bump. §6 testing → per-task TDD steps. §8 DoD → B7 Step 6 fleet re-test.

**Deliberate refinements of the spec (flagged for Jake):**
1. **`seq < highwater` (strict) instead of `<=`.** The spec §3.3 wrote `<=`; that would reject a legitimate idempotent re-resolve of the *same* record after the 15-min positive cache expires (same `seq` == highwater). Strict `<` rejects only strictly-older replays (the actual rollback threat) and accepts equal-`seq` (identical signed bytes — harmless).
2. **Diverse-relay realized as "query-all, freshest-by-seq" (`get_all` + `resolve_freshest`).** The spec assumed the resolver "already iterates a relay pool"; the code shows `RelayClient.get()` is first-hit and the resolver has no relay provenance. "Query all, take freshest" achieves the same security property (a single stale relay cannot pin the resolver) without provenance bookkeeping — simpler and strictly stronger (it consults *every* relay, not just "a different one").
3. **B4 coverage split.** The dial-retry control flow is verified by the DoD fleet re-test, not a unit test (it needs a live iroh endpoint); the reusable `endpoint_addr_from_routing` synthesis is unit-tested. Called out so it isn't a silent gap.

**Type/signature consistency:** `sign_new(routing_blob, harmony_identity_pub, announced_at_ms, valid_until_ms, identity_signing_key)` — used identically in A1, A2 tests, B2, B3. `verify_freshness(now_ms)` replaces `verify_skew(now_ms)` everywhere (same arity). `parse_relay_payload` → `(PkarrRoutingRecord, u64)` consumed in A3 (resolve), A4 (resolve_freshest), and wire tests. `REACHABILITY_RECORD_TTL_MS` defined once (B2) and imported by the four publishers + B3 test sites.

**No placeholders:** every code step shows real code; the only `<NEW_SHA>` is PR1's merge commit, which is unknowable until PR1 merges (B1), and the only `/* valid node id */` in B4 Step 1 is filled from a generated `SigningKey` as the surrounding tests already do.
