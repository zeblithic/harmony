# ZEB-382: Real pkarr wire format (pkarr-as-codec) — Design Spec

**Date:** 2026-06-05
**Ticket:** ZEB-382 (Urgent) — filed from the fleet smoke test, related to ZEB-330 / ZEB-321 Phase 2 discovery.
**Repo:** `harmony` (crate `harmony-pkarr`). Downstream `harmony-client` consumes via a pinned git rev.
**Status:** Approved design, pending implementation plan.

---

## 1. Goal

Make `harmony-pkarr` publish and resolve records that a **real** pkarr relay (`relay.pkarr.org`) accepts, so first-contact discovery (Case-A invite, Case-B identity, Case-C community, Case-D friend rendezvous) works on real hardware over the public relay. Today the crate speaks an in-house dialect that every real relay rejects with HTTP 400.

Success criterion: a record published by one node against `relay.pkarr.org` is resolvable + verifiable by a second node on different hardware, completing an invite redeem.

## 2. Background — root cause (verified 2026-06-05)

`harmony-pkarr`'s publish/resolve path was only ever validated against the in-process `MockPkarrRelay`, which accepts arbitrary keys and bytes. Against a real relay it fails because it diverges from the pkarr/BEP44 wire format in three places, all self-documented in the source as deferred "Phase 3 hardening":

1. **Key encoding.** `publisher.rs::z32_encode_public()` returns `hex::encode(pk)` (64 chars). Real relays require **z-base-32** (52 chars). The relay cannot parse the URL key and returns **400 before reading the body**. (`resolver.rs` mirrors the hex encoding.)
2. **Sequence endianness.** The envelope writes `seq.to_le_bytes()` (little-endian); BEP44 mandates **big-endian**. (`resolver.rs` reads `from_le_bytes` to match.) The BEP44 *signing material* `3:seqi{seq}e1:v{len}:` is already correct — only the on-wire envelope diverges.
3. **Payload `v`.** The payload is `PkarrRoutingRecord` canonical CBOR. Real pkarr relays parse `v` as a **compressed DNS packet** and reject anything else.

### 2.1 Relationship to ZEB-381 (the TLS fix)

ZEB-381 (merged: harmony #272 → `2f9541f`, harmony-client #193) was a **prerequisite, not a duplicate**. The two are sequential gates:

- **Gate 1 (TLS, fixed by ZEB-381):** `rustls-tls-native-roots` rejected the relay's certificate at the TLS handshake (`InvalidCertificate(UnknownIssuer)`), surfacing as a transport error → `NoRelaysAvailable`. We never reached an HTTP status.
- **Gate 2 (wire format, this ticket):** with TLS fixed, we now reach the relay and it returns **400** because of the three divergences in §2 (items 1–3).

Both gates produce the identical user-facing symptom: `RelayClient::get()` maps a 400 to the `Ok(_)` non-success arm → marks the relay on a 30 s cooldown → returns `NoRelaysAvailable`. That identical symptom string before and after ZEB-381 is why the TLS fix appeared inert; it had in fact advanced us from gate 1 to gate 2.

## 3. Approach

**Adopt the upstream `pkarr` crate (v6.0.1) as a no-network wire-format codec, and keep our own relay HTTP client.**

We use `pkarr` *only* for: deriving the z-base-32 key, building/signing the BEP44 `SignedPacket` (correct big-endian seq + DNS-packet `v`), and serializing/parsing the relay payload. We keep `harmony-pkarr::relay::RelayClient` (our reqwest client carrying the ZEB-381 webpki fix, the relay pool, and the 30 s cooldown) to perform the actual HTTP PUT/GET.

### 3.1 Why this approach

The root cause is a pure **serialization** defect. The minimal correct fix replaces exactly the serialization layer with an audited implementation and changes nothing else: our TLS fix, relay-pool/cooldown logic, two-layer signature model, and the entire `harmony-client` call surface all survive. `pkarr` enters as a codec dependency, not a framework.

### 3.2 Alternatives rejected

- **B — adopt pkarr's full relay `Client`.** Rejected. `pkarr::ClientBuilder` (v6.0.1) exposes **no** way to inject a custom `reqwest::Client`, TLS config, or extra roots (verified: the full builder method set has no such hook). It would therefore **discard the ZEB-381 webpki fix** with no override, re-inheriting the same platform-verifier TLS family that blocked us — and would demand a fresh cross-platform TLS spike plus a second `reqwest`/`url`/`futures-buffered` in the tree, for less control.
- **C — hand-roll real pkarr (no crate).** Rejected. Re-implements audited, security-sensitive wire code (z-base-32, BEP44, DNS-packet encode/decode) for zero benefit now that the codec is a public one-liner.

### 3.3 Decentralization posture (scope boundary)

ZEB-382 stays **relay-only**, matching the current architecture. The single-relay SPOF and any Mainline-DHT adoption (the genuinely decentralized substrate) are out of scope here and remain tracked under **ZEB-380**. We deliberately do *not* pull pkarr's `dht`/`mainline` stack.

## 4. Architecture

The public API of `harmony-pkarr` is unchanged. Only the bytes on the wire change.

```
PUBLISH:
  derive_ephemeral_key(case, ikm, info)         // derive.rs — UNCHANGED → ed25519_dalek::SigningKey
    -> Keypair::from_secret_key(&signing.to_bytes())   // pkarr; same ed25519 pubkey
    -> SignedPacket builder: one TXT record "_r" = base64url(record.to_canonical_cbor())
    -> sign with Keypair                          // pkarr: correct seq(BE) + DNS-packet v
    -> signed.to_relay_payload()  ── PUT via RelayClient (webpki + pool) ──▶ relay/<z32>

RESOLVE:
  z32(expected ephemeral pubkey) ── GET via RelayClient ──▶ relay response bytes
    -> SignedPacket::from_relay_payload(&public_key, &bytes)   // verifies OUTER (ephemeral) sig
    -> read TXT "_r" -> base64url-decode -> PkarrRoutingRecord::from_canonical_cbor
    -> verify_inner_sig + verify_skew + verify_identity_match   // record.rs — UNCHANGED
```

### 4.1 Module-level changes

| Module | Action |
|---|---|
| `derive.rs` | **Unchanged.** Same HKDF → `SigningKey`; the pinned reference vectors remain valid because we change only the *encoding* of the public key (z-base-32 instead of hex), not the derived key bytes. |
| `record.rs` | **Unchanged.** `PkarrRoutingRecord` (canonical CBOR + harmony inner signature) now rides inside a DNS TXT record. |
| `relay.rs` | **Unchanged.** `RelayClient`/`RelayPool` (reqwest + ZEB-381 webpki + pool + cooldown) already PUT/GET opaque bytes at `/{key}`. |
| `epoch.rs` | **Unchanged.** |
| `error.rs` | Add error mapping for pkarr verify/build failures; add a `RecordTooLarge` variant for the size guard. |
| `publisher.rs` | **Rewrite internals.** Delete `wrap_bep44_envelope` and `z32_encode_public`. Add: build `pkarr::Keypair` from the derived seed; build/sign a `SignedPacket`; `to_relay_payload()`; PUT via `RelayClient`. `PkarrPublisher`, `RecordBuilder`, `EphemeralKeyBuilder` public signatures unchanged. |
| `resolver.rs` | **Rewrite internals.** Delete `parse_and_verify`. Add: z-base-32 key for the GET URL; `SignedPacket::from_relay_payload`; read the TXT record; decode the CBOR; run the existing `record.rs` verifications. `PkarrResolver` public signature unchanged. |
| `testing.rs` | **Upgrade `MockPkarrRelay`** to the real format (see §6). |

`harmony-client`: **no code change** — only a `harmony-pkarr` rev-bump (as in ZEB-381 #193) plus a regenerated `Cargo.lock`.

### 4.2 Key consistency invariant

`pkarr::Keypair::from_secret_key(&seed)` and `ed25519_dalek::SigningKey::from_bytes(&seed)` derive the **same** ed25519 public key (pkarr uses ed25519-dalek internally; both apply the standard seed expansion). The z-base-32 string is base32 of those same 32 public-key bytes. This is asserted by a dedicated unit test (§6) and is what guarantees the `derive.rs` reference vectors stay valid.

## 5. Record → DNS TXT mapping

- **One TXT record**, fixed short label `_r`, value = **base64url-unpadded(`PkarrRoutingRecord::to_canonical_cbor()`)**. base64url because pkarr's TXT builder expects ASCII/text and our CBOR is binary.
- A helper pair `routing_record_to_txt(&PkarrRoutingRecord) -> Result<String, PkarrError>` / `txt_to_routing_record(&str) -> Result<PkarrRoutingRecord, PkarrError>` owns the base64 encode/decode and the DNS 255-byte character-string chunk/join, with round-trip tests.
- **Size budget.** The record is ~270–340 B CBOR → ~360–450 B base64url; the full DNS packet lands ~500 B against pkarr's ~1000 B `v` budget. The encoder **hard-checks the built packet is `< SignedPacket::MAX_BYTES`** and returns `PkarrError::RecordTooLarge` rather than letting an oversized future `routing_blob` fail opaquely at the relay.
- **TTL** is set to a small fixed value (300 s) and is not load-bearing: harmony freshness is governed by `PkarrRoutingRecord::verify_skew` (`SKEW_TOLERANCE_MS`) and epoch rotation, not DNS TTL.

## 6. Two signature layers + testing

The two-layer model is preserved exactly:

- **Outer signature** = pkarr `SignedPacket`, signed by the ephemeral (HKDF-derived) key. Verified by the relay and by `SignedPacket::from_relay_payload`.
- **Inner signature** = `PkarrRoutingRecord::inner_sig`, binding `(routing_blob, harmony_identity_pub, announced_at_ms)` to the publisher's harmony Ed25519 identity. Verified by the unchanged `record.rs` methods after decode.

Adopting pkarr fixes only the outer-layer encoding; identity authenticity is untouched.

### 6.1 Testing strategy

This bug existed because the suite only ever spoke to a permissive mock. The fix closes that gap:

1. **`MockPkarrRelay` upgraded to the real format.** On PUT, it parses the z-base-32 key from the URL and calls `SignedPacket::from_relay_payload` to validate the body — so it now **rejects the old in-house dialect**, and every default-CI test exercises *real* pkarr wire bytes through the round-trip.
2. **Unit tests** (default CI gate, no network):
   - pkarr-`Keypair` public key ≡ our `SigningKey` verifying key for the same seed (the §4.2 invariant).
   - `routing_record_to_txt` → `txt_to_routing_record` round-trip, including a value that exceeds one 255-byte DNS character-string.
   - oversize `routing_blob` → `PkarrError::RecordTooLarge`.
   - `from_relay_payload` rejects a tampered payload / public-key mismatch.
   - publish→(mock)→resolve end-to-end for at least one case, asserting inner-sig + skew + identity verification all pass.
3. **One gated real-relay test** against `relay.pkarr.org`: `#[ignore]` **and** env-gated (`HARMONY_PKARR_LIVE_RELAY=1`), kept **out of the default CI gate** because of network flakiness, but runnable on demand and in release smoke. It publishes then resolves a record over the live relay. This is the "can never silently regress to mock-only" guard, made CI-safe; it is also the cross-machine smoke test in automated form.

Wire-format fixtures: the existing `PkarrRoutingRecord` canonical-CBOR fixtures remain (the record type is unchanged). The `SignedPacket` envelope is **not** byte-pinnable (it embeds a non-deterministic timestamp + signature), so the outer layer is validated by round-trip + the existing record-CBOR fixture rather than by envelope byte-identity.

## 7. Dependencies

- Add to the workspace: `pkarr = { version = "3.10.0", default-features = false, features = ["signed_packet"] }` — a **no-network** dependency providing `Keypair`, `SignedPacket`, the builder, and `to_relay_payload`/`from_relay_payload`. The `signed_packet` feature pulls **no `reqwest`, no `mainline`, no `url`**. We use **3.10.0 specifically because the workspace already locks it transitively (via iroh)** — reusing the already-compiled crate and avoiding a second pkarr major version in the tree (which bot reviewers reliably flag). pkarr 3.10.0 declares no `rust-version`; edition 2021, no conflict with our 1.88 floor. `pkarr` and `harmony-pkarr` resolve to the same single `ed25519-dalek` 2.x, so `Keypair::from_secret_key` and our `SigningKey::from_bytes` share a code path — the key-consistency invariant (§4.2) is exact, not approximate.
- **Verified (2026-06-05) against pkarr 3.10.0:** `SignedPacket::to_relay_payload(&self) -> Bytes` and `SignedPacket::from_relay_payload(&PublicKey, &Bytes) -> Result<SignedPacket, SignedPacketVerifyError>` are public; plus `Keypair::from_secret_key(&[u8;32])`, `Keypair::to_z32()`, `PublicKey::try_from(&[u8;32])`, `SignedPacket::builder().txt(Name, TXT, ttl).sign(&keypair)`, `resource_records(name)`, and `SignedPacket::MAX_BYTES = 1104` — all under `signed_packet`. Implementation Task 1 re-confirms with a compiling in-memory round-trip + `cargo tree` (single pkarr 3.10.0).
- `harmony-pkarr` keeps `reqwest` (relay.rs), `ciborium`, `ed25519-dalek`, `serde_bytes`, `hkdf`, `sha2` (record/derive).

### 7.1 Error mapping

- `SignedPacket::from_relay_payload` error → `PkarrError::OuterSignatureInvalid` (RPK1).
- SignedPacket build / TXT encode error → `PkarrError::SerializeError(..)`.
- base64url decode / missing `_r` TXT / malformed CBOR → `PkarrError::InvalidRecord` (or `RelayResponseInvalid` for a structurally absent record).
- packet over `MAX_BYTES` → `PkarrError::RecordTooLarge` (new).

## 8. Backward compatibility and migration

**Hard cutover, no migration.** Nothing is deployed in the old dialect: every prior publish 400'd, so no real records exist on `relay.pkarr.org` in our format, and `MockPkarrRelay` is test-only. Both smoke-test machines rebuild from source. Therefore:

- `derive.rs` reference vectors stay valid (same derived keys).
- Existing `PkarrRoutingRecord` CBOR wire fixtures stay valid (record type unchanged).
- No wire-version negotiation is needed; going forward, the pkarr `SignedPacket` format **is** the harmony-pkarr relay wire format.

## 9. Rollout

1. **harmony PR** (this spec + the harmony-pkarr change) → review → merge to `main` (new SHA).
2. **harmony-client rev-bump PR** — bump the `harmony-pkarr` git `rev` to the new harmony `main` SHA (both the prod dep and the `test-fixtures` dev-dep), regenerate `Cargo.lock`. No harmony-client source change.
3. **Cross-machine milestone** — the gated real-relay test, run live: one node publishes a Case-A invite record, the second node resolves + redeems.

## 10. Security considerations

- The identity-binding inner signature and its verification (`verify_inner_sig` / `verify_identity_match` / `verify_skew`) are unchanged; the change does not weaken authenticity.
- Adopting pkarr's outer BEP44 signature replaces a hand-rolled signing/verification path with an audited one — a net reduction in security-sensitive bespoke code.
- The relay payload still contains only the opaque `routing_blob` plus the public identity key and signatures already published by design (Case-B is intentionally publicly discoverable per the existing `derive.rs` rationale); no new secret is exposed.
- The size guard (`RecordTooLarge`) prevents a future oversized `routing_blob` from producing malformed/truncated relay state.

## 11. Open items (resolved at implementation time, not blockers)

- (Resolved, §7) pkarr **3.10.0** with `features=["signed_packet"]`; builder is `SignedPacket::builder().txt(Name, TXT, ttl).sign(&keypair)`.
- Exact `simple_dns` (re-exported as `pkarr::dns`) TXT multi-character-string construction for base64 payloads > 255 bytes — confirmed by the Task 1 codec spike before the encode/decode helpers are written.
- `MockPkarrRelay` is upgraded by adding a **strict** validating constructor (parses the z32 key + runs `from_relay_payload` on PUT, returning 400 on failure) used by the wire-conformance + publisher/resolver tests, while the existing **lax** constructor is retained for the relay-pool/cooldown mechanics tests that intentionally PUT arbitrary bytes — §6.1.
