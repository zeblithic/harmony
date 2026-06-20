# pkarr Freshness: Signed-TTL Trust Model (facet-A) — Design Spec

**Status:** Approved (design) — Jake confirmed 2026-06-20; quorum-converged (Koya + AVALON judge + Ildwyn). Tracks ZEB-516 facet-A. (ZEB refs appear in this design file only — never in branch/commit/PR-title/body.)

**Goal:** Replace pkarr's hard ±30-minute wall-clock freshness window with a **publisher-signed TTL** (`valid_until`) as the primary freshness mechanism, so a record stays resolvable for its cryptographically-committed validity window rather than only ~30 min after each (rare) republish. This unblocks delayed / offline / store-and-forward resolution — first-contact hours/days after publish, and async-DM to an offline recipient (the deposit rungs tracked by ZEB-488 / ZEB-493).

**Repos:** `harmony` (core: `crates/harmony-pkarr`) + `harmony-client` (`src-tauri`). One flag-day wire change; coordinated 2-PR change (core first, then client), mirroring the ZEB-380 multi-relay split.

**Non-goal:** This spec does NOT implement the async-DM deposit rungs or their long TTLs — it builds the signed-TTL *framework* + applies it to reachability records. Deposit records reuse the framework later with their own (longer) TTLs.

---

## 1. Problem

The resolver drops any record whose `announced_at` is more than ±30 min from now:

- `PkarrRoutingRecord::verify_skew(now_ms)` — `record.rs:94-101`, `SKEW_TOLERANCE_MS = 30 * 60 * 1_000` (`record.rs:36`). Returns `PkarrError::StaleOrSkewed` (log tag RPK4).

But the publisher only republishes on the **7-day BEP44 epoch lifecycle**:

- `compute_next_publish_at` (`publisher.rs:238-253`) schedules at `epoch_start + 30 min` and `epoch_start + 3.5 d` (`EPOCH_DURATION_MS = 7 d`). So after the boot publish a record refreshes only **~twice per 7-day epoch** (~3.5-day gaps).

**Net:** a record is resolvable only for ~30 min after each republish, then unresolvable for up to ~3.5 days. First-contact has only ever worked when the redeem landed within 30 min of a publish — confirmed empirically 2026-06-20: a cross-machine first-contact greened *only* because the redeem was deliberately prompt (inside the window). For any delayed resolution — and structurally for **async-DM store-and-forward**, where an offline recipient resolves minutes-to-days later — the current guard cannot work.

**Root cause (category error):** the ±30 min check conflates *refresh-staleness* ("how long since the node last republished") with *truth-staleness* ("is this reachability claim still valid"). A slow republish cadence punishes a perfectly-valid record. Signed-TTL separates the two: the publisher declares the truth-window; refresh cadence stops being load-bearing for freshness.

---

## 2. Current State (code map)

### harmony-pkarr (`crates/harmony-pkarr`)
- **`record.rs`** — `PkarrRoutingRecord { routing_blob: Vec<u8>, harmony_identity_pub: [u8;64], announced_at_ms: u64, inner_sig: [u8;64] }`. Inner signature preimage = `canonical_signed_bytes(routing_blob, harmony_identity_pub, announced_at_ms)` — a CBOR 3-tuple (`record.rs:130-148`). `sign_new(routing_blob, harmony_identity_pub, announced_at_ms, identity_sk)`. `verify_skew` + `SKEW_TOLERANCE_MS` as above.
- **`resolver.rs`** — `PkarrResolver { relay, cache: LruCache<[u8;32], CachedResolution> }` (in-memory only, boot-lifetime, no persistence; `resolver.rs:29-32`). Validation sequence in `resolve()` (`resolver.rs:56-109`): relay GET → `wire::parse_relay_payload` (BEP44 outer sig) → `verify_inner_sig` → `verify_skew` → positive-cache 15 min / negative-cache 60 s. `verify_skew` also re-runs on cache hits.
- **`publisher.rs`** — `compute_next_publish_at` (cadence above); `RecordBuilder = Arc<dyn Fn(u64)->PkarrRoutingRecord>` (the `u64` is `now_ms`); `register`/`drive_pending`/`publish_one`.
- **`wire.rs`** — encodes `PkarrRoutingRecord` into a BEP44 `SignedPacket` (via the external `pkarr` crate). The `pkarr` crate sets the BEP44 `seq` to the current Unix-timestamp-in-seconds on `sign`; **harmony never reads or compares `seq` today** (no seq inspection anywhere in either repo).
- `RECORD_TTL = 300s` is the DNS TTL only — explicitly documented as non-load-bearing ("harmony freshness is governed by `verify_skew` + epoch rotation, not DNS TTL").

### harmony-client (`src-tauri`)
- **`reachability_record.rs`** — `ReachabilityAnnouncePayload` is CBOR-encoded into the `routing_blob` of `PkarrRoutingRecord` (and is separately the community-CRDT payload). Its `identity_signature` is **zero-filled** when used as a pkarr routing blob — the identity binding for the pkarr path is `PkarrRoutingRecord.inner_sig`, NOT the payload's own signature.
- **Publisher closures** — `pkarr_identity_publisher.rs:56-58`, `pkarr_community_publisher.rs:52-54`, `pkarr_friend_publisher.rs`, `pkarr_invite_publisher.rs` each build a `RecordBuilder` calling `PkarrRoutingRecord::sign_new(blob_builder(), id_pub, at_ms, &id_sk)`. **All cases publish the same record shape; only the ephemeral key differs.**
- **`pkarr_resolver_adapter.rs`** — re-checks inner sig + skew (RPK4) and decodes `routing_blob` → `ReachabilityAnnouncePayload`.
- **Resolver derive sites** — `lib.rs:41526` (`connectivity_discover_identity`) and `lib.rs:43026` (redeem / add-friend) derive case-B as `derive_ephemeral_key(Identity, identity_pub_64, epoch.to_be_bytes())` over `epoch_tolerance_window(now)`.

**Key architectural fact:** the freshness guard fires on the **outer `PkarrRoutingRecord.announced_at_ms`** (`at`), not the inner `ReachabilityAnnouncePayload.announced_at_ms` (`ts`). Therefore `valid_until` belongs on **`PkarrRoutingRecord`** (harmony-pkarr), and the harmony-client change is limited to *threading the TTL into the `sign_new` call* inside the publisher closures (+ fixture regen). `ReachabilityAnnouncePayload` itself does **not** need a new field.

---

## 3. Design

### 3.1 Signed `valid_until` (PRIMARY)
- Add `valid_until_ms: u64` to `PkarrRoutingRecord`, **inside** the signed preimage: `canonical_signed_bytes` becomes a CBOR 4-tuple `(routing_blob, harmony_identity_pub, announced_at_ms, valid_until_ms)`.
- `sign_new` gains a `valid_until_ms` parameter (or a `ttl_ms` it adds to `announced_at_ms`); the inner Ed25519 signature now covers it. Because it is signed by the identity key, `valid_until` cannot be forged or extended by a relay/attacker.
- The resolver honors a record while `now <= valid_until_ms`, replacing the ±30 min lower bound.

### 3.2 future-strict
- Replace `verify_skew` with `verify_freshness(now_ms)`:
  - reject if `announced_at_ms > now_ms + FUTURE_TOLERANCE_MS` (forged-future guard), and
  - reject if `now_ms > valid_until_ms` (expired).
- `FUTURE_TOLERANCE_MS` = small clock-skew allowance (proposed **5 min**). The old symmetric ±30 min is gone; the upper bound is now the signed TTL, the lower (future) bound stays tight.

### 3.3 seq anti-rollback — in-memory (v1)
- Surface the BEP44 `seq` from the parsed `SignedPacket` (`wire::parse_relay_payload`) up to the resolver (e.g., return it alongside the `PkarrRoutingRecord`, or attach to the resolution).
- The resolver keeps a per-key highest-`seq`-seen map **in memory** (alongside the LRU; e.g. `Arc<Mutex<HashMap<[u8;32], u64>>>`). On resolve: reject a record whose `seq <= stored_highwater`; on accept, update the highwater.
- **Not** in the signed preimage and **no wire change** — `seq` already rides the BEP44 envelope (DHT-CAS-enforced monotonic at PUT time). This adds resolver-side defense against a relay replaying an older (validly-signed, still-within-TTL) record *within a session*.

### 3.4 Diverse-relay re-resolve on dial-fail (DoD condition)
- The resolved endpoint is treated as **best-effort**. On a dial failure, re-resolve — and re-resolve from a **different** relay than the one that served the accepted record, taking the freshest result (highest `seq`; latest `announced_at` within TTL).
- This is what makes in-memory seq sound: a single hostile/stale relay cannot pin a resolver to an old-but-within-TTL record, because a dial-fail triggers a cross-relay check. Lives in the harmony-client dial path (`pkarr_resolver_adapter` / redeem+dial flow); the core resolver already iterates a relay pool, so this is mostly orchestration on the dial-retry.

### 3.5 TTL durations
- **Reachability records (identity / community / friend):** proposed TTL = **7 days (one epoch)**. Rationale: the ephemeral *key* rotates per epoch (`derive_ephemeral_key(..., epoch)`), so a record's natural relevance is already epoch-bounded; a 7-day TTL covers the ~3.5-day republish gap with margin and never exceeds the key's epoch lifetime. (A shorter TTL, e.g. 4 days, also covers the gap and bounds staleness tighter — open for spec review; default 7 d for alignment with the epoch.)
- **async-DM deposit records:** long TTL, **out of scope here** — set by the deposit-rung work (ZEB-488/493) reusing this same `valid_until` field.
- Encode TTL as a per-case constant the publisher closures pass to `sign_new`.

### 3.6 Flag-day migration — hard cutover (security-load-bearing)
- Adding `valid_until_ms` to the signed preimage is wire-incompatible: an old record signs a 3-tuple, a new record a 4-tuple; their inner signatures verify against different preimages.
- **Decision: hard cutover via a record version bump.** Old-format records fail inner-sig verification against the new preimage and are ignored; nodes republish fresh under the new format within a republish tick. Acceptable pre-launch (small fleet, frequent republish, the canonical Zeblithic node re-published on upgrade).
- **Why NOT a legacy fallback** ("if `valid_until` absent, use ±30 min"): that is a **downgrade attack** surface — an adversary serves/strips the field to force the resolver onto the weaker guard. A permanent fallback re-introduces exactly the weakness we're removing. Hard cutover eliminates it. (If a migration window is ever needed for a larger deployment, it must be time-boxed and removed — not a standing path.)

### 3.7 Gate fail-loud (minor, harmony-client, optional fold-in)
- Adjacent hardening surfaced during the 2026-06-20 debug (not the cause of anything that day): the `identity_discoverable` settings load (`PkarrSettings::load_or_default`) fails **closed and silent** — a parse error silently yields `identity_discoverable=false`.
- Fold in: WARN/ERROR when the gate reads false while a prior published record exists; treat "was-discoverable, now-unreadable" as an error state, not a silent default; keep parse-error distinct from explicit opt-out; **never fail-open on parse** (silently becoming discoverable would violate a real opt-out — privacy-fail-open is worse than a freeze). Minor; can ship in PR2 or be split out.

---

## 4. Security Analysis

| Threat | Old guard | New guard |
|---|---|---|
| Stale reachability served by a relay | ±30 min wall-clock | signed TTL (publisher commits) + seq anti-rollback + diverse-relay re-resolve; endpoint best-effort (dial-fail → retry) |
| Forged-future timestamp | ±30 min upper bound | future-strict reject (`FUTURE_TOLERANCE_MS`) |
| `valid_until` forgery / extension | n/a | signed in the inner preimage — unforgeable without the identity key |
| Downgrade to weaker guard | n/a | prevented by hard cutover (no legacy fallback path) |
| Replay of older signed record | partial (age) | seq highwater rejects `seq <= seen` (in-session) |

**Residual risk:** a relay serving a stale-but-within-TTL, validly-signed record. Mitigated by (a) seq anti-rollback rejecting an older record once a newer one was seen, (b) diverse-relay re-resolve on dial-fail cross-checking another relay, and (c) the endpoint being best-effort (a stale endpoint costs a retry, not a hard failure).

**Known limitation (accepted, deferred):** in-memory seq is forgotten across a resolver reboot, so a rollback *within a long TTL* could slip through once after a restart. This only matters for **long-TTL deposit records** (async-DM); reachability TTLs are short enough and republish often enough that the window is small. Persisted per-key seq is the fix and is **deferred to the async-DM milestone** (ZEB-488/493) — it needs no second flag-day (seq is not in the signed preimage).

---

## 5. Cross-repo split

**PR1 — harmony (core), merges first:**
- `record.rs`: add `valid_until_ms` to `PkarrRoutingRecord` + `canonical_signed_bytes` (4-tuple); update `sign_new`; record version bump; replace `verify_skew` with `verify_freshness` (future-strict + `valid_until`); remove/retire `SKEW_TOLERANCE_MS` (or repurpose into `FUTURE_TOLERANCE_MS`).
- `wire.rs`: surface BEP44 `seq` out of `parse_relay_payload`.
- `resolver.rs`: call `verify_freshness`; add in-memory per-key seq highwater + check; thread `seq` through.
- Fixture regen: `tests/wire_format_pkarr_routing_record_fixtures.rs`.

**PR2 — harmony-client, after PR1 (bumps the `harmony-pkarr` dep):**
- Thread the per-case TTL into the `sign_new` calls in `pkarr_identity_publisher.rs` / `pkarr_community_publisher.rs` / `pkarr_friend_publisher.rs` / `pkarr_invite_publisher.rs`.
- Diverse-relay re-resolve on dial-fail (`pkarr_resolver_adapter.rs` / redeem+dial path).
- `pkarr_resolver_adapter.rs`: align the re-checked freshness with `verify_freshness` (drop the RPK4 ±30 min re-check).
- Gate fail-loud (§3.7).
- Fixture regen: `src-tauri/tests/wire_format/*` (pkarr routing record + any record-shape pins).

---

## 6. Testing (TDD)

**harmony-pkarr (unit):**
- `verify_freshness`: accepts within-TTL (incl. >30 min old), rejects expired (`now > valid_until`), rejects future (`announced_at > now + tolerance`).
- `sign_new` + `verify_inner_sig` round-trip including `valid_until`; tamper of `valid_until` fails verification.
- seq highwater: rejects `seq <= seen`, accepts newer, updates on accept.
- fixture pin regen (golden CBOR with `valid_until`).

**harmony-client:**
- publisher closures set `valid_until = announced_at + TTL(case)` per case.
- diverse-relay re-resolve: on dial-fail, queries a *different* relay and picks the freshest (mock relay pool).
- gate fail-loud unit (parse error → loud, not silent; never fail-open).
- fixture regen.

**Integration (the core win):**
- publish → resolve **>30 min later** (e.g. simulated 1 hr+) within TTL **succeeds** (impossible under the old ±30 min guard).
- expired record (`now > valid_until`) rejected; future record rejected.

---

## 7. Out of Scope
- Persisted (on-disk) seq — deferred to async-DM (ZEB-488/493).
- The async-DM deposit rungs and their long TTLs (this spec provides the `valid_until` framework they reuse).
- Changing the epoch / republish cadence (orthogonal — signed-TTL makes cadence non-load-bearing for freshness, so the cadence can stay as-is).

## 8. Definition of Done
- Resolver honors signed `valid_until`; a reachability record resolves well after publish (1 hr+) while within TTL.
- Expired and forged-future records rejected; in-session seq rollback rejected.
- Diverse-relay re-resolve on dial-fail.
- Hard cutover: pre-`valid_until` records are ignored (no legacy ±30 min fallback path).
- Full gate green in both repos (`cargo fmt` + `clippy --locked --all-targets` + `nextest --locked --all-targets`, harmony-client with `--features test-fixtures`).
- Fleet re-test (ZEB-477): A-side re-runs first-contact with a **delayed** redeem (>30 min after publish) → resolves (the facet-A win), proving the async-DM precondition.
