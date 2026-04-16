# Design: SMTP RCPT Admission + Discovery-Backed Email→Hash Resolution (ZEB-120)

**Date:** 2026-04-15
**Ticket:** [ZEB-120](https://linear.app/zeblith/issue/ZEB-120) (parent: ZEB-113)
**Follows:** PR #240 (ZEB-113 PR A — SMTP remote delivery machinery)
**Phase:** ZEB-113 PR A' — admit non-local recipients at RCPT TO by resolving `local_part@domain → identity.address_hash`

---

## 1. Context and Problem Statement

### 1.1 What shipped in PR #240

PR #240 (ZEB-113 PR A, merged 2026-04-15) landed the downstream delivery machinery for remote mail:

- `RecipientResolver` trait (`harmony-mail/src/remote_delivery.rs:127-147`) — `fn resolve(&self, &IdentityHash) -> Option<Identity>`
- `HarmonyEnvelope::seal()` integration in the SMTP handler
- `publish_sealed_unicast()` publishing sealed envelopes to `harmony/msg/v1/unicast/{recipient_hash_hex}`
- Strict per-recipient `remote_accepted` semantics (sole-recipient miss → 451; batch succeeds if ≥1 recipient accepted)
- `identity_from_announce_record()` helper for reconstructing `Identity` from announce records (Ed25519 / X25519 only; ML-DSA-65 explicitly rejected)

### 1.2 The RCPT admission gap

`run_async_address_resolution` at `crates/harmony-mail/src/server.rs:1107-1133` rejects every non-local domain at RCPT TO. The delivery machinery from PR #240 is exercised only by tests that construct `DeliverToHarmony` directly, bypassing RCPT.

Root cause: `RecipientResolver::resolve` takes an `IdentityHash`, but RCPT TO gives us `local_part@domain`. The missing primitive is one level *above* `RecipientResolver`: a mapping from email address to identity hash. `AnnounceRecord` in `harmony-discovery` is purely cryptographic (hash + public_key + encryption_key + routing_hints); there are no email-identifying fields anywhere in the discovery layer.

### 1.3 Scope of this PR

1. New crate `harmony-mail-discovery` containing the `EmailResolver` abstraction and a production implementation backed by DNS + HTTPS.
2. Integration of `EmailResolver` into `run_async_address_resolution` so non-local domains are resolved (or rejected with appropriate SMTP responses) rather than blanket-rejected.
3. Collapse `run()`'s three `Option` parameters into a single `RemoteDeliveryContext` struct.
4. Production `RecipientResolver` implementation in `harmony-mail::remote_delivery` (Zenoh-backed lookup against `harmony/identity/{hash}/resolve`) so `main.rs` can wire the full chain.
5. `main.rs` wiring: load gateway `PrivateIdentity`, construct `DefaultEmailResolver` + production `RecipientResolver`, bundle into `RemoteDeliveryContext`, pass to `run()`.

### 1.4 Out of scope (follow-ups)

- Bundle endpoint for batched claim fetching (future ZEB ticket)
- Per-email signature chain for equivocation detection (reserved via claim format version byte)
- Key Transparency / Merkle-tree-based non-equivocation (reserved via claim format version byte)
- Post-quantum master keys (reserved via `Signature` enum and `alg` field in domain record)
- Phase 2 Zenoh distribution of signed claims (separate ticket; this PR ships HTTPS-only)
- `OfflineResolver` + store-and-forward for offline recipients (ZEB-113 PR B)

---

## 2. North Star and Phasing

### 2.1 Phase 2 target architecture (committed)

**Domain-scoped signed email→hash claims** with a two-tier key hierarchy:

1. **Master key** (Ed25519). Long-lived. Lives offline (hardware token or ops workstation). Published in DNS at `_harmony.<domain>` TXT record. Rotates only on compromise or algorithm deprecation (expected cadence: years).
2. **Signing key** (Ed25519). 90-day validity. Lives hot on the domain's gateway. Authorized by a master-signed `SigningKeyCert` that rides inline in every claim.
3. **Claim** (signed by signing key). Asserts `local_part@domain → identity_hash` with a monotonic per-email serial, a validity window, and a reference to the signing key that signed it.

### 2.2 Phase 1 (this PR)

Phase 2's trust model shipped end-to-end over HTTPS:

- Claims served from `GET https://<domain>/.well-known/harmony-users?h=<hashed_local_part>`
- Revocations served from `GET https://<domain>/.well-known/harmony-revocations`
- Wire format is identical to what Phase 2 will use over Zenoh. Phase 2 adds a transport; it does not change the cryptographic shape.

### 2.3 Explicit phasing invariant

**DNS only changes on master-key rotation.** Signing-key rotation happens entirely in-band via new claims carrying new certs. Master-key rotation (rare) is the only event that requires a DNS update.

---

## 3. Trust Model

### 3.1 Three trust layers, bootstrapped in order

| Layer | Trust anchor | Rotation cadence | Cache key |
|---|---|---|---|
| 1. DNS → master key | DNS (DNSSEC or TLS-rooted implicit trust) | Years (compromise-only) | `domain` |
| 2. Master key → signing key | Master key (from DNS) | 90 days (overlapping); sharp cutover via revocation cert | `(domain, signing_key_id)` |
| 3. Signing key → claim | Signing key (from cert) | Issued on-demand per user binding | `(domain, hashed_local_part)` |

### 3.2 Rotation protocol

**Planned rotation (signing key, every 90 days):**

- Operator generates next signing keypair ~2 weeks before current cert's `valid_until`
- Next cert is minted via offline master-key ceremony with `valid_from = current.valid_until - 2w`, `valid_until = current.valid_until + 90d`
- During overlap window, gateway may serve claims signed by either key; senders accept any claim whose cert is within its own `[valid_from, valid_until]` window
- After cutover, old signing key is destroyed

**Revocation (signing key, compromise):**

- Operator issues a master-signed revocation cert: same `SigningKeyCert` shape, but with `valid_until` set to the revocation timestamp (typically "now" or earlier)
- Revocation cert is published in the `/.well-known/harmony-revocations` list
- Senders detect revocation via background 6-hour refresh of the revocation list
- Worst-case propagation lag: 6 hours (the refresh interval). Claims issued before revocation are grandfathered through their own `expires_at`.

**Master-key rotation (years, compromise-only):**

- Single DNS TXT record is updated with new master pubkey
- All signing keys must be re-certified under the new master key
- Senders bootstrap the new master key on next DNS TTL expiry
- Explicitly a cutover event; no DNS-level overlap (we reject multiple `v=harmony1` TXT records as malformed)

### 3.3 Domain salt

Published alongside the master key in the DNS TXT record. Publicly derivable, explicitly **not** a secret (unlike Matrix MSC2134's "pepper," which was treated as secret and caused confusion). Purpose: raise enumeration cost from "curl" to "dictionary through SHA-256." Not a strong defense; a principled first-line deterrent.

Rotation: tied to master-key rotation. Salt rotation alone is unnecessary and would invalidate every cached claim for the domain.

### 3.4 What we rely on DNS/TLS for

- DNSSEC or implicit TLS trust to authenticate the `_harmony.<domain>` TXT record's master pubkey
- HTTPS TLS certificate validation for `.well-known/` endpoints (standard reqwest defaults)
- No redirect-following (DNS-rebinding and policy-attack defense)

### 3.5 What we do NOT rely on DNS/TLS for

- Claim integrity — signed end-to-end by the signing key under a master-signed cert
- Signing-key rotation — in-band via new claims
- Revocation — master-signed revocation list, verifiable independent of transport

---

## 4. Wire Formats

All artifacts are CBOR-encoded (deterministic encoding per RFC 8949 §4.2). Signature-covered byte ranges are explicitly the canonical-CBOR encoding of the payload struct, not the enclosing envelope — this keeps forward-compat open for sidecar fields.

### 4.1 DNS record

```
_harmony.q8.fyi. IN TXT "v=harmony1; k=<base64url(master_pubkey)>; salt=<base64url(salt)>; alg=ed25519"
```

Parsed struct:

```rust
pub struct DomainRecord {
    pub version: u8,                 // "harmony1" → 1; unknown versions rejected
    pub master_pubkey: MasterPubkey, // enum
    pub domain_salt: [u8; 16],
    pub alg: SignatureAlg,           // enum
}

pub enum MasterPubkey {
    Ed25519([u8; 32]),
    // MlDsa65(...)  — reserved; enum leaves room
}

pub enum SignatureAlg {
    Ed25519,
    // MlDsa65, Hybrid(...) — reserved
}
```

Rules:
- Fields are `k=v` pairs separated by `; ` (space after semicolon)
- Parser rejects unknown `v=` values
- Parser tolerates unknown k=v fields (forward-compat)
- Multiple TXT records matching `v=harmony1;` are rejected as malformed (no DNS-level overlap for master keys)
- Recommended DNS TTL: 1 hour

### 4.2 Signing-key cert

```rust
pub struct SigningKeyCert {
    pub version: u8,                       // 1
    pub signing_key_id: [u8; 8],           // random, 64-bit space
    pub signing_pubkey: SigningPubkey,     // enum, Ed25519 for now
    pub valid_from: u64,                   // unix seconds
    pub valid_until: u64,                  // unix seconds; can be in past (revocation cert)
    pub domain: String,                    // FQDN bound to the cert
    pub master_signature: Signature,       // over canonical CBOR of all preceding fields
}

pub enum Signature {
    Ed25519([u8; 64]),
    // MlDsa65(Box<[u8; 3309]>), Hybrid(Box<HybridSignature>)  — reserved
}
```

Why `domain` is inside the cert: closes cross-domain substitution attacks. A compromised signing key from domain A cannot be presented as a signing key for domain B.

Why 8-byte `signing_key_id`: random-assigned, 64-bit space puts birthday collisions at ~2³² certs — vastly beyond any domain's real issuance rate. Simpler than content-hash addressing, shorter than UUID.

### 4.3 Signed claim

```rust
pub struct ClaimPayload {
    pub version: u8,                        // 1; bumps on payload shape change (e.g., future sigchain or KT fields)
    pub domain: String,                     // must match cert.domain and query domain
    pub hashed_local_part: [u8; 32],        // SHA-256(local_part || 0x00 || domain_salt); canonical cache key
    pub email: String,                      // cleartext; audit/debug only; NOT the canonical key
    pub identity_hash: [u8; 16],            // Harmony IdentityHash — the thing we resolved to
    pub issued_at: u64,                     // unix seconds
    pub expires_at: u64,                    // unix seconds; default +7 days
    pub serial: u64,                        // strictly increasing per (domain, hashed_local_part)
    pub signing_key_id: [u8; 8],            // references the cert
}

pub struct SignedClaim {
    pub payload: ClaimPayload,
    pub cert: SigningKeyCert,               // embedded inline; enables claim-driven key discovery
    pub claim_signature: Signature,         // over canonical CBOR of payload
}
```

Verification algorithm:

1. Check `payload.domain == cert.domain == queried_domain`
2. Check `SHA-256(payload.email.local_part || 0x00 || domain_record.domain_salt) == payload.hashed_local_part`
3. Verify `cert.master_signature` under `domain_record.master_pubkey` over canonical CBOR of cert fields
4. Check `now ∈ [cert.valid_from - TOLERANCE, cert.valid_until + TOLERANCE]`
5. Check `cert.signing_key_id` against revocation view; if revoked, check grandfathering
6. Verify `claim.claim_signature` under `cert.signing_pubkey` over canonical CBOR of payload
7. Check `now < claim.expires_at + TOLERANCE`

`TOLERANCE = 60 seconds` for clock skew (applied to all time-bound checks).

Default claim lifetime: `expires_at - issued_at = 7 days`. Rationale: longer than a signing-key window would be nonsense; shorter makes cache worthless; 7 days balances revocation-by-expiry against re-fetch frequency.

### 4.4 Revocation list

```rust
pub struct RevocationList {
    pub version: u8,                          // 1
    pub domain: String,
    pub issued_at: u64,
    pub revoked_certs: Vec<SigningKeyCert>,   // full certs with valid_until in the past
    pub master_signature: Signature,          // over canonical CBOR of preceding fields
}
```

Why full revocation certs instead of just IDs: the revocation cert is a self-standing, auditable statement. An auditor can verify each entry independently. A revocation cert encountered in a claim response verifies identically to one in the list.

Grandfathering semantics:
- If `claim.signing_key_id == revoked_cert.signing_key_id`:
  - `claim.issued_at > revoked_cert.valid_until` → reject (`Revoked`)
  - `claim.issued_at ≤ revoked_cert.valid_until` → grandfather; continue verification (claim expires naturally at its own `expires_at`)

### 4.5 HTTP response envelopes

`GET https://<domain>/.well-known/harmony-users?h=<base64url(hashed_local_part)>`:
- **200** + `Content-Type: application/cbor` → CBOR-encoded `SignedClaim`
- **404** → no such claim
- Other → transient

`GET https://<domain>/.well-known/harmony-revocations`:
- **200** + `Content-Type: application/cbor` → CBOR-encoded `RevocationList`
- **404** → treated as authoritative "no revocations" (empty list cached)
- Other → transient; count against 24h safety valve

Constraints on all HTTP fetches:
- HTTPS only; no HTTP upgrade attempts
- No redirect-following
- 5s connect timeout + 10s total timeout
- 1 MB body size cap
- TLS validation non-negotiable (reqwest defaults)

### 4.6 Version bytes — forward compatibility

Four independent version bytes, one per artifact:

- `DomainRecord.version` — bumps on master-key algorithm space change
- `SigningKeyCert.version` — bumps on cert shape change
- `ClaimPayload.version` — bumps to add a per-email signature chain (future equivocation defense) or a Key-Transparency-style Merkle proof layer (future domain-level non-equivocation)
- `RevocationList.version` — bumps on reason codes or other structural additions

Unknown version → reject (no silent forward-compat). Cost: 4 bytes; benefit: format evolution doesn't have to guess whether old formats are still valid.

---

## 5. Components

New crate `harmony-mail-discovery` with five modules, ordered leaf-to-root:

### 5.1 `claim.rs` — types + verification

Pure, sans-I/O. All cryptographic correctness lives here.

**Public API:**

```rust
pub struct DomainRecord { /* §4.1 */ }
pub struct SigningKeyCert { /* §4.2 */ }
pub struct ClaimPayload { /* §4.3 */ }
pub struct SignedClaim { /* §4.3 */ }
pub struct RevocationList { /* §4.4 */ }
pub enum Signature { /* §4.2 — PQ-ready enum */ }

pub struct VerifiedBinding {
    pub domain: String,
    pub email: String,
    pub identity_hash: IdentityHash,
    pub serial: u64,
    pub claim_expires_at: u64,
    pub signing_key_id: [u8; 8],
}

impl SignedClaim {
    pub fn verify(
        &self,
        domain_record: &DomainRecord,
        revocations: &RevocationView,
        now: u64,
    ) -> Result<VerifiedBinding, VerifyError>;
}

pub enum VerifyError {
    DomainMismatch,
    HashedLocalPartMismatch,
    CertSignatureInvalid,
    ClaimSignatureInvalid,
    CertNotYetValid { valid_from: u64 },
    CertExpired { valid_until: u64 },
    CertRevoked { revoked_at: u64 },
    ClaimExpired { expires_at: u64 },
    UnsupportedVersion(u8),
    UnsupportedAlgorithm,
}
```

Dependencies: `harmony-crypto`, `ciborium`, `harmony-identity`.

### 5.2 `cache.rs` — four caches + soft-fail window

```rust
pub struct ResolverCaches {
    claim: DashMap<(String, [u8; 32]), CacheEntry<SignedClaim>>,
    signing_key: DashMap<(String, [u8; 8]), CacheEntry<VerifiedSigningKey>>,
    master_key: DashMap<String, CacheEntry<DomainRecord>>,
    revocation: DashMap<String, CacheEntry<RevocationView>>,
    domain_last_seen: DashMap<String, u64>,    // 72h soft-fail window
    negative: DashMap<(String, [u8; 32]), u64>, // 60s negative cache
}

pub struct CacheEntry<T> {
    pub value: T,
    pub expires_at: u64,
}
```

TTLs (configurable defaults):
- `claim` cache entry: claim's own `expires_at` (default 7d)
- `signing_key` cache entry: `cert.valid_until`
- `master_key` cache entry: DNS TTL (default 1h)
- `revocation` cache entry: `now + 6h`
- `domain_last_seen` entry: 72h
- `negative` entry: 60s

Eviction: lazy on access + periodic bulk sweep every 15 min from background task. LRU bound per cache (default 10k entries, configurable) for bounded memory under adversarial input.

Time injection: `TimeSource` trait — `fn now() -> u64`. Default impl uses `SystemTime`; tests use `FakeTimeSource`.

Dependencies: `dashmap`.

### 5.3 `dns.rs` — DNS TXT fetching

```rust
#[async_trait]
pub trait DnsClient: Send + Sync + 'static {
    async fn fetch_txt(&self, name: &str) -> Result<Vec<String>, DnsError>;
}

pub struct HickoryDnsClient { /* wraps hickory-resolver */ }

pub async fn fetch_domain_record(
    dns: &dyn DnsClient,
    domain: &str,
) -> Result<DomainRecord, DnsFetchError>;

pub enum DnsFetchError {
    NoRecord,             // NXDOMAIN or NOERROR-no-TXT
    Malformed(String),
    Transient(String),
    UnsupportedVersion(u8),
}
```

Parser strictness:
- Rejects unknown `v=` values
- Ignores unknown fields (forward-compat)
- Rejects missing required fields (`k`, `salt`, `alg`)
- Rejects malformed base64url
- Rejects multiple `v=harmony1;` TXT records

Dependencies: `hickory-resolver`, `async-trait`.

### 5.4 `http.rs` — HTTPS claim + revocation fetching

```rust
#[async_trait]
pub trait HttpClient: Send + Sync + 'static {
    async fn get(&self, url: &str) -> Result<HttpResponse, HttpError>;
}

pub struct ReqwestHttpClient { /* wraps reqwest::Client */ }

pub enum ClaimFetchResult {
    Found(SignedClaim),
    NotFound,
}

pub enum RevocationFetchResult {
    Found(RevocationList),
    Empty,     // 404 → authoritative empty
}

pub async fn fetch_claim(http: &dyn HttpClient, domain: &str, h: &[u8; 32])
    -> Result<ClaimFetchResult, HttpFetchError>;

pub async fn fetch_revocation_list(http: &dyn HttpClient, domain: &str)
    -> Result<RevocationFetchResult, HttpFetchError>;
```

URL construction: hardcoded to `https://<domain>/.well-known/harmony-users?h=<...>` and `https://<domain>/.well-known/harmony-revocations`. No HTTP; no redirects.

Dependencies: `reqwest` (rustls-backed), `async-trait`.

### 5.5 `resolver.rs` — orchestration

```rust
#[async_trait]
pub trait EmailResolver: Send + Sync + 'static {
    async fn resolve(&self, local_part: &str, domain: &str) -> ResolveOutcome;
}

pub enum ResolveOutcome {
    Resolved(IdentityHash),
    DomainDoesNotParticipate,
    UserUnknown,
    Transient { reason: &'static str },
    Revoked,
}

pub struct DefaultEmailResolver {
    dns: Arc<dyn DnsClient>,
    http: Arc<dyn HttpClient>,
    caches: Arc<ResolverCaches>,
    time: Arc<dyn TimeSource>,
    config: ResolverConfig,  // all tunables (soft-fail window, refresh intervals, etc.)
    background_task: Option<JoinHandle<()>>,
}

pub struct ResolverConfig {
    pub soft_fail_window_secs: u64,          // default: 72h
    pub negative_cache_secs: u64,            // default: 60s
    pub revocation_refresh_secs: u64,        // default: 6h
    pub revocation_safety_valve_secs: u64,   // default: 24h
    pub clock_skew_tolerance_secs: u64,      // default: 60s
    pub claim_cache_max_entries: usize,      // default: 10k
    pub dns_timeout_secs: u64,               // default: 5s
    pub http_timeout_secs: u64,              // default: 10s
    pub http_body_max_bytes: usize,          // default: 1 MB
}

impl DefaultEmailResolver {
    pub fn new(/* ... */) -> Self;
    pub fn spawn_background_refresh(&mut self);
}

#[async_trait]
impl EmailResolver for DefaultEmailResolver { /* §6 data flow */ }
```

Background task (single tokio task, 15-min tick):
- Evict expired entries across all caches
- Per-domain revocation refresh (cap 1 concurrent per domain) when `last_refreshed + 6h < now`
- Mark domains as `revocation_stale` if no successful refresh in >24h

### 5.6 `harmony-mail` integration

```rust
pub struct RemoteDeliveryContext {
    pub gateway_identity: Arc<PrivateIdentity>,
    pub recipient_resolver: Arc<dyn RecipientResolver>,
    pub email_resolver: Arc<dyn EmailResolver>,
}

pub async fn run(
    config: Config,
    remote_delivery: Option<RemoteDeliveryContext>,
) -> Result<(), Box<dyn std::error::Error>>;
```

Signature change: three `Option`s collapse to one. No "two of three" state.

Integration site: `server.rs:1107-1133`. Replaces the blanket-reject block with a call into `email_resolver.resolve()` and a match over `ResolveOutcome` mapping to SMTP codes per §7.

### 5.7 Production `RecipientResolver`

Lives in `harmony-mail::remote_delivery` alongside the existing `RecipientResolver` trait. Queries `harmony/identity/{hash_hex}/resolve` via Zenoh queryable, parses the returned `AnnounceRecord`, calls the existing `identity_from_announce_record()` helper.

Injected into `RemoteDeliveryContext` by `main.rs`.

### 5.8 File/module map

| Concern | Crate | File |
|---|---|---|
| Types + verification | `harmony-mail-discovery` | `claim.rs` |
| Caches + TTLs | `harmony-mail-discovery` | `cache.rs` |
| DNS fetching | `harmony-mail-discovery` | `dns.rs` |
| HTTPS fetching | `harmony-mail-discovery` | `http.rs` |
| Orchestrator + `EmailResolver` | `harmony-mail-discovery` | `resolver.rs` |
| Debug CLI | `harmony-mail-discovery` | `bin/harmony-mail-discovery-debug.rs` |
| `RemoteDeliveryContext` + `run()` | `harmony-mail` | `server.rs` |
| RCPT admission integration | `harmony-mail` | `server.rs` (~line 1107) |
| Production `RecipientResolver` | `harmony-mail` | `remote_delivery.rs` |
| Wiring | `harmony-mail` | `main.rs` |

---

## 6. Data Flow

### 6.1 Cold path — first resolution at a new domain

Sender receives `RCPT TO:<alice@q8.fyi>`; no cached state.

1. `smtp.rs::handle_rcpt_to` → `SmtpAction::ResolveHarmonyAddress`
2. `server.rs::run_async_address_resolution`: non-local domain → `ctx.email_resolver.resolve("alice", "q8.fyi")`
3. `DefaultEmailResolver::resolve`:
   a. `cache.master_key("q8.fyi")` → miss
   b. `dns.fetch_domain_record("q8.fyi")` → `_harmony.q8.fyi` TXT query; cache result with DNS TTL
   c. Compute `hashed_local_part = SHA-256("alice" || 0x00 || domain_salt)`
   d. `cache.negative` miss, `cache.claim` miss
   e. `http.fetch_claim("q8.fyi", hashed_local_part)` → 200, CBOR `SignedClaim`
   f. `cache.revocation("q8.fyi")` miss → `http.fetch_revocation_list("q8.fyi")` → 200 or 404 (cache as empty)
   g. `SignedClaim::verify(&domain_record, &revocation_view, now)` → `VerifiedBinding`
   h. Populate `signing_key`, `claim`, `revocation`, `domain_last_seen` caches
   i. Return `ResolveOutcome::Resolved(identity_hash)`
4. `run_async_address_resolution` emits `SmtpEvent::HarmonyResolved{ identity: Some(hash) }`
5. State machine: `resolved_recipients.push(hash)`, `250 OK`

Cost: 1 DNS query + 2 HTTPS GETs. Subsequent resolutions at same domain: 0 network calls until TTLs expire.

### 6.2 Hot path — cache hit

1. `cache.master_key` hit (within DNS TTL)
2. `cache.claim` hit (within `expires_at`)
3. `cache.revocation` hit (not stale)
4. Re-verify cached claim against cached revocation view (cheap; Ed25519 ~50µs) — catches revocations that arrived via background refresh since last hit
5. Return `Resolved`

Cost: 0 network. 1 cert-sig + 1 claim-sig verify.

### 6.3 Background revocation refresh

Every 15 min, background task:
1. `cache.evict_expired(now)` across all caches
2. For each domain in `revocation` cache with `last_refreshed + 6h < now`, spawn `http.fetch_revocation_list(domain)` (per-domain concurrency cap = 1)
3. On success: update cache with `now + 6h`
4. On failure: leave previous cache entry in place; increment failure counter
5. If no successful refresh in >24h, mark domain `revocation_stale`; `resolve()` returns `Transient` until recovery

### 6.4 Revocation-triggered rejection

- T+0: `q8.fyi` operator publishes updated `/.well-known/harmony-revocations` with new revocation cert
- T+0..6h: cached claims still serve as `Resolved` (operator responsible for simultaneously issuing fresh claims with `serial: N+1` if they need faster propagation)
- T+6h: background refresh picks up updated list; cache updated
- T+6h+ε: next `resolve()` → cache hit → re-verify sees revoked `signing_key_id` → check grandfathering → reject with `Revoked` (if `claim.issued_at > revoked_cert.valid_until`) or continue (if grandfathered)

Worst-case propagation lag: 6 hours.

### 6.5 Soft-fail window

- `q8.fyi` was seen participating yesterday (`domain_last_seen` within 72h)
- Operator breaks DNS; TXT record now malformed
- `cache.master_key` miss → `dns.fetch_domain_record` → `DnsFetchError::Malformed`
- `was_domain_recently_seen("q8.fyi", now, 72h)` → true
- Return `Transient{ reason: "dns_malformed_within_soft_fail" }` → SMTP 451
- Sender MTA retries; if operator fixes DNS within 72h, recovery
- If 72h elapses with no recovery, `domain_last_seen` expires; subsequent resolutions return `DomainDoesNotParticipate` → SMTP 550

### 6.6 Layering invariant

> Every `ResolveOutcome` other than `Resolved(hash)` produces an SMTP rejection at RCPT TO and leaves `resolved_recipients` untouched. The existing `DeliverToHarmony` construction + seal + publish chain from PR #240 does not change. This PR is purely upstream of the `remote_accepted` logic.

---

## 7. Error Handling

### 7.1 Philosophy

**Default to `Transient`. Reserve `DomainDoesNotParticipate` and `UserUnknown` for cryptographically-clean negative answers. Reserve `Revoked` for explicit operator intent.** Any inconclusive evidence (malformed DNS, HTTPS 5xx, CBOR broken, signature invalid) is `Transient` — sending MTAs retry, and retries are cheaper than permanent bounces on inconclusive evidence.

### 7.2 DNS failure mapping

| Condition | `ResolveOutcome` | SMTP |
|---|---|---|
| NXDOMAIN / NOERROR-no-TXT, domain within 72h soft-fail | `Transient{"dns_no_record_soft_fail"}` | 451 |
| NXDOMAIN / NOERROR-no-TXT, domain outside soft-fail | `DomainDoesNotParticipate` | 550 |
| Multiple `v=harmony1` TXT records | `Transient{"dns_multiple_records"}` | 451 |
| Malformed required fields, within soft-fail | `Transient{"dns_malformed"}` | 451 |
| Malformed required fields, outside soft-fail | `DomainDoesNotParticipate` | 550 |
| Unknown `v=` version | `Transient{"dns_unsupported_version"}` | 451 |
| DNS timeout / SERVFAIL | `Transient{"dns_timeout"}` or `Transient{"dns_error"}` | 451 |
| Domain name syntactically invalid (pre-flight) | `UserUnknown` | 550 |

### 7.3 HTTPS (claim fetch) failure mapping

| Condition | `ResolveOutcome` | Negative cache? |
|---|---|---|
| 200 + valid CBOR claim | → verify step | positive cache on success |
| 200 + malformed CBOR | `Transient{"claim_parse"}` | no |
| 200 + verification fails | `Transient{"claim_invalid"}` | no |
| 404 | `UserUnknown` | yes, 60s |
| 5xx / connection refused / TLS failure / timeout | `Transient{"http_*"}` | no |
| 3xx redirect | `Transient{"http_redirect_refused"}` | no |
| Body > 1 MB | `Transient{"claim_oversize"}` | no |
| TLS cert validation failure | `Transient{"tls_invalid"}` | no |

### 7.4 Revocation-list fetch failure mapping

| Condition | Effect |
|---|---|
| 200 + valid `RevocationList` | cache `now + 6h` |
| 200 + malformed | keep previous cache; retry next cycle |
| 404 | cache as authoritative empty list |
| 5xx / timeout / TLS failure | keep previous cache; `failed_since = now` |
| Master-signature fails | log at ERROR (potential attack); discard; keep previous |
| No successful fetch >24h | mark `revocation_stale`; `resolve()` returns `Transient{"revocation_stale"}` until recovery |
| **No cached list ever** + fetch fails | `Transient{"revocation_bootstrap_failed"}` — fail closed on first-ever bootstrap |

### 7.5 Claim verification failure mapping (from §5.1 `VerifyError`)

| `VerifyError` | `ResolveOutcome` | Log level |
|---|---|---|
| `DomainMismatch` | `Transient{"claim_domain_mismatch"}` | WARN |
| `HashedLocalPartMismatch` | `Transient{"claim_lp_mismatch"}` | WARN |
| `CertSignatureInvalid` | `Transient{"cert_sig"}` (60s neg-cache) | WARN |
| `ClaimSignatureInvalid` | `Transient{"claim_sig"}` (60s neg-cache) | WARN |
| `CertNotYetValid` | `Transient{"cert_future"}` | INFO |
| `CertExpired` | `Transient{"cert_expired"}` | WARN |
| `CertRevoked` | `Revoked` | INFO |
| `ClaimExpired` | `Transient{"claim_expired"}` | INFO |
| `UnsupportedVersion(n)` | `Transient{"unsupported_version"}` | WARN |
| `UnsupportedAlgorithm` | `Transient{"unsupported_alg"}` | WARN |

### 7.6 Clock-skew tolerance

Single constant `CLOCK_SKEW_TOLERANCE_SECS = 60` (configurable via `ResolverConfig`) applied to `cert.valid_from`, `cert.valid_until`, `claim.expires_at`. Not per-field.

### 7.7 Serial-number enforcement

On cache hit during re-verify, reject new claim with `serial < cached_claim.serial` as `Transient{"claim_serial_rollback"}`. First-time resolutions have no prior serial and accept any valid claim (inherent limitation of single-server lookups; a future per-email signature chain would close this gap by binding each claim to its predecessor's hash).

### 7.8 Resource exhaustion protection

| Attack | Defense |
|---|---|
| Cache-fill via many random local_parts | Per-cache LRU bound (default 10k entries) |
| DNS/HTTP flood via many random domains | SMTP connection rate limits (existing infra) |
| Oversized response body | 1 MB cap in `http.rs` |
| Slow-loris response | 10s total HTTP timeout, 5s DNS timeout |
| CBOR bomb (deep nesting) | `ciborium` depth limit (128, explicit) |
| Revocation refresh stampede | Per-domain concurrency cap 1; single bounded background task |

### 7.9 Observability

Metrics (via existing tracing/metrics infra):

- `email_resolve_outcomes_total{outcome="resolved"|"unknown"|"transient"|"revoked"|"no_participate"}`
- `email_resolve_cache_hits_total{cache="claim"|"signing_key"|"master_key"|"revocation"|"negative"}`
- `email_resolve_latency_seconds` (histogram)
- `revocation_refresh_failures_total{domain}`
- `revocation_stale_domains` (gauge)

Logs: every `Transient` at WARN with `reason` + `(domain, email)`. `Revoked` at INFO with `signing_key_id`. Attack-indicative failures (cross-domain substitution, revocation signature failure) at ERROR with full context. No PII beyond the email address. Claim bodies off by default.

### 7.10 Panic surface

- No `.unwrap()` / `.expect()` on parse paths (clippy-deny crate-wide)
- `cargo-fuzz` target lands for `claim.rs` deserialization + verification (not a CI gate)
- Checked/saturating integer ops for all arithmetic on untrusted fields (`expires_at + tolerance`, `serial`)

---

## 8. Testing

### 8.1 Unit tests — `harmony-mail-discovery`

`claim.rs` verification: ~17 tests, one per `VerifyError` variant plus happy path, time-boundary tolerance, revocation grandfathering, serial rollback. Pattern:

```rust
fn test_claim(mods: impl FnOnce(&mut ClaimBuilder)) -> SignedClaim { /* ... */ }

#[test]
fn rejects_invalid_claim_signature() {
    let mut claim = test_claim(|_| {});
    claim.claim_signature = Signature::Ed25519([0u8; 64]);
    let err = claim.verify(&valid_domain_record(), &empty_revocations(), now()).unwrap_err();
    assert!(matches!(err, VerifyError::ClaimSignatureInvalid));
}
```

`cache.rs`: ~6 tests. Injected `TimeSource`; verify TTL expiry, LRU eviction, negative cache, soft-fail window.

`dns.rs` / `http.rs` parsers: ~2 tests. Hand-constructed bytes → parsed structs or errors.

**No network I/O.** Coverage target: every `match` arm in `claim.rs` + `cache.rs`.

### 8.2 Component tests — `resolver.rs`

~17 tests using `FakeDnsClient` + `FakeHttpClient` + `FakeTimeSource`:

```
cold_path_resolves_and_caches
hot_path_serves_from_cache_without_network
dns_nxdomain_returns_domain_does_not_participate
dns_nxdomain_within_soft_fail_window_returns_transient
dns_timeout_returns_transient
dns_malformed_returns_transient
http_404_returns_user_unknown
http_404_cached_for_60s_negative
http_5xx_returns_transient
revocation_bootstrap_failure_fails_closed
revocation_list_cached_and_reused
claim_signed_by_revoked_cert_returns_revoked
claim_issued_before_revocation_is_grandfathered
background_refresh_updates_revocation_cache
stale_revocation_past_24h_returns_transient
claim_serial_rollback_returns_transient
soft_fail_expires_after_72h
```

`FakeTimeSource` is step-mutable — `time.advance(Duration::from_secs(6 * 3600))` — so time-bound behavior is testable without wall-clock waits.

### 8.3 Integration tests — `harmony-mail`

New file `crates/harmony-mail/tests/smtp_remote_rcpt_admission_integration.rs`:

- `smtp_rcpt_to_remote_domain_resolves_seals_publishes` — end-to-end: fake HTTP server + fake DNS + full `SmtpSession` with `RemoteDeliveryContext`; assert exactly-one sealed envelope captured; assert `HarmonyEnvelope::open` with recipient's `PrivateIdentity` recovers plaintext
- `smtp_rcpt_to_unknown_user_returns_550`
- `smtp_rcpt_to_non_participating_domain_returns_550`
- `smtp_rcpt_to_transient_dns_failure_returns_451`
- `smtp_rcpt_to_revoked_recipient_returns_550`
- `smtp_rcpt_to_mixed_local_and_remote_recipients_succeeds_per_recipient` — preserves PR #240's per-recipient semantics

### 8.4 Explicitly NOT tested

- Real DNS (tests hickory, not us; inject at `DnsClient` trait boundary)
- Real HTTPS endpoints (tests reqwest, not us; inject at `HttpClient` boundary)
- TLS cert validation (tests reqwest defaults)
- Ed25519 cryptographic correctness (tests `ed25519-dalek`)
- CBOR fuzz (target lands; CI gating is future work)
- End-to-end with real external domain (ops test; future work)

### 8.5 Tests that must keep passing

- `smtp_remote_delivery_integration.rs` (PR #240) — unchanged; bypasses RCPT via direct `DeliverToHarmony`
- PR #240's three unit tests (`remote_only_250`, `resolver_miss_451`, `remote_not_configured_451`) — mechanical update to use `RemoteDeliveryContext` instead of three `Option`s
- All 386 pre-existing `harmony-mail` tests — unchanged (local-delivery pathways don't touch `EmailResolver`)

### 8.6 Sizing

| Layer | New | Modified |
|---|---|---|
| Unit (`harmony-mail-discovery`) | ~25 | 0 |
| Component (`harmony-mail-discovery::resolver`) | ~17 | 0 |
| Integration (`harmony-mail`) | ~6 | 0 |
| PR #240 adjustments | 0 | ~3 |
| **Total** | **~48** | **~3** |

Execution time: <10s added to the existing <30s `harmony-mail` test suite.

---

## 9. Migration Path to Phase 2

This spec intentionally shapes Phase 1 so Phase 2 is purely additive:

| Phase 2 addition | Phase 1 enablement |
|---|---|
| Zenoh distribution of claims | Same `SignedClaim` wire format; new transport only |
| Zenoh distribution of revocation lists | Same `RevocationList` wire format; new transport only |
| Per-email signature chain (equivocation defense) | `ClaimPayload.version` bumps; old verifiers cleanly reject |
| Key Transparency / Merkle-proof layer (domain-level non-equivocation) | `ClaimPayload.version` bumps |
| PQ master / signing keys | `Signature` enum + `alg` field both extensible |
| Bundle endpoint for batched fetch | New endpoint; no change to existing `SignedClaim` shape |
| Hybrid Ed25519 + ML-DSA signatures | `Signature::Hybrid(...)` variant |

The only forward-compat boundary we deliberately DO NOT leave open is forward-compat across unknown version bytes. Every artifact has its own version byte; unknown versions are rejected rather than silently accepted. This is a deliberate choice: silent forward-compat is a cryptographic footgun.

---

## 10. Open Questions (Deferred)

Per Gemini research §8 and confirmed as follow-up work:

1. **Equivocation detection without heavy gossip.** If a domain's signing key is silently doubling up — signing one hash for sender A, another for sender B — we have no detection mechanism in Phase 1 or Phase 2 as specified. A per-email signature chain (`previous_claim_hash` field in a future `ClaimPayload` version) catches this within a single email's history; a domain-level Key Transparency layer (Meta's AKD / SEEMless shape) catches it across all users. Deferred until a concrete requirement or observed attack justifies the complexity.

2. **Per-user binding revocation.** The revocation list handles signing-key compromise (domain-wide) but not per-user binding revocation (e.g., "Alice's device was compromised, her old hash is now invalid"). Today the only mechanism is claim-expiry (up to 7 days) or roll-forward via a higher-serial claim that senders happen to re-fetch. A per-user revocation mechanism is deferred; the immediate guidance is that operators should reduce claim expiry (or trigger a domain-wide signing-key rotation) as the blunt-force answer to urgent user-level revocation.

3. **Private-information-retrieval rollout threshold.** ZipPIR maturity plus AVX-512 VNNI availability on gateway hardware determines when full private lookup (where the domain learns nothing about who was queried) becomes practical. Continue monitoring; not a Phase 2 blocker, but a Phase 3 candidate.

---

## 11. References

- **Linear:** [ZEB-120](https://linear.app/zeblith/issue/ZEB-120) (this ticket), [ZEB-113](https://linear.app/zeblith/issue/ZEB-113) (parent)
- **PR:** [#240 (ZEB-113 PR A)](https://github.com/zeblithic/harmony/pull/240)
- **Plan seed:** `docs/plans/2026-04-15-smtp-remote-delivery-plan.md` (PR A plan)
- **Code anchors:**
  - `crates/harmony-mail/src/server.rs:1069-1133` — RCPT admission entry point
  - `crates/harmony-mail/src/remote_delivery.rs:127-147` — `RecipientResolver` trait
  - `crates/harmony-discovery/src/record.rs:52-64` — `AnnounceRecord`
  - `crates/harmony-discovery/src/verify.rs:44-55` — address-hash binding check
  - `crates/harmony-zenoh/src/namespace.rs::msg::unicast_key` — PR #240's unicast topic
- **Research:** `docs/research/2026-04-15-email-hash-resolution-gemini-research.md` — Gemini Deep Research report on email-style identifier resolution in decentralized messaging; archived verbatim alongside this spec
- **Prior-art references (from research):** Nostr NIP-05, Matrix MSC2134, Signal CDS / Sealed Sender, WebFinger RFC 7033, did:web / did:webvh, DKIM RFC 6376, Keybase/Zoom sigchains, CONIKS / SEEMless / Parakeet / AKD, ZipPIR
