# ZEB-380 PR 1 — `harmony-pkarr` hot-swappable relay pool + per-relay health Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `harmony-pkarr`'s `RelayClient` pool hot-swappable at runtime, make its request-timeout / cooldown configurable, and expose a synchronous per-relay health accessor — the enabling core change for the harmony-client multi-relay manager (PR 2).

**Architecture:** `RelayClient` currently owns an immutable `pool: RelayPool` and two `const`s (`REQUEST_TIMEOUT`, `COOLDOWN`). This plan (a) replaces the consts with a `RelayConfig` supplied at construction, (b) wraps the pool in `std::sync::RwLock` so `set_relays(&self, …)` can swap it live (through an `Arc`, no `&mut`), and (c) adds a per-URL `RelayRecord` map plus a `relay_health() -> Vec<RelayHealth>` accessor that derives each relay's `Healthy`/`CoolingDown` state from the existing cooldown map. No behavior change to the publish/resolve hot path; the existing fall-through and cooldown semantics are preserved verbatim.

**Tech Stack:** Rust, `std::sync::{RwLock, Mutex}`, `reqwest`, `serde` (derive), `tokio` (tests). Crate: `crates/harmony-pkarr`.

**Spec:** `harmony-client/docs/specs/2026-06-05-zeb-380-configurable-multi-relay-pool-design.md` §4.

**Repo / branch:** `harmony` repo. Create branch `zeb-380-relay-pool-hotswap` off latest `origin/main`.

**Per-task gate (fast — small crate):** commit FIRST, then from `/Users/zeblith/work/zeblithic/harmony`:
```bash
cargo fmt -p harmony-pkarr -- --check \
  && cargo clippy -p harmony-pkarr --all-targets -- -D warnings \
  && cargo nextest run -p harmony-pkarr
```
Use `set -o pipefail` / check `${PIPESTATUS[0]}` if piping. 10-minute wall-clock kill switch per command; if a command stalls past that, commit WIP and report `DONE_WITH_CONCERNS`. If genuinely blocked, report `BLOCKED` rather than thrashing.

---

## File Structure

- **Modify:** `crates/harmony-pkarr/src/relay.rs` — all core changes (config, RwLock pool, health types + accessor). This is the only source file with logic changes; it stays a single focused module (~400 lines after changes — still cohesive).
- **Modify:** `crates/harmony-pkarr/src/lib.rs` — extend the `pub use relay::{…}` re-export with the new public types.

All tests live inline in `relay.rs`'s `#[cfg(test)] mod tests` (matching the existing convention — no separate test file).

---

## Task 1: `RelayConfig` — configurable timeout + cooldown

Replace the two module consts with a config struct supplied at construction. `RelayClient::new(pool)` keeps its signature (delegates to `with_config` with defaults); add `with_config`. **Not user-facing** — exists so tests use short values and a future ticket can wire knobs.

**Files:**
- Modify: `crates/harmony-pkarr/src/relay.rs`

- [ ] **Step 1: Write the failing test**

Add to the `#[cfg(test)] mod tests` block in `relay.rs`:

```rust
#[test]
fn relay_config_default_is_5s_30s() {
    let cfg = RelayConfig::default();
    assert_eq!(cfg.request_timeout, Duration::from_secs(5));
    assert_eq!(cfg.cooldown, Duration::from_secs(30));
}

#[tokio::test]
async fn with_config_short_cooldown_is_honored() {
    // A relay that fails goes on cooldown for the CONFIGURED duration.
    // With a 0ms cooldown the failed relay is immediately available again.
    let cfg = RelayConfig {
        request_timeout: Duration::from_millis(200),
        cooldown: Duration::from_millis(0),
    };
    let pool = RelayPool::new(vec!["http://192.0.2.1:80".to_string()]);
    let client = RelayClient::with_config(pool, cfg);
    // First get marks the unreachable relay on cooldown, then (cooldown=0)
    // it is available again on the next call — so we still see the relay,
    // not an empty pool. The point is `with_config` compiles + plumbs cfg.
    let _ = client.get("k").await; // Err(NoRelaysAvailable) or transport err
    assert_eq!(client.relay_health().len(), 1);
}
```

(The second test also forward-references `relay_health` from Task 3 — order Task 1 → 2 → 3, and this test compiles once Task 3 lands. If executing strictly per-task, gate Task 1 on `relay_config_default_is_5s_30s` alone and add the second assertion's body in Task 3. To keep it simple: write `relay_config_default_is_5s_30s` now; defer `with_config_short_cooldown_is_honored` to Task 3's test batch.)

- [ ] **Step 2: Run the default test, verify it fails**

Run: `cargo nextest run -p harmony-pkarr -E 'test(relay_config_default_is_5s_30s)'`
Expected: FAIL — `RelayConfig` does not exist.

- [ ] **Step 3: Implement `RelayConfig` + `with_config`**

In `relay.rs`, replace the two consts:

```rust
/// Per-request timeout for relay HTTP calls.
const REQUEST_TIMEOUT: Duration = Duration::from_secs(5);

/// How long a relay stays on cooldown after a timeout or 429 response.
const COOLDOWN: Duration = Duration::from_secs(30);
```

with:

```rust
/// Tunable timeouts for [`RelayClient`]. **Not user-facing** — multi-relay
/// redundancy (ZEB-380) already removes the "5 s timeout is terminal" failure
/// mode, so these stay code-level defaults. The struct exists so tests can use
/// short values and a future ticket can wire knobs if needed.
#[derive(Debug, Clone, Copy)]
pub struct RelayConfig {
    /// Per-request HTTP timeout. Default 5 s.
    pub request_timeout: Duration,
    /// How long a relay stays on cooldown after a timeout / 429 / 5xx. Default 30 s.
    pub cooldown: Duration,
}

impl Default for RelayConfig {
    fn default() -> Self {
        Self {
            request_timeout: Duration::from_secs(5),
            cooldown: Duration::from_secs(30),
        }
    }
}
```

Add a `config: RelayConfig` field to `RelayClient` (full struct rewrite happens in Task 2/3; for now add the field). Change construction:

```rust
impl RelayClient {
    /// Build a client with default [`RelayConfig`].
    pub fn new(pool: RelayPool) -> Self {
        Self::with_config(pool, RelayConfig::default())
    }

    /// Build a client with an explicit [`RelayConfig`] (test/forward-compat hook).
    pub fn with_config(pool: RelayPool, config: RelayConfig) -> Self {
        let http = reqwest::Client::builder()
            .timeout(config.request_timeout)
            // ZEB-381: trust Mozilla's webpki root bundle in addition to OS-native
            // roots. `rustls-tls-native-roots` alone failed to anchor relay.pkarr.org's
            // Let's Encrypt chain — InvalidCertificate(UnknownIssuer) — on BOTH macOS
            // and Windows, breaking every pkarr publish/resolve and thus every
            // first-contact invite redeem. webpki roots carry ISRG Root X1/X2 and
            // validate the chain; native stays enabled for any private/enterprise roots.
            .tls_built_in_webpki_certs(true)
            .build()
            .expect("reqwest client build should never fail with default settings");
        Self {
            pool, // becomes RwLock in Task 2
            http,
            config,
            cooldown: Mutex::new(HashMap::new()),
            // records: added in Task 3
        }
    }
    // ... existing put/get/available_relays/mark_cooldown unchanged for now
    // except mark_cooldown uses self.config.cooldown (Step 4).
}
```

- [ ] **Step 4: Replace `COOLDOWN` const usage**

In `mark_cooldown`, change `Instant::now() + COOLDOWN` to `Instant::now() + self.config.cooldown`:

```rust
fn mark_cooldown(&self, base: &str) {
    let mut cd = self.cooldown.lock().expect("cooldown poisoned");
    cd.insert(base.to_string(), Instant::now() + self.config.cooldown);
}
```

The `REQUEST_TIMEOUT` const is now unused (timeout moved into `with_config`); delete it. `COOLDOWN` const is now unused; delete it.

- [ ] **Step 5: Run the default test, verify it passes**

Run: `cargo nextest run -p harmony-pkarr -E 'test(relay_config_default_is_5s_30s)'`
Expected: PASS.

- [ ] **Step 6: Full crate gate + commit**

```bash
cargo fmt -p harmony-pkarr -- --check \
  && cargo clippy -p harmony-pkarr --all-targets -- -D warnings \
  && cargo nextest run -p harmony-pkarr
git add crates/harmony-pkarr/src/relay.rs
git commit -m "feat(zeb-380): RelayConfig — configurable relay timeout + cooldown"
```
Expected: all green, existing relay tests still pass.

---

## Task 2: Hot-swappable pool (`RwLock<RelayPool>` + `set_relays`)

Wrap the pool so it can be replaced live through the shared `Arc<RelayClient>`. `available_relays` reads a clone under a short read-lock; `set_relays` replaces the guarded value.

**Files:**
- Modify: `crates/harmony-pkarr/src/relay.rs`

- [ ] **Step 1: Write the failing test**

```rust
#[tokio::test]
async fn set_relays_takes_effect_mid_flight() {
    // Publish to pool A (relay_a), then swap the pool to relay_b and confirm
    // the next get hits relay_b (which has the record) — proving the swap is live.
    let relay_a = MockPkarrRelay::start().await;
    let relay_b = MockPkarrRelay::start().await;

    let client = RelayClient::new(RelayPool::new(vec![relay_a.base_url.clone()]));
    client.put("k1", b"in-a").await.expect("put to A");

    // Swap pool to B (which does NOT have k1 yet).
    client.set_relays(vec![relay_b.base_url.clone()]);
    assert_eq!(client.get("k1").await.expect("get from B"), None); // B has no k1

    // Publish to B via the swapped pool, then read it back from B.
    client.put("k1", b"in-b").await.expect("put to B");
    assert_eq!(
        client.get("k1").await.expect("get from B"),
        Some(b"in-b".to_vec())
    );
}
```

- [ ] **Step 2: Run it, verify it fails**

Run: `cargo nextest run -p harmony-pkarr -E 'test(set_relays_takes_effect_mid_flight)'`
Expected: FAIL — `set_relays` does not exist.

- [ ] **Step 3: Wrap the pool field in `RwLock`**

Change the struct:

```rust
pub struct RelayClient {
    /// Hot-swappable relay pool. Read-mostly (every put/get takes a short read
    /// lock); replaced wholesale by `set_relays`. `std::sync::RwLock` keeps the
    /// crate dependency-free (no `arc-swap`).
    pool: std::sync::RwLock<RelayPool>,
    http: reqwest::Client,
    config: RelayConfig,
    /// Maps relay base URL → `Instant` at which the cooldown expires.
    cooldown: Mutex<HashMap<String, Instant>>,
    // records: added in Task 3
}
```

In `with_config`, wrap the pool: `pool: std::sync::RwLock::new(pool),`.

Add `RwLock` to the existing `use std::sync::Mutex;` import → `use std::sync::{Mutex, RwLock};` and reference as `RwLock` (or keep the fully-qualified `std::sync::RwLock` — match the file's import style; the file currently imports `Mutex` bare, so add `RwLock` to that line).

- [ ] **Step 4: `available_relays` reads under the lock**

```rust
fn available_relays(&self) -> Vec<String> {
    let now = Instant::now();
    let cd = self.cooldown.lock().expect("cooldown poisoned");
    let pool = self.pool.read().expect("relay pool poisoned");
    pool.relays
        .iter()
        .filter(|r| cd.get(r.as_str()).is_none_or(|expiry| *expiry <= now))
        .cloned()
        .collect()
}
```

(Locks are acquired cooldown-then-pool and both released at end of function; no other code path holds both, so no deadlock risk. The clone-out keeps the lock hold short.)

- [ ] **Step 5: Add `set_relays`**

```rust
/// Replace the relay pool live. Takes effect on the next `put`/`get`.
///
/// Cooldown entries keyed on a now-removed relay simply never match a live
/// relay again and age out — harmless, no explicit pruning needed.
pub fn set_relays(&self, relays: Vec<String>) {
    *self.pool.write().expect("relay pool poisoned") = RelayPool::new(relays);
}
```

- [ ] **Step 6: Run the swap test + existing tests, verify pass**

Run: `cargo nextest run -p harmony-pkarr`
Expected: PASS — `set_relays_takes_effect_mid_flight`, `unreachable_relay_falls_through_to_next`, `all_relays_unavailable_yields_error`, `put_then_get_via_pool`, `get_returns_none_on_404` all green.

- [ ] **Step 7: Gate + commit**

```bash
cargo fmt -p harmony-pkarr -- --check \
  && cargo clippy -p harmony-pkarr --all-targets -- -D warnings \
  && cargo nextest run -p harmony-pkarr
git add crates/harmony-pkarr/src/relay.rs
git commit -m "feat(zeb-380): hot-swappable RelayPool via RwLock + set_relays"
```

---

## Task 3: Per-relay health types + `relay_health()` accessor

Add the `RelayHealth` / `RelayState` / `RelayOutcome` wire types, a per-URL `RelayRecord` map updated on each put/get branch, a `now_ms()` wall-clock helper, and the synchronous `relay_health()` accessor. `state` is derived from the cooldown map at read time; `until_ms` is the cooldown expiry converted to wall-clock millis.

**Design note (minor, honest refinement over spec §4.3):** `RelayOutcome` gains a `Transport` variant in addition to `Timeout`. `reqwest::Error::is_timeout()` distinguishes a genuine timeout from a connection-refused / DNS failure; collapsing both into `Timeout` would mislabel the health badge. Both still trip cooldown identically. A `404` on GET updates **no** record (the relay answered correctly; it is reachable and not failing — neither a success-for-us nor an error), so a relay that only ever 404s shows `Healthy` with `last_outcome: None`.

**Files:**
- Modify: `crates/harmony-pkarr/src/relay.rs`

- [ ] **Step 1: Write the failing tests**

```rust
#[tokio::test]
async fn relay_health_lists_pool_relays_healthy_by_default() {
    let relay = MockPkarrRelay::start().await;
    let client = RelayClient::new(RelayPool::new(vec![relay.base_url.clone()]));
    let health = client.relay_health();
    assert_eq!(health.len(), 1);
    assert_eq!(health[0].url, relay.base_url);
    assert_eq!(health[0].state, RelayState::Healthy);
    assert_eq!(health[0].last_outcome, None);
    assert_eq!(health[0].last_success_ms, None);
}

#[tokio::test]
async fn relay_health_records_success_after_put() {
    let relay = MockPkarrRelay::start().await;
    let client = RelayClient::new(RelayPool::new(vec![relay.base_url.clone()]));
    client.put("k1", b"v").await.expect("put");
    let h = &client.relay_health()[0];
    assert_eq!(h.state, RelayState::Healthy);
    assert_eq!(h.last_outcome, Some(RelayOutcome::Success));
    assert!(h.last_success_ms.is_some());
}

#[tokio::test]
async fn relay_health_reflects_cooldown_after_transport_failure() {
    // Unreachable relay → cooldown → state == CoolingDown with a future until_ms.
    let client = RelayClient::new(RelayPool::new(vec!["http://192.0.2.1:80".to_string()]));
    let _ = client.get("k").await; // trips cooldown on the unreachable relay
    let h = &client.relay_health()[0];
    match h.state {
        RelayState::CoolingDown { until_ms } => assert!(until_ms > 0),
        RelayState::Healthy => panic!("expected CoolingDown after transport failure"),
    }
    // A connection refused / DNS error is Transport, not Timeout.
    assert_eq!(h.last_outcome, Some(RelayOutcome::Transport));
}

#[tokio::test]
async fn relay_health_records_http_status_on_5xx() {
    // MockPkarrRelay supports a status-forcing variant; if not available in the
    // mock, this test uses a relay that 500s. See MockPkarrRelay API below —
    // if the mock cannot force a 5xx, assert via a 429 path instead, or skip
    // and rely on relay_health_records_success_after_put + the cooldown test.
    // (Implementer: confirm MockPkarrRelay's surface during Task 3; pick the
    // achievable assertion. Do NOT invent a mock API that doesn't exist.)
}

#[tokio::test]
async fn relay_health_only_lists_current_pool_after_swap() {
    let relay_a = MockPkarrRelay::start().await;
    let relay_b = MockPkarrRelay::start().await;
    let client = RelayClient::new(RelayPool::new(vec![relay_a.base_url.clone()]));
    client.set_relays(vec![relay_b.base_url.clone()]);
    let health = client.relay_health();
    assert_eq!(health.len(), 1);
    assert_eq!(health[0].url, relay_b.base_url);
}
```

> **Implementer:** before writing `relay_health_records_http_status_on_5xx`, read `crates/harmony-pkarr/src/testing.rs` to learn `MockPkarrRelay`'s actual API. If it can force a 5xx/429, assert `last_outcome == Some(RelayOutcome::Http(503))` (or the forced code). If it cannot, delete that test stub and rely on the `Success` + `Transport` + cooldown coverage already present. Do not fabricate a mock method.

- [ ] **Step 2: Run, verify they fail**

Run: `cargo nextest run -p harmony-pkarr -E 'test(relay_health)'`
Expected: FAIL — types + accessor don't exist.

- [ ] **Step 3: Add the health wire types**

At the top of `relay.rs` (after the `use` block), add `use serde::{Deserialize, Serialize};` and:

```rust
/// Whether a relay is currently usable or sitting out a cooldown.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum RelayState {
    /// Available for the next request.
    Healthy,
    /// On cooldown until `until_ms` (wall-clock Unix millis).
    CoolingDown { until_ms: u64 },
}

/// The last recorded result of talking to a relay.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum RelayOutcome {
    /// 2xx — record accepted (put) or returned (get).
    Success,
    /// Per-request timeout elapsed.
    Timeout,
    /// Non-timeout transport failure (connection refused, DNS, TLS).
    Transport,
    /// An explicit HTTP error status (429 / 5xx / other non-success).
    Http(u16),
}

/// A point-in-time health summary for one relay in the pool. One entry per
/// **current** pool relay (removed relays drop out; freshly added relays appear
/// with `last_outcome: None` until first use).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RelayHealth {
    pub url: String,
    pub state: RelayState,
    pub last_outcome: Option<RelayOutcome>,
    /// Wall-clock Unix millis of the last 2xx exchange (None if never).
    pub last_success_ms: Option<u64>,
}

/// Per-URL mutable record folded behind a `Mutex`. `state` is NOT stored here —
/// it is derived from the cooldown map at read time so there is a single source
/// of truth for "is this relay cooling down".
#[derive(Debug, Default, Clone)]
struct RelayRecord {
    last_outcome: Option<RelayOutcome>,
    last_success_ms: Option<u64>,
}

/// Wall-clock Unix millis (mirrors `network_health.rs::now_ms`).
fn now_ms() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}
```

- [ ] **Step 4: Add the `records` field + record-update helper**

Add to the struct:

```rust
    /// Per-URL last-outcome / last-success record (health observability).
    records: Mutex<HashMap<String, RelayRecord>>,
```

Initialize in `with_config`: `records: Mutex::new(HashMap::new()),`.

Add a helper:

```rust
fn record_outcome(&self, base: &str, outcome: RelayOutcome) {
    let mut recs = self.records.lock().expect("records poisoned");
    let rec = recs.entry(base.to_string()).or_default();
    rec.last_outcome = Some(outcome);
    if outcome == RelayOutcome::Success {
        rec.last_success_ms = Some(now_ms());
    }
}
```

- [ ] **Step 5: Update put/get branches to record outcomes**

In `put`, add `self.record_outcome(&base, …)` to each branch:
- success branch (`resp.status().is_success()`): `self.record_outcome(&base, RelayOutcome::Success);` before `return Ok(())`.
- 429 branch: after `self.mark_cooldown(&base);` add `self.record_outcome(&base, RelayOutcome::Http(429));`.
- other non-success: `self.record_outcome(&base, RelayOutcome::Http(status));` (where `status` is the captured `resp.status().as_u16()`).
- `Err(e)` transport branch: replace `Err(_)` with `Err(e)` and record:
  ```rust
  Err(e) => {
      self.mark_cooldown(&base);
      self.record_outcome(
          &base,
          if e.is_timeout() { RelayOutcome::Timeout } else { RelayOutcome::Transport },
      );
      continue;
  }
  ```

In `get`, mirror it:
- success branch: `self.record_outcome(&base, RelayOutcome::Success);` before returning the bytes.
- 404 branch: **no record update** (leave the comment noting 404 is a healthy, neutral answer).
- 429 branch: `self.record_outcome(&base, RelayOutcome::Http(429));` after `mark_cooldown`.
- `Ok(_)` other-status branch: capture the status and `self.record_outcome(&base, RelayOutcome::Http(resp.status().as_u16()));`. (You'll need to bind `resp`: change `Ok(_)` to `Ok(resp)` — note the existing get has `Ok(resp) if … 429`, then `Ok(_)`; convert the catch-all `Ok(_)` to `Ok(resp)` to read its status.)
- `Err(e)` branch: `if e.is_timeout() { Timeout } else { Transport }`, same as put.

> **Implementer:** keep the existing control flow / comments byte-for-byte except for the added `record_outcome` calls and the `Err(_) → Err(e)` / `Ok(_) → Ok(resp)` bindings. Do not restructure the match arms.

- [ ] **Step 6: Add `relay_health()`**

```rust
impl RelayClient {
    /// Synchronous per-relay health for the current pool — one entry per pool
    /// relay, in pool order. `state` is derived from the cooldown map; a relay
    /// whose cooldown expiry is in the future reports `CoolingDown { until_ms }`
    /// (cooldown `Instant` converted to wall-clock millis), else `Healthy`.
    pub fn relay_health(&self) -> Vec<RelayHealth> {
        let now_inst = Instant::now();
        let now_wall = now_ms();
        let cd = self.cooldown.lock().expect("cooldown poisoned");
        let recs = self.records.lock().expect("records poisoned");
        let pool = self.pool.read().expect("relay pool poisoned");
        pool.relays
            .iter()
            .map(|url| {
                let state = match cd.get(url.as_str()) {
                    Some(expiry) if *expiry > now_inst => {
                        let remaining = expiry.duration_since(now_inst).as_millis() as u64;
                        RelayState::CoolingDown {
                            until_ms: now_wall + remaining,
                        }
                    }
                    _ => RelayState::Healthy,
                };
                let rec = recs.get(url).cloned().unwrap_or_default();
                RelayHealth {
                    url: url.clone(),
                    state,
                    last_outcome: rec.last_outcome,
                    last_success_ms: rec.last_success_ms,
                }
            })
            .collect()
    }
}
```

(Lock order here is cooldown → records → pool, all read-only and released together; `available_relays` takes cooldown → pool. No path takes records then cooldown, so the orders are consistent — no deadlock.)

- [ ] **Step 7: Run the health tests + full crate, verify pass**

Run: `cargo nextest run -p harmony-pkarr`
Expected: PASS (after resolving the `relay_health_records_http_status_on_5xx` stub per the note — keep it only if the mock supports forcing a status).

- [ ] **Step 8: Gate + commit**

```bash
cargo fmt -p harmony-pkarr -- --check \
  && cargo clippy -p harmony-pkarr --all-targets -- -D warnings \
  && cargo nextest run -p harmony-pkarr
git add crates/harmony-pkarr/src/relay.rs
git commit -m "feat(zeb-380): per-relay health types + relay_health() accessor"
```

---

## Task 4: Public re-exports + workspace sweep

Surface the new public types and verify the whole workspace still builds (the crate is a workspace member; downstream `cargo` consumers resolve these names).

**Files:**
- Modify: `crates/harmony-pkarr/src/lib.rs`

- [ ] **Step 1: Extend the re-export**

Change line 36:

```rust
pub use relay::{RelayClient, RelayPool};
```

to:

```rust
pub use relay::{RelayClient, RelayConfig, RelayHealth, RelayOutcome, RelayPool, RelayState};
```

- [ ] **Step 2: Workspace build + crate test gate**

```bash
cargo build -p harmony-pkarr
cargo fmt -p harmony-pkarr -- --check \
  && cargo clippy -p harmony-pkarr --all-targets -- -D warnings \
  && cargo nextest run -p harmony-pkarr
```
Expected: all green; the new names are publicly importable.

- [ ] **Step 3: Commit**

```bash
git add crates/harmony-pkarr/src/lib.rs
git commit -m "feat(zeb-380): export RelayConfig + RelayHealth/State/Outcome"
```

---

## Final review + PR

After all tasks: dispatch a final code review over the whole branch diff (`git diff origin/main...HEAD`), then open the PR in the **harmony** repo.

**PR title:** `ZEB-380 PR 1: hot-swappable relay pool + per-relay health (harmony-pkarr)`

**PR body must:**
- Summarize: `RelayConfig` (configurable timeout/cooldown), `RwLock`-swappable pool + `set_relays`, `RelayHealth`/`RelayState`/`RelayOutcome` + sync `relay_health()`; no hot-path behavior change.
- Reference the design spec path (`harmony-client/docs/specs/2026-06-05-zeb-380-configurable-multi-relay-pool-design.md`) and that the harmony-client consumer PR (PR 2) bumps to this merge SHA.
- Note this is the enabling half; user-facing relay management ships in harmony-client.
- **Linear auto-close:** ZEB-380 is a *harmony-client* ticket. Mention it as context but DO **not** put a closing keyword for it here — only PR 2 carries the closing reference. Keep no other open ticket IDs in the body.
- Test plan: per-task `cargo fmt -p harmony-pkarr --check` + `clippy -D warnings` + `nextest -p harmony-pkarr`.

**Bot loop:** autonomous — CodeRabbit / Cursor / CodeAnt / Qodo. **NEVER** trigger Greptile and **never** write the literal `@greptile` (paid; charges the user — write plain "Greptile"). Scan all three comment buckets (inline review threads, PR issue-comments, PR reviews). One bundled push per round. Pushover at ready-to-merge; **do NOT self-merge** (Jake's gate). Pushover instantly on a true blocker.
