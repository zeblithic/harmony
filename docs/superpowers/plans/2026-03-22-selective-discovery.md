# Selective Discovery Opt-In Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
> **Note:** The `- [ ]` checkboxes below are execution tracking markers for the agentic worker, not persistent TODO items. Task tracking uses `bd` (beads) — see bead `harmony-0kq`.

**Goal:** Restrict tunnel routing hints to authenticated peers by splitting announces into a public broadcast (Reticulum-only) and an authenticated Zenoh queryable (full tunnel hints, gated by Discovery UCAN token).

**Architecture:** Three changes: (1) add `Discovery = 7` to `CapabilityType` in `harmony-identity`, (2) add `discover` namespace module in `harmony-zenoh`, (3) declare `harmony/discover/**` queryable in `NodeRuntime` with a token-validated handler that serves full or public announce records. The runtime stores two pre-serialized records and uses `last_unix_now` for token time-bound checks.

**Tech Stack:** Rust, `harmony-identity` (CapabilityType, PqUcanToken), `harmony-zenoh` (namespace), `harmony-crypto` (MlDsaPublicKey), `harmony-node` (runtime.rs, event_loop.rs)

**Spec:** `docs/superpowers/specs/2026-03-22-selective-discovery-design.md`

---

## File Structure

```
crates/harmony-identity/src/
└── ucan.rs              — Add Discovery = 7 to CapabilityType

crates/harmony-zenoh/src/
└── namespace.rs          — Add discover module (PREFIX, SUB, key())

crates/harmony-node/src/
├── runtime.rs            — Discover queryable, handle_discover_query(),
│                           local announce fields, local_dsa_pubkey,
│                           last_unix_now, TimerTick update
└── event_loop.rs         — Inject unix_now into TimerTick
```

---

### Task 1: Add Discovery capability type

**Files:**
- Modify: `crates/harmony-identity/src/ucan.rs`

- [ ] **Step 1: Add Discovery variant to CapabilityType**

After `Compute = 6` (line 42), add:

```rust
    /// Right to discover an identity's full routing hints (tunnel addresses).
    Discovery = 7,
```

- [ ] **Step 2: Update TryFrom<u8>**

In the `impl TryFrom<u8> for CapabilityType` match (line 48-58), add before the wildcard arm:

```rust
            7 => Ok(Self::Discovery),
```

- [ ] **Step 3: Write test**

Add to the existing test module in `ucan.rs`:

```rust
    #[test]
    fn discovery_capability_type_roundtrip() {
        let val = CapabilityType::Discovery as u8;
        assert_eq!(val, 7);
        let restored = CapabilityType::try_from(val).unwrap();
        assert_eq!(restored, CapabilityType::Discovery);
    }
```

- [ ] **Step 4: Run tests**

Run: `RUST_MIN_STACK=8388608 cargo test -p harmony-identity discovery_capability`

- [ ] **Step 5: Commit**

```bash
git commit -m "feat(identity): add Discovery = 7 capability type for selective discovery"
```

---

### Task 2: Add discover namespace module

**Files:**
- Modify: `crates/harmony-zenoh/src/namespace.rs`

- [ ] **Step 1: Add discover module**

After the `identity` module (around line 430), add:

```rust
/// Authenticated discovery queryable namespace.
///
/// Peers query `harmony/discover/{identity_hash_hex}` with a PQ UCAN token
/// to retrieve full routing hints (including tunnel addresses). Without a
/// valid token, the queryable responds with public hints only.
///
/// Distinct from `harmony/identity/` which handles DID document resolution.
pub mod discover {
    use alloc::{format, string::String};

    /// Base prefix: `harmony/discover`
    pub const PREFIX: &str = "harmony/discover";

    /// Subscribe/queryable pattern: `harmony/discover/**`
    pub const SUB: &str = "harmony/discover/**";

    /// Key for a specific identity: `harmony/discover/{identity_hash_hex}`
    pub fn key(identity_hash_hex: &str) -> String {
        format!("{PREFIX}/{identity_hash_hex}")
    }
}
```

- [ ] **Step 2: Write test**

Add to the existing test module:

```rust
    #[test]
    fn discover_key() {
        assert_eq!(
            discover::key("aabbccdd"),
            "harmony/discover/aabbccdd"
        );
    }

    #[test]
    fn discover_subscription_pattern() {
        assert_eq!(discover::SUB, "harmony/discover/**");
    }
```

- [ ] **Step 3: Run tests**

Run: `cargo test -p harmony-zenoh discover`

- [ ] **Step 4: Commit**

```bash
git commit -m "feat(zenoh): add discover namespace for authenticated routing hint queries"
```

---

### Task 3: Add runtime scaffolding (fields, queryable, TimerTick unix_now)

**Files:**
- Modify: `crates/harmony-node/src/runtime.rs`
- Modify: `crates/harmony-node/src/event_loop.rs`

This task adds the structural changes without the query handler logic.

- [ ] **Step 1: Add `local_dsa_pubkey` to NodeConfig**

In `NodeConfig` (around line 41), add after `local_identity_hash`:

```rust
    /// This node's ML-DSA-65 public verifying key bytes.
    /// Used to verify Discovery UCAN tokens (which are issued by this node).
    /// Defaults to empty; must be set from the loaded identity at startup.
    pub local_dsa_pubkey: Vec<u8>,
```

Update `Default` impl to include `local_dsa_pubkey: Vec::new()`.

- [ ] **Step 2: Add fields to NodeRuntime**

After `pubkey_cache` (around line 575), add:

```rust
    // This node's identity hash (copied from config for direct access).
    local_identity_hash: harmony_identity::IdentityHash,
    // Queryable ID for the discover namespace (harmony/discover/**)
    discover_queryable_id: QueryableId,
    // Pre-serialized public announce record (Reticulum-only hints).
    // Served to unauthorized discover queries and broadcast publicly.
    local_public_announce: Option<Vec<u8>>,
    // Pre-serialized full announce record (all hints including tunnel).
    // Served to authorized discover queries with valid Discovery token.
    local_full_announce: Option<Vec<u8>>,
    // Unix epoch seconds, updated on each TimerTick for token time-bound checks.
    last_unix_now: u64,
    // This node's ML-DSA-65 public key for verifying self-issued Discovery tokens.
    local_dsa_pubkey: Vec<u8>,
```

- [ ] **Step 3: Declare discover queryable in constructor**

In `NodeRuntime::new()`, after the memo queryable declaration, add:

```rust
        // Discover namespace: register queryable for authenticated routing hint queries.
        let (discover_qid, _) = queryable_router
            .declare(harmony_zenoh::namespace::discover::SUB)
            .expect("static key expression must be valid");
        actions.push(RuntimeAction::DeclareQueryable {
            key_expr: harmony_zenoh::namespace::discover::SUB.to_string(),
        });
```

Initialize fields in `Self { ... }`:

```rust
            local_identity_hash: config.local_identity_hash,
            discover_queryable_id: discover_qid,
            local_public_announce: None,
            local_full_announce: None,
            last_unix_now: 0,
            local_dsa_pubkey: config.local_dsa_pubkey,
```

- [ ] **Step 4: Add unix_now to TimerTick**

Change `RuntimeEvent::TimerTick` from:

```rust
    TimerTick { now: u64 },
```

to:

```rust
    /// Tier 1: Periodic timer tick for path expiry, announce scheduling.
    /// `now` is monotonic millis-since-start. `unix_now` is Unix epoch seconds.
    TimerTick { now: u64, unix_now: u64 },
```

Update `push_event` handler for `TimerTick`:

```rust
            RuntimeEvent::TimerTick { now, unix_now } => {
                self.last_now = now;
                self.last_unix_now = unix_now;
                self.router_queue.push_back(NodeEvent::TimerTick { now });
            }
```

- [ ] **Step 5: Update event_loop.rs to inject unix_now**

In `event_loop.rs`, find where `TimerTick` events are constructed. Add `unix_now` from `SystemTime`:

```rust
use std::time::{SystemTime, UNIX_EPOCH};

// Where TimerTick is pushed:
let unix_now = SystemTime::now()
    .duration_since(UNIX_EPOCH)
    .map(|d| d.as_secs())
    .unwrap_or(0);
runtime.push_event(RuntimeEvent::TimerTick { now: now_ms(), unix_now });
```

- [ ] **Step 6: Add stub handler and route_query dispatch**

Add placeholder method:

```rust
    fn handle_discover_query(
        &self,
        query_id: u64,
        _key_expr: &str,
        _payload: &[u8],
    ) -> Vec<RuntimeAction> {
        // TODO: implement in Task 4
        Vec::new()
    }
```

Add dispatch in `route_query()`, after the `memo_queryable_id` arm:

```rust
                } else if queryable_id == self.discover_queryable_id {
                    let actions = self.handle_discover_query(query_id, &key_expr, &payload);
                    self.pending_direct_actions.extend(actions);
                }
```

- [ ] **Step 7: Add public accessors for test injection**

```rust
    /// Set the pre-serialized public announce record (Reticulum-only hints).
    pub fn set_local_public_announce(&mut self, data: Vec<u8>) {
        self.local_public_announce = Some(data);
    }

    /// Set the pre-serialized full announce record (all hints including tunnel).
    pub fn set_local_full_announce(&mut self, data: Vec<u8>) {
        self.local_full_announce = Some(data);
    }
```

- [ ] **Step 8: Fix ALL existing tests that construct TimerTick**

Every test that uses `RuntimeEvent::TimerTick { now: ... }` needs `unix_now` added. Search for `TimerTick {` in the test module. For all existing tests, add `unix_now: 0` since they don't care about Unix time:

```rust
// Before:
RuntimeEvent::TimerTick { now: 0 }
// After:
RuntimeEvent::TimerTick { now: 0, unix_now: 0 }
```

- [ ] **Step 9: Update constructor test assertion**

The `constructor_returns_startup_actions` test asserts queryable counts. Update:

```rust
        // 16 shard + 1 stats + 1 compute + 1 page + 1 memo + 1 discover = 21
        assert_eq!(queryable_count, 21);
```

Also update variant completeness tests to include the new `TimerTick` field and any new variants.

- [ ] **Step 10: Run tests**

Run: `RUST_MIN_STACK=8388608 cargo test -p harmony-node`

- [ ] **Step 11: Commit**

```bash
git commit -m "feat(runtime): discover queryable scaffolding, TimerTick unix_now, local announce fields

Declare queryable on harmony/discover/**. Add local_public_announce,
local_full_announce, last_unix_now, local_dsa_pubkey fields. Add
unix_now to TimerTick for token time-bound validation."
```

---

### Task 4: Implement handle_discover_query

**Files:**
- Modify: `crates/harmony-node/src/runtime.rs`

- [ ] **Step 1: Write tests**

```rust
    #[test]
    fn discover_query_missing_token_returns_public_hints() {
        let (mut rt, _) = make_runtime();

        // Set up pre-serialized announce records
        let public_data = b"public-announce-record".to_vec();
        let full_data = b"full-announce-record-with-tunnel".to_vec();
        rt.set_local_public_announce(public_data.clone());
        rt.set_local_full_announce(full_data.clone());

        // Query with empty payload (no token)
        let identity_hex = hex::encode(rt.local_identity_hash());
        let key_expr = harmony_zenoh::namespace::discover::key(&identity_hex);
        rt.push_event(RuntimeEvent::QueryReceived {
            query_id: 42,
            key_expr,
            payload: vec![],
        });

        let actions = rt.tick();
        let reply = actions
            .iter()
            .find(|a| matches!(a, RuntimeAction::SendReply { query_id: 42, .. }));
        assert!(reply.is_some(), "should reply with public hints");

        if let Some(RuntimeAction::SendReply { payload, .. }) = reply {
            assert_eq!(payload, &public_data, "should return public record");
        }
    }

    #[test]
    fn discover_query_valid_token_returns_full_hints() {
        use harmony_identity::PqPrivateIdentity;
        use rand::rngs::OsRng;

        let identity = PqPrivateIdentity::generate(&mut OsRng);
        let pub_id = identity.public_identity();
        let id_ref = harmony_identity::IdentityRef::from(pub_id);

        // Build config with this identity's hash and pubkey
        let mut config = NodeConfig::default();
        config.local_identity_hash = id_ref.hash;
        config.local_dsa_pubkey = pub_id.verifying_key.as_bytes();
        config.node_addr = hex::encode(id_ref.hash);
        let (mut rt, _) = NodeRuntime::new(config, MemoryBookStore::new());

        let public_data = b"public-only".to_vec();
        let full_data = b"full-with-tunnel".to_vec();
        rt.set_local_public_announce(public_data.clone());
        rt.set_local_full_announce(full_data.clone());

        // Set last_unix_now via TimerTick
        rt.push_event(RuntimeEvent::TimerTick { now: 1000, unix_now: 1500 });
        rt.tick();

        // Issue a Discovery token: local identity authorizes a peer
        let peer_hash = [0xBB; 16];
        let token = identity.issue_pq_root_token(
            &mut OsRng,
            &peer_hash,
            harmony_identity::CapabilityType::Discovery,
            &id_ref.hash,  // resource = our identity hash
            1000,   // not_before
            2000,   // expires_at
        )
        .unwrap();
        let token_bytes = token.to_bytes();

        let identity_hex = hex::encode(id_ref.hash);
        let key_expr = harmony_zenoh::namespace::discover::key(&identity_hex);
        rt.push_event(RuntimeEvent::QueryReceived {
            query_id: 43,
            key_expr,
            payload: token_bytes,
        });

        let actions = rt.tick();
        let reply = actions
            .iter()
            .find(|a| matches!(a, RuntimeAction::SendReply { query_id: 43, .. }));
        assert!(reply.is_some(), "should reply with full hints");

        if let Some(RuntimeAction::SendReply { payload, .. }) = reply {
            assert_eq!(payload, &full_data, "should return full record");
        }
    }

    #[test]
    fn discover_query_expired_token_returns_public_hints() {
        use harmony_identity::PqPrivateIdentity;
        use rand::rngs::OsRng;

        let identity = PqPrivateIdentity::generate(&mut OsRng);
        let pub_id = identity.public_identity();
        let id_ref = harmony_identity::IdentityRef::from(pub_id);

        let mut config = NodeConfig::default();
        config.local_identity_hash = id_ref.hash;
        config.local_dsa_pubkey = pub_id.verifying_key.as_bytes();
        config.node_addr = hex::encode(id_ref.hash);
        let (mut rt, _) = NodeRuntime::new(config, MemoryBookStore::new());

        let public_data = b"public-only".to_vec();
        let full_data = b"full-with-tunnel".to_vec();
        rt.set_local_public_announce(public_data.clone());
        rt.set_local_full_announce(full_data);

        // Set unix_now AFTER the token expires
        rt.push_event(RuntimeEvent::TimerTick { now: 1000, unix_now: 3000 });
        rt.tick();

        // Token expires at 2000, unix_now is 3000 → expired
        let peer_hash = [0xBB; 16];
        let token = identity.issue_pq_root_token(
            &mut OsRng,
            &peer_hash,
            harmony_identity::CapabilityType::Discovery,
            &id_ref.hash,
            1000,
            2000,  // expires_at = 2000
        )
        .unwrap();
        let token_bytes = token.to_bytes();

        let identity_hex = hex::encode(id_ref.hash);
        let key_expr = harmony_zenoh::namespace::discover::key(&identity_hex);
        rt.push_event(RuntimeEvent::QueryReceived {
            query_id: 44,
            key_expr,
            payload: token_bytes,
        });

        let actions = rt.tick();
        let reply = actions
            .iter()
            .find(|a| matches!(a, RuntimeAction::SendReply { query_id: 44, .. }));
        assert!(reply.is_some(), "should reply with public hints on expired token");

        if let Some(RuntimeAction::SendReply { payload, .. }) = reply {
            assert_eq!(payload, &public_data, "expired token → public record");
        }
    }

    #[test]
    fn discover_query_wrong_capability_returns_public_hints() {
        use harmony_identity::PqPrivateIdentity;
        use rand::rngs::OsRng;

        let identity = PqPrivateIdentity::generate(&mut OsRng);
        let pub_id = identity.public_identity();
        let id_ref = harmony_identity::IdentityRef::from(pub_id);

        let mut config = NodeConfig::default();
        config.local_identity_hash = id_ref.hash;
        config.local_dsa_pubkey = pub_id.verifying_key.as_bytes();
        config.node_addr = hex::encode(id_ref.hash);
        let (mut rt, _) = NodeRuntime::new(config, MemoryBookStore::new());

        let public_data = b"public-only".to_vec();
        rt.set_local_public_announce(public_data.clone());
        rt.set_local_full_announce(b"full".to_vec());

        rt.push_event(RuntimeEvent::TimerTick { now: 1000, unix_now: 1500 });
        rt.tick();

        // Issue a Content token instead of Discovery
        let peer_hash = [0xBB; 16];
        let token = identity.issue_pq_root_token(
            &mut OsRng,
            &peer_hash,
            harmony_identity::CapabilityType::Content,  // WRONG capability
            &id_ref.hash,
            1000,
            2000,
        )
        .unwrap();
        let token_bytes = token.to_bytes();

        let identity_hex = hex::encode(id_ref.hash);
        let key_expr = harmony_zenoh::namespace::discover::key(&identity_hex);
        rt.push_event(RuntimeEvent::QueryReceived {
            query_id: 45,
            key_expr,
            payload: token_bytes,
        });

        let actions = rt.tick();
        let reply = actions
            .iter()
            .find(|a| matches!(a, RuntimeAction::SendReply { query_id: 45, .. }));
        assert!(reply.is_some());

        if let Some(RuntimeAction::SendReply { payload, .. }) = reply {
            assert_eq!(payload, &public_data, "wrong capability → public record");
        }
    }

    #[test]
    fn discover_query_invalid_token_returns_public_hints() {
        let (mut rt, _) = make_runtime();

        let public_data = b"public-announce-record".to_vec();
        rt.set_local_public_announce(public_data.clone());
        rt.set_local_full_announce(b"full".to_vec());

        // Query with garbage token bytes (fails from_bytes)
        let identity_hex = hex::encode(rt.local_identity_hash());
        let key_expr = harmony_zenoh::namespace::discover::key(&identity_hex);
        rt.push_event(RuntimeEvent::QueryReceived {
            query_id: 47,
            key_expr,
            payload: vec![0xFF; 100], // garbage
        });

        let actions = rt.tick();
        let reply = actions
            .iter()
            .find(|a| matches!(a, RuntimeAction::SendReply { query_id: 47, .. }));
        assert!(reply.is_some(), "should reply with public hints on invalid token");

        if let Some(RuntimeAction::SendReply { payload, .. }) = reply {
            assert_eq!(payload, &public_data, "invalid token → public record");
        }
    }

    #[test]
    fn discover_query_wrong_identity_no_reply() {
        let (mut rt, _) = make_runtime();

        rt.set_local_public_announce(b"public".to_vec());
        rt.set_local_full_announce(b"full".to_vec());

        // Query for a different identity than ours
        let wrong_hex = hex::encode([0xFF; 16]);
        let key_expr = harmony_zenoh::namespace::discover::key(&wrong_hex);
        rt.push_event(RuntimeEvent::QueryReceived {
            query_id: 46,
            key_expr,
            payload: vec![],
        });

        let actions = rt.tick();
        let reply = actions
            .iter()
            .find(|a| matches!(a, RuntimeAction::SendReply { query_id: 46, .. }));
        assert!(reply.is_none(), "should not reply for unknown identity");
    }
```

Note: The tests use `rt.local_identity_hash()` — the `local_identity_hash` field is stored directly on `NodeRuntime` (added in Task 3). Add a public accessor:

```rust
    /// This node's identity hash.
    pub fn local_identity_hash(&self) -> [u8; 16] {
        self.local_identity_hash
    }
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `RUST_MIN_STACK=8388608 cargo test -p harmony-node discover_query`
Expected: FAIL — stub handler returns empty vec.

- [ ] **Step 3: Implement handle_discover_query**

Replace the stub:

```rust
    /// Handle a discover query: validate Discovery UCAN token, respond with
    /// full or public announce record.
    ///
    /// Key expression: `harmony/discover/{identity_hash_hex}`
    /// Payload: serialized PqUcanToken (or empty for public hints)
    /// Response: serialized AnnounceRecord bytes
    fn handle_discover_query(
        &self,
        query_id: u64,
        key_expr: &str,
        payload: &[u8],
    ) -> Vec<RuntimeAction> {
        // 1. Parse identity_hash from key expression
        let identity_hex = match key_expr
            .strip_prefix(harmony_zenoh::namespace::discover::PREFIX)
            .and_then(|s| s.strip_prefix('/'))
        {
            Some(hex) => hex.split('/').next().unwrap_or(""),
            None => return Vec::new(),
        };

        let id_bytes = match hex::decode(identity_hex) {
            Ok(b) if b.len() == 16 => b,
            _ => return Vec::new(),
        };
        let mut queried_hash = [0u8; 16];
        queried_hash.copy_from_slice(&id_bytes);

        // 2. Check if this is our identity
        let local_hash = self.local_identity_hash;
        if queried_hash != local_hash {
            return Vec::new(); // Not our identity — no reply
        }

        // 3. If no records available, no reply
        let public_bytes = match &self.local_public_announce {
            Some(b) => b,
            None => return Vec::new(),
        };

        // 4. Empty payload → public hints
        if payload.is_empty() {
            return vec![RuntimeAction::SendReply {
                query_id,
                payload: public_bytes.clone(),
            }];
        }

        // 5. Parse and validate Discovery UCAN token
        let full_bytes = match &self.local_full_announce {
            Some(b) => b,
            None => {
                return vec![RuntimeAction::SendReply {
                    query_id,
                    payload: public_bytes.clone(),
                }];
            }
        };

        // Size guard
        const MAX_TOKEN_BYTES: usize = 8 * 1024;
        if payload.len() > MAX_TOKEN_BYTES {
            return vec![RuntimeAction::SendReply {
                query_id,
                payload: public_bytes.clone(),
            }];
        }

        let token = match harmony_identity::PqUcanToken::from_bytes(payload) {
            Ok(t) => t,
            Err(_) => {
                return vec![RuntimeAction::SendReply {
                    query_id,
                    payload: public_bytes.clone(),
                }];
            }
        };

        // Check capability
        if token.capability != harmony_identity::CapabilityType::Discovery {
            return vec![RuntimeAction::SendReply {
                query_id,
                payload: public_bytes.clone(),
            }];
        }

        // Check resource matches our identity hash
        if token.resource.len() != 16 || token.resource[..] != local_hash[..] {
            return vec![RuntimeAction::SendReply {
                query_id,
                payload: public_bytes.clone(),
            }];
        }

        // Check expiry
        if token.expires_at != 0 && self.last_unix_now > token.expires_at {
            return vec![RuntimeAction::SendReply {
                query_id,
                payload: public_bytes.clone(),
            }];
        }

        // Check not-before
        if token.not_before > self.last_unix_now {
            return vec![RuntimeAction::SendReply {
                query_id,
                payload: public_bytes.clone(),
            }];
        }

        // Verify ML-DSA signature with LOCAL key (we issued this token)
        let pubkey = match harmony_crypto::ml_dsa::MlDsaPublicKey::from_bytes(
            &self.local_dsa_pubkey,
        ) {
            Ok(pk) => pk,
            Err(_) => {
                return vec![RuntimeAction::SendReply {
                    query_id,
                    payload: public_bytes.clone(),
                }];
            }
        };
        if token.verify_signature(&pubkey).is_err() {
            return vec![RuntimeAction::SendReply {
                query_id,
                payload: public_bytes.clone(),
            }];
        }

        // Check issuer matches our identity (token was issued by us)
        if token.issuer != local_hash {
            return vec![RuntimeAction::SendReply {
                query_id,
                payload: public_bytes.clone(),
            }];
        }

        // All checks passed — serve full record
        vec![RuntimeAction::SendReply {
            query_id,
            payload: full_bytes.clone(),
        }]
    }
```

- [ ] **Step 4: Run tests**

Run: `RUST_MIN_STACK=8388608 cargo test -p harmony-node discover_query`
Expected: all 6 tests pass.

Note: The spec lists two additional tests (`public_announce_broadcast_excludes_tunnel_hints` and `runtime_builds_public_and_full_records`) that depend on outbound announce publishing, which is not yet wired (existing TODO). These tests are deferred to the announce publishing bead.

- [ ] **Step 5: Commit**

```bash
git commit -m "feat(runtime): authenticated discover query handler

handle_discover_query() validates Discovery UCAN tokens against the
local node's ML-DSA key. Valid tokens get full routing hints (including
tunnel); invalid/missing tokens get Reticulum-only public hints.
6 tests."
```

---

### Task 5: Cleanup and verification

- [ ] **Step 1: Run clippy**

Run: `cargo clippy -p harmony-identity -p harmony-zenoh -p harmony-node`
Fix any warnings.

- [ ] **Step 2: Run workspace tests**

Run: `RUST_MIN_STACK=8388608 cargo test --workspace`

- [ ] **Step 3: Run fmt check**

Run: `cargo fmt --all -- --check`

- [ ] **Step 4: Commit (if clippy/fmt fixes needed)**

```bash
git commit -m "chore: clippy/fmt fixes for selective discovery"
```

---

## Summary

| Task | Description | Key Output |
|------|-------------|------------|
| 1 | Discovery capability type | `CapabilityType::Discovery = 7` |
| 2 | Discover namespace module | `harmony/discover/{identity_hash}` key expressions |
| 3 | Runtime scaffolding | Queryable declaration, fields, `TimerTick.unix_now`, stub handler |
| 4 | Discover query handler | `handle_discover_query()` with token validation |
| 5 | Cleanup | Clippy clean, workspace tests pass |

**End-to-end flow after this bead:**
1. Alice issues a Discovery UCAN token to Bob (out-of-band)
2. Bob queries `harmony/discover/{alice_hash}` with the token as payload
3. Alice's node validates: capability, resource, time bounds, ML-DSA signature
4. Valid → Alice responds with full announce (including tunnel NodeId, relay URL)
5. Invalid/missing → Alice responds with public announce (Reticulum-only)
6. Bob uses the tunnel hints to initiate an iroh-net connection to Alice
