# Memo Fetch Orchestration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
> **Note:** The `- [ ]` checkboxes below are execution tracking markers for the agentic worker, not persistent TODO items. Task tracking uses `bd` (beads) — see bead `harmony-5vf`.

**Goal:** Add memo queryable handler (serving) and memo fetch orchestration (requesting) to NodeRuntime, enabling peers to query and retrieve signed memo attestations over Zenoh.

**Architecture:** Two complementary pieces in `runtime.rs`: (1) a queryable on `harmony/memo/**` that serves local memos in a length-prefixed binary format, and (2) a fetch pipeline triggered by `MemoFetchRequest` that checks the local store, deduplicates in-flight queries, issues a Zenoh query, and inserts verified responses into the MemoStore. A `pubkey_cache`-backed `CredentialKeyResolver` verifies memo signatures on ingest.

**Tech Stack:** Rust, `harmony-memo` (Memo, serialize/deserialize, verify_memo), `harmony-credential` (CredentialKeyResolver), `harmony-zenoh` (namespace::memo), `harmony-node` (runtime.rs)

**Spec:** `docs/superpowers/specs/2026-03-22-memo-fetch-orchestration-design.md`

---

## File Structure

```
crates/harmony-node/src/
└── runtime.rs     — All changes: new event/action variants, memo queryable setup,
                     handle_memo_query(), handle_memo_fetch_request(),
                     handle_memo_fetch_response(), PubkeyCacheKeyResolver,
                     pending_memo_fetches field, tick-based cleanup
```

---

### Task 1: Add new RuntimeEvent and RuntimeAction variants + constants

**Files:**
- Modify: `crates/harmony-node/src/runtime.rs`

- [ ] **Step 1: Add MemoFetchRequest and MemoFetchResponse to RuntimeEvent**

After the `PeerPublicKeyLearned` variant (around line 203), add:

```rust
    /// Memo fetch: caller requests memos for an input CID.
    MemoFetchRequest { input: ContentId },

    /// Memo fetch: Zenoh query reply arrived with memo data.
    /// `unix_now` is Unix epoch seconds — injected by the event loop for sans-I/O testability.
    MemoFetchResponse {
        key_expr: String,
        payload: Vec<u8>,
        unix_now: u64,
    },
```

You'll need to add `use harmony_content::ContentId;` to the imports if not already present. Check by searching for it — it's likely already imported via `StorageTier` usage.

- [ ] **Step 2: Add QueryMemo to RuntimeAction**

After the `ReplicaPullResponse` variant (around line 253), add:

```rust
    /// Memo fetch: emit a Zenoh session.get() query for memos.
    /// The event loop calls session.get(key_expr) and feeds each reply
    /// back as RuntimeEvent::MemoFetchResponse.
    QueryMemo { key_expr: String },
```

- [ ] **Step 3: Add constants**

At the top of the file (near other constants), add:

```rust
/// Ticks before an in-flight memo fetch expires (20 × 250ms = 5s).
const MEMO_FETCH_TIMEOUT_TICKS: u64 = 20;

/// Maximum number of memos in a single query response.
const MAX_MEMO_RESPONSE_COUNT: usize = 256;

/// Maximum total response payload size (1 MiB).
const MAX_MEMO_RESPONSE_BYTES: usize = 1_048_576;
```

- [ ] **Step 4: Add fields to NodeRuntime struct**

In the `NodeRuntime` struct (around line 548, after `pubkey_cache`), add:

```rust
    // Queryable ID for the memo namespace (harmony/memo/**)
    memo_queryable_id: QueryableId,
    // In-flight memo fetches: input CID → tick when fetch was started.
    // Prevents re-querying for the same input while a fetch is in-flight.
    pending_memo_fetches: HashMap<ContentId, u64>,
```

- [ ] **Step 5: Update constructor to initialize new fields and declare memo queryable**

In `NodeRuntime::new()`, after the page queryable declaration (around line 630), add:

```rust
        // Memo namespace: register queryable for memo lookups.
        let (memo_qid, _) = queryable_router
            .declare(harmony_zenoh::namespace::memo::SUB)
            .expect("static key expression must be valid");
        actions.push(RuntimeAction::DeclareQueryable {
            key_expr: harmony_zenoh::namespace::memo::SUB.to_string(),
        });
```

In the `Self { ... }` struct initializer (around line 684), add:

```rust
            memo_queryable_id: memo_qid,
            pending_memo_fetches: HashMap::new(),
```

- [ ] **Step 6: Add stub handlers for new events in push_event**

In `push_event()`, add match arms for the new events (after the `PeerPublicKeyLearned` arm):

```rust
            RuntimeEvent::MemoFetchRequest { input } => {
                self.handle_memo_fetch_request(input);
            }
            RuntimeEvent::MemoFetchResponse {
                key_expr,
                payload,
                unix_now,
            } => {
                self.handle_memo_fetch_response(&key_expr, &payload, unix_now);
            }
```

Add placeholder methods so the code compiles:

```rust
    fn handle_memo_fetch_request(&mut self, _input: ContentId) {
        // TODO: implement in Task 3
    }

    fn handle_memo_fetch_response(&mut self, _key_expr: &str, _payload: &[u8], _unix_now: u64) {
        // TODO: implement in Task 3
    }
```

- [ ] **Step 7: Update constructor test assertion**

The `constructor_returns_startup_actions` test (around line 2262) asserts exact counts. Update:

```rust
        // 16 shard queryables + 1 stats queryable + 1 compute activity queryable + 1 page queryable + 1 memo queryable = 20
        assert_eq!(queryable_count, 20);
```

Also update the variant completeness test (search for `_a5 = RuntimeAction::DeclareQueryable`) to include the new variants:

```rust
        let _a_qm = RuntimeAction::QueryMemo {
            key_expr: "harmony/memo/aa/**".into(),
        };
```

And for RuntimeEvent variants:

```rust
        let _e_mfr = RuntimeEvent::MemoFetchRequest {
            input: ContentId::from_bytes([0u8; 32]),
        };
        let _e_mfp = RuntimeEvent::MemoFetchResponse {
            key_expr: "harmony/memo/aa/**".into(),
            payload: vec![],
            unix_now: 1000,
        };
```

- [ ] **Step 8: Run tests**

Run: `RUST_MIN_STACK=8388608 cargo test -p harmony-node`

- [ ] **Step 9: Commit**

```bash
git commit -m "feat(runtime): add memo fetch event/action variants and queryable setup

New RuntimeEvent: MemoFetchRequest, MemoFetchResponse.
New RuntimeAction: QueryMemo.
Declare queryable on harmony/memo/** at startup.
Constants: MEMO_FETCH_TIMEOUT_TICKS, MAX_MEMO_RESPONSE_COUNT, MAX_MEMO_RESPONSE_BYTES."
```

---

### Task 2: Implement memo queryable handler (serving)

**Files:**
- Modify: `crates/harmony-node/src/runtime.rs`

- [ ] **Step 1: Write tests for handle_memo_query**

Add to the `#[cfg(test)] mod tests` section:

```rust
    #[test]
    fn memo_queryable_serves_local_memos() {
        use harmony_identity::PqPrivateIdentity;
        use rand::rngs::OsRng;

        let (mut rt, _) = make_runtime();
        let identity = PqPrivateIdentity::generate(&mut OsRng);
        let input = ContentId::from_bytes([0x11; 32]);
        let output = ContentId::from_bytes([0x22; 32]);

        let memo = harmony_memo::create::create_memo(
            input, output, &identity, &mut OsRng, 1000, 2000,
        )
        .unwrap();
        rt.memo_store_mut().insert(memo);

        // Simulate a query for this input
        let input_hex = hex::encode(input.to_bytes());
        let key_expr = format!("harmony/memo/{input_hex}/**");
        rt.push_event(RuntimeEvent::QueryReceived {
            query_id: 42,
            key_expr,
            payload: vec![],
        });

        let actions = rt.tick();
        let reply = actions
            .iter()
            .find(|a| matches!(a, RuntimeAction::SendReply { query_id: 42, .. }));
        assert!(reply.is_some(), "should reply to memo query");

        // Parse response format: [u16 LE count][u32 LE len][bytes]...
        if let Some(RuntimeAction::SendReply { payload, .. }) = reply {
            assert!(payload.len() >= 2, "response too short");
            let count = u16::from_le_bytes([payload[0], payload[1]]) as usize;
            assert_eq!(count, 1, "should have 1 memo");
            // Verify the memo can be deserialized
            let memo_len = u32::from_le_bytes([payload[2], payload[3], payload[4], payload[5]]) as usize;
            let memo_bytes = &payload[6..6 + memo_len];
            let restored = harmony_memo::deserialize(memo_bytes).expect("should deserialize");
            assert_eq!(restored.input, input);
            assert_eq!(restored.output, output);
        }
    }

    #[test]
    fn memo_queryable_empty_no_reply() {
        let (mut rt, _) = make_runtime();

        let input_hex = hex::encode([0xFF; 32]);
        let key_expr = format!("harmony/memo/{input_hex}/**");
        rt.push_event(RuntimeEvent::QueryReceived {
            query_id: 99,
            key_expr,
            payload: vec![],
        });

        let actions = rt.tick();
        let reply = actions
            .iter()
            .find(|a| matches!(a, RuntimeAction::SendReply { query_id: 99, .. }));
        assert!(reply.is_none(), "should not reply when no memos exist");
    }

    #[test]
    fn memo_queryable_caps_response_count() {
        use harmony_identity::PqPrivateIdentity;
        use rand::rngs::OsRng;

        let (mut rt, _) = make_runtime();
        let input = ContentId::from_bytes([0x11; 32]);

        // Insert MAX_MEMO_RESPONSE_COUNT + 10 memos with different signers
        for _ in 0..(MAX_MEMO_RESPONSE_COUNT + 10) {
            let identity = PqPrivateIdentity::generate(&mut OsRng);
            let output = ContentId::from_bytes([0x22; 32]);
            let memo = harmony_memo::create::create_memo(
                input, output, &identity, &mut OsRng, 1000, 2000,
            )
            .unwrap();
            rt.memo_store_mut().insert(memo);
        }

        let input_hex = hex::encode(input.to_bytes());
        let key_expr = format!("harmony/memo/{input_hex}/**");
        rt.push_event(RuntimeEvent::QueryReceived {
            query_id: 50,
            key_expr,
            payload: vec![],
        });

        let actions = rt.tick();
        let reply = actions
            .iter()
            .find(|a| matches!(a, RuntimeAction::SendReply { query_id: 50, .. }));
        assert!(reply.is_some(), "should reply");

        if let Some(RuntimeAction::SendReply { payload, .. }) = reply {
            let count = u16::from_le_bytes([payload[0], payload[1]]) as usize;
            assert_eq!(count, MAX_MEMO_RESPONSE_COUNT, "should cap at MAX_MEMO_RESPONSE_COUNT");
        }
    }
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `RUST_MIN_STACK=8388608 cargo test -p harmony-node memo_queryable`
Expected: FAIL — `handle_memo_query` not yet implemented (queries go to route_query but no dispatch for memo queryable ID).

- [ ] **Step 3: Implement handle_memo_query**

Add the dispatch in `route_query()`. Find the `else if queryable_id == self.page_queryable_id` arm and add after it:

```rust
                } else if queryable_id == self.memo_queryable_id {
                    let memo_actions = self.handle_memo_query(query_id, &key_expr);
                    self.pending_direct_actions.extend(memo_actions);
                }
```

Add the implementation method:

```rust
    /// Handle a memo query: look up MemoStore by input CID from key expression.
    ///
    /// Key expression: `harmony/memo/{input_hex}/**`
    /// Response: `[u16 LE count][u32 LE len][memo_bytes]...`
    /// Returns empty vec (no reply) if no memos found or key malformed.
    fn handle_memo_query(&self, query_id: u64, key_expr: &str) -> Vec<RuntimeAction> {
        // Strip prefix and trailing wildcard to extract input_hex.
        let input_hex = match key_expr
            .strip_prefix(harmony_zenoh::namespace::memo::PREFIX)
            .and_then(|s| s.strip_prefix('/'))
        {
            Some(rest) => {
                // rest is e.g. "aabb.../**" — strip trailing /**
                rest.split('/').next().unwrap_or("")
            }
            None => return Vec::new(),
        };

        // Decode hex to ContentId
        let cid_bytes = match hex::decode(input_hex) {
            Ok(b) if b.len() == 32 => b,
            _ => return Vec::new(),
        };
        let mut arr = [0u8; 32];
        arr.copy_from_slice(&cid_bytes);
        let input_cid = ContentId::from_bytes(arr);

        let memos = self.memo_store.get_by_input(&input_cid);
        if memos.is_empty() {
            return Vec::new();
        }

        // Serialize up to MAX_MEMO_RESPONSE_COUNT memos into a temp buffer,
        // then write the actual count. This avoids a count mismatch if any
        // serialize() call fails.
        let mut memo_buf = Vec::new();
        let mut actual_count: u16 = 0;

        for memo in memos.iter().take(MAX_MEMO_RESPONSE_COUNT) {
            if let Ok(bytes) = harmony_memo::serialize(memo) {
                memo_buf.extend_from_slice(&(bytes.len() as u32).to_le_bytes());
                memo_buf.extend_from_slice(&bytes);
                actual_count += 1;
            }
        }

        if actual_count == 0 {
            return Vec::new();
        }

        let mut payload = Vec::with_capacity(2 + memo_buf.len());
        payload.extend_from_slice(&actual_count.to_le_bytes());
        payload.extend_from_slice(&memo_buf);

        vec![RuntimeAction::SendReply { query_id, payload }]
    }
```

- [ ] **Step 4: Run tests**

Run: `RUST_MIN_STACK=8388608 cargo test -p harmony-node memo_queryable`
Expected: all 3 tests pass.

- [ ] **Step 5: Commit**

```bash
git commit -m "feat(runtime): memo queryable handler serves local memos

handle_memo_query() parses input CID from key expression, looks up
MemoStore, serializes up to MAX_MEMO_RESPONSE_COUNT memos into a
length-prefixed response. 3 tests."
```

---

### Task 3: Implement memo fetch orchestration (requesting)

**Files:**
- Modify: `crates/harmony-node/src/runtime.rs`

- [ ] **Step 1: Write tests for memo fetch request**

```rust
    #[test]
    fn memo_fetch_request_local_short_circuit() {
        use harmony_identity::PqPrivateIdentity;
        use rand::rngs::OsRng;

        let (mut rt, _) = make_runtime();
        let identity = PqPrivateIdentity::generate(&mut OsRng);
        let input = ContentId::from_bytes([0x11; 32]);
        let output = ContentId::from_bytes([0x22; 32]);

        let memo = harmony_memo::create::create_memo(
            input, output, &identity, &mut OsRng, 1000, 2000,
        )
        .unwrap();
        rt.memo_store_mut().insert(memo);

        // Request memos for an input we already have locally
        rt.push_event(RuntimeEvent::MemoFetchRequest { input });
        let actions = rt.tick();

        // Should NOT emit QueryMemo — local data is sufficient
        assert!(
            !actions.iter().any(|a| matches!(a, RuntimeAction::QueryMemo { .. })),
            "should not query when memos exist locally"
        );
    }

    #[test]
    fn memo_fetch_request_emits_query() {
        let (mut rt, _) = make_runtime();
        let input = ContentId::from_bytes([0x11; 32]);

        rt.push_event(RuntimeEvent::MemoFetchRequest { input });
        let actions = rt.tick();

        let query = actions
            .iter()
            .find(|a| matches!(a, RuntimeAction::QueryMemo { .. }));
        assert!(query.is_some(), "should emit QueryMemo for unknown input");

        if let Some(RuntimeAction::QueryMemo { key_expr }) = query {
            let expected = format!("harmony/memo/{}/**", hex::encode(input.to_bytes()));
            assert_eq!(key_expr, &expected);
        }
    }

    #[test]
    fn memo_fetch_request_dedup() {
        let (mut rt, _) = make_runtime();
        let input = ContentId::from_bytes([0x11; 32]);

        // First request → should emit query
        rt.push_event(RuntimeEvent::MemoFetchRequest { input });
        let actions1 = rt.tick();
        assert!(
            actions1.iter().any(|a| matches!(a, RuntimeAction::QueryMemo { .. })),
            "first request should emit query"
        );

        // Second request for same input → should NOT emit query (in-flight)
        rt.push_event(RuntimeEvent::MemoFetchRequest { input });
        let actions2 = rt.tick();
        assert!(
            !actions2.iter().any(|a| matches!(a, RuntimeAction::QueryMemo { .. })),
            "duplicate request should be suppressed"
        );
    }

    #[test]
    fn memo_fetch_timeout_clears_pending() {
        let (mut rt, _) = make_runtime();
        let input = ContentId::from_bytes([0x11; 32]);

        // Issue a fetch
        rt.push_event(RuntimeEvent::MemoFetchRequest { input });
        rt.tick();

        // Advance past timeout
        for _ in 0..=MEMO_FETCH_TIMEOUT_TICKS {
            rt.push_event(RuntimeEvent::TimerTick { now: 0 });
            rt.tick();
        }

        // Should be able to re-fetch now
        rt.push_event(RuntimeEvent::MemoFetchRequest { input });
        let actions = rt.tick();
        assert!(
            actions.iter().any(|a| matches!(a, RuntimeAction::QueryMemo { .. })),
            "should re-emit query after timeout"
        );
    }
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `RUST_MIN_STACK=8388608 cargo test -p harmony-node memo_fetch_request`
Expected: FAIL — stub handlers don't emit actions.

- [ ] **Step 3: Implement handle_memo_fetch_request**

Replace the stub:

```rust
    /// Handle a memo fetch request: check local store, dedup, issue Zenoh query.
    fn handle_memo_fetch_request(&mut self, input: ContentId) {
        // 1. Local check — short-circuit if we already have memos
        if !self.memo_store.get_by_input(&input).is_empty() {
            return;
        }

        // 2. Dedup check — skip if already in-flight
        if self.pending_memo_fetches.contains_key(&input) {
            return;
        }

        // 3. Issue Zenoh query
        let input_hex = hex::encode(input.to_bytes());
        let key_expr = harmony_zenoh::namespace::memo::input_query(&input_hex);
        self.pending_direct_actions
            .push(RuntimeAction::QueryMemo { key_expr });

        // 4. Track in-flight
        self.pending_memo_fetches.insert(input, self.tick_count);
    }
```

- [ ] **Step 4: Add timeout cleanup in tick()**

In the `tick()` method, find the section where existing tick-based cleanup happens (after the filter broadcast logic). Add:

```rust
        // Expire in-flight memo fetches that have timed out.
        self.pending_memo_fetches
            .retain(|_, started| self.tick_count.saturating_sub(*started) <= MEMO_FETCH_TIMEOUT_TICKS);
```

- [ ] **Step 5: Run tests**

Run: `RUST_MIN_STACK=8388608 cargo test -p harmony-node memo_fetch_request`
Expected: all 4 tests pass.

- [ ] **Step 6: Commit**

```bash
git commit -m "feat(runtime): memo fetch request with local short-circuit and dedup

handle_memo_fetch_request() checks local MemoStore first, deduplicates
in-flight requests via pending_memo_fetches HashMap, emits QueryMemo
action. Tick-based timeout cleanup after MEMO_FETCH_TIMEOUT_TICKS."
```

---

### Task 4: Implement memo fetch response handler

**Files:**
- Modify: `crates/harmony-node/src/runtime.rs`
- Modify: `crates/harmony-node/Cargo.toml`

- [ ] **Step 1: Add harmony-credential dependency**

`harmony-credential` is needed for the `CredentialKeyResolver` trait. Add it to `crates/harmony-node/Cargo.toml` under `[dependencies]`:

```toml
harmony-credential.workspace = true
```

- [ ] **Step 2: Implement PubkeyCacheKeyResolver**

The `verify_memo()` function requires a `CredentialKeyResolver`. The runtime has `pubkey_cache: HashMap<[u8; 16], Vec<u8>>`. Add a thin wrapper struct:

```rust
/// Adapts the runtime's pubkey_cache to the CredentialKeyResolver trait
/// for memo signature verification.
struct PubkeyCacheKeyResolver<'a> {
    cache: &'a HashMap<[u8; 16], Vec<u8>>,
}

impl<'a> harmony_credential::CredentialKeyResolver for PubkeyCacheKeyResolver<'a> {
    fn resolve(&self, issuer: &harmony_identity::IdentityRef, _issued_at: u64) -> Option<Vec<u8>> {
        self.cache.get(&issuer.hash).cloned()
    }
}
```

Place this near the `handle_pull_with_token` method (which uses the same pubkey_cache pattern) or at module level before the `impl` block.

- [ ] **Step 3: Write tests for memo fetch response**

```rust
    #[test]
    fn memo_fetch_response_inserts_verified_memos() {
        use harmony_identity::PqPrivateIdentity;
        use rand::rngs::OsRng;

        let (mut rt, _) = make_runtime();
        let identity = PqPrivateIdentity::generate(&mut OsRng);
        let pub_id = identity.public_identity();
        let input = ContentId::from_bytes([0x11; 32]);
        let output = ContentId::from_bytes([0x22; 32]);

        // Cache the signer's public key via the production code path
        let id_ref = harmony_identity::IdentityRef::from(pub_id);
        rt.push_event(RuntimeEvent::PeerPublicKeyLearned {
            identity_hash: id_ref.hash,
            dsa_pubkey: pub_id.verifying_key.as_bytes().to_vec(),
        });

        // Build a valid response payload
        let memo = harmony_memo::create::create_memo(
            input, output, &identity, &mut OsRng, 1000, 2000,
        )
        .unwrap();
        let memo_bytes = harmony_memo::serialize(&memo).unwrap();

        let mut payload = Vec::new();
        payload.extend_from_slice(&1u16.to_le_bytes()); // count = 1
        payload.extend_from_slice(&(memo_bytes.len() as u32).to_le_bytes());
        payload.extend_from_slice(&memo_bytes);

        let input_hex = hex::encode(input.to_bytes());
        let key_expr = format!("harmony/memo/{input_hex}/**");
        rt.push_event(RuntimeEvent::MemoFetchResponse {
            key_expr,
            payload,
            unix_now: 1500,
        });
        rt.tick();

        // Memo should now be in the store
        let stored = rt.memo_store().get_by_input(&input);
        assert_eq!(stored.len(), 1);
        assert_eq!(stored[0].output, output);
    }

    #[test]
    fn memo_fetch_response_rejects_invalid_memos() {
        let (mut rt, _) = make_runtime();
        let input = ContentId::from_bytes([0x11; 32]);

        // Build a payload with garbage memo bytes
        let garbage = vec![0xFF; 100];
        let mut payload = Vec::new();
        payload.extend_from_slice(&1u16.to_le_bytes()); // count = 1
        payload.extend_from_slice(&(garbage.len() as u32).to_le_bytes());
        payload.extend_from_slice(&garbage);

        let input_hex = hex::encode(input.to_bytes());
        let key_expr = format!("harmony/memo/{input_hex}/**");
        rt.push_event(RuntimeEvent::MemoFetchResponse {
            key_expr,
            payload,
            unix_now: 1500,
        });
        rt.tick();

        // No memo should be stored
        assert!(rt.memo_store().get_by_input(&input).is_empty());
    }

    #[test]
    fn memo_fetch_response_oversized_rejected() {
        let (mut rt, _) = make_runtime();
        let input = ContentId::from_bytes([0x11; 32]);

        // Build a payload that exceeds MAX_MEMO_RESPONSE_BYTES
        let payload = vec![0u8; MAX_MEMO_RESPONSE_BYTES + 1];

        let input_hex = hex::encode(input.to_bytes());
        let key_expr = format!("harmony/memo/{input_hex}/**");
        rt.push_event(RuntimeEvent::MemoFetchResponse {
            key_expr,
            payload,
            unix_now: 1500,
        });
        rt.tick();

        assert!(rt.memo_store().get_by_input(&input).is_empty());
    }

    #[test]
    fn memo_fetch_multiple_responses_all_inserted() {
        use harmony_identity::PqPrivateIdentity;
        use rand::rngs::OsRng;

        let (mut rt, _) = make_runtime();
        let input = ContentId::from_bytes([0x11; 32]);

        // Two different signers
        let alice = PqPrivateIdentity::generate(&mut OsRng);
        let bob = PqPrivateIdentity::generate(&mut OsRng);

        let alice_pub = alice.public_identity();
        let bob_pub = bob.public_identity();

        // Cache both public keys via the production code path
        let alice_ref = harmony_identity::IdentityRef::from(alice_pub);
        rt.push_event(RuntimeEvent::PeerPublicKeyLearned {
            identity_hash: alice_ref.hash,
            dsa_pubkey: alice_pub.verifying_key.as_bytes().to_vec(),
        });
        let bob_ref = harmony_identity::IdentityRef::from(bob_pub);
        rt.push_event(RuntimeEvent::PeerPublicKeyLearned {
            identity_hash: bob_ref.hash,
            dsa_pubkey: bob_pub.verifying_key.as_bytes().to_vec(),
        });

        let output = ContentId::from_bytes([0x22; 32]);
        let input_hex = hex::encode(input.to_bytes());
        let key_expr = format!("harmony/memo/{input_hex}/**");

        // First response: Alice's memo
        let memo_alice = harmony_memo::create::create_memo(
            input, output, &alice, &mut OsRng, 1000, 2000,
        ).unwrap();
        let alice_bytes = harmony_memo::serialize(&memo_alice).unwrap();
        let mut payload1 = Vec::new();
        payload1.extend_from_slice(&1u16.to_le_bytes());
        payload1.extend_from_slice(&(alice_bytes.len() as u32).to_le_bytes());
        payload1.extend_from_slice(&alice_bytes);

        rt.push_event(RuntimeEvent::MemoFetchResponse {
            key_expr: key_expr.clone(),
            payload: payload1,
            unix_now: 1500,
        });
        rt.tick();

        // Second response: Bob's memo
        let memo_bob = harmony_memo::create::create_memo(
            input, output, &bob, &mut OsRng, 1000, 2000,
        ).unwrap();
        let bob_bytes = harmony_memo::serialize(&memo_bob).unwrap();
        let mut payload2 = Vec::new();
        payload2.extend_from_slice(&1u16.to_le_bytes());
        payload2.extend_from_slice(&(bob_bytes.len() as u32).to_le_bytes());
        payload2.extend_from_slice(&bob_bytes);

        rt.push_event(RuntimeEvent::MemoFetchResponse {
            key_expr,
            payload: payload2,
            unix_now: 1500,
        });
        rt.tick();

        // Both memos should be in the store
        let stored = rt.memo_store().get_by_input(&input);
        assert_eq!(stored.len(), 2, "both Alice's and Bob's memos should be stored");
    }
```

- [ ] **Step 4: Run tests to verify they fail**

Run: `RUST_MIN_STACK=8388608 cargo test -p harmony-node memo_fetch_response`
Expected: FAIL — stub handler doesn't parse or insert.

- [ ] **Step 5: Implement handle_memo_fetch_response**

Replace the stub:

```rust
    /// Handle a memo fetch response: parse, verify, insert into MemoStore.
    ///
    /// Each Zenoh reply becomes a separate MemoFetchResponse event.
    /// Multiple peers may respond — all are processed independently.
    fn handle_memo_fetch_response(&mut self, key_expr: &str, payload: &[u8], unix_now: u64) {
        // 1. Parse input_cid from key_expr
        let input_hex = match key_expr
            .strip_prefix(harmony_zenoh::namespace::memo::PREFIX)
            .and_then(|s| s.strip_prefix('/'))
        {
            Some(rest) => rest.split('/').next().unwrap_or(""),
            None => return,
        };

        let cid_bytes = match hex::decode(input_hex) {
            Ok(b) if b.len() == 32 => b,
            _ => return,
        };
        let mut arr = [0u8; 32];
        arr.copy_from_slice(&cid_bytes);
        let _input_cid = ContentId::from_bytes(arr);

        // 2. Size guard
        if payload.len() > MAX_MEMO_RESPONSE_BYTES {
            return;
        }

        // 3. Decode response: [u16 LE count][u32 LE len][bytes]...
        if payload.len() < 2 {
            return;
        }
        let count = u16::from_le_bytes([payload[0], payload[1]]) as usize;
        let mut offset = 2;

        let resolver = PubkeyCacheKeyResolver {
            cache: &self.pubkey_cache,
        };

        for _ in 0..count {
            if offset + 4 > payload.len() {
                break;
            }
            let memo_len = u32::from_le_bytes([
                payload[offset],
                payload[offset + 1],
                payload[offset + 2],
                payload[offset + 3],
            ]) as usize;
            offset += 4;

            if offset + memo_len > payload.len() {
                break;
            }
            let memo_bytes = &payload[offset..offset + memo_len];
            offset += memo_len;

            // Deserialize
            let memo = match harmony_memo::deserialize(memo_bytes) {
                Ok(m) => m,
                Err(_) => continue,
            };

            // Verify
            if harmony_memo::verify::verify_memo(&memo, unix_now, &resolver).is_err() {
                continue;
            }

            // Insert
            self.memo_store.insert(memo);
        }
    }
```

- [ ] **Step 6: Run tests**

Run: `RUST_MIN_STACK=8388608 cargo test -p harmony-node memo_fetch_response`
Expected: all 4 tests pass.

- [ ] **Step 7: Run all memo tests together**

Run: `RUST_MIN_STACK=8388608 cargo test -p harmony-node memo`
Expected: all memo-related tests pass.

- [ ] **Step 8: Commit**

```bash
git commit -m "feat(runtime): memo fetch response handler with verification

handle_memo_fetch_response() parses length-prefixed memos, verifies
each via verify_memo() using PubkeyCacheKeyResolver, inserts valid
memos into MemoStore. Rejects oversized payloads, malformed entries,
and memos with unknown issuers. 4 tests."
```

---

### Task 5: Cleanup and verification

- [ ] **Step 1: Run clippy**

Run: `cargo clippy -p harmony-node`
Fix any warnings.

- [ ] **Step 2: Run workspace tests**

Run: `RUST_MIN_STACK=8388608 cargo test --workspace`

- [ ] **Step 3: Run fmt check**

Run: `cargo fmt --all -- --check`

- [ ] **Step 4: Commit (if clippy/fmt fixes needed)**

```bash
git commit -m "chore: clippy/fmt fixes for memo fetch orchestration"
```

---

## Summary

| Task | Description | Key Output |
|------|-------------|------------|
| 1 | Event/action variants, constants, queryable setup | New `MemoFetchRequest`, `MemoFetchResponse`, `QueryMemo`, memo queryable declared |
| 2 | Memo queryable handler (serving) | `handle_memo_query()` serves local memos in length-prefixed format |
| 3 | Memo fetch request (dedup + query) | `handle_memo_fetch_request()` with local short-circuit and in-flight dedup |
| 4 | Memo fetch response (verify + insert) | `handle_memo_fetch_response()` with `PubkeyCacheKeyResolver` verification |
| 5 | Cleanup | Clippy clean, workspace tests pass, fmt check |

**End-to-end flow after this bead:**
1. Workflow engine needs memos for input CID → pushes `MemoFetchRequest`
2. Runtime checks local MemoStore (short-circuit if found)
3. Runtime checks `pending_memo_fetches` (skip if in-flight)
4. Runtime emits `QueryMemo` → event loop calls Zenoh `session.get()`
5. Peers with `harmony/memo/**` queryable receive query → `handle_memo_query()` replies
6. Replies arrive as `MemoFetchResponse` → parsed, verified, inserted into MemoStore
7. Caller's next `memo_store().get_by_input()` finds the fetched memos
