# ZEB-475 — Reticulum teardown (core) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove Reticulum from harmony-core — delete the ~14.6k-LOC `harmony-reticulum` crate, the `harmony-runtime` router that drives it, harmony-node's Reticulum usage, the tunnel/rawlink Reticulum-carrier branches, and the Reticulum-concept cruft across five more crates — while preserving every load-bearing piece (device-hash formula, shared-crypto interop tests, the tunnel/rawlink transports, mDNS PeerTable, `PeerAction::InitiateLink`).

**Architecture:** This is a deletion. Reticulum's only live consumer was DM unicast, already retired client-side by ZEB-474 (#271, merged). Now remove the core that drove it. The teardown proceeds **leaf → trunk**: independent dead modules first, then concept-cruft variants (each removed atomically with all consumers), then the runtime router + harmony-node router-consumers, then the carrier branches, then the crate itself, then a doc-comment + final-gate sweep. Each task leaves the workspace **compiling + tests-green**; transient `dead_code` warnings are tolerated mid-stream and swept to `clippy -D warnings`-clean in the final task.

**Tech Stack:** Rust workspace (harmony-core), `cargo nextest`, `cargo clippy --all-targets -- -D warnings`, `cargo fmt`. harmony has **no GitHub CI** — these local gates ARE the gate.

**Base:** branch `zeb-475-reticulum-teardown-core` off `origin/main` @ `7d05583` (ZEB-472 / #279 merged — `FrameTag::Dm = 0x04` present; this teardown removes `FrameTag::Reticulum = 0x01`, orthogonal).

**Spec:** `harmony-client/docs/specs/2026-06-15-reticulum-teardown-move-2-design.md` (§4.2, §4.2.1, §4.3 KEEP-list, §8 entanglements).

---

## Conventions for every task

- **Line numbers below are a guide** (captured by recon against the post-#279 tree). The implementer MUST grep the named symbol to confirm the exact site before editing — the #279 merge may have drifted a few lines in harmony-tunnel/harmony-node.
- **Remove tests with the code they cover** (never `#[ignore]` them). A Reticulum-only test dies with its subject; a generic test that merely *used* `FrameTag::Reticulum` as a sample is re-pointed to a surviving tag (e.g. `FrameTag::Zenoh`).
- **Green checkpoint per task:** `cargo build --workspace` + `cargo nextest run -p <touched crates>` pass. `clippy -D warnings` clean is enforced in Task 7 (transient `dead_code` between tasks is fine under plain build).
- **Commit after each task** with a `ZEB-475:` prefixed message.
- **Do NOT touch** (spec §4.3): the `DeviceIdentityHash`/`IdentityHash` formula in `harmony-identity/src/identity.rs` (only restyle the "Reticulum address" comment, never the derivation); `harmony-identity/tests/reticulum_interop.rs` (shared-crypto coverage — keep as-is, framing rename optional); `harmony-tunnel`/`harmony-rawlink` as transports (only their Reticulum branches go); the mDNS `PeerTable` (keyed by `[u8;16]`, no `harmony_reticulum` dep — woven through the node event loop, out-of-scope per "NOT a full node retirement"); `PeerAction::InitiateLink` (also emitted non-Reticulum at `manager.rs:92`).

---

## Task 1: Leaf dead-modules + doc-comment framing

Fully isolated removals — no logic consumers. Lands green immediately.

**Files:**
- Delete: `crates/harmony-content/src/reticulum_bridge.rs`
- Modify: `crates/harmony-content/src/lib.rs` (remove `pub mod reticulum_bridge;`, ~line 21)
- Modify: `crates/harmony-zenoh/src/namespace.rs` (delete the `pub mod reticulum { … }` block, ~lines 44–82)
- Modify: `crates/harmony-platform/src/network.rs` (rewrite stale `Interface`-trait doc comment, ~lines 12–13)

- [ ] **Step 1:** Confirm `harmony-content::reticulum_bridge` is unreferenced: `rg "reticulum_bridge" crates/ | grep -v "crates/harmony-content/src/reticulum_bridge.rs\|lib.rs:"` → expect zero hits. Then delete the file and remove the `pub mod reticulum_bridge;` line.
- [ ] **Step 2:** Confirm `harmony-zenoh::namespace::reticulum` is unreferenced: `rg "namespace::reticulum|reticulum::(ANNOUNCE|LINK|DIAGNOSTICS|announce_key|link_key|diagnostics_key)" crates/` → expect zero hits outside `namespace.rs`. Delete the `pub mod reticulum { … }` block (and any now-unused imports it pulled in).
- [ ] **Step 3:** Rewrite `harmony-platform/src/network.rs:12–13` to drop the reference to the deleted crate's `Interface` trait — e.g. "`NetworkInterface` is the platform byte-I/O bridge." (no mention of harmony-reticulum).
- [ ] **Step 4:** Doc-comment-only framing (KEEP every function/type; change comment text only — drop "Reticulum" framing, keep the crypto/format facts). Sites:
  - `crates/harmony-identity/src/identity.rs:3` (module doc "and Reticulum"), `:112` (encrypt doc)
  - `crates/harmony-identity/src/crypto_suite.rs:13` (`CryptoSuite::Ed25519` doc "Reticulum-compatible")
  - `crates/harmony-crypto/src/aead.rs:3`, `fernet.rs:1,3,38,72`, `hash.rs:6,18,31,34`, `hkdf.rs:8,11,19,40`
  - `crates/harmony-zenoh/src/unicast.rs:57`, `crates/harmony-mailbox/src/message.rs:4`, `crates/harmony-peers/src/state.rs:5–6`
  - **Do NOT alter** the `IdentityHash`/`address_hash` derivation in `identity.rs` or the hash/fernet function bodies — comments only.
- [ ] **Step 5:** Verify: `cargo build --workspace` + `cargo nextest run -p harmony-content -p harmony-zenoh -p harmony-platform`. Expected: PASS.
- [ ] **Step 6:** Commit: `git commit -am "ZEB-475: remove dead reticulum_bridge + zenoh reticulum namespace + doc framing"`

---

## Task 2: harmony-discovery `RoutingHint::Reticulum` (+ node caller + runtime test)

`RoutingHint` lives in `harmony-discovery`; removing its `Reticulum` variant breaks the harmony-node announce builder and one runtime test. All land in one task.

**Files:**
- Modify: `crates/harmony-discovery/src/record.rs` (remove `Reticulum` variant ~26–27; production deser use ~231; test builders ~305, 325)
- Modify: `crates/harmony-discovery/src/verify.rs` (test uses ~88, 174, 203)
- Modify: `crates/harmony-discovery/src/manager.rs` (test uses ~305, 735, 826)
- Modify: `crates/harmony-node/src/event_loop.rs` (`build_local_announce` ~262–307: strip `reticulum_addr` param + the `RoutingHint::Reticulum` hint block ~302–307; update caller ~597 to stop passing the addr as the routing hint — KEEP `mdns_addr` flowing to `PeerTable::new`/`start_mdns`)
- Modify: `crates/harmony-runtime/src/runtime.rs` (remove the `discovery_no_tunnel_hints_is_noop` test's `RoutingHint::Reticulum` construction ~7087; verify `process_discovered_tunnel_hints` ~3009/3029 stays exhaustive)

- [ ] **Step 1:** `rg "RoutingHint::Reticulum|RoutingHint :: Reticulum" crates/` to enumerate every site (the list above is the recon snapshot; confirm).
- [ ] **Step 2:** Remove the `Reticulum { destination_hash: [u8;16] }` variant from `RoutingHint` (`record.rs:26–27`). Fix the production site at `record.rs:231` (announce deserialization) — if it `match`es `RoutingHint`, remove the `Reticulum` arm; ensure remaining arms stay exhaustive.
- [ ] **Step 3:** In `harmony-node` `build_local_announce`: remove the `reticulum_addr: Option<[u8;16]>` parameter and the `if let Some(dest_hash) = reticulum_addr { builder.add_routing_hint(RoutingHint::Reticulum {…}) }` block. At the caller (`event_loop.rs:597`), drop the `mdns_addr` argument to `build_local_announce` only (the `mdns_addr` local still feeds `PeerTable::new` at ~370 and `start_mdns` at ~372 — unchanged).
- [ ] **Step 4:** Remove `RoutingHint::Reticulum` uses in `verify.rs`/`manager.rs` tests (re-point to `RoutingHint::Tunnel` where the test needs a non-matching hint, or delete the Reticulum-specific case). Remove the runtime test construction at `runtime.rs:7087` (delete the test if it solely asserts the Reticulum-hint no-op; otherwise re-point to a Tunnel hint).
- [ ] **Step 5:** Verify: `cargo build --workspace` + `cargo nextest run -p harmony-discovery -p harmony-node -p harmony-runtime`. Expected: PASS.
- [ ] **Step 6:** Commit: `git commit -am "ZEB-475: remove harmony-discovery RoutingHint::Reticulum + node announce hint"`

---

## Task 3: harmony-contacts `ContactAddress::Reticulum` (+ peers consumer + version bump)

Removing the `Reticulum` variant (discriminant 0) shifts the postcard discriminant of `Tunnel`. The store is **never persisted to disk** (per `store.rs` comment) so no real data breaks, but **bump `FORMAT_VERSION` v3→v4** as the hygienic, mechanism-consistent fail-closed. The one live consumer is the harmony-peers fallback dial.

**Files:**
- Modify: `crates/harmony-contacts/src/contact.rs` (remove `ContactAddress::Reticulum` variant ~8–10; postcard round-trip tests ~148–155 and the Reticulum case in ~176–209)
- Modify: `crates/harmony-contacts/src/store.rs` (bump `FORMAT_VERSION` 3→4 + update the "expected v3" error string to v4 ~109; remove `find_by_tunnel_node_id_skips_reticulum_addresses` test ~285–296)
- Modify: `crates/harmony-peers/src/manager.rs` (remove the `has_reticulum` fallback block ~244–259; the `with_reticulum` test-helper param ~418–432; the two tests `tunnel_dropped_falls_back_to_reticulum` ~1119–1147 and `tunnel_dropped_no_fallback_without_reticulum_address` ~1149–1177)
- Verify-only: `crates/harmony-runtime/src/runtime.rs:1573–1582` (`try_initiate_tunnel` already matches only `ContactAddress::Tunnel` via `if let` — confirm no exhaustive `match` breaks); `crates/harmony-node/src/main.rs:784` (constructs `Tunnel` only — no change)

- [ ] **Step 1:** `rg "ContactAddress::Reticulum|ContactAddress :: Reticulum" crates/` to enumerate sites (confirm the list above).
- [ ] **Step 2:** Remove the `Reticulum { destination_hash: [u8;16] }` variant from `ContactAddress` (`contact.rs:8`). Find `FORMAT_VERSION` (`rg "FORMAT_VERSION" crates/harmony-contacts/src/store.rs`) and bump its `const FORMAT_VERSION: u8 = 3;` to `4`; update the error string at `store.rs:109` ("expected v3" → "expected v4").
- [ ] **Step 3:** Remove the `has_reticulum` block in `harmony-peers/src/manager.rs` (the `let has_reticulum = …` + `if has_reticulum { actions.push(PeerAction::InitiateLink …); peer.status = Connecting; }`). **KEEP `PeerAction::InitiateLink`** — it is still emitted at `manager.rs:92`. Remove the `with_reticulum` helper arg + the two fallback tests.
- [ ] **Step 4:** Remove the Reticulum cases from the `contact.rs` postcard tests and the `store.rs` skip test. Confirm the `runtime.rs` and `main.rs` `ContactAddress` sites need no change (verify-only).
- [ ] **Step 5:** Verify: `cargo build --workspace` + `cargo nextest run -p harmony-contacts -p harmony-peers -p harmony-runtime -p harmony-node`. Expected: PASS (incl. the surviving `InitiateLink`-emitting peers tests).
- [ ] **Step 6:** Commit: `git commit -am "ZEB-475: remove ContactAddress::Reticulum + peers fallback; bump ContactStore v4"`

---

## Task 4: harmony-runtime router surface + harmony-node router-consumers

The trunk. Remove the Reticulum packet router from `harmony-runtime` and every harmony-node arm that bridged runtime actions/events to Reticulum transport. After this task, **neither `harmony-runtime` nor `harmony-node` references `harmony_reticulum::*`** (the crate dep is dropped in Task 6). This is the largest task — remove the full slice and resolve all compile errors; `dead_code` orphans in harmony-node's tunnel_bridge/tunnel_task (e.g. `try_send_reticulum`) are removed in Task 5, so a transient `dead_code` warning here is expected and fine under plain build.

**Files — `crates/harmony-runtime/src/runtime.rs`:**
- `use harmony_reticulum::node::{Node, NodeAction, NodeEvent};` import (~23) — remove
- `router: Node` field (~754); `router_queue: VecDeque<NodeEvent>` (~766); `router_starved: u32` (~792); `pending_unicast_sends: VecDeque<([u8;16], Vec<u8>)>` (~776–782); `local_public_announce`/`local_full_announce` fields (~860–866)
- `Node::new()` + `udp0` `register_interface` (~954–962); `register_announcing_destination` block (~967–989)
- `register_interface`/`unregister_interface` on `TunnelHandshakeComplete` (~1910–1914), `TunnelClosed` (~1929), `L2InterfaceReady` (~2106–2110), `L2InterfaceClosed` (~2114) — remove the `self.router.*` calls ONLY; **keep** the surrounding tunnel/L2 lifecycle + PeerManager wiring
- `push_event` `SendUnicastToDevice` arm (~2190–2206); the tick() unicast drain block (~2353–2422); `router_queue` push sites (~1755, 1764, 1921–1925); `router_queue` drain in tick (~2301–2318); `dispatch_router_actions` fn (~2592–2700); accessors `router_queue_len` (~1344–1346), `pending_unicast_sends_len` (~1360–1361); `router_starved` in the starvation tuple (~2281); the router tier in `order`/tick scheduling (~2279, 2301)
- public methods `register_local_destination`/`unregister_local_destination` (~1672–1688), `lookup_destination_identity` (~1709–1713), `set_local_public_announce`/`set_local_full_announce` (~1649–1657)
- **RuntimeEvent** Reticulum-only variants: `InboundPacket` (~197–201), `SendUnicastToDevice` (+doc ~205–266), `TunnelReticulumReceived` (~296–301), `L2InterfaceReady` (~362–363), `L2InterfaceClosed` (~364–365). **KEEP `TimerTick`** (~202–204 — dual-use; only its `router_queue.push_back(NodeEvent::TimerTick)` at ~1764 goes; the Discovery republish at ~2258–2266 stays).
- **RuntimeAction** Reticulum-only variants: `SendOnInterface` (~405–412), `UnicastReceived` (~436–445)
- `reticulum_identity_bytes` config field in `NodeConfig` (~79–84) + its `default()` (~176)
- runtime tests calling `harmony_reticulum::packet::*`/`path_table::*`/`PacketContext` (~5005–5054, 5080–5130, 5203–5230, 5387–5413, 5505–5560, 8655–8690) — remove
- **KEEP** the Discovery(zenoh) `DiscoveryAnnounceReceived` ingestion (~1947–2004) and `local_identity_hash` (~65–66)
- Modify `crates/harmony-runtime/src/lib.rs:22` re-export list (drop removed `RuntimeAction`/`RuntimeEvent` Reticulum variants from the public surface)

**Files — `crates/harmony-node/src`:**
- `event_loop.rs`: the `RuntimeAction::SendOnInterface` arm in full (~1908–1944 — both `l2:` and `tunnel-` branches); the `TunnelBridgeEvent::ReticulumReceived` select arm (~1122–1137); the `RuntimeEvent::UnicastReceived`/`InboundPacket`/`TunnelReticulumReceived`/`L2InterfaceReady`/`L2InterfaceClosed` push/handle sites; the L2 rawlink `#[cfg(all(target_os="linux", feature="rawlink"))]` block (~450–507) incl. `ret_inbound_rx`/`ret_outbound_tx`/`rawlink_iface_name`, the `BridgeConfig { reticulum_inbound_tx: Some(..) }`, the `L2InterfaceReady` push (~484–486), and the arm-9 inbound handler (~1366–1397)
- `main.rs`: the Ed25519→`reticulum_identity_bytes` derivation (~589–593) + the `reticulum_identity_bytes,` field in the `NodeConfig` construction (~710) + the CLI doc comment (~79–81). **Trace the `ed25519` local** (`main.rs:557`/`589`): after removing the Reticulum derivation, if `ed25519` is otherwise unused (the `pq` key covers all other uses), remove its load too — leave **no dangling `Zeroizing` secret** (spec §8.3). Keep `local_identity_hash`/`our_addr_bytes` if still consumed by PeerManager/mDNS.

- [ ] **Step 1:** `rg "harmony_reticulum|router\.|router_queue|SendUnicastToDevice|SendOnInterface|UnicastReceived|InboundPacket|TunnelReticulumReceived|reticulum_identity_bytes|register_announcing_destination" crates/harmony-runtime/src crates/harmony-node/src` to map every site against the recon list above.
- [ ] **Step 2 (runtime):** Remove the router field/queue/import, the `SendUnicastToDevice` path + drain, `register_announcing_destination`, the `register_interface` router calls on tunnel/L2 lifecycle (keep the lifecycle), `dispatch_router_actions`, the router accessors, the RuntimeAction/RuntimeEvent Reticulum variants, the `reticulum_identity_bytes` config field, and the harmony_reticulum-using tests. Update `lib.rs:22` re-exports.
- [ ] **Step 3 (node):** Remove the `SendOnInterface` arm, the `ReticulumReceived` bridge arm, the removed-RuntimeEvent push/handle sites, the L2 rawlink block, and the `main.rs` `reticulum_identity_bytes` derivation + field (+ dangling `ed25519` cleanup). Where a `match` on `RuntimeAction`/`RuntimeEvent` loses arms, ensure it stays exhaustive (a `_ => {}` wildcard already covers the client-style drop in many node arms — confirm).
- [ ] **Step 4:** Verify: `cargo build --workspace` (hard errors must be zero; `dead_code` warnings on harmony-node tunnel_bridge/tunnel_task Reticulum helpers are expected and resolved in Task 5). `cargo nextest run -p harmony-runtime -p harmony-node`. Expected: PASS.
- [ ] **Step 5:** Commit: `git commit -am "ZEB-475: remove harmony-runtime Reticulum router + harmony-node router consumers"`

---

## Task 5: harmony-tunnel + harmony-node tunnel Reticulum-carrier branch

Remove `FrameTag::Reticulum` and the `SendReticulum`/`ReticulumReceived` tunnel carrier, plus the now-orphaned harmony-node tunnel plumbing. KEEP `FrameTag::Dm` (0x04), `Zenoh` (0x02), `Replication` (0x03), `Keepalive` (0x00) and the tunnel session itself.

**Files — `crates/harmony-tunnel/src`:**
- `frame.rs`: remove `Reticulum = 0x01` (~12) + the `0x01 => Ok(Self::Reticulum)` arm in `from_byte` (~24). Re-point the two generic tests that used it as a sample tag — `frame_encode_decode_roundtrip` (~140–151) and `wrong_aad_fails_decryption` (~191–200) — to `FrameTag::Zenoh` (preserves coverage; the test is tag-agnostic).
- `event.rs`: remove `TunnelEvent::SendReticulum { packet, now_ms }` (~9) and `TunnelAction::ReticulumReceived { packet }` (~28).
- `session.rs`: remove the `handle_event` `SendReticulum` arm (~198–200) and the `handle_encrypted_frame` `FrameTag::Reticulum` arm (~272–274). Remove the Reticulum-path tests (~501–523, ~675–681, ~798+) — these specifically exercise the Reticulum send/receive; the `Dm`/`Zenoh` analogues remain as coverage.

**Files — `crates/harmony-node/src`:**
- `tunnel_bridge.rs`: remove `TunnelBridgeEvent::ReticulumReceived` (~23–28), `TunnelCommand::SendReticulum` (~61–62), and the `try_send_reticulum` method (~90–95).
- `tunnel_task.rs`: remove the `Some(TunnelCommand::SendReticulum { packet })` handler arm (~358–372) and the `TunnelAction::ReticulumReceived { packet } =>` forwarding arm (~513–522). (The `TunnelAction::DmReceived` debug arm added by #279 stays.)

- [ ] **Step 1:** `rg "FrameTag::Reticulum|SendReticulum|ReticulumReceived|try_send_reticulum" crates/harmony-tunnel/src crates/harmony-node/src` to confirm sites.
- [ ] **Step 2:** Remove the harmony-tunnel `FrameTag::Reticulum` + `from_byte` arm + the `SendReticulum`/`ReticulumReceived` event/action + session arms; re-point the two generic frame tests to `FrameTag::Zenoh`; delete the Reticulum-specific session tests.
- [ ] **Step 3:** Remove the harmony-node `tunnel_bridge.rs` Reticulum variants + `try_send_reticulum`, and the `tunnel_task.rs` arms. This clears the Task-4 transient `dead_code`.
- [ ] **Step 4:** Verify: `cargo build --workspace` + `cargo nextest run -p harmony-tunnel -p harmony-node`. Expected: PASS (incl. the surviving `Dm` round-trip + unknown-tag-rejection tests, and the #279 keepalive-timing regression test).
- [ ] **Step 5:** Commit: `git commit -am "ZEB-475: remove harmony-tunnel FrameTag::Reticulum carrier + node plumbing"`

---

## Task 6: harmony-rawlink carrier branch + drop deps + delete the crate

Remove the rawlink `0x00` Reticulum frame-type + bridge channels, then drop the `harmony-reticulum` deps and delete the crate. After Task 4 the node no longer wires rawlink's Reticulum channels, so they are now dead.

**Files — `crates/harmony-rawlink/src`:**
- `lib.rs`: remove `pub const RETICULUM: u8 = 0x00;` from `frame_type` (~27–28). KEEP `SCOUT=0x01`, `DATA=0x02`, `BATCH=0x03`.
- `bridge.rs`: remove `BridgeConfig.reticulum_inbound_tx` (~100–101, default ~111); `Bridge.reticulum_outbound_rx` (~123); the `Bridge::new` `reticulum_outbound_rx` param (~138); the outbound drain block (~234–248); the inbound `frame_type::RETICULUM` dispatch arm (~332–339); the tests `reticulum_frame_routed_to_channel`/`reticulum_outbound_encoding`/`interleaved_frame_types_routed_correctly`/`batch_reticulum_and_scout` (~729–849) — re-point interleave/batch tests to `SCOUT`/`DATA` where they need a second frame type.

**Crate deletion:**
- Modify: `crates/harmony-node/Cargo.toml` (remove `harmony-reticulum = …`, ~44)
- Modify: `crates/harmony-runtime/Cargo.toml` (remove dep ~45; dev-dep ~58–64; the `"harmony-reticulum/std",` feature entry ~20)
- Modify: the workspace root `Cargo.toml` (remove `crates/harmony-reticulum` from `[workspace] members` and any `[workspace.dependencies] harmony-reticulum = …` entry)
- Delete: `crates/harmony-reticulum/` (entire directory, incl. `tests/reticulum_interop.rs` — the crate-level one; the **harmony-identity** `tests/reticulum_interop.rs` is a different file and stays)

- [ ] **Step 1:** `rg "frame_type::RETICULUM|reticulum_inbound_tx|reticulum_outbound_rx" crates/` to confirm rawlink sites are dead (callers removed in Task 4).
- [ ] **Step 2:** Remove the rawlink `RETICULUM` const + bridge channels + dispatch/drain + tests (re-point interleave/batch tests to surviving frame types).
- [ ] **Step 3:** `rg "harmony_reticulum|harmony-reticulum" crates/ --glob '!crates/harmony-reticulum/**'` → expect **zero** hits (proves no remaining consumer). If any remain, fix them before deleting.
- [ ] **Step 4:** Drop the deps in `harmony-node/Cargo.toml` + `harmony-runtime/Cargo.toml` (prod + dev + feature list) + the workspace root members/deps. `git rm -r crates/harmony-reticulum`.
- [ ] **Step 5:** Verify: `cargo build --workspace` + `cargo nextest run -p harmony-rawlink -p harmony-node -p harmony-runtime`. Expected: PASS. (`Cargo.lock` updates — commit it.)
- [ ] **Step 6:** Commit: `git commit -am "ZEB-475: remove rawlink Reticulum frame-type + delete harmony-reticulum crate"`

---

## Task 7: Final sweep + full-workspace gate

Sweep every deletion orphan and prove the whole tree is clean.

- [ ] **Step 1:** `rg -i "reticulum" crates/ --glob '!crates/harmony-identity/tests/reticulum_interop.rs'` and review every remaining hit. Acceptable survivors: the kept `harmony-identity/tests/reticulum_interop.rs`; any shared-crypto doc comment intentionally left (should be none after Task 1 Step 4). Everything else is a missed site — remove it.
- [ ] **Step 2:** `cargo fmt --all`
- [ ] **Step 3:** `cargo clippy --all-targets --all-features -- -D warnings` — fix ALL `dead_code`/`unused_import`/`unused_variable` orphans from the teardown (this is where the Task-4/5 transients are cleaned). Note: a small set of **pre-existing** harmony-node-own-code lints (`useless_vec`, `needless_borrow`, never-read fields, const asserts in tunnel_task tests) + harmony-content `manual_checked_ops` exist on `main` independent of this work — if `-D warnings` trips on those, do NOT expand scope; record them and confirm they predate this branch (`git stash` + clippy on `origin/main` to compare), leaving them for the separate clippy-cleanup ticket. The teardown's own orphans MUST be clean.
- [ ] **Step 4:** `cargo nextest run --workspace --all-targets`. Expected: all pass (Reticulum tests gone, not skipped). Known orphan-flakes (iroh/zenoh transport first-bind, channel_backfill catch-up under parallel load — ZEB-476-class) are non-blocking; re-run to confirm green.
- [ ] **Step 5:** Confirm the KEEP-list intact: `harmony-identity/tests/reticulum_interop.rs` present + passing; `IdentityHash` derivation in `identity.rs` byte-identical to `origin/main` (`git diff origin/main -- crates/harmony-identity/src/identity.rs` shows comment-only changes); `harmony-tunnel`/`harmony-rawlink` still build as transports; mDNS `PeerTable` intact.
- [ ] **Step 6:** Commit: `git commit -am "ZEB-475: final reticulum sweep — fmt + clippy + workspace tests green"`

---

## Testing strategy (spec §7)

- **Non-regression is the whole proof:** workspace builds + all non-Reticulum tests pass after each task; the full `clippy --all-targets -D warnings` + `nextest --workspace` is green at Task 7. Reticulum tests are **removed, not skipped**.
- **No dangling secret:** Task 4 confirms `reticulum_identity_bytes` removal leaves no constructed-but-unused identity-secret derivation in node boot (`rg "Zeroizing|to_private_bytes" crates/harmony-node/src/main.rs` post-edit).
- **KEEP-list preserved:** the device-hash formula diff is comment-only; `reticulum_interop.rs` (identity) still green; tunnel/rawlink transports + mDNS PeerTable intact.

## PR

Single PR `ZEB-475` against harmony. Body references the Move 2 spec + this plan + the parent only as needed (keep parent epics OUT of the body per the Linear auto-close rule — body names **ZEB-475** only). harmony has no CI; the bot loop is Qodo + CodeAnt → address → one CodeRabbit pass (manual). After merge, harmony-client gets a trivial pinned-rev bump (Move 2 spec §6 PR-3 — separate, bundled into the next client PR).

## Self-review

- **Spec coverage:** §4.2 router/node/tunnel/rawlink/crate/Cargo → Tasks 4,5,6. §4.2.1 content/contacts/peers/platform/match-arms → Tasks 1,3. §4.3 KEEP-list → Conventions + Task 7 Step 5. §8 entanglements: identity untouched (Conventions); device-hash formula comment-only (Task 1/7); dangling secret (Task 4); `compute_dm_destination_hash` is client-side (ZEB-474, already merged) — N/A core. Recon extras (harmony-discovery RoutingHint, harmony-zenoh namespace) → Tasks 2,1.
- **Ordering soundness:** leaf/independent (1,2,3) → trunk router (4) → carrier branches (5,6) → crate delete (6) → sweep (7). Each compiles; clippy-clean deferred to 7 with the transient-`dead_code` note made explicit.
- **No invented symbols:** every removal targets a recon-confirmed existing site; `FORMAT_VERSION` bump verified against `store.rs:91/107`; `PeerAction::InitiateLink` kept (live at `manager.rs:92`); mDNS kept (out-of-scope).
