# Two-Node Mesh Discovery Test — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Automated tests proving two Harmony unikernel nodes discover each other via Reticulum announces and exchange heartbeats — both at the host-side unit test level and in QEMU.

**Architecture:** Two test layers. Layer 1: host-side unit tests in harmony-unikernel that create two `UnikernelRuntime` instances and shuttle packets between them in a simulated tick loop. Layer 2: a `just test-mesh` recipe that launches two QEMU instances on a multicast virtual LAN and verifies serial output.

**Tech Stack:** Rust (harmony-unikernel, no_std compatible), QEMU x86_64, just (task runner), bash

**Design doc:** `docs/plans/2026-03-08-two-node-mesh-test-design.md`

---

### Task 1: Refactor test entropy helper to accept a seed

**Files:**
- Modify: `crates/harmony-unikernel/src/event_loop.rs:397-412` (test module helpers)

**Context:** The existing `test_entropy()` always starts its counter at 42, and `make_runtime()` uses it to generate an identity. Two runtimes built with the same seed would have identical keypairs and identical address hashes, making peer discovery impossible (a node ignores its own announces). We need a parameterized version so two runtimes get distinct identities.

**Step 1: Add `test_entropy_with_seed` and `make_runtime_with_seed`**

In the `#[cfg(test)] mod tests` block of `event_loop.rs`, add these two new helpers **above** the existing `test_entropy` and `make_runtime` functions. Then rewrite the originals to delegate:

```rust
fn test_entropy_with_seed(seed: u8) -> KernelEntropy<impl FnMut(&mut [u8])> {
    let mut counter: u8 = seed;
    KernelEntropy::new(move |buf: &mut [u8]| {
        for byte in buf.iter_mut() {
            *byte = counter;
            counter = counter.wrapping_add(7);
        }
    })
}

fn make_runtime_with_seed(
    seed: u8,
) -> UnikernelRuntime<KernelEntropy<impl FnMut(&mut [u8])>, MemoryState> {
    let mut entropy = test_entropy_with_seed(seed);
    let identity = PrivateIdentity::generate(&mut entropy);
    let persistence = MemoryState::new();
    UnikernelRuntime::new(identity, entropy, persistence)
}

fn test_entropy() -> KernelEntropy<impl FnMut(&mut [u8])> {
    test_entropy_with_seed(42)
}

fn make_runtime() -> UnikernelRuntime<KernelEntropy<impl FnMut(&mut [u8])>, MemoryState> {
    make_runtime_with_seed(42)
}
```

**Step 2: Run existing tests to verify no regressions**

Run: `cargo test -p harmony-unikernel`
Expected: All 10 existing tests pass — the originals delegate to the seeded versions with the same seed (42), so behavior is identical.

**Step 3: Commit**

```bash
git add crates/harmony-unikernel/src/event_loop.rs
git commit -m "refactor: parameterize test entropy seed for multi-runtime tests"
```

---

### Task 2: Two runtimes discover each other (packet shuttle test)

**Files:**
- Modify: `crates/harmony-unikernel/src/event_loop.rs` (add test at end of test module)

**Context:** This is the core test. Two `UnikernelRuntime` instances with distinct seeds simulate a network by collecting `SendOnInterface` actions from each runtime's `tick()` and feeding them as `handle_packet()` input to the other. The Reticulum Node sets `next_announce_at = now` on registration, so the first `TimerTick` (where `now >= registration_time`) triggers an announce. Each runtime should receive the other's announce and emit `PeerDiscovered`.

**Step 1: Write the failing test**

Add this test at the end of the `mod tests` block in `event_loop.rs`:

```rust
#[test]
fn two_runtimes_discover_each_other() {
    let mut rt_a = make_runtime_with_seed(42);
    let mut rt_b = make_runtime_with_seed(99);

    // Verify distinct identities (same seed = same keypair = useless test).
    assert_ne!(
        rt_a.identity().public_identity().address_hash,
        rt_b.identity().public_identity().address_hash,
        "runtimes must have distinct identities"
    );

    rt_a.register_interface("net0");
    rt_b.register_interface("net0");
    rt_a.register_announcing_destination("harmony", &["node"], 300_000, 0);
    rt_b.register_announcing_destination("harmony", &["node"], 300_000, 0);

    let mut a_discovered_b = false;
    let mut b_discovered_a = false;

    let addr_a = rt_a.identity().public_identity().address_hash;
    let addr_b = rt_b.identity().public_identity().address_hash;

    for tick in 0..100u64 {
        let now = tick * 1_000; // 1 second per tick

        // Tick both runtimes — collects announces and heartbeats.
        let actions_a = rt_a.tick(now);
        let actions_b = rt_b.tick(now);

        // Shuttle A's outbound packets to B.
        for action in &actions_a {
            if let RuntimeAction::SendOnInterface { raw, .. } = action {
                let results = rt_b.handle_packet("net0", raw.clone(), now);
                for r in &results {
                    if let RuntimeAction::PeerDiscovered { address_hash, .. } = r {
                        if *address_hash == addr_a {
                            b_discovered_a = true;
                        }
                    }
                }
            }
        }

        // Shuttle B's outbound packets to A.
        for action in &actions_b {
            if let RuntimeAction::SendOnInterface { raw, .. } = action {
                let results = rt_a.handle_packet("net0", raw.clone(), now);
                for r in &results {
                    if let RuntimeAction::PeerDiscovered { address_hash, .. } = r {
                        if *address_hash == addr_b {
                            a_discovered_b = true;
                        }
                    }
                }
            }
        }

        if a_discovered_b && b_discovered_a {
            break;
        }
    }

    assert!(b_discovered_a, "B should have discovered A via announce");
    assert!(a_discovered_b, "A should have discovered B via announce");
    assert_eq!(rt_a.peer_count(), 1);
    assert_eq!(rt_b.peer_count(), 1);
}
```

**Step 2: Run test to verify it passes**

Run: `cargo test -p harmony-unikernel two_runtimes_discover`
Expected: PASS. The first tick at `now=0` fires announces (because `next_announce_at=0`). But the Node internally checks `now >= next_announce_at` using **seconds** (`now_secs = now / 1000`), and `0 >= 0` is true. So both runtimes announce on tick 0 (now=0). The shuttle delivers each announce to the other runtime on the same tick iteration, triggering `PeerDiscovered`. Discovery should complete by tick 1 at most.

If the test fails because the Node requires `now > 0` (strict inequality or similar), the fix is to start the tick loop at `tick = 1` so `now = 1000` (1 second). Adjust accordingly.

**Step 3: Commit**

```bash
git add crates/harmony-unikernel/src/event_loop.rs
git commit -m "test: two runtimes discover each other via announce shuttle"
```

---

### Task 3: Two runtimes exchange heartbeats

**Files:**
- Modify: `crates/harmony-unikernel/src/event_loop.rs` (add test at end of test module)

**Context:** After mutual discovery (Task 2), heartbeats fire every `heartbeat_interval_ms` (5000ms). This test extends the tick loop past the heartbeat interval and asserts both runtimes receive `HeartbeatReceived` events. This proves bidirectional data packet routing works, not just announce processing.

**Step 1: Write the test**

Add after the previous test:

```rust
#[test]
fn two_runtimes_exchange_heartbeats() {
    let mut rt_a = make_runtime_with_seed(42);
    let mut rt_b = make_runtime_with_seed(99);

    rt_a.register_interface("net0");
    rt_b.register_interface("net0");
    rt_a.register_announcing_destination("harmony", &["node"], 300_000, 0);
    rt_b.register_announcing_destination("harmony", &["node"], 300_000, 0);

    let mut a_got_heartbeat = false;
    let mut b_got_heartbeat = false;

    // Run for 10 simulated seconds (heartbeat_interval_ms = 5000).
    // Discovery happens on tick 0-1. Heartbeats fire at ~6s.
    for tick in 0..100u64 {
        let now = tick * 100; // 100ms per tick, 100 ticks = 10 seconds

        let actions_a = rt_a.tick(now);
        let actions_b = rt_b.tick(now);

        // Shuttle A → B
        for action in &actions_a {
            if let RuntimeAction::SendOnInterface { raw, .. } = action {
                let results = rt_b.handle_packet("net0", raw.clone(), now);
                for r in &results {
                    if matches!(r, RuntimeAction::HeartbeatReceived { .. }) {
                        b_got_heartbeat = true;
                    }
                }
            }
        }

        // Shuttle B → A
        for action in &actions_b {
            if let RuntimeAction::SendOnInterface { raw, .. } = action {
                let results = rt_a.handle_packet("net0", raw.clone(), now);
                for r in &results {
                    if matches!(r, RuntimeAction::HeartbeatReceived { .. }) {
                        a_got_heartbeat = true;
                    }
                }
            }
        }

        if a_got_heartbeat && b_got_heartbeat {
            break;
        }
    }

    assert!(a_got_heartbeat, "A should have received a heartbeat from B");
    assert!(b_got_heartbeat, "B should have received a heartbeat from A");
}
```

**Step 2: Run test to verify it passes**

Run: `cargo test -p harmony-unikernel two_runtimes_exchange`
Expected: PASS. Discovery happens in the first few ticks (now < 1000ms). Heartbeats fire when `now - last_heartbeat_ms >= 5000` and there are live peers. With 100ms per tick, tick 50 hits now=5000ms. Heartbeats should appear around tick 50-60.

If heartbeats don't appear, check:
1. Does `last_heartbeat_ms` initialize to 0? (Yes, in `UnikernelRuntime::new`.) So the first heartbeat fires when `now >= 5000`.
2. Does `build_data_packet` + `route_packet` successfully route to the peer? The peer's `dest_hash` is stored in the peer table from the announce. `route_packet` broadcasts if no path exists, which should work since we shuttle all `SendOnInterface` packets.

**Step 3: Commit**

```bash
git add crates/harmony-unikernel/src/event_loop.rs
git commit -m "test: two runtimes exchange heartbeats after discovery"
```

---

### Task 4: QEMU two-node mesh test recipe

**Files:**
- Modify: `justfile` (add `test-mesh` recipe)

**Context:** This is the full-stack integration test. Two QEMU instances boot the same disk image, connect via multicast virtual LAN (mcast=230.0.0.1:1234), and must discover each other. The boot sequence logs `[PEER+] <hex>` when a peer is discovered and `[HBT] <hex> uptime=<ms>` when a heartbeat arrives. We capture serial output to temp files and poll for these markers.

**Important details:**
- Use `-serial file:<path>` (not `stdio`) so both QEMUs run in background.
- No `-device isa-debug-exit` — these instances run until killed.
- 30-second timeout is generous. Discovery should happen within 2-3 seconds (first announce fires on first tick), heartbeats within 8 seconds.
- QEMU auto-assigns distinct MAC addresses per instance, which means RDRAND produces distinct entropy, which means distinct identities.

**Step 1: Add the test-mesh recipe to justfile**

Add this recipe after the `test-qemu` recipe (after line 80):

```just
# Two-node mesh test — verifies peer discovery and heartbeat exchange over virtual LAN
test-mesh: build
    #!/usr/bin/env bash
    set -euo pipefail
    echo "Two-node mesh test: launching peers..."
    LOG_A=$(mktemp)
    LOG_B=$(mktemp)
    PID_A=""
    PID_B=""
    cleanup() {
        [ -n "$PID_A" ] && kill "$PID_A" 2>/dev/null || true
        [ -n "$PID_B" ] && kill "$PID_B" 2>/dev/null || true
        wait "$PID_A" 2>/dev/null || true
        wait "$PID_B" 2>/dev/null || true
        rm -f "$LOG_A" "$LOG_B"
    }
    trap cleanup EXIT

    qemu-system-x86_64 \
        -drive format=raw,file=target/harmony-boot-bios.img \
        -serial file:"$LOG_A" \
        -display none \
        -cpu qemu64,+rdrand \
        -device virtio-net-pci,netdev=n0 \
        -netdev socket,id=n0,mcast=230.0.0.1:1234 &
    PID_A=$!

    qemu-system-x86_64 \
        -drive format=raw,file=target/harmony-boot-bios.img \
        -serial file:"$LOG_B" \
        -display none \
        -cpu qemu64,+rdrand \
        -device virtio-net-pci,netdev=n0 \
        -netdev socket,id=n0,mcast=230.0.0.1:1234 &
    PID_B=$!

    for i in $(seq 1 30); do
        sleep 1
        A_PEER=$(grep -c '\[PEER+\]' "$LOG_A" 2>/dev/null || echo 0)
        B_PEER=$(grep -c '\[PEER+\]' "$LOG_B" 2>/dev/null || echo 0)
        A_HBT=$(grep -c '\[HBT\]' "$LOG_A" 2>/dev/null || echo 0)
        B_HBT=$(grep -c '\[HBT\]' "$LOG_B" 2>/dev/null || echo 0)
        echo "  [${i}s] A: ${A_PEER} peers, ${A_HBT} heartbeats | B: ${B_PEER} peers, ${B_HBT} heartbeats"
        if [ "$A_PEER" -gt 0 ] && [ "$B_PEER" -gt 0 ] \
           && [ "$A_HBT" -gt 0 ] && [ "$B_HBT" -gt 0 ]; then
            echo ""
            echo "=== Node A log ==="
            cat "$LOG_A"
            echo ""
            echo "=== Node B log ==="
            cat "$LOG_B"
            echo ""
            echo "Two-node mesh test: PASSED"
            exit 0
        fi
    done

    echo ""
    echo "=== Node A log ==="
    cat "$LOG_A"
    echo ""
    echo "=== Node B log ==="
    cat "$LOG_B"
    echo ""
    echo "Two-node mesh test: FAILED (timeout after 30s)"
    exit 1
```

**Step 2: Verify the recipe runs**

Run: `just test-mesh`
Expected: Both QEMUs boot, discover each other within a few seconds, exchange heartbeats within ~8 seconds, and the script exits with "PASSED". If it times out, the full logs from both nodes are printed for debugging.

If QEMU is not installed, the script will fail immediately with a clear error. This is expected in CI environments without QEMU.

**Step 3: Commit**

```bash
git add justfile
git commit -m "test: add just test-mesh for two-node QEMU discovery"
```

---

### Task 5: Quality gates

**Files:** None (verification only)

**Step 1: Run all workspace tests**

Run: `cargo test --workspace`
Expected: All tests pass including the 2 new two-runtime tests.

**Step 2: Run clippy**

Run: `cargo clippy --workspace`
Expected: Zero warnings.

**Step 3: Run format check**

Run: `cargo fmt --all -- --check`
Expected: No formatting issues.

**Step 4: Verify test-mesh passes (if QEMU available)**

Run: `just test-mesh`
Expected: PASSED within 30 seconds.

If QEMU is not available (e.g., CI machine), skip this step — the host-side unit tests cover the protocol logic. Document that `just test-mesh` requires `qemu-system-x86_64`.
