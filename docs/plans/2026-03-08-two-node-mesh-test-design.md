# Two-Node Mesh Discovery Test

**Date:** 2026-03-08
**Status:** Approved
**Scope:** harmony-os (crates/harmony-unikernel, justfile)

## Problem

The Harmony unikernel has a VirtIO-net driver, a Reticulum Node state machine, and an event loop that wires them together. The justfile has `run` and `run-peer` targets that launch two QEMU instances on the same multicast LAN. But there is no automated test proving that two nodes actually discover each other and exchange data. Manual testing with two terminals is the only verification path.

**Goal:** Automated tests at two layers — a fast host-side unit test and a QEMU integration test — proving two Harmony nodes discover each other via Reticulum announces and exchange heartbeats.

## Architecture

Two layers, each validating different parts of the stack:

```
Layer 1: Host-side unit test (cargo test -p harmony-unikernel)
  UnikernelRuntime A ──SendOnInterface──→ handle_packet() on B
  UnikernelRuntime B ──SendOnInterface──→ handle_packet() on A
  Assert: both emit PeerDiscovered, then HeartbeatReceived

Layer 2: QEMU integration test (just test-mesh)
  QEMU 1 ──VirtIO TX──→ multicast LAN ──→ VirtIO RX──→ QEMU 2
  QEMU 2 ──VirtIO TX──→ multicast LAN ──→ VirtIO RX──→ QEMU 1
  Assert: both serial logs contain [PEER+] and [HBT] lines
```

The unit test catches protocol/logic bugs fast. The QEMU test catches driver/framing/boot bugs.

## Layer 1: Two-Runtime Packet Shuttle Test

### Approach

Create two `UnikernelRuntime` instances with distinct identities in a `#[test]`. Register an interface and announcing destination on each. Run a simulated tick loop that collects `SendOnInterface` actions from each runtime and feeds them as `handle_packet` input to the other.

### Test Helper Refactor

The existing `make_runtime()` uses a fixed entropy seed (42). Add a `make_runtime_with_seed(seed: u8)` helper so two runtimes get distinct keypairs.

### Test 1: `two_runtimes_discover_each_other`

1. Create runtime A (seed 42) and runtime B (seed 99)
2. Register interface "net0" and announcing destination on each
3. Tick loop (up to 100 iterations, 100ms simulated per tick):
   - Tick both runtimes
   - Shuttle A's SendOnInterface outputs to B via handle_packet
   - Shuttle B's SendOnInterface outputs to A via handle_packet
   - Track PeerDiscovered events on both sides
4. Assert both discovered each other

The first announce fires on the first tick (next_announce_at = now on registration). Discovery should happen within 2-3 ticks.

### Test 2: `two_runtimes_exchange_heartbeats`

Extends test 1: after mutual discovery, continue ticking past `heartbeat_interval_ms` (5000ms). Assert both runtimes receive `HeartbeatReceived` events.

## Layer 2: QEMU Integration Test

### Approach

A `just test-mesh` recipe that launches two QEMU instances on the same multicast LAN, captures their serial output to temp files, and polls for discovery + heartbeat log lines.

### QEMU Configuration

Identical to existing `run`/`run-peer` targets:
- `-netdev socket,id=n0,mcast=230.0.0.1:1234` (multicast virtual LAN)
- `-device virtio-net-pci,netdev=n0`
- `-serial file:<tempfile>` (instead of stdio, so both run in background)
- `-cpu qemu64,+rdrand`
- No `-device isa-debug-exit` (instances run until killed)

### Success Criteria

Poll both log files for up to 30 seconds:
- Both contain `[PEER+]` (mutual peer discovery via Reticulum announce)
- Both contain `[HBT]` (heartbeat exchange — proves bidirectional data packets work)

### Timing

- First announce fires on the first timer tick after registration
- `register_announcing_destination` sets `next_announce_at = now`
- PIT timer starts at 0, first tick fires within ~10ms of entering the event loop
- Heartbeat interval is 5000ms, so heartbeats appear ~5s after discovery
- 30-second timeout provides ample margin

## Changes Required

### harmony-unikernel (crates/harmony-unikernel/src/event_loop.rs)

- Refactor `make_runtime()` to `make_runtime_with_seed(seed: u8)`
- Keep `make_runtime()` as `make_runtime_with_seed(42)` for backward compat
- Add `two_runtimes_discover_each_other` test
- Add `two_runtimes_exchange_heartbeats` test

### harmony-os repo root (justfile)

- Add `test-mesh: build` recipe

### No changes to

- harmony-reticulum (Ring 0) — protocol logic is correct
- harmony-boot — boot sequence unchanged
- VirtIO driver — already works

## Testing

### Layer 1 (host-side)
```bash
cargo test -p harmony-unikernel  # includes new two-runtime tests
```

### Layer 2 (QEMU)
```bash
just test-mesh  # builds + launches two QEMUs + verifies discovery
```

### Quality gates
```bash
cargo test --workspace
cargo clippy --workspace
cargo fmt --all -- --check
```

## Success Criteria

1. `cargo test -p harmony-unikernel` — new two-runtime tests pass
2. `just test-mesh` — two QEMU instances discover each other and exchange heartbeats within 30 seconds
3. `cargo clippy --workspace` — zero warnings
4. No changes to Ring 0 (harmony core)

## Future Work (NOT in scope)

- Multi-hop routing (3+ nodes)
- Message delivery beyond heartbeats
- aarch64 / Raspberry Pi 5 port
- Performance benchmarks
