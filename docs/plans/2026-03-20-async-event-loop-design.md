# Async Event Loop — Design

**Date:** 2026-03-20
**Status:** Proposed
**Bead:** harmony-622

## Problem

`NodeRuntime` is a complete sans-I/O state machine with a three-tier priority event loop (router → storage → compute), but nothing drives it. `harmony run` constructs the runtime, prints startup info, and exits. The node cannot participate in the mesh.

## Solution

Wire tokio + Zenoh + a UDP socket to `NodeRuntime` via a single `tokio::select!` loop. Three event sources (UDP recv, Zenoh callbacks, timer) feed `push_event()`. After any event, `tick()` produces actions dispatched to the appropriate I/O channel.

## Dependencies

**New dependencies for `harmony-node`:**

| Crate | Features | Purpose |
|-------|----------|---------|
| `tokio` | `rt`, `net`, `time`, `macros`, `sync` | Async runtime, UDP socket, timer, `#[tokio::main]`, mpsc channels |
| `zenoh` | default | Pub/sub session for content layer (queryables, subscriptions, queries) |

Zenoh brings tokio transitively — using tokio directly avoids impedance mismatch.

**Binary size impact:** Stripped cross-compiled binary grows from ~3 MB to ~6-8 MB. Acceptable for routers with 1-2 GB RAM.

## Event Loop Architecture

```
┌─────────────────────────────────────────────────────┐
│                  tokio::select! loop                 │
│                                                      │
│  UDP recv ──→ push_event(InboundPacket)              │
│  Timer    ──→ push_event(TimerTick)                  │
│  Zenoh rx ──→ push_event(QueryReceived /             │
│               SubscriptionMessage / ComputeQuery)    │
│                                                      │
│  ───── after any event ─────                         │
│  actions = runtime.tick()                            │
│  for action in actions:                              │
│    SendOnInterface  → udp_socket.send_to()           │
│    SendReply        → zenoh query.reply()            │
│    Publish          → zenoh session.put()            │
│    DeclareQueryable → zenoh session.declare_queryable│
│    Subscribe        → zenoh session.declare_subscriber│
│    FetchContent     → tokio::spawn zenoh get() →     │
│                       mpsc → ContentFetchResponse    │
│    FetchModule      → tokio::spawn zenoh get() →     │
│                       mpsc → ModuleFetchResponse     │
└─────────────────────────────────────────────────────┘
```

### Event Sources

**UDP socket** (`tokio::net::UdpSocket`): Bound to `0.0.0.0:4242` (default, overridable via `--listen-address`). Each `recv_from` produces a `RuntimeEvent::InboundPacket` with the raw bytes, interface name `"udp0"`, and current monotonic timestamp.

**Timer** (`tokio::time::interval`): Fires every 250ms. Produces `RuntimeEvent::TimerTick` with the current monotonic timestamp. 250ms matches the Reticulum announce/expiry granularity.

**Zenoh channel** (`tokio::sync::mpsc`): Zenoh subscribers and queryables deliver events via async callbacks. Each callback sends a `RuntimeEvent` through an `mpsc::Sender`. The `select!` loop receives from the corresponding `mpsc::Receiver`.

### Action Dispatch

After `tick()` returns actions, each is dispatched:

| Action | Dispatch |
|--------|----------|
| `SendOnInterface` | `udp_socket.send_to(&raw, peer_addr)` |
| `SendReply` | Look up pending query by `query_id`, call `query.reply()` |
| `Publish` | `zenoh_session.put(key_expr, payload)` |
| `DeclareQueryable` | `zenoh_session.declare_queryable(key_expr)` with callback → mpsc |
| `Subscribe` | `zenoh_session.declare_subscriber(key_expr)` with callback → mpsc |
| `FetchContent` | `tokio::spawn` a Zenoh `get()`, send result back as `ContentFetchResponse` via mpsc |
| `FetchModule` | `tokio::spawn` a Zenoh `get()`, send result back as `ModuleFetchResponse` via mpsc |

### Pending Queries

`SendReply` references a `query_id` that must map back to the original Zenoh `Query` object (needed to call `query.reply()`). The event loop maintains a `HashMap<u64, zenoh::query::Query>` populated when `QueryReceived` events are created, drained when `SendReply` actions are dispatched.

### Startup Sequence

1. Load identity from key file (already done — `identity_file::load_or_generate`)
2. Bind `tokio::net::UdpSocket` on `--listen-address` (default `0.0.0.0:4242`)
3. Open Zenoh session (`zenoh::open(zenoh::Config::default())`)
4. Construct `NodeRuntime::new(config, store)` → get `(runtime, startup_actions)`
5. Execute startup actions: declare 18 queryables + 4 subscribers, store handles
6. Enter `select!` loop

### Zenoh Session Configuration

Use `zenoh::Config::default()` for v1. This enables:
- **Scouting** — automatic peer discovery via UDP multicast (works on LAN)
- **Peer mode** — nodes connect directly to each other (no router required)

Custom Zenoh config (routers, TCP endpoints, TLS) is deferred to harmony-r4o (config file support).

## File Structure

| File | Responsibility |
|------|---------------|
| Create: `crates/harmony-node/src/event_loop.rs` | `run()` async fn — select loop, action dispatch, Zenoh bridge |
| Modify: `crates/harmony-node/src/main.rs` | `#[tokio::main]`, `--listen-address`, call `event_loop::run()` |
| Modify: `crates/harmony-node/Cargo.toml` | Add `tokio`, `zenoh` |

`event_loop.rs` is the only new file. `runtime.rs` is untouched — sans-I/O stays sans-I/O.

## CLI Changes

Add one flag to the `Run` subcommand:

```rust
/// UDP listen address for Reticulum mesh packets
#[arg(long, default_value = "0.0.0.0:4242")]
listen_address: String,
```

## Timestamps

`NodeRuntime` uses `u64` timestamps in milliseconds (abstract kernel time). The event loop uses `std::time::Instant::now()` at startup as the epoch and derives all `now` values as `elapsed().as_millis() as u64`. This matches the pattern used in harmony-os's boot code.

## Error Handling

| Condition | Behavior |
|-----------|----------|
| UDP bind fails (port in use) | Error and exit: "Failed to bind <addr>: <error>" |
| Zenoh session open fails | Error and exit: "Failed to open Zenoh session: <error>" |
| UDP recv error | Log warning, continue loop (transient network errors are normal) |
| Zenoh callback error | Log warning, continue loop |
| `tick()` panics | Unrecoverable — let it propagate (bug in sans-I/O logic) |

The event loop runs until the process receives SIGTERM (handled by a future bead, harmony-438). For now, Ctrl+C / SIGINT terminates the tokio runtime naturally.

## Testing

### Unit tests (existing, unaffected)
- `NodeRuntime` 70+ tests — sans-I/O, no async, all pass unchanged

### New CLI tests
- Parse `--listen-address` flag with custom value
- Parse `--listen-address` with default (0.0.0.0:4242)

### Integration test
- Spawn `harmony run` as a child process
- Verify it binds the UDP port (doesn't exit)
- Send a UDP packet, verify the process is still alive
- Kill the process, verify clean exit

### Not tested (deferred)
- Zenoh session interactions — requires Zenoh router/peer, deferred to manual testing
- Full Reticulum announce/link protocol — covered by existing sans-I/O unit tests

## What Does NOT Change

- `NodeRuntime` / `runtime.rs` — untouched, sans-I/O preserved
- `identity_file.rs` — untouched
- All library crates — no changes to crypto, identity, reticulum, content, etc.
- harmony-os — drives the same `NodeRuntime` from its own bare-metal event loop
- Cross-compilation — tokio + zenoh build for musl (both support aarch64-unknown-linux-musl)

## Future Work

| Item | Bead | Notes |
|------|------|-------|
| Graceful shutdown (SIGTERM/SIGHUP) | harmony-438 | `tokio::signal` handler in the select loop |
| Structured logging | harmony-jri | Replace `eprintln!` with `tracing` |
| Config file (TOML) | harmony-r4o | Zenoh config, per-interface settings |
| LAN peer discovery (mDNS) | harmony-k4g | Complement Zenoh scouting |
| Enable OpenWRT service | harmony-os-v5b | Bump PKG_SOURCE_VERSION, enable by default |
| Firewall rules | harmony-os-b9o | Allow UDP 4242 through OpenWRT firewall |
| WiFi mesh interface | harmony-os-pxp | Configure dedicated mesh interface |
