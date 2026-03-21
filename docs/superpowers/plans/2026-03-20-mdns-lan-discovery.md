# mDNS/DNS-SD LAN Peer Discovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Zero-config LAN peer discovery for harmony-node using mDNS/DNS-SD, with unicast fan-out to discovered peers alongside broadcast.

**Architecture:** Add `mdns-sd` crate for service advertisement and browsing. A new `discovery.rs` module manages a `PeerTable` that tracks discovered peers. The event loop gains a 5th select! arm for mDNS events, unicast fan-out in the send path, and passive liveness tracking via inbound packet timestamps.

**Tech Stack:** Rust, mdns-sd 0.18 (flume async), tokio current_thread, harmony-node event loop

**Spec:** `docs/superpowers/specs/2026-03-20-mdns-lan-discovery-design.md`

---

## File Structure

| File | Responsibility |
|------|---------------|
| `crates/harmony-node/Cargo.toml` | Add `mdns-sd` dependency |
| `crates/harmony-node/src/discovery.rs` | **New**: PeerTable struct, mDNS setup, TXT parsing |
| `crates/harmony-node/src/event_loop.rs` | 5th select! arm, unicast fan-out, mark_seen, stale eviction |
| `crates/harmony-node/src/main.rs` | `mod discovery;`, `--no-mdns`/`--mdns-stale-timeout` flags, wire `our_addr` bytes |
| `harmony-openwrt: harmony-node.init` | Two new UCI options |
| `harmony-openwrt: harmony-node.conf` | Defaults for new options |

---

### Task 1: PeerTable — data structure and unit tests

**Files:**
- Create: `crates/harmony-node/src/discovery.rs`
- Modify: `crates/harmony-node/src/main.rs:1` (add `mod discovery;`)

This is the core data structure. It has no I/O dependencies — pure HashMap + Instant tracking. TDD: write all tests first, then implement.

- [ ] **Step 1: Add `mod discovery;` to main.rs**

In `crates/harmony-node/src/main.rs`, add after the existing mod declarations (line 4):

```rust
mod discovery;
```

- [ ] **Step 2: Write the failing tests for PeerTable**

Create `crates/harmony-node/src/discovery.rs` with test module only:

```rust
use std::collections::{HashMap, HashSet};
use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::time::{Duration, Instant};

/// Tracks LAN peers discovered via mDNS.
///
/// Keyed by `SocketAddr` (IP + port). A reverse index maps Reticulum address
/// to the set of socket addresses for that peer (a single mDNS peer may
/// resolve to multiple IPs: IPv4 + IPv6, or multiple interfaces).
pub struct PeerTable {
    // will be filled in implementation step
}

/// Information about a discovered LAN peer.
pub struct PeerInfo {
    pub reticulum_addr: [u8; 16],
    pub proto_version: u8,
    pub last_seen: Instant,
    pub discovered_at: Instant,
}

impl PeerTable {
    pub fn new(our_addr: [u8; 16], stale_timeout: Duration) -> Self {
        todo!()
    }

    pub fn add_peer(&mut self, addr: SocketAddr, reticulum_addr: [u8; 16], proto_version: u8) {
        todo!()
    }

    pub fn remove_by_reticulum_addr(&mut self, reticulum_addr: &[u8; 16]) -> usize {
        todo!()
    }

    pub fn mark_seen(&mut self, src_addr: &SocketAddr) {
        todo!()
    }

    pub fn evict_stale(&mut self) -> Vec<SocketAddr> {
        todo!()
    }

    pub fn peer_addrs(&self) -> impl Iterator<Item = &SocketAddr> {
        std::iter::empty()
    }

    pub fn peer_count(&self) -> usize {
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn our_addr() -> [u8; 16] {
        [0xAA; 16]
    }

    fn peer_addr_a() -> [u8; 16] {
        [0xBB; 16]
    }

    fn peer_addr_b() -> [u8; 16] {
        [0xCC; 16]
    }

    fn socket(port: u16) -> SocketAddr {
        SocketAddr::new(IpAddr::V4(Ipv4Addr::new(192, 168, 1, port as u8)), 4242)
    }

    #[test]
    fn add_and_iterate_peers() {
        let mut table = PeerTable::new(our_addr(), Duration::from_secs(60));
        table.add_peer(socket(10), peer_addr_a(), 1);
        table.add_peer(socket(11), peer_addr_b(), 1);
        assert_eq!(table.peer_count(), 2);
        let addrs: Vec<_> = table.peer_addrs().collect();
        assert!(addrs.contains(&&socket(10)));
        assert!(addrs.contains(&&socket(11)));
    }

    #[test]
    fn self_discovery_filtered() {
        let mut table = PeerTable::new(our_addr(), Duration::from_secs(60));
        table.add_peer(socket(10), our_addr(), 1);
        assert_eq!(table.peer_count(), 0);
    }

    #[test]
    fn idempotent_add() {
        let mut table = PeerTable::new(our_addr(), Duration::from_secs(60));
        table.add_peer(socket(10), peer_addr_a(), 1);
        table.add_peer(socket(10), peer_addr_a(), 1);
        assert_eq!(table.peer_count(), 1);
    }

    #[test]
    fn multiple_addrs_per_reticulum_peer() {
        let mut table = PeerTable::new(our_addr(), Duration::from_secs(60));
        // Same Reticulum address, different socket addrs (IPv4 + IPv6 scenario)
        table.add_peer(socket(10), peer_addr_a(), 1);
        let ipv6_addr = SocketAddr::new(
            IpAddr::V6("::1".parse().unwrap()),
            4242,
        );
        table.add_peer(ipv6_addr, peer_addr_a(), 1);
        assert_eq!(table.peer_count(), 2);
    }

    #[test]
    fn remove_by_reticulum_addr_removes_all_sockets() {
        let mut table = PeerTable::new(our_addr(), Duration::from_secs(60));
        table.add_peer(socket(10), peer_addr_a(), 1);
        let ipv6_addr = SocketAddr::new(
            IpAddr::V6("::1".parse().unwrap()),
            4242,
        );
        table.add_peer(ipv6_addr, peer_addr_a(), 1);
        assert_eq!(table.peer_count(), 2);

        let removed = table.remove_by_reticulum_addr(&peer_addr_a());
        assert_eq!(removed, 2);
        assert_eq!(table.peer_count(), 0);
    }

    #[test]
    fn remove_by_reticulum_addr_unknown_is_zero() {
        let mut table = PeerTable::new(our_addr(), Duration::from_secs(60));
        let removed = table.remove_by_reticulum_addr(&peer_addr_a());
        assert_eq!(removed, 0);
    }

    #[test]
    fn mark_seen_updates_timestamp() {
        let mut table = PeerTable::new(our_addr(), Duration::from_secs(60));
        table.add_peer(socket(10), peer_addr_a(), 1);
        let before = table.peers.get(&socket(10)).unwrap().last_seen;
        // Spin briefly to ensure Instant advances
        std::thread::sleep(Duration::from_millis(2));
        table.mark_seen(&socket(10));
        let after = table.peers.get(&socket(10)).unwrap().last_seen;
        assert!(after > before);
    }

    #[test]
    fn mark_seen_unknown_addr_is_noop() {
        let mut table = PeerTable::new(our_addr(), Duration::from_secs(60));
        table.mark_seen(&socket(99)); // should not panic
        assert_eq!(table.peer_count(), 0);
    }

    #[test]
    fn evict_stale_removes_old_peers() {
        // Use a very short timeout so we can test eviction without long sleeps
        let mut table = PeerTable::new(our_addr(), Duration::from_millis(10));
        table.add_peer(socket(10), peer_addr_a(), 1);
        std::thread::sleep(Duration::from_millis(20));
        let evicted = table.evict_stale();
        assert_eq!(evicted.len(), 1);
        assert_eq!(evicted[0], socket(10));
        assert_eq!(table.peer_count(), 0);
    }

    #[test]
    fn evict_stale_keeps_fresh_peers() {
        let mut table = PeerTable::new(our_addr(), Duration::from_secs(60));
        table.add_peer(socket(10), peer_addr_a(), 1);
        let evicted = table.evict_stale();
        assert!(evicted.is_empty());
        assert_eq!(table.peer_count(), 1);
    }

    #[test]
    fn evict_stale_cleans_reverse_index() {
        let mut table = PeerTable::new(our_addr(), Duration::from_millis(10));
        table.add_peer(socket(10), peer_addr_a(), 1);
        std::thread::sleep(Duration::from_millis(20));
        table.evict_stale();
        // Reverse index should also be cleaned
        assert!(table.addr_to_sockets.is_empty());
    }
}
```

- [ ] **Step 3: Run tests — verify they fail**

Run: `cargo test -p harmony-node discovery::tests -- --nocapture 2>&1 | head -30`
Expected: compilation errors or `todo!()` panics

- [ ] **Step 4: Implement PeerTable**

Replace the struct and method bodies in `discovery.rs`:

```rust
pub struct PeerTable {
    pub(crate) peers: HashMap<SocketAddr, PeerInfo>,
    pub(crate) addr_to_sockets: HashMap<[u8; 16], HashSet<SocketAddr>>,
    our_addr: [u8; 16],
    stale_timeout: Duration,
}

impl PeerTable {
    pub fn new(our_addr: [u8; 16], stale_timeout: Duration) -> Self {
        Self {
            peers: HashMap::new(),
            addr_to_sockets: HashMap::new(),
            our_addr,
            stale_timeout,
        }
    }

    pub fn add_peer(&mut self, addr: SocketAddr, reticulum_addr: [u8; 16], proto_version: u8) {
        if reticulum_addr == self.our_addr {
            return;
        }
        let now = Instant::now();
        self.peers
            .entry(addr)
            .and_modify(|p| {
                p.last_seen = now;
                p.proto_version = proto_version;
            })
            .or_insert(PeerInfo {
                reticulum_addr,
                proto_version,
                last_seen: now,
                discovered_at: now,
            });
        self.addr_to_sockets
            .entry(reticulum_addr)
            .or_default()
            .insert(addr);
    }

    pub fn remove_by_reticulum_addr(&mut self, reticulum_addr: &[u8; 16]) -> usize {
        let sockets = match self.addr_to_sockets.remove(reticulum_addr) {
            Some(s) => s,
            None => return 0,
        };
        let count = sockets.len();
        for addr in &sockets {
            self.peers.remove(addr);
        }
        count
    }

    pub fn mark_seen(&mut self, src_addr: &SocketAddr) {
        if let Some(peer) = self.peers.get_mut(src_addr) {
            peer.last_seen = Instant::now();
        }
    }

    pub fn evict_stale(&mut self) -> Vec<SocketAddr> {
        let now = Instant::now();
        let stale: Vec<SocketAddr> = self
            .peers
            .iter()
            .filter(|(_, info)| now.duration_since(info.last_seen) > self.stale_timeout)
            .map(|(addr, _)| *addr)
            .collect();

        for addr in &stale {
            if let Some(info) = self.peers.remove(addr) {
                if let Some(sockets) = self.addr_to_sockets.get_mut(&info.reticulum_addr) {
                    sockets.remove(addr);
                    if sockets.is_empty() {
                        self.addr_to_sockets.remove(&info.reticulum_addr);
                    }
                }
            }
        }
        stale
    }

    pub fn peer_addrs(&self) -> impl Iterator<Item = &SocketAddr> {
        self.peers.keys()
    }

    pub fn peer_count(&self) -> usize {
        self.peers.len()
    }
}
```

- [ ] **Step 5: Run tests — verify they pass**

Run: `cargo test -p harmony-node discovery::tests -v`
Expected: all 10 tests pass

- [ ] **Step 6: Commit**

```bash
git add crates/harmony-node/src/discovery.rs crates/harmony-node/src/main.rs
git commit -m "feat(harmony-node): add PeerTable for mDNS LAN peer discovery

TDD: 10 unit tests covering add, remove, mark_seen, evict_stale,
self-filtering, idempotent add, multi-address peers, reverse index cleanup."
```

---

### Task 2: mDNS setup and TXT parsing

**Files:**
- Modify: `crates/harmony-node/Cargo.toml` (add mdns-sd dep)
- Modify: `crates/harmony-node/src/discovery.rs` (add start_mdns, parse helpers)

Add the `mdns-sd` crate and the functions that start the mDNS daemon and parse TXT records from discovered services.

- [ ] **Step 1: Add mdns-sd dependency**

In `crates/harmony-node/Cargo.toml`, add to `[dependencies]`:

```toml
mdns-sd = { version = "0.18", features = ["async"] }
```

- [ ] **Step 2: Write failing tests for TXT parsing**

Add to `discovery.rs` (above the `#[cfg(test)]` module):

```rust
use mdns_sd::{ServiceDaemon, ServiceEvent, ServiceInfo};

const SERVICE_TYPE: &str = "_harmony._udp.local.";

/// Parse the Reticulum address from a DNS-SD TXT record property set.
///
/// Expects a property `addr=<32 hex chars>`. Returns `None` if missing or malformed.
pub fn parse_txt_addr(properties: &mdns_sd::TxtProperties) -> Option<[u8; 16]> {
    todo!()
}

/// Parse the protocol version from a DNS-SD TXT record property set.
///
/// Expects a property `proto=<u8>`. Returns 0 if missing or malformed.
pub fn parse_txt_proto(properties: &mdns_sd::TxtProperties) -> u8 {
    todo!()
}

/// Parse the Reticulum address from an mDNS instance fullname.
///
/// The fullname has the format `<32 hex chars>._harmony._udp.local.`
/// Returns `None` if the prefix is not valid hex or wrong length.
pub fn parse_instance_addr(fullname: &str) -> Option<[u8; 16]> {
    todo!()
}

/// Start the mDNS daemon: register our service and browse for peers.
///
/// Returns the daemon handle (for shutdown) and the flume Receiver for events.
/// On failure (e.g., can't bind port 5353), returns `Err` — caller should log
/// and proceed without discovery.
pub fn start_mdns(
    listen_port: u16,
    reticulum_addr: &[u8; 16],
) -> Result<(ServiceDaemon, mdns_sd::Receiver<ServiceEvent>), Box<dyn std::error::Error + Send + Sync>> {
    todo!()
}
```

Add these tests to the `tests` module:

```rust
    #[test]
    fn parse_instance_addr_valid() {
        let hex_addr = "aabbccdd11223344aabbccdd11223344";
        let fullname = format!("{hex_addr}._harmony._udp.local.");
        let result = parse_instance_addr(&fullname);
        assert!(result.is_some());
        let bytes = result.unwrap();
        assert_eq!(hex::encode(bytes), hex_addr);
    }

    #[test]
    fn parse_instance_addr_invalid_hex() {
        let fullname = "not_valid_hex_here_0000000000000._harmony._udp.local.";
        assert!(parse_instance_addr(fullname).is_none());
    }

    #[test]
    fn parse_instance_addr_wrong_suffix() {
        let fullname = "aabbccdd11223344aabbccdd11223344._other._tcp.local.";
        // Still parses — we only look at the prefix before the first dot
        let result = parse_instance_addr(fullname);
        assert!(result.is_some());
    }

    #[test]
    fn parse_instance_addr_too_short() {
        let fullname = "aabb._harmony._udp.local.";
        assert!(parse_instance_addr(fullname).is_none());
    }
```

- [ ] **Step 3: Run tests — verify they fail**

Run: `cargo test -p harmony-node discovery::tests -- --nocapture 2>&1 | head -30`
Expected: `todo!()` panics for the new tests, existing tests still pass

- [ ] **Step 4: Implement parse functions**

```rust
pub fn parse_txt_addr(properties: &mdns_sd::TxtProperties) -> Option<[u8; 16]> {
    let val_str = properties.get_property_val_str("addr")?;
    let bytes = hex::decode(val_str).ok()?;
    if bytes.len() != 16 {
        return None;
    }
    let mut arr = [0u8; 16];
    arr.copy_from_slice(&bytes);
    Some(arr)
}

pub fn parse_txt_proto(properties: &mdns_sd::TxtProperties) -> u8 {
    properties
        .get_property_val_str("proto")
        .and_then(|s| s.parse::<u8>().ok())
        .unwrap_or(0)
}

pub fn parse_instance_addr(fullname: &str) -> Option<[u8; 16]> {
    let prefix = fullname.split('.').next()?;
    if prefix.len() != 32 {
        return None;
    }
    let bytes = hex::decode(prefix).ok()?;
    if bytes.len() != 16 {
        return None;
    }
    let mut arr = [0u8; 16];
    arr.copy_from_slice(&bytes);
    Some(arr)
}
```

- [ ] **Step 5: Implement start_mdns**

```rust
pub fn start_mdns(
    listen_port: u16,
    reticulum_addr: &[u8; 16],
) -> Result<(ServiceDaemon, mdns_sd::Receiver<ServiceEvent>), Box<dyn std::error::Error + Send + Sync>> {
    let daemon = ServiceDaemon::new()?;
    let instance_name = hex::encode(reticulum_addr);
    let host = format!("{instance_name}.local.");

    let hex_addr = hex::encode(reticulum_addr);
    let properties: [(&str, &str); 2] = [
        ("addr", &hex_addr),
        ("proto", "1"),
    ];

    let service = ServiceInfo::new(
        SERVICE_TYPE,
        &instance_name,
        &host,
        "",  // empty = auto-detect IP addresses
        listen_port,
        &properties[..],
    )?;

    daemon.register(service)?;

    let receiver = daemon.browse(SERVICE_TYPE)?;

    Ok((daemon, receiver))
}
```

Note: The `ServiceInfo::new()` API may require adjustments depending on the exact mdns-sd 0.18 signature. The implementer should check `docs.rs/mdns-sd` for the current constructor. The key fields are: service type, instance name, host, IP (empty for auto), port, and TXT properties.

Note: `parse_txt_addr` and `parse_txt_proto` are not unit-tested because
`mdns_sd::TxtProperties` has no public constructor — it can only be obtained
from a real `ServiceResolved` event. These functions are exercised by the
integration test in Task 3 (two daemons discovering each other) and by manual
testing on the router. The `parse_instance_addr` function IS testable (plain
string parsing) and has 4 unit tests above.

- [ ] **Step 6: Run tests — verify parse tests pass**

Run: `cargo test -p harmony-node discovery::tests -v`
Expected: all 14 tests pass (10 PeerTable + 4 parse_instance_addr)

- [ ] **Step 7: Commit**

```bash
git add crates/harmony-node/Cargo.toml crates/harmony-node/src/discovery.rs
git commit -m "feat(harmony-node): add mdns-sd service registration and TXT parsing

Registers _harmony._udp.local. with addr + proto TXT records.
Browses for peers. Parse helpers for ServiceResolved/ServiceRemoved events."
```

---

### Task 3: Event loop integration — 5th select! arm and unicast fan-out

**Files:**
- Modify: `crates/harmony-node/src/event_loop.rs:1-384`
- Modify: `crates/harmony-node/src/main.rs:185-276` (Run command)

This is the core wiring: add mDNS events to the select loop, change outbound UDP to broadcast+unicast, add passive liveness tracking. No new tests — this is I/O integration code tested by the existing event loop and by manual/integration testing.

- [ ] **Step 1: Add CLI flags to Run command**

In `crates/harmony-node/src/main.rs`, add two new fields to the `Run` variant (after `listen_address`):

```rust
        /// Disable mDNS peer discovery (broadcast-only mode)
        #[arg(long)]
        no_mdns: bool,
        /// Seconds before evicting a silent mDNS peer (default: 60)
        #[arg(long, default_value_t = 60)]
        mdns_stale_timeout: u64,
```

- [ ] **Step 2: Derive `our_addr` bytes and pass to event loop**

In the `Commands::Run` match arm in `main.rs`, after the identity is loaded (around line 241), save the raw address bytes before dropping the identity:

```rust
            let our_addr_bytes: [u8; 16] = identity.ed25519.public_identity().address_hash;
            let node_addr = hex::encode(our_addr_bytes);
```

Update the `event_loop::run()` call to pass the new parameters:

```rust
            crate::event_loop::run(
                rt,
                startup_actions,
                listen_addr,
                if no_mdns { None } else { Some(our_addr_bytes) },
                Duration::from_secs(mdns_stale_timeout),
            ).await
                .map_err(|e| -> Box<dyn std::error::Error> { e.to_string().into() })?;
```

Add `use std::time::Duration;` if not already imported.

Also destructure the new fields in the `Commands::Run` pattern:

```rust
            no_mdns,
            mdns_stale_timeout,
```

- [ ] **Step 3: Update `event_loop::run` signature**

Change the signature of `run()` in `event_loop.rs`:

```rust
pub async fn run(
    mut runtime: NodeRuntime<MemoryBookStore>,
    startup_actions: Vec<RuntimeAction>,
    listen_addr: SocketAddr,
    mdns_addr: Option<[u8; 16]>,
    mdns_stale_timeout: Duration,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
```

- [ ] **Step 4: Initialize mDNS and PeerTable**

After the UDP socket setup and before the Zenoh session, add:

```rust
    // ── mDNS peer discovery (optional) ──────────────────────────────────────
    use crate::discovery::{self, PeerTable};
    let mut peer_table = PeerTable::new(
        mdns_addr.unwrap_or([0; 16]),
        mdns_stale_timeout,
    );
    let mdns_state = match mdns_addr {
        Some(addr) => match discovery::start_mdns(listen_addr.port(), &addr) {
            Ok((daemon, rx)) => {
                tracing::info!("mDNS discovery started on _harmony._udp.local.");
                Some((daemon, rx))
            }
            Err(e) => {
                tracing::warn!(err = %e, "mDNS discovery failed to start — broadcast only");
                None
            }
        },
        None => {
            tracing::info!("mDNS discovery disabled (--no-mdns)");
            None
        }
    };
```

- [ ] **Step 5: Update `dispatch_action` to accept `&PeerTable`**

Change the signature:

```rust
async fn dispatch_action(
    action: RuntimeAction,
    session: &zenoh::Session,
    zenoh_tx: &mpsc::Sender<ZenohEvent>,
    udp: &UdpSocket,
    broadcast_addr: &SocketAddr,
    peer_table: Option<&PeerTable>,
) {
```

Update the `SendOnInterface` arm:

```rust
        RuntimeAction::SendOnInterface { raw, .. } => {
            // Always broadcast (preserves compatibility with non-mDNS nodes)
            if let Err(e) = udp.send_to(&raw, broadcast_addr).await {
                tracing::warn!(err = %e, "UDP broadcast send error");
            }
            // Additionally unicast to each known mDNS peer
            if let Some(peers) = peer_table {
                for addr in peers.peer_addrs() {
                    if let Err(e) = udp.send_to(&raw, addr).await {
                        tracing::warn!(peer = %addr, err = %e, "UDP unicast send error");
                    }
                }
            }
        }
```

Update all `dispatch_action` call sites to pass `Some(&peer_table)` for runtime calls and `None` for startup calls (startup happens before mDNS is initialized):

Startup (before select loop):
```rust
        dispatch_action(action, &session, &zenoh_tx, &udp, &broadcast_addr, None).await;
```

Inside select loop (after `runtime.tick()`):
```rust
                dispatch_action(action, &session, &zenoh_tx, &udp, &broadcast_addr, Some(&peer_table)).await;
```

- [ ] **Step 6: Add Arm 1 mark_seen and Arm 2 stale eviction**

In Arm 1 (UDP recv), after the `push_event` call (inside the `Ok((len, _src))` arm), change `_src` to `src` and add:

```rust
                    Ok((len, src)) => {
                        peer_table.mark_seen(&src);
                        runtime.push_event(RuntimeEvent::InboundPacket {
```

In Arm 2 (timer tick), after `should_tick = true;`, add stale eviction:

```rust
            _ = timer.tick() => {
                runtime.push_event(RuntimeEvent::TimerTick { now: now_ms() });
                should_tick = true;
                // Evict stale mDNS peers
                for addr in peer_table.evict_stale() {
                    tracing::info!(peer = %addr, "evicted stale mDNS peer");
                }
            }
```

- [ ] **Step 7: Add Arm 5 — mDNS events**

Add a new arm to the `tokio::select!` block, between Arm 3 (Zenoh) and Arm 4 (shutdown). This arm is conditionally active only when mDNS is enabled:

```rust
            // Arm 5: mDNS discovery events (when enabled).
            Some(Ok(event)) = async {
                match &mdns_state {
                    Some((_, rx)) => Some(rx.recv_async().await),
                    None => std::future::pending::<Option<Result<mdns_sd::ServiceEvent, mdns_sd::Error>>>().await,
                }
            } => {
                match event {
                    mdns_sd::ServiceEvent::ServiceResolved(info) => {
                        let properties = info.get_properties();
                        if let Some(reticulum_addr) = discovery::parse_txt_addr(properties) {
                            let proto = discovery::parse_txt_proto(properties);
                            for ip in info.get_addresses() {
                                let socket_addr = SocketAddr::new(ip.to_ip_addr(), info.get_port());
                                if peer_table.add_peer(socket_addr, reticulum_addr, proto) {
                                    tracing::info!(
                                        peer = %socket_addr,
                                        addr = %hex::encode(reticulum_addr),
                                        proto,
                                        "mDNS peer discovered"
                                    );
                                }
                            }
                        }
                    }
                    mdns_sd::ServiceEvent::ServiceRemoved(_service_type, fullname) => {
                        if let Some(reticulum_addr) = discovery::parse_instance_addr(&fullname) {
                            let removed = peer_table.remove_by_reticulum_addr(&reticulum_addr);
                            if removed > 0 {
                                tracing::info!(
                                    addr = %hex::encode(reticulum_addr),
                                    removed,
                                    "mDNS peer removed (goodbye)"
                                );
                            }
                        }
                    }
                    _ => {} // ServiceFound (triggers resolve), SearchStarted, SearchStopped
                }
            }
```

Note: `add_peer` currently returns `()`. The implementer should change it to return `bool` (true if this is a new peer, false if update) so we only log on first discovery. This is a minor change to the PeerTable:

```rust
    pub fn add_peer(&mut self, addr: SocketAddr, reticulum_addr: [u8; 16], proto_version: u8) -> bool {
        if reticulum_addr == self.our_addr {
            return false;
        }
        let now = Instant::now();
        let is_new = !self.peers.contains_key(&addr);
        self.peers
            .entry(addr)
            .and_modify(|p| {
                p.last_seen = now;
                p.proto_version = proto_version;
            })
            .or_insert(PeerInfo {
                reticulum_addr,
                proto_version,
                last_seen: now,
                discovered_at: now,
            });
        self.addr_to_sockets
            .entry(reticulum_addr)
            .or_default()
            .insert(addr);
        is_new
    }
```

Update the existing tests to handle the `bool` return value (they can ignore it or assert on it).

- [ ] **Step 8: Add shutdown cleanup**

After the `loop { ... }` block exits, before `Ok(())`, add:

```rust
    // Shutdown mDNS daemon — sends goodbye packet so peers learn immediately
    if let Some((daemon, _)) = mdns_state {
        if let Err(e) = daemon.shutdown() {
            tracing::warn!(err = %e, "mDNS shutdown error");
        }
    }
```

- [ ] **Step 9: Run existing tests — verify nothing breaks**

Run: `cargo test -p harmony-node -v`
Expected: all existing tests pass (CLI tests don't exercise the event loop)

- [ ] **Step 10: Commit**

```bash
git add crates/harmony-node/src/event_loop.rs crates/harmony-node/src/main.rs crates/harmony-node/src/discovery.rs
git commit -m "feat(harmony-node): wire mDNS discovery into event loop

5th select! arm for mdns-sd events. Broadcast always sent, plus unicast
to each known mDNS peer. Passive liveness via inbound packet timestamps.
Stale eviction on timer tick. Graceful mDNS shutdown on SIGTERM."
```

---

### Task 4: CLI tests for new flags

**Files:**
- Modify: `crates/harmony-node/src/main.rs` (tests module)

- [ ] **Step 1: Write tests for new CLI flags**

Add to the `#[cfg(test)] mod tests` in `main.rs`:

```rust
    #[test]
    fn cli_parses_no_mdns_flag() {
        let cli = Cli::try_parse_from(["harmony", "run", "--no-mdns"]).unwrap();
        if let Commands::Run { no_mdns, .. } = cli.command {
            assert!(no_mdns);
        } else {
            panic!("expected Run command");
        }
    }

    #[test]
    fn cli_no_mdns_default_false() {
        let cli = Cli::try_parse_from(["harmony", "run"]).unwrap();
        if let Commands::Run { no_mdns, .. } = cli.command {
            assert!(!no_mdns);
        } else {
            panic!("expected Run command");
        }
    }

    #[test]
    fn cli_parses_mdns_stale_timeout() {
        let cli = Cli::try_parse_from(["harmony", "run", "--mdns-stale-timeout", "120"]).unwrap();
        if let Commands::Run { mdns_stale_timeout, .. } = cli.command {
            assert_eq!(mdns_stale_timeout, 120);
        } else {
            panic!("expected Run command");
        }
    }

    #[test]
    fn cli_mdns_stale_timeout_default() {
        let cli = Cli::try_parse_from(["harmony", "run"]).unwrap();
        if let Commands::Run { mdns_stale_timeout, .. } = cli.command {
            assert_eq!(mdns_stale_timeout, 60);
        } else {
            panic!("expected Run command");
        }
    }
```

- [ ] **Step 2: Run tests**

Run: `cargo test -p harmony-node tests:: -v`
Expected: all tests pass (new + existing)

- [ ] **Step 3: Commit**

```bash
git add crates/harmony-node/src/main.rs
git commit -m "test(harmony-node): CLI tests for --no-mdns and --mdns-stale-timeout"
```

---

### Task 5: OpenWRT UCI integration

**Files:**
- Modify: `harmony-openwrt/harmony-node/files/harmony-node.init`
- Modify: `harmony-openwrt/harmony-node/files/harmony-node.conf`

This task is in the `harmony-openwrt` repo (separate from harmony core).

- [ ] **Step 1: Add UCI options to conf**

In `harmony-openwrt/harmony-node/files/harmony-node.conf`, add after `no_public_ephemeral_announce`:

```
	option no_mdns '0'
	option mdns_stale_timeout '60'
```

- [ ] **Step 2: Add UCI options to init script**

In `harmony-openwrt/harmony-node/files/harmony-node.init`:

Add to the `local` declarations at the top of `start_service()`:

```sh
	local no_mdns mdns_stale_timeout
```

Add config_get lines after the `no_public_ephemeral_announce` line:

```sh
	config_get_bool no_mdns main no_mdns 0
	config_get mdns_stale_timeout main mdns_stale_timeout 60
```

Add the flag appending after the `no_public_ephemeral_announce` append:

```sh
	[ "$no_mdns" -eq 1 ] && \
		procd_append_param command --no-mdns
	procd_append_param command --mdns-stale-timeout "$mdns_stale_timeout"
```

- [ ] **Step 3: Commit (in harmony-openwrt repo)**

```bash
cd /Users/zeblith/work/zeblithic/harmony-openwrt
git add harmony-node/files/harmony-node.init harmony-node/files/harmony-node.conf
git commit -m "feat: UCI options for mDNS peer discovery (no_mdns, mdns_stale_timeout)"
```

---

### Task 6: Full workspace verification

**Files:** None (verification only)

- [ ] **Step 1: Run full workspace tests in harmony core**

Run: `cargo test --workspace`
Expected: all tests pass (365+ existing + ~14 new discovery tests)

- [ ] **Step 2: Run clippy**

Run: `cargo clippy --workspace`
Expected: no warnings on new code

- [ ] **Step 3: Run fmt check**

Run: `cargo fmt --all -- --check`
Expected: no formatting issues

- [ ] **Step 4: Verify OpenWRT init script syntax**

Run: `sh -n harmony-openwrt/harmony-node/files/harmony-node.init`
Expected: no syntax errors
