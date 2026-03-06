# Platform-Abstract Traits — Design

**Date:** 2026-03-06
**Status:** Approved
**Bead:** harmony-2yl

## Problem

Ring 0's sans-I/O state machines need platform services (randomness, networking, persistence) but must remain decoupled from any specific runtime. Currently these boundaries are implicit: RNG is passed per-call, network I/O flows through events/actions without a formal trait, and persistent state doesn't exist at all. Ring 1 (unikernel) needs concrete trait bounds to implement against.

## Solution

A new `harmony-platform` crate containing three platform-abstract traits, an error type, and one in-memory test implementation. The crate is `no_std` compatible and sits at the bottom of the dependency graph alongside `harmony-crypto`.

## Crate Position in Dependency Graph

```
harmony-platform  (new: traits only, no_std)
  └── harmony-crypto       (adds EntropySource usage)
      └── harmony-identity (inherits)
          └── harmony-reticulum (adds NetworkInterface, PersistentState)
harmony-platform
  └── harmony-zenoh        (adds NetworkInterface)
  └── harmony-content      (adds PersistentState)
```

## Trait 1: EntropySource

Platform-provided cryptographic randomness. Decouples Ring 0 from `rand_core` as a public API surface while remaining compatible via blanket impl.

```rust
pub trait EntropySource {
    /// Fill `buf` with cryptographically secure random bytes.
    fn fill_bytes(&mut self, buf: &mut [u8]);
}

// Blanket impl: anything implementing CryptoRngCore also
// implements EntropySource, so existing callers work unchanged.
impl<T: rand_core::CryptoRngCore> EntropySource for T {
    fn fill_bytes(&mut self, buf: &mut [u8]) {
        rand_core::RngCore::fill_bytes(self, buf);
    }
}
```

**Design rationale:** The current pattern passes `&mut impl CryptoRngCore` into each call that needs randomness (Node::announce, Link::initiate, Identity::encrypt). A wrapper trait lets bare-metal Ring 1 implement `EntropySource` directly against hardware RNG without pulling in `getrandom`. The blanket impl means existing code passing `&mut OsRng` continues to work.

**Note:** harmony-platform depends on `rand_core` (without `getrandom`) for the blanket impl. This is trait-only — no entropy source is included.

## Trait 2: NetworkInterface

Platform-provided network I/O for the event loop. Bridges real I/O to the sans-I/O event/action model.

```rust
pub trait NetworkInterface {
    /// Human-readable interface name (e.g., "eth0", "lora0").
    fn name(&self) -> &str;

    /// Maximum transmission unit in bytes.
    fn mtu(&self) -> usize;

    /// Send raw bytes on this interface.
    fn send(&mut self, data: &[u8]) -> Result<(), PlatformError>;

    /// Poll for one inbound packet. Returns None if no data available.
    fn receive(&mut self) -> Option<Vec<u8>>;
}
```

**Design rationale:** The existing `Interface` trait in harmony-reticulum has Reticulum-specific metadata (mode, bitrate, stats, direction). `NetworkInterface` is the platform bridge — focused on byte I/O. The event loop uses `NetworkInterface::receive()` to get raw bytes, feeds them as `InboundPacket` events to the state machine, and uses `NetworkInterface::send()` to actuate `SendOnInterface` actions.

**Relationship to existing `Interface` trait:** The existing trait stays in harmony-reticulum for protocol-level metadata. `NetworkInterface` is what Ring 1 platform code implements. An event loop can hold a type that implements both.

## Trait 3: PersistentState

Platform-provided key-value persistence for node state. Callers serialize their own state to bytes.

```rust
pub trait PersistentState {
    /// Save a byte blob under the given key, overwriting any prior value.
    fn save(&mut self, key: &str, data: &[u8]) -> Result<(), PlatformError>;

    /// Load a previously saved blob. Returns None if the key doesn't exist.
    fn load(&self, key: &str) -> Option<Vec<u8>>;

    /// Delete a key and its data. No-op if key doesn't exist.
    fn delete(&mut self, key: &str) -> Result<(), PlatformError>;

    /// List all stored keys.
    fn keys(&self) -> Vec<&str>;
}
```

**Design rationale:** Key-value bytes is the minimal abstraction that works across flash, files, databases, and in-memory storage. Each caller (identity, path table, node config) serializes its own state — no serde dependency at the trait boundary.

**Planned key conventions:**
- `identity` — 64-byte PrivateIdentity (Ed25519 + X25519 private keys)
- `path_table` — serialized path entries (format TBD by harmony-reticulum)
- `node_config` — node configuration blob
- `announce_cache` — recent announce data for faster restart

## Error Type

```rust
#[derive(Debug)]
pub enum PlatformError {
    /// Network send failed (interface down, buffer full, etc.)
    SendFailed,
    /// Persistence operation failed (flash full, I/O error, etc.)
    StorageFailed,
}
```

Deliberately simple. Platform implementors log detailed errors internally — the trait boundary signals success/failure. Implements `core::fmt::Display` and (with `std` feature) `std::error::Error`.

## In-Memory Test Implementations

The crate provides `MemoryPersistentState` — a `HashMap<String, Vec<u8>>` for use in tests. `EntropySource` gets its test impl from the blanket impl over `rand::rngs::StdRng` or `OsRng`.

No `MemoryNetworkInterface` in this crate — `LoopbackInterface` in harmony-reticulum already serves that role for packet-level testing.

## What This Bead Does NOT Do

- **Does not refactor callers** — Existing code continues to use `CryptoRngCore` directly. Migration to `EntropySource` is a follow-up.
- **Does not add serialization** — Path table serialization is a separate task.
- **Does not wire traits into NodeRuntime** — The runtime integration is Ring 1 work.
- **Does not change BlobStore** — The existing trait in harmony-content is already a good platform abstraction.
