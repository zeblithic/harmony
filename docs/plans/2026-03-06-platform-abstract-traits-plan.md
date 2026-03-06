# Platform-Abstract Traits Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Create the `harmony-platform` crate with three platform-abstract traits (EntropySource, NetworkInterface, PersistentState) that decouple Ring 0 from platform-specific I/O.

**Architecture:** New `no_std`-compatible crate at the bottom of the dependency graph. Contains only traits, a simple error type, a blanket impl bridging `EntropySource` to `rand_core::CryptoRngCore`, and one in-memory test implementation (`MemoryPersistentState`).

**Tech Stack:** Rust, `no_std` + `alloc`, `rand_core` (traits only), `thiserror`, `hashbrown` (for no_std HashMap in MemoryPersistentState)

**Design doc:** `docs/plans/2026-03-06-platform-abstract-traits-design.md`

---

### Task 1: Scaffold the crate and workspace integration

**Files:**
- Modify: `Cargo.toml` (workspace root, lines 3-12 members list, line 50+ workspace deps)
- Create: `crates/harmony-platform/Cargo.toml`
- Create: `crates/harmony-platform/src/lib.rs`

**Step 1: Create the crate directory**

```bash
mkdir -p crates/harmony-platform/src
```

**Step 2: Write `crates/harmony-platform/Cargo.toml`**

```toml
[package]
name = "harmony-platform"
description = "Platform-abstract traits for the Harmony decentralized network stack"
version.workspace = true
edition.workspace = true
license.workspace = true
repository.workspace = true
rust-version.workspace = true

[dependencies]
rand_core = { workspace = true }
thiserror = { workspace = true }
hashbrown = { workspace = true }

[features]
default = ["std"]
std = ["rand_core/getrandom"]

[dev-dependencies]
rand = { workspace = true }
```

**Step 3: Write `crates/harmony-platform/src/lib.rs`** (minimal, just the no_std preamble)

```rust
#![cfg_attr(not(feature = "std"), no_std)]
extern crate alloc;

pub mod error;
```

**Step 4: Create a placeholder error module** at `crates/harmony-platform/src/error.rs`

```rust
/// Errors from platform operations.
#[derive(Debug, thiserror::Error)]
pub enum PlatformError {
    #[error("network send failed")]
    SendFailed,

    #[error("persistent storage operation failed")]
    StorageFailed,
}
```

**Step 5: Add to workspace `Cargo.toml`**

Add `"crates/harmony-platform"` to `[workspace.members]` (after `harmony-node`).

Add to `[workspace.dependencies]` (after `harmony-identity`):
```toml
harmony-platform = { path = "crates/harmony-platform" }
```

**Step 6: Verify it compiles**

```bash
cargo check -p harmony-platform
cargo check -p harmony-platform --no-default-features
```

Expected: both pass with no errors.

**Step 7: Commit**

```bash
git add crates/harmony-platform/ Cargo.toml
git commit -m "feat(platform): scaffold harmony-platform crate (harmony-2yl)"
```

---

### Task 2: EntropySource trait with blanket impl and tests

**Files:**
- Create: `crates/harmony-platform/src/entropy.rs`
- Modify: `crates/harmony-platform/src/lib.rs`

**Step 1: Write the failing test** in `crates/harmony-platform/src/entropy.rs`

```rust
/// Platform-provided cryptographic randomness.
///
/// Ring 0 state machines accept `&mut impl EntropySource` wherever they need
/// randomness. Ring 1+ implements this against hardware RNG, OS getrandom,
/// or a seeded CSPRNG.
///
/// A blanket implementation is provided for all types implementing
/// [`rand_core::CryptoRngCore`], so existing callers passing `&mut OsRng`
/// or `&mut StdRng` work unchanged.
pub trait EntropySource {
    /// Fill `buf` with cryptographically secure random bytes.
    fn fill_bytes(&mut self, buf: &mut [u8]);
}

impl<T: rand_core::CryptoRngCore> EntropySource for T {
    fn fill_bytes(&mut self, buf: &mut [u8]) {
        rand_core::RngCore::fill_bytes(self, buf);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blanket_impl_works_with_std_rng() {
        use rand::SeedableRng;
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let mut buf = [0u8; 32];
        EntropySource::fill_bytes(&mut rng, &mut buf);
        // Seeded RNG should produce non-zero output.
        assert_ne!(buf, [0u8; 32]);
    }

    #[test]
    fn deterministic_with_same_seed() {
        use rand::SeedableRng;
        let mut rng1 = rand::rngs::StdRng::seed_from_u64(99);
        let mut rng2 = rand::rngs::StdRng::seed_from_u64(99);
        let mut buf1 = [0u8; 16];
        let mut buf2 = [0u8; 16];
        EntropySource::fill_bytes(&mut rng1, &mut buf1);
        EntropySource::fill_bytes(&mut rng2, &mut buf2);
        assert_eq!(buf1, buf2);
    }

    #[test]
    fn accepts_entropy_source_trait_object() {
        use rand::SeedableRng;
        let mut rng = rand::rngs::StdRng::seed_from_u64(7);
        let source: &mut dyn EntropySource = &mut rng;
        let mut buf = [0u8; 8];
        source.fill_bytes(&mut buf);
        assert_ne!(buf, [0u8; 8]);
    }
}
```

**Step 2: Add the module to `lib.rs`**

Add `pub mod entropy;` and re-export:
```rust
pub mod entropy;

pub use entropy::EntropySource;
```

**Step 3: Run tests**

```bash
cargo test -p harmony-platform
```

Expected: 3 tests pass.

**Step 4: Verify no_std compilation**

```bash
cargo check -p harmony-platform --no-default-features
```

Expected: pass.

**Step 5: Commit**

```bash
git add crates/harmony-platform/src/
git commit -m "feat(platform): add EntropySource trait with CryptoRngCore blanket impl"
```

---

### Task 3: NetworkInterface trait and tests

**Files:**
- Create: `crates/harmony-platform/src/network.rs`
- Modify: `crates/harmony-platform/src/lib.rs`

**Step 1: Write the trait and tests** in `crates/harmony-platform/src/network.rs`

```rust
use alloc::vec::Vec;

use crate::error::PlatformError;

/// Platform-provided network I/O for the event loop.
///
/// The event loop polls [`NetworkInterface::receive`] for inbound bytes and
/// calls [`NetworkInterface::send`] to actuate `SendOnInterface` actions from
/// the sans-I/O state machines. This bridges real I/O to the event/action model.
///
/// The existing `Interface` trait in `harmony-reticulum` carries
/// protocol-specific metadata (mode, bitrate, stats). `NetworkInterface`
/// is the platform bridge — focused purely on byte I/O.
pub trait NetworkInterface {
    /// Human-readable interface name (e.g., `"eth0"`, `"lora0"`).
    fn name(&self) -> &str;

    /// Maximum transmission unit in bytes.
    fn mtu(&self) -> usize;

    /// Send raw bytes on this interface.
    fn send(&mut self, data: &[u8]) -> Result<(), PlatformError>;

    /// Poll for one inbound packet. Returns `None` if no data is available.
    fn receive(&mut self) -> Option<Vec<u8>>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::collections::VecDeque;
    use alloc::string::String;

    /// Minimal loopback for testing the trait contract.
    struct TestLoopback {
        name: String,
        mtu: usize,
        inbox: VecDeque<Vec<u8>>,
        sent: Vec<Vec<u8>>,
    }

    impl TestLoopback {
        fn new(name: &str, mtu: usize) -> Self {
            TestLoopback {
                name: name.into(),
                mtu,
                inbox: VecDeque::new(),
                sent: Vec::new(),
            }
        }

        fn inject(&mut self, data: &[u8]) {
            self.inbox.push_back(data.to_vec());
        }
    }

    impl NetworkInterface for TestLoopback {
        fn name(&self) -> &str {
            &self.name
        }

        fn mtu(&self) -> usize {
            self.mtu
        }

        fn send(&mut self, data: &[u8]) -> Result<(), PlatformError> {
            if data.len() > self.mtu {
                return Err(PlatformError::SendFailed);
            }
            self.sent.push(data.to_vec());
            Ok(())
        }

        fn receive(&mut self) -> Option<Vec<u8>> {
            self.inbox.pop_front()
        }
    }

    #[test]
    fn send_and_receive_round_trip() {
        let mut iface = TestLoopback::new("lo0", 500);
        iface.inject(b"hello mesh");
        let received = iface.receive().unwrap();
        assert_eq!(received, b"hello mesh");
        assert!(iface.receive().is_none());
    }

    #[test]
    fn send_records_outbound_data() {
        let mut iface = TestLoopback::new("eth0", 1500);
        iface.send(b"packet-one").unwrap();
        iface.send(b"packet-two").unwrap();
        assert_eq!(iface.sent.len(), 2);
        assert_eq!(iface.sent[0], b"packet-one");
    }

    #[test]
    fn send_exceeding_mtu_fails() {
        let mut iface = TestLoopback::new("lora0", 4);
        let result = iface.send(b"too-long");
        assert!(result.is_err());
    }

    #[test]
    fn name_and_mtu_accessors() {
        let iface = TestLoopback::new("wlan0", 1400);
        assert_eq!(iface.name(), "wlan0");
        assert_eq!(iface.mtu(), 1400);
    }

    #[test]
    fn works_as_trait_object() {
        let mut iface = TestLoopback::new("dyn0", 500);
        let net: &mut dyn NetworkInterface = &mut iface;
        assert_eq!(net.name(), "dyn0");
        net.send(b"via-dyn").unwrap();
    }
}
```

**Step 2: Add the module to `lib.rs`**

Add `pub mod network;` and re-export:
```rust
pub mod network;

pub use network::NetworkInterface;
```

**Step 3: Run tests**

```bash
cargo test -p harmony-platform
```

Expected: 8 tests pass (3 entropy + 5 network).

**Step 4: Verify no_std**

```bash
cargo check -p harmony-platform --no-default-features
```

**Step 5: Commit**

```bash
git add crates/harmony-platform/src/
git commit -m "feat(platform): add NetworkInterface trait"
```

---

### Task 4: PersistentState trait, MemoryPersistentState, and tests

**Files:**
- Create: `crates/harmony-platform/src/persistence.rs`
- Modify: `crates/harmony-platform/src/lib.rs`

**Step 1: Write the trait, in-memory impl, and tests** in `crates/harmony-platform/src/persistence.rs`

```rust
use alloc::string::String;
use alloc::vec::Vec;
#[cfg(feature = "std")]
use std::collections::HashMap;
#[cfg(not(feature = "std"))]
use hashbrown::HashMap;

use crate::error::PlatformError;

/// Platform-provided key-value persistence for node state.
///
/// Callers serialize their own state to bytes. Implementations may
/// back this with flash, files, a database, or in-memory (volatile).
pub trait PersistentState {
    /// Save a byte blob under the given key, overwriting any prior value.
    fn save(&mut self, key: &str, data: &[u8]) -> Result<(), PlatformError>;

    /// Load a previously saved blob. Returns `None` if the key doesn't exist.
    fn load(&self, key: &str) -> Option<Vec<u8>>;

    /// Delete a key and its data. No-op if key doesn't exist.
    fn delete(&mut self, key: &str) -> Result<(), PlatformError>;

    /// List all stored keys.
    fn keys(&self) -> Vec<String>;
}

/// In-memory volatile implementation of [`PersistentState`].
///
/// Useful for tests and for nodes that don't need persistence across
/// restarts (e.g., ephemeral unikernel appliances).
pub struct MemoryPersistentState {
    data: HashMap<String, Vec<u8>>,
}

impl MemoryPersistentState {
    pub fn new() -> Self {
        MemoryPersistentState {
            data: HashMap::new(),
        }
    }

    /// Number of stored entries.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Whether the store is empty.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

impl Default for MemoryPersistentState {
    fn default() -> Self {
        Self::new()
    }
}

impl PersistentState for MemoryPersistentState {
    fn save(&mut self, key: &str, data: &[u8]) -> Result<(), PlatformError> {
        self.data.insert(key.into(), data.to_vec());
        Ok(())
    }

    fn load(&self, key: &str) -> Option<Vec<u8>> {
        self.data.get(key).cloned()
    }

    fn delete(&mut self, key: &str) -> Result<(), PlatformError> {
        self.data.remove(key);
        Ok(())
    }

    fn keys(&self) -> Vec<String> {
        self.data.keys().cloned().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn save_and_load() {
        let mut store = MemoryPersistentState::new();
        store.save("identity", b"my-64-byte-key-material").unwrap();
        let loaded = store.load("identity").unwrap();
        assert_eq!(loaded, b"my-64-byte-key-material");
    }

    #[test]
    fn load_missing_key_returns_none() {
        let store = MemoryPersistentState::new();
        assert!(store.load("nonexistent").is_none());
    }

    #[test]
    fn save_overwrites_existing() {
        let mut store = MemoryPersistentState::new();
        store.save("config", b"v1").unwrap();
        store.save("config", b"v2").unwrap();
        assert_eq!(store.load("config").unwrap(), b"v2");
        assert_eq!(store.len(), 1);
    }

    #[test]
    fn delete_removes_key() {
        let mut store = MemoryPersistentState::new();
        store.save("temp", b"data").unwrap();
        assert_eq!(store.len(), 1);
        store.delete("temp").unwrap();
        assert!(store.load("temp").is_none());
        assert!(store.is_empty());
    }

    #[test]
    fn delete_missing_key_is_noop() {
        let mut store = MemoryPersistentState::new();
        store.delete("ghost").unwrap(); // should not panic or error
    }

    #[test]
    fn keys_lists_all_stored() {
        let mut store = MemoryPersistentState::new();
        store.save("identity", b"key").unwrap();
        store.save("path_table", b"paths").unwrap();
        store.save("node_config", b"cfg").unwrap();
        let mut keys = store.keys();
        keys.sort();
        assert_eq!(keys, vec!["identity", "node_config", "path_table"]);
    }

    #[test]
    fn default_is_empty() {
        let store = MemoryPersistentState::default();
        assert!(store.is_empty());
    }

    #[test]
    fn works_as_trait_object() {
        let mut store = MemoryPersistentState::new();
        let ps: &mut dyn PersistentState = &mut store;
        ps.save("via-dyn", b"dynamic dispatch").unwrap();
        assert_eq!(ps.load("via-dyn").unwrap(), b"dynamic dispatch");
    }
}
```

**Step 2: Add module to `lib.rs`**

Add `pub mod persistence;` and re-exports:
```rust
pub mod persistence;

pub use persistence::{MemoryPersistentState, PersistentState};
```

**Step 3: Run tests**

```bash
cargo test -p harmony-platform
```

Expected: 16 tests pass (3 entropy + 5 network + 8 persistence).

**Step 4: Verify no_std**

```bash
cargo check -p harmony-platform --no-default-features
```

**Step 5: Commit**

```bash
git add crates/harmony-platform/src/
git commit -m "feat(platform): add PersistentState trait with MemoryPersistentState"
```

---

### Task 5: Error Display test and final verification

**Files:**
- Modify: `crates/harmony-platform/src/error.rs` (add tests)

**Step 1: Add error display tests** to `crates/harmony-platform/src/error.rs`

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use alloc::format;

    #[test]
    fn display_send_failed() {
        let err = PlatformError::SendFailed;
        assert_eq!(format!("{err}"), "network send failed");
    }

    #[test]
    fn display_storage_failed() {
        let err = PlatformError::StorageFailed;
        assert_eq!(format!("{err}"), "persistent storage operation failed");
    }

    #[test]
    fn debug_impl_exists() {
        let err = PlatformError::SendFailed;
        let dbg = format!("{err:?}");
        assert!(dbg.contains("SendFailed"));
    }
}
```

**Step 2: Run full test suite**

```bash
cargo test -p harmony-platform
```

Expected: 19 tests pass.

**Step 3: Run clippy**

```bash
cargo clippy -p harmony-platform
```

Expected: zero warnings.

**Step 4: Verify full workspace still compiles and tests pass**

```bash
cargo test --workspace
cargo clippy --workspace
```

Expected: all existing tests pass, zero warnings.

**Step 5: Verify no_std for the new crate**

```bash
cargo check -p harmony-platform --no-default-features
```

**Step 6: Commit**

```bash
git add crates/harmony-platform/src/error.rs
git commit -m "test(platform): add error display tests, verify full suite"
```

---

## File Summary

| Action | Path |
|--------|------|
| Modify | `Cargo.toml` (workspace root) |
| Create | `crates/harmony-platform/Cargo.toml` |
| Create | `crates/harmony-platform/src/lib.rs` |
| Create | `crates/harmony-platform/src/error.rs` |
| Create | `crates/harmony-platform/src/entropy.rs` |
| Create | `crates/harmony-platform/src/network.rs` |
| Create | `crates/harmony-platform/src/persistence.rs` |

**7 files total** (1 modified, 6 created). ~300 lines of Rust.
