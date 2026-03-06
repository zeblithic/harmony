# Ring 1 Boot Stub Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Create a minimal bootable x86_64 image that boots in QEMU, generates a Harmony identity, and enters the unikernel event loop with serial output.

**Architecture:** Two new crates in harmony-os: `harmony-boot` (x86_64 binary, `bootloader_api` entry point) and expanded `harmony-unikernel` (portable lib with platform trait impls, serial writer, event loop). An `xtask` crate builds disk images. The `bootloader` crate handles the boot ceremony; we own everything after the Rust entry point.

**Tech Stack:** `bootloader_api` 0.11, `linked_list_allocator` 0.10, `x86_64` 0.15, `bootloader` 0.11 (xtask only), Ring 0 crates with `default-features = false` for no_std.

**Repos:** Primary work in `harmony-os` (`zeblithic/harmony-os/`). Design doc already committed in `harmony` (`zeblithic/harmony/`).

**Design doc:** `docs/plans/2026-03-05-ring1-boot-stub-design.md`

---

### Task 1: Repository and Workspace Setup

**Context:** Create the task branch in harmony-os, add new workspace dependencies, scaffold the `harmony-boot` and `xtask` crates with minimal compilable stubs.

**Files:**
- Modify: `harmony-os/Cargo.toml`
- Create: `harmony-os/crates/harmony-boot/Cargo.toml`
- Create: `harmony-os/crates/harmony-boot/src/main.rs`
- Create: `harmony-os/xtask/Cargo.toml`
- Create: `harmony-os/xtask/src/main.rs`

**Step 1: Create task branch in harmony-os**

```bash
cd harmony-os  # relative to workspace root
git checkout main && git pull origin main
git checkout -b jake-os-boot-stub
```

**Step 2: Update workspace Cargo.toml**

Add `harmony-boot` to workspace members, add new workspace dependencies:

```toml
[workspace]
resolver = "2"
members = [
    "crates/harmony-unikernel",
    "crates/harmony-microkernel",
    "crates/harmony-os",
    "crates/harmony-boot",
]

[workspace.package]
version = "0.1.0"
edition = "2021"
license = "GPL-2.0-or-later"
repository = "https://github.com/zeblithic/harmony-os"
rust-version = "1.75"

[workspace.dependencies]
# Ring 0 — harmony core (consumed under MIT license arm)
harmony-crypto = { git = "https://github.com/zeblithic/harmony.git" }
harmony-identity = { git = "https://github.com/zeblithic/harmony.git" }
harmony-platform = { git = "https://github.com/zeblithic/harmony.git" }
harmony-reticulum = { git = "https://github.com/zeblithic/harmony.git" }
harmony-zenoh = { git = "https://github.com/zeblithic/harmony.git" }
harmony-content = { git = "https://github.com/zeblithic/harmony.git" }
harmony-compute = { git = "https://github.com/zeblithic/harmony.git" }
harmony-workflow = { git = "https://github.com/zeblithic/harmony.git" }
# Note: harmony-node is a binary crate (no lib target), not usable as a dependency

# Internal workspace crates
harmony-unikernel = { path = "crates/harmony-unikernel" }
harmony-microkernel = { path = "crates/harmony-microkernel" }
harmony-os = { path = "crates/harmony-os" }

# Shared dependencies
rand_core = { version = "0.6", default-features = false }
```

**Step 3: Create harmony-boot crate scaffold**

`crates/harmony-boot/Cargo.toml`:
```toml
[package]
name = "harmony-boot"
description = "Ring 1: x86_64 boot stub for Harmony unikernel (QEMU target)"
version.workspace = true
edition.workspace = true
license.workspace = true
repository.workspace = true

[dependencies]
harmony-unikernel = { workspace = true, default-features = false }
bootloader_api = "0.11"
linked_list_allocator = "0.10"
x86_64 = "0.15"
```

`crates/harmony-boot/src/main.rs` (minimal compilable stub):
```rust
// SPDX-License-Identifier: GPL-2.0-or-later
#![no_std]
#![no_main]

use bootloader_api::{entry_point, BootInfo};
use core::panic::PanicInfo;

entry_point!(kernel_main);

fn kernel_main(_boot_info: &'static mut BootInfo) -> ! {
    loop {
        x86_64::instructions::hlt();
    }
}

#[panic_handler]
fn panic(_info: &PanicInfo) -> ! {
    loop {
        x86_64::instructions::hlt();
    }
}
```

**Step 4: Create xtask crate (outside workspace)**

`xtask/Cargo.toml`:
```toml
[package]
name = "xtask"
version = "0.1.0"
edition = "2021"
publish = false

[dependencies]
bootloader = "0.11"
```

`xtask/src/main.rs` (stub):
```rust
use std::path::PathBuf;
use std::process::Command;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(|s| s.as_str()) {
        Some("build-image") => build_image(),
        Some("run") => { build_image(); run_qemu(); }
        _ => eprintln!("Usage: cargo xtask [build-image|run]"),
    }
}

fn project_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).parent().unwrap().to_path_buf()
}

fn kernel_binary() -> PathBuf {
    project_root().join("target/x86_64-unknown-none/release/harmony-boot")
}

fn image_path() -> PathBuf {
    project_root().join("target/harmony-boot-bios.img")
}

fn build_image() {
    // Build kernel ELF
    let status = Command::new("cargo")
        .args(["build", "-p", "harmony-boot", "--target", "x86_64-unknown-none", "--release"])
        .current_dir(project_root())
        .status()
        .expect("failed to build kernel");
    assert!(status.success(), "kernel build failed");

    // Create BIOS disk image
    bootloader::BiosBoot::new(&kernel_binary())
        .create_disk_image(&image_path())
        .expect("failed to create BIOS disk image");
    println!("Created: {}", image_path().display());
}

fn run_qemu() {
    let status = Command::new("qemu-system-x86_64")
        .args([
            "-drive", &format!("format=raw,file={}", image_path().display()),
            "-serial", "stdio",
            "-display", "none",
            "-device", "isa-debug-exit,iobase=0xf4,iosize=0x04",
        ])
        .status()
        .expect("failed to launch QEMU");
    std::process::exit(status.code().unwrap_or(1));
}
```

**Step 5: Verify workspace compiles**

```bash
cd harmony-os  # relative to workspace root
cargo check -p harmony-unikernel
cargo check -p harmony-boot --target x86_64-unknown-none
```

Expected: both pass (stubs compile). Note: `harmony-boot` must be checked with the bare-metal target.

**Step 6: Commit**

```bash
git add -A
git commit -m "scaffold: harmony-boot crate and xtask for Ring 1 boot stub"
```

---

### Task 2: harmony-unikernel no_std and Module Structure

**Context:** Convert harmony-unikernel from a placeholder to a proper no_std lib with feature flags and module structure for the portable kernel logic.

**Files:**
- Modify: `harmony-os/crates/harmony-unikernel/Cargo.toml`
- Modify: `harmony-os/crates/harmony-unikernel/src/lib.rs`
- Create: `harmony-os/crates/harmony-unikernel/src/serial.rs`
- Create: `harmony-os/crates/harmony-unikernel/src/platform/mod.rs`
- Create: `harmony-os/crates/harmony-unikernel/src/platform/entropy.rs`
- Create: `harmony-os/crates/harmony-unikernel/src/platform/persistence.rs`
- Create: `harmony-os/crates/harmony-unikernel/src/event_loop.rs`

**Step 1: Update Cargo.toml with no_std deps and features**

```toml
[package]
name = "harmony-unikernel"
description = "Ring 1: Bootable single-purpose Harmony node (bare metal / hypervisor)"
version.workspace = true
edition.workspace = true
license.workspace = true
repository.workspace = true
rust-version.workspace = true

[dependencies]
# Ring 0 — harmony protocol core (no_std when std feature disabled)
harmony-crypto = { workspace = true, default-features = false }
harmony-identity = { workspace = true, default-features = false }
harmony-platform = { workspace = true, default-features = false }
harmony-reticulum = { workspace = true, default-features = false }
rand_core = { workspace = true }

[features]
default = ["std"]
std = [
    "harmony-crypto/std",
    "harmony-identity/std",
    "harmony-platform/std",
    "harmony-reticulum/std",
]

[dev-dependencies]
```

**Step 2: Rewrite lib.rs with module structure**

```rust
// SPDX-License-Identifier: GPL-2.0-or-later
#![cfg_attr(not(feature = "std"), no_std)]
extern crate alloc;

pub mod event_loop;
pub mod platform;
pub mod serial;

pub use event_loop::UnikernelRuntime;
pub use platform::entropy::KernelEntropy;
pub use platform::persistence::MemoryState;
pub use serial::SerialWriter;
```

**Step 3: Create empty module files**

`src/serial.rs`:
```rust
// SPDX-License-Identifier: GPL-2.0-or-later
// Arch-agnostic structured serial output.
```

`src/platform/mod.rs`:
```rust
// SPDX-License-Identifier: GPL-2.0-or-later
pub mod entropy;
pub mod persistence;
```

`src/platform/entropy.rs`:
```rust
// SPDX-License-Identifier: GPL-2.0-or-later
// Arch-agnostic EntropySource wrapper.
```

`src/platform/persistence.rs`:
```rust
// SPDX-License-Identifier: GPL-2.0-or-later
// In-memory PersistentState implementation.
```

`src/event_loop.rs`:
```rust
// SPDX-License-Identifier: GPL-2.0-or-later
// Unikernel event loop driving Ring 0 state machines.
```

**Step 4: Verify compiles in both modes**

```bash
cargo check -p harmony-unikernel                          # std (default)
cargo check -p harmony-unikernel --no-default-features     # no_std
```

Expected: both pass (empty modules).

**Step 5: Commit**

```bash
git add -A
git commit -m "feat(unikernel): no_std module structure for platform, serial, event_loop"
```

---

### Task 3: SerialWriter (TDD)

**Context:** Arch-agnostic serial output writer that takes a closure for byte output. Implements `core::fmt::Write`. Provides structured `[TAG] message` logging.

**Files:**
- Modify: `harmony-os/crates/harmony-unikernel/src/serial.rs`

**Step 1: Write failing tests**

```rust
// SPDX-License-Identifier: GPL-2.0-or-later

use core::fmt::Write;

use alloc::vec::Vec;
use alloc::string::String;

pub struct SerialWriter<F: FnMut(u8)> {
    write_byte: F,
}

impl<F: FnMut(u8)> SerialWriter<F> {
    pub fn new(write_byte: F) -> Self {
        SerialWriter { write_byte }
    }

    pub fn log(&mut self, tag: &str, msg: &str) {
        let _ = write!(self, "[{}] {}\n", tag, msg);
    }
}

impl<F: FnMut(u8)> Write for SerialWriter<F> {
    fn write_str(&mut self, s: &str) -> core::fmt::Result {
        for byte in s.bytes() {
            (self.write_byte)(byte);
        }
        Ok(())
    }
}

/// Format 16 bytes as 32-char lowercase hex string.
pub fn hex_encode(bytes: &[u8], buf: &mut [u8]) {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    for (i, &b) in bytes.iter().enumerate() {
        buf[i * 2] = HEX[(b >> 4) as usize];
        buf[i * 2 + 1] = HEX[(b & 0xf) as usize];
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::sync::Arc;
    use core::cell::RefCell;

    fn capture_writer() -> (SerialWriter<impl FnMut(u8)>, Arc<RefCell<Vec<u8>>>) {
        let buf = Arc::new(RefCell::new(Vec::new()));
        let buf_clone = buf.clone();
        let writer = SerialWriter::new(move |byte| {
            buf_clone.borrow_mut().push(byte);
        });
        (writer, buf)
    }

    #[test]
    fn write_str_captures_bytes() {
        let (mut writer, buf) = capture_writer();
        write!(writer, "hello").unwrap();
        assert_eq!(&*buf.borrow(), b"hello");
    }

    #[test]
    fn log_formats_structured_tag() {
        let (mut writer, buf) = capture_writer();
        writer.log("BOOT", "Harmony unikernel v0.1.0");
        let output = String::from_utf8(buf.borrow().clone()).unwrap();
        assert_eq!(output, "[BOOT] Harmony unikernel v0.1.0\n");
    }

    #[test]
    fn log_identity_format() {
        let (mut writer, buf) = capture_writer();
        writer.log("IDENTITY", "deadbeef01234567deadbeef01234567");
        let output = String::from_utf8(buf.borrow().clone()).unwrap();
        assert!(output.starts_with("[IDENTITY] "));
        assert!(output.ends_with("\n"));
    }

    #[test]
    fn hex_encode_produces_lowercase_hex() {
        let bytes = [0xde, 0xad, 0xbe, 0xef];
        let mut buf = [0u8; 8];
        hex_encode(&bytes, &mut buf);
        assert_eq!(&buf, b"deadbeef");
    }

    #[test]
    fn hex_encode_16_bytes_produces_32_chars() {
        let bytes = [0x01, 0x23, 0x45, 0x67, 0x89, 0xab, 0xcd, 0xef,
                     0xfe, 0xdc, 0xba, 0x98, 0x76, 0x54, 0x32, 0x10];
        let mut buf = [0u8; 32];
        hex_encode(&bytes, &mut buf);
        assert_eq!(&buf, b"0123456789abcdeffedcba9876543210");
    }
}
```

**Step 2: Run tests to verify they pass**

Since this is TDD but the implementation is included with the tests (it's small and self-contained):

```bash
cargo test -p harmony-unikernel -- serial
```

Expected: all 5 tests pass.

**Step 3: Commit**

```bash
git add -A
git commit -m "feat(unikernel): SerialWriter with structured [TAG] logging and hex_encode"
```

---

### Task 4: KernelEntropy (TDD)

**Context:** Arch-agnostic entropy wrapper implementing `rand_core::RngCore + CryptoRng` (which gives `CryptoRngCore` and thus `EntropySource` via blanket impl). Takes a caller-provided `FnMut(&mut [u8])` closure — `harmony-boot` will pass RDRAND.

**Files:**
- Modify: `harmony-os/crates/harmony-unikernel/src/platform/entropy.rs`

**Step 1: Write implementation and tests**

```rust
// SPDX-License-Identifier: GPL-2.0-or-later

use rand_core::{CryptoRng, RngCore};

/// Arch-agnostic entropy source wrapping a platform-provided fill function.
///
/// The fill function is injected by the boot crate:
/// - x86_64: RDRAND instruction
/// - aarch64: MRS RNDR (future)
/// - test: deterministic seed
pub struct KernelEntropy<F: FnMut(&mut [u8])> {
    fill_fn: F,
}

impl<F: FnMut(&mut [u8])> KernelEntropy<F> {
    pub fn new(fill_fn: F) -> Self {
        KernelEntropy { fill_fn }
    }
}

impl<F: FnMut(&mut [u8])> RngCore for KernelEntropy<F> {
    fn next_u32(&mut self) -> u32 {
        let mut buf = [0u8; 4];
        self.fill_bytes(&mut buf);
        u32::from_le_bytes(buf)
    }

    fn next_u64(&mut self) -> u64 {
        let mut buf = [0u8; 8];
        self.fill_bytes(&mut buf);
        u64::from_le_bytes(buf)
    }

    fn fill_bytes(&mut self, dest: &mut [u8]) {
        (self.fill_fn)(dest);
    }

    fn try_fill_bytes(&mut self, dest: &mut [u8]) -> Result<(), rand_core::Error> {
        self.fill_bytes(dest);
        Ok(())
    }
}

// SAFETY: The caller guarantees the fill function provides cryptographically
// secure randomness (RDRAND, RNDR, etc.). This marker enables use with
// PrivateIdentity::generate() and the EntropySource blanket impl.
impl<F: FnMut(&mut [u8])> CryptoRng for KernelEntropy<F> {}

#[cfg(test)]
mod tests {
    use super::*;
    use harmony_platform::EntropySource;

    /// Deterministic fill for testing: fills with incrementing bytes.
    fn counting_fill() -> impl FnMut(&mut [u8]) {
        let mut counter: u8 = 0;
        move |buf: &mut [u8]| {
            for byte in buf.iter_mut() {
                *byte = counter;
                counter = counter.wrapping_add(1);
            }
        }
    }

    #[test]
    fn fill_bytes_uses_provided_function() {
        let mut entropy = KernelEntropy::new(counting_fill());
        let mut buf = [0u8; 4];
        entropy.fill_bytes(&mut buf);
        assert_eq!(buf, [0, 1, 2, 3]);
    }

    #[test]
    fn next_u32_returns_le_bytes() {
        let mut entropy = KernelEntropy::new(counting_fill());
        let val = entropy.next_u32();
        assert_eq!(val, u32::from_le_bytes([0, 1, 2, 3]));
    }

    #[test]
    fn next_u64_returns_le_bytes() {
        let mut entropy = KernelEntropy::new(counting_fill());
        let val = entropy.next_u64();
        assert_eq!(val, u64::from_le_bytes([0, 1, 2, 3, 4, 5, 6, 7]));
    }

    #[test]
    fn satisfies_entropy_source_trait() {
        let mut entropy = KernelEntropy::new(counting_fill());
        let source: &mut dyn EntropySource = &mut entropy;
        let mut buf = [0u8; 4];
        source.fill_bytes(&mut buf);
        assert_eq!(buf, [0, 1, 2, 3]);
    }

    #[test]
    fn satisfies_crypto_rng_core() {
        let mut entropy = KernelEntropy::new(counting_fill());
        // CryptoRngCore is CryptoRng + RngCore — verify it compiles
        fn takes_crypto_rng(_rng: &mut impl rand_core::CryptoRngCore) {}
        takes_crypto_rng(&mut entropy);
    }

    #[test]
    fn sequential_calls_advance_state() {
        let mut entropy = KernelEntropy::new(counting_fill());
        let mut buf1 = [0u8; 4];
        let mut buf2 = [0u8; 4];
        entropy.fill_bytes(&mut buf1);
        entropy.fill_bytes(&mut buf2);
        assert_eq!(buf1, [0, 1, 2, 3]);
        assert_eq!(buf2, [4, 5, 6, 7]);
    }
}
```

**Step 2: Run tests**

```bash
cargo test -p harmony-unikernel -- platform::entropy
```

Expected: all 6 tests pass.

**Step 3: Commit**

```bash
git add -A
git commit -m "feat(unikernel): KernelEntropy — arch-agnostic RngCore+CryptoRng wrapper"
```

---

### Task 5: MemoryState (TDD)

**Context:** In-memory `BTreeMap`-based `PersistentState` implementation. Uses `alloc::collections::BTreeMap` to avoid `hashbrown` dependency in the unikernel crate.

**Files:**
- Modify: `harmony-os/crates/harmony-unikernel/src/platform/persistence.rs`

**Step 1: Write implementation and tests**

```rust
// SPDX-License-Identifier: GPL-2.0-or-later

use alloc::collections::BTreeMap;
use alloc::string::String;
use alloc::vec::Vec;

use harmony_platform::{PersistentState, PlatformError};

/// In-memory volatile implementation of [`PersistentState`].
///
/// Data is lost on reboot. Suitable for unikernel nodes that regenerate
/// identity on each boot or for testing.
pub struct MemoryState {
    store: BTreeMap<String, Vec<u8>>,
}

impl MemoryState {
    pub fn new() -> Self {
        MemoryState {
            store: BTreeMap::new(),
        }
    }
}

impl Default for MemoryState {
    fn default() -> Self {
        Self::new()
    }
}

impl PersistentState for MemoryState {
    fn save(&mut self, key: &str, data: &[u8]) -> Result<(), PlatformError> {
        self.store.insert(key.into(), data.to_vec());
        Ok(())
    }

    fn load(&self, key: &str) -> Option<Vec<u8>> {
        self.store.get(key).cloned()
    }

    fn delete(&mut self, key: &str) -> Result<(), PlatformError> {
        self.store.remove(key);
        Ok(())
    }

    fn keys(&self) -> Vec<String> {
        self.store.keys().cloned().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn save_and_load_round_trip() {
        let mut state = MemoryState::new();
        state.save("identity", b"key-material-64-bytes").unwrap();
        assert_eq!(state.load("identity").unwrap(), b"key-material-64-bytes");
    }

    #[test]
    fn load_missing_returns_none() {
        let state = MemoryState::new();
        assert!(state.load("nonexistent").is_none());
    }

    #[test]
    fn save_overwrites() {
        let mut state = MemoryState::new();
        state.save("config", b"v1").unwrap();
        state.save("config", b"v2").unwrap();
        assert_eq!(state.load("config").unwrap(), b"v2");
    }

    #[test]
    fn delete_removes_key() {
        let mut state = MemoryState::new();
        state.save("temp", b"data").unwrap();
        state.delete("temp").unwrap();
        assert!(state.load("temp").is_none());
    }

    #[test]
    fn delete_missing_is_noop() {
        let mut state = MemoryState::new();
        state.delete("ghost").unwrap();
    }

    #[test]
    fn keys_lists_all_sorted() {
        let mut state = MemoryState::new();
        state.save("zebra", b"z").unwrap();
        state.save("alpha", b"a").unwrap();
        state.save("middle", b"m").unwrap();
        // BTreeMap keys are sorted
        assert_eq!(state.keys(), vec!["alpha", "middle", "zebra"]);
    }

    #[test]
    fn works_as_trait_object() {
        let mut state = MemoryState::new();
        let ps: &mut dyn PersistentState = &mut state;
        ps.save("via-dyn", b"dynamic").unwrap();
        assert_eq!(ps.load("via-dyn").unwrap(), b"dynamic");
    }
}
```

**Step 2: Run tests**

```bash
cargo test -p harmony-unikernel -- platform::persistence
```

Expected: all 7 tests pass.

**Step 3: Commit**

```bash
git add -A
git commit -m "feat(unikernel): MemoryState — BTreeMap-based volatile PersistentState"
```

---

### Task 6: UnikernelRuntime (TDD)

**Context:** The core event loop struct that wires Ring 0's `Node` state machine with platform trait impls. Uses `Node::new()` (leaf mode) and `handle_event(NodeEvent::TimerTick)`.

**Important API note:** The Node uses `handle_event(NodeEvent) -> Vec<NodeAction>`, not a `tick()` method. The runtime's `tick()` wraps this.

**Files:**
- Modify: `harmony-os/crates/harmony-unikernel/src/event_loop.rs`

**Step 1: Write implementation and tests**

```rust
// SPDX-License-Identifier: GPL-2.0-or-later

use alloc::vec::Vec;

use harmony_identity::PrivateIdentity;
use harmony_platform::{EntropySource, PersistentState};
use harmony_reticulum::{Node, NodeAction, NodeEvent};

/// Unikernel event loop driving Ring 0 state machines.
///
/// Generic over entropy and persistence to allow arch-specific boot
/// crates to inject their platform implementations.
pub struct UnikernelRuntime<E: EntropySource, P: PersistentState> {
    node: Node,
    identity: PrivateIdentity,
    entropy: E,
    persistence: P,
    tick_count: u64,
}

impl<E: EntropySource, P: PersistentState> UnikernelRuntime<E, P> {
    pub fn new(identity: PrivateIdentity, entropy: E, persistence: P) -> Self {
        let node = Node::new();
        UnikernelRuntime {
            node,
            identity,
            entropy,
            persistence,
            tick_count: 0,
        }
    }

    /// One iteration of the event loop.
    ///
    /// Sends a timer tick to the Node and processes resulting actions.
    /// Returns the list of actions produced (empty when no interfaces are registered).
    pub fn tick(&mut self, now: u64) -> Vec<NodeAction> {
        self.tick_count += 1;
        let actions = self.node.handle_event(NodeEvent::TimerTick { now });
        // Future: execute SendOnInterface actions via NetworkInterface impls
        actions
    }

    pub fn tick_count(&self) -> u64 {
        self.tick_count
    }

    pub fn identity(&self) -> &PrivateIdentity {
        &self.identity
    }

    pub fn entropy(&mut self) -> &mut E {
        &mut self.entropy
    }

    pub fn persistence(&mut self) -> &mut P {
        &mut self.persistence
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::platform::entropy::KernelEntropy;
    use crate::platform::persistence::MemoryState;

    fn test_entropy() -> KernelEntropy<impl FnMut(&mut [u8])> {
        // Deterministic seed for reproducible tests
        let mut counter: u8 = 42;
        KernelEntropy::new(move |buf: &mut [u8]| {
            for byte in buf.iter_mut() {
                *byte = counter;
                counter = counter.wrapping_add(7);
            }
        })
    }

    fn make_runtime() -> UnikernelRuntime<KernelEntropy<impl FnMut(&mut [u8])>, MemoryState> {
        let mut entropy = test_entropy();
        let identity = PrivateIdentity::generate(&mut entropy);
        let persistence = MemoryState::new();
        UnikernelRuntime::new(identity, entropy, persistence)
    }

    #[test]
    fn runtime_initializes() {
        let runtime = make_runtime();
        assert_eq!(runtime.tick_count(), 0);
    }

    #[test]
    fn tick_increments_counter() {
        let mut runtime = make_runtime();
        runtime.tick(1000);
        runtime.tick(1001);
        assert_eq!(runtime.tick_count(), 2);
    }

    #[test]
    fn tick_without_interfaces_returns_empty_or_minimal_actions() {
        let mut runtime = make_runtime();
        let actions = runtime.tick(1000);
        // With no registered interfaces, timer tick produces no send actions
        // (may produce PathsExpired or nothing)
        for action in &actions {
            match action {
                NodeAction::SendOnInterface { .. } => panic!("no interfaces, should not send"),
                _ => {} // other actions are fine
            }
        }
    }

    #[test]
    fn identity_is_accessible() {
        let runtime = make_runtime();
        let addr = runtime.identity().public_identity().address_hash;
        // Address hash is 16 bytes, should be non-zero
        assert_ne!(addr, [0u8; 16]);
    }

    #[test]
    fn persistence_is_usable() {
        let mut runtime = make_runtime();
        runtime.persistence().save("test", b"data").unwrap();
        assert_eq!(runtime.persistence().load("test").unwrap(), b"data");
    }
}
```

**Step 2: Run tests**

```bash
cargo test -p harmony-unikernel -- event_loop
```

Expected: all 5 tests pass.

**Step 3: Verify all unikernel tests pass together**

```bash
cargo test -p harmony-unikernel
```

Expected: all tests pass (serial + entropy + persistence + event_loop).

**Step 4: Commit**

```bash
git add -A
git commit -m "feat(unikernel): UnikernelRuntime — event loop driving Ring 0 Node"
```

---

### Task 7: harmony-boot Entry Point

**Context:** The x86_64-specific boot binary. Initializes serial (UART 0x3F8), heap (linked_list_allocator from boot memory map), IDT (exception handlers), RDRAND entropy, generates identity, and enters the event loop. All arch-specific code lives here.

**Files:**
- Modify: `harmony-os/crates/harmony-boot/src/main.rs`

**Step 1: Write the full entry point**

```rust
// SPDX-License-Identifier: GPL-2.0-or-later
#![no_std]
#![no_main]

extern crate alloc;

use bootloader_api::{entry_point, BootInfo, BootloaderConfig};
use bootloader_api::info::MemoryRegionKind;
use core::fmt::Write;
use core::panic::PanicInfo;
use linked_list_allocator::LockedHeap;
use x86_64::instructions::port::Port;

use harmony_unikernel::{KernelEntropy, MemoryState, SerialWriter, UnikernelRuntime};
use harmony_unikernel::serial::hex_encode;
use harmony_identity::PrivateIdentity;

// --- Bootloader configuration ---

pub static BOOTLOADER_CONFIG: BootloaderConfig = {
    let mut config = BootloaderConfig::new_default();
    config.kernel_stack_size = 128 * 1024; // 128 KiB stack
    config
};

entry_point!(kernel_main, config = &BOOTLOADER_CONFIG);

// --- Global heap allocator ---

#[global_allocator]
static ALLOCATOR: LockedHeap = LockedHeap::empty();

const HEAP_SIZE: usize = 1024 * 1024; // 1 MiB

// --- Serial I/O (UART 0x3F8) ---

const SERIAL_PORT: u16 = 0x3F8;

fn init_serial() {
    unsafe {
        // Disable interrupts
        Port::new(SERIAL_PORT + 1).write(0x00u8);
        // Enable DLAB (set baud rate divisor)
        Port::new(SERIAL_PORT + 3).write(0x80u8);
        // Set divisor to 1 (115200 baud)
        Port::new(SERIAL_PORT + 0).write(0x01u8); // low byte
        Port::new(SERIAL_PORT + 1).write(0x00u8); // high byte
        // 8 bits, no parity, one stop bit
        Port::new(SERIAL_PORT + 3).write(0x03u8);
        // Enable FIFO, clear, 14-byte threshold
        Port::new(SERIAL_PORT + 2).write(0xC7u8);
        // IRQs enabled, RTS/DSR set
        Port::new(SERIAL_PORT + 4).write(0x0Bu8);
    }
}

fn serial_write_byte(byte: u8) {
    unsafe {
        // Wait for transmit buffer empty
        while Port::<u8>::new(SERIAL_PORT + 5).read() & 0x20 == 0 {}
        Port::new(SERIAL_PORT).write(byte);
    }
}

// --- IDT (interrupt descriptor table) ---

use x86_64::structures::idt::{InterruptDescriptorTable, InterruptStackFrame};

static mut IDT: InterruptDescriptorTable = InterruptDescriptorTable::new();

fn init_idt() {
    unsafe {
        IDT.breakpoint.set_handler_fn(breakpoint_handler);
        IDT.double_fault.set_handler_fn(double_fault_handler);
        IDT.page_fault.set_handler_fn(page_fault_handler);
        IDT.load();
    }
}

extern "x86-interrupt" fn breakpoint_handler(stack_frame: InterruptStackFrame) {
    let mut serial = SerialWriter::new(serial_write_byte);
    let _ = write!(serial, "[TRAP] breakpoint at {:#x}\n",
        stack_frame.instruction_pointer.as_u64());
}

extern "x86-interrupt" fn double_fault_handler(
    stack_frame: InterruptStackFrame, _error_code: u64,
) -> ! {
    let mut serial = SerialWriter::new(serial_write_byte);
    let _ = write!(serial, "[FATAL] double fault at {:#x}\n",
        stack_frame.instruction_pointer.as_u64());
    loop { x86_64::instructions::hlt(); }
}

extern "x86-interrupt" fn page_fault_handler(
    stack_frame: InterruptStackFrame,
    error_code: x86_64::structures::idt::PageFaultErrorCode,
) {
    let mut serial = SerialWriter::new(serial_write_byte);
    let _ = write!(serial, "[FAULT] page fault at {:#x} (code: {:?})\n",
        stack_frame.instruction_pointer.as_u64(), error_code);
    loop { x86_64::instructions::hlt(); }
}

// --- RDRAND entropy ---

fn rdrand_fill(buf: &mut [u8]) {
    let rdrand = x86_64::instructions::random::RdRand::new()
        .expect("RDRAND not supported");
    // Fill 8 bytes at a time
    let mut i = 0;
    while i + 8 <= buf.len() {
        let val = rdrand.get_u64().expect("RDRAND failed");
        buf[i..i + 8].copy_from_slice(&val.to_le_bytes());
        i += 8;
    }
    // Fill remaining bytes
    if i < buf.len() {
        let val = rdrand.get_u64().expect("RDRAND failed");
        let bytes = val.to_le_bytes();
        buf[i..].copy_from_slice(&bytes[..buf.len() - i]);
    }
}

// --- QEMU debug exit ---

fn qemu_exit(code: u32) {
    unsafe {
        Port::new(0xf4).write(code);
    }
}

// --- Kernel main ---

fn kernel_main(boot_info: &'static mut BootInfo) -> ! {
    // 1. Init serial
    init_serial();
    let mut serial = SerialWriter::new(serial_write_byte);
    serial.log("BOOT", "Harmony unikernel v0.1.0");

    // 2. Init heap from boot memory map
    let heap_region = boot_info.memory_regions.iter()
        .find(|r| r.kind == MemoryRegionKind::Usable && (r.end - r.start) as usize >= HEAP_SIZE)
        .expect("no usable memory region large enough for heap");

    unsafe {
        ALLOCATOR.lock().init(heap_region.start as *mut u8, HEAP_SIZE);
    }
    serial.log("HEAP", &{
        let mut buf = [0u8; 20];
        let n = format_usize(HEAP_SIZE, &mut buf);
        // SAFETY: digits are always valid ASCII
        core::str::from_utf8(&buf[..n]).unwrap_or("?").to_owned()
    });

    // 3. Init IDT
    init_idt();
    serial.log("IDT", "loaded");

    // 4. Init RDRAND entropy
    let mut entropy = KernelEntropy::new(rdrand_fill);
    serial.log("ENTROPY", "RDRAND available");

    // 5. Generate identity
    let identity = PrivateIdentity::generate(&mut entropy);
    let addr = identity.public_identity().address_hash;
    let mut hex_buf = [0u8; 32];
    hex_encode(&addr, &mut hex_buf);
    serial.log("IDENTITY", core::str::from_utf8(&hex_buf).unwrap());

    // 6. Enter event loop
    let persistence = MemoryState::new();
    let mut runtime = UnikernelRuntime::new(identity, entropy, persistence);
    serial.log("READY", "entering event loop");

    // Signal QEMU test success (exit code 0x10 -> QEMU returns (0x10 << 1) | 1 = 0x21 = 33)
    qemu_exit(0x10);

    let mut tick: u64 = 0;
    loop {
        runtime.tick(tick);
        tick += 1;
        x86_64::instructions::hlt();
    }
}

/// Format a usize as decimal into a byte buffer. Returns number of bytes written.
fn format_usize(mut n: usize, buf: &mut [u8]) -> usize {
    if n == 0 {
        buf[0] = b'0';
        return 1;
    }
    let mut i = 0;
    let mut digits = [0u8; 20];
    while n > 0 {
        digits[i] = b'0' + (n % 10) as u8;
        n /= 10;
        i += 1;
    }
    for j in 0..i {
        buf[j] = digits[i - 1 - j];
    }
    i
}

// --- Panic handler ---

#[panic_handler]
fn panic(info: &PanicInfo) -> ! {
    init_serial();
    let mut serial = SerialWriter::new(serial_write_byte);
    let _ = write!(serial, "[PANIC] {}\n", info);
    loop { x86_64::instructions::hlt(); }
}
```

**Step 2: Verify kernel compiles**

```bash
cargo build -p harmony-boot --target x86_64-unknown-none --release
```

Expected: compiles without errors. This does NOT run tests (bare-metal binary can't run host tests).

**Step 3: Commit**

```bash
cd harmony-os  # relative to workspace root
git add -A
git commit -m "feat(boot): x86_64 entry point with serial, heap, IDT, RDRAND, identity"
```

---

### Task 8: Build Tooling (xtask + justfile)

**Context:** Set up the disk image builder and QEMU launch recipes. The xtask crate uses `bootloader::BiosBoot` to wrap the kernel ELF into a bootable image. The justfile provides convenient recipes.

**Files:**
- Modify: `harmony-os/xtask/src/main.rs` (already scaffolded in Task 1, finalize)
- Create: `harmony-os/justfile`

**Step 1: Finalize xtask (should already be correct from Task 1 scaffold)**

Verify `xtask/src/main.rs` matches the scaffold from Task 1. If any paths need adjusting based on actual build output, update now.

**Step 2: Create justfile**

```makefile
# Harmony OS build recipes

# Default: show available recipes
default:
    @just --list

# Build kernel ELF only
build-kernel:
    cargo build -p harmony-boot --target x86_64-unknown-none --release

# Build bootable BIOS disk image
build: build-kernel
    cargo run --manifest-path xtask/Cargo.toml -- build-image

# Run in QEMU interactively (serial on terminal)
run: build
    qemu-system-x86_64 \
        -drive format=raw,file=target/harmony-boot-bios.img \
        -serial stdio \
        -display none \
        -device isa-debug-exit,iobase=0xf4,iosize=0x04

# Run all host-native unit tests
test:
    cargo test --workspace

# Boot in QEMU and verify identity announcement appears
test-qemu: build
    @echo "Booting QEMU and checking for [IDENTITY] line..."
    timeout 10 qemu-system-x86_64 \
        -drive format=raw,file=target/harmony-boot-bios.img \
        -serial stdio \
        -display none \
        -device isa-debug-exit,iobase=0xf4,iosize=0x04 \
        2>/dev/null | tee /dev/stderr | grep -q '\[IDENTITY\]' \
        && echo "QEMU boot test: PASSED" \
        || (echo "QEMU boot test: FAILED" && exit 1)

# Lint all workspace crates
clippy:
    cargo clippy --workspace
    cargo clippy -p harmony-boot --target x86_64-unknown-none

# Format check
fmt-check:
    cargo fmt --all -- --check

# Full quality gate
check: test clippy fmt-check test-qemu
    @echo "All checks passed."
```

**Step 3: Verify build pipeline**

```bash
cd harmony-os  # relative to workspace root
just build
```

Expected: kernel ELF builds, xtask creates `target/harmony-boot-bios.img`.

**Step 4: Commit**

```bash
git add -A
git commit -m "build: justfile and xtask for BIOS image creation and QEMU launch"
```

---

### Task 9: QEMU Integration Test

**Context:** Boot the image in QEMU and verify the serial output contract.

**Step 1: Run QEMU interactively to verify output**

```bash
cd harmony-os  # relative to workspace root
just run
```

Expected serial output:
```
[BOOT] Harmony unikernel v0.1.0
[HEAP] 1048576
[IDT] loaded
[ENTROPY] RDRAND available
[IDENTITY] <32 hex chars>
[READY] entering event loop
```

If output doesn't match, debug and fix. Common issues:
- RDRAND not available in QEMU: add `-cpu host` or `-cpu qemu64,+rdrand` to QEMU args
- Heap region not found: check memory map iteration logic
- Serial not working: verify UART init sequence

**Step 2: Run automated test**

```bash
just test-qemu
```

Expected: `QEMU boot test: PASSED`

**Step 3: Fix any issues found, commit fixes**

```bash
git add -A
git commit -m "fix(boot): adjustments from QEMU integration testing"
```

(Only if fixes were needed.)

---

### Task 10: Final Verification

**Context:** Run all quality gates before declaring the bead complete.

**Step 1: All host tests**

```bash
cd harmony-os  # relative to workspace root
cargo test --workspace
```

Expected: all tests pass (unikernel + microkernel + os placeholders).

**Step 2: Clippy**

```bash
cargo clippy --workspace
cargo clippy -p harmony-boot --target x86_64-unknown-none
```

Expected: no warnings.

**Step 3: Format**

```bash
cargo fmt --all -- --check
```

Expected: no changes needed.

**Step 4: QEMU test**

```bash
just test-qemu
```

Expected: PASSED.

**Step 5: Commit any final cleanup**

If any quality gate fixes were needed:

```bash
git add -A
git commit -m "chore: clippy and fmt cleanup"
```

---

## Notes

### QEMU CPU flag
If RDRAND isn't available in your QEMU build, add `-cpu qemu64,+rdrand` to the QEMU command. Update xtask and justfile accordingly.

### `alloc` in `format!` on no_std
The `format!` macro requires `alloc`. In `harmony-boot` (no_std binary), use `write!` to a `SerialWriter` instead of `format!`. The `format_usize` helper avoids pulling in the full formatting machinery for simple numbers.

### Dependency versions
The plan specifies `bootloader_api = "0.11"`, `bootloader = "0.11"`, `linked_list_allocator = "0.10"`, `x86_64 = "0.15"`. Check crates.io for latest compatible versions when implementing. If APIs differ from what's shown, adjust accordingly.

### The `to_owned()` call in heap logging
The heap size logging in main.rs uses `to_owned()` which requires alloc's String. An alternative is to pre-format into a stack buffer and pass a `&str`. The implementer may simplify this.

### Multi-repo branches
Both `harmony` and `harmony-os` should have matching `jake-os-boot-stub` branches. The harmony branch already exists with the design doc commit. Create the harmony-os branch in Task 1.
