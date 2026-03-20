# Identity Persistence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Persist the node's PQ and Ed25519 identities to a binary key file so the address survives restarts.

**Architecture:** A new `identity_file` module in `harmony-node` handles load/save of a 161-byte binary file (version + PQ seed + Ed25519 key). `main.rs` calls it before constructing `NodeRuntime`, wiring the loaded identity's address into `NodeConfig::node_addr`.

**Tech Stack:** Rust std::fs, zeroize crate, tempfile (dev-dep for tests)

**Spec:** `docs/plans/2026-03-20-identity-persistence-design.md`

**Worktree:** `/Users/zeblith/work/zeblithic/harmony/.claude/worktrees/jake-node-identity-persist`

---

## File Structure

| File | Responsibility |
|------|---------------|
| Create: `crates/harmony-node/src/identity_file.rs` | Load/save identity key file (161 bytes), validation, permissions |
| Modify: `crates/harmony-node/src/main.rs` | Add `--identity-file` flag, call identity_file before runtime setup, wire `node_addr` |
| Modify: `crates/harmony-node/Cargo.toml` | Add `zeroize` dep, `tempfile` dev-dep |

---

### Task 1: Add dependencies to Cargo.toml

**Files:**
- Modify: `crates/harmony-node/Cargo.toml`

- [ ] **Step 1: Add zeroize and tempfile**

Add `zeroize` to `[dependencies]` and `tempfile` to `[dev-dependencies]`:

```toml
[dependencies]
# ... existing deps ...
zeroize = { workspace = true, features = ["std"] }

[dev-dependencies]
tempfile = "3"
```

- [ ] **Step 2: Verify it compiles**

Run: `cargo check -p harmony-node`
Expected: success

- [ ] **Step 3: Commit**

```bash
git add crates/harmony-node/Cargo.toml
git commit -m "chore: add zeroize and tempfile deps to harmony-node"
```

---

### Task 2: Identity file module — core load/save logic

**Files:**
- Create: `crates/harmony-node/src/identity_file.rs`

- [ ] **Step 1: Write the failing tests**

Create `crates/harmony-node/src/identity_file.rs` with tests at the bottom and stub types/functions at the top. The tests come first — the implementation stubs will fail to compile without the function signatures, so include minimal signatures that compile but `todo!()`.

```rust
//! Identity key file persistence.
//!
//! File format: `[1B version][96B PQ seed][64B Ed25519 key]` = 161 bytes.
//! Default path: `~/.harmony/identity.key`.

use std::path::{Path, PathBuf};

use harmony_identity::{PqPrivateIdentity, PrivateIdentity};
use zeroize::Zeroizing;

const VERSION: u8 = 0x01;
const PQ_KEY_LEN: usize = 96;
const ED25519_KEY_LEN: usize = 64;
const FILE_LEN: usize = 1 + PQ_KEY_LEN + ED25519_KEY_LEN; // 161

/// Both identities loaded from (or generated for) a key file.
pub struct NodeIdentity {
    pub pq: PqPrivateIdentity,
    pub ed25519: PrivateIdentity,
}

/// Resolve the identity file path from an optional CLI override.
///
/// Returns the `--identity-file` path if provided, otherwise
/// `$HOME/.harmony/identity.key`. Errors if `$HOME` is not set
/// and no override was given.
pub fn resolve_path(cli_override: Option<&Path>) -> Result<PathBuf, String> {
    todo!()
}

/// Load identities from an existing key file.
///
/// Validates version byte, file length, and deserializes both identities.
/// Zeroizes the raw buffer on drop.
pub fn load(path: &Path) -> Result<NodeIdentity, String> {
    todo!()
}

/// Save identities to a key file using atomic write (write tmp, fsync, rename).
///
/// Creates parent directories with mode 0o700. File is written with mode 0o600.
/// Zeroizes the raw buffer on drop.
pub fn save(path: &Path, identity: &NodeIdentity) -> Result<(), String> {
    todo!()
}

/// Load identities from a key file, or generate and save new ones if the file
/// does not exist.
pub fn load_or_generate(path: &Path) -> Result<NodeIdentity, String> {
    todo!()
}

/// Check file permissions and warn if too open. Unix-only.
#[cfg(unix)]
fn warn_permissions(path: &Path) {
    todo!()
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn round_trip_save_and_load() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("identity.key");

        let pq = PqPrivateIdentity::generate(&mut rand::rngs::OsRng);
        let ed = PrivateIdentity::generate(&mut rand::rngs::OsRng);
        let pq_addr = pq.public_identity().address_hash;
        let ed_addr = ed.public_identity().address_hash;

        let identity = NodeIdentity { pq, ed25519: ed };
        save(&path, &identity).unwrap();

        let loaded = load(&path).unwrap();
        assert_eq!(loaded.pq.public_identity().address_hash, pq_addr);
        assert_eq!(loaded.ed25519.public_identity().address_hash, ed_addr);
    }

    #[test]
    fn load_wrong_version_fails() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("identity.key");
        let mut data = vec![0x99u8]; // wrong version
        data.extend_from_slice(&[0u8; PQ_KEY_LEN + ED25519_KEY_LEN]);
        std::fs::write(&path, &data).unwrap();

        let err = load(&path).unwrap_err();
        assert!(err.contains("version"), "expected version error, got: {err}");
    }

    #[test]
    fn load_truncated_file_fails() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("identity.key");
        std::fs::write(&path, &[VERSION; 10]).unwrap();

        let err = load(&path).unwrap_err();
        assert!(err.contains("161"), "expected length error, got: {err}");
    }

    #[test]
    fn load_empty_file_fails() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("identity.key");
        std::fs::write(&path, &[]).unwrap();

        let err = load(&path).unwrap_err();
        assert!(err.contains("161"), "expected length error, got: {err}");
    }

    #[test]
    fn load_oversized_file_fails() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("identity.key");
        std::fs::write(&path, &[VERSION; 200]).unwrap();

        let err = load(&path).unwrap_err();
        assert!(err.contains("161"), "expected length error, got: {err}");
    }

    #[test]
    fn load_or_generate_creates_new_file() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("subdir").join("identity.key");
        assert!(!path.exists());

        let identity = load_or_generate(&path).unwrap();
        assert!(path.exists());

        // Reload and verify same addresses
        let reloaded = load(&path).unwrap();
        assert_eq!(
            identity.pq.public_identity().address_hash,
            reloaded.pq.public_identity().address_hash,
        );
        assert_eq!(
            identity.ed25519.public_identity().address_hash,
            reloaded.ed25519.public_identity().address_hash,
        );
    }

    #[test]
    fn load_or_generate_loads_existing() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("identity.key");

        // Generate first
        let first = load_or_generate(&path).unwrap();
        let addr = first.pq.public_identity().address_hash;

        // Load existing — must get same address
        let second = load_or_generate(&path).unwrap();
        assert_eq!(second.pq.public_identity().address_hash, addr);
    }

    #[test]
    fn resolve_path_uses_override() {
        let p = resolve_path(Some(Path::new("/custom/path.key"))).unwrap();
        assert_eq!(p, PathBuf::from("/custom/path.key"));
    }

    #[test]
    fn resolve_path_uses_home() {
        // Only run if $HOME is set (CI and dev machines)
        if std::env::var("HOME").is_ok() {
            let p = resolve_path(None).unwrap();
            assert!(p.to_str().unwrap().contains(".harmony"));
            assert!(p.to_str().unwrap().ends_with("identity.key"));
        }
    }

    #[test]
    fn file_length_constant_is_correct() {
        assert_eq!(FILE_LEN, 161);
        assert_eq!(1 + PQ_KEY_LEN + ED25519_KEY_LEN, 161);
    }

    #[cfg(unix)]
    #[test]
    fn save_creates_directory_and_file_with_correct_permissions() {
        use std::os::unix::fs::PermissionsExt;

        let dir = TempDir::new().unwrap();
        let path = dir.path().join("nested").join("identity.key");

        let identity = NodeIdentity {
            pq: PqPrivateIdentity::generate(&mut rand::rngs::OsRng),
            ed25519: PrivateIdentity::generate(&mut rand::rngs::OsRng),
        };
        save(&path, &identity).unwrap();

        let dir_perms = std::fs::metadata(path.parent().unwrap())
            .unwrap()
            .permissions()
            .mode()
            & 0o777;
        assert_eq!(dir_perms, 0o700, "directory should be 0700");

        let file_perms = std::fs::metadata(&path).unwrap().permissions().mode() & 0o777;
        assert_eq!(file_perms, 0o600, "file should be 0600");
    }
}
```

- [ ] **Step 2: Register the module in main.rs**

Add `mod identity_file;` at the top of `crates/harmony-node/src/main.rs` (after `mod runtime;`):

```rust
mod compute;
mod identity_file;
mod runtime;
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `cargo test -p harmony-node -- identity_file`
Expected: FAIL — all tests hit `todo!()`

- [ ] **Step 4: Implement `resolve_path`**

Replace the `todo!()` in `resolve_path`:

```rust
pub fn resolve_path(cli_override: Option<&Path>) -> Result<PathBuf, String> {
    if let Some(p) = cli_override {
        return Ok(p.to_path_buf());
    }
    let home = std::env::var("HOME")
        .map_err(|_| "Cannot determine identity file path: $HOME not set. Use --identity-file.".to_string())?;
    Ok(PathBuf::from(home).join(".harmony").join("identity.key"))
}
```

- [ ] **Step 5: Implement `load`**

Replace the `todo!()` in `load`:

```rust
pub fn load(path: &Path) -> Result<NodeIdentity, String> {
    let mut buf = Zeroizing::new(
        std::fs::read(path).map_err(|e| format!("Failed to read {}: {e}", path.display()))?,
    );

    if buf.is_empty() || buf.len() != FILE_LEN {
        return Err(format!(
            "Corrupt identity file: expected {FILE_LEN} bytes, got {}",
            buf.len()
        ));
    }
    if buf[0] != VERSION {
        return Err(format!(
            "Unsupported identity file version: {:#04x}",
            buf[0]
        ));
    }

    let pq = PqPrivateIdentity::from_private_bytes(&buf[1..1 + PQ_KEY_LEN])
        .map_err(|e| format!("Corrupt PQ identity in key file: {e}"))?;
    let ed25519 = PrivateIdentity::from_private_bytes(&buf[1 + PQ_KEY_LEN..])
        .map_err(|e| format!("Corrupt Ed25519 identity in key file: {e}"))?;

    #[cfg(unix)]
    warn_permissions(path);

    Ok(NodeIdentity { pq, ed25519 })
}
```

- [ ] **Step 6: Implement `save`**

Replace the `todo!()` in `save`:

```rust
pub fn save(path: &Path, identity: &NodeIdentity) -> Result<(), String> {
    // Create parent directory.
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| format!("Failed to create {}: {e}", parent.display()))?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let _ = std::fs::set_permissions(parent, std::fs::Permissions::from_mode(0o700));
        }
    }

    // Serialize into a zeroizing buffer.
    let pq_bytes = Zeroizing::new(identity.pq.to_private_bytes());
    let ed_bytes = identity.ed25519.to_private_bytes();
    let mut buf = Zeroizing::new(Vec::with_capacity(FILE_LEN));
    buf.push(VERSION);
    buf.extend_from_slice(&pq_bytes);
    buf.extend_from_slice(&ed_bytes);

    // Atomic write: tmp → fsync → rename.
    let tmp_path = path.with_extension("key.tmp");
    std::fs::write(&tmp_path, &*buf)
        .map_err(|e| format!("Failed to write {}: {e}", tmp_path.display()))?;

    // Set permissions before rename so the file is never world-readable.
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&tmp_path, std::fs::Permissions::from_mode(0o600))
            .map_err(|e| format!("Failed to set permissions on {}: {e}", tmp_path.display()))?;
    }

    // fsync the file.
    let f = std::fs::File::open(&tmp_path)
        .map_err(|e| format!("Failed to open {}: {e}", tmp_path.display()))?;
    f.sync_all()
        .map_err(|e| format!("Failed to fsync {}: {e}", tmp_path.display()))?;

    // Rename into place.
    std::fs::rename(&tmp_path, path)
        .map_err(|e| format!("Failed to rename {} → {}: {e}", tmp_path.display(), path.display()))?;

    Ok(())
}
```

- [ ] **Step 7: Implement `load_or_generate`**

Replace the `todo!()` in `load_or_generate`:

```rust
pub fn load_or_generate(path: &Path) -> Result<NodeIdentity, String> {
    if path.exists() {
        return load(path);
    }
    let pq = PqPrivateIdentity::generate(&mut rand::rngs::OsRng);
    let ed25519 = PrivateIdentity::generate(&mut rand::rngs::OsRng);
    let identity = NodeIdentity { pq, ed25519 };
    save(path, &identity)?;
    Ok(identity)
}
```

- [ ] **Step 8: Implement `warn_permissions`**

Replace the `todo!()` in `warn_permissions`:

```rust
#[cfg(unix)]
fn warn_permissions(path: &Path) {
    use std::os::unix::fs::PermissionsExt;
    if let Ok(meta) = std::fs::metadata(path) {
        let mode = meta.permissions().mode() & 0o777;
        if mode & 0o077 != 0 {
            eprintln!(
                "Warning: {} has open permissions ({:#05o}), should be 0600",
                path.display(),
                mode,
            );
        }
    }
}
```

- [ ] **Step 9: Run tests**

Run: `cargo test -p harmony-node -- identity_file`
Expected: all 10 tests pass

- [ ] **Step 10: Commit**

```bash
git add crates/harmony-node/src/identity_file.rs crates/harmony-node/src/main.rs
git commit -m "feat: add identity_file module — load/save/generate 161-byte key file"
```

---

### Task 3: Wire identity into `harmony run`

**Files:**
- Modify: `crates/harmony-node/src/main.rs:20-43` (add CLI flag)
- Modify: `crates/harmony-node/src/main.rs:149-257` (load identity, wire node_addr)

- [ ] **Step 1: Add `--identity-file` to the Run subcommand**

In `Commands::Run`, add after the existing fields:

```rust
        /// Path to the identity key file
        #[arg(long, value_name = "PATH")]
        identity_file: Option<std::path::PathBuf>,
```

- [ ] **Step 2: Add identity_file to the destructure in run()**

In the `Commands::Run { ... }` match arm (around line 149), add `identity_file` to the destructured fields:

```rust
        Commands::Run {
            cache_capacity,
            compute_budget,
            encrypted_durable_persist,
            encrypted_durable_announce,
            no_public_ephemeral_announce,
            filter_broadcast_ticks,
            filter_mutation_threshold,
            identity_file,
        } => {
```

- [ ] **Step 3: Load identity and wire node_addr**

At the beginning of the `Commands::Run` handler block (after the `use` imports, before the `cache_capacity == 0` check), add:

```rust
            // Load or generate node identity.
            let id_path = crate::identity_file::resolve_path(
                identity_file.as_deref(),
            )?;
            let identity = crate::identity_file::load_or_generate(&id_path)?;
            let node_addr = hex::encode(identity.ed25519.public_identity().address_hash);
            eprintln!("Identity: {node_addr} ({})", id_path.display());
```

- [ ] **Step 4: Replace the `node_addr` placeholder**

Replace the `node_addr: "local".to_string()` line (and its comment) with:

```rust
                node_addr,
```

- [ ] **Step 5: Run all tests**

Run: `cargo test -p harmony-node`
Expected: all tests pass (identity_file tests + existing CLI tests)

- [ ] **Step 6: Verify the binary works**

Run: `cargo run -p harmony-node -- run 2>&1 | head -5`
Expected: First line on stderr shows `Identity: <hex> (~/.harmony/identity.key)`, then the existing runtime output.

Run again: `cargo run -p harmony-node -- run 2>&1 | head -1`
Expected: Same hex address (identity persisted).

- [ ] **Step 7: Clean up**

Run: `rm -rf ~/.harmony/identity.key` (remove test identity)

- [ ] **Step 8: Commit**

```bash
git add crates/harmony-node/src/main.rs
git commit -m "feat: wire identity persistence into harmony run — load/generate on startup"
```

---

### Task 4: Push and create PR

- [ ] **Step 1: Run full test suite**

Run: `cargo test -p harmony-node`
Expected: all tests pass

- [ ] **Step 2: Push**

```bash
git push -u origin jake-node-identity-persist
```

- [ ] **Step 3: Create PR**

```bash
gh pr create --title "feat: persist node identity across restarts" --body "$(cat <<'PREOF'
## Summary

- New `identity_file` module: load/save 161-byte binary key file (version + PQ seed + Ed25519 key)
- `harmony run` loads identity on startup or generates one on first run
- `--identity-file <PATH>` override, default `~/.harmony/identity.key`
- Atomic write (tmp → fsync → rename), 0o600 permissions, zeroized buffers
- 10 tests covering round-trip, corruption, permissions, path resolution

## Why

`harmony run` generated a fresh identity on every invocation — the node's address changed each restart, breaking peer recognition. Now the address is stable.

## Test plan

- [x] Round-trip: generate → save → load → compare addresses
- [x] Wrong version byte → error
- [x] Truncated/empty/oversized file → error
- [x] First run creates file, second run loads same address
- [x] Directory + file permissions (Unix)
- [x] Path resolution (override vs $HOME default)
- [x] `cargo run -- run` prints identity and persists it

Tracked by: `harmony-e88`

🤖 Generated with [Claude Code](https://claude.com/claude-code)
PREOF
)"
```
