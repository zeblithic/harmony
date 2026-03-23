# Nix Build → Memo Signing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A shell script that runs `nix build`, computes input/output CIDs, and creates a signed memo attestation.

**Architecture:** Add a `harmony-node cid` subcommand that computes ContentId from file/stdin, then a shell script that orchestrates `nix build → cid → memo sign`. The CID subcommand handles files of any size: ≤1MB uses `ContentId::for_book` directly, >1MB SHA-256 hashes first then wraps in a Book CID.

**Tech Stack:** Rust (clap CLI, harmony-content), Bash (Nix orchestration)

**Spec:** `docs/superpowers/specs/2026-03-22-nix-memo-signing-design.md`

---

## File Structure

| File | Responsibility |
|------|---------------|
| `crates/harmony-node/src/main.rs` | Add `Cid` variant to `Commands` enum, implement handler |
| `deploy/memo-sign-build.sh` | Shell script: nix build → cid → memo sign |

---

### Task 1: `harmony-node cid` subcommand

**Files:**
- Modify: `crates/harmony-node/src/main.rs`

- [ ] **Step 1: Add `Cid` variant to `Commands` enum**

In `crates/harmony-node/src/main.rs`, in the `Commands` enum (after the `Run` variant, around line 94), add:

```rust
    /// Compute ContentId for a file or stdin
    Cid {
        /// Path to file (reads stdin if omitted)
        #[arg(long, value_name = "PATH")]
        file: Option<std::path::PathBuf>,
    },
```

- [ ] **Step 2: Add handler in main()**

In the `main()` function, find the match on `cli.command`. After the `Commands::Run { .. }` arm (but before the closing brace), add a new arm:

```rust
        Commands::Cid { file } => {
            use harmony_content::cid::{ContentFlags, ContentId, MAX_PAYLOAD_SIZE};

            let data = match file {
                Some(path) => std::fs::read(&path)
                    .map_err(|e| format!("Failed to read {}: {e}", path.display()))?,
                None => {
                    use std::io::Read;
                    let mut buf = Vec::new();
                    std::io::stdin()
                        .read_to_end(&mut buf)
                        .map_err(|e| format!("Failed to read stdin: {e}"))?;
                    buf
                }
            };

            if data.is_empty() {
                return Err("Empty input — cannot compute CID".into());
            }

            let cid = if data.len() <= MAX_PAYLOAD_SIZE {
                ContentId::for_book(&data, ContentFlags::default())
                    .map_err(|e| format!("CID computation failed: {e}"))?
            } else {
                // For large files, hash first then wrap in a Book CID.
                let digest = harmony_crypto::hash::full_hash(&data);
                ContentId::for_book(&digest, ContentFlags::default())
                    .map_err(|e| format!("CID computation failed: {e}"))?
            };

            println!("{}", hex::encode(cid.to_bytes()));
        }
```

**Important:** Check the exact import paths. `ContentId`, `ContentFlags`, and `MAX_PAYLOAD_SIZE` are in `harmony_content::cid`. `harmony_crypto::hash::full_hash` returns `[u8; 32]`. The `hex` crate is already used in main.rs.

- [ ] **Step 3: Add CLI parsing tests**

In the `#[cfg(test)]` module at the bottom of main.rs, add:

```rust
    #[test]
    fn cli_parses_cid_with_file() {
        let cli = Cli::try_parse_from(["harmony", "cid", "--file", "/tmp/test.bin"]).unwrap();
        if let Commands::Cid { file } = cli.command {
            assert_eq!(file, Some(std::path::PathBuf::from("/tmp/test.bin")));
        } else {
            panic!("expected Cid command");
        }
    }

    #[test]
    fn cli_parses_cid_stdin_mode() {
        let cli = Cli::try_parse_from(["harmony", "cid"]).unwrap();
        if let Commands::Cid { file } = cli.command {
            assert!(file.is_none());
        } else {
            panic!("expected Cid command");
        }
    }
```

- [ ] **Step 4: Add integration tests**

Add a new integration test file `crates/harmony-node/tests/cid_command.rs`:

```rust
use std::io::Write;
use std::process::Command;

fn harmony_bin() -> Command {
    Command::new(env!("CARGO_BIN_EXE_harmony"))
}

#[test]
fn cid_from_file() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("test.bin");
    std::fs::write(&path, b"hello harmony cid test").unwrap();

    let output = harmony_bin()
        .args(["cid", "--file", path.to_str().unwrap()])
        .output()
        .unwrap();

    assert!(output.status.success(), "stderr: {}", String::from_utf8_lossy(&output.stderr));
    let hex = String::from_utf8(output.stdout).unwrap().trim().to_string();
    assert_eq!(hex.len(), 64, "CID should be 64 hex chars, got {}", hex.len());
    // Verify it's valid hex
    assert!(hex::decode(&hex).is_ok());
}

#[test]
fn cid_from_stdin() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("test.bin");
    let data = b"hello harmony cid test";
    std::fs::write(&path, data).unwrap();

    // Get CID from file
    let file_output = harmony_bin()
        .args(["cid", "--file", path.to_str().unwrap()])
        .output()
        .unwrap();
    let file_cid = String::from_utf8(file_output.stdout).unwrap().trim().to_string();

    // Get CID from stdin with same data
    let mut child = harmony_bin()
        .args(["cid"])
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .unwrap();

    child.stdin.take().unwrap().write_all(data).unwrap();
    let output = child.wait_with_output().unwrap();
    assert!(output.status.success());
    let stdin_cid = String::from_utf8(output.stdout).unwrap().trim().to_string();

    assert_eq!(file_cid, stdin_cid, "file and stdin should produce same CID");
}

#[test]
fn cid_deterministic() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("test.bin");
    std::fs::write(&path, b"determinism check").unwrap();

    let cid1 = harmony_bin()
        .args(["cid", "--file", path.to_str().unwrap()])
        .output()
        .unwrap();
    let cid2 = harmony_bin()
        .args(["cid", "--file", path.to_str().unwrap()])
        .output()
        .unwrap();

    assert_eq!(cid1.stdout, cid2.stdout, "same input must produce same CID");
}

#[test]
fn cid_empty_input_fails() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("empty.bin");
    std::fs::write(&path, b"").unwrap();

    let output = harmony_bin()
        .args(["cid", "--file", path.to_str().unwrap()])
        .output()
        .unwrap();

    assert!(!output.status.success(), "empty input should fail");
}
```

**Note:** Check if `tempfile` is already a dev-dependency of harmony-node. If not, add it to `Cargo.toml`:
```toml
[dev-dependencies]
tempfile = "3"
```

Also check the binary name — it might be `harmony` not `harmony-node` in the Cargo.toml `[[bin]]` section. Use `CARGO_BIN_EXE_harmony` or `CARGO_BIN_EXE_harmony-node` matching the actual binary name.

- [ ] **Step 5: Run tests**

Run: `cargo test -p harmony-node`
Expected: all tests pass (existing + new CLI + integration tests)

- [ ] **Step 6: Commit**

```bash
git add crates/harmony-node/src/main.rs crates/harmony-node/tests/cid_command.rs
git commit -m "feat(cli): add harmony-node cid subcommand

Computes ContentId from file or stdin. Files ≤1MB use for_book
directly; larger files SHA-256 hash first then wrap in a Book CID.
Prints 64-char lowercase hex to stdout.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

If `Cargo.toml` was modified (tempfile dep), include it in the commit.

---

### Task 2: Shell script

**Files:**
- Create: `deploy/memo-sign-build.sh`

- [ ] **Step 1: Create the script**

Create `deploy/memo-sign-build.sh`:

```bash
#!/usr/bin/env bash
# deploy/memo-sign-build.sh — Build a Nix package and sign a memo attestation.
#
# Creates a signed memo linking the Nix derivation (input CID) to the
# built binary (output CID). The memo can be published to the Harmony
# network so other nodes can verify build provenance.
#
# Environment variables:
#   PACKAGE        — Nix package attribute (default: harmony-node)
#   IDENTITY_FILE  — Path to signing identity (optional; defaults to ~/.harmony/identity.key)
#   HARMONY        — Path to harmony-node binary (default: harmony-node from PATH)
#   EXPIRES_IN     — Memo expiry in seconds (default: 31536000 = 365 days)
#
# Usage:
#   ./deploy/memo-sign-build.sh
#   PACKAGE=iroh-relay-x86_64-linux ./deploy/memo-sign-build.sh
#   IDENTITY_FILE=/path/to/key ./deploy/memo-sign-build.sh

set -euo pipefail

PACKAGE="${PACKAGE:-harmony-node}"
HARMONY="${HARMONY:-harmony-node}"
IDENTITY_FILE="${IDENTITY_FILE:-}"
EXPIRES_IN="${EXPIRES_IN:-31536000}"
NIX_FLAGS=(--extra-experimental-features "nix-command flakes")

echo "Building ${PACKAGE}..." >&2
STORE_PATH=$(nix build ".#${PACKAGE}" --print-out-paths --no-link "${NIX_FLAGS[@]}")
BINARY="${STORE_PATH}/bin/${PACKAGE}"

if [ ! -f "${BINARY}" ]; then
    echo "Error: binary not found at ${BINARY}" >&2
    exit 1
fi

echo "Computing input CID from derivation..." >&2
INPUT_CID=$(nix derivation show ".#${PACKAGE}" "${NIX_FLAGS[@]}" | "${HARMONY}" cid)

echo "Computing output CID from ${BINARY}..." >&2
OUTPUT_CID=$("${HARMONY}" cid --file "${BINARY}")

echo "Input CID:  ${INPUT_CID}" >&2
echo "Output CID: ${OUTPUT_CID}" >&2

IDENTITY_ARGS=()
if [ -n "${IDENTITY_FILE}" ]; then
    IDENTITY_ARGS=(--identity-file "${IDENTITY_FILE}")
fi

echo "Signing memo..." >&2
"${HARMONY}" memo sign \
    --input "${INPUT_CID}" \
    --output "${OUTPUT_CID}" \
    --expires-in "${EXPIRES_IN}" \
    "${IDENTITY_ARGS[@]+"${IDENTITY_ARGS[@]}"}"
```

- [ ] **Step 2: Make executable**

```bash
chmod +x deploy/memo-sign-build.sh
```

- [ ] **Step 3: Commit**

```bash
git add deploy/memo-sign-build.sh
git commit -m "feat(deploy): nix build → memo signing script

Orchestrates nix build, CID computation from derivation and binary,
and memo signing in one step. Configurable via PACKAGE, IDENTITY_FILE,
HARMONY, and EXPIRES_IN environment variables.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Full verification

- [ ] **Step 1: Run full workspace tests**

Run: `cargo test --workspace`

- [ ] **Step 2: Run clippy**

Run: `cargo clippy -p harmony-node`

- [ ] **Step 3: Run fmt**

Run: `cargo fmt --all -- --check`
