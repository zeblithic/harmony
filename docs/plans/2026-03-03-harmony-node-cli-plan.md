# harmony-node CLI Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Create a `harmony` CLI binary with identity management commands (new, show, sign, verify).

**Architecture:** New binary crate `harmony-node` using clap derive for argument parsing. Wraps `harmony-identity` APIs. Ephemeral keys only — no persistence. Integration tests via `std::process::Command`.

**Tech Stack:** Rust, clap (derive), harmony-identity, hex, rand (OsRng)

---

### Task 1: Scaffold the harmony-node crate

**Files:**
- Create: `crates/harmony-node/Cargo.toml`
- Create: `crates/harmony-node/src/main.rs`
- Modify: `Cargo.toml` (workspace members list)

**Step 1: Create Cargo.toml for the binary crate**

```toml
[package]
name = "harmony-node"
version.workspace = true
edition.workspace = true
license.workspace = true
repository.workspace = true
rust-version.workspace = true

[[bin]]
name = "harmony"
path = "src/main.rs"

[dependencies]
harmony-identity = { path = "../harmony-identity" }
clap = { version = "4", features = ["derive"] }
hex = "0.4"
rand = "0.8"
```

**Step 2: Create minimal main.rs with clap skeleton**

```rust
use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "harmony", about = "Harmony decentralized network tools")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Identity management commands
    Identity {
        #[command(subcommand)]
        action: IdentityAction,
    },
}

#[derive(Subcommand)]
enum IdentityAction {
    /// Generate a new identity keypair
    New,
    /// Display public info from a private key
    Show {
        /// Hex-encoded private key (128 hex chars)
        private_key: String,
    },
    /// Sign a message with a private key
    Sign {
        /// Hex-encoded private key (128 hex chars)
        private_key: String,
        /// Message to sign (UTF-8 string)
        message: String,
    },
    /// Verify a signature against a public key
    Verify {
        /// Hex-encoded public key (128 hex chars)
        public_key: String,
        /// Original message (UTF-8 string)
        message: String,
        /// Hex-encoded signature (128 hex chars)
        signature: String,
    },
}

fn main() {
    let cli = Cli::parse();
    if let Err(e) = run(cli) {
        eprintln!("Error: {e}");
        std::process::exit(1);
    }
}

fn run(cli: Cli) -> Result<(), Box<dyn std::error::Error>> {
    match cli.command {
        Commands::Identity { action } => match action {
            IdentityAction::New => todo!("identity new"),
            IdentityAction::Show { .. } => todo!("identity show"),
            IdentityAction::Sign { .. } => todo!("identity sign"),
            IdentityAction::Verify { .. } => todo!("identity verify"),
        },
    }
}
```

**Step 3: Add harmony-node to workspace members**

In the root `Cargo.toml`, add `"crates/harmony-node"` to the `members` list.

**Step 4: Verify it compiles**

Run: `cargo build -p harmony-node`
Expected: Compiles successfully (commands panic with `todo!()`)

**Step 5: Commit**

```
feat(node): scaffold harmony-node CLI crate with clap skeleton
```

---

### Task 2: Implement `identity new`

**Files:**
- Modify: `crates/harmony-node/src/main.rs`
- Test: `crates/harmony-node/tests/identity_cli.rs`

**Step 1: Write the failing integration test**

Create `crates/harmony-node/tests/identity_cli.rs`:

```rust
use std::process::Command;

fn harmony_cmd() -> Command {
    Command::new(env!("CARGO_BIN_EXE_harmony"))
}

#[test]
fn identity_new_prints_three_fields() {
    let output = harmony_cmd()
        .args(["identity", "new"])
        .output()
        .expect("failed to run harmony");
    assert!(output.status.success(), "exit code should be 0");

    let stdout = String::from_utf8_lossy(&output.stdout);
    let lines: Vec<&str> = stdout.lines().collect();
    assert_eq!(lines.len(), 3, "expected 3 lines: {stdout}");

    // Address: 32 hex chars (16 bytes)
    assert!(lines[0].starts_with("Address:"), "line 0: {}", lines[0]);
    let addr_hex = lines[0].trim_start_matches("Address:").trim();
    assert_eq!(addr_hex.len(), 32, "address hex length");
    hex::decode(addr_hex).expect("address should be valid hex");

    // Public key: 128 hex chars (64 bytes)
    assert!(lines[1].starts_with("Public key:"), "line 1: {}", lines[1]);
    let pub_hex = lines[1].trim_start_matches("Public key:").trim();
    assert_eq!(pub_hex.len(), 128, "pubkey hex length");
    hex::decode(pub_hex).expect("pubkey should be valid hex");

    // Private key: 128 hex chars (64 bytes)
    assert!(lines[2].starts_with("Private key:"), "line 2: {}", lines[2]);
    let priv_hex = lines[2].trim_start_matches("Private key:").trim();
    assert_eq!(priv_hex.len(), 128, "privkey hex length");
    hex::decode(priv_hex).expect("privkey should be valid hex");
}
```

Add `hex` as a dev-dependency in `crates/harmony-node/Cargo.toml`:
```toml
[dev-dependencies]
hex = "0.4"
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p harmony-node --test identity_cli identity_new_prints_three_fields`
Expected: FAIL (todo panic)

**Step 3: Implement `identity new`**

Replace the `IdentityAction::New` arm in `run()`:

```rust
IdentityAction::New => {
    let id = harmony_identity::PrivateIdentity::generate(&mut rand::rngs::OsRng);
    let pub_id = id.public_identity();
    println!("Address:     {}", hex::encode(pub_id.address_hash));
    println!("Public key:  {}", hex::encode(pub_id.to_public_bytes()));
    println!("Private key: {}", hex::encode(id.to_private_bytes()));
    Ok(())
}
```

**Step 4: Run test to verify it passes**

Run: `cargo test -p harmony-node --test identity_cli identity_new_prints_three_fields`
Expected: PASS

**Step 5: Commit**

```
feat(node): implement 'harmony identity new' command
```

---

### Task 3: Implement `identity show`

**Files:**
- Modify: `crates/harmony-node/src/main.rs`
- Modify: `crates/harmony-node/tests/identity_cli.rs`

**Step 1: Write the failing test**

Add to `identity_cli.rs`:

```rust
#[test]
fn identity_show_round_trips_from_new() {
    // Generate a key with 'identity new'.
    let new_output = harmony_cmd()
        .args(["identity", "new"])
        .output()
        .expect("failed to run identity new");
    let new_stdout = String::from_utf8_lossy(&new_output.stdout);
    let new_lines: Vec<&str> = new_stdout.lines().collect();
    let expected_addr = new_lines[0].trim_start_matches("Address:").trim();
    let expected_pub = new_lines[1].trim_start_matches("Public key:").trim();
    let priv_hex = new_lines[2].trim_start_matches("Private key:").trim();

    // Show should produce the same address and pubkey.
    let show_output = harmony_cmd()
        .args(["identity", "show", priv_hex])
        .output()
        .expect("failed to run identity show");
    assert!(show_output.status.success());

    let show_stdout = String::from_utf8_lossy(&show_output.stdout);
    let show_lines: Vec<&str> = show_stdout.lines().collect();
    assert_eq!(show_lines.len(), 2, "show should print 2 lines");

    let show_addr = show_lines[0].trim_start_matches("Address:").trim();
    let show_pub = show_lines[1].trim_start_matches("Public key:").trim();
    assert_eq!(show_addr, expected_addr);
    assert_eq!(show_pub, expected_pub);
}

#[test]
fn identity_show_rejects_invalid_hex() {
    let output = harmony_cmd()
        .args(["identity", "show", "not-valid-hex"])
        .output()
        .expect("failed to run");
    assert!(!output.status.success(), "should fail on bad hex");
}
```

**Step 2: Run tests to verify they fail**

Run: `cargo test -p harmony-node --test identity_cli identity_show`
Expected: FAIL (todo panic)

**Step 3: Implement `identity show`**

Add a hex-decoding helper and implement the `Show` arm:

```rust
fn decode_hex_key(hex_str: &str, expected_len: usize, label: &str) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let bytes = hex::decode(hex_str).map_err(|e| format!("invalid {label} hex: {e}"))?;
    if bytes.len() != expected_len {
        return Err(format!("{label}: expected {expected_len} bytes, got {}", bytes.len()).into());
    }
    Ok(bytes)
}
```

```rust
IdentityAction::Show { private_key } => {
    let bytes = decode_hex_key(&private_key, 64, "private key")?;
    let id = harmony_identity::PrivateIdentity::from_private_bytes(&bytes)?;
    let pub_id = id.public_identity();
    println!("Address:    {}", hex::encode(pub_id.address_hash));
    println!("Public key: {}", hex::encode(pub_id.to_public_bytes()));
    Ok(())
}
```

**Step 4: Run tests**

Run: `cargo test -p harmony-node --test identity_cli identity_show`
Expected: PASS (both tests)

**Step 5: Commit**

```
feat(node): implement 'harmony identity show' command
```

---

### Task 4: Implement `identity sign`

**Files:**
- Modify: `crates/harmony-node/src/main.rs`
- Modify: `crates/harmony-node/tests/identity_cli.rs`

**Step 1: Write the failing test**

Add to `identity_cli.rs`:

```rust
#[test]
fn identity_sign_produces_valid_signature() {
    // Generate a key.
    let new_output = harmony_cmd()
        .args(["identity", "new"])
        .output()
        .unwrap();
    let new_stdout = String::from_utf8_lossy(&new_output.stdout);
    let new_lines: Vec<&str> = new_stdout.lines().collect();
    let priv_hex = new_lines[2].trim_start_matches("Private key:").trim();

    // Sign a message.
    let sign_output = harmony_cmd()
        .args(["identity", "sign", priv_hex, "hello harmony"])
        .output()
        .unwrap();
    assert!(sign_output.status.success());

    let sign_stdout = String::from_utf8_lossy(&sign_output.stdout);
    let sig_line = sign_stdout.lines().next().expect("should have output");
    assert!(sig_line.starts_with("Signature:"), "line: {sig_line}");
    let sig_hex = sig_line.trim_start_matches("Signature:").trim();
    assert_eq!(sig_hex.len(), 128, "signature should be 128 hex chars (64 bytes)");
    hex::decode(sig_hex).expect("signature should be valid hex");
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p harmony-node --test identity_cli identity_sign`
Expected: FAIL (todo panic)

**Step 3: Implement `identity sign`**

```rust
IdentityAction::Sign { private_key, message } => {
    let bytes = decode_hex_key(&private_key, 64, "private key")?;
    let id = harmony_identity::PrivateIdentity::from_private_bytes(&bytes)?;
    let signature = id.sign(message.as_bytes());
    println!("Signature: {}", hex::encode(signature));
    Ok(())
}
```

**Step 4: Run test**

Run: `cargo test -p harmony-node --test identity_cli identity_sign`
Expected: PASS

**Step 5: Commit**

```
feat(node): implement 'harmony identity sign' command
```

---

### Task 5: Implement `identity verify`

**Files:**
- Modify: `crates/harmony-node/src/main.rs`
- Modify: `crates/harmony-node/tests/identity_cli.rs`

**Step 1: Write the failing tests**

Add to `identity_cli.rs`:

```rust
#[test]
fn sign_then_verify_round_trip() {
    // Generate key, sign, then verify.
    let new_output = harmony_cmd()
        .args(["identity", "new"])
        .output()
        .unwrap();
    let new_stdout = String::from_utf8_lossy(&new_output.stdout);
    let new_lines: Vec<&str> = new_stdout.lines().collect();
    let pub_hex = new_lines[1].trim_start_matches("Public key:").trim();
    let priv_hex = new_lines[2].trim_start_matches("Private key:").trim();

    let message = "test message for verification";

    // Sign.
    let sign_output = harmony_cmd()
        .args(["identity", "sign", priv_hex, message])
        .output()
        .unwrap();
    let sign_stdout = String::from_utf8_lossy(&sign_output.stdout);
    let sig_hex = sign_stdout
        .lines()
        .next()
        .unwrap()
        .trim_start_matches("Signature:")
        .trim();

    // Verify — should succeed.
    let verify_output = harmony_cmd()
        .args(["identity", "verify", pub_hex, message, sig_hex])
        .output()
        .unwrap();
    assert!(verify_output.status.success(), "valid signature should verify");
    let verify_stdout = String::from_utf8_lossy(&verify_output.stdout);
    assert!(verify_stdout.contains("Valid"), "should print Valid");
}

#[test]
fn verify_rejects_corrupted_signature() {
    // Generate and sign.
    let new_output = harmony_cmd()
        .args(["identity", "new"])
        .output()
        .unwrap();
    let new_stdout = String::from_utf8_lossy(&new_output.stdout);
    let new_lines: Vec<&str> = new_stdout.lines().collect();
    let pub_hex = new_lines[1].trim_start_matches("Public key:").trim();
    let priv_hex = new_lines[2].trim_start_matches("Private key:").trim();

    let sign_output = harmony_cmd()
        .args(["identity", "sign", priv_hex, "original"])
        .output()
        .unwrap();
    let sign_stdout = String::from_utf8_lossy(&sign_output.stdout);
    let sig_hex = sign_stdout
        .lines()
        .next()
        .unwrap()
        .trim_start_matches("Signature:")
        .trim();

    // Verify with wrong message — should fail.
    let verify_output = harmony_cmd()
        .args(["identity", "verify", pub_hex, "tampered", sig_hex])
        .output()
        .unwrap();
    assert!(!verify_output.status.success(), "tampered message should fail verification");
}
```

**Step 2: Run tests to verify they fail**

Run: `cargo test -p harmony-node --test identity_cli verify`
Expected: FAIL (todo panic)

**Step 3: Implement `identity verify`**

```rust
IdentityAction::Verify { public_key, message, signature } => {
    let pub_bytes = decode_hex_key(&public_key, 64, "public key")?;
    let sig_bytes = decode_hex_key(&signature, 64, "signature")?;
    let identity = harmony_identity::Identity::from_public_bytes(&pub_bytes)?;
    let sig_array: [u8; 64] = sig_bytes.try_into().unwrap();
    match identity.verify(message.as_bytes(), &sig_array) {
        Ok(()) => {
            println!("Valid");
            Ok(())
        }
        Err(_) => {
            eprintln!("Invalid");
            std::process::exit(1);
        }
    }
}
```

**Step 4: Run tests**

Run: `cargo test -p harmony-node --test identity_cli verify`
Expected: PASS (both tests)

**Step 5: Commit**

```
feat(node): implement 'harmony identity verify' command
```

---

### Task 6: Final quality gates

**Step 1: Run all harmony-node tests**

Run: `cargo test -p harmony-node`
Expected: All 5 integration tests pass

**Step 2: Run clippy**

Run: `cargo clippy -p harmony-node`
Expected: No warnings

**Step 3: Run format check**

Run: `cargo fmt --all -- --check`
Expected: No formatting issues

**Step 4: Run full workspace tests (regression check)**

Run: `cargo test --workspace`
Expected: All 365+ tests pass

**Step 5: Commit any cleanups, then the bead is ready for delivery**

Claim the bead: `bd update harmony-971 --claim`
