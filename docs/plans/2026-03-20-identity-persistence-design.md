# Identity Persistence — Design

**Date:** 2026-03-20
**Status:** Proposed
**Bead:** harmony-e88

## Problem

`harmony run` generates a fresh `PrivateIdentity` (Ed25519) on every invocation via `OsRng`. The node's Reticulum address changes each time, breaking link establishment, peer recognition, and announce continuity. For a persistent daemon (especially on an OpenWRT router), the identity must survive restarts.

## Solution

Persist both the PQ identity (`PqPrivateIdentity`, 96-byte seed) and the Ed25519 identity (`PrivateIdentity`, 64-byte key material) to a binary key file. Generate on first run, reload on subsequent runs. PQ is the primary identity; Ed25519 is retained for Reticulum wire compatibility.

## File Format

Raw binary, 161 bytes:

```
[1B version][96B PQ private seed][64B Ed25519 private key]
```

| Offset | Length | Content |
|--------|--------|---------|
| 0 | 1 | Version byte (`0x01`) |
| 1 | 96 | `PqPrivateIdentity::to_private_bytes()` — ML-KEM-768 seed (64B) + ML-DSA-65 seed (32B) |
| 97 | 64 | `PrivateIdentity::to_private_bytes()` — X25519 secret (32B) + Ed25519 secret (32B) |

No header magic, no checksums — the version byte and fixed length are sufficient for validation. The file is not human-editable.

## File Path

**Default:** `~/.harmony/identity.key` (resolved from `$HOME` environment variable).

**Override:** `--identity-file <path>` CLI flag.

**OpenWRT:** The procd init script passes `--identity-file /etc/harmony/identity.key` so the key persists across firmware upgrades (OpenWRT's `/etc` is on overlay).

Follows the `~/.reticulum/` convention from the Python Reticulum reference implementation.

## Behavior

### `harmony run`

1. Resolve key file path (`--identity-file` or `$HOME/.harmony/identity.key`)
2. **If file exists:**
   - Read file contents
   - Validate: version == `0x01`, length == 161 bytes
   - Deserialize PQ identity from bytes `[1..97]`
   - Deserialize Ed25519 identity from bytes `[97..161]`
   - Zeroize the raw byte buffer after deserialization
   - Log: `Identity loaded: <hex address>`
3. **If file does not exist:**
   - Generate PQ identity (`PqPrivateIdentity::generate`)
   - Generate Ed25519 identity (`PrivateIdentity::generate`)
   - Create parent directory with `mkdir -p`, mode `0o700`
   - Write file with mode `0o600`
   - Zeroize the raw byte buffer after writing
   - Log: `Identity generated: <hex address> (saved to <path>)`
4. **If file is corrupt** (wrong version, wrong length, deserialization error):
   - Error and exit — never silently regenerate over an existing file
   - Message: `Error: corrupt identity file at <path>: <reason>`

### `harmony identity new`

Unchanged — still prints keys to stdout. Not wired to the key file. A `--save` flag is YAGNI for now.

### `harmony identity show`

Unchanged — still takes hex-encoded private key from CLI args.

## File Permissions

- **Directory** (`~/.harmony/`): created with mode `0o700`
- **File** (`identity.key`): created with mode `0o600`
- **On load:** check file permissions. If group-readable or world-readable, print a warning to stderr: `Warning: <path> has open permissions (<mode>), should be 0600`. Do not refuse to load — matching ssh's behavior for non-strict mode.

## CLI Changes

Add one flag to the `Run` subcommand:

```rust
/// Path to the identity key file
#[arg(long, value_name = "PATH")]
identity_file: Option<PathBuf>,
```

No changes to `Commands::Identity` subcommands.

## OpenWRT Integration

Update the harmony-openwrt init script and UCI config:

```
# UCI config addition
option identity_file '/etc/harmony/identity.key'
```

```sh
# Init script addition
config_get identity_file main identity_file '/etc/harmony/identity.key'
procd_set_param command /usr/bin/harmony run \
    --identity-file "$identity_file" \
    ...
```

## Error Handling

| Condition | Behavior |
|-----------|----------|
| `$HOME` not set, no `--identity-file` | Error: "Cannot determine identity file path: $HOME not set. Use --identity-file." |
| Parent directory creation fails | Error with OS error (e.g., "Permission denied creating ~/.harmony/") |
| File write fails | Error with OS error |
| File read fails | Error with OS error |
| Version byte != 0x01 | Error: "Unsupported identity file version: <N>" |
| File length != 161 | Error: "Corrupt identity file: expected 161 bytes, got <N>" |
| PQ deserialization fails | Error: "Corrupt PQ identity in key file: <inner error>" |
| Ed25519 deserialization fails | Error: "Corrupt Ed25519 identity in key file: <inner error>" |

All errors exit with non-zero status. No fallback to ephemeral identity — if the user specified a path (or the default exists), it must work.

## Zeroization

- Raw byte buffer from `fs::read()` is zeroized after deserialization
- Raw byte buffer for `fs::write()` is zeroized after writing
- Both identity types implement `Zeroize` and `ZeroizeOnDrop` via derive

Uses `zeroize` crate (already a workspace dependency).

## Testing

| Test | Verifies |
|------|----------|
| Round-trip: generate → save → load → compare addresses | Both identities survive serialization |
| Version byte validation | Wrong version byte → error |
| Length validation | Truncated file → error, oversized file → error |
| Missing file → generates new identity | First-run behavior |
| Existing file → loads identity | Subsequent-run behavior |
| Corrupt PQ bytes → error | Deserialization failure |
| Corrupt Ed25519 bytes → error | Deserialization failure |
| Directory creation | `~/.harmony/` created with 0o700 |
| File permissions | `identity.key` created with 0o600 |

Tests use `tempdir` for isolation — no writes to real `~/.harmony/`.

## What Does NOT Change

- `harmony identity new` — still prints to stdout
- `harmony identity show` / `sign` / `verify` — still use CLI hex args
- `PrivateIdentity` / `PqPrivateIdentity` types — no changes to identity crate
- `NodeRuntime` — receives identity as a parameter, doesn't know about persistence
