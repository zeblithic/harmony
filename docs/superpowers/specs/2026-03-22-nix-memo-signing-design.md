# Nix Build → Memo Signing Integration

**Bead:** harmony-s1i
**Date:** 2026-03-22
**Status:** Draft

## Problem

After a Nix build completes, there's no way to create a signed attestation
linking the build inputs (derivation) to the build output (binary). The memo
system can express "input → output" attestations, and the CLI can sign them,
but nothing computes the CIDs from Nix artifacts or orchestrates the flow.

## Solution

A shell script (`deploy/memo-sign-build.sh`) that runs `nix build`, computes
input/output CIDs via a new `harmony-node cid` subcommand, and calls
`harmony-node memo sign`. The CID subcommand encapsulates ContentId formatting
so the script doesn't depend on internal CID layout.

## Design Decisions

### Shell script, not a Rust subcommand

The existing deploy infrastructure is shell-based (`deploy.sh`,
`nix-cache-setup.sh`). A shell script keeps the orchestration composable —
callable from CI, deploy scripts, or manually. The memo CLI already does the
cryptographic heavy lifting; this is just glue.

### `harmony-node cid` subcommand for CID computation

Computing a ContentId requires the 4-byte header + 28-byte hash format. Rather
than encoding this in bash (fragile, duplicates logic), a small subcommand
reads a file or stdin and prints the 64-char hex CID. Useful beyond this
script.

### Derivation JSON as input CID source

`nix derivation show .#pkg` outputs the full derivation JSON — all inputs,
compiler flags, source hashes. This is deterministic: same build inputs always
produce the same derivation. Hashing this captures "what was asked for."

### Binary file as output CID source

The built binary at `$STORE_PATH/bin/$PACKAGE` is the concrete build result.
Hashing it captures "what was produced." Different compilers or flags produce
different binaries, different CIDs.

## Architecture

### `harmony-node cid` subcommand

```
harmony-node cid [--file PATH]
```

- `--file PATH`: Read file at PATH, compute CID
- No `--file`: Read stdin to EOF, compute CID
- Output: 64-char lowercase hex ContentId to stdout
- Errors: empty input, file not found, read failure
- Files ≤ `MAX_PAYLOAD_SIZE` (~1MB): `ContentId::for_book(data, ContentFlags::default())`
- Files > `MAX_PAYLOAD_SIZE`: SHA-256 hash the data first, then `ContentId::for_book(&hash, ContentFlags::default())` — wraps the digest in a Book CID

### `deploy/memo-sign-build.sh`

```bash
#!/usr/bin/env bash
set -euo pipefail

PACKAGE="${PACKAGE:-harmony-node}"
IDENTITY_FILE="${IDENTITY_FILE:-}"
HARMONY="${HARMONY:-harmony-node}"

STORE_PATH=$(nix build ".#${PACKAGE}" --print-out-paths --no-link \
    --extra-experimental-features "nix-command flakes")
BINARY="${STORE_PATH}/bin/${PACKAGE}"

INPUT_CID=$(nix derivation show ".#${PACKAGE}" \
    --extra-experimental-features "nix-command flakes" \
    | "${HARMONY}" cid)
OUTPUT_CID=$("${HARMONY}" cid --file "${BINARY}")

IDENTITY_ARGS=()
if [ -n "${IDENTITY_FILE}" ]; then
    IDENTITY_ARGS=(--identity-file "${IDENTITY_FILE}")
fi

"${HARMONY}" memo sign \
    --input "${INPUT_CID}" \
    --output "${OUTPUT_CID}" \
    "${IDENTITY_ARGS[@]+"${IDENTITY_ARGS[@]}"}"
```

Environment variables:
- `PACKAGE` — Nix package name (default: `harmony-node`)
- `IDENTITY_FILE` — path to signing identity (optional; defaults to `~/.harmony/identity.key` when omitted)
- `HARMONY` — path to harmony-node binary (default: `harmony-node` from PATH)

## File Changes

| File | Change |
|------|--------|
| `crates/harmony-node/src/main.rs` | Add `Cid` subcommand with `--file` option |
| `deploy/memo-sign-build.sh` | New shell script: nix build → cid → memo sign |

## What is NOT in Scope

- No changes to the memo crate or signing logic
- No Nix post-build hooks (requires identity in build sandbox)
- No automatic Zenoh publishing of the memo (script prints to stdout)
- No verification flow (existing `memo verify` handles that)
- No multi-output support (single binary per invocation)

## Testing

- `cid_from_file` — compute CID from temp file, verify 64 hex chars
- `cid_from_stdin` — pipe data, verify same CID as file mode
- `cid_empty_input_fails` — empty stdin returns error
- `cid_deterministic` — same input always produces same CID
- Shell script: manual integration test against real `nix build`
