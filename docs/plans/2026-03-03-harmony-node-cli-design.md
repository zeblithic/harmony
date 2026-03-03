# harmony-node CLI Design

**Goal:** Create a `harmony` CLI binary for identity management — generating keypairs, displaying public keys/addresses, and signing/verifying messages. Foundation for manual testing of the cryptographic stack.

**Scope:** Identity-only MVP. No network runtime, no key persistence. Ephemeral keys only.

---

## Crate Structure

New binary crate: `crates/harmony-node/`

- Workspace member with `[[bin]]` target named `harmony`
- Depends on: `harmony-identity`, `clap` (derive), `hex`
- Library crates remain dependency-free of CLI concerns

## Commands

### `harmony identity new`

Generate a new identity and print to stdout:

```
Address:     <32 hex chars>  (16-byte truncated SHA-256)
Public key:  <128 hex chars> (64 bytes: X25519 || Ed25519)
Private key: <128 hex chars> (64 bytes: X25519 || Ed25519)
```

Implementation: `PrivateIdentity::generate(&mut OsRng)`, then format fields.

### `harmony identity show <PRIVATE_KEY_HEX>`

Reconstruct identity from private key bytes and display public info:

```
Address:    <32 hex chars>
Public key: <128 hex chars>
```

Implementation: `PrivateIdentity::from_private_bytes()`, extract `public_identity()`.

### `harmony identity sign <PRIVATE_KEY_HEX> <MESSAGE>`

Sign a UTF-8 message and print the signature:

```
Signature: <128 hex chars> (64-byte Ed25519)
```

Implementation: `private_identity.sign(message.as_bytes())`.

### `harmony identity verify <PUBLIC_KEY_HEX> <MESSAGE> <SIGNATURE_HEX>`

Verify a signature against a public key and message:

```
Valid       (exit code 0)
Invalid     (exit code 1)
```

Implementation: `Identity::from_public_bytes()`, then `identity.verify(message, signature)`.

## Output Format

Plain text, one field per line, labels left-aligned. Hex-encoded values for all
binary data. No JSON (YAGNI). Exit code 0 for success, 1 for errors/invalid
signatures.

## Error Handling

All errors go to stderr via `clap`'s built-in error formatting or `eprintln!`.
Invalid hex, wrong key lengths, and bad signatures produce human-readable
messages and exit code 1.

## Dependencies

| Crate | Purpose |
|-------|---------|
| `harmony-identity` | Keypair generation, signing, verification |
| `clap` (derive) | CLI argument parsing |
| `hex` | Hex encoding/decoding |
| `rand` | `OsRng` for key generation |

## Testing

Integration tests that invoke the binary via `std::process::Command`:

1. `new` produces valid hex output with correct field lengths
2. `show` round-trips: `new` → extract private key → `show` → same address/pubkey
3. `sign` + `verify` round-trip: sign a message, verify succeeds
4. `verify` rejects corrupted signature (exit code 1)
5. `show` rejects invalid hex input (exit code 1)

## Non-Goals

- Key persistence / wallet (future `harmony identity save/load`)
- Network runtime / `node start` (future, needs tokio + transport)
- JSON output (add `--format json` later if needed)
- Encryption/decryption commands (future, needs recipient key exchange)
