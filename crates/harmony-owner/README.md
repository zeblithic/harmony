# harmony-owner

Two-tier owner→device identity binding: cert types, CRDT state, lifecycle
flows, trust evaluation.

See `docs/superpowers/specs/2026-04-25-harmony-owner-device-binding-design.md`
for the full design.

## Quick example

```rust
use harmony_owner::lifecycle::mint_owner;

let unix_now: u64 = 1_744_000_000;
let mint = mint_owner(unix_now).unwrap();
// Save mint.recovery_artifact (BIP39 mnemonic via ZEB-175)
// Use mint.device_signing_key for ongoing device #1 operations
// mint.state has device #1 enrolled
```

## Status

v1 (ZEB-173). Network propagation (Zenoh) and harmony-client wiring tracked
separately under ZEB-169 Track A.
