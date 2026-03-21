# Nix Derivations for Harmony

**Date:** 2026-03-21
**Status:** Approved
**Scope:** flake.nix + deploy script integration
**Bead:** harmony-qnw

## Overview

Add a Nix flake to the harmony repo that builds `harmony-node` and
`iroh-relay` as reproducible, cross-compilable derivations. This
eliminates the "30-minute compile on e2-micro" problem: the deploy
script cross-compiles from any host in ~2 minutes via Nix, then
SCPs the binary to the VM.

## Motivation

Deploying iroh-relay to a GCP e2-micro currently requires building
from source on the VM (0.25 vCPU, 1GB RAM). This takes 30+ minutes,
saturates the CPU so badly that SSH connections drop, and produces
a binary that could have been built anywhere. Nix solves this:
builds are deterministic functions of their inputs, so any machine
can produce the identical output, and the result is cacheable by
content hash.

## Design

### flake.nix

A single `flake.nix` at the repo root exposing four packages:

```
packages.x86_64-linux.harmony-node
packages.x86_64-linux.iroh-relay
packages.aarch64-linux.harmony-node
packages.aarch64-linux.iroh-relay
```

**Build framework:** [crane](https://github.com/ipetkov/crane) —
the standard Nix build tool for Rust/Cargo workspaces. Crane
understands Cargo workspaces, handles incremental dependency
caching, and integrates with Nix's cross-compilation infrastructure.

**harmony-node:** Built from the local workspace source. Crane
builds the full workspace dependency tree, then compiles the
`harmony-node` binary crate.

**iroh-relay:** Built from a pinned source fetch (GitHub archive
at tag v0.91.2, matching the workspace's Cargo.lock). This is an
external project — we fetch, build, and package it, but don't
modify its source.

**Cross-compilation:** Uses Nix's `pkgsCross` infrastructure to
cross-compile for Linux targets from any host (including macOS).
No Docker, no QEMU — Nix provides the complete cross toolchain
(linker, sysroot, pkg-config wrappers) declaratively.

### Inputs

```nix
{
  nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  crane.url = "github:ipetkov/crane";
  iroh-src = {
    url = "github:n0-computer/iroh/v0.91.2";
    flake = false;
  };
}
```

- `nixpkgs` — system packages, cross-compilation toolchains
- `crane` — Rust build framework
- `iroh-src` — pinned iroh source (non-flake, just the tarball)

### Deploy Script Changes

The build strategy in `deploy/relay/deploy.sh` becomes four tiers:

1. **Pre-built binary** (`LOCAL_BINARY`) — unchanged
2. **Nix build** — `nix build .#iroh-relay`, cross-compiles for
   x86_64-linux from any host. Result at `./result/bin/iroh-relay`.
   Replaces the old "local cargo build" path.
3. **Nix cache** (`NIX_CACHE_URL`) — placeholder for future binary
   cache integration (harmony-m5y). Not wired up in this bead.
4. **Remote build on VM** — unchanged fallback for hosts without Nix

The architecture check (`uname -sm == Linux x86_64`) is removed —
Nix cross-compiles correctly regardless of host platform.

### Updated Detection Logic

```bash
if [ -n "$LOCAL_BINARY" ] && [ -f "$LOCAL_BINARY" ]; then
    # Tier 1: pre-built binary
elif command -v nix &>/dev/null; then
    # Tier 2: nix build (cross-compiles for target)
    nix build .#iroh-relay
    BINARY_PATH="./result/bin/iroh-relay"
else
    # Tier 4: remote build on VM (slow fallback)
fi
```

## What This Bead Delivers

- `flake.nix` at repo root
- `flake.lock` (pinned inputs)
- Updated `deploy/relay/deploy.sh` with Nix build tier
- Updated `deploy/relay/README.md` with Nix prerequisites
- No Rust code changes

## What This Bead Does NOT Deliver

- Binary cache integration (harmony-m5y)
- Zenoh memo publication (harmony-m5y)
- Signed artifact records (harmony-m5y)
- Dev shell / contributor tooling (future bead)

## Testing

- `nix build .#harmony-node` produces a working ELF binary
- `nix build .#iroh-relay` produces a working ELF binary
- `nix flake check` passes
- `file result/bin/harmony-node` confirms ELF x86_64 (not Mach-O)
- Deploy script uses Nix-built binary successfully

## Future: Connection to harmony-m5y

The Nix derivation hashes produced by this flake become the keys
in the Zenoh memo namespace (`harmony/artifacts/{arch}/{drv_hash}`).
The output paths map to Book CIDs in harmony-content. harmony-m5y
will add:

- Signed memo publication on every build
- Trust threshold check (>= 0xF0 for MVP)
- Bloom filter broadcast of available artifacts
- Cache fetch as tier 3 in the deploy script
