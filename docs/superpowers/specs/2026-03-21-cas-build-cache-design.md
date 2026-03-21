# CAS-Backed Build Cache: Nix Binary Cache for Relay Deployment

**Bead:** harmony-bef
**Date:** 2026-03-21
**Status:** Draft

## Problem

The relay deploy script (`deploy/relay/deploy.sh`) falls back to compiling
`iroh-relay` from source on a GCP e2-micro VM when no pre-built binary is
available. This takes 15-30 minutes on the constrained VM. The workstation
(or any capable machine) should build once and the result should be available
to any device that needs it.

## Solution

Use the existing `flake.nix` (which already builds `iroh-relay-x86_64-linux`
via crane) to build deterministically on the workstation. Push the Nix closure
to the relay VM via `nix copy` over `gcloud compute ssh`. Run `nix-serve` on
the VM to serve the cached binary over HTTP (port 5000) so any Nix-enabled
machine can fetch it without SSH access.

Replace the cargo build paths in `deploy.sh` with a Nix-first strategy.

## Architecture

### Nix Binary Cache on the Relay VM

`nix-serve` on `i.q8.fyi:5000`, serving directly from `/nix/store`.

**Signing:** The cache signs all NARs with a private key on the VM
(`/etc/nix/cache-key.pem`). Clients verify with the public key committed
to the repo (`deploy/relay/cache-key.pub`). Standard Nix trust model.

**No reverse proxy.** `nix-serve` listens on port 5000 directly. Binary
cache integrity comes from signing keys, not transport security. Avoids
disrupting the existing iroh-relay TLS setup on 80/443.

**Note:** `nix-serve` exposes the entire `/nix/store` contents on port 5000.
This is intentional — NARs are content-addressed and signed, so exposure is
low-risk. Only iroh-relay closures should be pushed to the VM.

**VM requirements:**
- Nix installed (multi-user with daemon, so `nix copy` can write to the store
  regardless of which SSH user connects via `gcloud compute ssh`)
- `nix-serve` running as a systemd service
- Port 5000 open in GCP firewall

**Memory:** `nix-serve` is a lightweight Perl script (~5 MB RSS). Acceptable
alongside `iroh-relay` on the 1 GB e2-micro. Can be stopped if memory becomes
tight — deploy still works via direct SSH copy.

### SSH Transport: gcloud compute ssh

The existing deploy script uses `gcloud compute ssh` exclusively — not direct
`ssh root@host`. Port 22 is not open in the GCP firewall, and root login is
typically disabled on GCP VMs.

`nix copy --to ssh://` requires a direct SSH connection. Instead, we use
`nix copy` with a custom SSH command that wraps `gcloud compute ssh`:

```bash
NIX_SSHOPTS="-o ProxyCommand='gcloud compute ssh %h --zone=$ZONE --command=nc\ %h\ %p'"
```

Alternatively, the simpler approach: build locally with `nix build`, serialize
the closure with `nix-store --export`, pipe it through `gcloud compute ssh`
and import on the VM with `nix-store --import`. This avoids the `nix copy`
SSH protocol entirely:

```bash
# Build locally, export closure, import on VM via gcloud ssh
STORE_PATH=$(nix build .#iroh-relay-x86_64-linux --print-out-paths --no-link)
nix-store --export $(nix-store -qR "$STORE_PATH") | \
    gcloud compute ssh "$VM_NAME" --zone="$ZONE" -- \
    "sudo nix-store --import"
```

This is the recommended approach: it uses the same `gcloud compute ssh`
transport as the rest of deploy.sh, requires no firewall changes for SSH,
and works with multi-user Nix on the VM.

### Deploy Script Strategy

New fallback order in `deploy.sh`:

1. **`LOCAL_BINARY` env var** — escape hatch, scp binary to VM (unchanged)
2. **Check VM Nix store** — query whether the expected store path already
   exists on the VM (via `gcloud compute ssh ... nix path-info <path>`). If
   present, skip build and copy — just activate. Fast path for repeat deploys.
3. **`nix build` locally + push** — if `nix` is on the deployer's machine,
   build `.#iroh-relay-x86_64-linux` from the flake (~2 min on workstation),
   export + import the closure to the VM. Sign on the VM.

**Store path identification:** `nix build --print-out-paths` captures the
exact `/nix/store/...-iroh-relay-0.91.2/bin/iroh-relay` path. This is passed
to the VM for activation (`cp <store-path>/bin/iroh-relay /usr/local/bin/`).

**What gets removed:**
- `cargo build` local fallback
- Remote-build-on-VM section (the 30-minute pain point)
- `rustup` installation logic on the VM
- All cargo/rustc version checks

**What stays:**
- `LOCAL_BINARY` path
- All infrastructure setup (GCP instance, firewall, DNS, TLS)
- systemd service management
- Rate limiting, health checks

**Activation:** After the binary is in the VM's Nix store, copy it to
`/usr/local/bin/iroh-relay` and restart the systemd service. Same as today.

**Rollback:** Previous versions remain in `/nix/store`. To roll back, the
deployer re-activates a previous store path — no rebuild needed. The script
can record the previous active path before overwriting for instant rollback.

### Version Management

The `flake.nix` pins `iroh` at `v0.91.2` via a flake input. To deploy a
different version, update the flake input and lock:

```bash
nix flake update iroh-relay-src  # or edit flake.nix input URL
nix build .#iroh-relay-x86_64-linux
```

The `IROH_VERSION` env var in deploy.sh is removed — version is controlled
by the flake lock, which is deterministic and reproducible. The deploy script
no longer needs to know the version; it just builds whatever the flake defines.

### Push Workflow

```bash
# Build on workstation
STORE_PATH=$(nix build .#iroh-relay-x86_64-linux --print-out-paths --no-link)

# Push to VM's Nix store (via gcloud ssh)
nix-store --export $(nix-store -qR "$STORE_PATH") | \
    gcloud compute ssh "$VM_NAME" --zone="$ZONE" -- \
    "sudo nix-store --import"

# Sign on VM (specific paths, not --all)
gcloud compute ssh "$VM_NAME" --zone="$ZONE" -- \
    "sudo nix store sign --key-file /etc/nix/cache-key.pem '$STORE_PATH'"
```

### Client Configuration

Any Nix-enabled machine can use the cache:

```nix
# In nix.conf or flake.nix nixConfig
substituters = https://cache.nixos.org http://i.q8.fyi:5000
trusted-public-keys = cache.nixos.org-1:... i.q8.fyi:PUBLIC_KEY_HERE
```

## One-Time Setup Script

`deploy/relay/nix-cache-setup.sh` performs the following on the relay VM
(run via `gcloud compute ssh`):

1. Install Nix (multi-user with daemon) if not present
2. Generate binary cache signing keypair if not present
3. Install `nix-serve` from nixpkgs
4. Create systemd service for `nix-serve --port 5000`
5. Open port 5000 in GCP firewall
6. Print the public key for the deployer to commit to the repo

## File Changes

| File | Change |
|------|--------|
| `deploy/relay/nix-cache-setup.sh` | New: one-time VM setup |
| `deploy/relay/deploy.sh` | Rewrite build section: Nix-first, remove cargo. Add port 5000 to firewall rules. |
| `deploy/relay/cache-key.pub` | New: public signing key |
| `deploy/relay/README.md` | Update to reflect Nix workflow |

## What is NOT in Scope

- No changes to `flake.nix` (already builds iroh-relay)
- No CI/CD pipeline (manual push for now — CI is a follow-up)
- No Harmony CAS/NDN integration (future: serve NARs as Books via Zenoh)
- No reverse proxy or TLS for the cache endpoint
- No Cachix or third-party hosting

## Error Handling

| Scenario | Behavior |
|----------|----------|
| `nix` not installed on deployer | Error: "Install Nix or set LOCAL_BINARY" |
| `nix build` fails | Error with build log, suggest `LOCAL_BINARY` fallback |
| `nix-store --export` pipe fails | Error with SSH diagnostic |
| VM Nix store already has the binary | Skip build and copy, just activate (fast path) |
| `nix-serve` is down | Other machines can't fetch, but deploy still works via SSH pipe |
| Signing key missing on VM | Warn and skip signing — nix-serve serves unsigned NARs |

## Testing

- Run `nix build .#iroh-relay-x86_64-linux` locally — verify binary produced
- Run `nix-cache-setup.sh` on the VM — verify nix-serve starts and responds
- Run modified `deploy.sh` — verify it builds via Nix, copies, activates
- Verify `curl http://i.q8.fyi:5000/nix-cache-info` returns valid response
- Verify another machine can fetch from the substituter and gets a cache hit
- Deploy again without changes — verify fast path (store path already present)
