# Nix Remote Builder Setup

Configure a development workstation (e.g., MacBook) to offload `nix build`
invocations to a more powerful remote machine (e.g., AVALON WSL2) over SSH.
Builds happen on the remote host; results are copied back transparently.

## Overview

```
MacBook (aarch64-darwin)
  └── nix build .#harmony-node-x86_64-linux
        └── SSH ──→ AVALON (x86_64-linux, 24 cores)
                      ├── builds natively
                      └── returns store paths ──→ MacBook
```

This eliminates multi-hour cold-store rebuilds on machines with limited
resources. Combined with [sccache](sccache-setup.md) (Rust compilation
cache) and the Nix binary cache on the relay, most builds are fast regardless
of which machine you're on.

## Prerequisites

- Nix installed on **both** machines (multi-user mode recommended)
- SSH access from the client to the builder
- The builder's `nix-daemon` must be running

## Builder Setup (remote host)

### 1. Ensure nix-daemon is running

```bash
sudo systemctl enable --now nix-daemon
```

### 2. Configure supported platforms

Edit `/etc/nix/nix.conf` on the builder:

```ini
# Native platform
system = x86_64-linux

# Additional platforms this machine can build for.
# Requires binfmt/qemu registration for cross-arch (e.g., aarch64-linux
# on x86_64 host). Omit if not needed.
extra-platforms = aarch64-linux

# Allow the builder user to be trusted (needed for remote builds)
trusted-users = root <your-username>
```

Restart the daemon after editing:

```bash
sudo systemctl restart nix-daemon
```

### 3. Ensure SSH access

The client needs to SSH into the builder as a user listed in `trusted-users`.
Add the client's public key to `~/.ssh/authorized_keys` on the builder:

```bash
# On the client, copy your public key:
cat ~/.ssh/id_ed25519.pub

# On the builder, append to authorized_keys:
echo "ssh-ed25519 AAAA... you@client" >> ~/.ssh/authorized_keys
```

#### WSL2 note: SSH port forwarding

WSL2 doesn't expose SSH directly to the LAN. You have two options:

**Option A — Windows SSH port forward (recommended):**

On the Windows host, forward a port to WSL's SSH:

```powershell
# In an elevated PowerShell prompt
netsh interface portproxy add v4tov4 listenport=2222 listenaddress=0.0.0.0 connectport=22 connectaddress=$(wsl hostname -I | % { $_.Trim() })
```

The client then connects to `<windows-ip>:2222`.

**Option B — Start SSH inside WSL:**

```bash
sudo apt install openssh-server
sudo systemctl enable --now ssh
```

Then forward port 22 from Windows as above.

### 4. Test SSH connectivity

From the client:

```bash
ssh <builder-user>@<builder-host> -p <port> "nix-store --version"
```

This must succeed before configuring the builder in Nix.

## Client Setup (local machine)

### 1. Add the builder to nix.conf

Edit `/etc/nix/nix.conf` (or `~/.config/nix/nix.conf` for single-user Nix):

```ini
builders = ssh-ng://<builder-user>@<builder-host>?port=<port> x86_64-linux - <max-jobs> <speed-factor> <supported-features>

# Allow the builder to fetch from cache.nixos.org on your behalf
builders-use-substitutes = true
```

Example for AVALON (WSL2 via port forward on 2222, 24 cores):

```ini
builders = ssh-ng://zebli@avalon-ip?port=2222 x86_64-linux - 24 1 benchmark,big-parallel

builders-use-substitutes = true
```

**Field reference:**

| Field | Example | Meaning |
|---|---|---|
| URI | `ssh-ng://zebli@avalon-ip?port=2222` | SSH connection to builder |
| Platform | `x86_64-linux` | What the builder can build |
| Features | `-` | SSH key file (- = default) |
| Max jobs | `24` | Concurrent build jobs |
| Speed factor | `1` | Priority vs other builders |
| Supported features | `benchmark,big-parallel` | Nix system features |

### 2. Configure SSH

Add an entry to `~/.ssh/config` for convenience:

```
Host avalon-builder
    HostName <windows-lan-ip>
    Port 2222
    User zebli
    IdentityFile ~/.ssh/id_ed25519
    # Nix remote builds transfer large closures; keep the connection alive
    ServerAliveInterval 60
    ServerAliveCountMax 3
```

Then use the host alias in nix.conf:

```ini
builders = ssh-ng://avalon-builder x86_64-linux - 24 1 benchmark,big-parallel
```

### 3. Restart nix-daemon (if multi-user)

```bash
sudo systemctl restart nix-daemon
```

### 4. Verify

```bash
# Check the builder is recognized
nix show-config | grep builders

# Run a build that targets the builder's platform
nix build .#harmony-node-x86_64-linux --print-build-logs
```

Build logs should show `building on ssh-ng://...` instead of local compilation.

## Multiple Builders

Add multiple builders as semicolon-separated entries:

```ini
builders = ssh-ng://avalon-builder x86_64-linux - 24 1 benchmark,big-parallel ; ssh-ng://other-builder aarch64-linux - 8 2 benchmark
```

Nix distributes builds across available builders based on platform match
and speed factor.

## Troubleshooting

### "cannot connect to builder"

1. Verify SSH works: `ssh avalon-builder "echo ok"`
2. Check the builder's `nix-daemon` is running: `systemctl status nix-daemon`
3. Ensure the SSH user is in `trusted-users` in the builder's nix.conf

### "builder does not support platform"

The builder's `system` and `extra-platforms` in nix.conf must include the
target platform. Check with: `ssh avalon-builder "nix show-config | grep system"`

### Builds still happen locally

- Multi-user Nix: you must restart `nix-daemon` after editing nix.conf
- The `builders` field must use `ssh-ng://` (not `ssh://`) for the Nix 2.x protocol
- Check `nix show-config | grep builders` shows the expected value

### WSL2 IP changed after reboot

WSL2 gets a new IP on each Windows restart. Re-run the `netsh` port proxy
command, or use a script to auto-update it. Alternatively, use `localhost`
if SSH is forwarded through Windows.

## Related

- [sccache setup](sccache-setup.md) — Rust compilation cache (complements remote builders)
- [Relay deploy](../deploy/relay/README.md) — Uses Nix binary cache for relay VM deployment
