# Harmony Relay Server Deployment

Deploy a stock [iroh-relay](https://github.com/n0-computer/iroh/tree/main/iroh-relay)
server on GCP for cross-NAT tunnel bootstrapping.

## What This Does

Harmony peers behind different NATs can't reach each other directly.
The relay server acts as a rendezvous point: peers connect to the relay,
which forwards encrypted bytes between them until they can hole-punch
a direct QUIC connection. The relay never sees plaintext (PQ encryption
is end-to-end).

## Architecture

```
Peer A (behind NAT) ──QUIC──→ i.q8.fyi ←──QUIC── Peer B (behind NAT)
                                 │
                          stock iroh-relay
                          e2-micro (free tier)
                          Let's Encrypt TLS
```

- **Single GCP e2-micro instance** (free tier eligible in `us-west1`)
- **Stock `iroh-relay` binary** — no Harmony-specific server code
- **Let's Encrypt TLS** via built-in ACME support
- **Ports:** 80/tcp (HTTP), 443/tcp (HTTPS), 7842/udp (QUIC)

## Quick Start

### Prerequisites

**Automated (`deploy.sh`):**
- [Google Cloud SDK](https://cloud.google.com/sdk/docs/install) (`gcloud` CLI)
- A GCP project with billing enabled
- A domain with DNS you control
- [Nix package manager](https://nixos.org/download/) (used to build `iroh-relay` via the binary cache)

**Manual runbook (Step 4 builds via Nix):**
- All of the above, plus Nix installed locally
- Alternatively, set `LOCAL_BINARY=/path/to/iroh-relay` to skip the Nix build and supply a pre-built binary directly

### Automated Deployment

```bash
# Set your hostname and GCP project
export RELAY_HOSTNAME="i.q8.fyi"
export GCP_PROJECT="your-gcp-project-id"

# Run the deployment script
./deploy/relay/deploy.sh
```

The script will:
1. Create an `e2-micro` VM in `us-west1-b` with Debian 12
2. Reserve a static external IP
3. Configure firewall rules (80/tcp, 443/tcp, 7842/udp)
4. Pull `iroh-relay` from the Nix binary cache (or build via `nix build` if not cached)
5. Install the systemd unit and config
6. Start the service

After the script completes, it prints the static IP. Point your DNS
A record at that IP:

```
i.q8.fyi.  A  <static-ip>
```

### Manual Deployment

See the step-by-step runbook below.

## Binary Cache

The deployment uses a Nix binary cache on the relay VM so that `iroh-relay`
builds are fast and reproducible without a local Rust toolchain.

### First-Time Setup

After the VM is created (Step 1 of the runbook), run the cache setup script:

```bash
gcloud compute ssh harmony-relay --zone=us-west1-b -- \
    'bash -s' < ./deploy/relay/nix-cache-setup.sh
```

This generates a signing keypair at `/etc/nix/cache-key` on the VM, installs
Nix, and configures the VM to serve as an HTTP binary cache on port 5000.
After running, copy the public key back into the repo:

```bash
gcloud compute ssh harmony-relay --zone=us-west1-b -- \
    cat /etc/nix/cache-key.pub > ./deploy/relay/cache-key.pub
```

Commit `cache-key.pub` so clients can verify cache signatures.

### How It Works

```
deploy.sh:
  1. Check VM has Nix (pre-flight)
  2. Pass --extra-substituters to nix build (best-effort cache hit)
  3. nix build .#iroh-relay-x86_64-linux
     └── check local /nix/store
     └── check binary cache (http://<relay>:5000) if cache-key.pub is set
     └── build from source (if both miss)
  4. Check if VM already has this store path (nix-store --check-validity)
  5. If not: push closure via nix-store --export | gcloud ssh | nix-store --import
  6. Activate binary and restart service

Note: nix-serve signs NARs on-the-fly via NIX_SECRET_KEY_FILE (configured
by nix-cache-setup.sh). No explicit signing step in the deploy script.
```

`deploy.sh` configures the VM's binary cache as an `extra-substituter` via
`--extra-substituters` CLI flags, so `nix build` can pull pre-built artifacts. The closure is
pushed to the VM via `nix-store --export/--import` over `gcloud compute ssh`
(not `nix copy`). On subsequent deploys from any machine with the cache
configured, `nix build` is a cache hit — only the push step runs.

### Client Configuration

To use the relay's binary cache on your local machine, add to `/etc/nix/nix.conf`
(or `~/.config/nix/nix.conf`):

```
substituters = https://cache.nixos.org http://i.q8.fyi:5000
trusted-public-keys = cache.nixos.org-1:6NCHdD59X431o0gWypbMrAURkbJ16ZPMQFGspcDShjY= <contents of deploy/relay/cache-key.pub>
```

Replace `<contents of deploy/relay/cache-key.pub>` with the single line from that file.

### Updating the Version

To deploy a new version of `iroh-relay`, update the flake input:

```bash
# Edit flake.nix to change the iroh input rev/tag, then:
nix flake update iroh-src

# Commit the updated flake.lock
git add flake.lock && git commit -m "chore: bump iroh-relay to <new-version>"

# Re-run deploy.sh — it will build the new version and push to cache
./deploy/relay/deploy.sh
```

### Rollback

Previous versions remain in `/nix/store` on the VM. To roll back:

```bash
# List recent iroh-relay store paths
gcloud compute ssh harmony-relay --zone=us-west1-b --command="
    ls -lt /nix/store | grep iroh-relay | head -5
"

# Activate an older store path directly
gcloud compute ssh harmony-relay --zone=us-west1-b --command="
    sudo systemctl stop iroh-relay &&
    sudo cp -f /nix/store/<old-path>/bin/iroh-relay /usr/local/bin/iroh-relay &&
    sudo systemctl start iroh-relay
"
```

## Runbook: Step-by-Step Manual Deployment

### 1. Create the VM

```bash
export GCP_PROJECT="your-gcp-project-id"
export RELAY_HOSTNAME="i.q8.fyi"

gcloud config set project "$GCP_PROJECT"

# Reserve a static IP
gcloud compute addresses create harmony-relay \
    --region=us-west1

# Note the IP for DNS
gcloud compute addresses describe harmony-relay \
    --region=us-west1 --format='get(address)'

# Create the VM
gcloud compute instances create harmony-relay \
    --zone=us-west1-b \
    --machine-type=e2-micro \
    --image-family=debian-12 \
    --image-project=debian-cloud \
    --boot-disk-size=10GB \
    --address=harmony-relay \
    --tags=harmony-relay \
    --metadata=startup-script='#!/bin/bash
        apt-get update -qq
        apt-get install -y -qq curl xz-utils'
```

### 2. Configure Firewall Rules

```bash
# HTTP (Let's Encrypt ACME challenge + relay fallback)
gcloud compute firewall-rules create harmony-relay-http \
    --allow=tcp:80 \
    --target-tags=harmony-relay \
    --description="Harmony relay HTTP"

# HTTPS (primary relay traffic)
gcloud compute firewall-rules create harmony-relay-https \
    --allow=tcp:443 \
    --target-tags=harmony-relay \
    --description="Harmony relay HTTPS"

# QUIC (NAT traversal / address discovery)
gcloud compute firewall-rules create harmony-relay-quic \
    --allow=udp:7842 \
    --target-tags=harmony-relay \
    --description="Harmony relay QUIC address discovery"
```

### 3. Point DNS

Add an A record with your registrar:

```
i.q8.fyi.  A  <static-ip-from-step-1>
```

Wait for propagation (check with `dig i.q8.fyi`).

### 4. Build and Upload iroh-relay via Nix

Run the cache setup first (see **Binary Cache** section above), then build and upload:

```bash
# Build iroh-relay using Nix (pulls from binary cache if available)
nix build .#iroh-relay-x86_64-linux

# Upload to the VM
gcloud compute scp result/bin/iroh-relay \
    harmony-relay:/tmp/iroh-relay --zone=us-west1-b
```

If you have a pre-built binary, set `LOCAL_BINARY` instead:

```bash
export LOCAL_BINARY=/path/to/iroh-relay
gcloud compute scp "$LOCAL_BINARY" \
    harmony-relay:/tmp/iroh-relay --zone=us-west1-b
```

### 5. Configure the Server

SSH into the VM:

```bash
gcloud compute ssh harmony-relay --zone=us-west1-b
```

Then on the VM, set your hostname:

```bash
export RELAY_HOSTNAME="i.q8.fyi"  # replace with your domain
```

Then on the VM:

```bash
# Install the binary
sudo mv /tmp/iroh-relay /usr/local/bin/iroh-relay
sudo chmod +x /usr/local/bin/iroh-relay

# Create service user
sudo useradd -r -s /bin/false iroh-relay

# Create directories
sudo mkdir -p /etc/iroh-relay /var/lib/iroh-relay/certs
sudo chown -R iroh-relay:iroh-relay /var/lib/iroh-relay

# Write config (replace RELAY_HOSTNAME with your domain)
sudo tee /etc/iroh-relay/config.toml << 'TOML'
enable_relay = true
enable_quic_addr_discovery = true
http_bind_addr = "[::]:80"

[tls]
https_bind_addr = "[::]:443"
quic_bind_addr = "[::]:7842"
cert_mode = "LetsEncrypt"
hostname = "RELAY_HOSTNAME_PLACEHOLDER"
contact = "admin@RELAY_HOSTNAME_PLACEHOLDER"
prod_tls = true
cert_dir = "/var/lib/iroh-relay/certs"

[limits.client.rx]
bytes_per_second = 102400
max_burst_bytes = 512000
TOML

# Replace placeholder with actual hostname
sudo sed -i "s/RELAY_HOSTNAME_PLACEHOLDER/$RELAY_HOSTNAME/g" \
    /etc/iroh-relay/config.toml

# Install systemd unit
sudo tee /etc/systemd/system/iroh-relay.service << 'UNIT'
[Unit]
Description=Harmony Relay Server (iroh-relay)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=iroh-relay
Group=iroh-relay
WorkingDirectory=/var/lib/iroh-relay
ExecStart=/usr/local/bin/iroh-relay --config-path /etc/iroh-relay/config.toml
Restart=always
RestartSec=5s
NoNewPrivileges=true
ProtectSystem=strict
ProtectHome=yes
PrivateTmp=yes
ReadWritePaths=/var/lib/iroh-relay
LimitNOFILE=65536
StandardOutput=journal
StandardError=journal
SyslogIdentifier=iroh-relay
Environment="RUST_LOG=info"

# Allow binding to privileged ports (80, 443)
AmbientCapabilities=CAP_NET_BIND_SERVICE
CapabilityBoundingSet=CAP_NET_BIND_SERVICE

[Install]
WantedBy=multi-user.target
UNIT

# Enable and start
sudo systemctl daemon-reload
sudo systemctl enable iroh-relay
sudo systemctl start iroh-relay

# Check status
sudo systemctl status iroh-relay
sudo journalctl -u iroh-relay -f
```

### 6. Verify

From your local machine:

```bash
# Check HTTPS is responding
curl -s -o /dev/null -w "%{http_code}" https://i.q8.fyi/

# Check the relay is accepting connections (from a harmony-node)
cargo run -p harmony-node -- --relay-url https://i.q8.fyi
```

## Connecting Harmony Nodes

Once the relay is running, nodes connect with:

```bash
harmony-node --relay-url https://i.q8.fyi
```

Or in a TOML config file:

```toml
relay_url = "https://i.q8.fyi"
```

The relay is optional — nodes on the same LAN communicate directly
via mDNS without any relay.

## Monitoring

View logs:

```bash
gcloud compute ssh harmony-relay --zone=us-west1-b -- \
    sudo journalctl -u iroh-relay -f
```

Check service health:

```bash
gcloud compute ssh harmony-relay --zone=us-west1-b -- \
    sudo systemctl status iroh-relay
```

## Updating

See the **Binary Cache → Updating the Version** section above for the Nix-based
update workflow. For a quick in-place update without Nix:

```bash
# Build or obtain the new binary, then upload and restart
gcloud compute scp /path/to/new/iroh-relay \
    harmony-relay:/tmp/iroh-relay --zone=us-west1-b

gcloud compute ssh harmony-relay --zone=us-west1-b -- \
    'sudo systemctl stop iroh-relay && \
     sudo mv /tmp/iroh-relay /usr/local/bin/iroh-relay && \
     sudo chmod +x /usr/local/bin/iroh-relay && \
     sudo systemctl start iroh-relay'
```

## Cost

- **e2-micro in us-west1:** Free tier eligible (1 instance per billing account)
- **Static IP:** Free while attached to a running instance
- **Egress:** 1 GB/month free, then $0.12/GB
- **Expected monthly cost:** $0 for light usage (< 1 GB relay traffic)

## Future Improvements

- Rate-limiting sidecar with UCAN-based tiered access
- N+2 cluster behind a load balancer for high availability
- Prometheus metrics collection
- Automated binary builds via CI
