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

**Manual runbook (Step 4 builds locally):**
- All of the above, plus a local Rust toolchain

### Automated Deployment

```bash
# Set your hostname and GCP project
export RELAY_HOSTNAME="i.q8.fyi"
export GCP_PROJECT="your-gcp-project-id"

# Run the deployment script
./deploy/relay/deploy.sh
```

The script will:
1. Create a `e2-micro` VM in `us-west1-b` with Container-Optimized OS
2. Reserve a static external IP
3. Configure firewall rules (80/tcp, 443/tcp, 7842/udp)
4. Build and upload the `iroh-relay` binary
5. Install the systemd unit and config
6. Start the service

After the script completes, it prints the static IP. Point your DNS
A record at that IP:

```
i.q8.fyi.  A  <static-ip>
```

### Manual Deployment

See the step-by-step runbook below.

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
        apt-get install -y -qq build-essential pkg-config libssl-dev'
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

### 4. Build and Upload iroh-relay

On your local machine (or on the VM):

```bash
# Clone iroh and build the relay binary
git clone https://github.com/n0-computer/iroh.git /tmp/iroh-build
cd /tmp/iroh-build
cargo build --release -p iroh-relay --features server --bin iroh-relay

# Upload to the VM
gcloud compute scp /tmp/iroh-build/target/release/iroh-relay \
    harmony-relay:/tmp/iroh-relay --zone=us-west1-b
```

### 5. Configure the Server

SSH into the VM:

```bash
gcloud compute ssh harmony-relay --zone=us-west1-b
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

```bash
# Build new version locally
cd /tmp/iroh-build && git pull && cargo build --release -p iroh-relay --features server

# Upload and restart
gcloud compute scp /tmp/iroh-build/target/release/iroh-relay \
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
