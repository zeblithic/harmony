#!/usr/bin/env bash
# deploy.sh — Provision and configure a Harmony relay server on GCP.
#
# Usage:
#   RELAY_HOSTNAME="i.q8.fyi" GCP_PROJECT="my-project" ./deploy.sh
#
# Prerequisites:
#   - gcloud CLI authenticated (gcloud auth login)
#   - A GCP project with billing enabled
#   - A domain with DNS you control
#   - DNS: After this script runs, point RELAY_HOSTNAME's A record
#     at the printed static IP address.
#
# Note: Rust is installed on the remote VM automatically — no local
# Rust toolchain needed for automated deployment.

set -euo pipefail

# ── Configuration ─────────────────────────────────────────────────
RELAY_HOSTNAME="${RELAY_HOSTNAME:?Set RELAY_HOSTNAME (e.g. i.q8.fyi)}"
GCP_PROJECT="${GCP_PROJECT:?Set GCP_PROJECT (e.g. my-gcp-project)}"
GCP_ZONE="${GCP_ZONE:-us-west1-b}"
GCP_REGION="${GCP_REGION:-us-west1}"
VM_NAME="${VM_NAME:-harmony-relay}"
MACHINE_TYPE="${MACHINE_TYPE:-e2-micro}"
IROH_REPO="${IROH_REPO:-https://github.com/n0-computer/iroh.git}"
IROH_VERSION="${IROH_VERSION:-v0.35.0}"
CONTACT_EMAIL="${CONTACT_EMAIL:-admin@${RELAY_HOSTNAME}}"

# Rate limits (bytes/sec). Defaults: 100 KB/s sustained, 500 KB burst.
RATE_LIMIT_BPS="${RATE_LIMIT_BPS:-102400}"
RATE_LIMIT_BURST="${RATE_LIMIT_BURST:-512000}"

echo "=== Harmony Relay Deployment ==="
echo "  Hostname:  ${RELAY_HOSTNAME}"
echo "  Project:   ${GCP_PROJECT}"
echo "  Zone:      ${GCP_ZONE}"
echo "  VM:        ${VM_NAME} (${MACHINE_TYPE})"
echo ""

gcloud config set project "$GCP_PROJECT" --quiet

# ── Step 1: Reserve static IP ─────────────────────────────────────
echo "--- Reserving static IP..."
if gcloud compute addresses describe "$VM_NAME" --region="$GCP_REGION" &>/dev/null; then
    echo "    Static IP '$VM_NAME' already exists."
else
    gcloud compute addresses create "$VM_NAME" --region="$GCP_REGION"
fi

STATIC_IP=$(gcloud compute addresses describe "$VM_NAME" \
    --region="$GCP_REGION" --format='get(address)')
echo "    Static IP: ${STATIC_IP}"

# ── Step 2: Create firewall rules ─────────────────────────────────
echo "--- Configuring firewall rules..."
for rule in \
    "${VM_NAME}-http:tcp:80:HTTP" \
    "${VM_NAME}-https:tcp:443:HTTPS" \
    "${VM_NAME}-quic:udp:7842:QUIC address discovery"; do
    IFS=: read -r name proto port desc <<< "$rule"
    if gcloud compute firewall-rules describe "$name" &>/dev/null; then
        echo "    Rule '$name' already exists."
    else
        gcloud compute firewall-rules create "$name" \
            --allow="${proto}:${port}" \
            --target-tags="$VM_NAME" \
            --description="Harmony relay ${desc}" \
            --quiet
        echo "    Created rule: $name (${proto}:${port})"
    fi
done

# ── Step 3: Create VM ─────────────────────────────────────────────
echo "--- Creating VM..."
if gcloud compute instances describe "$VM_NAME" --zone="$GCP_ZONE" &>/dev/null; then
    echo "    VM '$VM_NAME' already exists."
else
    gcloud compute instances create "$VM_NAME" \
        --zone="$GCP_ZONE" \
        --machine-type="$MACHINE_TYPE" \
        --image-family=debian-12 \
        --image-project=debian-cloud \
        --boot-disk-size=10GB \
        --address="$VM_NAME" \
        --tags="$VM_NAME" \
        --quiet
    echo "    Created VM: $VM_NAME"

    echo "    Waiting for SSH to become available..."
    ssh_ready=0
    for i in $(seq 1 30); do
        if gcloud compute ssh "$VM_NAME" --zone="$GCP_ZONE" \
            --command="echo ready" &>/dev/null; then
            ssh_ready=1
            break
        fi
        sleep 5
    done
    if [ "$ssh_ready" -eq 0 ]; then
        echo "ERROR: VM SSH not available after 150s. Check VM logs:"
        echo "  gcloud compute instances get-serial-port-output $VM_NAME --zone=$GCP_ZONE"
        exit 1
    fi
fi

# ── Step 4: Install build tools on VM ─────────────────────────────
echo "--- Installing build dependencies on VM..."
gcloud compute ssh "$VM_NAME" --zone="$GCP_ZONE" --command="
    set -euo pipefail
    if [ ! -f \"\$HOME/.cargo/bin/cargo\" ]; then
        sudo apt-get update -qq
        sudo apt-get install -y -qq build-essential pkg-config libssl-dev curl git
        curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    fi
    source \"\$HOME/.cargo/env\"
    echo 'Build tools ready.'
"

# ── Step 5: Build iroh-relay on VM ────────────────────────────────
echo "--- Building iroh-relay on VM (this takes a few minutes)..."
gcloud compute ssh "$VM_NAME" --zone="$GCP_ZONE" --command="
    set -euo pipefail
    source \"\$HOME/.cargo/env\"
    if [ ! -d /tmp/iroh-build ]; then
        git clone --depth=1 --branch '${IROH_VERSION}' '${IROH_REPO}' /tmp/iroh-build
    else
        cd /tmp/iroh-build
        git fetch --depth=1 origin '${IROH_VERSION}'
        git checkout FETCH_HEAD
    fi
    cd /tmp/iroh-build
    cargo build --release -p iroh-relay --features server --bin iroh-relay
    echo 'Build complete.'
"

# ── Step 6: Install and configure ─────────────────────────────────
echo "--- Installing iroh-relay service..."
gcloud compute ssh "$VM_NAME" --zone="$GCP_ZONE" --command="
    set -euo pipefail

    # Install binary
    sudo cp -f /tmp/iroh-build/target/release/iroh-relay /usr/local/bin/iroh-relay
    sudo chmod +x /usr/local/bin/iroh-relay

    # Create service user (idempotent)
    id -u iroh-relay &>/dev/null || sudo useradd -r -s /bin/false iroh-relay

    # Create directories
    sudo mkdir -p /etc/iroh-relay /var/lib/iroh-relay/certs
    sudo chown -R iroh-relay:iroh-relay /var/lib/iroh-relay

    # Write config
    sudo tee /etc/iroh-relay/config.toml > /dev/null << TOML
enable_relay = true
enable_quic_addr_discovery = true
http_bind_addr = \"[::]:80\"

[tls]
https_bind_addr = \"[::]:443\"
quic_bind_addr = \"[::]:7842\"
cert_mode = \"LetsEncrypt\"
hostname = \"${RELAY_HOSTNAME}\"
contact = \"${CONTACT_EMAIL}\"
prod_tls = true
cert_dir = \"/var/lib/iroh-relay/certs\"

[limits.client.rx]
bytes_per_second = ${RATE_LIMIT_BPS}
max_burst_bytes = ${RATE_LIMIT_BURST}
TOML

    # Install systemd unit
    sudo tee /etc/systemd/system/iroh-relay.service > /dev/null << 'UNIT'
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
Environment=RUST_LOG=info
AmbientCapabilities=CAP_NET_BIND_SERVICE
CapabilityBoundingSet=CAP_NET_BIND_SERVICE

[Install]
WantedBy=multi-user.target
UNIT

    # Enable and start
    sudo systemctl daemon-reload
    sudo systemctl enable iroh-relay
    sudo systemctl restart iroh-relay

    echo 'Service installed and started.'
"

# ── Done ──────────────────────────────────────────────────────────
echo ""
echo "=== Deployment Complete ==="
echo ""
echo "Static IP: ${STATIC_IP}"
echo ""
echo "NEXT STEP: Add this DNS A record with your registrar:"
echo ""
echo "    ${RELAY_HOSTNAME}.  A  ${STATIC_IP}"
echo ""
echo "After DNS propagates, verify with:"
echo "    curl -s -o /dev/null -w '%{http_code}' https://${RELAY_HOSTNAME}/"
echo ""
echo "Connect Harmony nodes with:"
echo "    harmony-node --relay-url https://${RELAY_HOSTNAME}"
echo ""
echo "View logs:"
echo "    gcloud compute ssh ${VM_NAME} --zone=${GCP_ZONE} -- sudo journalctl -u iroh-relay -f"
