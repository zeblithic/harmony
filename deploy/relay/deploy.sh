#!/usr/bin/env bash
# deploy.sh — Provision and configure a Harmony relay server on GCP.
#
# Usage:
#   RELAY_HOSTNAME="i.q8.fyi" GCP_PROJECT="my-project" ./deploy/relay/deploy.sh
#
# Prerequisites:
#   - gcloud CLI authenticated (gcloud auth login)
#   - A GCP project with billing enabled
#   - A domain with DNS you control
#
# The script builds iroh-relay locally if Rust is available (fast, ~2 min),
# otherwise falls back to building on the remote VM (slow, ~15 min on
# e2-micro). Set LOCAL_BINARY to skip building entirely:
#
#   LOCAL_BINARY=/path/to/iroh-relay ./deploy/relay/deploy.sh

set -euo pipefail

# ── Configuration ─────────────────────────────────────────────────
RELAY_HOSTNAME="${RELAY_HOSTNAME:?Set RELAY_HOSTNAME (e.g. i.q8.fyi)}"
GCP_PROJECT="${GCP_PROJECT:?Set GCP_PROJECT (e.g. my-gcp-project)}"
GCP_ZONE="${GCP_ZONE:-us-west1-b}"
# Derive region from zone (strip trailing -a/-b/-c) to prevent mismatch.
GCP_REGION="${GCP_ZONE%-*}"
VM_NAME="${VM_NAME:-harmony-relay}"
MACHINE_TYPE="${MACHINE_TYPE:-e2-micro}"
IROH_REPO="${IROH_REPO:-https://github.com/n0-computer/iroh.git}"
IROH_VERSION="${IROH_VERSION:-v0.91.2}"
CONTACT_EMAIL="${CONTACT_EMAIL:-admin@${RELAY_HOSTNAME}}"
LOCAL_BINARY="${LOCAL_BINARY:-}"

# Rate limits (bytes/sec). Defaults: 100 KB/s sustained, 500 KB burst.
RATE_LIMIT_BPS="${RATE_LIMIT_BPS:-102400}"
RATE_LIMIT_BURST="${RATE_LIMIT_BURST:-512000}"

echo "=== Harmony Relay Deployment ==="
echo "  Hostname:     ${RELAY_HOSTNAME}"
echo "  Project:      ${GCP_PROJECT}"
echo "  Zone:         ${GCP_ZONE} (region: ${GCP_REGION})"
echo "  VM:           ${VM_NAME} (${MACHINE_TYPE})"
echo "  iroh version: ${IROH_VERSION}"
echo ""

gcloud config set project "$GCP_PROJECT" --quiet

# ── Step 1: Reserve static IP ─────────────────────────────────────
echo "--- Step 1: Reserving static IP..."
if gcloud compute addresses describe "$VM_NAME" --region="$GCP_REGION" &>/dev/null; then
    echo "    Static IP '$VM_NAME' already exists."
else
    gcloud compute addresses create "$VM_NAME" --region="$GCP_REGION"
fi

STATIC_IP=$(gcloud compute addresses describe "$VM_NAME" \
    --region="$GCP_REGION" --format='get(address)')
echo "    Static IP: ${STATIC_IP}"

# ── Step 2: Create firewall rules ─────────────────────────────────
echo "--- Step 2: Configuring firewall rules..."
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
echo "--- Step 3: Creating VM..."
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

# ── Step 4: Build iroh-relay ──────────────────────────────────────
# Three strategies, in order of preference:
#   1. Use a pre-built binary (LOCAL_BINARY env var)
#   2. Nix cross-compile (works from any host, ~5 min)
#   3. Build on the VM (slow fallback, ~15 min on e2-micro)

BINARY_PATH=""
BUILD_STRATEGY=""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

if [ -n "$LOCAL_BINARY" ] && [ -f "$LOCAL_BINARY" ]; then
    echo "--- Step 4: Using pre-built binary: ${LOCAL_BINARY}"
    BINARY_PATH="$LOCAL_BINARY"
    BUILD_STRATEGY="prebuilt"

elif command -v nix &>/dev/null; then
    echo "--- Step 4: Cross-compiling iroh-relay via Nix..."
    STORE_PATH=$(nix build "${REPO_ROOT}#iroh-relay-x86_64-linux" \
        --extra-experimental-features "nix-command flakes" \
        --print-out-paths --no-link | head -1)
    if [ -z "$STORE_PATH" ]; then
        echo "ERROR: nix build returned empty store path"
        exit 1
    fi
    BINARY_PATH="${STORE_PATH}/bin/iroh-relay"
    BUILD_STRATEGY="nix"
    echo "    Nix build complete: ${STORE_PATH}"

else
    echo "--- Step 4: No Nix found — building on VM (slow, ~15 min on e2-micro)..."

    BUILD_STRATEGY="remote"

    # Install build deps on VM
    echo "    Installing build dependencies..."
    gcloud compute ssh "$VM_NAME" --zone="$GCP_ZONE" --command="
        set -euo pipefail
        if [ ! -f \"\$HOME/.cargo/bin/cargo\" ]; then
            sudo apt-get update -qq > /dev/null
            sudo apt-get install -y -qq build-essential pkg-config libssl-dev curl git > /dev/null
            curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y > /dev/null 2>&1
        fi
        [ -f \"\$HOME/.cargo/env\" ] && . \"\$HOME/.cargo/env\"
        cargo --version
    "

    # Build on VM
    echo "    Building iroh-relay on VM..."
    gcloud compute ssh "$VM_NAME" --zone="$GCP_ZONE" --command="
        set -euo pipefail
        [ -f \"\$HOME/.cargo/env\" ] && . \"\$HOME/.cargo/env\"
        if [ ! -d /tmp/iroh-build ]; then
            git clone --depth=1 --branch '${IROH_VERSION}' '${IROH_REPO}' /tmp/iroh-build
        else
            cd /tmp/iroh-build
            git fetch --depth=1 origin '${IROH_VERSION}'
            git checkout FETCH_HEAD
        fi
        cd /tmp/iroh-build
        cargo build --release -p iroh-relay --features server --bin iroh-relay 2>&1 | tail -3
        echo 'Remote build complete.'
    "
fi

# ── Step 5: Upload binary and install service ─────────────────────
echo "--- Step 5: Installing iroh-relay service..."

# Upload binary if we built locally or have a pre-built one
if [ -n "$BINARY_PATH" ]; then
    echo "    Uploading binary to VM..."
    gcloud compute scp "$BINARY_PATH" "$VM_NAME:/tmp/iroh-relay" --zone="$GCP_ZONE"
    gcloud compute ssh "$VM_NAME" --zone="$GCP_ZONE" --command="
        sudo mv -f /tmp/iroh-relay /usr/local/bin/iroh-relay
        sudo chmod +x /usr/local/bin/iroh-relay
    "
elif [ "$BUILD_STRATEGY" = "remote" ]; then
    # Install from remote build directory
    gcloud compute ssh "$VM_NAME" --zone="$GCP_ZONE" --command="
        sudo cp -f /tmp/iroh-build/target/release/iroh-relay /usr/local/bin/iroh-relay
        sudo chmod +x /usr/local/bin/iroh-relay
    "
fi

gcloud compute ssh "$VM_NAME" --zone="$GCP_ZONE" --command="
    set -euo pipefail

    # Verify binary exists
    /usr/local/bin/iroh-relay --help > /dev/null 2>&1 || {
        echo 'ERROR: iroh-relay binary not found or not executable.'
        exit 1
    }

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

    # Enable but do NOT start yet — ACME needs DNS to resolve first.
    sudo systemctl daemon-reload
    sudo systemctl enable iroh-relay

    echo 'Service installed and enabled.'
"

# ── Step 6: DNS check and service start ───────────────────────────
echo ""
echo "=== VM provisioned. Static IP: ${STATIC_IP} ==="
echo ""
echo "DNS A record needed:"
echo "    ${RELAY_HOSTNAME}.  A  ${STATIC_IP}"
echo ""

# Check if DNS already resolves (user may have set it up in advance)
resolved=$(dig +short "${RELAY_HOSTNAME}" @8.8.8.8 2>/dev/null | head -1)
if [ "$resolved" = "$STATIC_IP" ]; then
    echo "DNS already resolves correctly. Starting service..."
    gcloud compute ssh "$VM_NAME" --zone="$GCP_ZONE" --command="sudo systemctl start iroh-relay"
    echo ""
    echo "=== Deployment Complete ==="
    echo "Relay running at https://${RELAY_HOSTNAME}"
    echo ""
    echo "Connect nodes:  harmony-node --relay-url https://${RELAY_HOSTNAME}"
    echo "View logs:      gcloud compute ssh ${VM_NAME} --zone=${GCP_ZONE} -- sudo journalctl -u iroh-relay -f"
    exit 0
fi

# Interactive mode: wait for DNS
if [ -t 0 ]; then
    read -rp "Press Enter after you've added the DNS record (or Ctrl-C to finish later)..."

    echo "--- Waiting for DNS to resolve ${RELAY_HOSTNAME} to ${STATIC_IP}..."
    dns_ready=0
    for i in $(seq 1 60); do
        resolved=$(dig +short "${RELAY_HOSTNAME}" @8.8.8.8 2>/dev/null | head -1)
        if [ "$resolved" = "$STATIC_IP" ]; then
            dns_ready=1
            break
        fi
        echo "    Attempt $i/60: got '${resolved:-<empty>}', waiting..."
        sleep 10
    done

    if [ "$dns_ready" -eq 1 ]; then
        echo "    DNS resolved! Starting iroh-relay..."
        gcloud compute ssh "$VM_NAME" --zone="$GCP_ZONE" --command="sudo systemctl start iroh-relay"
        echo ""
        echo "=== Deployment Complete ==="
        echo "Relay running at https://${RELAY_HOSTNAME}"
    else
        echo ""
        echo "WARNING: DNS did not resolve after 10 minutes."
        echo "Start manually after DNS propagates:"
        echo "    gcloud compute ssh ${VM_NAME} --zone=${GCP_ZONE} -- sudo systemctl start iroh-relay"
    fi
else
    # Non-interactive: just print instructions
    echo "Non-interactive mode. After adding the DNS record, start the service:"
    echo "    gcloud compute ssh ${VM_NAME} --zone=${GCP_ZONE} -- sudo systemctl start iroh-relay"
fi

echo ""
echo "Connect nodes:  harmony-node --relay-url https://${RELAY_HOSTNAME}"
echo "View logs:      gcloud compute ssh ${VM_NAME} --zone=${GCP_ZONE} -- sudo journalctl -u iroh-relay -f"
