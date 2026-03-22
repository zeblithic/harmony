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
# The script uses Nix to build iroh-relay as a cross-compiled static musl
# binary, then pushes the Nix closure to the VM.  Set LOCAL_BINARY to skip
# building entirely:
#
#   LOCAL_BINARY=/path/to/iroh-relay ./deploy/relay/deploy.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# ── Configuration ─────────────────────────────────────────────────
RELAY_HOSTNAME="${RELAY_HOSTNAME:?Set RELAY_HOSTNAME (e.g. i.q8.fyi)}"
GCP_PROJECT="${GCP_PROJECT:?Set GCP_PROJECT (e.g. my-gcp-project)}"
GCP_ZONE="${GCP_ZONE:-us-west1-b}"
# Derive region from zone (strip trailing -a/-b/-c) to prevent mismatch.
GCP_REGION="${GCP_ZONE%-*}"
VM_NAME="${VM_NAME:-harmony-relay}"
MACHINE_TYPE="${MACHINE_TYPE:-e2-micro}"
CONTACT_EMAIL="${CONTACT_EMAIL:-admin@${RELAY_HOSTNAME}}"
LOCAL_BINARY="${LOCAL_BINARY:-}"

# Rate limits (bytes/sec). Defaults: 100 KB/s sustained, 500 KB burst.
RATE_LIMIT_BPS="${RATE_LIMIT_BPS:-102400}"
RATE_LIMIT_BURST="${RATE_LIMIT_BURST:-512000}"

# ── Input validation ──────────────────────────────────────────────
# Prevent shell injection via SSH commands and TOML config corruption.
# RFC 952: labels start/end with alphanumeric, hyphens only in the middle.
# No underscores — they cause ACME/Let's Encrypt certificate failures.
[[ "${RELAY_HOSTNAME}" =~ ^[a-zA-Z0-9]([a-zA-Z0-9-]*[a-zA-Z0-9])?(\.[a-zA-Z0-9]([a-zA-Z0-9-]*[a-zA-Z0-9])?)+$ ]] \
    || { echo "ERROR: RELAY_HOSTNAME must be a valid hostname (RFC 952: alphanumeric labels separated by dots, no leading/trailing hyphens)"; exit 1; }
[[ "${CONTACT_EMAIL}" =~ ^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9._-]+\.[a-zA-Z]{2,}$ ]] \
    || { echo "ERROR: CONTACT_EMAIL must be a valid email address (user@domain.tld)"; exit 1; }
# Validate LOCAL_BINARY early: if set but missing, fail now.
if [ -n "${LOCAL_BINARY}" ] && [ ! -f "${LOCAL_BINARY}" ]; then
    echo "ERROR: LOCAL_BINARY set to '${LOCAL_BINARY}' but file does not exist"
    exit 1
fi
# IROH_VERSION and IROH_REPO removed — version is controlled by flake.nix.
# TOML forbids leading zeros in integers — match (0|[1-9][0-9]*).
[[ "${RATE_LIMIT_BPS}" =~ ^(0|[1-9][0-9]*)$ ]] \
    || { echo "ERROR: RATE_LIMIT_BPS must be a non-negative integer (no leading zeros)"; exit 1; }
[[ "${RATE_LIMIT_BURST}" =~ ^(0|[1-9][0-9]*)$ ]] \
    || { echo "ERROR: RATE_LIMIT_BURST must be a non-negative integer (no leading zeros)"; exit 1; }

echo "=== Harmony Relay Deployment ==="
echo "  Hostname:     ${RELAY_HOSTNAME}"
echo "  Project:      ${GCP_PROJECT}"
echo "  Zone:         ${GCP_ZONE} (region: ${GCP_REGION})"
echo "  VM:           ${VM_NAME} (${MACHINE_TYPE})"
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
    "${VM_NAME}-quic:udp:7842:QUIC address discovery" \
    "${VM_NAME}-nix-cache:tcp:5000:Nix binary cache"; do
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
# Two strategies, in order of preference:
#   1. Use a pre-built binary (LOCAL_BINARY env var)
#   2. Nix build locally, push closure to VM

BINARY_PATH=""
BUILD_STRATEGY=""

if [ -n "$LOCAL_BINARY" ] && [ -f "$LOCAL_BINARY" ]; then
    echo "--- Step 4: Using pre-built binary: ${LOCAL_BINARY}"
    BINARY_PATH="$LOCAL_BINARY"
    BUILD_STRATEGY="prebuilt"

elif command -v nix &>/dev/null; then
    echo "--- Step 4: Building iroh-relay via Nix..."

    # gcloud compute ssh runs a non-login, non-interactive shell where
    # /etc/profile.d/nix.sh is never sourced. Prefix all remote commands
    # with Nix PATH setup so nix-store is found (NixOS/nix#2587).
    NIX_SSH_PREFIX='export PATH="/nix/var/nix/profiles/default/bin:$PATH";'
    # Absolute path for sudo calls — Debian's secure_path doesn't include Nix.
    NIX_STORE_BIN="/nix/var/nix/profiles/default/bin/nix-store"

    # Verify Nix is installed on the VM BEFORE building locally (~2 min).
    # On a brand-new VM, auto-run nix-cache-setup.sh to install Nix + nix-serve.
    if ! gcloud compute ssh "$VM_NAME" --zone="$GCP_ZONE" \
        --command="${NIX_SSH_PREFIX} command -v nix-store" &>/dev/null; then
        echo "    Nix not found on ${VM_NAME} — running nix-cache-setup.sh..."
        SETUP_SCRIPT="${SCRIPT_DIR}/nix-cache-setup.sh"
        if [ ! -f "$SETUP_SCRIPT" ]; then
            echo "ERROR: ${SETUP_SCRIPT} not found."
            exit 1
        fi
        gcloud compute ssh "$VM_NAME" --zone="$GCP_ZONE" -- "bash -s" < "$SETUP_SCRIPT"
        echo "    Nix setup complete."
        echo ""
        echo "    ACTION REQUIRED: Copy the generated public key into the repo:"
        echo "      gcloud compute ssh ${VM_NAME} --zone=${GCP_ZONE} -- cat /etc/nix/cache-key.pub > ${SCRIPT_DIR}/cache-key.pub"
        echo "    Then commit cache-key.pub so future deploys can use the binary cache."
        echo ""
        # Verify the daemon is ready after setup before proceeding
        if ! gcloud compute ssh "$VM_NAME" --zone="$GCP_ZONE" \
            --command="${NIX_SSH_PREFIX} nix-store --version" &>/dev/null; then
            echo "ERROR: Nix daemon on ${VM_NAME} is not responding after setup."
            echo "  gcloud compute ssh ${VM_NAME} --zone=${GCP_ZONE} -- sudo journalctl -u nix-daemon -n 20"
            exit 1
        fi
    fi

    # Configure the VM's binary cache as a substituter so nix build can
    # pull pre-built artifacts instead of compiling. The public key is in
    # the repo; if it's a placeholder (comment-only), skip the substituter.
    CACHE_KEY_FILE="${SCRIPT_DIR}/cache-key.pub"
    EXTRA_NIX_ARGS=()
    # Extract the actual key line (strip comments and blank lines)
    CACHE_PUB_KEY=""
    if [ -f "$CACHE_KEY_FILE" ] && [ -s "$CACHE_KEY_FILE" ]; then
        # || true: grep exits 1 when no lines match (e.g. comment-only placeholder).
        # Under set -euo pipefail, this would abort the script.
        CACHE_PUB_KEY=$(grep -v '^#' "$CACHE_KEY_FILE" | grep -v '^$' | head -1 || true)
    fi
    if [ -n "$CACHE_PUB_KEY" ]; then
        # Best-effort: ask nix build to check the relay's binary cache.
        # In multi-user Nix, extra-substituters from non-trusted users are
        # silently ignored by the daemon. If the cache isn't consulted, the
        # build compiles from source locally (~2 min) and the push path
        # handles getting it to the VM. To enable the cache shortcut, add
        # your username to trusted-users in /etc/nix/nix.conf:
        #   trusted-users = root @wheel <your-username>
        EXTRA_NIX_ARGS+=(
            --extra-substituters "http://${RELAY_HOSTNAME}:5000"
            # Note: on first deploy DNS may not resolve yet; the substituter miss is harmless.
            --extra-trusted-public-keys "$CACHE_PUB_KEY"
        )
        echo "    Binary cache configured (requires trusted-users for cache hits)"
    fi

    # Build locally (cross-compiles to x86_64-linux-musl on any host).
    # If the binary cache has this derivation, nix build fetches instead of compiling.
    # --extra-experimental-features ensures flakes work even if the developer hasn't
    # enabled them globally in ~/.config/nix/nix.conf.
    echo "    Running: nix build .#iroh-relay-x86_64-linux (from ${REPO_ROOT})"
    STORE_PATH=$(nix build "${REPO_ROOT}#iroh-relay-x86_64-linux" \
        ${EXTRA_NIX_ARGS[@]+"${EXTRA_NIX_ARGS[@]}"} \
        --extra-experimental-features "nix-command flakes" \
        --print-out-paths --no-link)
    if [ -z "$STORE_PATH" ]; then
        echo "ERROR: nix build returned empty store path; check that the flake target exists"
        exit 1
    fi
    echo "    Store path: ${STORE_PATH}"

    # Check if VM already has this exact store path (fast path).
    # Uses nix-store (stable CLI) instead of nix path-info (requires experimental nix-command).
    if gcloud compute ssh "$VM_NAME" --zone="$GCP_ZONE" \
        --command="${NIX_SSH_PREFIX} nix-store --check-validity '$STORE_PATH'" &>/dev/null; then
        echo "    VM already has this store path — skipping push."
    else
        echo "    Pushing Nix closure to VM..."
        # Collect all closure paths into an array first, then export in a
        # single nix-store --export call. xargs would split into multiple
        # invocations for large closures, and nix-store --import only reads
        # one export stream — subsequent streams would be silently dropped.
        # Uses while-read loop (bash 3.2 compatible) instead of mapfile (bash 4+).
        CLOSURE=()
        while IFS= read -r path; do
            CLOSURE+=("$path")
        done < <(nix-store -qR "$STORE_PATH")
        if [ ${#CLOSURE[@]} -eq 0 ]; then
            echo "ERROR: nix-store -qR returned no paths for ${STORE_PATH}"
            exit 1
        fi
        nix-store --export "${CLOSURE[@]}" | \
            gcloud compute ssh "$VM_NAME" --zone="$GCP_ZONE" -- \
            "sudo ${NIX_STORE_BIN} --import"
        echo "    Closure pushed."
    fi

    # No explicit signing step needed — nix-serve signs NARs on-the-fly
    # when NIX_SECRET_KEY_FILE is set in its systemd unit (configured by
    # nix-cache-setup.sh). This avoids the nix-store --sign / nix store sign
    # CLI compatibility issue entirely.

    BUILD_STRATEGY="nix"
    echo "    Nix build strategy complete."

else
    echo "ERROR: Nix is not installed. Install Nix or set LOCAL_BINARY."
    exit 1
fi

# ── Step 5: Upload binary and install service ─────────────────────
echo "--- Step 5: Installing iroh-relay service..."

# Install binary on the VM
if [ -n "$BINARY_PATH" ]; then
    # Prebuilt: SCP + mv
    echo "    Uploading binary to VM..."
    gcloud compute scp "$BINARY_PATH" "$VM_NAME:/tmp/iroh-relay" --zone="$GCP_ZONE"
    gcloud compute ssh "$VM_NAME" --zone="$GCP_ZONE" --command="
        sudo mv -f /tmp/iroh-relay /usr/local/bin/iroh-relay
        sudo chmod +x /usr/local/bin/iroh-relay
    "
elif [ "$BUILD_STRATEGY" = "nix" ]; then
    # Nix: copy from store path on VM
    echo "    Installing from Nix store path on VM..."
    gcloud compute ssh "$VM_NAME" --zone="$GCP_ZONE" --command="
        sudo cp -f '${STORE_PATH}/bin/iroh-relay' /usr/local/bin/iroh-relay
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
