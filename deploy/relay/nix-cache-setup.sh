#!/usr/bin/env bash
# nix-cache-setup.sh — One-time setup for Nix binary cache on relay VM.
#
# Usage (from local machine):
#   gcloud compute ssh VM_NAME --zone=ZONE -- "bash -s" < deploy/relay/nix-cache-setup.sh
#
# Idempotent — safe to re-run.

set -euo pipefail

# ---------------------------------------------------------------------------
# 1. Install multi-user Nix (with daemon) if not present
# ---------------------------------------------------------------------------
if ! command -v nix &>/dev/null; then
  echo "[nix-cache-setup] Installing multi-user Nix..."
  # Multi-user Nix installer requires root for daemon setup
  curl -fsSL https://nixos.org/nix/install | sudo sh -s -- --daemon --yes
else
  echo "[nix-cache-setup] Nix already installed — skipping."
fi

# Always source Nix profile — gcloud ssh runs a non-login shell so
# /etc/profile.d/ is never loaded automatically. This is needed on
# both first install AND re-runs.
if [ -e /etc/profile.d/nix.sh ]; then
  # shellcheck source=/dev/null
  . /etc/profile.d/nix.sh
elif [ -e /nix/var/nix/profiles/default/etc/profile.d/nix-daemon.sh ]; then
  # shellcheck source=/dev/null
  . /nix/var/nix/profiles/default/etc/profile.d/nix-daemon.sh
fi

# Ensure the nix daemon is running before proceeding
if command -v systemctl &>/dev/null; then
  sudo systemctl enable --now nix-daemon.service 2>/dev/null || true
fi

# ---------------------------------------------------------------------------
# 2. Generate binary cache signing keypair if not present
# ---------------------------------------------------------------------------
KEY_DIR=/etc/nix
PRIVATE_KEY="${KEY_DIR}/cache-key.pem"
PUBLIC_KEY="${KEY_DIR}/cache-key.pub"

if [ ! -f "${PRIVATE_KEY}" ]; then
  echo "[nix-cache-setup] Generating signing keypair..."
  sudo install -d -m 755 "${KEY_DIR}"
  # nix-store --generate-binary-cache-key produces:
  #   <name>.pem  — private key (keep secret)
  #   <name>.pub  — public key  (add to clients' nix.conf trusted-public-keys)
  # The key name is embedded in signatures, so use a stable host-based name.
  KEY_NAME="$(hostname -f)-cache-1"
  sudo nix-store --generate-binary-cache-key "${KEY_NAME}" "${PRIVATE_KEY}" "${PUBLIC_KEY}"
  sudo chmod 600 "${PRIVATE_KEY}"
  sudo chmod 644 "${PUBLIC_KEY}"
  echo "[nix-cache-setup] Signing keypair written to ${KEY_DIR}."
else
  echo "[nix-cache-setup] Signing keypair already exists — skipping."
fi

# ---------------------------------------------------------------------------
# 3. Install nix-serve from nixpkgs
# ---------------------------------------------------------------------------
if ! command -v nix-serve &>/dev/null; then
  echo "[nix-cache-setup] Installing nix-serve..."
  # Ensure the current user has the nixpkgs channel (multi-user installer
  # only adds it for root; gcloud ssh connects as a regular user).
  if ! nix-channel --list 2>/dev/null | grep -q nixpkgs; then
    # Use stable channel for production infrastructure (not nixpkgs-unstable)
    nix-channel --add https://nixos.org/channels/nixos-24.11 nixpkgs
  fi
  nix-channel --update nixpkgs
  nix-env -iA nixpkgs.nix-serve
else
  echo "[nix-cache-setup] nix-serve already installed — skipping."
fi

# Resolve through all symlinks to get the canonical /nix/store/... path.
# This is critical: ProtectHome=yes makes ~/.nix-profile inaccessible to
# the service process, so ExecStart must point to the store path directly.
NIX_SERVE_BIN="$(readlink -f "$(command -v nix-serve)")"

# ---------------------------------------------------------------------------
# 4. Create systemd service for nix-serve on port 5000
# ---------------------------------------------------------------------------
SERVICE_FILE=/etc/systemd/system/nix-serve.service

# Create a dedicated unprivileged user for nix-serve.
# nix-serve only needs read access to /nix/store (world-readable)
# and read access to the signing key.
if ! id -u nix-serve &>/dev/null; then
    sudo useradd -r -s /bin/false -d /nonexistent nix-serve
    echo "[nix-cache-setup] Created nix-serve user."
fi
# Grant nix-serve group read on the signing key
sudo chgrp nix-serve "${PRIVATE_KEY}" 2>/dev/null || true
sudo chmod 640 "${PRIVATE_KEY}" 2>/dev/null || true

sudo tee "${SERVICE_FILE}" > /dev/null <<EOF
[Unit]
Description=Nix binary cache server
After=network.target nix-daemon.service
Wants=nix-daemon.service

[Service]
Type=simple
User=nix-serve
Group=nix-serve
Environment=NIX_SECRET_KEY_FILE=${PRIVATE_KEY}
ExecStart=${NIX_SERVE_BIN} --listen 0.0.0.0:5000
Restart=on-failure
RestartSec=5s
NoNewPrivileges=true
# ProtectSystem=full (not strict) — nix-serve needs access to the Nix
# daemon socket at /nix/var/nix/daemon-socket/socket for store queries.
# strict would mount / read-only and block the socket connect().
ProtectSystem=full
ProtectHome=yes
PrivateTmp=yes

[Install]
WantedBy=multi-user.target
EOF

echo "[nix-cache-setup] Wrote ${SERVICE_FILE}."

if command -v systemctl &>/dev/null; then
  sudo systemctl daemon-reload
  sudo systemctl enable nix-serve.service
  # try-restart: applies new unit file on re-runs without failing if stopped
  sudo systemctl try-restart nix-serve.service
  # Start if not already running (first run)
  sudo systemctl start nix-serve.service 2>/dev/null || true
  echo "[nix-cache-setup] nix-serve service enabled and started."
fi

# ---------------------------------------------------------------------------
# 5. Print public key info for the deployer
# ---------------------------------------------------------------------------
echo ""
echo "======================================================================"
echo " Nix binary cache setup complete."
echo "======================================================================"
echo " Listening on: http://$(hostname -f):5000"
echo ""
echo " Public key (add to clients' nix.conf trusted-public-keys):"
echo "   $(cat "${PUBLIC_KEY}")"
echo ""
echo " Also add to clients' nix.conf substituters:"
echo "   http://$(hostname -f):5000"
echo "======================================================================"
