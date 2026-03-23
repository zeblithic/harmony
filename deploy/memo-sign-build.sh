#!/usr/bin/env bash
# deploy/memo-sign-build.sh — Build a Nix package and sign a memo attestation.
#
# Creates a signed memo linking the Nix derivation (input CID) to the
# built binary (output CID). The memo can be published to the Harmony
# network so other nodes can verify build provenance.
#
# Environment variables:
#   PACKAGE        — Nix package attribute (default: harmony-node)
#   BINARY_NAME    — Binary name inside store path (default: harmony)
#   IDENTITY_FILE  — Path to signing identity (optional; defaults to ~/.harmony/identity.key)
#   HARMONY        — Path to harmony binary (default: harmony from PATH)
#   EXPIRES_IN     — Memo expiry in seconds (default: 31536000 = 365 days)
#
# Usage:
#   ./deploy/memo-sign-build.sh
#   PACKAGE=iroh-relay-x86_64-linux BINARY_NAME=iroh-relay ./deploy/memo-sign-build.sh
#   IDENTITY_FILE=/path/to/key ./deploy/memo-sign-build.sh

set -euo pipefail

PACKAGE="${PACKAGE:-harmony-node}"
BINARY_NAME="${BINARY_NAME:-harmony}"
HARMONY="${HARMONY:-harmony}"
IDENTITY_FILE="${IDENTITY_FILE:-}"
EXPIRES_IN="${EXPIRES_IN:-31536000}"
NIX_FLAGS=(--extra-experimental-features "nix-command flakes")

echo "Building ${PACKAGE}..." >&2
# Capture all output paths, then extract the first via parameter expansion
# (no pipe, no SIGPIPE risk under pipefail for multi-output packages).
ALL_PATHS=$(nix build ".#${PACKAGE}" --print-out-paths --no-link "${NIX_FLAGS[@]}")
STORE_PATH="${ALL_PATHS%%$'\n'*}"
BINARY="${STORE_PATH}/bin/${BINARY_NAME}"

if [ ! -f "${BINARY}" ]; then
    echo "Error: binary not found at ${BINARY}" >&2
    exit 1
fi

echo "Computing input CID from derivation..." >&2
INPUT_CID=$(nix derivation show ".#${PACKAGE}" "${NIX_FLAGS[@]}" | "${HARMONY}" cid)

echo "Computing output CID from ${BINARY}..." >&2
OUTPUT_CID=$("${HARMONY}" cid --file "${BINARY}")

echo "Input CID:  ${INPUT_CID}" >&2
echo "Output CID: ${OUTPUT_CID}" >&2

IDENTITY_ARGS=()
if [ -n "${IDENTITY_FILE}" ]; then
    IDENTITY_ARGS=(--identity-file "${IDENTITY_FILE}")
fi

echo "Signing memo..." >&2
"${HARMONY}" memo sign \
    --input "${INPUT_CID}" \
    --output "${OUTPUT_CID}" \
    --expires-in "${EXPIRES_IN}" \
    "${IDENTITY_ARGS[@]+"${IDENTITY_ARGS[@]}"}"
