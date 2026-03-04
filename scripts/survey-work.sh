#!/usr/bin/env bash
# survey-work.sh — Gather all context needed for /findtask in one shot.
#
# Usage: ./scripts/survey-work.sh
#
# Outputs: ready beads with details, recent merges, open PRs, current branch.

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

echo "═══════════════════════════════════════════════════"
echo "  Harmony Work Survey — $(date +%Y-%m-%d)"
echo "═══════════════════════════════════════════════════"

# ── Current state ──────────────────────────────────────────────────
echo ""
echo "── Current State ──"
echo "Branch: $(git branch --show-current 2>/dev/null || echo '(detached)')"
OPEN_PRS=$(gh pr list --state open --limit 5 --json number,title,headRefName 2>/dev/null || echo "[]")
if [[ "$OPEN_PRS" != "[]" ]]; then
  echo "Open PRs:"
  echo "$OPEN_PRS" | jq -r '.[] | "  #\(.number) [\(.headRefName)] \(.title)"'
else
  echo "Open PRs: none"
fi

# ── Recent merges ─────────────────────────────────────────────────
echo ""
echo "── Recent Merges (last 5) ──"
git log --oneline --merges -5 2>/dev/null || echo "(no merges)"

# ── Ready beads ───────────────────────────────────────────────────
echo ""
echo "── Ready Beads (unblocked) ──"
READY_OUTPUT=$(bd ready 2>/dev/null || echo "(bd not available)")
echo "$READY_OUTPUT"

# Extract bead IDs and show details for each
BEAD_IDS=$(echo "$READY_OUTPUT" | grep -oE 'harmony-[a-z0-9]+' | head -10)

if [[ -n "$BEAD_IDS" ]]; then
  echo ""
  echo "── Bead Details ──"
  for BEAD_ID in $BEAD_IDS; do
    echo ""
    echo "--- $BEAD_ID ---"
    bd show "$BEAD_ID" 2>/dev/null | head -20 || echo "(failed to fetch)"
  done
fi

# ── All beads (for blocked/deferred context) ──────────────────────
echo ""
echo "── All Beads ──"
bd list 2>/dev/null || echo "(bd not available)"

echo ""
echo "═══════════════════════════════════════════════════"
echo "Survey complete. Use this to decide what to work on next."
