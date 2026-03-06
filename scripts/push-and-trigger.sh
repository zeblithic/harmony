#!/usr/bin/env bash
# push-and-trigger.sh — Push current branch and trigger both reviewers.
#
# Usage: ./scripts/push-and-trigger.sh [PR_NUMBER]
#   If no PR number given, infers from current branch.
#
# Gates on cargo test + clippy before pushing. Triggers both Bugbot and
# Greptile after push, then runs review-state.sh to confirm REVIEWS_PENDING.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Work from the caller's git root — supports both main repo and worktrees.
# (git rev-parse --show-toplevel returns the worktree root when in a worktree,
# or the main repo root when in the main checkout.)
GIT_ROOT="$(git rev-parse --show-toplevel)"
cd "$GIT_ROOT"

OWNER="zeblithic"
REPO="harmony"

# ── Verify not on main ─────────────────────────────────────────────
BRANCH=$(git branch --show-current)
if [[ "$BRANCH" == "main" || "$BRANCH" == "master" ]]; then
  echo "ERROR: Refusing to push from '$BRANCH'. Switch to a task branch first."
  exit 1
fi

# ── Resolve PR number ──────────────────────────────────────────────
PR="${1:-}"
if [[ -z "$PR" ]]; then
  PR=$(gh pr view --json number --jq '.number' 2>/dev/null || echo "")
  if [[ -z "$PR" ]]; then
    echo "ERROR: No open PR for branch '$BRANCH'. Create one with /delivertask first."
    exit 1
  fi
fi

TITLE=$(gh pr view "$PR" --json title --jq '.title')
echo "PR: #${PR} — ${TITLE}"
echo "Branch: ${BRANCH}"
echo "─────────────────────────────────────────────────"

# ── Quality gate: tests + clippy ───────────────────────────────────
echo ""
echo "Running quality gates..."

echo "  cargo test --workspace..."
if ! cargo test --workspace 2>&1 | tail -3; then
  echo "ERROR: Tests failed. Fix before pushing."
  exit 1
fi

echo "  cargo clippy --workspace..."
CLIPPY_OUTPUT=$(cargo clippy --workspace 2>&1)
if echo "$CLIPPY_OUTPUT" | grep -q "^warning\|^error"; then
  echo "$CLIPPY_OUTPUT" | grep "^warning\|^error"
  echo "ERROR: Clippy has warnings/errors. Fix before pushing."
  exit 1
fi
echo "  All clear."

# ── Push ───────────────────────────────────────────────────────────
echo ""
echo "Pushing to origin/${BRANCH}..."
git push

# ── Trigger both reviewers ─────────────────────────────────────────
echo ""
echo "Triggering reviewers..."
gh pr comment "$PR" --body "bugbot run"
echo "  Bugbot triggered."
gh pr comment "$PR" --body "@greptile"
echo "  Greptile triggered."

# ── Confirm state ──────────────────────────────────────────────────
echo ""
echo "═════════════════════════════════════════════════"
bash "$SCRIPT_DIR/review-state.sh" "$PR"
