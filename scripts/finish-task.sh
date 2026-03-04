#!/usr/bin/env bash
# finish-task.sh — Merge a reviewed PR, clean up branches, return to main.
#
# Usage: ./scripts/finish-task.sh [PR_NUMBER]
#   If no PR number given, infers from current branch.
#
# Checks review state before merging. Refuses to merge unless state is
# REVIEWS_COMPLETE_ALL_CLEAR (override with --force).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_DIR"

OWNER="zeblithic"
REPO="harmony"

FORCE=false
PR=""

# ── Parse args ─────────────────────────────────────────────────────
for arg in "$@"; do
  case "$arg" in
    --force) FORCE=true ;;
    *) PR="$arg" ;;
  esac
done

# ── Resolve PR number ──────────────────────────────────────────────
BRANCH=$(git branch --show-current)
if [[ -z "$PR" ]]; then
  PR=$(gh pr view --json number --jq '.number' 2>/dev/null || echo "")
  if [[ -z "$PR" ]]; then
    echo "ERROR: No open PR for branch '$BRANCH'. Pass PR number as argument."
    exit 1
  fi
fi

TITLE=$(gh pr view "$PR" --json title --jq '.title')
PR_BRANCH=$(gh pr view "$PR" --json headRefName --jq '.headRefName')
echo "PR: #${PR} — ${TITLE}"
echo "Branch: ${PR_BRANCH}"
echo "─────────────────────────────────────────────────"

# ── Check for uncommitted changes ──────────────────────────────────
if [[ -n "$(git status --short)" ]]; then
  echo ""
  echo "WARNING: Uncommitted changes detected:"
  git status --short
  echo ""
  echo "Commit or stash these before merging. Aborting."
  exit 1
fi

# ── Verify review state ───────────────────────────────────────────
echo ""
echo "Checking review state..."
STATE_OUTPUT=$(bash "$SCRIPT_DIR/review-state.sh" "$PR" 2>&1)
echo "$STATE_OUTPUT"

STATE=$(echo "$STATE_OUTPUT" | grep "^State:" | awk '{print $2}')

if [[ "$STATE" != "REVIEWS_COMPLETE_ALL_CLEAR" ]]; then
  if [[ "$FORCE" == "true" ]]; then
    echo ""
    echo "WARNING: State is ${STATE}, but --force was specified. Proceeding."
  else
    echo ""
    echo "ERROR: State is ${STATE}. PR is not ready to merge."
    echo "  Use --force to override, or address the feedback first."
    exit 1
  fi
fi

# ── Merge ──────────────────────────────────────────────────────────
echo ""
echo "═════════════════════════════════════════════════"
echo "Merging PR #${PR}..."
gh pr merge "$PR" --merge --delete-branch

# ── Switch to main and pull ────────────────────────────────────────
echo ""
echo "Switching to main..."
git checkout main
git pull origin main

# ── Delete local branch (if it still exists) ──────────────────────
if git branch --list "$PR_BRANCH" | grep -q "$PR_BRANCH"; then
  git branch -d "$PR_BRANCH"
  echo "Deleted local branch: ${PR_BRANCH}"
fi

# ── Report ─────────────────────────────────────────────────────────
echo ""
echo "═════════════════════════════════════════════════"
echo "Merged:  PR #${PR} — ${TITLE}"
echo "Branch:  ${PR_BRANCH} deleted (local + remote)"
echo "Main:    $(git log --oneline -1)"
echo ""
echo "Task complete. Use /findtask to survey what's next."
