#!/usr/bin/env bash
# review-state.sh — Deterministic PR review state checker for zeblithic/harmony.
#
# Usage: ./scripts/review-state.sh [PR_NUMBER]
#   If no PR number given, infers from current branch.
#
# Checks Bugbot (cursor[bot]) and Greptile (greptile-apps) review status
# and derives the review state machine state.

set -euo pipefail

OWNER="zeblithic"
REPO="harmony"

# ── Resolve PR number ────────────────────────────────────────────────
PR="${1:-}"
if [[ -z "$PR" ]]; then
  PR=$(gh pr view --json number --jq '.number' 2>/dev/null || echo "")
  if [[ -z "$PR" ]]; then
    echo "ERROR: No open PR for current branch. Pass PR number as argument."
    exit 1
  fi
fi

TITLE=$(gh pr view "$PR" --json title --jq '.title')
echo "PR: #${PR} — ${TITLE}"
echo "─────────────────────────────────────────────────"

# ── Gather data ──────────────────────────────────────────────────────

# Latest commit
LAST_COMMIT_SHA=$(gh pr view "$PR" --json commits --jq '.commits[-1].oid')
LAST_COMMIT_DATE=$(gh pr view "$PR" --json commits --jq '.commits[-1].committedDate')
echo "Last commit:  ${LAST_COMMIT_SHA:0:7} @ ${LAST_COMMIT_DATE}"

# Fetch all issue comments once (paginated — default 30/page is not enough for active PRs).
# --paginate may emit multiple JSON arrays; --slurp + add merges them into one.
ISSUE_COMMENTS=$(gh api "repos/${OWNER}/${REPO}/issues/${PR}/comments?per_page=100" --paginate --jq '.' 2>/dev/null | jq -s 'add // []')

# Fetch all PR review comments once (paginated — this is where Bugbot inline findings live)
PR_REVIEW_COMMENTS=$(gh api "repos/${OWNER}/${REPO}/pulls/${PR}/comments?per_page=100" --paginate --jq '.' 2>/dev/null | jq -s 'add // []')

# Fetch all PR reviews once (paginated)
PR_REVIEWS=$(gh api "repos/${OWNER}/${REPO}/pulls/${PR}/reviews?per_page=100" --paginate --jq '.' 2>/dev/null | jq -s 'add // []')

# Latest "bugbot run" trigger (from issue comments)
BUGBOT_TRIGGER=$(echo "$ISSUE_COMMENTS" | jq -r '[.[] | select(.body == "bugbot run")] | last // empty | .created_at // empty' 2>/dev/null)
BUGBOT_TRIGGER_ID=$(echo "$ISSUE_COMMENTS" | jq -r '[.[] | select(.body == "bugbot run")] | last // empty | .id // empty' 2>/dev/null)
echo "Bugbot trigger: ${BUGBOT_TRIGGER:-none}"

# Latest "@greptile" trigger (from issue comments)
GREPTILE_TRIGGER=$(echo "$ISSUE_COMMENTS" | jq -r '[.[] | select(.body == "@greptile")] | last // empty | .created_at // empty' 2>/dev/null)
GREPTILE_TRIGGER_ID=$(echo "$ISSUE_COMMENTS" | jq -r '[.[] | select(.body == "@greptile")] | last // empty | .id // empty' 2>/dev/null)
echo "Greptile trigger: ${GREPTILE_TRIGGER:-none}"

# Eyes emoji on triggers
BUGBOT_EYES=0
GREPTILE_EYES=0
if [[ -n "$BUGBOT_TRIGGER_ID" ]]; then
  BUGBOT_EYES=$(gh api "repos/${OWNER}/${REPO}/issues/comments/${BUGBOT_TRIGGER_ID}" \
    --jq '.reactions.eyes // 0' 2>/dev/null)
fi
if [[ -n "$GREPTILE_TRIGGER_ID" ]]; then
  GREPTILE_EYES=$(gh api "repos/${OWNER}/${REPO}/issues/comments/${GREPTILE_TRIGGER_ID}" \
    --jq '.reactions.eyes // 0' 2>/dev/null)
fi
echo "Bugbot eyes:  ${BUGBOT_EYES}"
echo "Greptile eyes: ${GREPTILE_EYES}"

# Thumbs-up on triggers
BUGBOT_THUMBSUP=0
GREPTILE_THUMBSUP=0
if [[ -n "$BUGBOT_TRIGGER_ID" ]]; then
  BUGBOT_THUMBSUP=$(gh api "repos/${OWNER}/${REPO}/issues/comments/${BUGBOT_TRIGGER_ID}" \
    --jq '.reactions["+1"] // 0' 2>/dev/null)
fi
if [[ -n "$GREPTILE_TRIGGER_ID" ]]; then
  GREPTILE_THUMBSUP=$(gh api "repos/${OWNER}/${REPO}/issues/comments/${GREPTILE_TRIGGER_ID}" \
    --jq '.reactions["+1"] // 0' 2>/dev/null)
fi
echo "Bugbot thumbs-up:  ${BUGBOT_THUMBSUP}"
echo "Greptile thumbs-up: ${GREPTILE_THUMBSUP}"

# Latest cursor[bot] review (from PR reviews API)
BUGBOT_LAST_REVIEW=$(echo "$PR_REVIEWS" | jq -r '[.[] | select(.user.login == "cursor[bot]")] | last // empty | .submitted_at // empty' 2>/dev/null)
echo "Bugbot last review: ${BUGBOT_LAST_REVIEW:-none}"

# Latest cursor[bot] inline comment (from PR review comments API — THIS is where findings live)
BUGBOT_LAST_INLINE=$(echo "$PR_REVIEW_COMMENTS" | jq -r '[.[] | select(.user.login == "cursor[bot]")] | last // empty | .created_at // empty' 2>/dev/null)
echo "Bugbot last inline: ${BUGBOT_LAST_INLINE:-none}"

# Count of Bugbot inline comments AFTER latest commit
BUGBOT_NEW_COMMENTS=$(echo "$PR_REVIEW_COMMENTS" | jq -r "[.[] | select(.user.login == \"cursor[bot]\" and .created_at > \"${LAST_COMMIT_DATE}\")] | length" 2>/dev/null)
echo "Bugbot new inlines: ${BUGBOT_NEW_COMMENTS}"

# Latest greptile-apps PR comment (from issue comments, NOT reviews)
GREPTILE_LAST_COMMENT=$(echo "$ISSUE_COMMENTS" | jq -r '[.[] | select(.user.login == "greptile-apps[bot]")] | last // empty | .created_at // empty' 2>/dev/null)
echo "Greptile last comment: ${GREPTILE_LAST_COMMENT:-none}"

# Count of Greptile comments AFTER latest commit
GREPTILE_NEW_COMMENTS=$(echo "$ISSUE_COMMENTS" | jq -r "[.[] | select(.user.login == \"greptile-apps[bot]\" and .created_at > \"${LAST_COMMIT_DATE}\")] | length" 2>/dev/null)
echo "Greptile new comments: ${GREPTILE_NEW_COMMENTS}"

echo "─────────────────────────────────────────────────"

# ── Derive state ─────────────────────────────────────────────────────

# Helper: is timestamp A newer than B?
newer() { [[ "$1" > "$2" ]]; }

# Bugbot responded = has a review OR inline comment newer than the commit
BUGBOT_RESPONDED=false
if [[ -n "$BUGBOT_LAST_REVIEW" ]] && newer "$BUGBOT_LAST_REVIEW" "$LAST_COMMIT_DATE"; then
  BUGBOT_RESPONDED=true
fi
if [[ -n "$BUGBOT_LAST_INLINE" ]] && newer "$BUGBOT_LAST_INLINE" "$LAST_COMMIT_DATE"; then
  BUGBOT_RESPONDED=true
fi

# Greptile responded = has a comment newer than the commit
GREPTILE_RESPONDED=false
if [[ -n "$GREPTILE_LAST_COMMENT" ]] && newer "$GREPTILE_LAST_COMMENT" "$LAST_COMMIT_DATE"; then
  GREPTILE_RESPONDED=true
fi
# Also count thumbs-up on trigger as "responded with no new findings"
if [[ -n "$GREPTILE_TRIGGER" ]] && newer "$GREPTILE_TRIGGER" "$LAST_COMMIT_DATE" && [[ "$GREPTILE_THUMBSUP" -gt 0 ]]; then
  GREPTILE_RESPONDED=true
fi

# Check if triggers are pending (newer than responses)
BUGBOT_PENDING=false
if [[ -n "$BUGBOT_TRIGGER" ]]; then
  if [[ "$BUGBOT_RESPONDED" == "false" ]]; then
    if [[ "$BUGBOT_EYES" -gt 0 ]]; then
      BUGBOT_PENDING=true
    fi
  fi
fi

GREPTILE_PENDING=false
if [[ -n "$GREPTILE_TRIGGER" ]]; then
  if [[ "$GREPTILE_RESPONDED" == "false" ]]; then
    if [[ "$GREPTILE_EYES" -gt 0 ]]; then
      GREPTILE_PENDING=true
    fi
  fi
fi

# Has feedback?
BUGBOT_HAS_FEEDBACK=false
if [[ "$BUGBOT_NEW_COMMENTS" -gt 0 ]]; then
  BUGBOT_HAS_FEEDBACK=true
fi

GREPTILE_HAS_FEEDBACK=false
if [[ "$GREPTILE_NEW_COMMENTS" -gt 0 ]]; then
  # Check if they're just trigger acknowledgments vs actual feedback
  GREPTILE_CONTENT_COMMENTS=$(echo "$ISSUE_COMMENTS" | jq -r "[.[] | select(.user.login == \"greptile-apps[bot]\" and .created_at > \"${LAST_COMMIT_DATE}\" and (.body | test(\"Additional Comments|Summary|Issues|suggestion\")))] | length" 2>/dev/null)
  if [[ "$GREPTILE_CONTENT_COMMENTS" -gt 0 ]]; then
    GREPTILE_HAS_FEEDBACK=true
  fi
fi

# ── State machine ────────────────────────────────────────────────────

if [[ "$BUGBOT_PENDING" == "true" || "$GREPTILE_PENDING" == "true" ]]; then
  STATE="REVIEWS_PENDING"
  BUGBOT_STATUS="running"
  [[ "$BUGBOT_RESPONDED" == "true" ]] && BUGBOT_STATUS="complete"
  GREPTILE_STATUS="running"
  [[ "$GREPTILE_RESPONDED" == "true" ]] && GREPTILE_STATUS="complete"
  ACTION="Reviews in progress. Do NOT push or run bd commands."
elif [[ "$BUGBOT_RESPONDED" == "true" && "$GREPTILE_RESPONDED" == "true" ]]; then
  if [[ "$BUGBOT_HAS_FEEDBACK" == "true" || "$GREPTILE_HAS_FEEDBACK" == "true" ]]; then
    STATE="REVIEWS_COMPLETE_WITH_FEEDBACK"
    BUGBOT_STATUS="complete (${BUGBOT_NEW_COMMENTS} new inline comments)"
    GREPTILE_STATUS="complete (${GREPTILE_NEW_COMMENTS} new comments)"
    ACTION="Fix issues locally, then push + re-trigger."
  else
    STATE="REVIEWS_COMPLETE_ALL_CLEAR"
    BUGBOT_STATUS="complete (no new findings)"
    GREPTILE_STATUS="complete (no new findings)"
    ACTION="Ready for /finishtask to merge."
  fi
elif [[ "$BUGBOT_RESPONDED" == "true" || "$GREPTILE_RESPONDED" == "true" ]]; then
  STATE="PARTIAL_REVIEWS"
  BUGBOT_STATUS="pending"
  [[ "$BUGBOT_RESPONDED" == "true" ]] && BUGBOT_STATUS="complete"
  GREPTILE_STATUS="pending"
  [[ "$GREPTILE_RESPONDED" == "true" ]] && GREPTILE_STATUS="complete"
  ACTION="Waiting for remaining reviewer(s). Do NOT push."
else
  # Neither responded — check if triggers exist
  if [[ -n "$BUGBOT_TRIGGER" ]] && newer "$BUGBOT_TRIGGER" "$LAST_COMMIT_DATE"; then
    STATE="REVIEWS_PENDING"
    BUGBOT_STATUS="triggered, waiting"
    GREPTILE_STATUS="triggered, waiting"
    ACTION="Reviews in progress. Do NOT push or run bd commands."
  else
    STATE="NO_REVIEWS_TRIGGERED"
    BUGBOT_STATUS="not triggered"
    GREPTILE_STATUS="not triggered"
    ACTION="Trigger reviews: gh pr comment ${PR} --body 'bugbot run' && gh pr comment ${PR} --body '@greptile'"
  fi
fi

echo ""
echo "Bugbot:   ${BUGBOT_STATUS}"
echo "Greptile: ${GREPTILE_STATUS}"
echo "State:    ${STATE}"
echo "Action:   ${ACTION}"

# If there are new Bugbot findings, show them
if [[ "$BUGBOT_NEW_COMMENTS" -gt 0 ]]; then
  echo ""
  echo "── Bugbot findings (inline review comments on diff) ──"
  echo "$PR_REVIEW_COMMENTS" | jq -r ".[] | select(.user.login == \"cursor[bot]\" and .created_at > \"${LAST_COMMIT_DATE}\") | \"  [\(.created_at)] \(.path):\(.line // \"?\") — \(.body | split(\"\n\")[0][:100])\"" 2>/dev/null
fi

# If there are new Greptile findings, show them
if [[ "$GREPTILE_HAS_FEEDBACK" == "true" ]]; then
  echo ""
  echo "── Greptile findings (PR-level issue comments) ──"
  echo "$ISSUE_COMMENTS" | jq -r ".[] | select(.user.login == \"greptile-apps[bot]\" and .created_at > \"${LAST_COMMIT_DATE}\") | \"  [\(.created_at)] \(.body | split(\"\n\")[0][:100])\"" 2>/dev/null
fi
