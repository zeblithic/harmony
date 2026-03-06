#!/usr/bin/env bash
# review-state.sh — Deterministic PR review state checker.
#
# Usage: ./scripts/review-state.sh [--repo OWNER/REPO] [PR_NUMBER]
#   --repo: GitHub repo (default: auto-detect from gh CLI / current directory)
#   If no PR number given, infers from current branch via gh pr view.
#
# Works with any repo (harmony, harmony-os, harmony-client). When run from
# the repo directory, auto-detects the repo. Or pass --repo explicitly.
#
# Checks Bugbot (cursor[bot]) and Greptile (greptile-apps[bot]) review status,
# extracts all findings, and derives the review state machine state.

set -euo pipefail

# ── Helpers ─────────────────────────────────────────────────────────

# Parse ISO 8601 date to epoch seconds (macOS + Linux)
to_epoch() {
  local date_str="$1"
  if [[ -z "$date_str" ]]; then
    echo "0"
    return
  fi
  date -jf "%Y-%m-%dT%H:%M:%SZ" "$date_str" +%s 2>/dev/null \
    || date -d "$date_str" +%s 2>/dev/null \
    || echo "0"
}

# ── 1. Argument parsing & PR detection ──────────────────────────────
#
# Usage: ./scripts/review-state.sh [--repo OWNER/REPO] [PR_NUMBER]
#   --repo: GitHub repo in OWNER/REPO format (default: auto-detect from gh)
#   If no PR number given, infers from current branch via gh pr view.

REPO=""
PR_NUMBER=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo)
      REPO="$2"
      shift 2
      ;;
    *)
      PR_NUMBER="$1"
      shift
      ;;
  esac
done

# Auto-detect repo from gh CLI if not specified
if [[ -z "$REPO" ]]; then
  REPO=$(gh repo view --json nameWithOwner --jq '.nameWithOwner' 2>/dev/null || echo "")
  if [[ -z "$REPO" ]]; then
    echo "ERROR: Could not detect repo. Use --repo OWNER/REPO or run from a git repo."
    exit 1
  fi
fi

if [[ -z "$PR_NUMBER" ]]; then
  PR_NUMBER=$(gh pr view --json number --jq '.number' 2>/dev/null || echo "")
  if [[ -z "$PR_NUMBER" ]]; then
    echo "ERROR: No open PR for current branch. Pass PR number as argument."
    exit 1
  fi
fi

# Fetch PR metadata
PR_META=$(gh pr view "$PR_NUMBER" --json number,title,url,commits,createdAt)
PR_TITLE=$(echo "$PR_META" | jq -r '.title')
PR_URL=$(echo "$PR_META" | jq -r '.url')

echo "=== REVIEW STATE: PR #${PR_NUMBER} ==="
echo "Title: ${PR_TITLE}"
echo "URL:   ${PR_URL}"

# ── 2. Latest commit info ──────────────────────────────────────────

LAST_COMMIT_SHA=$(echo "$PR_META" | jq -r '.commits[-1].oid')
LAST_COMMIT_DATE=$(echo "$PR_META" | jq -r '.commits[-1].committedDate')
LAST_COMMIT_EPOCH=$(to_epoch "$LAST_COMMIT_DATE")

echo ""
echo "--- LATEST COMMIT ---"
echo "SHA:  ${LAST_COMMIT_SHA}"
echo "Date: ${LAST_COMMIT_DATE}"

# ── 3. Paginated data fetch ────────────────────────────────────────
# gh api --paginate emits one JSON array per page; jq -s 'add // []' merges them.

ALL_ISSUE_COMMENTS=$(gh api "repos/$REPO/issues/$PR_NUMBER/comments?per_page=100" --paginate 2>/dev/null | jq -s 'add // []')
ALL_PR_COMMENTS=$(gh api "repos/$REPO/pulls/$PR_NUMBER/comments?per_page=100" --paginate 2>/dev/null | jq -s 'add // []')
ALL_PR_REVIEWS=$(gh api "repos/$REPO/pulls/$PR_NUMBER/reviews?per_page=100" --paginate 2>/dev/null | jq -s 'add // []')

# ── 4. Trigger detection with reactions ─────────────────────────────

# Latest "bugbot run" trigger
BUGBOT_TRIGGER_JSON=$(echo "$ALL_ISSUE_COMMENTS" | jq '[.[] | select(.body == "bugbot run")] | last // empty')
BUGBOT_TRIGGER_DATE=$(echo "$BUGBOT_TRIGGER_JSON" | jq -r '.created_at // empty' 2>/dev/null)
BUGBOT_TRIGGER_ID=$(echo "$BUGBOT_TRIGGER_JSON" | jq -r '.id // empty' 2>/dev/null)

BUGBOT_TRIGGER_EYES=0
BUGBOT_TRIGGER_THUMBSUP=0
if [[ -n "$BUGBOT_TRIGGER_ID" && "$BUGBOT_TRIGGER_ID" != "null" ]]; then
  BUGBOT_REACTIONS=$(gh api "repos/$REPO/issues/comments/$BUGBOT_TRIGGER_ID" --jq '.reactions' 2>/dev/null || echo '{}')
  BUGBOT_TRIGGER_EYES=$(echo "$BUGBOT_REACTIONS" | jq -r '.eyes // 0')
  BUGBOT_TRIGGER_THUMBSUP=$(echo "$BUGBOT_REACTIONS" | jq -r '.["+1"] // 0')
fi

# Latest "@greptile" trigger — only match comments by humans (not greptile-apps[bot] itself)
GREPTILE_TRIGGER_JSON=$(echo "$ALL_ISSUE_COMMENTS" | jq '[.[] | select(.body == "@greptile" and .user.login != "greptile-apps[bot]")] | last // empty')
GREPTILE_TRIGGER_DATE=$(echo "$GREPTILE_TRIGGER_JSON" | jq -r '.created_at // empty' 2>/dev/null)
GREPTILE_TRIGGER_ID=$(echo "$GREPTILE_TRIGGER_JSON" | jq -r '.id // empty' 2>/dev/null)

# Greptile auto-triggers on PR creation. If no explicit @greptile comment exists,
# use PR creation time as baseline so the initial auto-triggered review is captured.
GREPTILE_AUTO_TRIGGER=false
if [[ -z "$GREPTILE_TRIGGER_DATE" ]]; then
  GREPTILE_TRIGGER_DATE=$(echo "$PR_META" | jq -r '.createdAt // empty')
  GREPTILE_AUTO_TRIGGER=true
fi

GREPTILE_TRIGGER_EYES=0
GREPTILE_TRIGGER_THUMBSUP=0
if [[ -n "$GREPTILE_TRIGGER_ID" && "$GREPTILE_TRIGGER_ID" != "null" ]]; then
  GREPTILE_REACTIONS=$(gh api "repos/$REPO/issues/comments/$GREPTILE_TRIGGER_ID" --jq '.reactions' 2>/dev/null || echo '{}')
  GREPTILE_TRIGGER_EYES=$(echo "$GREPTILE_REACTIONS" | jq -r '.eyes // 0')
  GREPTILE_TRIGGER_THUMBSUP=$(echo "$GREPTILE_REACTIONS" | jq -r '.["+1"] // 0')
fi

echo ""
echo "--- TRIGGERS ---"
if [[ -n "$BUGBOT_TRIGGER_DATE" ]]; then
  echo "Latest \"bugbot run\": ${BUGBOT_TRIGGER_DATE} (eyes: ${BUGBOT_TRIGGER_EYES}, thumbsup: ${BUGBOT_TRIGGER_THUMBSUP})"
else
  echo "Latest \"bugbot run\": none"
fi
if [[ -n "$GREPTILE_TRIGGER_DATE" ]]; then
  if [[ "$GREPTILE_AUTO_TRIGGER" == "true" ]]; then
    echo "Latest \"@greptile\":  auto (PR created ${GREPTILE_TRIGGER_DATE})"
  else
    echo "Latest \"@greptile\":  ${GREPTILE_TRIGGER_DATE} (eyes: ${GREPTILE_TRIGGER_EYES}, thumbsup: ${GREPTILE_TRIGGER_THUMBSUP})"
  fi
else
  echo "Latest \"@greptile\":  none"
fi

# ── 5. Bugbot findings (cursor[bot]) ───────────────────────────────

echo ""
echo "--- BUGBOT (cursor[bot]) ---"

BUGBOT_HAS_RESPONSE=false
BUGBOT_ISSUE_COUNT=0
BUGBOT_LATEST_REVIEW=""
BUGBOT_INLINE_COUNT=0

if [[ -n "$BUGBOT_TRIGGER_DATE" ]]; then
  # PR reviews from cursor[bot] after trigger
  BUGBOT_LATEST_REVIEW=$(echo "$ALL_PR_REVIEWS" | jq -r \
    "[.[] | select(.user.login == \"cursor[bot]\" and .submitted_at > \"$BUGBOT_TRIGGER_DATE\")] | last // empty | .submitted_at // empty" 2>/dev/null)

  # Inline PR comments from cursor[bot] after trigger
  BUGBOT_INLINE_COMMENTS=$(echo "$ALL_PR_COMMENTS" | jq \
    "[.[] | select(.user.login == \"cursor[bot]\" and .created_at > \"$BUGBOT_TRIGGER_DATE\")]" 2>/dev/null)
  BUGBOT_INLINE_COUNT=$(echo "$BUGBOT_INLINE_COMMENTS" | jq 'length' 2>/dev/null || echo "0")

  if [[ -n "$BUGBOT_LATEST_REVIEW" || "$BUGBOT_INLINE_COUNT" -gt 0 ]]; then
    BUGBOT_HAS_RESPONSE=true
  fi

  # Thumbs-up on trigger without formal review = "reviewed, no issues"
  # This happens when the change is trivial (e.g., applying the reviewer's own suggestion).
  if [[ "$BUGBOT_HAS_RESPONSE" == "false" && "$BUGBOT_TRIGGER_THUMBSUP" -gt 0 && "$BUGBOT_TRIGGER_EYES" -eq 0 ]]; then
    BUGBOT_HAS_RESPONSE=true
  fi

  # Count issues (inline comments are findings)
  BUGBOT_ISSUE_COUNT="$BUGBOT_INLINE_COUNT"
else
  BUGBOT_INLINE_COMMENTS="[]"
fi

if [[ "$BUGBOT_HAS_RESPONSE" == "true" ]]; then
  echo "Status: complete"
  echo "Latest review: ${BUGBOT_LATEST_REVIEW:-none}"
  echo "Inline comments after trigger: ${BUGBOT_INLINE_COUNT}"
  echo "Issues found: ${BUGBOT_ISSUE_COUNT}"

  # Print each finding
  if [[ "$BUGBOT_INLINE_COUNT" -gt 0 ]]; then
    echo ""
    local_idx=0
    while IFS= read -r finding_json; do
      local_idx=$((local_idx + 1))
      f_file=$(echo "$finding_json" | jq -r '.path // "unknown"')
      f_line=$(echo "$finding_json" | jq -r '.line // .original_line // "?"')
      f_body=$(echo "$finding_json" | jq -r '.body // ""')
      # Parse severity from body
      f_severity="Unknown"
      if echo "$f_body" | grep -qi "critical severity"; then
        f_severity="Critical"
      elif echo "$f_body" | grep -qi "high severity"; then
        f_severity="High"
      elif echo "$f_body" | grep -qi "medium severity"; then
        f_severity="Medium"
      elif echo "$f_body" | grep -qi "low severity"; then
        f_severity="Low"
      fi
      echo "Finding ${local_idx}:"
      echo "  File: ${f_file}"
      echo "  Line: ${f_line}"
      echo "  Severity: ${f_severity}"
      echo "  Body:"
      echo "$f_body" | sed 's/^/    /'
      echo ""
    done < <(echo "$BUGBOT_INLINE_COMMENTS" | jq -c '.[]')
  fi
else
  if [[ -n "$BUGBOT_TRIGGER_DATE" ]]; then
    echo "Status: pending"
    echo "Triggered at: ${BUGBOT_TRIGGER_DATE}"
    echo "Inline comments after trigger: 0"
    echo "Issues found: 0"
  else
    echo "Status: not triggered"
  fi
fi

# ── 6. Greptile findings (greptile-apps[bot]) ──────────────────────

echo ""
echo "--- GREPTILE (greptile-apps[bot]) ---"

GREPTILE_HAS_RESPONSE=false
GREPTILE_ISSUE_COUNT=0
GREPTILE_LATEST_COMMENT=""
GREPTILE_COMMENT_COUNT=0

GREPTILE_INLINE_COMMENTS="[]"
GREPTILE_INLINE_COUNT=0

if [[ -n "$GREPTILE_TRIGGER_DATE" ]]; then
  # Greptile posts issue comments (summary) AND inline PR review comments (findings)
  GREPTILE_RESPONSE_COMMENTS=$(echo "$ALL_ISSUE_COMMENTS" | jq \
    "[.[] | select(.user.login == \"greptile-apps[bot]\" and .created_at > \"$GREPTILE_TRIGGER_DATE\")]" 2>/dev/null)
  GREPTILE_COMMENT_COUNT=$(echo "$GREPTILE_RESPONSE_COMMENTS" | jq 'length' 2>/dev/null || echo "0")
  GREPTILE_LATEST_COMMENT=$(echo "$GREPTILE_RESPONSE_COMMENTS" | jq -r 'last // empty | .created_at // empty' 2>/dev/null)

  # Inline PR comments from greptile-apps[bot] after trigger (this is where findings live)
  GREPTILE_INLINE_COMMENTS=$(echo "$ALL_PR_COMMENTS" | jq \
    "[.[] | select(.user.login == \"greptile-apps[bot]\" and .created_at > \"$GREPTILE_TRIGGER_DATE\")]" 2>/dev/null)
  GREPTILE_INLINE_COUNT=$(echo "$GREPTILE_INLINE_COMMENTS" | jq 'length' 2>/dev/null || echo "0")

  # Also check PR reviews (Greptile posts formal reviews with empty bodies)
  GREPTILE_LATEST_REVIEW=$(echo "$ALL_PR_REVIEWS" | jq -r \
    "[.[] | select(.user.login == \"greptile-apps[bot]\" and .submitted_at > \"$GREPTILE_TRIGGER_DATE\")] | last // empty | .submitted_at // empty" 2>/dev/null)

  # Track the latest timestamp from any response type
  if [[ -z "$GREPTILE_LATEST_COMMENT" && -n "$GREPTILE_LATEST_REVIEW" ]]; then
    GREPTILE_LATEST_COMMENT="$GREPTILE_LATEST_REVIEW"
  fi

  if [[ "$GREPTILE_COMMENT_COUNT" -gt 0 || "$GREPTILE_INLINE_COUNT" -gt 0 || -n "$GREPTILE_LATEST_REVIEW" ]]; then
    GREPTILE_HAS_RESPONSE=true
    # Inline PR comments are the actual findings
    GREPTILE_ISSUE_COUNT="$GREPTILE_INLINE_COUNT"
  fi

  # Thumbs-up on trigger without formal response = "reviewed, no issues"
  # This happens when the change is trivial (e.g., applying the reviewer's own suggestion).
  if [[ "$GREPTILE_HAS_RESPONSE" == "false" && "$GREPTILE_TRIGGER_THUMBSUP" -gt 0 && "$GREPTILE_TRIGGER_EYES" -eq 0 ]]; then
    GREPTILE_HAS_RESPONSE=true
  fi
else
  GREPTILE_RESPONSE_COMMENTS="[]"
fi

if [[ "$GREPTILE_HAS_RESPONSE" == "true" ]]; then
  echo "Status: complete"
  echo "Latest comment: ${GREPTILE_LATEST_COMMENT}"
  echo "Issue comments after trigger: ${GREPTILE_COMMENT_COUNT}"
  echo "Inline comments after trigger: ${GREPTILE_INLINE_COUNT}"
  echo "Issues found: ${GREPTILE_ISSUE_COUNT}"

  # Print each inline finding (these are the actual code issues)
  if [[ "$GREPTILE_INLINE_COUNT" -gt 0 ]]; then
    echo ""
    local_idx=0
    while IFS= read -r finding_json; do
      local_idx=$((local_idx + 1))
      f_file=$(echo "$finding_json" | jq -r '.path // "unknown"')
      f_line=$(echo "$finding_json" | jq -r '.line // .original_line // "?"')
      f_body=$(echo "$finding_json" | jq -r '.body // ""')
      echo "Finding ${local_idx}:"
      echo "  File: ${f_file}"
      echo "  Line: ${f_line}"
      echo "  Body:"
      echo "$f_body" | sed 's/^/    /'
      echo ""
    done < <(echo "$GREPTILE_INLINE_COMMENTS" | jq -c '.[]')
  fi

  # Print summary comment if present
  if [[ "$GREPTILE_COMMENT_COUNT" -gt 0 ]]; then
    echo ""
    echo "Summary comment:"
    echo "$GREPTILE_RESPONSE_COMMENTS" | jq -r 'last // empty | .body // ""' | sed 's/^/    /'
    echo ""
  fi
else
  if [[ -n "$GREPTILE_TRIGGER_DATE" ]]; then
    echo "Status: pending"
    echo "Triggered at: ${GREPTILE_TRIGGER_DATE}"
    echo "Issues found: 0"
  else
    echo "Status: not triggered"
  fi
fi

# ── 7. State machine derivation ────────────────────────────────────

# Determine latest trigger date (for staleness check)
LATEST_TRIGGER_EPOCH=0
if [[ -n "$BUGBOT_TRIGGER_DATE" ]]; then
  BUGBOT_TRIGGER_EPOCH=$(to_epoch "$BUGBOT_TRIGGER_DATE")
  if [[ "$BUGBOT_TRIGGER_EPOCH" -gt "$LATEST_TRIGGER_EPOCH" ]]; then
    LATEST_TRIGGER_EPOCH="$BUGBOT_TRIGGER_EPOCH"
  fi
fi
if [[ -n "$GREPTILE_TRIGGER_DATE" ]]; then
  GREPTILE_TRIGGER_EPOCH=$(to_epoch "$GREPTILE_TRIGGER_DATE")
  if [[ "$GREPTILE_TRIGGER_EPOCH" -gt "$LATEST_TRIGGER_EPOCH" ]]; then
    LATEST_TRIGGER_EPOCH="$GREPTILE_TRIGGER_EPOCH"
  fi
fi

NOW_EPOCH=$(date +%s)

# State machine rules (priority order):
# 1. No triggers -> NO_TRIGGERS
# 2. Latest commit newer than latest trigger -> STALE
# 3. Bugbot trigger exists, no response, >3 min, no eyes -> BUGBOT_STUCK
# 4. Only one bot responded -> PARTIAL_REVIEWS
# 5. Both responded, issues > 0 -> REVIEWS_COMPLETE_WITH_FEEDBACK
# 6. Both responded, no issues -> REVIEWS_COMPLETE_ALL_CLEAR
# 7. Fallback -> REVIEWS_PENDING

STATE=""
BUGBOT_STATE_LABEL=""
GREPTILE_STATE_LABEL=""
ACTION=""

# Shorthand
HAS_BUGBOT_TRIGGER=false
HAS_GREPTILE_TRIGGER=false
[[ -n "$BUGBOT_TRIGGER_DATE" ]] && HAS_BUGBOT_TRIGGER=true
[[ -n "$GREPTILE_TRIGGER_DATE" ]] && HAS_GREPTILE_TRIGGER=true

if [[ "$HAS_BUGBOT_TRIGGER" == "false" && "$HAS_GREPTILE_TRIGGER" == "false" ]]; then
  # Rule 1: No triggers
  STATE="NO_TRIGGERS"
  BUGBOT_STATE_LABEL="not triggered"
  GREPTILE_STATE_LABEL="not triggered"
  ACTION="Trigger reviews: comment 'bugbot run' and '@greptile' on the PR."

elif [[ "$LAST_COMMIT_EPOCH" -gt "$LATEST_TRIGGER_EPOCH" ]]; then
  # Rule 2: Commit newer than triggers -> STALE
  STATE="STALE"
  BUGBOT_STATE_LABEL="stale (commit after trigger)"
  GREPTILE_STATE_LABEL="stale (commit after trigger)"
  ACTION="Re-trigger reviews: comment 'bugbot run' and '@greptile' on the PR."

elif [[ "$HAS_BUGBOT_TRIGGER" == "true" && "$BUGBOT_HAS_RESPONSE" == "false" ]]; then
  # Check rule 3: Bugbot stuck? (BUGBOT_TRIGGER_EPOCH computed above)
  ELAPSED=$(( NOW_EPOCH - BUGBOT_TRIGGER_EPOCH ))
  if [[ "$ELAPSED" -gt 180 && "$BUGBOT_TRIGGER_EYES" -eq 0 ]]; then
    STATE="BUGBOT_STUCK"
    BUGBOT_STATE_LABEL="stuck (no eyes after ${ELAPSED}s)"
    if [[ "$GREPTILE_HAS_RESPONSE" == "true" ]]; then
      GREPTILE_STATE_LABEL="complete (${GREPTILE_ISSUE_COUNT} issues)"
    elif [[ "$HAS_GREPTILE_TRIGGER" == "true" ]]; then
      GREPTILE_STATE_LABEL="pending"
    else
      GREPTILE_STATE_LABEL="not triggered"
    fi
    ACTION="Bugbot may be stuck. Re-trigger with 'bugbot run' comment."
  fi
fi

# If no state set yet, continue with rules 4-7
if [[ -z "$STATE" ]]; then
  if [[ "$BUGBOT_HAS_RESPONSE" == "true" && "$GREPTILE_HAS_RESPONSE" == "true" ]]; then
    TOTAL_ISSUES=$(( BUGBOT_ISSUE_COUNT + GREPTILE_ISSUE_COUNT ))
    if [[ "$TOTAL_ISSUES" -gt 0 ]]; then
      # Rule 5: Both responded, issues found
      STATE="REVIEWS_COMPLETE_WITH_FEEDBACK"
      BUGBOT_STATE_LABEL="complete (${BUGBOT_ISSUE_COUNT} issues)"
      GREPTILE_STATE_LABEL="complete (${GREPTILE_ISSUE_COUNT} issues)"
      ACTION="Fix issues locally, then push + re-trigger when ready. Safe to use bd commands."
    else
      # Rule 6: Both responded, no issues
      STATE="REVIEWS_COMPLETE_ALL_CLEAR"
      BUGBOT_STATE_LABEL="complete (0 issues)"
      GREPTILE_STATE_LABEL="complete (0 issues)"
      ACTION="All clear. Ready for /finishtask to merge."
    fi
  elif [[ "$BUGBOT_HAS_RESPONSE" == "true" || "$GREPTILE_HAS_RESPONSE" == "true" ]]; then
    # Rule 4: Only one responded
    STATE="PARTIAL_REVIEWS"
    if [[ "$BUGBOT_HAS_RESPONSE" == "true" ]]; then
      BUGBOT_STATE_LABEL="complete (${BUGBOT_ISSUE_COUNT} issues)"
    elif [[ "$HAS_BUGBOT_TRIGGER" == "true" ]]; then
      BUGBOT_STATE_LABEL="pending"
    else
      BUGBOT_STATE_LABEL="not triggered"
    fi
    if [[ "$GREPTILE_HAS_RESPONSE" == "true" ]]; then
      GREPTILE_STATE_LABEL="complete (${GREPTILE_ISSUE_COUNT} issues)"
    elif [[ "$HAS_GREPTILE_TRIGGER" == "true" ]]; then
      GREPTILE_STATE_LABEL="pending"
    else
      GREPTILE_STATE_LABEL="not triggered"
    fi
    ACTION="Waiting for remaining reviewer(s). Do NOT push or run bd commands."
  else
    # Rule 7: Fallback — triggers exist but no responses yet
    STATE="REVIEWS_PENDING"
    if [[ "$HAS_BUGBOT_TRIGGER" == "true" ]]; then
      BUGBOT_STATE_LABEL="pending"
    else
      BUGBOT_STATE_LABEL="not triggered"
    fi
    if [[ "$HAS_GREPTILE_TRIGGER" == "true" ]]; then
      GREPTILE_STATE_LABEL="pending"
    else
      GREPTILE_STATE_LABEL="not triggered"
    fi
    ACTION="Reviews in progress. Do NOT push or run bd commands."
  fi
fi

echo ""
echo "--- STATE ---"
echo "State:    ${STATE}"
echo "Bugbot:   ${BUGBOT_STATE_LABEL}"
echo "Greptile: ${GREPTILE_STATE_LABEL}"
echo "Action:   ${ACTION}"
