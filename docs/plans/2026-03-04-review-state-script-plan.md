# review-state.sh Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Create `scripts/review-state.sh` — a deterministic shell script that gathers all PR review signals and derives the review state machine state, with full finding extraction.

**Architecture:** Single bash script using `gh api --paginate` + `jq` to fetch data from three GitHub API endpoints (PR reviews, PR inline comments, issue comments), then applies the state machine rules to derive the current review state. Outputs structured text with labeled sections.

**Tech Stack:** bash, gh CLI, jq

---

### Task 1: Create script skeleton with argument parsing and PR detection

**Files:**
- Create: `scripts/review-state.sh`

**Step 1: Create the scripts directory and the script file**

```bash
#!/usr/bin/env bash
set -euo pipefail

REPO="zeblithic/harmony"
PR_NUMBER="${1:-}"

# Auto-detect PR from current branch if not provided
if [[ -z "$PR_NUMBER" ]]; then
  PR_JSON=$(gh pr view --json number,title,url,state 2>/dev/null) || {
    echo "ERROR: No PR number provided and no PR found for current branch." >&2
    exit 1
  }
  PR_NUMBER=$(echo "$PR_JSON" | jq -r '.number')
  PR_TITLE=$(echo "$PR_JSON" | jq -r '.title')
  PR_URL=$(echo "$PR_JSON" | jq -r '.url')
else
  PR_JSON=$(gh pr view "$PR_NUMBER" --json number,title,url,state 2>/dev/null) || {
    echo "ERROR: PR #$PR_NUMBER not found." >&2
    exit 1
  }
  PR_TITLE=$(echo "$PR_JSON" | jq -r '.title')
  PR_URL=$(echo "$PR_JSON" | jq -r '.url')
fi

echo "=== REVIEW STATE: PR #$PR_NUMBER ==="
echo "Title: $PR_TITLE"
echo "URL: $PR_URL"
echo ""
```

**Step 2: Run it to verify basic output**

Run: `bash scripts/review-state.sh 35`
Expected: Prints PR header with number, title, URL.

Run: `bash scripts/review-state.sh`
Expected: Auto-detects PR from current branch (jake-workflow-durable-execution → PR #35).

Run: `bash scripts/review-state.sh 99999`
Expected: `ERROR: PR #99999 not found.` and exit 1.

**Step 3: Commit**

```bash
git add scripts/review-state.sh
git commit -m "feat: add review-state.sh skeleton with PR detection"
```

---

### Task 2: Fetch latest commit info

**Files:**
- Modify: `scripts/review-state.sh`

**Step 1: Add latest commit section after PR header**

```bash
# --- Latest commit ---
LATEST_COMMIT_JSON=$(gh pr view "$PR_NUMBER" --json commits --jq '.commits[-1] | {sha: .oid, date: .committedDate}')
LATEST_COMMIT_SHA=$(echo "$LATEST_COMMIT_JSON" | jq -r '.sha')
LATEST_COMMIT_DATE=$(echo "$LATEST_COMMIT_JSON" | jq -r '.date')

echo "--- LATEST COMMIT ---"
echo "SHA: $LATEST_COMMIT_SHA"
echo "Date: $LATEST_COMMIT_DATE"
echo ""
```

**Step 2: Run and verify**

Run: `bash scripts/review-state.sh 35`
Expected: Shows commit SHA and timestamp after the header.

**Step 3: Commit**

```bash
git add scripts/review-state.sh
git commit -m "feat(review-state): add latest commit info"
```

---

### Task 3: Fetch trigger comments and reactions

**Files:**
- Modify: `scripts/review-state.sh`

**Step 1: Add trigger detection**

Fetch all issue comments (paginated), find the latest "bugbot run" and "@greptile" comments, extract their timestamps and reaction counts.

```bash
# --- Triggers ---
# Fetch all issue comments (includes "bugbot run" and "@greptile" triggers)
ALL_ISSUE_COMMENTS=$(gh api "repos/$REPO/issues/$PR_NUMBER/comments?per_page=100" --paginate)

# Latest "bugbot run" trigger
BUGBOT_TRIGGER_JSON=$(echo "$ALL_ISSUE_COMMENTS" | jq -r '
  [.[] | select(.body == "bugbot run")] | last //empty')
if [[ -n "$BUGBOT_TRIGGER_JSON" ]]; then
  BUGBOT_TRIGGER_DATE=$(echo "$BUGBOT_TRIGGER_JSON" | jq -r '.created_at')
  BUGBOT_TRIGGER_EYES=$(echo "$BUGBOT_TRIGGER_JSON" | jq -r '.reactions.eyes')
  BUGBOT_TRIGGER_THUMBSUP=$(echo "$BUGBOT_TRIGGER_JSON" | jq -r '.reactions["+1"]')
else
  BUGBOT_TRIGGER_DATE=""
  BUGBOT_TRIGGER_EYES="0"
  BUGBOT_TRIGGER_THUMBSUP="0"
fi

# Latest "@greptile" trigger
GREPTILE_TRIGGER_JSON=$(echo "$ALL_ISSUE_COMMENTS" | jq -r '
  [.[] | select(.body == "@greptile")] | last // empty')
if [[ -n "$GREPTILE_TRIGGER_JSON" ]]; then
  GREPTILE_TRIGGER_DATE=$(echo "$GREPTILE_TRIGGER_JSON" | jq -r '.created_at')
  GREPTILE_TRIGGER_EYES=$(echo "$GREPTILE_TRIGGER_JSON" | jq -r '.reactions.eyes')
  GREPTILE_TRIGGER_THUMBSUP=$(echo "$GREPTILE_TRIGGER_JSON" | jq -r '.reactions["+1"]')
else
  GREPTILE_TRIGGER_DATE=""
  GREPTILE_TRIGGER_EYES="0"
  GREPTILE_TRIGGER_THUMBSUP="0"
fi

echo "--- TRIGGERS ---"
if [[ -n "$BUGBOT_TRIGGER_DATE" ]]; then
  echo "Latest \"bugbot run\": $BUGBOT_TRIGGER_DATE (eyes: $BUGBOT_TRIGGER_EYES, thumbsup: $BUGBOT_TRIGGER_THUMBSUP)"
else
  echo "Latest \"bugbot run\": none"
fi
if [[ -n "$GREPTILE_TRIGGER_DATE" ]]; then
  echo "Latest \"@greptile\":  $GREPTILE_TRIGGER_DATE (eyes: $GREPTILE_TRIGGER_EYES, thumbsup: $GREPTILE_TRIGGER_THUMBSUP)"
else
  echo "Latest \"@greptile\":  none"
fi
echo ""
```

**Important:** The `--paginate` flag on `gh api` returns a JSON array per page. Multiple pages produce concatenated arrays. The jq expressions operate on each array, so we must handle this. Using `gh api --paginate` with `--jq` processes each page separately. To get all results into one array, fetch raw JSON and pipe to jq.

**Step 2: Run and verify**

Run: `bash scripts/review-state.sh 35`
Expected: Shows trigger timestamps with reaction counts.

**Step 3: Commit**

```bash
git add scripts/review-state.sh
git commit -m "feat(review-state): add trigger detection with reactions"
```

---

### Task 4: Fetch Bugbot findings (inline review comments)

**Files:**
- Modify: `scripts/review-state.sh`

**Step 1: Add Bugbot section**

Bugbot posts findings as inline PR review comments on the diff. These are accessible via `pulls/{N}/comments` API. Also check `pulls/{N}/reviews` for the summary review body.

```bash
# --- Bugbot ---
# Inline review comments (findings on specific diff lines)
ALL_PR_COMMENTS=$(gh api "repos/$REPO/pulls/$PR_NUMBER/comments?per_page=100" --paginate)

# Filter: cursor[bot] comments newer than latest bugbot trigger
if [[ -n "$BUGBOT_TRIGGER_DATE" ]]; then
  BUGBOT_FINDINGS=$(echo "$ALL_PR_COMMENTS" | jq --arg trigger "$BUGBOT_TRIGGER_DATE" '
    [.[] | select(.user.login == "cursor[bot]" and .created_at > $trigger)]')
else
  BUGBOT_FINDINGS="[]"
fi

BUGBOT_FINDING_COUNT=$(echo "$BUGBOT_FINDINGS" | jq 'length')

# Also get the latest review body (summary)
ALL_PR_REVIEWS=$(gh api "repos/$REPO/pulls/$PR_NUMBER/reviews?per_page=100" --paginate)

if [[ -n "$BUGBOT_TRIGGER_DATE" ]]; then
  BUGBOT_LATEST_REVIEW=$(echo "$ALL_PR_REVIEWS" | jq --arg trigger "$BUGBOT_TRIGGER_DATE" '
    [.[] | select(.user.login == "cursor[bot]" and .submitted_at > $trigger)] | last // empty')
  BUGBOT_REVIEW_DATE=$(echo "$BUGBOT_LATEST_REVIEW" | jq -r '.submitted_at // empty')
  BUGBOT_REVIEW_BODY=$(echo "$BUGBOT_LATEST_REVIEW" | jq -r '.body // empty')
else
  BUGBOT_REVIEW_DATE=""
  BUGBOT_REVIEW_BODY=""
fi

# Determine Bugbot status
BUGBOT_HAS_RESPONSE="false"
if [[ -n "$BUGBOT_REVIEW_DATE" ]] || [[ "$BUGBOT_FINDING_COUNT" -gt 0 ]]; then
  BUGBOT_HAS_RESPONSE="true"
fi

echo "--- BUGBOT (cursor[bot]) ---"
if [[ -z "$BUGBOT_TRIGGER_DATE" ]]; then
  echo "Status: not triggered"
elif [[ "$BUGBOT_HAS_RESPONSE" == "true" ]]; then
  echo "Status: complete"
  echo "Latest review: ${BUGBOT_REVIEW_DATE:-none}"
  echo "Inline comments after trigger: $BUGBOT_FINDING_COUNT"
  echo "Issues found: $BUGBOT_FINDING_COUNT"
  echo ""
  # Print each finding
  for i in $(seq 0 $((BUGBOT_FINDING_COUNT - 1))); do
    FINDING=$(echo "$BUGBOT_FINDINGS" | jq ".[$i]")
    F_PATH=$(echo "$FINDING" | jq -r '.path')
    F_LINE=$(echo "$FINDING" | jq -r '.line // .original_line // "N/A"')
    F_BODY=$(echo "$FINDING" | jq -r '.body')
    # Parse severity from body (Bugbot puts "Low/Medium/High Severity" in markdown)
    F_SEVERITY=$(echo "$F_BODY" | grep -oE '(Low|Medium|High|Critical) Severity' | head -1)
    F_SEVERITY="${F_SEVERITY:- Unknown Severity}"
    echo "Finding $((i + 1)):"
    echo "  File: $F_PATH"
    echo "  Line: $F_LINE"
    echo "  Severity: $F_SEVERITY"
    echo "  Body: $F_BODY"
    echo ""
  done
else
  # No response yet — check if stuck
  NOW_EPOCH=$(date +%s)
  TRIGGER_EPOCH=$(date -jf "%Y-%m-%dT%H:%M:%SZ" "$BUGBOT_TRIGGER_DATE" +%s 2>/dev/null || date -d "$BUGBOT_TRIGGER_DATE" +%s 2>/dev/null || echo "0")
  ELAPSED=$(( NOW_EPOCH - TRIGGER_EPOCH ))
  if [[ "$BUGBOT_TRIGGER_EYES" -gt 0 ]] || [[ "$ELAPSED" -lt 180 ]]; then
    echo "Status: running"
  else
    echo "Status: stuck (triggered ${ELAPSED}s ago, no response, no eyes reaction)"
  fi
fi
echo ""
```

**Step 2: Run and verify**

Run: `bash scripts/review-state.sh 35`
Expected: Shows Bugbot findings with file, line, severity, full body for each.

**Step 3: Commit**

```bash
git add scripts/review-state.sh
git commit -m "feat(review-state): add Bugbot finding extraction"
```

---

### Task 5: Fetch Greptile findings (issue comments)

**Files:**
- Modify: `scripts/review-state.sh`

**Step 1: Add Greptile section**

Greptile posts findings as PR-level issue comments. These are already fetched in `ALL_ISSUE_COMMENTS`. Filter for `greptile-apps[bot]` comments newer than the latest trigger.

```bash
# --- Greptile ---
if [[ -n "$GREPTILE_TRIGGER_DATE" ]]; then
  GREPTILE_COMMENTS=$(echo "$ALL_ISSUE_COMMENTS" | jq --arg trigger "$GREPTILE_TRIGGER_DATE" '
    [.[] | select(.user.login == "greptile-apps[bot]" and .created_at > $trigger)]')
else
  GREPTILE_COMMENTS="[]"
fi

GREPTILE_COMMENT_COUNT=$(echo "$GREPTILE_COMMENTS" | jq 'length')

# Count individual findings within comments (each <details> block or bold file reference is a finding)
# Greptile typically posts comments with multiple findings inside <details><summary>Additional Comments (N)</summary>
# Also posts a main summary comment. Count all of them.
GREPTILE_FINDING_COUNT=0
GREPTILE_HAS_RESPONSE="false"

if [[ "$GREPTILE_COMMENT_COUNT" -gt 0 ]]; then
  GREPTILE_HAS_RESPONSE="true"
  # Count findings: look for "Additional Comments (N)" pattern and the main summary
  for i in $(seq 0 $((GREPTILE_COMMENT_COUNT - 1))); do
    COMMENT_BODY=$(echo "$GREPTILE_COMMENTS" | jq -r ".[$i].body")
    # Count individual findings: each **`file`** or **`file`, line N** pattern
    FILE_REFS=$(echo "$COMMENT_BODY" | grep -cE '\*\*`[^`]+`' || true)
    if [[ "$FILE_REFS" -gt 0 ]]; then
      GREPTILE_FINDING_COUNT=$((GREPTILE_FINDING_COUNT + FILE_REFS))
    elif echo "$COMMENT_BODY" | grep -q "Greptile Summary"; then
      # Main summary comment — count as 1 finding if it has key issues
      GREPTILE_FINDING_COUNT=$((GREPTILE_FINDING_COUNT + 1))
    fi
  done
fi

GREPTILE_LATEST_DATE=""
if [[ "$GREPTILE_COMMENT_COUNT" -gt 0 ]]; then
  GREPTILE_LATEST_DATE=$(echo "$GREPTILE_COMMENTS" | jq -r 'last | .created_at')
fi

echo "--- GREPTILE (greptile-apps[bot]) ---"
if [[ -z "$GREPTILE_TRIGGER_DATE" ]]; then
  echo "Status: not triggered"
elif [[ "$GREPTILE_HAS_RESPONSE" == "true" ]]; then
  echo "Status: complete"
  echo "Latest comment: $GREPTILE_LATEST_DATE"
  echo "Comments after trigger: $GREPTILE_COMMENT_COUNT"
  echo "Issues found: $GREPTILE_FINDING_COUNT"
  echo ""
  # Print each comment's findings
  for i in $(seq 0 $((GREPTILE_COMMENT_COUNT - 1))); do
    COMMENT_BODY=$(echo "$GREPTILE_COMMENTS" | jq -r ".[$i].body")
    COMMENT_DATE=$(echo "$GREPTILE_COMMENTS" | jq -r ".[$i].created_at")
    echo "Comment ($COMMENT_DATE):"
    echo "  Body: $COMMENT_BODY"
    echo ""
  done
else
  NOW_EPOCH=$(date +%s)
  TRIGGER_EPOCH=$(date -jf "%Y-%m-%dT%H:%M:%SZ" "$GREPTILE_TRIGGER_DATE" +%s 2>/dev/null || date -d "$GREPTILE_TRIGGER_DATE" +%s 2>/dev/null || echo "0")
  ELAPSED=$(( NOW_EPOCH - TRIGGER_EPOCH ))
  if [[ "$GREPTILE_TRIGGER_EYES" -gt 0 ]] || [[ "$ELAPSED" -lt 180 ]]; then
    echo "Status: running"
  else
    echo "Status: pending (triggered ${ELAPSED}s ago, no response)"
  fi
fi
echo ""
```

**Step 2: Run and verify**

Run: `bash scripts/review-state.sh 35`
Expected: Shows Greptile findings with full body for each comment.

**Step 3: Commit**

```bash
git add scripts/review-state.sh
git commit -m "feat(review-state): add Greptile finding extraction"
```

---

### Task 6: Derive state machine state and print summary

**Files:**
- Modify: `scripts/review-state.sh`

**Step 1: Add state derivation at the end of the script**

```bash
# --- State derivation ---
STATE=""
ACTION=""

# Rule 1: No triggers
if [[ -z "$BUGBOT_TRIGGER_DATE" ]] && [[ -z "$GREPTILE_TRIGGER_DATE" ]]; then
  STATE="NO_TRIGGERS"
  ACTION="Not in review cycle. Use /delivertask to push and trigger reviews."

# Rule 2: Latest commit newer than latest trigger (stale)
elif [[ -n "$BUGBOT_TRIGGER_DATE" && "$LATEST_COMMIT_DATE" > "$BUGBOT_TRIGGER_DATE" ]] || \
     [[ -n "$GREPTILE_TRIGGER_DATE" && "$LATEST_COMMIT_DATE" > "$GREPTILE_TRIGGER_DATE" ]]; then
  STATE="STALE"
  ACTION="Commits pushed after last trigger. Re-trigger: gh pr comment $PR_NUMBER --body 'bugbot run' && gh pr comment $PR_NUMBER --body '@greptile'"

# Rule 3: Bugbot stuck (trigger >3 min, no response, no eyes)
elif [[ "$BUGBOT_HAS_RESPONSE" == "false" ]] && [[ -n "$BUGBOT_TRIGGER_DATE" ]]; then
  NOW_EPOCH=$(date +%s)
  TRIGGER_EPOCH=$(date -jf "%Y-%m-%dT%H:%M:%SZ" "$BUGBOT_TRIGGER_DATE" +%s 2>/dev/null || date -d "$BUGBOT_TRIGGER_DATE" +%s 2>/dev/null || echo "0")
  ELAPSED=$(( NOW_EPOCH - TRIGGER_EPOCH ))
  if [[ "$ELAPSED" -gt 180 ]] && [[ "$BUGBOT_TRIGGER_EYES" -eq 0 ]]; then
    STATE="BUGBOT_STUCK"
    ACTION="Bugbot may be stuck (${ELAPSED}s, no response). Re-trigger: gh pr comment $PR_NUMBER --body 'bugbot run'"
  else
    STATE="REVIEWS_PENDING"
    ACTION="Reviews in progress. Do NOT push or run bd commands."
  fi

# Rule 4: Partial reviews (only one bot responded)
elif [[ "$BUGBOT_HAS_RESPONSE" == "true" ]] && [[ "$GREPTILE_HAS_RESPONSE" == "false" ]]; then
  STATE="PARTIAL_REVIEWS"
  ACTION="Bugbot complete, waiting for Greptile. Do NOT push."
elif [[ "$BUGBOT_HAS_RESPONSE" == "false" ]] && [[ "$GREPTILE_HAS_RESPONSE" == "true" ]]; then
  STATE="PARTIAL_REVIEWS"
  ACTION="Greptile complete, waiting for Bugbot. Do NOT push."

# Rule 5: Both responded
elif [[ "$BUGBOT_HAS_RESPONSE" == "true" ]] && [[ "$GREPTILE_HAS_RESPONSE" == "true" ]]; then
  TOTAL_ISSUES=$((BUGBOT_FINDING_COUNT + GREPTILE_FINDING_COUNT))
  if [[ "$TOTAL_ISSUES" -gt 0 ]]; then
    STATE="REVIEWS_COMPLETE_WITH_FEEDBACK"
    ACTION="Fix issues locally, then push + re-trigger when ready. Safe to use bd commands."
  else
    STATE="REVIEWS_COMPLETE_ALL_CLEAR"
    ACTION="All reviews passed. Ready for /finishtask to merge."
  fi

# Fallback
else
  STATE="REVIEWS_PENDING"
  ACTION="Reviews in progress. Do NOT push or run bd commands."
fi

# Print summary
BUGBOT_STATUS="pending"
if [[ "$BUGBOT_HAS_RESPONSE" == "true" ]]; then
  BUGBOT_STATUS="complete ($BUGBOT_FINDING_COUNT issues)"
elif [[ -z "$BUGBOT_TRIGGER_DATE" ]]; then
  BUGBOT_STATUS="not triggered"
fi

GREPTILE_STATUS="pending"
if [[ "$GREPTILE_HAS_RESPONSE" == "true" ]]; then
  GREPTILE_STATUS="complete ($GREPTILE_FINDING_COUNT issues)"
elif [[ -z "$GREPTILE_TRIGGER_DATE" ]]; then
  GREPTILE_STATUS="not triggered"
fi

echo "--- STATE ---"
echo "State: $STATE"
echo "Bugbot: $BUGBOT_STATUS"
echo "Greptile: $GREPTILE_STATUS"
echo "Action: $ACTION"
```

**Step 2: Run and verify against PR #35**

Run: `bash scripts/review-state.sh 35`
Expected: Full output with all sections. State should be `REVIEWS_COMPLETE_WITH_FEEDBACK` since both bots have posted findings.

**Step 3: Commit**

```bash
git add scripts/review-state.sh
git commit -m "feat(review-state): add state machine derivation and summary output"
```

---

### Task 7: Handle pagination edge case and make script executable

**Files:**
- Modify: `scripts/review-state.sh`

**Step 1: Fix pagination**

`gh api --paginate` without `--jq` returns one JSON array per page, concatenated. When piped to jq, this means the input may contain multiple root-level arrays. Fix by wrapping the raw output in `jq -s 'add'` to merge all pages into a single array.

Replace all three `gh api --paginate` calls:
```bash
# Instead of:
ALL_ISSUE_COMMENTS=$(gh api "repos/$REPO/issues/$PR_NUMBER/comments?per_page=100" --paginate)
# Use:
ALL_ISSUE_COMMENTS=$(gh api "repos/$REPO/issues/$PR_NUMBER/comments?per_page=100" --paginate | jq -s 'add // []')
```

Same pattern for `ALL_PR_COMMENTS` and `ALL_PR_REVIEWS`.

**Step 2: Make executable**

```bash
chmod +x scripts/review-state.sh
```

**Step 3: Run full test**

Run: `bash scripts/review-state.sh 35`
Expected: Clean output, no jq parse errors, correct state derivation.

**Step 4: Commit**

```bash
git add scripts/review-state.sh
git commit -m "fix(review-state): handle pagination with jq -s add, make executable"
```

---

### Task 8: Update monitorreviews command to remove step 2 API calls

Since the script now does full extraction, the monitorreviews command's step 2 (manual API calls for finding content) is redundant. The command should state that the script output contains everything.

**Files:**
- Modify: `.claude/plugins/harmony-deliver/commands/monitorreviews.md`

**Step 1: Simplify step 2**

Replace the current step 2 (which has manual `gh api` calls) with:

```markdown
### 2. Parse the script output

The script output contains ALL findings with full bodies. Parse the output to:
- Extract the STATE line for the status table
- Extract all findings for the issue summary
- No additional API calls needed — the script handles pagination and both endpoints.
```

**Step 2: Commit**

```bash
git add .claude/plugins/harmony-deliver/commands/monitorreviews.md
git commit -m "docs(monitorreviews): simplify to use review-state.sh full extraction"
```
