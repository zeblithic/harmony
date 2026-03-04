---
description: Check the review status of a PR and determine what action is needed next
argument-hint: [pr-number]
allowed-tools: Bash(gh pr view:*), Bash(gh pr comments:*), Bash(gh api:*), Bash(git log:*), Bash(git branch:*), Bash(bash scripts/review-state.sh:*), Bash(cd:*)
---

## Context

- Harmony repo: `/Users/zeblith/work/zeblithic/harmony`
- Current branch: !`cd /Users/zeblith/work/zeblithic/harmony && git branch --show-current 2>/dev/null || echo "(not in repo)"`
- Open PRs: !`cd /Users/zeblith/work/zeblithic/harmony && gh pr list --state open --limit 5 2>/dev/null || echo "(gh not available)"`

**All workflow commands must run from the harmony repo.** Start with: `cd /Users/zeblith/work/zeblithic/harmony`

## Arguments

PR number (optional): $ARGUMENTS

If no PR number is provided, infer from the current branch's associated PR.

## Review Monitoring Workflow

### 1. Run the review-state script

The script deterministically gathers all review signals and derives the state machine state. It handles pagination and checks both API endpoints.

```bash
cd /Users/zeblith/work/zeblithic/harmony && bash scripts/review-state.sh [PR_NUMBER]
```

This script outputs: trigger timestamps, reaction counts, latest responses from each bot, new comment counts, derived state, and recommended action. It also prints any new findings.

### 2. If state is REVIEWS_COMPLETE_WITH_FEEDBACK or REVIEWS_COMPLETE_ALL_CLEAR: READ the actual content

The script gives you the state and a summary of findings, but you MUST still read the full review content to understand details.

**Where each reviewer posts findings — CRITICAL:**
- **Bugbot (`cursor[bot]`)** posts findings as **inline PR review comments on the diff** — accessible via `pulls/{number}/comments` API (NOT the `issues/{number}/comments` endpoint). These show up as comments on specific lines of code in the PR.
- **Greptile (`greptile-apps[bot]`)** posts findings as **PR-level issue comments** — accessible via `issues/{number}/comments` API or `gh pr view --comments`.
- **These are DIFFERENT API endpoints.** You must check BOTH.

```bash
# Bugbot inline review comments (findings on specific diff lines) — PAGINATED
gh api "repos/zeblithic/harmony/pulls/{number}/comments?per_page=100" --paginate \
  --jq '.[] | select(.user.login == "cursor[bot]") | {path: .path, line: .line, created: .created_at, body: .body[:300]}'

# Bugbot's latest PR review body (summary, may be empty if only inline comments)
gh api "repos/zeblithic/harmony/pulls/{number}/reviews?per_page=100" --paginate \
  --jq '[.[] | select(.user.login == "cursor[bot]")] | last | .body'

# Greptile's latest comment body — PAGINATED
gh api "repos/zeblithic/harmony/issues/{number}/comments?per_page=100" --paginate \
  --jq '[.[] | select(.user.login == "greptile-apps[bot]")] | last | .body'
```

**Scan both for:** suggestions, code changes, severity labels, "could"/"should"/"consider" language, ```suggestion blocks. Any actionable feedback = WITH_FEEDBACK.

**CRITICAL: Thumbs-up on trigger comments = "completed review", NOT "approved". You MUST read the actual review content to determine whether there are actionable issues.**

### 3. Report

Print a clean status table based on the script output:

```
PR:        #<number> — <title>
Bugbot:    <pending|running|complete (N issues)|stale>
Greptile:  <pending|running|complete (N issues)|stale>
State:     <state from script>
Action:    <action from script>
```

If REVIEWS_COMPLETE_WITH_FEEDBACK, summarize the specific issues from BOTH reviewers.

If the state is REVIEWS_PENDING or PARTIAL_REVIEWS, suggest: "While waiting for reviews, this is a good time to `/compact` if context is getting long."

**STOP HERE. Report the state and recommended action, then yield to the human.**
