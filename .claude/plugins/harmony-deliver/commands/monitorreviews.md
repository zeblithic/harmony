---
description: Check the review status of a PR and determine what action is needed next
argument-hint: [pr-number]
allowed-tools: Bash(gh pr view:*), Bash(gh pr comments:*), Bash(gh api:*), Bash(git log:*), Bash(git branch:*), Bash(bash scripts/review-state.sh:*), Bash(cd:*), Bash(pwd:*)
---

## Context

- Harmony repo: `/Users/zeblith/work/zeblithic/harmony`
- Session directory: !`pwd`
- Git root: !`git rev-parse --show-toplevel 2>/dev/null || echo "(not in repo)"`
- Current branch: !`git branch --show-current 2>/dev/null || echo "(not in repo)"`
- Open PRs: !`cd /Users/zeblith/work/zeblithic/harmony && gh pr list --state open --limit 5 2>/dev/null || echo "(gh not available)"`

**Worktree note:** The review-state script uses the GitHub API with explicit repo paths, so it works from either the main repo or a worktree. Run it from wherever you are.

## Arguments

PR number (optional): $ARGUMENTS

If no PR number is provided, infer from the current branch's associated PR.

## Review Monitoring Workflow

### 1. Run the review-state script

The script deterministically gathers all review signals and derives the state machine state. It handles pagination and checks both API endpoints.

```bash
bash scripts/review-state.sh [PR_NUMBER]
```

This script outputs: trigger timestamps, reaction counts, latest responses from each bot, new comment counts, derived state, and recommended action. It also prints any new findings.

### 2. Parse the script output

The script output contains ALL findings with full bodies from both reviewers. Parse the output to:
- Extract the `--- STATE ---` section for the status table
- Extract all findings from the `--- BUGBOT ---` and `--- GREPTILE ---` sections for the issue summary
- No additional API calls needed — the script handles pagination and both endpoints (inline PR comments for Bugbot, issue comments for Greptile).

**CRITICAL: Thumbs-up on trigger comments = "completed review", NOT "approved". The script extracts full finding bodies — read them to determine whether there are actionable issues.**

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
