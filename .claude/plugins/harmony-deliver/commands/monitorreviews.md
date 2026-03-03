---
description: Check the review status of a PR and determine what action is needed next
argument-hint: [pr-number]
allowed-tools: Bash(gh pr view:*), Bash(gh pr comments:*), Bash(gh api:*), Bash(git log:*), Bash(git branch:*)
---

## Context

- Current branch: !`git branch --show-current`
- Open PRs: !`gh pr list --state open --limit 5 2>/dev/null || echo "(gh not available)"`

## Arguments

PR number (optional): $ARGUMENTS

If no PR number is provided, infer from the current branch's associated PR.

## Review Monitoring Workflow

### 1. Identify the PR

- If `$ARGUMENTS` provides a PR number, use it
- Otherwise: `gh pr view --json number,title,url,state,headRefName`
- If no open PR found, report and stop

### 2. Gather review signals

Fetch ALL of these in parallel:

```bash
# All reviews (Greptile, Bugbot, humans)
gh api repos/{owner}/{repo}/pulls/{number}/reviews \
  --jq '.[] | {user: .user.login, state: .state, submitted: .submitted_at}'

# All inline review comments with timestamps
gh api repos/{owner}/{repo}/pulls/{number}/comments \
  --jq '.[] | {user: .user.login, created: .created_at, id: .id}'

# All PR-level comments (includes "bugbot run" triggers)
gh pr view {number} --comments --json comments \
  --jq '.comments[] | {author: .author.login, body: .body, created: .createdAt}'

# Latest commit SHA and timestamp
gh pr view {number} --json commits \
  --jq '.commits[-1] | {sha: .oid, date: .committedDate}'

# Reactions on comments (thumbs-up signals completion)
# Check the "bugbot run" trigger comments for reactions
gh api repos/{owner}/{repo}/issues/{number}/comments \
  --jq '.[] | select(.body == "bugbot run" or .body == "@greptile") | {id: .id, body: .body, created: .created_at, reactions: .reactions}'
```

### 3. Determine review state

Apply these rules in order to determine the current state:

#### State: REVIEWS_PENDING

**Condition:** "bugbot run" or "@greptile" comment exists that is newer than the latest `cursor[bot]` or `greptile-apps[bot]` response, AND either:
- The trigger comment has an "eyes" reaction (reviewer acknowledged, working on it)
- Less than 3 minutes have passed since the trigger comment (still starting up)

**Action:** Report "Reviews in progress. Do NOT push or run `bd` commands." and STOP.

#### State: BUGBOT_STUCK

**Condition:** "bugbot run" comment exists, more than 3 minutes old, no "eyes" reaction, no `cursor[bot]` response after it.

**Action:** Report "Bugbot may be stuck. Recommend re-triggering with `gh pr comment {number} --body 'bugbot run'`"

#### State: BUGBOT_CANCELED

**Condition:** Commits exist AFTER the latest `cursor[bot]` review. The push canceled the Bugbot run.

**Action:** Report "Bugbot results are stale (commits pushed after last review). Need to re-trigger: `gh pr comment {number} --body 'bugbot run'`"

#### State: REVIEWS_COMPLETE_WITH_FEEDBACK

**Condition:** Both `cursor[bot]` and `greptile-apps[bot]` have posted reviews that are newer than the latest push, AND at least one found issues.

**Action:** Report the issues summary. Print: "Reviews complete with feedback. Fix issues locally, then push + re-trigger when ready. Safe to use `bd` commands during fix work."

#### State: REVIEWS_COMPLETE_ALL_CLEAR

**Condition:** Both `cursor[bot]` and `greptile-apps[bot]` have posted reviews newer than the latest push, with no HIGH severity issues and/or thumbs-up reactions on trigger comments.

**Action:** Report: "All reviews passed. Ready for `/finishtask` to merge."

#### State: PARTIAL_REVIEWS

**Condition:** Only one reviewer has responded so far.

**Action:** Report which reviewer is done and which is still pending. Print: "Waiting for remaining reviewer(s). Do NOT push."

### 4. Report

Print a clean status table:

```
PR:        #<number> — <title>
Bugbot:    <pending|running|complete (N issues)|stale>
Greptile:  <pending|running|complete (N issues)|stale>
State:     <REVIEWS_PENDING|BUGBOT_STUCK|REVIEWS_COMPLETE_WITH_FEEDBACK|REVIEWS_COMPLETE_ALL_CLEAR|PARTIAL_REVIEWS>
Action:    <what to do next>
```

**STOP HERE. Report the state and recommended action, then yield to the human.**
