---
description: Check the review status of a PR and determine what action is needed next
argument-hint: [pr-number]
allowed-tools: Bash(gh pr view:*), Bash(gh pr comments:*), Bash(gh api:*), Bash(git log:*), Bash(git branch:*), Bash(cd:*), Bash(pwd:*)
---

## Context

- Harmony repo: `/Users/zeblith/work/zeblithic/harmony`
- Session directory: !`pwd`
- Git root: !`git rev-parse --show-toplevel 2>/dev/null || echo "(not in repo)"`
- Current branch: !`git branch --show-current 2>/dev/null || echo "(not in repo)"`
- Open PRs: !`cd /Users/zeblith/work/zeblithic/harmony && gh pr list --state open --limit 5 2>/dev/null || echo "(gh not available)"`

## Arguments

PR number (optional): $ARGUMENTS

If no PR number is provided, infer from the current branch's associated PR.

## Review Monitoring Workflow

### 1. Gather review signals

Bugbot and Greptile report in different places:

- **Bugbot**: Appears in the PR **Checks** section (like CI). Also posts inline review comments on the PR.
- **Greptile**: Posts an empty review body in the `reviews` endpoint, actual findings as inline PR review comments in `pulls/{n}/comments`, and a summary in `issues/{n}/comments`.

**NEVER trigger either reviewer.** Bugbot auto-reviews every commit. Greptile auto-triggers on PR creation. Only the human can re-trigger Greptile ($1/review).

```bash
# PR metadata + checks status (Bugbot appears here)
gh pr view <number> --json headRefOid,title,statusCheckRollup --jq '{title: .title, sha: .headRefOid, checks: [.statusCheckRollup[] | select(.name | test("bugbot|cursor"; "i")) | {name: .name, status: .status, conclusion: .conclusion}]}'

# Bugbot inline review comments on the PR
gh api "repos/{owner}/{repo}/pulls/<number>/reviews?per_page=100" --paginate \
  --jq '.[] | select(.user.login == "cursor[bot]") | {user: .user.login, state: .state, submitted_at: .submitted_at, body: .body[:500]}'

# Inline PR review comments (Bugbot findings + Greptile findings)
gh api "repos/{owner}/{repo}/pulls/<number>/comments?per_page=100" --paginate \
  --jq '.[] | select(.user.login == "cursor[bot]" or .user.login == "greptile-apps[bot]") | {user: .user.login, path: .path, line: .line, created_at: .created_at, body: .body[:500]}'

# Issue-level comments (Greptile summary)
gh api "repos/{owner}/{repo}/issues/<number>/comments?per_page=100" --paginate \
  --jq '.[] | select(.user.login == "greptile-apps[bot]") | {user: .user.login, created_at: .created_at, body: .body[:500]}'
```

### 2. Determine state

Compare reviewer response timestamps against the latest commit timestamp:

- **REVIEWS_PENDING**: Reviewers haven't responded yet (or only "eyes" emoji, meaning still working)
- **PARTIAL_REVIEWS**: One reviewer done, other still pending
- **REVIEWS_COMPLETE_ALL_CLEAR**: Both reviewers responded, no actionable issues
- **REVIEWS_COMPLETE_WITH_FEEDBACK**: Both reviewers responded, issues found — summarize them
- **STALE**: Commits are newer than the latest reviewer response — results are outdated

### 3. Report

Print a clean status:

```
PR:        #<number> — <title>
Bugbot:    <pending|complete (N issues)|stale>
Greptile:  <pending|complete (N issues)|stale>
Action:    <what to do next>
```

If there are issues, summarize the specific findings from both reviewers.

If reviews are still pending, suggest: "While waiting, this is a good time to `/compact` if context is getting long."

**STOP HERE. Report the state and recommended action, then yield to the human.**
