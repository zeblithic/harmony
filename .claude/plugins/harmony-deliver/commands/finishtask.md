---
description: Merge reviewed PR, clean up branches, return to latest main
argument-hint: [pr-number]
allowed-tools: Bash(git checkout:*), Bash(git branch:*), Bash(git pull:*), Bash(git push:*), Bash(git log:*), Bash(git status:*), Bash(gh pr merge:*), Bash(gh pr view:*), Bash(gh api:*), Bash(git add:*), Bash(git commit:*), Bash(git diff:*)
---

## Context

- Harmony repo: `/Users/zeblith/work/zeblithic/harmony`
- Current branch: !`cd /Users/zeblith/work/zeblithic/harmony && git branch --show-current 2>/dev/null || echo "(not in repo)"`
- Git status: !`cd /Users/zeblith/work/zeblithic/harmony && git status --short 2>/dev/null`
- Uncommitted changes: !`cd /Users/zeblith/work/zeblithic/harmony && git diff --stat 2>/dev/null`
- Open PRs: !`cd /Users/zeblith/work/zeblithic/harmony && gh pr list --state open --limit 5 2>/dev/null || echo "(gh not available)"`

**All workflow commands must run from the harmony repo.** Start with: `cd /Users/zeblith/work/zeblithic/harmony`

## Arguments

PR number (optional): $ARGUMENTS

If no PR number is provided, infer from the current branch's associated PR.

## Finish Task Workflow

Complete ALL steps below in order. These steps are sequential — each depends on the previous.

### 1. Identify the PR

- If `$ARGUMENTS` provides a PR number, use it
- Otherwise, check if the current branch has an open PR: `gh pr view --json number,title,state`
- Print the PR title and number for confirmation

### 2. Check for uncommitted local changes

- If there are unstaged or uncommitted changes, stage and commit them before proceeding
- These are changes that need to be part of the PR before merge
- Use `git add <specific-files>` and commit with a clear message

### 3. Verify review state (the "ready to merge" check)

Run the review-state script first for a deterministic check:

```bash
cd /Users/zeblith/work/zeblithic/harmony && bash scripts/review-state.sh {number}
```

If the script reports `REVIEWS_COMPLETE_ALL_CLEAR`, proceed to merge.

If it reports anything else, fetch detail to explain why:

```bash
# Bugbot inline review comments (findings on diff lines) — PAGINATED
gh api "repos/zeblithic/harmony/pulls/{number}/comments?per_page=100" --paginate \
  --jq '.[] | select(.user.login == "cursor[bot]") | {path: .path, line: .line, created: .created_at, body: .body[:300]}'

# Greptile's latest comment — PAGINATED
gh api "repos/zeblithic/harmony/issues/{number}/comments?per_page=100" --paginate \
  --jq '[.[] | select(.user.login == "greptile-apps[bot]")] | last | .body'
```

**Where each reviewer posts findings:**
- **Bugbot (`cursor[bot]`)**: Inline PR **review comments** on the diff (`pulls/{number}/comments` API)
- **Greptile (`greptile-apps[bot]`)**: PR-level **issue comments** (`issues/{number}/comments` API)

**A PR is ready to merge when:**
1. `cursor[bot]` has posted a review/inline comments AND `greptile-apps[bot]` has posted a comment — both newer than the latest commit
2. No unresolved HIGH severity issues in the actual review content
3. There are **no additional commits** after the approval responses

**If any condition fails:**
- If reviews are stale (commits after latest review): warn and suggest pushing + re-triggering reviews
- If a reviewer hasn't responded: warn and suggest `/monitorreviews`
- If there are unresolved issues: warn and list them
- The human can override any of these ("merge it anyway") — but flag it clearly

### 4. Merge the PR

- Use standard merge (not squash, not rebase):
  ```
  gh pr merge <number> --merge --delete-branch
  ```
- The `--delete-branch` flag handles remote branch cleanup automatically

### 5. Switch to main and pull

- `git checkout main`
- `git pull origin main`
- Delete the local task branch: `git branch -d <branch-name>` (safe — git confirms fully merged)
- Verify the merge commit is present in the log: `git log --oneline -3`

### 6. Report

Print a clean summary:

```
Merged:  PR #<number> — <title>
Branch:  <branch-name> deleted (local + remote)
Main:    up to date
```

Then print:

> Task complete. Use `/findtask` to survey what's next.

**STOP HERE. The human will invoke `/findtask` when ready for the next cycle.**
