---
description: Merge reviewed PR, clean up branches, return to latest main
argument-hint: [pr-number]
allowed-tools: Bash(git checkout:*), Bash(git branch:*), Bash(git pull:*), Bash(git push:*), Bash(git log:*), Bash(git status:*), Bash(gh pr merge:*), Bash(gh pr view:*), Bash(gh api:*), Bash(git add:*), Bash(git commit:*), Bash(git diff:*)
---

## Context

- Current branch: !`git branch --show-current`
- Git status: !`git status --short`
- Uncommitted changes: !`git diff --stat`
- Open PRs: !`gh pr list --state open --limit 5 2>/dev/null || echo "(gh not available)"`

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

Fetch the review signals and verify ALL of these conditions:

```bash
# Latest commit timestamp
gh pr view {number} --json commits --jq '.commits[-1] | {sha: .oid, date: .committedDate}'

# Latest reviews from both bots
gh api repos/{owner}/{repo}/pulls/{number}/reviews \
  --jq '.[] | select(.user.login == "cursor[bot]" or .user.login == "greptile-apps[bot]") | {user: .user.login, submitted: .submitted_at}'

# Thumbs-up on trigger comments (approval signal)
gh api repos/{owner}/{repo}/issues/{number}/comments \
  --jq '.[] | select(.body == "bugbot run" or .body == "@greptile") | {body: .body, created: .created_at, thumbsup: .reactions["+1"]}'
```

**A PR is ready to merge when:**
1. Both `cursor[bot]` and `greptile-apps[bot]` have posted reviews
2. Their reviews are **newer** than the latest commit (not stale)
3. The most recent "bugbot run" and "@greptile" trigger comments have thumbs-up reactions (approval)
4. There are **no additional commits** after the approval reviews
5. No unresolved HIGH severity issues

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
- Verify the merge commit is present in the log: `git log --oneline -3`

### 6. Delete local task branch

- Delete the local task branch that was just merged:
  ```
  git branch -d <branch-name>
  ```
- Use `-d` (not `-D`) so git confirms the branch is fully merged first

### 7. Report

Print a clean summary:

```
Merged:  PR #<number> — <title>
Branch:  <branch-name> deleted (local + remote)
Main:    up to date
```

Then print:

> Task complete. Use `/findtask` to survey what's next.

**STOP HERE. The human will invoke `/findtask` when ready for the next cycle.**
