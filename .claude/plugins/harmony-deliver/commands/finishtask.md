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

Fetch the review signals and verify ALL of these conditions:

```bash
# Latest commit timestamp
gh pr view {number} --json commits --jq '.commits[-1] | {sha: .oid, date: .committedDate}'

# Bugbot reviews (cursor[bot] posts GitHub PR reviews)
gh api repos/{owner}/{repo}/pulls/{number}/reviews \
  --jq '.[] | select(.user.login == "cursor[bot]") | {user: .user.login, state: .state, submitted: .submitted_at}'

# Greptile responses (greptile-apps posts PR comments, NOT GitHub reviews)
gh pr view {number} --comments --json comments \
  --jq '.comments[] | select(.author.login == "greptile-apps") | {author: .author.login, created: .createdAt, bodyPreview: (.body | .[0:80])}'

# Thumbs-up on trigger comments (approval signal for BOTH reviewers)
gh api repos/{owner}/{repo}/issues/{number}/comments \
  --jq '.[] | select(.body == "bugbot run" or .body == "@greptile") | {body: .body, created: .created_at, thumbsup: .reactions["+1"]}'
```

**How each reviewer signals completion:**
- **Bugbot (`cursor[bot]`)**: Posts a GitHub PR **review** (shows in reviews API). Approval = thumbs-up on "bugbot run" trigger comment.
- **Greptile (`greptile-apps`)**: Posts a PR **comment** (NOT a review — won't appear in reviews API). Approval = thumbs-up on "@greptile" trigger comment.

**A PR is ready to merge when:**
1. `cursor[bot]` has posted a review AND `greptile-apps` has posted a comment — both newer than the latest commit
2. The most recent "bugbot run" and "@greptile" trigger comments have thumbs-up reactions (approval)
3. There are **no additional commits** after the approval responses
4. No unresolved HIGH severity issues

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
