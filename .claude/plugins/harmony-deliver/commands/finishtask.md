---
description: Merge reviewed PR, clean up branches, return to latest main
argument-hint: [pr-number]
allowed-tools: Bash(git checkout:*), Bash(git branch:*), Bash(git pull:*), Bash(git push:*), Bash(git log:*), Bash(git status:*), Bash(git worktree:*), Bash(git rev-parse:*), Bash(gh pr merge:*), Bash(gh pr view:*), Bash(gh api:*), Bash(git add:*), Bash(git commit:*), Bash(git diff:*), Bash(cd:*), Bash(pwd:*)
---

## Context

- Harmony repo: `/Users/zeblith/work/zeblithic/harmony`
- Session directory: !`pwd`
- Git root: !`git rev-parse --show-toplevel 2>/dev/null || echo "(not in repo)"`
- Current branch: !`git branch --show-current 2>/dev/null || echo "(not in repo)"`
- Git status: !`git status --short 2>/dev/null`
- Uncommitted changes: !`git diff --stat 2>/dev/null`
- Open PRs: !`cd /Users/zeblith/work/zeblithic/harmony && gh pr list --state open --limit 5 2>/dev/null || echo "(gh not available)"`

## Arguments

PR number (optional): $ARGUMENTS

If no PR number is provided, infer from the current branch's associated PR.

## Finish Task Workflow

### 1. Identify the PR

- If `$ARGUMENTS` provides a PR number, use it
- Otherwise: `gh pr view --json number,title,state`

### 2. Handle uncommitted local changes

- Stage and commit any remaining changes before merging
- Use `git add <specific-files>` and commit with a clear message

### 3. Verify reviews are clear

Check that the human has confirmed reviews are acceptable. If unsure, ask — the human can override ("merge it anyway") but flag the state clearly.

### 4. Merge, switch to main, clean up

**CRITICAL ORDERING: If in a worktree, remove the worktree BEFORE merging.** `gh pr merge --delete-branch` tries to delete the local branch, which fails if a worktree is using it. Worse, if the shell CWD is inside the worktree, all subsequent commands break.

**If in a worktree:**
```bash
# 1. Save branch name before leaving the worktree
BRANCH=$(git branch --show-current)
WORKTREE=$(git rev-parse --show-toplevel)

# 2. Switch to main repo FIRST (escape the worktree CWD)
MAIN_REPO=$(git worktree list --porcelain | head -1 | sed 's/^worktree //')
cd "$MAIN_REPO"

# 3. Remove the worktree (frees the local branch)
git worktree remove "$WORKTREE"

# 4. Now merge (local + remote branch deletion works cleanly)
gh pr merge <number> --merge --delete-branch

# 5. Pull and verify
git pull origin main
git log --oneline -3
```

**If in main repo (standard):**
```bash
gh pr merge <number> --merge --delete-branch
git checkout main && git pull origin main
git log --oneline -3
```

### 5. Report

```
Merged:  PR #<number> — <title>
Branch:  <branch-name> deleted (local + remote)
Main:    up to date
```

> Task complete. Use `/findtask` to survey what's next.

**STOP HERE. The human will invoke `/findtask` when ready for the next cycle.**
