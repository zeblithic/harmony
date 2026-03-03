---
description: Merge reviewed PR, clean up branches, return to latest main
argument-hint: [pr-number]
allowed-tools: Bash(git checkout:*), Bash(git branch:*), Bash(git pull:*), Bash(git push:*), Bash(git log:*), Bash(git status:*), Bash(gh pr merge:*), Bash(gh pr view:*)
---

## Context

- Current branch: !`git branch --show-current`
- Git status: !`git status --short`
- Open PRs: !`gh pr list --state open --limit 5 2>/dev/null || echo "(gh not available)"`

## Arguments

PR number (optional): $ARGUMENTS

If no PR number is provided, infer from the current branch's associated PR.

## Finish Task Workflow

Complete ALL steps below in order. These steps are sequential — each depends on the previous.

### 1. Identify the PR

- If `$ARGUMENTS` provides a PR number, use it
- Otherwise, check if the current branch has an open PR: `gh pr view --json number,title,state`
- Confirm the PR is in a mergeable state (approved/no conflicts)
- Print the PR title and number for confirmation

### 2. Merge the PR

- Use standard merge (not squash, not rebase):
  ```
  gh pr merge <number> --merge --delete-branch
  ```
- The `--delete-branch` flag handles remote branch cleanup automatically

### 3. Switch to main and pull

- `git checkout main`
- `git pull origin main`
- Verify the merge commit is present in the log: `git log --oneline -3`

### 4. Delete local task branch

- Delete the local task branch that was just merged:
  ```
  git branch -d <branch-name>
  ```
- Use `-d` (not `-D`) so git confirms the branch is fully merged first

### 5. Report

Print a clean summary:

```
Merged:  PR #<number> — <title>
Branch:  <branch-name> deleted (local + remote)
Main:    up to date
```

Then print:

> Task complete. Use `/findtask` to survey what's next.

**STOP HERE. The human will invoke `/findtask` when ready for the next cycle.**
