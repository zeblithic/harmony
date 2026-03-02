---
description: Finalize reviewed PR — merge, clean up branches, find next task
argument-hint: [pr-number]
allowed-tools: Bash(git checkout:*), Bash(git branch:*), Bash(git pull:*), Bash(git push:*), Bash(git log:*), Bash(git status:*), Bash(gh pr merge:*), Bash(gh pr view:*), Bash(bd list:*), Bash(bd show:*)
---

## Context

- Current branch: !`git branch --show-current`
- Git status: !`git status --short`
- Open PRs: !`gh pr list --state open --limit 5 2>/dev/null || echo "(gh not available)"`
- Active beads: !`bd list 2>/dev/null | head -10 || echo "(no beads)"`

## Arguments

PR number (optional): $ARGUMENTS

If no PR number is provided, infer from the current branch's associated PR.

## Finalize Workflow

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

### 5. Survey beads for next work

- Run `bd list` to show all available beads
- Analyze which tasks are ready to work on (considering dependencies, priorities)
- Suggest the most promising next task with a brief rationale
- If a bead has dependencies, note whether they're satisfied

### 6. Report

- Confirm the PR was merged and branches cleaned up
- Show the suggested next bead(s) to work on
- Ask if I should start working on the suggested task
