---
description: Claim a bead, create a task branch, and start planning the work
argument-hint: <bead-id>
allowed-tools: Bash(bd show:*), Bash(bd update:*), Bash(git checkout:*), Bash(git branch:*), Bash(git pull:*), Bash(git status:*), Bash(git worktree:*)
---

## Context

- Harmony repo: `/Users/zeblith/work/zeblithic/harmony`
- Current branch: !`cd /Users/zeblith/work/zeblithic/harmony && git branch --show-current 2>/dev/null || echo "(not in repo)"`
- Git status: !`cd /Users/zeblith/work/zeblithic/harmony && git status --short 2>/dev/null`
- Bead argument: $ARGUMENTS

**All workflow commands must run from the harmony repo.** Start with: `cd /Users/zeblith/work/zeblithic/harmony`

## Arguments

Bead ID (required): $ARGUMENTS

If no bead ID is provided, ask the user which bead to claim (suggest running `/findtask` first).

## Claim Task Workflow

### 1. Validate the bead

- Run `bd show $ARGUMENTS` to get the bead's full details (title, description, dependencies, status)
- Verify the bead is not already closed or in a blocked state
- If the bead has unsatisfied dependencies, warn the user and ask if they want to proceed anyway

### 2. Claim the bead

- Run `bd update $ARGUMENTS --claim` to mark it as claimed/in-progress
- `bd` commands must run from the main repo (`cd /Users/zeblith/work/zeblithic/harmony`)

### 3. Switch to latest main

- `cd /Users/zeblith/work/zeblithic/harmony`
- `git checkout main`
- `git pull origin main`
- Verify the working tree is clean (`git status --short` should be empty)
- If there are uncommitted changes on main, warn the user before proceeding

### 4. Create worktree and task branch

- Derive branch name from the bead title: `jake-<crate-short>-<slug>`
  - Example: bead "Content transport bridges" on harmony-content becomes `jake-content-transport-bridges`
  - Example: bead "Debug impls for core types" becomes `jake-debug-impls`
- Create a worktree with a new branch (from the main repo):
  ```bash
  cd /Users/zeblith/work/zeblithic/harmony
  git worktree add .claude/worktrees/<branch-name> -b <branch-name>
  ```
- Switch to the worktree for all subsequent work:
  ```bash
  cd /Users/zeblith/work/zeblithic/harmony/.claude/worktrees/<branch-name>
  ```

### 5. Report and begin planning

Print a summary:

```
Claimed:   <bead-id> — <title>
Branch:    <branch-name>
Worktree:  .claude/worktrees/<branch-name>
```

**All subsequent work happens in the worktree.** The main repo stays on `main` — this prevents `bd` auto-pushes from landing on `main` and keeps worktrees isolated from each other.

Then begin helping the user plan and execute the work:

- If the bead references design docs or plans, read them
- If the work scope is non-trivial, suggest brainstorming or planning
- If the work is straightforward, offer to start implementation

**The user drives what happens next** — brainstorming, planning, or diving straight into code.
