---
description: Close bead, push task branch, create PR, trigger agent reviews, then stop
argument-hint: [bead-id]
allowed-tools: Bash(git add:*), Bash(git status:*), Bash(git commit:*), Bash(git push:*), Bash(git checkout:*), Bash(git branch:*), Bash(git diff:*), Bash(git log:*), Bash(git rev-parse:*), Bash(gh pr create:*), Bash(gh pr comment:*), Bash(bd show:*), Bash(bd list:*), Bash(bd close:*), Bash(cd:*), Bash(pwd:*)
---

## Context

- Harmony repo: `/Users/zeblith/work/zeblithic/harmony`
- Session directory: !`pwd`
- Git root: !`git rev-parse --show-toplevel 2>/dev/null || echo "(not in repo)"`
- Current branch: !`git branch --show-current 2>/dev/null || echo "(not in repo)"`
- Git status: !`git status --short 2>/dev/null`
- Changes summary: !`git diff HEAD --stat 2>/dev/null`
- Recent commits: !`git log --oneline -5 2>/dev/null`
- Active beads: !`cd /Users/zeblith/work/zeblithic/harmony && bd list --status=active 2>/dev/null || echo "(no active beads)"`

## Arguments

Bead reference (optional): $ARGUMENTS

If no bead reference is provided, infer from the active beads context above or the current branch name.

## Worktree Awareness

**Detect worktree vs main repo:** Compare `git rev-parse --show-toplevel` with `/Users/zeblith/work/zeblithic/harmony`.

- **If they match:** You're in the main repo. Standard flow.
- **If they differ:** You're in a worktree. Follow the worktree-specific notes below.

**In a worktree:**
- `bd` commands (close, show, list) must run from the main repo: `cd /Users/zeblith/work/zeblithic/harmony && bd ...`
- `git` commands (add, commit, push) run from the worktree (where the branch is checked out)
- `gh pr create` runs from the worktree (so HEAD resolves to the task branch, not main)
- `gh pr comment` works from either location (uses explicit PR number)

## Delivery Workflow

Complete ALL steps below. Use parallel tool calls where steps are independent.

### 1. Identify the work unit

- Determine which bead this delivery is for (from `$ARGUMENTS`, active beads, or branch name)
- If a bead ID is provided, run `cd /Users/zeblith/work/zeblithic/harmony && bd show <bead-id>` to get its title

### 2. Ensure work is on a task branch (NOT main)

**CRITICAL: `bd` auto-commits and pushes to the current branch. If on main, this pushes unreviewed code. All work MUST be on a task branch.**

- Branch naming convention: `jake-<crate-short>-<slug>` derived from the bead title

**If already on a task branch:** proceed to step 3.

**If on main:** STOP and warn the user. Do not attempt to retroactively fix this — ask the user how to proceed.

### 3. Close the bead

- Close the bead from the main repo: `cd /Users/zeblith/work/zeblithic/harmony && bd close <bead-id> --reason "Delivered — PR pending review"`
- Once work goes up for review, the PR is the state tracker — the bead's job is done

### 4. Stage and commit (if needed)

- If there are unstaged changes, stage with `git add <specific-files>` (prefer explicit paths)
- Do NOT stage secrets (.env, credentials, keys, .dolt/)
- **In a worktree:** run git commands from the worktree directory (where the branch is checked out)
- Create a commit with a clear message summarizing the work
- End the commit message with:
  ```
  Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
  ```
- Use a HEREDOC for the commit message to ensure correct formatting

### 5. Push to origin

- From the worktree (or main repo if not using worktrees): `git push -u origin <branch-name>`

### 6. Create PR against main

**From the worktree** (so HEAD resolves correctly), or use `--head <branch-name>` if running from main repo:

```
gh pr create --title "<concise title, under 70 chars>" --body "$(cat <<'EOF'
## Summary
<1-3 bullet points describing what changed and why>

## Test plan
- [ ] `cargo test -p <crate>` — all tests pass
- [ ] `cargo clippy --workspace` — zero warnings

Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

### 7. Stop and report

**Do NOT trigger any reviewers.** Bugbot auto-reviews every commit (appears in PR Checks section). Greptile auto-triggers on PR creation. The agent must NEVER comment "bugbot run" or "@greptile".

- Print the PR URL
- Print this reminder:

> **Waiting for reviews.** The PR is now in `REVIEWS_PENDING` state.
> - Do NOT push or run `bd` commands while reviews are pending.
> - Bugbot auto-reviews (check PR Checks section). Greptile auto-triggered on PR creation.
> - Use `/monitorreviews` to check review status.
> - When reviews are complete and any feedback is addressed, use `/finishtask` to merge.
> - **Good time to `/compact`** — reviews take a few minutes, reclaim context while waiting.

**STOP HERE. Do not proceed further. The human will review and invoke `/finishtask` when ready.**
