---
description: Close bead, push task branch, create PR, trigger agent reviews, then stop
argument-hint: [bead-id]
allowed-tools: Bash(git add:*), Bash(git status:*), Bash(git commit:*), Bash(git push:*), Bash(git checkout:*), Bash(git branch:*), Bash(git diff:*), Bash(git log:*), Bash(gh pr create:*), Bash(gh pr comment:*), Bash(bd show:*), Bash(bd list:*), Bash(bd close:*)
---

## Context

- Harmony repo: `/Users/zeblith/work/zeblithic/harmony`
- Git status: !`cd /Users/zeblith/work/zeblithic/harmony && git status --short 2>/dev/null`
- Current branch: !`cd /Users/zeblith/work/zeblithic/harmony && git branch --show-current 2>/dev/null || echo "(not in repo)"`
- Changes summary: !`cd /Users/zeblith/work/zeblithic/harmony && git diff HEAD --stat 2>/dev/null`
- Recent commits: !`cd /Users/zeblith/work/zeblithic/harmony && git log --oneline -5 2>/dev/null`
- Active beads: !`cd /Users/zeblith/work/zeblithic/harmony && bd list --status=active 2>/dev/null || echo "(no active beads)"`

**All workflow commands must run from the harmony repo.** Start with: `cd /Users/zeblith/work/zeblithic/harmony`

## Arguments

Bead reference (optional): $ARGUMENTS

If no bead reference is provided, infer from the active beads context above or the current branch name.

## Delivery Workflow

Complete ALL steps below. Use parallel tool calls where steps are independent.

### 1. Identify the work unit

- Determine which bead this delivery is for (from `$ARGUMENTS`, active beads, or branch name)
- If a bead ID is provided, run `bd show <bead-id>` to get its title

### 2. Ensure work is on a task branch (NOT main)

**CRITICAL: `bd` auto-commits and pushes to the current branch. If on main, this pushes unreviewed code. All work MUST be on a task branch.**

- Branch naming convention: `jake-<crate-short>-<slug>` derived from the bead title

**If already on a task branch:** proceed to step 3.

**If on main:** STOP and warn the user. Do not attempt to retroactively fix this — ask the user how to proceed.

### 3. Close the bead

- Close the bead with `bd close <bead-id> --reason "Delivered — PR pending review"`
- Once work goes up for review, the PR is the state tracker — the bead's job is done

### 4. Stage and commit (if needed)

- If there are unstaged changes, stage with `git add <specific-files>` (prefer explicit paths)
- Do NOT stage secrets (.env, credentials, keys, .dolt/)
- Create a commit with a clear message summarizing the work
- End the commit message with:
  ```
  Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
  ```
- Use a HEREDOC for the commit message to ensure correct formatting

### 5. Push to origin

- `git push -u origin <branch-name>`

### 6. Create PR against main

Use `gh pr create --base main` with this format:

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

### 7. Trigger reviews

After the PR is created, immediately trigger Bugbot:

```
gh pr comment <pr-number> --body "bugbot run"
```

**First delivery:** Greptile triggers automatically on PR creation — no action needed.
**Re-delivery (pushing fixes after feedback):** Greptile does NOT auto-trigger on subsequent pushes. Trigger it explicitly:

```
gh pr comment <pr-number> --body "@greptile"
```

### 8. Stop and report

- Print the PR URL
- Print this reminder:

> **Waiting for reviews.** The PR is now in `REVIEWS_PENDING` state.
> - Do NOT push or run `bd` commands — pushing cancels Bugbot.
> - Use `/monitorreviews` to check review status.
> - For subsequent fix cycles: `bash scripts/push-and-trigger.sh` handles push + both triggers + test gate in one command.
> - When reviews are complete and any feedback is addressed, use `/finishtask` to merge.
> - **Good time to `/compact`** — reviews take a few minutes, reclaim context while waiting.

**STOP HERE. Do not proceed further. The human will review and invoke `/finishtask` when ready.**
