---
description: Deliver bead work — commit, push, PR, trigger agent reviews
argument-hint: [bead-id]
allowed-tools: Bash(git add:*), Bash(git status:*), Bash(git commit:*), Bash(git push:*), Bash(git checkout:*), Bash(git branch:*), Bash(git diff:*), Bash(git log:*), Bash(gh pr create:*), Bash(gh pr comment:*), Bash(bd show:*), Bash(bd list:*)
---

## Context

- Git status: !`git status --short`
- Current branch: !`git branch --show-current`
- Changes summary: !`git diff HEAD --stat`
- Recent commits: !`git log --oneline -5`
- Active beads: !`bd list --status=active 2>/dev/null || echo "(no active beads)"`

## Arguments

Bead reference (optional): $ARGUMENTS

If no bead reference is provided, infer from the active beads context above or the current branch name.

## Delivery Workflow

Complete ALL steps below. Use parallel tool calls where steps are independent.

### 1. Identify the work unit

- Determine which bead this delivery is for (from `$ARGUMENTS`, active beads, or branch name)
- If a bead ID is provided, run `bd show <bead-id>` to get its title

### 2. Create task branch (if needed)

- If on `main` or `master`: create and switch to a task branch
  - Branch naming convention: `jake-<crate-short>-<slug>` derived from the bead title
  - Example: bead "transport relay with reverse table" on crate `harmony-reticulum` becomes `jake-harmony-ret-transport-relay-reversetbl`
- If already on a task branch: use it as-is

### 3. Stage and commit

- Stage relevant changed files with `git add <specific-files>` (prefer explicit paths over `git add .`)
- Do NOT stage secrets (.env, credentials, keys, .dolt/)
- Create a commit with a clear message summarizing the work
- End the commit message with:
  ```
  Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
  ```
- Use a HEREDOC for the commit message to ensure correct formatting

### 4. Push to origin

- `git push -u origin <branch-name>`

### 5. Create PR against main

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

### 6. Trigger Bugbot review

After the PR is created, immediately comment on it:

```
gh pr comment <pr-number> --body "bugbot run"
```

Greptile reviews trigger automatically on PR creation — no action needed for Greptile.

### 7. Report

- Print the PR URL
- Print this reminder:

> Do NOT push additional changes while reviews are in progress.
> Pushing resets both Greptile and Bugbot review agents.
> Make local fixes and wait for the signal to push again.
