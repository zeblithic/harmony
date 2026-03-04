---
name: task-lifecycle
description: Use when the user mentions tasks, beads, delivery, PRs, merging, shipping, finding work, claiming work, review status, bugbot, greptile, or any part of the development lifecycle. Also use proactively when work appears complete (tests passing, clippy clean) to suggest /delivertask, when a PR is merged to suggest /findtask, or when reviews may be complete to suggest /monitorreviews. Provides the zeblithic/harmony 4-step task lifecycle with review state machine.
version: 3.0.0
---

# Harmony Task Lifecycle

## The Cycle

Four commands plus a review monitor, each ending with a human-in-the-loop checkpoint:

```
/findtask → /claimtask → /delivertask → [/monitorreviews ...] → /finishtask → /findtask ...
```

| Step | Command | What it does | Stops when |
|------|---------|-------------|------------|
| 1 | `/findtask` | Survey beads, recommend next task | Recommendations presented |
| 2 | `/claimtask <id>` | Claim bead, create branch, start planning | Branch created, planning begins |
| 3 | `/delivertask` | Close bead, push, PR, trigger reviews | PR created, reviews pending |
| 3.5 | `/monitorreviews` | Check review state, report status + next action | State reported |
| 4 | `/finishtask` | Merge PR, clean up, return to main | Main updated, ready for next cycle |

## Helper Scripts (`scripts/`)

These scripts handle deterministic, multi-step operations atomically:

| Script | Used by | Purpose |
|--------|---------|---------|
| `review-state.sh [PR]` | `/monitorreviews`, `/finishtask` | Gather all review signals (paginated), derive state machine state |
| `push-and-trigger.sh [PR]` | Re-delivery cycles | Test gate → push → trigger both reviewers → confirm state |
| `finish-task.sh [PR]` | `/finishtask` | Check review state → merge → main → cleanup → report |
| `survey-work.sh` | `/findtask` | Gather all beads, merges, PRs, bead details in one shot |

## Critical Rules

### Branch-First Rule
**ALWAYS create a task branch BEFORE any work** — including brainstorming, coding, or `bd` commands. `bd sync` and `bd` backup auto-commit and auto-push to the current branch. If on `main`, this pushes unreviewed code directly.

### Merge Strategy
- **Standard merge** (not squash, not rebase) — preserves full commit history
- **Always delete branches** after merge — both remote and local

### Branch Naming
Convention: `jake-<crate-short>-<slug>` derived from the bead title.

Examples:
- "W-TinyLFU cache admission" → `jake-content-wtinylfu-cache`
- "Content transport bridges" → `jake-content-transport-bridges`
- "Debug impls for core types" → `jake-debug-impls`

## Review State Machine

The review cycle lives in the gap between `/delivertask` and `/finishtask`. Use `/monitorreviews` to check current state.

### States

```
/delivertask
  ├── close bead (BEFORE push — bd auto-pushes!)
  ├── commit + push
  ├── "bugbot run" + (first run: automatic greptile / re-run: "@greptile")
  └── STOP → enters REVIEWS_PENDING

  ┌─── REVIEWS_PENDING ─────────────────────────┐
  │ Signals:                                     │
  │ - "eyes" emoji on trigger comment = working  │
  │ - cursor[bot] review posted = Bugbot done    │
  │ - greptile-apps[bot] review posted = done    │
  │ - thumbs-up on trigger comment = all clear   │
  │                                              │
  │ Rules:                                       │
  │ - NO git push (cancels Bugbot)               │
  │ - NO bd commands (bd auto-pushes)            │
  │ - Local edits only                           │
  └──────────────┬──────────────┬────────────────┘
          issues found      all clear
               │                │
               ▼                ▼
     WORKING_ON_FEEDBACK   READY_TO_MERGE
     (safe for bd + edits)  → /finishtask
               │
               ▼
          push fixes
          "bugbot run" + "@greptile"
               │
               ▼
         REVIEWS_PENDING (again)
```

### Reviewer Behavior

| Reviewer | First run trigger | Re-run trigger | Canceled by push? | Completion signal |
|----------|-------------------|----------------|--------------------|--------------------|
| **Cursor Bugbot** | `gh pr comment N --body "bugbot run"` | Same | **YES** — any push cancels it | `cursor[bot]` posts review; thumbs-up on "bugbot run" comment |
| **Greptile** | Automatic on PR creation | `gh pr comment N --body "@greptile"` | No — runs asynchronously | `greptile-apps[bot]` posts review; thumbs-up on PR description (first) or trigger comment (re-run) |

### Signal Detection

| What to look for | Where | Meaning |
|------------------|-------|---------|
| "eyes" emoji on "bugbot run" comment | Comment reactions | Bugbot acknowledged, working |
| "eyes" emoji on "@greptile" comment | Comment reactions | Greptile acknowledged, working |
| `cursor[bot]` review with inline comments | PR reviews | Bugbot finished — check for issues |
| `greptile-apps[bot]` review with analysis | PR reviews | Greptile finished — check for issues |
| Thumbs-up on trigger comment | Comment reactions | Reviewer finished with no blocking issues |
| Commits newer than latest reviewer response | PR commits vs review timestamps | Reviewer results are **stale** — need re-trigger |
| "bugbot run" as last comment, >3 min, no response | PR comments | Bugbot may be stuck — re-trigger |

### Ordering Rules (Critical)

1. **Close beads BEFORE push** — `bd close` auto-commits and pushes; if done after pushing, it cancels Bugbot
2. **Push once, trigger once** — after pushing, immediately trigger "bugbot run" (and "@greptile" on re-runs)
3. **Freeze during REVIEWS_PENDING** — no pushes, no `bd` commands, local edits only
4. **Safe to use `bd` during WORKING_ON_FEEDBACK** — reviews are complete, you're fixing code
5. **Each push resets the cycle** — fix → push → trigger → wait → check
6. **Both reviewers must clear** for automation to proceed — OR human explicitly overrides ("merge it", "good enough")

### Recovery from Accidental Push During Review

If you accidentally push while reviews are pending:
1. Bugbot's run was canceled — results are stale
2. Greptile is fine — it runs asynchronously
3. Re-trigger Bugbot: `gh pr comment N --body "bugbot run"`
4. You do NOT need to re-trigger Greptile unless you want fresh analysis of the new changes

## Context Window Management

Before yielding to the human at any step boundary, check your context usage. If less than ~25% of the context window remains, **compact first** before stopping. A compaction that still leaves you over 75% full is marginal — in that case, warn the human that the session is getting long and suggest restarting after the current step completes.

The goal: never hand control back to the human in a state where the next command they invoke would immediately hit context limits and lose important working state.

## When to Suggest Commands

| Condition | Suggest |
|-----------|---------|
| Just merged a PR / just finished `/finishtask` | `/findtask` |
| User chose a bead from `/findtask` output | `/claimtask <id>` |
| Tests pass + clippy clean + work complete | `/delivertask` |
| PR has open reviews, user asks about status | `/monitorreviews` |
| User asks "are reviews done?", "check bugbot", "what's the PR status?" | `/monitorreviews` |
| Time has passed since `/delivertask`, reviews may be done | `/monitorreviews` |
| PR reviews approved + feedback addressed | `/finishtask` |
| User says "ship it", "deliver", "send for review" | `/delivertask` |
| User says "merge it", "looks good", "wrap it up" | `/finishtask` |
| User says "what's next", "find work" | `/findtask` |
