---
name: task-lifecycle
description: Use when the user mentions tasks, beads, delivery, PRs, merging, shipping, finding work, claiming work, review status, bugbot, greptile, or any part of the development lifecycle. Also use proactively when work appears complete (tests passing, clippy clean) to suggest /delivertask, when a PR is merged to suggest /findtask, or when reviews may be complete to suggest /monitorreviews. Provides the zeblithic/harmony 4-step task lifecycle with review state machine.
version: 3.1.0
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
  ├── (Bugbot auto-reviews; Greptile auto-triggers on PR creation)
  └── STOP → enters REVIEWS_PENDING

  ┌─── REVIEWS_PENDING ─────────────────────────┐
  │ Signals:                                     │
  │ - "eyes" emoji on trigger comment = working  │
  │ - cursor[bot] review posted = Bugbot done    │
  │ - greptile-apps[bot] review posted = done    │
  │ - thumbs-up on trigger comment = all clear   │
  │                                              │
  │ Rules:                                       │
  │ - NO git push (wait for reviews)             │
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
          (Bugbot auto-reviews new commit)
               │
               ▼
         REVIEWS_PENDING (again)
```

### Reviewer Behavior

| Reviewer | Trigger | Re-run | Cost | Completion signal |
|----------|---------|--------|------|-------------------|
| **Cursor Bugbot** | Automatic on every PR and every commit | Automatic | $40/month flat | `cursor[bot]` posts review |
| **Greptile** | Automatic on PR creation only | **ONLY the human** comments `@greptile` — agent must NEVER do this | **$1 per review** | `greptile-apps[bot]` posts review |

**CRITICAL: The agent must NEVER comment "bugbot run" or "@greptile" on any PR.** Bugbot triggers automatically. Greptile re-runs are a spending decision that only the human makes.

### Signal Detection

| What to look for | Where | Meaning |
|------------------|-------|---------|
| Bugbot check status (pending/success/failure) | PR Checks section (`gh pr checks`) | Bugbot auto-reviewing or complete |
| `cursor[bot]` review with inline comments | PR reviews + inline comments | Bugbot findings — check for issues |
| `greptile-apps[bot]` review with analysis | PR reviews + inline comments | Greptile finished — check for issues |
| Commits newer than latest reviewer response | PR commits vs review timestamps | Bugbot will auto-review new commit; Greptile results may be stale (only human can re-trigger) |

### Ordering Rules (Critical)

1. **Close beads BEFORE push** — `bd close` auto-commits and pushes; if done after pushing, it creates noise during review
2. **Push once, wait** — after pushing, Bugbot auto-reviews. Do NOT trigger anything.
3. **NEVER comment "bugbot run" or "@greptile"** — Bugbot is automatic; Greptile re-runs are the human's spending decision ($1/review)
4. **Freeze during REVIEWS_PENDING** — no pushes, no `bd` commands, local edits only
5. **Safe to use `bd` during WORKING_ON_FEEDBACK** — reviews are complete, you're fixing code
6. **Each push resets Bugbot** — fix → push → Bugbot auto-reviews → wait → check
7. **Both reviewers must clear** for automation to proceed — OR human explicitly overrides ("merge it", "good enough")

### Recovery from Accidental Push During Review

If you accidentally push while reviews are pending:
1. Bugbot will auto-review the new commit — no action needed
2. Greptile's initial review is unaffected — it runs asynchronously
3. Do NOT trigger any reviewers — wait for Bugbot to pick up the new commit automatically

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
