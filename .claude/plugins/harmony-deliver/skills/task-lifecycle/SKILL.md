---
name: task-lifecycle
description: Use when the user mentions tasks, beads, delivery, PRs, merging, shipping, finding work, claiming work, or any part of the development lifecycle. Also use proactively when work appears complete (tests passing, clippy clean) to suggest /delivertask, or when a PR is merged to suggest /findtask. Provides the zeblithic/harmony 4-step task lifecycle.
version: 2.0.0
---

# Harmony Task Lifecycle

## The Cycle

Four commands, each ending with a human-in-the-loop checkpoint:

```
/findtask → (human chooses) → /claimtask → (human works) → /delivertask → (reviews) → /finishtask → /findtask ...
```

| Step | Command | What it does | Stops when |
|------|---------|-------------|------------|
| 1 | `/findtask` | Survey beads, recommend next task | Recommendations presented |
| 2 | `/claimtask <id>` | Claim bead, create branch, start planning | Branch created, planning begins |
| 3 | `/delivertask` | Close bead, push, PR, trigger reviews | PR created, reviews pending |
| 4 | `/finishtask` | Merge PR, clean up, return to main | Main updated, ready for next cycle |

## Critical Rules

### Branch-First Rule
**ALWAYS create a task branch BEFORE any work** — including brainstorming, coding, or `bd` commands. `bd sync` and `bd` backup auto-commit and auto-push to the current branch. If on `main`, this pushes unreviewed code directly.

### Review Discipline
- **Never push during active reviews** — pushing resets Bugbot and Greptile agents
- **Always trigger bugbot** — comment "bugbot run" on every PR
- **Greptile is automatic** — triggers on PR creation
- **Local fixes only during review** — edit locally, do NOT push until reviews complete

### Merge Strategy
- **Standard merge** (not squash, not rebase) — preserves full commit history
- **Always delete branches** after merge — both remote and local

### Branch Naming
Convention: `jake-<crate-short>-<slug>` derived from the bead title.

Examples:
- "W-TinyLFU cache admission" → `jake-content-wtinylfu-cache`
- "Content transport bridges" → `jake-content-transport-bridges`
- "Debug impls for core types" → `jake-debug-impls`

## When to Suggest Commands

| Condition | Suggest |
|-----------|---------|
| Just merged a PR / just finished `/finishtask` | `/findtask` |
| User chose a bead from `/findtask` output | `/claimtask <id>` |
| Tests pass + clippy clean + work complete | `/delivertask` |
| PR reviews approved + feedback addressed | `/finishtask` |
| User says "ship it", "deliver", "send for review" | `/delivertask` |
| User says "merge it", "looks good", "wrap it up" | `/finishtask` |
| User says "what's next", "find work" | `/findtask` |
