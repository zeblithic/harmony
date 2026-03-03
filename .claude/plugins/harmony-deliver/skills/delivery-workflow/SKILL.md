---
name: delivery-workflow
description: This skill should be used when the user says "deliver", "ship it", "let's get this reviewed", "create a PR", "push this up", "send for review", mentions "bugbot", or when a bead's work appears to be complete (all tests passing, clippy clean). Provides the zeblithic/harmony delivery workflow for getting agent reviews via Bugbot and Greptile.
version: 1.0.0
---

# Harmony Delivery Workflow

## CRITICAL: Branch-First Rule

**ALWAYS create a task branch BEFORE starting any work — including brainstorming, coding, or `bd` commands.**

`bd sync` and `bd` backup auto-commit and auto-push to the current branch. If you're on `main`, this pushes unreviewed code directly to main. This has caused painful retroactive PR cleanups. The fix is simple: create the task branch first.

**When picking up a bead:**
1. `git checkout -b jake-<crate>-<slug>` — create task branch FIRST
2. Then do all work (brainstorm, code, tests, `bd` interactions) on the branch
3. When done, `/deliver` to ship

## Delivery Process

When a unit of work from a bead is complete:

1. **Close bead** — the PR becomes the state tracker from here
2. **Commit** changes on the task branch
3. **Push** to origin (once only)
4. **Create PR** against main
5. **Trigger reviews**: Comment "bugbot run" on the PR (Greptile triggers automatically)
6. **Wait**: Do NOT push new changes while reviews are in progress

Use `/deliver` to execute this workflow automatically.

## When to suggest delivery

Proactively suggest `/deliver` when ALL of these are true:
- A bead's implementation is complete
- All tests pass (`cargo test -p <crate>`)
- Clippy is clean (`cargo clippy --workspace`)
- The user hasn't already created a PR for this work
- Changes are uncommitted or committed but not yet pushed

## Critical rules

- **Branch first, always** — never start work on main; `bd` auto-pushes will ruin your day
- **Never push during active reviews** — pushing resets Bugbot and Greptile review agents
- **Always trigger bugbot** — comment "bugbot run" on every new PR (including the first)
- **Greptile is automatic** — it reviews on PR creation, no manual trigger needed
- **One push per review cycle** — the initial push creates the PR; do not push again until reviews complete and feedback is addressed
- **Local fixes only during review** — address review feedback with local edits, stage them, but do NOT push until the user gives the go-ahead
