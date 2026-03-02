---
name: delivery-workflow
description: This skill should be used when the user says "deliver", "ship it", "let's get this reviewed", "create a PR", "push this up", "send for review", mentions "bugbot", or when a bead's work appears to be complete (all tests passing, clippy clean). Provides the zeblithic/harmony delivery workflow for getting agent reviews via Bugbot and Greptile.
version: 1.0.0
---

# Harmony Delivery Workflow

When a unit of work from a bead is complete, the standard delivery process is:

1. **Commit** changes on a task branch (named after the bead)
2. **Push** to origin (once only)
3. **Create PR** against main
4. **Trigger reviews**: Comment "bugbot run" on the PR (Greptile triggers automatically)
5. **Wait**: Do NOT push new changes while reviews are in progress

Use `/deliver` to execute this workflow automatically.

## When to suggest delivery

Proactively suggest `/deliver` when ALL of these are true:
- A bead's implementation is complete
- All tests pass (`cargo test -p <crate>`)
- Clippy is clean (`cargo clippy --workspace`)
- The user hasn't already created a PR for this work
- Changes are uncommitted or committed but not yet pushed

## Critical rules

- **Never push during active reviews** — pushing resets Bugbot and Greptile review agents
- **Always trigger bugbot** — comment "bugbot run" on every new PR (including the first)
- **Greptile is automatic** — it reviews on PR creation, no manual trigger needed
- **One push per review cycle** — the initial push creates the PR; do not push again until reviews complete and feedback is addressed
- **Local fixes only during review** — address review feedback with local edits, stage them, but do NOT push until the user gives the go-ahead
