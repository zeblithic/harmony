---
name: finalize-workflow
description: This skill should be used when the user says "finalize", "merge it", "looks good let's merge", "ship it to main", "close the PR", "reviews look good", "let's wrap this up", or when a PR's reviews are all approved and the user indicates satisfaction with the result. Provides the zeblithic/harmony finalization workflow for merging reviewed PRs and transitioning to the next task.
version: 1.0.0
---

# Harmony Finalize Workflow

After a PR has been reviewed by Bugbot and Greptile (and any feedback addressed), the finalization process is:

1. **Merge** the PR into main (standard merge via `gh pr merge --merge --delete-branch`)
2. **Switch** to main and pull the latest
3. **Clean up** the local task branch
4. **Survey beads** to find the most promising next task

Use `/finalize` to execute this workflow automatically.

## When to suggest finalization

Proactively suggest `/finalize` when ALL of these are true:
- A PR exists for the current branch
- Bugbot and Greptile reviews are complete
- Any review feedback has been addressed (fixes pushed, re-reviewed)
- The user has indicated they're happy with the state ("looks good", "let's merge", etc.)

## Critical rules

- **Standard merge** (not squash, not rebase) — preserves the full commit history
- **Always delete branches** after merge — both remote (via `--delete-branch`) and local (via `git branch -d`)
- **Pull main after merge** — ensures the local main is up to date before starting new work
- **Always check beads** — after cleanup, survey available tasks so there's no idle time
