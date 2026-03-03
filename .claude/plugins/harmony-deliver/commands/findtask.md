---
description: Survey beads and project state to determine what to work on next
allowed-tools: Bash(bd list:*), Bash(bd show:*), Bash(bd ready:*), Bash(git log:*), Bash(git branch:*), Bash(gh pr list:*)
---

## Context

- Harmony repo: `/Users/zeblith/work/zeblithic/harmony`
- Available beads: !`cd /Users/zeblith/work/zeblithic/harmony && bd ready 2>/dev/null || echo "(bd not available)"`
- All beads: !`cd /Users/zeblith/work/zeblithic/harmony && bd list 2>/dev/null || echo "(bd not available)"`
- Recent merges: !`cd /Users/zeblith/work/zeblithic/harmony && git log --oneline --merges -5 2>/dev/null || echo "(no merges)"`
- Current branch: !`cd /Users/zeblith/work/zeblithic/harmony && git branch --show-current 2>/dev/null || echo "(not in repo)"`
- Open PRs: !`cd /Users/zeblith/work/zeblithic/harmony && gh pr list --state open --limit 5 2>/dev/null || echo "(no open PRs)"`

**All workflow commands must run from the harmony repo.** Start with: `cd /Users/zeblith/work/zeblithic/harmony`

## Find Task Workflow

Help determine the most promising next unit of work.

### 1. Survey available work

- Review the beads listed above (both `ready` and `list` output)
- For each candidate bead, run `bd show <bead-id>` to understand its scope, dependencies, and description
- Check if any beads have unsatisfied dependencies (blocked)

### 2. Assess context

- Check recent merges and PRs to understand what was just completed
- Consider whether any recent work creates momentum toward a particular bead
- Look at priority levels (P1 > P2 > P3 > P4)
- Check if any project design docs in `docs/plans/` are relevant

### 3. Recommend next task

Present 2-3 candidate beads ranked by impact, with a brief rationale for each:

```
1. **<bead-id>** (P<n>) — <title>
   <Why this is a good next step — 1-2 sentences>

2. **<bead-id>** (P<n>) — <title>
   <Why this is a good next step — 1-2 sentences>
```

Lead with your top recommendation and explain why.

### 4. Stop and wait

**Do NOT proceed to claiming or starting work.** Present the recommendations and let the user choose. The user will invoke `/claimtask <bead-id>` when ready.
