---
description: Survey beads and project state to determine what to work on next
allowed-tools: Bash(bd list:*), Bash(bd show:*), Bash(bd ready:*), Bash(git log:*), Bash(git branch:*), Bash(gh pr list:*), Bash(cd:*)
---

## Context

- Harmony repo: `/Users/zeblith/work/zeblithic/harmony`
- Current branch: !`cd /Users/zeblith/work/zeblithic/harmony && git branch --show-current 2>/dev/null || echo "(not in repo)"`
- Open PRs: !`cd /Users/zeblith/work/zeblithic/harmony && gh pr list --state open --limit 5 2>/dev/null || echo "(no open PRs)"`

**All workflow commands must run from the harmony repo.** Start with: `cd /Users/zeblith/work/zeblithic/harmony`

## Find Task Workflow

Help determine the most promising next unit of work.

### 1. Survey available work

Gather context using `bd` and git directly:

```bash
cd /Users/zeblith/work/zeblithic/harmony

# Ready beads (unblocked)
bd ready

# Recent merges for momentum context
git log --oneline --merges -10

# Open PRs
gh pr list --state open --limit 10
```

If specific beads look interesting, `bd show <id>` for details.

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
