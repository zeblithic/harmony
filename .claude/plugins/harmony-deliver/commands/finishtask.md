---
description: Merge reviewed PR, clean up branches, return to latest main
argument-hint: [pr-number]
allowed-tools: Bash(git checkout:*), Bash(git branch:*), Bash(git pull:*), Bash(git push:*), Bash(git log:*), Bash(git status:*), Bash(gh pr merge:*), Bash(gh pr view:*), Bash(gh api:*), Bash(git add:*), Bash(git commit:*), Bash(git diff:*), Bash(bash scripts/finish-task.sh:*), Bash(bash scripts/review-state.sh:*), Bash(cd:*)
---

## Context

- Harmony repo: `/Users/zeblith/work/zeblithic/harmony`
- Current branch: !`cd /Users/zeblith/work/zeblithic/harmony && git branch --show-current 2>/dev/null || echo "(not in repo)"`
- Git status: !`cd /Users/zeblith/work/zeblithic/harmony && git status --short 2>/dev/null`
- Uncommitted changes: !`cd /Users/zeblith/work/zeblithic/harmony && git diff --stat 2>/dev/null`
- Open PRs: !`cd /Users/zeblith/work/zeblithic/harmony && gh pr list --state open --limit 5 2>/dev/null || echo "(gh not available)"`

**All workflow commands must run from the harmony repo.** Start with: `cd /Users/zeblith/work/zeblithic/harmony`

## Arguments

PR number (optional): $ARGUMENTS

If no PR number is provided, infer from the current branch's associated PR.

## Finish Task Workflow

### Quick path (preferred)

If there are no uncommitted changes, use the finish-task script which handles everything atomically:

```bash
cd /Users/zeblith/work/zeblithic/harmony && bash scripts/finish-task.sh [PR_NUMBER]
```

The script: checks review state → merges (standard merge) → switches to main → pulls → deletes local branch → reports. It refuses to merge unless state is `REVIEWS_COMPLETE_ALL_CLEAR` (use `--force` to override).

If the script succeeds, print the output and stop. If it fails, fall back to the manual steps below.

### Manual path (when script can't handle it)

Use this when there are uncommitted changes or the script needs to be overridden.

#### 1. Identify the PR

- If `$ARGUMENTS` provides a PR number, use it
- Otherwise: `gh pr view --json number,title,state`

#### 2. Handle uncommitted local changes

- Stage and commit any remaining changes before merging
- Use `git add <specific-files>` and commit with a clear message

#### 3. Verify review state

```bash
cd /Users/zeblith/work/zeblithic/harmony && bash scripts/review-state.sh {number}
```

If not `REVIEWS_COMPLETE_ALL_CLEAR`, read the actual findings:

```bash
# Bugbot inline comments (findings on diff lines) — PAGINATED
gh api "repos/zeblithic/harmony/pulls/{number}/comments?per_page=100" --paginate \
  --jq '.[] | select(.user.login == "cursor[bot]") | {path: .path, line: .line, body: .body[:300]}'

# Greptile comments — PAGINATED
gh api "repos/zeblithic/harmony/issues/{number}/comments?per_page=100" --paginate \
  --jq '[.[] | select(.user.login == "greptile-apps[bot]")] | last | .body'
```

The human can override ("merge it anyway") — but flag the state clearly.

#### 4. Merge, switch to main, clean up

```bash
gh pr merge <number> --merge --delete-branch
git checkout main && git pull origin main
git branch -d <branch-name>
git log --oneline -3
```

#### 5. Report

```
Merged:  PR #<number> — <title>
Branch:  <branch-name> deleted (local + remote)
Main:    up to date
```

> Task complete. Use `/findtask` to survey what's next.

**STOP HERE. The human will invoke `/findtask` when ready for the next cycle.**
