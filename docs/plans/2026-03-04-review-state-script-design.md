# Design: `scripts/review-state.sh`

## Purpose

Deterministic shell script that gathers all review signals for a PR and derives
the review state machine state. Used by `/monitorreviews` and `/finishtask`.

## Input/Output

**Input:** `bash scripts/review-state.sh [PR_NUMBER]`
- Auto-detects PR from current branch if no number given.

**Output:** Structured text with labeled sections:

```
=== REVIEW STATE: PR #N ===
Title: ...
URL: ...

--- LATEST COMMIT ---
SHA: ...
Date: ...

--- TRIGGERS ---
Latest "bugbot run": <timestamp> (eyes: N, thumbsup: N)
Latest "@greptile":  <timestamp> (eyes: N, thumbsup: N)

--- BUGBOT (cursor[bot]) ---
Status: <pending|running|complete|stale>
Latest review: <timestamp>
Reviews after latest commit: N
Inline comments after latest commit: N
Issues found: N

Finding 1:
  File: ...
  Line: ...
  Severity: ...
  Body: <full body>

--- GREPTILE (greptile-apps[bot]) ---
Status: <pending|running|complete|stale>
Latest comment: <timestamp>
Comments after latest commit: N
Issues found: N

Finding 1:
  Body: <full body>

--- STATE ---
State: <REVIEWS_PENDING|BUGBOT_STUCK|PARTIAL_REVIEWS|REVIEWS_COMPLETE_WITH_FEEDBACK|REVIEWS_COMPLETE_ALL_CLEAR>
Bugbot: <status> (N issues)
Greptile: <status> (N issues)
Action: <recommended next step>
```

## State Derivation (priority order)

1. No triggers → error, not in review cycle
2. Latest commit newer than latest trigger → STALE
3. Triggers exist, no bot response after trigger:
   - <3 min or "eyes" reaction → REVIEWS_PENDING
   - Bugbot trigger >3 min, no eyes, no response → BUGBOT_STUCK
4. Only one bot responded after latest trigger → PARTIAL_REVIEWS
5. Both responded:
   - Issues > 0 → REVIEWS_COMPLETE_WITH_FEEDBACK
   - No issues → REVIEWS_COMPLETE_ALL_CLEAR

## Finding Extraction

- **Bugbot:** `pulls/{N}/comments` API (inline review comments by `cursor[bot]`).
  Only comments newer than latest trigger. Parse severity from markdown body.
- **Greptile:** `issues/{N}/comments` API (PR-level comments by `greptile-apps[bot]`).
  Only comments newer than latest trigger. Count findings in `<details>` blocks.
- All API calls use `--paginate`.

## Dependencies

- `gh` CLI (authenticated)
- `jq`
- Standard POSIX shell utilities (`date`, `awk`, `grep`)
