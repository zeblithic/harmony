# Beads + Dolt: Issue Tracking for Harmony

Comprehensive guide to how issue tracking is set up and used across all
Zeblithic Harmony repositories.

---

## CRITICAL: The Git-Tracking Trap

> **If `.beads/` files are tracked by git, beads WILL break on every branch
> switch, stash, merge, or checkout.** This is the single most common cause
> of beads failures in AI-assisted workflows.

### What happens

1. `bd init` creates `.beads/metadata.json` with `dolt_mode: "embedded"`
2. If `.beads/` files get committed to git (e.g., `bd init` without `--stealth`,
   or an AI agent runs `git add -A`), git now tracks `metadata.json`
3. You fix `metadata.json` locally to `"embedded"` — it works
4. Next `git checkout`, `git stash pop`, `git merge`, or branch switch **silently
   reverts `metadata.json`** to whatever version git has committed (often `"server"`)
5. Every `bd` command now fails with "database not found on Dolt server"
6. You fix it again. Git reverts it again. Repeat for months.

### Why `.git/info/exclude` is NOT sufficient

`bd init --stealth` adds `.beads/` to `.git/info/exclude`. This is a **local-only
gitignore** that:
- Prevents new untracked `.beads/` files from showing in `git status`
- Does **NOT** untrack files already committed to git
- Does **NOT** propagate to other clones or machines

If `.beads/` files were ever committed (even once), `.git/info/exclude` does nothing
to protect them from git operations.

### The correct setup (mandatory for all repos)

**`.beads/` must be in the committed `.gitignore` AND untracked from the git index:**

```bash
# 1. Add to .gitignore (committed, works for all clones)
echo ".beads/" >> .gitignore

# 2. Remove from git index (keeps local files, stops git from tracking them)
git rm -r --cached .beads/

# 3. Commit
git add .gitignore
git commit -m "chore: untrack .beads/ directory from git"
git push
```

After this, git will never touch `.beads/` again — no matter what branch
operations happen. The `.beads/` directory stays on disk for `bd` to use,
but git doesn't know it exists.

**Verify with:** `git ls-files --cached .beads/` — must return nothing.

### After untracking: fix Dolt remote sync

Removing `.beads/` from git moves the git remote's HEAD ahead of what Dolt's
internal `refs/dolt/data` ref tracks. This causes `bd dolt push` to fail with
"non-fast-forward". Fix with a one-time force push:

```bash
bd dolt push --force
```

### Auto-sync configuration (required)

After fixing the git-tracking issue, ensure auto-sync is enabled so changes
push automatically instead of silently accumulating locally:

```bash
bd config set dolt.auto-commit on      # Commit after every bd write operation
bd config set dolt.auto-push true      # Push to remote after commits
```

Verify with:
```bash
bd config get dolt.auto-commit    # Should show: on
bd config get dolt.auto-push      # Should show: true
```

Without `dolt.auto-push`, changes only sync when you manually run `bd dolt push`.
With it enabled, every `bd create`, `bd update`, `bd close`, etc. automatically
commits and pushes (throttled to every 5 minutes to avoid excessive network calls).

---

## Scope

This applies to **every** `zeblithic/harmony-*` repo:

| Repo | Status | Dolt DB name |
|---|---|---|
| `harmony` | Active | `harmony` |
| `harmony-client` | Active | `harmony_client` |
| `harmony-glitch` | Active | `glitch` |
| `harmony-os` | Active | `harmony_os` |
| `harmony-openwrt` | Active | `harmony_openwrt` |
| `harmony-browser` | Not yet initialized | — |
| `harmony-stq8` | Not yet initialized | — |

The Dolt database name is derived from the repo directory name at init time.
Hyphens become underscores (MySQL identifier rules). The exception is
`harmony-glitch` whose prefix was set to just `glitch`.

---

## How It Works

### The Stack

- **Beads** (`bd` CLI, v0.62.0+) — issue tracking that lives in-repo
- **Dolt** (v1.84.0+) — version-controlled SQL database (git for data)
- **Embedded mode** — Dolt runs in-process inside `bd`, no external server

### What "Embedded Mode" Means

Beads can run Dolt in two modes:

| | Embedded Mode (what we use) | Server Mode |
|---|---|---|
| Dolt process | Runs inside `bd` (in-process) | Separate `dolt sql-server` on a TCP port |
| Data location | `.beads/embeddeddolt/<db>/` | `.beads/dolt/<db>/` |
| Concurrency | Single-process (file lock) | Multi-writer via TCP |
| `metadata.json` | `"dolt_mode": "embedded"` | `"dolt_mode": "server"` or absent |
| Config needed | None — zero config | Host, port, sometimes password |
| `bd dolt status` | Not supported (no server to check) | Shows server PID and port |
| `bd dolt show` | Not supported | Shows connection details |

**We use embedded mode exclusively.** If a machine has `dolt_mode: "server"` or
is missing the `embeddeddolt/` directory, it's in the wrong mode and needs to be
re-initialized (see [Fixing Wrong Mode](#fixing-wrong-mode-server-instead-of-embedded)).

### Where Data Lives

```
<repo>/
└── .beads/
    ├── config.yaml                  # Beads settings
    ├── metadata.json                # Backend config (MUST say dolt_mode: embedded)
    ├── .gitignore                   # Keeps DB files out of git commits
    ├── README.md                    # Auto-generated quickstart
    ├── push-state.json              # Tracks last push (timestamp + commit hash)
    ├── interactions.jsonl           # Audit log (local, not synced)
    ├── last-touched                 # Timestamp of last bd operation
    ├── hooks/                       # Git hooks (unused — we set no-git-ops: true)
    ├── dolt/                        # EMPTY — this is the server-mode data dir
    │                                #   If you see data here, you're in server mode
    └── embeddeddolt/                # THIS is where our data actually lives
        ├── .lock                    # Process-exclusive file lock
        └── <db_name>/              # e.g., "harmony", "harmony_client", "glitch"
            └── .dolt/
                ├── repo_state.json  # Dolt remotes, branch refs
                ├── config.json      # Dolt config (usually empty {})
                ├── noms/            # Content-addressed chunk store
                │   ├── manifest     # Root manifest
                │   ├── journal.idx  # Write-ahead journal
                │   └── LOCK         # Dolt's internal write lock
                └── git-remote-cache/  # Cached git remote data
```

### What Gets Synced vs What's Local-Only

| Synced via Dolt remote | Local-only (per-machine) |
|---|---|
| Issue data (titles, descriptions, status, deps) | `push-state.json` |
| Dolt commit history | `interactions.jsonl` |
| Schema migrations | `last-touched` |
| | `embeddeddolt/.lock` |
| | `config.yaml` (rebuilt at init) |
| | `metadata.json` (rebuilt at init) |

`.beads/` is listed in the repo's committed `.gitignore` to prevent any of
this from entering git. Beads data syncs through Dolt remotes only — never
through git commits/PRs. See [The Git-Tracking Trap](#critical-the-git-tracking-trap)
for why this must be in `.gitignore` and not just `.git/info/exclude`.

### How Dolt Remotes Work

Each repo's Dolt remote points to the **same GitHub repo** as the source code,
using a `git+ssh://` URL:

```
origin    git+ssh://git@github.com/zeblithic/<repo>.git
```

Dolt stores its data in a separate git ref namespace (`refs/dolt/...`) that
doesn't interfere with source code branches. When you `bd dolt push`, it pushes
Dolt refs — not your code.

---

## Setting Up a New Machine

### Prerequisites

1. **bd** (beads CLI) — build from source:
   ```bash
   cd ~/work/steveyegge/beads
   go build -o ~/.local/bin/bd ./cmd/bd
   ```
   Or install from release. Verify: `bd version` should show `>= 0.62.0`.

2. **dolt** — install the Dolt binary:
   ```bash
   sudo bash -c 'curl -L https://github.com/dolthub/dolt/releases/latest/download/install.sh | bash'
   ```
   Or via Homebrew: `brew install dolt`. Verify: `dolt version` should show `>= 1.84.0`.

3. **SSH key** registered with GitHub (Dolt remotes use `git+ssh://`).

4. **Git identity** configured (`git config user.name` / `git config user.email`).
   Beads uses this for the Dolt commit author.

### Step-by-Step: Initialize Each Repo

Do this for every repo in the workspace. Here's `harmony` as the example;
repeat for each repo.

#### 1. Initialize beads in stealth/embedded mode

```bash
cd ~/work/zeblithic/harmony
bd init --stealth
```

**What this does:**
- Creates `.beads/` directory
- Sets `dolt_mode: "embedded"` in `metadata.json`
- Sets `no-git-ops: true` in `config.yaml` (stealth = no git hooks, no auto-commits)
- Adds `.beads/` to `.git/info/exclude` (hides from collaborators)
- Creates the embedded Dolt database at `.beads/embeddeddolt/harmony/`
- Runs schema migrations and makes an initial Dolt commit
- Auto-detects git origin and adds it as a Dolt remote

#### 2. Verify the remote was auto-configured

```bash
bd dolt remote list
```

Expected output:
```
origin    git+ssh://git@github.com/zeblithic/harmony.git
```

If it's missing, add it manually:
```bash
bd dolt remote add origin git+ssh://git@github.com/zeblithic/harmony.git
```

#### 3. Pull existing data from the remote

```bash
cd .beads/embeddeddolt/harmony
dolt pull origin main
```

> **Why `dolt pull` directly instead of `bd dolt pull`?**
>
> There's a known bug in bd v0.62.0: `bd dolt pull` fails in embedded mode with
> "did not specify a branch" because the `branches` map in `repo_state.json` is
> empty (no tracking relationship set). The `dolt` CLI accepts `origin main` as
> explicit arguments and works fine.

If this is the very first machine and no data has been pushed yet, this will
fail with "branch not found on remote" — that's fine, skip it and push first
instead.

#### 4. Verify

```bash
bd list    # Should show issues
```

#### 5. Repeat for each repo

Use the correct database name for each:

```bash
# harmony-client
cd ~/work/zeblithic/harmony-client
bd init --stealth
cd .beads/embeddeddolt/harmony_client && dolt pull origin main

# harmony-glitch
cd ~/work/zeblithic/harmony-glitch
bd init --stealth
cd .beads/embeddeddolt/glitch && dolt pull origin main

# harmony-os
cd ~/work/zeblithic/harmony-os
bd init --stealth
cd .beads/embeddeddolt/harmony_os && dolt pull origin main

# harmony-openwrt
cd ~/work/zeblithic/harmony-openwrt
bd init --stealth
cd .beads/embeddeddolt/harmony_openwrt && dolt pull origin main
```

### Verify correct metadata.json

After init, each repo's `.beads/metadata.json` should look like:

```json
{
  "database": "dolt",
  "backend": "dolt",
  "dolt_mode": "embedded",
  "dolt_database": "<db_name>",
  "project_id": "<some-uuid>"
}
```

The critical field is `"dolt_mode": "embedded"`. If it says `"server"` or is
missing, you're in the wrong mode.

### Verify correct config.yaml

Each repo's `.beads/config.yaml` should contain:

```yaml
no-git-ops: true
```

It may also have `backup: enabled: false`. The key thing is `no-git-ops: true`
(stealth mode).

---

## Day-to-Day Usage

### Working with issues

```bash
bd list                                  # List all issues
bd ready                                 # Show unblocked, ready-to-work issues
bd show <id>                             # View full issue details
bd create "Title" -p 2 --json            # Create issue (priority 2)
bd create "Title" --deps discovered-from:<parent-id> --json
bd update <id> --claim                   # Claim work (sets assignee + in_progress)
bd close <id> --reason "Done"            # Complete work
bd search "keyword"                      # Search issues
```

### Pushing changes to remote

After creating/updating/closing issues, push to sync with other machines:

```bash
bd dolt commit     # Commit pending changes to Dolt history
bd dolt push       # Push Dolt commits to GitHub remote
```

`bd dolt push` handles both the commit and the push in practice — if there are
uncommitted changes, it commits them first. But explicitly committing is clearer.

### Pulling changes from remote

```bash
# Navigate to the embedded Dolt database directory
cd <repo>/.beads/embeddeddolt/<db_name>

# Pull latest from remote
dolt pull origin main
```

Or fetch + inspect first:
```bash
cd <repo>/.beads/embeddeddolt/<db_name>
dolt fetch origin
dolt log --oneline remotes/origin/main | head -5   # See what's new
dolt merge remotes/origin/main                      # Merge it in
```

### Checking sync status

```bash
cd <repo>/.beads/embeddeddolt/<db_name>
dolt fetch origin
dolt branch -av
```

This shows both `main` (local HEAD) and `remotes/origin/main` (last fetched
remote). If the commit hashes match, you're in sync. If they differ, you need
to pull or push.

```bash
# Compare logs side by side
dolt log --oneline main | head -5
dolt log --oneline remotes/origin/main | head -5
```

---

## Multi-Machine Sync

### Normal workflow (no conflicts)

```
Machine A:  bd dolt commit && bd dolt push
Machine B:  cd .beads/embeddeddolt/<db> && dolt pull origin main
```

Both machines now share the same Dolt history. Future pushes from either
machine will fast-forward cleanly.

### When push is rejected (non-fast-forward)

This means the other machine pushed since your last pull. Resolve like git:

```bash
cd .beads/embeddeddolt/<db_name>

# 1. Fetch the remote state
dolt fetch origin

# 2. Merge remote into local
dolt merge remotes/origin/main

# 3. If conflicts exist, resolve them
dolt conflicts cat <table>         # See conflicting rows
dolt conflicts resolve --theirs    # Accept remote version
# or: dolt conflicts resolve --ours  # Keep local version

# 4. Push the merged result
dolt push origin main
```

### When histories have no common ancestor

This happens when one machine re-initializes (`bd init`) and force pushes,
creating an entirely new Dolt history that shares no commits with the other
machine's local data.

**Symptoms:**
- `dolt merge remotes/origin/main` fails with "no common ancestor"
- `dolt log` shows different root commits locally vs remote

**To fix** (from the machine with the richer/correct data):

```bash
cd .beads/embeddeddolt/<db_name>
dolt push --force origin main
```

Then on the **other machine**, discard its local history and adopt the remote:

```bash
cd .beads/embeddeddolt/<db_name>
dolt fetch origin
dolt reset --hard remotes/origin/main
```

This is destructive to the other machine's local Dolt history — make sure the
force-pushing machine has the data you want to keep.

---

## Fixing Wrong Mode (Server Instead of Embedded)

If a machine was initialized with `bd init` (without `--stealth`) or with
`bd init --server`, it may be in server mode. Signs:

- `.beads/metadata.json` says `"dolt_mode": "server"` or has no `dolt_mode` field
- `.beads/dolt/<db>/` has data but `.beads/embeddeddolt/` is missing or empty
- `bd dolt status` works (shows a running server) instead of erroring
- `bd dolt show` works (shows connection config) instead of erroring

### How to fix: re-initialize in embedded mode

**Option A: Re-init from scratch and pull from remote** (cleanest)

```bash
cd ~/work/zeblithic/<repo>

# 1. Back up any local issues not yet pushed
bd export -o /tmp/<repo>-issues-backup.jsonl

# 2. Remove the old beads directory
rm -rf .beads

# 3. Re-initialize in embedded/stealth mode
bd init --stealth

# 4. Verify metadata
cat .beads/metadata.json   # Must show "dolt_mode": "embedded"

# 5. Pull data from remote
cd .beads/embeddeddolt/<db_name>
dolt pull origin main

# 6. Verify
cd ~/work/zeblithic/<repo>
bd list
```

**Option B: Import from JSONL backup** (if remote has no data or wrong data)

```bash
cd ~/work/zeblithic/<repo>

# 1. Export current issues (while still in server mode)
bd export -o /tmp/<repo>-issues.jsonl

# 2. Kill any running dolt server
bd dolt stop 2>/dev/null
bd dolt killall 2>/dev/null

# 3. Remove and re-initialize
rm -rf .beads
bd init --stealth

# 4. Import the backup
bd import /tmp/<repo>-issues.jsonl

# 5. Commit and push to establish remote history
bd dolt commit
bd dolt push
```

### Verifying you're in embedded mode

Run these checks:

```bash
# 1. metadata.json must say embedded
cat .beads/metadata.json | grep dolt_mode
# Expected: "dolt_mode": "embedded"

# 2. embeddeddolt directory must exist with data
ls .beads/embeddeddolt/
# Expected: <db_name>/ directory

# 3. These commands should ERROR (they only work in server mode)
bd dolt status    # Expected: "not supported in embedded mode"
bd dolt show      # Expected: "not supported in embedded mode"

# 4. These should work
bd list           # Shows issues
bd dolt remote list   # Shows origin remote
```

---

## Reference: Exact Config Files

### metadata.json (per-repo)

```json
{
  "database": "dolt",
  "backend": "dolt",
  "dolt_mode": "embedded",
  "dolt_database": "harmony",
  "project_id": "67558276-43d5-49c0-bfb8-a0585b63e2ef"
}
```

- `database` and `backend`: Always `"dolt"` (legacy compat)
- `dolt_mode`: **Must be `"embedded"`**
- `dolt_database`: The SQL database name (used as directory name under `embeddeddolt/`)
- `project_id`: UUID for cross-project isolation — auto-generated, unique per init

### config.yaml (per-repo)

```yaml
backup:
    enabled: false
no-git-ops: true
dolt:
    auto-commit: "on"
    auto-push: true
```

- `no-git-ops: true`: Stealth mode — bd never runs git commands, installs hooks, or auto-commits
- `backup.enabled: false`: No auto-backup exports
- `dolt.auto-commit: "on"`: Commit to Dolt history after every `bd` write operation
- `dolt.auto-push: true`: Push to remote after commits (throttled to every 5 minutes)

### repo_state.json (inside embeddeddolt/<db>/.dolt/)

```json
{
  "head": "refs/heads/main",
  "remotes": {
    "origin": {
      "name": "origin",
      "url": "git+ssh://git@github.com/zeblithic/harmony.git",
      "fetch_specs": ["refs/heads/*:refs/remotes/origin/*"],
      "params": {}
    }
  },
  "backups": {},
  "branches": {}
}
```

- `head`: Current branch (always `main`)
- `remotes.origin.url`: The git+ssh URL for Dolt push/pull
- `branches`: Empty map — this is the known bug that causes `bd dolt pull` to
  fail with "did not specify a branch"

---

## Reference: Command Cheat Sheet

### Issue management

| Command | What it does |
|---|---|
| `bd list` | List all issues |
| `bd ready` | List unblocked issues ready for work |
| `bd show <id>` | View issue details |
| `bd create "Title" -p <0-4> --json` | Create issue |
| `bd update <id> --claim` | Claim issue (assign + set in_progress) |
| `bd close <id>` | Mark done |
| `bd search "keyword"` | Full-text search |
| `bd export -o file.jsonl` | Export all issues to JSONL |
| `bd import file.jsonl` | Import issues from JSONL |

### Dolt / sync

| Command | What it does |
|---|---|
| `bd dolt commit` | Commit pending changes to Dolt history |
| `bd dolt push` | Push to remote (commits first if needed) |
| `bd dolt remote list` | Show configured remotes |
| `bd dolt remote add <name> <url>` | Add a remote |
| `bd dolt remote remove <name>` | Remove a remote |
| `bd dolt killall` | Kill orphaned dolt server processes |

### Direct dolt commands (in embeddeddolt/<db>/ dir)

| Command | What it does |
|---|---|
| `dolt fetch origin` | Fetch remote refs without merging |
| `dolt pull origin main` | Fetch + merge from remote |
| `dolt push origin main` | Push local to remote |
| `dolt push --force origin main` | Force push (overwrites remote) |
| `dolt branch -av` | Show all branches + remote tracking |
| `dolt log --oneline` | Compact commit history |
| `dolt status` | Show working set changes |
| `dolt merge remotes/origin/main` | Merge remote into local |
| `dolt reset --hard remotes/origin/main` | Discard local, adopt remote |
| `dolt conflicts cat <table>` | Show conflicting rows |
| `dolt conflicts resolve --theirs` | Resolve conflicts using remote |

---

## PR Process and Code Review

All functional code changes across **every** Zeblithic Harmony repo go through
a pull request process with automated review:

### Required reviewers

- **Greptile** — AI code review that understands the full codebase context.
  Checks for architectural consistency, potential bugs, and adherence to
  project patterns.
- **Bugbot** — Automated bug detection. Scans for common issues, security
  vulnerabilities, and regressions.

Both must pass before a PR can be merged.

### Workflow

1. Create a feature branch from `main`
2. Make your changes, commit, push the branch
3. Open a PR against `main`
4. Greptile and Bugbot automatically review
5. Address any findings
6. Merge once both reviewers are satisfied

### What goes through PRs vs what doesn't

| Through PRs (git) | Through Dolt remotes (not PRs) |
|---|---|
| Source code changes | Issue data |
| Config file changes | Dolt commit history |
| Documentation | Push/pull state |
| CI/CD changes | |

Beads data never appears in PRs. The `.beads/.gitignore` and
`.git/info/exclude` (stealth mode) ensure this.

---

## Appendix: How We Got Here (Historical Context)

Issues were originally created on the MacBook Pro and exported to JSONL files.
On 2026-03-29, the Vultr dev server was set up with embedded Dolt:

1. `bd init --stealth` on each repo
2. `bd import <repo>-issues.jsonl` to load the exported issues
3. `bd dolt remote add origin git+ssh://...` to configure remotes
4. `bd dolt push` to push to GitHub

The MacBook later re-initialized some repos and force-pushed bare `Initialize
data repository` commits, clobbering the server's richer history for
`harmony-client`, `harmony-glitch`, and `harmony-os`. These were reconciled by
force-pushing from the Vultr server (which had the full data) back to the
remote.

**To prevent this in the future:** always pull before re-initializing. If you
must re-init, export first (`bd export -o backup.jsonl`) so you don't lose data.

### The Git-Tracking Bug (2026-03-30)

For months, beads intermittently broke across sessions. The root cause was
finally identified: `bd init` had committed `.beads/metadata.json` to git with
`dolt_mode: "server"`. The file was later changed locally to `"embedded"`, but
because git tracked it, every branch switch or stash reverted it to `"server"`.

The fix applied across all 5 zeblithic repos:
1. `git rm -r --cached .beads/` — untrack from git index
2. Added `.beads/` to committed `.gitignore` (not just `.git/info/exclude`)
3. Rewrote `metadata.json` to `dolt_mode: "embedded"`
4. `bd dolt push --force` — reset remote `refs/dolt/data` ref
5. `bd config set dolt.auto-push true` — enable automatic sync

This is now the documented canonical setup. Any new repo or machine setup
must follow the steps in [The Git-Tracking Trap](#critical-the-git-tracking-trap).
