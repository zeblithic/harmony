# Harmony File Manager Design

## Overview

A "Google Drive"-like file management tab in the harmony-client desktop app. Provides personal content lifecycle management — upload, organize, replicate, publish, and clean up content within the Harmony decentralized network.

**Key insight:** The file manager manages **private** content. Published content leaves your custody entirely — the network absorbs it, it costs you nothing, and you can't take it back. Jain (the content lifecycle engine) only governs private content.

## Architecture

**Approach:** Full-stack vertical integration into the existing harmony-client app. Third top-level mode (`'messages' | 'vines' | 'files'`) with its own layout, backed by a `FileManagerService` class (TypeScript). Mock data initially, progressive backend integration via Tauri commands into `harmony-jain`, `harmony-content`, and `harmony-roxy`.

**Tech stack:** Svelte 5 (runes), TypeScript, Tauri v2, vitest + @testing-library/svelte. No new dependencies — follows all existing conventions.

---

## Content Model

### Private vs. Published

| Property | Private Content | Published Content |
|----------|----------------|-------------------|
| **Quota cost** | Counts against your quota (each replica too) | Free — the network owns it |
| **Managed by** | You + Jain | The network (caches, libraries, repos) |
| **Deletable** | Yes (burn) | No — information is permanent |
| **Replication** | You pay for durability | Network handles distribution |
| **Jain involvement** | Full lifecycle (staleness, cleanup, repair) | None |

### Five Content Actions

| Action | Permanence | Privacy | Quota | Confirmation |
|--------|-----------|---------|-------|-------------|
| **Pin** | Stays forever | Private | Costs quota | None |
| **Archive** | Cold storage | Private | Costs quota | Single confirm |
| **Release** | Best-effort | Public, ephemeral | **Free** | Double confirm |
| **Publish** | Permanent | Public, durable | **Free** | Graduated (see Publishing) |
| **Burn** | Gone forever | Gone | **Free** | Graduated (see Cleanup) |

### Release vs. Publish

- **Publish (durable):** "I'm proud of this, I want it preserved." Libraries subscribe, replicate, and preserve it. Uses `ContentFlags { ephemeral: false }`.
- **Release (ephemeral):** "Someone might want this, but I won't be sad if it disappears." Available while caches are warm, may fade. Uses `ContentFlags { ephemeral: true }`. The digital equivalent of leaving a box on the curb with a "FREE" sign.

---

## Replication Model

### Replica Counting

`replica_count` = your local copy + shares (people with keys) + blind storage buddies.

| Tier | Total Copies | Others Needed | Use Case |
|------|-------------|---------------|----------|
| Expendable | 1 | 0 (just you) | Won't miss it |
| Light | 2 | 1 | Minimal durability |
| Default | 3 | 2 | Standard (N+2) |
| High | 5 | 4 | Important content |
| Ultra | 9 | 8 (3×3 distributed) | Critical, max spread |

Default tier is 3 — classic N+2. Lose one expected, lose one unexpected, still have a copy and immediately re-replicate.

If you share a file with 1 person and your tier is Default (3), the system only needs 1 storage buddy to reach 3 total.

### Two Peer Relationships

| Relationship | Icon | They Can Read It? | Managed Where |
|-------------|------|-------------------|---------------|
| **Share** | Unlocked lock | Yes (have decryption key) | Per-file in detail panel |
| **Storage buddy** | Sealed envelope | No (hold encrypted blobs) | Standing relationship in NavPanel |

**Shares** = "Here's a lockbox and here's your key." A Roxy licensing/access grant.

**Storage buddies** = "Hold this sealed package for me." Pure durability, no access. Pool of trusted-but-blind replication peers. Contacts from Messages mode are the candidate pool; you explicitly enable them for storage in the file manager.

**Reciprocity (future):** Storage buddy relationships are bilateral — you hold blobs for them, they hold for you. Both sides pay quota. Stubbed initially.

### Settings

- **Default replication tier:** Dropdown (default: Default/3). Applied to new uploads, overridable per-file and per-folder (bundles).
- **Quota limit:** Configurable number in GB (default: 10 GB). Private content only.

---

## Layout & Navigation

### Mode Integration

Add `'files'` to `AppMode`. NavPanel footer gets a third toggle. App.svelte conditionally renders the file manager layout.

### 3-Column Layout (reuses existing CSS grid)

```
[NavPanel 240px]     [FileBrowser flex-1]        [DetailPanel 320px]

- Folder tree        - Toolbar (view toggle,     - File metadata
- Quick filters        sort, search, upload)     - Staleness score
  (category,         - Breadcrumbs               - Replication status
   status,           - File list OR grid         - Sensitivity badge
   repl. tier)       - CleanupView (sub-view)    - Share list
- Storage buddies    - PublishedView (sub-view)   - Storage buddy list
  (collapsed)        - QuotaBar (bottom,          - Actions (publish,
                       private content only)        release, burn,
                                                    archive, pin, tier)
```

Same responsive collapse at 768px.

### File Browser

**Two view modes** (toolbar toggle):
- **List view:** Sortable columns — name, category icon, size, last accessed, staleness badge, replication tier, sensitivity icon.
- **Grid view:** Thumbnail cards with staleness badge overlay.

**Navigation:** Folder tree in NavPanel, breadcrumbs in FileBrowser, double-click folders, Backspace to go up.

**Two sections** (toggled via NavPanel or toolbar):
- **My Private Content** — the real file manager. Quota, replication, staleness, cleanup.
- **My Published Content** — read-only catalog. Browse/search your published works. No quota bar, no cleanup. Stats (popularity, library count) in future.

### Staleness Badges (Private Content Only)

| Score Range | Badge | Color | Meaning |
|-------------|-------|-------|---------|
| 0.0 – 0.3 | none | — | Fresh |
| 0.3 – 0.6 | dot | yellow | Getting stale |
| 0.6 – 0.85 | dot | orange | Archive candidate |
| 0.85 – 1.0 | dot | red | Burn candidate |

Pinned/licensed content never shows a badge.

### Quick Filters (NavPanel)

- By category: Music, Video, Text, Image, Software, Dataset
- By status: Stale, Pinned, Licensed, Under-replicated
- By replication tier: Expendable → Ultra

---

## Upload & Import

1. "Upload" button in toolbar (or drag-and-drop onto FileBrowser)
2. Tauri native file dialog — single or multi-select
3. Pre-upload panel:
   - File list with sizes
   - Sensitivity picker per file (default: `Private`)
   - Replication tier (inherits global default, overridable)
   - Total quota impact: "This will use 2.3 GB of your 10 GB quota"
4. Confirm → files chunked, CIDs generated, `ContentRecord` created with `origin: SelfCreated`
5. Progress bar near quota bar

**Export (stub):** "Export to device" action exists in UI and service interface. Shows "Export coming soon" toast. Future: reassemble Merkle DAG, write via Tauri file save dialog.

---

## Publishing Flow

**What publishing means:** Making content `Public`. Anyone in the world can access, copy, and redistribute it. Once published, you cannot unpublish — that's how information works. Published content costs zero quota.

### Confirmation Gates (sensitivity sets floor, user ratchets up in settings)

**Publish (durable):**

| Sensitivity | Minimum | Gates |
|------------|---------|-------|
| Private | Double | Gate 1 → Gate 2 |
| Intimate | Triple | Gate 1 → Gate 2 → Gate 3 |
| Confidential | Triple+OOB | Gate 1 → Gate 2 → Gate 3 → Gate 4 |

- **Gate 1:** "Publishing makes this content permanently public. Anyone in the world can access, copy, and redistribute it. This cannot be undone." → Cancel / Continue
- **Gate 2:** "You are about to publish [filename]. This is irreversible." → Cancel / "Yes, publish"
- **Gate 3:** Type-to-confirm — "Type `[filename]` to confirm publication"
- **Gate 4:** Out-of-band verification stub — 6-digit code, "Confirm on your other device." Escape hatch with warning. (Future: actual cross-device verification.)

**Release (ephemeral):** Always double-confirm (Gate 1 + Gate 2), with adjusted wording: "This will make your content publicly available. It may persist on the network or fade over time. You can't take it back, but nobody's obligated to keep it either."

### Visual State

- **Private:** Lock icon colored by sensitivity level
- **Published (durable):** Globe icon
- **Released (ephemeral):** Open-hand / wind icon (something that conveys "set free")

---

## Cleanup & Storage Management

### Cleanup View (Private Content Only)

Accessed via quota bar tap or "Cleanup" toolbar button. Sub-view within FileBrowser (not a separate window).

**Layout:**
- **Quota summary** — used/total, breakdown by category (bar chart)
- **Recommendations list** — Jain's `RecommendBurn` and `RecommendArchive` plus "consider publishing/releasing" suggestions, sorted by confidence

Each recommendation shows:
- File name, category icon, size
- Reason: Stale, DuplicateOfPublic, OverReplicated, Expired
- Space recoverable
- Staleness score bar
- Actions: Burn, Archive, Release, Publish, Pin (dismiss)

**Jain's "consider publishing" suggestion:** "This costs you 1.5 GB across 3 devices. Publish to preserve it forever, release to free quota with no preservation guarantee, or burn if it's truly disposable."

### Bulk Operations

- Select all burn/archive/release/publish candidates
- Single confirmation dialog for bulk action
- Live running total of space to be recovered

### Burn Confirmation

- Single confirm for `Public`-sensitivity content (wait — this is already public, so it wouldn't be here. All content in cleanup is private.)
- Double confirm for `Private`
- Sensitivity floor applies (same as publishing, since burning Confidential data is also consequential — it's gone forever)

---

## Service Architecture

### FileManagerService (TypeScript)

```
FileManagerService
├── getContents(folderId?, filters?) → ContentItem[]
├── getContentDetail(cid) → ContentDetail
├── getQuotaStatus() → QuotaStatus
├── getCleanupRecommendations() → CleanupRecommendation[]
├── getStorageBuddies() → StorageBuddy[]
├── getPublishedContent(filters?) → PublishedItem[]
│
├── upload(files, sensitivity, tier) → UploadResult
├── burn(cids) → void
├── archive(cids) → void
├── publish(cids) → void
├── release(cids) → void
├── pin(cid) / unpin(cid) → void
├── setReplicationTier(cids, tier) → void
├── shareTo(cid, peerAddress) → void
├── addStorageBuddy(peerAddress) → void
├── removeStorageBuddy(peerAddress) → void
├── exportToDevice(cids) → void  (stub)
│
└── settings
    ├── defaultReplicationTier: ReplicationTier
    ├── quotaBytes: number
    └── confirmationOverrides: Map<Sensitivity, ConfirmLevel>
```

### Data Flow

```
[Svelte Components]
       ↕ props + events
[FileManagerService]  ← mock data initially
       ↕ invoke()
[Tauri Commands]      ← stubs initially
       ↕ function calls
[harmony-jain]        ← staleness, cleanup recs, filtering
[harmony-content]     ← CIDs, chunking, Merkle DAG
[harmony-roxy]        ← licensing, publishing
```

### Key TypeScript Types

```typescript
type AppMode = 'messages' | 'vines' | 'files'
type ReplicationTier = 'expendable' | 'light' | 'default' | 'high' | 'ultra'
type ContentSensitivity = 'public' | 'private' | 'intimate' | 'confidential'
type FileViewMode = 'list' | 'grid'
type ContentSection = 'private' | 'published'
type PublishMode = 'durable' | 'ephemeral'
```

---

## Component Hierarchy

```
App.svelte (mode === 'files')
│
├── NavPanel.svelte (files mode content)
│   ├── FolderTree.svelte
│   ├── QuickFilters.svelte
│   └── StorageBuddySummary.svelte
│
├── FileBrowser.svelte
│   ├── BrowserToolbar.svelte
│   ├── Breadcrumbs.svelte
│   ├── FileList.svelte
│   │   └── FileRow.svelte
│   ├── FileGrid.svelte
│   │   └── FileCard.svelte
│   ├── CleanupView.svelte
│   │   ├── QuotaSummary.svelte
│   │   └── RecommendationCard.svelte
│   ├── PublishedView.svelte
│   └── QuotaBar.svelte
│
└── FileDetailPanel.svelte
    ├── FileMetadata.svelte
    ├── StalenessIndicator.svelte
    ├── ReplicationStatus.svelte
    ├── SensitivityBadge.svelte
    ├── ShareList.svelte
    ├── StorageBuddyList.svelte
    ├── PublishButton.svelte
    └── FileActions.svelte

Shared (reusable confirmation gates):
├── ConfirmDialog.svelte
├── DoubleConfirmDialog.svelte
├── TypeToConfirmDialog.svelte
└── OutOfBandVerify.svelte
```

~25 new components, each small and single-purpose. Each gets a vitest test file.

---

## Stubbed vs. Real (Initial Scope)

| Feature | Initial State | What's Real |
|---------|--------------|-------------|
| File browser (list/grid) | Mock data | Full UI, navigation, sorting, filtering |
| Staleness badges | Mock scores | Visual system, threshold mapping |
| Quota bar | Configurable constant | UI, percentage math, color gradient |
| Upload flow | Mock delay + add to list | Full UI, sensitivity picker, tier picker |
| Publish/Release flow | Mock state change | All confirmation gates, visual state changes |
| Burn/Archive | Mock removal from list | Confirmation dialogs, bulk select |
| Cleanup view | Mock recommendations | Full UI, recommendation cards, bulk actions |
| Replication status | Mock counts | Detail panel display, tier dropdown |
| Storage buddies | Mock peer list | Add/remove UI, buddy list |
| Sharing | Mock share list | Contact picker, share/unshare |
| Export | Toast "coming soon" | Button exists, wired in service interface |
| Published content view | Mock catalog | Browse/search published items |
| Tauri commands | Hardcoded returns | Command signatures, DTOs typed |
| Jain integration | None | Future: wire tick/reconcile through Tauri |
| Real chunking/CIDs | None | Future: wire harmony-content |
| Real network replication | None | Future: Zenoh transport |

**Principle:** Every user-facing interaction works end-to-end with mock data. Architecture is ready to swap in real backends layer by layer without touching the UI.
