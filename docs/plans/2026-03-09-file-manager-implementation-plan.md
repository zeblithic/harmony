# File Manager Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a Google Drive-like file management tab in harmony-client with upload, replication, graduated publishing, Jain-powered cleanup, and quota management — all backed by mock data, architecture-ready for progressive backend integration.

**Architecture:** Third top-level mode (`'files'`) in the existing Svelte 5 app. New `FileManagerService` class (same pattern as `TrustService`). ~25 new components in 3-column layout reusing the existing CSS grid. Private content is managed by Jain; published content is a read-only catalog. Leverages harmony-roxy types (`ContentCategory`, `LicenseManifest`, `UsageRights`) for the publish/release pipeline.

**Tech Stack:** Svelte 5 (runes: `$state`, `$derived`, `$props`, `$effect`), TypeScript, vitest + @testing-library/svelte, CSS custom properties. No new dependencies.

**Design doc:** `docs/plans/2026-03-09-file-manager-design.md`

**Conventions reference:**
- Services: class-based, instantiated at App level, passed as props (see `src/lib/trust-service.ts`)
- Components: `$props()` destructuring, ARIA attributes on all interactive elements
- Tests: `render()` + `screen.getByRole()` pattern (see `__tests__/AriaAnnouncer.test.ts`)
- State: Svelte 5 runes only, no external store library
- CSS: use `:root` variables from `src/app.css`
- Events: `onclick={handler}` (NOT `on:click`)

---

### Task 1: TypeScript Types

**Files:**
- Modify: `src/lib/types.ts`
- Test: `src/lib/types.test.ts`

**Context:** All new types go in the existing `types.ts` alongside `AppMode`, `Message`, `VineVideo`, etc. The existing `AppMode` is `'messages' | 'vines'` on line 117. We add `'files'` and all file-manager types.

**Step 1: Write the type tests**

Create `src/lib/types.test.ts`:

```typescript
import { describe, it, expect } from 'vitest';
import type {
  AppMode,
  ReplicationTier,
  ContentSensitivity,
  FileViewMode,
  ContentSection,
  PublishMode,
  ContentItem,
  ContentDetail,
  QuotaStatus,
  CleanupRecommendation,
  CleanupReason,
  StorageBuddy,
  PublishedItem,
  UploadCandidate,
  FileManagerSettings,
} from './types';

describe('File manager types', () => {
  it('AppMode includes files', () => {
    const mode: AppMode = 'files';
    expect(mode).toBe('files');
  });

  it('ReplicationTier has all five levels', () => {
    const tiers: ReplicationTier[] = ['expendable', 'light', 'default', 'high', 'ultra'];
    expect(tiers).toHaveLength(5);
  });

  it('ContentItem has required fields', () => {
    const item: ContentItem = {
      cid: 'abc123',
      name: 'photo.jpg',
      category: 'image',
      sensitivity: 'private',
      sizeBytes: 1024,
      storedAt: 1000,
      lastAccessed: 2000,
      accessCount: 5,
      stalenessScore: 0.3,
      replicationTier: 'default',
      replicaCount: 3,
      pinned: false,
      licensed: false,
      parentCid: null,
      isFolder: false,
    };
    expect(item.cid).toBe('abc123');
    expect(item.isFolder).toBe(false);
  });

  it('ContentDetail extends ContentItem with peer info', () => {
    const detail: ContentDetail = {
      cid: 'abc123',
      name: 'photo.jpg',
      category: 'image',
      sensitivity: 'private',
      sizeBytes: 1024,
      storedAt: 1000,
      lastAccessed: 2000,
      accessCount: 5,
      stalenessScore: 0.3,
      replicationTier: 'default',
      replicaCount: 3,
      pinned: false,
      licensed: false,
      parentCid: null,
      isFolder: false,
      sharedWith: [{ address: 'peer1', displayName: 'Alice' }],
      storageBuddies: [{ address: 'peer2', displayName: 'Bob' }],
      origin: 'self-created',
    };
    expect(detail.sharedWith).toHaveLength(1);
    expect(detail.origin).toBe('self-created');
  });

  it('QuotaStatus has usage fields', () => {
    const quota: QuotaStatus = {
      usedBytes: 5_000_000_000,
      totalBytes: 10_000_000_000,
      byCategory: { image: 2_000_000_000, video: 3_000_000_000 },
    };
    expect(quota.usedBytes).toBeLessThan(quota.totalBytes);
  });

  it('CleanupRecommendation has action-relevant fields', () => {
    const rec: CleanupRecommendation = {
      cid: 'abc123',
      name: 'old-doc.txt',
      category: 'text',
      sizeBytes: 50_000,
      reason: 'stale',
      stalenessScore: 0.87,
      spaceRecoverable: 150_000,
      confidence: 0.87,
    };
    expect(rec.reason).toBe('stale');
    expect(rec.confidence).toBeGreaterThan(0.8);
  });

  it('StorageBuddy tracks storage used', () => {
    const buddy: StorageBuddy = {
      address: 'peer3',
      displayName: 'Charlie',
      storageUsedBytes: 500_000_000,
      online: true,
    };
    expect(buddy.online).toBe(true);
  });

  it('PublishedItem includes publish mode', () => {
    const item: PublishedItem = {
      cid: 'pub1',
      name: 'my-song.mp3',
      category: 'music',
      sizeBytes: 8_000_000,
      publishedAt: 1000,
      publishMode: 'durable',
    };
    expect(item.publishMode).toBe('durable');
  });
});
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/zeblith/work/zeblithic/harmony-client && npx vitest run src/lib/types.test.ts`
Expected: FAIL — types don't exist yet

**Step 3: Add the types to `types.ts`**

Modify `src/lib/types.ts`. Change line 117 from:
```typescript
export type AppMode = 'messages' | 'vines';
```
to:
```typescript
export type AppMode = 'messages' | 'vines' | 'files';
```

Then append the following types at the end of the file:

```typescript
// ── File Manager Types ──────────────────────────────────────────────

export type ReplicationTier = 'expendable' | 'light' | 'default' | 'high' | 'ultra';
export type ContentSensitivity = 'public' | 'private' | 'intimate' | 'confidential';
export type FileViewMode = 'list' | 'grid';
export type ContentSection = 'private' | 'published';
export type PublishMode = 'durable' | 'ephemeral';
export type ContentOrigin = 'self-created' | 'peer-replicated' | 'downloaded' | 'cached-in-transit';
export type CleanupReason = 'stale' | 'duplicate-of-public' | 'over-replicated' | 'expired';

/** Mirrors harmony-roxy ContentCategory. */
export type ContentCategory = 'music' | 'video' | 'text' | 'image' | 'software' | 'dataset' | 'bundle';

/** Mirrors harmony-roxy UsageRights bitflags as a simpler TS set. */
export type UsageRight = 'stream' | 'download' | 'remix' | 'reshare';

export interface PeerRef {
  address: string;
  displayName: string;
}

export interface ContentItem {
  cid: string;
  name: string;
  category: ContentCategory;
  sensitivity: ContentSensitivity;
  sizeBytes: number;
  storedAt: number;
  lastAccessed: number;
  accessCount: number;
  stalenessScore: number;
  replicationTier: ReplicationTier;
  replicaCount: number;
  pinned: boolean;
  licensed: boolean;
  parentCid: string | null;
  isFolder: boolean;
}

export interface ContentDetail extends ContentItem {
  sharedWith: PeerRef[];
  storageBuddies: PeerRef[];
  origin: ContentOrigin;
}

export interface QuotaStatus {
  usedBytes: number;
  totalBytes: number;
  byCategory: Partial<Record<ContentCategory, number>>;
}

export interface CleanupRecommendation {
  cid: string;
  name: string;
  category: ContentCategory;
  sizeBytes: number;
  reason: CleanupReason;
  stalenessScore: number;
  spaceRecoverable: number;
  confidence: number;
}

export interface StorageBuddy {
  address: string;
  displayName: string;
  storageUsedBytes: number;
  online: boolean;
}

export interface PublishedItem {
  cid: string;
  name: string;
  category: ContentCategory;
  sizeBytes: number;
  publishedAt: number;
  publishMode: PublishMode;
}

export interface UploadCandidate {
  file: File;
  sensitivity: ContentSensitivity;
  replicationTier: ReplicationTier;
}

export interface FileManagerSettings {
  defaultReplicationTier: ReplicationTier;
  quotaBytes: number;
  defaultViewMode: FileViewMode;
  confirmationOverrides: Partial<Record<ContentSensitivity, number>>;
}
```

**Step 4: Run tests to verify they pass**

Run: `cd /Users/zeblith/work/zeblithic/harmony-client && npx vitest run src/lib/types.test.ts`
Expected: PASS — all 8 tests green

**Step 5: Run existing tests to verify no regressions**

Run: `cd /Users/zeblith/work/zeblithic/harmony-client && npx vitest run`
Expected: All existing tests pass. The `AppMode` change may cause TypeScript errors in components that exhaustively match modes — fix in Task 2.

**Step 6: Commit**

```bash
cd /Users/zeblith/work/zeblithic/harmony-client
git add src/lib/types.ts src/lib/types.test.ts
git commit -m "feat(files): add file manager TypeScript types

Add ReplicationTier, ContentSensitivity, ContentItem, ContentDetail,
QuotaStatus, CleanupRecommendation, StorageBuddy, PublishedItem, and
supporting types. Extend AppMode with 'files'. Types mirror harmony-roxy
ContentCategory and UsageRights for future Tauri bridge integration."
```

---

### Task 2: FileManagerService & Mock Data

**Files:**
- Create: `src/lib/file-manager-service.ts`
- Create: `src/lib/mock-file-data.ts`
- Test: `src/lib/file-manager-service.test.ts`

**Context:** Services are class-based (see `trust-service.ts`). Instantiated at App level, passed as props. This service wraps all file manager operations and returns mock data. The mock data file follows the pattern of `mock-data.ts` — exported arrays/objects.

**Step 1: Create mock data**

Create `src/lib/mock-file-data.ts`:

```typescript
import type {
  ContentItem,
  ContentDetail,
  CleanupRecommendation,
  StorageBuddy,
  PublishedItem,
  QuotaStatus,
  PeerRef,
} from './types';

const now = Date.now();
const DAY = 86_400_000;

export const mockPeers: PeerRef[] = [
  { address: 'a1b2c3d4', displayName: 'Alice' },
  { address: 'e5f6a7b8', displayName: 'Bob' },
  { address: 'c9d0e1f2', displayName: 'Charlie' },
  { address: 'd3e4f5a6', displayName: 'Diana' },
];

export const mockStorageBuddies: StorageBuddy[] = [
  { address: 'e5f6a7b8', displayName: 'Bob', storageUsedBytes: 500_000_000, online: true },
  { address: 'c9d0e1f2', displayName: 'Charlie', storageUsedBytes: 200_000_000, online: false },
];

export const mockPrivateContent: ContentItem[] = [
  {
    cid: 'cid-photos-2026',
    name: '2026 Photos',
    category: 'bundle',
    sensitivity: 'private',
    sizeBytes: 2_500_000_000,
    storedAt: now - 90 * DAY,
    lastAccessed: now - 30 * DAY,
    accessCount: 12,
    stalenessScore: 0.35,
    replicationTier: 'default',
    replicaCount: 3,
    pinned: false,
    licensed: false,
    parentCid: null,
    isFolder: true,
  },
  {
    cid: 'cid-vacation-pic',
    name: 'sunset-beach.jpg',
    category: 'image',
    sensitivity: 'private',
    sizeBytes: 4_500_000,
    storedAt: now - 60 * DAY,
    lastAccessed: now - 45 * DAY,
    accessCount: 3,
    stalenessScore: 0.52,
    replicationTier: 'default',
    replicaCount: 3,
    pinned: false,
    licensed: false,
    parentCid: 'cid-photos-2026',
    isFolder: false,
  },
  {
    cid: 'cid-tax-docs',
    name: 'Tax Returns 2025',
    category: 'bundle',
    sensitivity: 'confidential',
    sizeBytes: 15_000_000,
    storedAt: now - 365 * DAY,
    lastAccessed: now - 200 * DAY,
    accessCount: 2,
    stalenessScore: 0.15,
    replicationTier: 'high',
    replicaCount: 5,
    pinned: true,
    licensed: false,
    parentCid: null,
    isFolder: true,
  },
  {
    cid: 'cid-old-draft',
    name: 'blog-draft-v1.txt',
    category: 'text',
    sensitivity: 'private',
    sizeBytes: 12_000,
    storedAt: now - 180 * DAY,
    lastAccessed: now - 170 * DAY,
    accessCount: 1,
    stalenessScore: 0.88,
    replicationTier: 'expendable',
    replicaCount: 1,
    pinned: false,
    licensed: false,
    parentCid: null,
    isFolder: false,
  },
  {
    cid: 'cid-my-song',
    name: 'late-night-demo.mp3',
    category: 'music',
    sensitivity: 'private',
    sizeBytes: 8_000_000,
    storedAt: now - 30 * DAY,
    lastAccessed: now - 2 * DAY,
    accessCount: 25,
    stalenessScore: 0.08,
    replicationTier: 'default',
    replicaCount: 3,
    pinned: false,
    licensed: false,
    parentCid: null,
    isFolder: false,
  },
  {
    cid: 'cid-project-archive',
    name: 'harmony-prototype-v0.tar',
    category: 'software',
    sensitivity: 'private',
    sizeBytes: 150_000_000,
    storedAt: now - 120 * DAY,
    lastAccessed: now - 100 * DAY,
    accessCount: 1,
    stalenessScore: 0.72,
    replicationTier: 'light',
    replicaCount: 2,
    pinned: false,
    licensed: false,
    parentCid: null,
    isFolder: false,
  },
  {
    cid: 'cid-journal',
    name: 'personal-journal.txt',
    category: 'text',
    sensitivity: 'intimate',
    sizeBytes: 250_000,
    storedAt: now - 60 * DAY,
    lastAccessed: now - 1 * DAY,
    accessCount: 45,
    stalenessScore: 0.02,
    replicationTier: 'high',
    replicaCount: 5,
    pinned: true,
    licensed: false,
    parentCid: null,
    isFolder: false,
  },
  {
    cid: 'cid-dataset-research',
    name: 'mesh-latency-data.csv',
    category: 'dataset',
    sensitivity: 'private',
    sizeBytes: 75_000_000,
    storedAt: now - 45 * DAY,
    lastAccessed: now - 40 * DAY,
    accessCount: 2,
    stalenessScore: 0.61,
    replicationTier: 'default',
    replicaCount: 2,
    pinned: false,
    licensed: false,
    parentCid: null,
    isFolder: false,
  },
  {
    cid: 'cid-under-replicated',
    name: 'important-contract.pdf',
    category: 'text',
    sensitivity: 'confidential',
    sizeBytes: 2_000_000,
    storedAt: now - 10 * DAY,
    lastAccessed: now - 5 * DAY,
    accessCount: 3,
    stalenessScore: 0.0,
    replicationTier: 'high',
    replicaCount: 2,
    pinned: false,
    licensed: false,
    parentCid: null,
    isFolder: false,
  },
];

export const mockPublishedContent: PublishedItem[] = [
  {
    cid: 'pub-cid-song',
    name: 'Midnight Drive.mp3',
    category: 'music',
    sizeBytes: 9_500_000,
    publishedAt: now - 14 * DAY,
    publishMode: 'durable',
  },
  {
    cid: 'pub-cid-article',
    name: 'Decentralizing the Web.md',
    category: 'text',
    sizeBytes: 35_000,
    publishedAt: now - 7 * DAY,
    publishMode: 'durable',
  },
  {
    cid: 'pub-cid-dataset',
    name: 'network-benchmarks-2026.csv',
    category: 'dataset',
    sizeBytes: 12_000_000,
    publishedAt: now - 2 * DAY,
    publishMode: 'ephemeral',
  },
];

export const mockCleanupRecommendations: CleanupRecommendation[] = [
  {
    cid: 'cid-old-draft',
    name: 'blog-draft-v1.txt',
    category: 'text',
    sizeBytes: 12_000,
    reason: 'stale',
    stalenessScore: 0.88,
    spaceRecoverable: 12_000,
    confidence: 0.88,
  },
  {
    cid: 'cid-project-archive',
    name: 'harmony-prototype-v0.tar',
    category: 'software',
    sizeBytes: 150_000_000,
    reason: 'stale',
    stalenessScore: 0.72,
    spaceRecoverable: 300_000_000,
    confidence: 0.72,
  },
  {
    cid: 'cid-dataset-research',
    name: 'mesh-latency-data.csv',
    category: 'dataset',
    sizeBytes: 75_000_000,
    reason: 'stale',
    stalenessScore: 0.61,
    spaceRecoverable: 225_000_000,
    confidence: 0.61,
  },
];

export function mockQuotaStatus(): QuotaStatus {
  const items = mockPrivateContent;
  const used = items.reduce((sum, item) => sum + item.sizeBytes, 0);
  const byCategory: Partial<Record<string, number>> = {};
  for (const item of items) {
    byCategory[item.category] = (byCategory[item.category] ?? 0) + item.sizeBytes;
  }
  return {
    usedBytes: used,
    totalBytes: 10_000_000_000,
    byCategory,
  };
}
```

**Step 2: Write the service tests**

Create `src/lib/file-manager-service.test.ts`:

```typescript
import { describe, it, expect } from 'vitest';
import { FileManagerService } from './file-manager-service';

describe('FileManagerService', () => {
  it('constructs with default settings', () => {
    const service = new FileManagerService();
    expect(service.settings.defaultReplicationTier).toBe('default');
    expect(service.settings.quotaBytes).toBe(10_000_000_000);
  });

  it('returns private content', () => {
    const service = new FileManagerService();
    const items = service.getContents();
    expect(items.length).toBeGreaterThan(0);
    expect(items.every((i) => i.cid && i.name)).toBe(true);
  });

  it('filters content by parentCid', () => {
    const service = new FileManagerService();
    const root = service.getContents(null);
    expect(root.every((i) => i.parentCid === null)).toBe(true);
    const children = service.getContents('cid-photos-2026');
    expect(children.every((i) => i.parentCid === 'cid-photos-2026')).toBe(true);
  });

  it('returns quota status', () => {
    const service = new FileManagerService();
    const quota = service.getQuotaStatus();
    expect(quota.totalBytes).toBe(10_000_000_000);
    expect(quota.usedBytes).toBeGreaterThan(0);
    expect(quota.usedBytes).toBeLessThan(quota.totalBytes);
  });

  it('returns cleanup recommendations sorted by confidence', () => {
    const service = new FileManagerService();
    const recs = service.getCleanupRecommendations();
    expect(recs.length).toBeGreaterThan(0);
    for (let i = 1; i < recs.length; i++) {
      expect(recs[i - 1].confidence).toBeGreaterThanOrEqual(recs[i].confidence);
    }
  });

  it('returns storage buddies', () => {
    const service = new FileManagerService();
    const buddies = service.getStorageBuddies();
    expect(buddies.length).toBeGreaterThan(0);
    expect(buddies[0].address).toBeTruthy();
  });

  it('returns published content', () => {
    const service = new FileManagerService();
    const published = service.getPublishedContent();
    expect(published.length).toBeGreaterThan(0);
    expect(published.every((p) => p.publishMode === 'durable' || p.publishMode === 'ephemeral')).toBe(true);
  });

  it('burn removes content and frees quota', () => {
    const service = new FileManagerService();
    const before = service.getContents();
    const target = before.find((i) => i.cid === 'cid-old-draft');
    expect(target).toBeTruthy();
    service.burn(['cid-old-draft']);
    const after = service.getContents();
    expect(after.find((i) => i.cid === 'cid-old-draft')).toBeUndefined();
  });

  it('pin toggles pinned state', () => {
    const service = new FileManagerService();
    const item = service.getContents().find((i) => !i.pinned);
    expect(item).toBeTruthy();
    service.pin(item!.cid);
    const updated = service.getContents().find((i) => i.cid === item!.cid);
    expect(updated!.pinned).toBe(true);
  });

  it('publish moves content to published and removes from private', () => {
    const service = new FileManagerService();
    const beforePrivate = service.getContents().length;
    const beforePublished = service.getPublishedContent().length;
    service.publish(['cid-my-song']);
    expect(service.getContents().length).toBe(beforePrivate - 1);
    expect(service.getPublishedContent().length).toBe(beforePublished + 1);
    const pub = service.getPublishedContent().find((p) => p.cid === 'cid-my-song');
    expect(pub?.publishMode).toBe('durable');
  });

  it('release moves content to published with ephemeral mode', () => {
    const service = new FileManagerService();
    service.release(['cid-old-draft']);
    const pub = service.getPublishedContent().find((p) => p.cid === 'cid-old-draft');
    expect(pub?.publishMode).toBe('ephemeral');
  });

  it('setReplicationTier updates tier', () => {
    const service = new FileManagerService();
    service.setReplicationTier(['cid-my-song'], 'high');
    const item = service.getContents().find((i) => i.cid === 'cid-my-song');
    expect(item?.replicationTier).toBe('high');
  });
});
```

**Step 3: Run tests to verify they fail**

Run: `cd /Users/zeblith/work/zeblithic/harmony-client && npx vitest run src/lib/file-manager-service.test.ts`
Expected: FAIL — `FileManagerService` doesn't exist

**Step 4: Implement the service**

Create `src/lib/file-manager-service.ts`:

```typescript
import type {
  ContentItem,
  ContentDetail,
  QuotaStatus,
  CleanupRecommendation,
  StorageBuddy,
  PublishedItem,
  FileManagerSettings,
  ReplicationTier,
  ContentCategory,
} from './types';
import {
  mockPrivateContent,
  mockPublishedContent,
  mockCleanupRecommendations,
  mockStorageBuddies,
  mockQuotaStatus,
  mockPeers,
} from './mock-file-data';

export class FileManagerService {
  readonly settings: FileManagerSettings;
  private privateContent: ContentItem[];
  private publishedContent: PublishedItem[];

  constructor(settings?: Partial<FileManagerSettings>) {
    this.settings = {
      defaultReplicationTier: settings?.defaultReplicationTier ?? 'default',
      quotaBytes: settings?.quotaBytes ?? 10_000_000_000,
      defaultViewMode: settings?.defaultViewMode ?? 'list',
      confirmationOverrides: settings?.confirmationOverrides ?? {},
    };
    this.privateContent = structuredClone(mockPrivateContent);
    this.publishedContent = structuredClone(mockPublishedContent);
  }

  getContents(parentCid?: string | null): ContentItem[] {
    if (parentCid === undefined) return this.privateContent;
    return this.privateContent.filter((i) => i.parentCid === parentCid);
  }

  getContentDetail(cid: string): ContentDetail | undefined {
    const item = this.privateContent.find((i) => i.cid === cid);
    if (!item) return undefined;
    return {
      ...item,
      sharedWith: cid === 'cid-my-song' ? [mockPeers[0]] : [],
      storageBuddies: mockStorageBuddies.map((b) => ({ address: b.address, displayName: b.displayName })),
      origin: item.parentCid ? 'self-created' : 'self-created',
    };
  }

  getQuotaStatus(): QuotaStatus {
    const used = this.privateContent.reduce((sum, item) => sum + item.sizeBytes, 0);
    const byCategory: Partial<Record<ContentCategory, number>> = {};
    for (const item of this.privateContent) {
      byCategory[item.category] = (byCategory[item.category] ?? 0) + item.sizeBytes;
    }
    return { usedBytes: used, totalBytes: this.settings.quotaBytes, byCategory };
  }

  getCleanupRecommendations(): CleanupRecommendation[] {
    return mockCleanupRecommendations
      .filter((r) => this.privateContent.some((i) => i.cid === r.cid))
      .sort((a, b) => b.confidence - a.confidence);
  }

  getStorageBuddies(): StorageBuddy[] {
    return structuredClone(mockStorageBuddies);
  }

  getPublishedContent(): PublishedItem[] {
    return this.publishedContent;
  }

  burn(cids: string[]): void {
    this.privateContent = this.privateContent.filter((i) => !cids.includes(i.cid));
  }

  archive(cids: string[]): void {
    // In the real implementation, moves to cold storage tier.
    // For now, just mark by removing from active list.
    this.privateContent = this.privateContent.filter((i) => !cids.includes(i.cid));
  }

  publish(cids: string[]): void {
    for (const cid of cids) {
      const idx = this.privateContent.findIndex((i) => i.cid === cid);
      if (idx === -1) continue;
      const item = this.privateContent[idx];
      this.publishedContent.push({
        cid: item.cid,
        name: item.name,
        category: item.category,
        sizeBytes: item.sizeBytes,
        publishedAt: Date.now(),
        publishMode: 'durable',
      });
      this.privateContent.splice(idx, 1);
    }
  }

  release(cids: string[]): void {
    for (const cid of cids) {
      const idx = this.privateContent.findIndex((i) => i.cid === cid);
      if (idx === -1) continue;
      const item = this.privateContent[idx];
      this.publishedContent.push({
        cid: item.cid,
        name: item.name,
        category: item.category,
        sizeBytes: item.sizeBytes,
        publishedAt: Date.now(),
        publishMode: 'ephemeral',
      });
      this.privateContent.splice(idx, 1);
    }
  }

  pin(cid: string): void {
    const item = this.privateContent.find((i) => i.cid === cid);
    if (item) item.pinned = true;
  }

  unpin(cid: string): void {
    const item = this.privateContent.find((i) => i.cid === cid);
    if (item) item.pinned = false;
  }

  setReplicationTier(cids: string[], tier: ReplicationTier): void {
    for (const cid of cids) {
      const item = this.privateContent.find((i) => i.cid === cid);
      if (item) item.replicationTier = tier;
    }
  }

  exportToDevice(_cids: string[]): void {
    // Stub: real implementation writes reassembled Merkle DAG via Tauri file dialog.
  }
}
```

**Step 5: Run tests to verify they pass**

Run: `cd /Users/zeblith/work/zeblithic/harmony-client && npx vitest run src/lib/file-manager-service.test.ts`
Expected: PASS — all 12 tests green

**Step 6: Run all tests**

Run: `cd /Users/zeblith/work/zeblithic/harmony-client && npx vitest run`
Expected: All pass

**Step 7: Commit**

```bash
cd /Users/zeblith/work/zeblithic/harmony-client
git add src/lib/file-manager-service.ts src/lib/file-manager-service.test.ts src/lib/mock-file-data.ts
git commit -m "feat(files): add FileManagerService with mock data

Class-based service (same pattern as TrustService) wrapping all file
manager operations: getContents, quota, cleanup recs, publish, release,
burn, archive, pin, replication tier. All backed by mock data."
```

---

### Task 3: Mode Integration — Layout & NavPanel

**Files:**
- Modify: `src/lib/components/Layout.svelte`
- Modify: `src/lib/components/NavPanel.svelte`
- Modify: `src/App.svelte`
- Test: `src/lib/components/__tests__/Layout.test.ts` (new)

**Context:** This task wires the `'files'` mode into the app shell. The mode toggle becomes a 3-way toggle. Layout.svelte gets a `files-mode` grid. App.svelte instantiates `FileManagerService` and renders placeholder file components.

**Step 1: Write Layout test for files mode**

Create `src/lib/components/__tests__/Layout.test.ts`:

```typescript
import { render, screen } from '@testing-library/svelte';
import { describe, it, expect } from 'vitest';
import Layout from '../Layout.svelte';

describe('Layout', () => {
  it('renders files-mode class when mode is files', () => {
    const { container } = render(Layout, {
      props: {
        mode: 'files',
        collapsed: false,
        showSettings: false,
        nav: () => {},
        textFeed: () => {},
        mediaFeed: () => {},
        fileBrowser: () => {},
        fileDetailPanel: () => {},
      },
    });
    const layout = container.querySelector('.layout');
    expect(layout?.classList.contains('files-mode')).toBe(true);
  });

  it('does not render text-area in files mode', () => {
    const { container } = render(Layout, {
      props: {
        mode: 'files',
        collapsed: false,
        showSettings: false,
        nav: () => {},
        textFeed: () => {},
        mediaFeed: () => {},
        fileBrowser: () => {},
        fileDetailPanel: () => {},
      },
    });
    expect(container.querySelector('.text-area')).toBeNull();
  });
});
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/zeblith/work/zeblithic/harmony-client && npx vitest run src/lib/components/__tests__/Layout.test.ts`
Expected: FAIL — `files-mode` class doesn't exist, `fileBrowser` snippet not accepted

**Step 3: Update Layout.svelte**

Add `fileBrowser` and `fileDetailPanel` snippet props to the props interface. Add a `files-mode` branch to the template, and a `.files-mode` grid CSS rule.

In the `<script>` section, add the new snippet props alongside the existing ones:

```typescript
let {
  mode = 'messages',
  collapsed = false,
  showSettings = false,
  nav,
  textFeed,
  mediaFeed,
  settingsPanel,
  vineFeed,
  fileBrowser,
  fileDetailPanel,
}: {
  mode?: AppMode;
  collapsed?: boolean;
  showSettings?: boolean;
  nav: Snippet;
  textFeed: Snippet;
  mediaFeed: Snippet;
  settingsPanel?: Snippet;
  vineFeed?: Snippet;
  fileBrowser?: Snippet;
  fileDetailPanel?: Snippet;
} = $props();
```

**Important:** Import `AppMode` from `'../types'` at the top of the script, and `Snippet` from `'svelte'`.

In the template, add a `files-mode` branch. The full conditional becomes:

```svelte
<div class="layout" class:collapsed class:vine-mode={mode === 'vines' && vineFeed} class:files-mode={mode === 'files' && fileBrowser}>
  <aside class="nav-area">
    {@render nav()}
  </aside>
  {#if mode === 'files' && fileBrowser}
    <main class="files-area">
      {@render fileBrowser()}
    </main>
    {#if !collapsed && fileDetailPanel}
      <section class="detail-area">
        {@render fileDetailPanel()}
      </section>
    {/if}
  {:else if mode === 'vines' && vineFeed}
    <main class="vine-area">
      {@render vineFeed()}
    </main>
  {:else}
    <main class="text-area">
      {@render textFeed()}
    </main>
    {#if !collapsed}
      <section class="media-area">
        {#if showSettings && settingsPanel}
          {@render settingsPanel()}
        {:else}
          {@render mediaFeed()}
        {/if}
      </section>
    {/if}
  {/if}
</div>
```

Add CSS for files mode:

```css
.layout.files-mode {
  grid-template-columns: var(--nav-width) 1fr 320px;
  grid-template-areas: "nav files detail";
}

.layout.files-mode.collapsed {
  grid-template-columns: var(--nav-width-collapsed) 1fr 320px;
  grid-template-areas: "nav files detail";
}

.files-area {
  grid-area: files;
  background: var(--bg-primary);
  overflow-y: auto;
  display: flex;
  flex-direction: column;
}

.detail-area {
  grid-area: detail;
  background: var(--bg-secondary);
  overflow-y: auto;
  padding: 12px;
  border-left: 1px solid var(--border);
}
```

**Step 4: Update NavPanel.svelte for 3-way mode toggle**

In the mode toggle area (around lines 133-142), replace the single toggle button with a button group:

```svelte
<div class="mode-toggles" role="group" aria-label="App mode">
  <button
    type="button"
    class="nav-action-btn mode-toggle"
    class:active={appMode === 'messages'}
    aria-label="Messages"
    aria-pressed={appMode === 'messages'}
    onclick={() => onModeChange?.('messages')}
  >
    Messages
  </button>
  <button
    type="button"
    class="nav-action-btn mode-toggle"
    class:active={appMode === 'vines'}
    aria-label="Vines"
    aria-pressed={appMode === 'vines'}
    onclick={() => onModeChange?.('vines')}
  >
    Vines
  </button>
  <button
    type="button"
    class="nav-action-btn mode-toggle"
    class:active={appMode === 'files'}
    aria-label="Files"
    aria-pressed={appMode === 'files'}
    onclick={() => onModeChange?.('files')}
  >
    Files
  </button>
</div>
```

Update the props to replace `onModeToggle` with `onModeChange`:

```typescript
onModeChange?: (mode: AppMode) => void;
```

Add CSS for the button group:

```css
.mode-toggles {
  display: flex;
  gap: 2px;
}

.mode-toggles .mode-toggle {
  flex: 1;
  font-size: 0.75rem;
  padding: 4px 6px;
}
```

**Step 5: Update App.svelte**

Import `FileManagerService`:

```typescript
import { FileManagerService } from './lib/file-manager-service';
```

Initialize alongside other services (near line 57):

```typescript
const fileManagerService = new FileManagerService();
let fileManagerVersion = $state(0);
```

Replace the mode toggle handler (near line 201) — change from:
```typescript
onModeToggle={() => { appMode = appMode === 'messages' ? 'vines' : 'messages'; showSettings = false; }}
```
to:
```typescript
onModeChange={(mode) => { appMode = mode; showSettings = false; }}
```

Add file-mode snippet props to the `<Layout>` component:

```svelte
{#snippet fileBrowser()}
  <div class="placeholder" style="padding: 24px; color: var(--text-secondary);">
    File Browser — coming in Task 5
  </div>
{/snippet}

{#snippet fileDetailPanel()}
  <div class="placeholder" style="padding: 24px; color: var(--text-secondary);">
    Detail Panel — coming in Task 8
  </div>
{/snippet}
```

Pass them to Layout:

```svelte
<Layout
  {mode}
  ...
  {fileBrowser}
  {fileDetailPanel}
>
```

**Step 6: Run all tests**

Run: `cd /Users/zeblith/work/zeblithic/harmony-client && npx vitest run`
Expected: All pass (including new Layout test). Existing NavPanel tests may need updating — the `onModeToggle` prop is now `onModeChange`. Update those tests to pass the new prop name.

**Step 7: Commit**

```bash
cd /Users/zeblith/work/zeblithic/harmony-client
git add src/lib/components/Layout.svelte src/lib/components/NavPanel.svelte src/App.svelte src/lib/components/__tests__/Layout.test.ts
git commit -m "feat(files): integrate files mode into Layout and NavPanel

Three-way mode toggle (Messages/Vines/Files). Layout.svelte adds
files-mode grid with files-area and detail-area columns. Placeholder
snippets in App.svelte for progressive component replacement."
```

---

### Task 4: Confirmation Gate Components

**Files:**
- Create: `src/lib/components/ConfirmDialog.svelte`
- Create: `src/lib/components/DoubleConfirmDialog.svelte`
- Create: `src/lib/components/TypeToConfirmDialog.svelte`
- Test: `src/lib/components/__tests__/ConfirmDialog.test.ts`
- Test: `src/lib/components/__tests__/DoubleConfirmDialog.test.ts`
- Test: `src/lib/components/__tests__/TypeToConfirmDialog.test.ts`

**Context:** These are shared, reusable dialog components used by publish, release, and burn flows. Built before the file-specific components so they're available to compose. The `OutOfBandVerify` component is deferred — it's a stub within the triple-confirm flow for now.

**Step 1: Write ConfirmDialog test**

Create `src/lib/components/__tests__/ConfirmDialog.test.ts`:

```typescript
import { render, screen, fireEvent } from '@testing-library/svelte';
import { describe, it, expect, vi } from 'vitest';
import ConfirmDialog from '../ConfirmDialog.svelte';

describe('ConfirmDialog', () => {
  it('renders title and message', () => {
    render(ConfirmDialog, {
      props: {
        title: 'Confirm Delete',
        message: 'Are you sure?',
        confirmLabel: 'Delete',
        onConfirm: vi.fn(),
        onCancel: vi.fn(),
      },
    });
    expect(screen.getByText('Confirm Delete')).toBeTruthy();
    expect(screen.getByText('Are you sure?')).toBeTruthy();
  });

  it('has role="dialog" with aria-modal', () => {
    render(ConfirmDialog, {
      props: {
        title: 'Test',
        message: 'Test message',
        confirmLabel: 'OK',
        onConfirm: vi.fn(),
        onCancel: vi.fn(),
      },
    });
    const dialog = screen.getByRole('dialog');
    expect(dialog.getAttribute('aria-modal')).toBe('true');
  });

  it('calls onConfirm when confirm button clicked', async () => {
    const onConfirm = vi.fn();
    render(ConfirmDialog, {
      props: {
        title: 'Test',
        message: 'Msg',
        confirmLabel: 'Confirm',
        onConfirm,
        onCancel: vi.fn(),
      },
    });
    await fireEvent.click(screen.getByRole('button', { name: 'Confirm' }));
    expect(onConfirm).toHaveBeenCalledOnce();
  });

  it('calls onCancel when cancel button clicked', async () => {
    const onCancel = vi.fn();
    render(ConfirmDialog, {
      props: {
        title: 'Test',
        message: 'Msg',
        confirmLabel: 'OK',
        onConfirm: vi.fn(),
        onCancel,
      },
    });
    await fireEvent.click(screen.getByRole('button', { name: 'Cancel' }));
    expect(onCancel).toHaveBeenCalledOnce();
  });

  it('applies destructive class when destructive prop is true', () => {
    const { container } = render(ConfirmDialog, {
      props: {
        title: 'Test',
        message: 'Msg',
        confirmLabel: 'Delete',
        destructive: true,
        onConfirm: vi.fn(),
        onCancel: vi.fn(),
      },
    });
    expect(container.querySelector('.confirm-btn.destructive')).toBeTruthy();
  });
});
```

**Step 2: Write DoubleConfirmDialog test**

Create `src/lib/components/__tests__/DoubleConfirmDialog.test.ts`:

```typescript
import { render, screen, fireEvent } from '@testing-library/svelte';
import { describe, it, expect, vi } from 'vitest';
import DoubleConfirmDialog from '../DoubleConfirmDialog.svelte';

describe('DoubleConfirmDialog', () => {
  const defaultProps = {
    title: 'Publish Content',
    firstMessage: 'Anyone in the world can access this.',
    secondMessage: 'This is irreversible.',
    confirmLabel: 'Yes, publish',
    onConfirm: vi.fn(),
    onCancel: vi.fn(),
  };

  it('shows first gate initially', () => {
    render(DoubleConfirmDialog, { props: defaultProps });
    expect(screen.getByText('Anyone in the world can access this.')).toBeTruthy();
  });

  it('advances to second gate on Continue', async () => {
    render(DoubleConfirmDialog, { props: defaultProps });
    await fireEvent.click(screen.getByRole('button', { name: 'Continue' }));
    expect(screen.getByText('This is irreversible.')).toBeTruthy();
  });

  it('calls onConfirm only after second gate', async () => {
    const onConfirm = vi.fn();
    render(DoubleConfirmDialog, { props: { ...defaultProps, onConfirm } });
    await fireEvent.click(screen.getByRole('button', { name: 'Continue' }));
    await fireEvent.click(screen.getByRole('button', { name: 'Yes, publish' }));
    expect(onConfirm).toHaveBeenCalledOnce();
  });

  it('Cancel at gate 1 calls onCancel', async () => {
    const onCancel = vi.fn();
    render(DoubleConfirmDialog, { props: { ...defaultProps, onCancel } });
    await fireEvent.click(screen.getByRole('button', { name: 'Cancel' }));
    expect(onCancel).toHaveBeenCalledOnce();
  });
});
```

**Step 3: Write TypeToConfirmDialog test**

Create `src/lib/components/__tests__/TypeToConfirmDialog.test.ts`:

```typescript
import { render, screen, fireEvent } from '@testing-library/svelte';
import { describe, it, expect, vi } from 'vitest';
import TypeToConfirmDialog from '../TypeToConfirmDialog.svelte';

describe('TypeToConfirmDialog', () => {
  const defaultProps = {
    title: 'Confirm Publication',
    message: 'Type the file name to confirm.',
    confirmText: 'my-file.txt',
    confirmLabel: 'Publish',
    onConfirm: vi.fn(),
    onCancel: vi.fn(),
  };

  it('confirm button is disabled until text matches', () => {
    render(TypeToConfirmDialog, { props: defaultProps });
    const btn = screen.getByRole('button', { name: 'Publish' });
    expect(btn.hasAttribute('disabled')).toBe(true);
  });

  it('enables confirm button when typed text matches', async () => {
    render(TypeToConfirmDialog, { props: defaultProps });
    const input = screen.getByRole('textbox');
    await fireEvent.input(input, { target: { value: 'my-file.txt' } });
    const btn = screen.getByRole('button', { name: 'Publish' });
    expect(btn.hasAttribute('disabled')).toBe(false);
  });

  it('match is case-sensitive', async () => {
    render(TypeToConfirmDialog, { props: defaultProps });
    const input = screen.getByRole('textbox');
    await fireEvent.input(input, { target: { value: 'My-File.txt' } });
    const btn = screen.getByRole('button', { name: 'Publish' });
    expect(btn.hasAttribute('disabled')).toBe(true);
  });
});
```

**Step 4: Run tests to verify they fail**

Run: `cd /Users/zeblith/work/zeblithic/harmony-client && npx vitest run src/lib/components/__tests__/ConfirmDialog.test.ts src/lib/components/__tests__/DoubleConfirmDialog.test.ts src/lib/components/__tests__/TypeToConfirmDialog.test.ts`
Expected: FAIL — components don't exist

**Step 5: Implement all three dialog components**

Create `src/lib/components/ConfirmDialog.svelte`:

```svelte
<script lang="ts">
  let {
    title,
    message,
    confirmLabel,
    destructive = false,
    onConfirm,
    onCancel,
  }: {
    title: string;
    message: string;
    confirmLabel: string;
    destructive?: boolean;
    onConfirm: () => void;
    onCancel: () => void;
  } = $props();
</script>

<div class="dialog-overlay">
  <div class="dialog" role="dialog" aria-modal="true" aria-labelledby="dialog-title">
    <h2 id="dialog-title" class="dialog-title">{title}</h2>
    <p class="dialog-message">{message}</p>
    <div class="dialog-actions">
      <button type="button" class="dialog-btn cancel-btn" onclick={onCancel}>Cancel</button>
      <button type="button" class="dialog-btn confirm-btn" class:destructive onclick={onConfirm}>
        {confirmLabel}
      </button>
    </div>
  </div>
</div>

<style>
  .dialog-overlay {
    position: fixed;
    inset: 0;
    background: rgba(0, 0, 0, 0.6);
    display: flex;
    align-items: center;
    justify-content: center;
    z-index: 100;
  }
  .dialog {
    background: var(--bg-secondary);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 24px;
    max-width: 480px;
    width: 90%;
  }
  .dialog-title {
    color: var(--text-primary);
    font-size: 1.1rem;
    margin: 0 0 12px;
  }
  .dialog-message {
    color: var(--text-secondary);
    font-size: 0.9rem;
    line-height: 1.5;
    margin: 0 0 20px;
  }
  .dialog-actions {
    display: flex;
    justify-content: flex-end;
    gap: 8px;
  }
  .dialog-btn {
    padding: 8px 16px;
    border-radius: 4px;
    border: none;
    cursor: pointer;
    font-size: 0.85rem;
  }
  .cancel-btn {
    background: var(--bg-tertiary);
    color: var(--text-secondary);
  }
  .confirm-btn {
    background: var(--accent);
    color: var(--text-primary);
  }
  .confirm-btn.destructive {
    background: #d83c3e;
  }
</style>
```

Create `src/lib/components/DoubleConfirmDialog.svelte`:

```svelte
<script lang="ts">
  let {
    title,
    firstMessage,
    secondMessage,
    confirmLabel,
    destructive = false,
    onConfirm,
    onCancel,
  }: {
    title: string;
    firstMessage: string;
    secondMessage: string;
    confirmLabel: string;
    destructive?: boolean;
    onConfirm: () => void;
    onCancel: () => void;
  } = $props();

  let gate = $state(1);
</script>

<div class="dialog-overlay">
  <div class="dialog" role="dialog" aria-modal="true" aria-labelledby="dialog-title">
    <h2 id="dialog-title" class="dialog-title">{title}</h2>
    {#if gate === 1}
      <p class="dialog-message">{firstMessage}</p>
      <div class="dialog-actions">
        <button type="button" class="dialog-btn cancel-btn" onclick={onCancel}>Cancel</button>
        <button type="button" class="dialog-btn confirm-btn" onclick={() => gate = 2}>Continue</button>
      </div>
    {:else}
      <p class="dialog-message">{secondMessage}</p>
      <div class="dialog-actions">
        <button type="button" class="dialog-btn cancel-btn" onclick={onCancel}>Cancel</button>
        <button type="button" class="dialog-btn confirm-btn" class:destructive onclick={onConfirm}>
          {confirmLabel}
        </button>
      </div>
    {/if}
  </div>
</div>

<style>
  .dialog-overlay {
    position: fixed;
    inset: 0;
    background: rgba(0, 0, 0, 0.6);
    display: flex;
    align-items: center;
    justify-content: center;
    z-index: 100;
  }
  .dialog {
    background: var(--bg-secondary);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 24px;
    max-width: 480px;
    width: 90%;
  }
  .dialog-title {
    color: var(--text-primary);
    font-size: 1.1rem;
    margin: 0 0 12px;
  }
  .dialog-message {
    color: var(--text-secondary);
    font-size: 0.9rem;
    line-height: 1.5;
    margin: 0 0 20px;
  }
  .dialog-actions {
    display: flex;
    justify-content: flex-end;
    gap: 8px;
  }
  .dialog-btn {
    padding: 8px 16px;
    border-radius: 4px;
    border: none;
    cursor: pointer;
    font-size: 0.85rem;
  }
  .cancel-btn {
    background: var(--bg-tertiary);
    color: var(--text-secondary);
  }
  .confirm-btn {
    background: var(--accent);
    color: var(--text-primary);
  }
  .confirm-btn.destructive {
    background: #d83c3e;
  }
</style>
```

Create `src/lib/components/TypeToConfirmDialog.svelte`:

```svelte
<script lang="ts">
  let {
    title,
    message,
    confirmText,
    confirmLabel,
    onConfirm,
    onCancel,
  }: {
    title: string;
    message: string;
    confirmText: string;
    confirmLabel: string;
    onConfirm: () => void;
    onCancel: () => void;
  } = $props();

  let typed = $state('');
  let matches = $derived(typed === confirmText);
</script>

<div class="dialog-overlay">
  <div class="dialog" role="dialog" aria-modal="true" aria-labelledby="dialog-title">
    <h2 id="dialog-title" class="dialog-title">{title}</h2>
    <p class="dialog-message">{message}</p>
    <p class="confirm-hint">Type <code>{confirmText}</code> to confirm:</p>
    <input
      type="text"
      class="confirm-input"
      role="textbox"
      bind:value={typed}
      aria-label="Type to confirm"
    />
    <div class="dialog-actions">
      <button type="button" class="dialog-btn cancel-btn" onclick={onCancel}>Cancel</button>
      <button type="button" class="dialog-btn confirm-btn" disabled={!matches} onclick={onConfirm}>
        {confirmLabel}
      </button>
    </div>
  </div>
</div>

<style>
  .dialog-overlay {
    position: fixed;
    inset: 0;
    background: rgba(0, 0, 0, 0.6);
    display: flex;
    align-items: center;
    justify-content: center;
    z-index: 100;
  }
  .dialog {
    background: var(--bg-secondary);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 24px;
    max-width: 480px;
    width: 90%;
  }
  .dialog-title {
    color: var(--text-primary);
    font-size: 1.1rem;
    margin: 0 0 12px;
  }
  .dialog-message {
    color: var(--text-secondary);
    font-size: 0.9rem;
    line-height: 1.5;
    margin: 0 0 12px;
  }
  .confirm-hint {
    color: var(--text-secondary);
    font-size: 0.85rem;
    margin: 0 0 8px;
  }
  .confirm-hint code {
    background: var(--bg-tertiary);
    padding: 2px 6px;
    border-radius: 3px;
    color: var(--text-primary);
  }
  .confirm-input {
    width: 100%;
    padding: 8px 12px;
    background: var(--bg-primary);
    border: 1px solid var(--border);
    border-radius: 4px;
    color: var(--text-primary);
    font-size: 0.9rem;
    margin-bottom: 20px;
    box-sizing: border-box;
  }
  .dialog-actions {
    display: flex;
    justify-content: flex-end;
    gap: 8px;
  }
  .dialog-btn {
    padding: 8px 16px;
    border-radius: 4px;
    border: none;
    cursor: pointer;
    font-size: 0.85rem;
  }
  .cancel-btn {
    background: var(--bg-tertiary);
    color: var(--text-secondary);
  }
  .confirm-btn {
    background: var(--accent);
    color: var(--text-primary);
  }
  .confirm-btn:disabled {
    opacity: 0.4;
    cursor: not-allowed;
  }
</style>
```

**Step 6: Run tests to verify they pass**

Run: `cd /Users/zeblith/work/zeblithic/harmony-client && npx vitest run src/lib/components/__tests__/ConfirmDialog.test.ts src/lib/components/__tests__/DoubleConfirmDialog.test.ts src/lib/components/__tests__/TypeToConfirmDialog.test.ts`
Expected: PASS — all 12 tests green

**Step 7: Run all tests**

Run: `cd /Users/zeblith/work/zeblithic/harmony-client && npx vitest run`
Expected: All pass

**Step 8: Commit**

```bash
cd /Users/zeblith/work/zeblithic/harmony-client
git add src/lib/components/ConfirmDialog.svelte src/lib/components/DoubleConfirmDialog.svelte src/lib/components/TypeToConfirmDialog.svelte src/lib/components/__tests__/ConfirmDialog.test.ts src/lib/components/__tests__/DoubleConfirmDialog.test.ts src/lib/components/__tests__/TypeToConfirmDialog.test.ts
git commit -m "feat(files): add graduated confirmation gate components

ConfirmDialog (single gate), DoubleConfirmDialog (two gates),
TypeToConfirmDialog (type-to-confirm). Reusable across publish,
release, and burn flows. Full ARIA dialog semantics."
```

---

### Task 5: QuotaBar Component

**Files:**
- Create: `src/lib/components/QuotaBar.svelte`
- Test: `src/lib/components/__tests__/QuotaBar.test.ts`

**Context:** Persistent bar at the bottom of the FileBrowser column. Shows used/total with color gradient. Clicking opens cleanup view. Only shows private content quota.

**Step 1: Write test**

Create `src/lib/components/__tests__/QuotaBar.test.ts`:

```typescript
import { render, screen, fireEvent } from '@testing-library/svelte';
import { describe, it, expect, vi } from 'vitest';
import QuotaBar from '../QuotaBar.svelte';

describe('QuotaBar', () => {
  it('renders usage text', () => {
    render(QuotaBar, {
      props: { usedBytes: 5_000_000_000, totalBytes: 10_000_000_000, onCleanupClick: vi.fn() },
    });
    expect(screen.getByText(/5\.0 GB/)).toBeTruthy();
    expect(screen.getByText(/10\.0 GB/)).toBeTruthy();
  });

  it('has role="progressbar" with aria attributes', () => {
    render(QuotaBar, {
      props: { usedBytes: 3_000_000_000, totalBytes: 10_000_000_000, onCleanupClick: vi.fn() },
    });
    const bar = screen.getByRole('progressbar');
    expect(bar.getAttribute('aria-valuenow')).toBe('30');
    expect(bar.getAttribute('aria-valuemin')).toBe('0');
    expect(bar.getAttribute('aria-valuemax')).toBe('100');
  });

  it('calls onCleanupClick when clicked', async () => {
    const onClick = vi.fn();
    render(QuotaBar, {
      props: { usedBytes: 5_000_000_000, totalBytes: 10_000_000_000, onCleanupClick: onClick },
    });
    await fireEvent.click(screen.getByRole('button'));
    expect(onClick).toHaveBeenCalledOnce();
  });

  it('shows warning color when usage exceeds 85%', () => {
    const { container } = render(QuotaBar, {
      props: { usedBytes: 9_000_000_000, totalBytes: 10_000_000_000, onCleanupClick: vi.fn() },
    });
    expect(container.querySelector('.quota-fill.warning')).toBeTruthy();
  });
});
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/zeblith/work/zeblithic/harmony-client && npx vitest run src/lib/components/__tests__/QuotaBar.test.ts`
Expected: FAIL

**Step 3: Implement QuotaBar.svelte**

Create `src/lib/components/QuotaBar.svelte`:

```svelte
<script lang="ts">
  let {
    usedBytes,
    totalBytes,
    onCleanupClick,
  }: {
    usedBytes: number;
    totalBytes: number;
    onCleanupClick: () => void;
  } = $props();

  let percent = $derived(Math.min(100, Math.round((usedBytes / totalBytes) * 100)));
  let warning = $derived(percent >= 85);

  function formatBytes(bytes: number): string {
    if (bytes >= 1_000_000_000) return (bytes / 1_000_000_000).toFixed(1) + ' GB';
    if (bytes >= 1_000_000) return (bytes / 1_000_000).toFixed(1) + ' MB';
    if (bytes >= 1_000) return (bytes / 1_000).toFixed(1) + ' KB';
    return bytes + ' B';
  }
</script>

<button type="button" class="quota-bar" onclick={onCleanupClick} aria-label="Storage quota — click to manage">
  <div class="quota-track">
    <div
      class="quota-fill"
      class:warning
      style="width: {percent}%"
      role="progressbar"
      aria-valuenow={percent}
      aria-valuemin={0}
      aria-valuemax={100}
      aria-label="Storage usage"
    ></div>
  </div>
  <span class="quota-text">{formatBytes(usedBytes)} of {formatBytes(totalBytes)} used ({percent}%)</span>
</button>

<style>
  .quota-bar {
    display: flex;
    flex-direction: column;
    gap: 4px;
    padding: 8px 12px;
    background: var(--bg-secondary);
    border-top: 1px solid var(--border);
    cursor: pointer;
    border: none;
    width: 100%;
    text-align: left;
  }
  .quota-bar:hover {
    background: var(--bg-tertiary);
  }
  .quota-track {
    height: 6px;
    background: var(--bg-tertiary);
    border-radius: 3px;
    overflow: hidden;
  }
  .quota-fill {
    height: 100%;
    background: var(--accent);
    border-radius: 3px;
    transition: width 0.3s ease;
  }
  .quota-fill.warning {
    background: #d83c3e;
  }
  .quota-text {
    font-size: 0.75rem;
    color: var(--text-muted);
  }
</style>
```

**Step 4: Run tests**

Run: `cd /Users/zeblith/work/zeblithic/harmony-client && npx vitest run src/lib/components/__tests__/QuotaBar.test.ts`
Expected: PASS

**Step 5: Commit**

```bash
cd /Users/zeblith/work/zeblithic/harmony-client
git add src/lib/components/QuotaBar.svelte src/lib/components/__tests__/QuotaBar.test.ts
git commit -m "feat(files): add QuotaBar with progress indicator

Shows used/total bytes with color gradient (accent → red at 85%).
Clickable to open cleanup view. ARIA progressbar semantics."
```

---

### Task 6: FileRow, FileCard, and StalenessIndicator

**Files:**
- Create: `src/lib/components/StalenessIndicator.svelte`
- Create: `src/lib/components/FileRow.svelte`
- Create: `src/lib/components/FileCard.svelte`
- Test: `src/lib/components/__tests__/StalenessIndicator.test.ts`
- Test: `src/lib/components/__tests__/FileRow.test.ts`
- Test: `src/lib/components/__tests__/FileCard.test.ts`

**Context:** These are the building blocks for list and grid views. StalenessIndicator is a small badge (dot) with color mapped from staleness score thresholds. FileRow is a table row for list view. FileCard is a thumbnail card for grid view. Both display staleness badges.

Follow TDD: write tests first, run to confirm failure, implement, run to confirm pass. Tests should cover: rendering content item data, staleness badge color thresholds (none/yellow/orange/red), click handler, accessibility (row should be focusable/clickable, card should have alt text).

The staleness badge thresholds are: 0-0.3 = no badge, 0.3-0.6 = yellow, 0.6-0.85 = orange, 0.85-1.0 = red. Pinned items never show a badge.

Category icons (use Unicode/emoji as placeholder): music=♪, video=▶, text=📄, image=🖼, software=⚙, dataset=📊, bundle=📁.

Sensitivity icons: public=🌐, private=🔒, intimate=🔒 (red tint), confidential=🔒 (gold tint). Replication tier shown as "3/3" replica count.

**Commit message:**

```
feat(files): add FileRow, FileCard, and StalenessIndicator

FileRow for list view, FileCard for grid view, both with staleness
badges mapped from score thresholds. Category and sensitivity icons.
```

---

### Task 7: FileBrowser with FileList, FileGrid, Breadcrumbs, and BrowserToolbar

**Files:**
- Create: `src/lib/components/BrowserToolbar.svelte`
- Create: `src/lib/components/Breadcrumbs.svelte`
- Create: `src/lib/components/FileList.svelte`
- Create: `src/lib/components/FileGrid.svelte`
- Create: `src/lib/components/FileBrowser.svelte`
- Test: `src/lib/components/__tests__/BrowserToolbar.test.ts`
- Test: `src/lib/components/__tests__/Breadcrumbs.test.ts`
- Test: `src/lib/components/__tests__/FileBrowser.test.ts`

**Context:** FileBrowser is the center column main component. It composes BrowserToolbar (view toggle, sort, search, upload button), Breadcrumbs (path navigation), FileList or FileGrid (based on view mode), and QuotaBar (bottom). It takes `FileManagerService` as a prop and manages navigation state (current folder CID, view mode, sort, search filter).

BrowserToolbar has: search input, view toggle (list/grid), sort dropdown (name/date/size/staleness), upload button. Breadcrumbs shows path segments, each clickable. FileList renders FileRow items in a sortable table. FileGrid renders FileCard items in a CSS grid.

The FileBrowser also manages sub-views: when `showCleanup` is true, it renders CleanupView (Task 10) instead of the file list. When `section` is `'published'`, it renders PublishedView (Task 11).

For now, implement the private content browsing. CleanupView and PublishedView are placeholders that get implemented in later tasks.

**Commit message:**

```
feat(files): add FileBrowser with toolbar, breadcrumbs, list and grid views

Center column file browser with search, view toggle (list/grid),
sortable columns, breadcrumb navigation, and QuotaBar. Navigates
folder hierarchy via parentCid. Powered by FileManagerService.
```

---

### Task 8: FileDetailPanel and Sub-Components

**Files:**
- Create: `src/lib/components/FileDetailPanel.svelte`
- Create: `src/lib/components/FileMetadata.svelte`
- Create: `src/lib/components/ReplicationStatus.svelte`
- Create: `src/lib/components/SensitivityBadge.svelte`
- Create: `src/lib/components/FileActions.svelte`
- Test: `src/lib/components/__tests__/FileDetailPanel.test.ts`
- Test: `src/lib/components/__tests__/ReplicationStatus.test.ts`
- Test: `src/lib/components/__tests__/SensitivityBadge.test.ts`

**Context:** Right column panel, shown when a file is selected in the browser. Composes: FileMetadata (name, size, category, dates), StalenessIndicator (score bar, not just dot — a wider horizontal bar here), ReplicationStatus (tier dropdown, "3 of 3" count, breakdown of local/shared/buddies), SensitivityBadge (lock icon colored by level), and FileActions (buttons: Publish, Release, Burn, Archive, Pin/Unpin, Export, Change Tier).

ReplicationStatus shows: tier name, current/target count, visual breakdown: "You (local) · 1 shared · 1 storage buddy". Under-replicated shows a yellow warning. Tier is a dropdown to change.

FileActions fires callbacks: `onPublish`, `onRelease`, `onBurn`, `onArchive`, `onPin`, `onExport`. Each callback is passed the CID. The parent (App.svelte or FileBrowser) handles confirmation gates.

**Commit message:**

```
feat(files): add FileDetailPanel with metadata, replication, actions

Right column detail panel showing file metadata, staleness bar,
replication status with tier dropdown, sensitivity badge, and
action buttons (publish, release, burn, archive, pin, export).
```

---

### Task 9: NavPanel Files Mode — FolderTree, QuickFilters, StorageBuddySummary

**Files:**
- Create: `src/lib/components/FolderTree.svelte`
- Create: `src/lib/components/QuickFilters.svelte`
- Create: `src/lib/components/StorageBuddySummary.svelte`
- Modify: `src/lib/components/NavPanel.svelte`
- Test: `src/lib/components/__tests__/FolderTree.test.ts`
- Test: `src/lib/components/__tests__/QuickFilters.test.ts`
- Test: `src/lib/components/__tests__/StorageBuddySummary.test.ts`

**Context:** When NavPanel is in `'files'` mode, the middle content area (between search and footer) shows FolderTree, QuickFilters, and StorageBuddySummary instead of the channel/DM tree.

FolderTree: renders the bundle hierarchy as a tree. Root items are `parentCid === null` and `isFolder === true`. Children loaded from `FileManagerService.getContents(parentCid)`. Clicking a folder fires `onFolderSelect(cid)`.

QuickFilters: collapsible sections with checkboxes/buttons for filtering by category, status (stale/pinned/licensed/under-replicated), and replication tier. Fires `onFilterChange(filters)`.

StorageBuddySummary: shows buddy count and online count. "Manage" link fires `onManageBuddies()` which opens buddy management in the detail panel.

NavPanel conditionally renders these when `appMode === 'files'` — add an `{#if appMode === 'files'}` block in the nav content area.

**Commit message:**

```
feat(files): add FolderTree, QuickFilters, StorageBuddySummary in NavPanel

NavPanel renders file-specific navigation when in files mode:
folder tree for bundle hierarchy, quick filters by category/status/tier,
and storage buddy count with manage link.
```

---

### Task 10: CleanupView with Recommendations

**Files:**
- Create: `src/lib/components/CleanupView.svelte`
- Create: `src/lib/components/QuotaSummary.svelte`
- Create: `src/lib/components/RecommendationCard.svelte`
- Test: `src/lib/components/__tests__/CleanupView.test.ts`
- Test: `src/lib/components/__tests__/RecommendationCard.test.ts`

**Context:** Sub-view within FileBrowser, replacing the file list when "Cleanup" is active. Accessed by clicking the QuotaBar or a toolbar button.

QuotaSummary: shows total used/total, breakdown by category as colored horizontal bars. Simple bar chart — no charting library needed, just CSS width percentages with category colors.

RecommendationCard: shows one Jain recommendation. Displays: file name, category icon, size, reason badge (stale/duplicate/etc.), staleness score bar, space recoverable. Five action buttons: Burn, Archive, Release, Publish, Pin. Each fires a callback with the CID. The card also has a checkbox for bulk selection.

CleanupView composes: QuotaSummary at top, then a list of RecommendationCards. Has a "Select all" checkbox and a bulk action bar that appears when items are selected, showing: "N items selected — X.X GB recoverable" with Burn All / Archive All / Release All / Publish All buttons. Each bulk action triggers the appropriate confirmation gate (from Task 4).

The "Consider publishing" copy from the design: recommendations include a subtle suggestion line — "This costs you X across N devices. Publish to preserve it forever, release to free quota, or burn if disposable."

**Commit message:**

```
feat(files): add CleanupView with quota summary and recommendation cards

Jain-powered cleanup sub-view with quota breakdown by category,
recommendation cards with burn/archive/release/publish/pin actions,
bulk selection, and graduated confirmation gates.
```

---

### Task 11: PublishedView (Catalog)

**Files:**
- Create: `src/lib/components/PublishedView.svelte`
- Test: `src/lib/components/__tests__/PublishedView.test.ts`

**Context:** Read-only catalog of published content. Accessed via a "Published" toggle in BrowserToolbar or NavPanel. No quota bar (published content is free). Shows: name, category, size, publish date, publish mode (durable 🌐 / ephemeral 🌬). Searchable and filterable by category. No cleanup actions — published content is the network's responsibility.

Simple list view (no grid needed for catalog). Each row shows the publish mode icon prominently — globe for durable, wind/open-hand for ephemeral.

**Commit message:**

```
feat(files): add PublishedView catalog for published content

Read-only catalog showing published content with durable/ephemeral
mode indicators. No quota cost, no cleanup actions. Searchable
and filterable by category.
```

---

### Task 12: ShareList and StorageBuddyList in Detail Panel

**Files:**
- Create: `src/lib/components/ShareList.svelte`
- Create: `src/lib/components/StorageBuddyList.svelte`
- Modify: `src/lib/components/FileDetailPanel.svelte`
- Test: `src/lib/components/__tests__/ShareList.test.ts`
- Test: `src/lib/components/__tests__/StorageBuddyList.test.ts`

**Context:** Two peer lists in the detail panel, visually distinct to enforce the critical share vs. storage buddy distinction.

ShareList: shows peers who have decryption keys. Each row: avatar, name, "can view" label, unlocked lock icon. "Share with..." button opens a contact picker (mock: dropdown of `mockPeers`). "Remove" button per peer.

StorageBuddyList: shows peers holding encrypted blobs. Each row: avatar, name, "storing X MB" label, sealed envelope icon, online/offline indicator. "Add buddy" button, "Remove" button per buddy. Remove warns that replicas need reassignment.

Both lists use the existing `Avatar.svelte` component for peer avatars.

Visual distinction is critical: ShareList has a section header "Shared with (can view)" with an unlocked lock icon. StorageBuddyList has "Stored by (encrypted)" with a sealed envelope icon. Different background tints if needed.

**Commit message:**

```
feat(files): add ShareList and StorageBuddyList in detail panel

Visually distinct peer lists: ShareList (unlocked lock, "can view")
for decryption key holders, StorageBuddyList (sealed envelope,
"encrypted") for blind storage peers. Critical UX distinction.
```

---

### Task 13: Publish and Release Flow Integration

**Files:**
- Create: `src/lib/components/PublishButton.svelte`
- Modify: `src/lib/components/FileActions.svelte`
- Modify: `src/lib/components/FileBrowser.svelte` (or App.svelte — wherever dialog state lives)
- Test: `src/lib/components/__tests__/PublishButton.test.ts`

**Context:** Wire the confirmation gate components (Task 4) into the publish and release actions. This is where the graduated confirmation flows live.

PublishButton: a split button or dropdown with two options — "Publish (permanent)" and "Release (ephemeral)". Clicking either starts the appropriate gate sequence.

**Publish (durable) flow:**
1. Determine confirmation level from sensitivity (Private=double, Intimate=triple, Confidential=triple+OOB)
2. Check user's `confirmationOverrides` for ratcheted-up levels
3. Gate 1: DoubleConfirmDialog with `firstMessage: "Publishing makes this content permanently public. Anyone in the world can access, copy, and redistribute it. This cannot be undone."` and `secondMessage: "You are about to publish [filename]. This is irreversible."`
4. If triple: after gate 2, show TypeToConfirmDialog with `confirmText: filename`
5. If triple+OOB: after type-confirm, show a stub OOB dialog (just a ConfirmDialog saying "In a future version, verify on another device. For now, confirm here.")
6. On final confirm: call `fileManagerService.publish([cid])`

**Release (ephemeral) flow:**
1. Always double-confirm regardless of sensitivity
2. DoubleConfirmDialog with `firstMessage: "This will make your content publicly available. It may persist on the network or fade over time. You can't take it back, but nobody's obligated to keep it either."` and `secondMessage: "You are about to release [filename]."`
3. On confirm: call `fileManagerService.release([cid])`

The dialog state (`showPublishDialog`, `publishTarget`, `publishGate`) should live in FileBrowser or App.svelte — wherever the FileManagerService is accessible. Dialogs render as overlays.

**Commit message:**

```
feat(files): wire publish and release flows with graduated confirmation

PublishButton with durable/ephemeral split. Graduated gates:
double-confirm for Private, triple for Intimate, triple+OOB stub
for Confidential. Release always double-confirm. Leverages Roxy
content model for publish mode (durable vs ephemeral ContentFlags).
```

---

### Task 14: Wire Everything into App.svelte

**Files:**
- Modify: `src/App.svelte`
- Modify: `src/lib/components/Layout.svelte` (if snippet wiring needs updates)

**Context:** Replace the placeholder snippets from Task 3 with real components. Wire `FileManagerService` through to all child components. Manage selection state (which file is selected for the detail panel), dialog state (which confirmation is open), and navigation state (current folder).

Key state to add in App.svelte:

```typescript
const fileManagerService = new FileManagerService();
let fileManagerVersion = $state(0);
let selectedFileCid = $state<string | null>(null);
let currentFolderCid = $state<string | null>(null);
let fileViewMode = $state<FileViewMode>('list');
let showCleanup = $state(false);
let fileSection = $state<ContentSection>('private');
```

The `fileBrowser` snippet renders `<FileBrowser>` with all required props. The `fileDetailPanel` snippet renders `<FileDetailPanel>` when `selectedFileCid` is set, otherwise a placeholder "Select a file" message.

Mutation callbacks (burn, publish, etc.) call the service method, then increment `fileManagerVersion` to trigger reactivity.

**Commit message:**

```
feat(files): wire file manager components into App.svelte

Replace placeholder snippets with real FileBrowser and FileDetailPanel.
Manage selection, navigation, and dialog state at app level. Service
mutations trigger version bumps for Svelte reactivity.
```

---

### Task 15: Final Integration Test & Polish

**Files:**
- Create: `src/lib/components/__tests__/FileBrowser.integration.test.ts`
- Modify: various (CSS tweaks, accessibility fixes found during testing)

**Context:** Write an integration test that renders the full file manager flow: mode switch → browse files → select file → see detail → publish → confirm gates → verify published catalog. This tests the happy path end-to-end with mock data.

Also: run `npx vitest run` for all tests, `npm run build` for build verification, and manually check accessibility (all interactive elements should have ARIA labels, dialogs should trap focus, etc.).

Fix any issues found. Final commit.

**Commit message:**

```
feat(files): integration test and polish

End-to-end test covering mode switch, file browsing, selection,
publish flow with confirmation gates. Accessibility and CSS fixes.
```

---

## Task Summary

| Task | Description | New Files | Tests |
|------|-------------|-----------|-------|
| 1 | TypeScript types | 0 new, 1 modified | 1 |
| 2 | FileManagerService + mock data | 2 new | 1 |
| 3 | Mode integration (Layout, NavPanel, App) | 0 new, 3 modified | 1 |
| 4 | Confirmation gate dialogs | 3 new | 3 |
| 5 | QuotaBar | 1 new | 1 |
| 6 | FileRow, FileCard, StalenessIndicator | 3 new | 3 |
| 7 | FileBrowser, FileList, FileGrid, Breadcrumbs, Toolbar | 5 new | 3 |
| 8 | FileDetailPanel + sub-components | 5 new | 3 |
| 9 | NavPanel files mode (FolderTree, Filters, Buddies) | 3 new, 1 modified | 3 |
| 10 | CleanupView + recommendations | 3 new | 2 |
| 11 | PublishedView catalog | 1 new | 1 |
| 12 | ShareList + StorageBuddyList | 2 new, 1 modified | 2 |
| 13 | Publish/Release flow integration | 1 new, 2 modified | 1 |
| 14 | Wire into App.svelte | 0 new, 2 modified | 0 |
| 15 | Integration test + polish | 0 new | 1 |
| **Total** | | **~29 new files** | **~26 test files** |
