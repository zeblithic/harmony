# Trust Network Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add an 8-bit trust scoring system to the Harmony client with trust editor, badges, and overview table, integrated with the existing media trust service.

**Architecture:** Pure TypeScript data model with bit-manipulation helpers, a mock graph service generating synthetic trust edges between existing mock peers, and three Svelte 5 components (editor, badge, overview table). The existing `TrustService` gains a fallback to the trust graph for media trust resolution.

**Tech Stack:** Svelte 5 (runes), TypeScript, vitest + @testing-library/svelte, jsdom

**Working directory:** `/Users/zeblith/work/zeblithic/harmony-client`

**Design doc:** `/Users/zeblith/work/zeblithic/harmony/docs/plans/2026-03-07-trust-network-design.md`

**Conventions:**
- Svelte 5 runes: `$state()`, `$derived()`, `$props()`, `onclick={handler}` (NOT `on:click`)
- Test: `npx vitest run` | Build: `npm run build`
- Accessibility is a design requirement (WAI-ARIA, keyboard nav, `aria-label`)

---

### Task 1: Trust Score Types and Bit Manipulation Helpers

**Files:**
- Create: `src/lib/trust-score.ts`
- Create: `src/lib/trust-score.test.ts`

**Step 1: Write the failing tests**

```typescript
// src/lib/trust-score.test.ts
import { describe, it, expect } from 'vitest';
import {
  buildScore,
  getIdentity,
  getCompliance,
  getAssociation,
  getEndorsement,
  trustScoreColor,
  trustScoreLabel,
} from './trust-score';

describe('buildScore', () => {
  it('encodes all zeros as 0x00', () => {
    expect(buildScore(0, 0, 0, 0)).toBe(0x00);
  });

  it('encodes all threes as 0xFF', () => {
    expect(buildScore(3, 3, 3, 3)).toBe(0xFF);
  });

  it('encodes identity in bits 0-1', () => {
    expect(buildScore(2, 0, 0, 0)).toBe(0b00000010);
  });

  it('encodes compliance in bits 2-3', () => {
    expect(buildScore(0, 2, 0, 0)).toBe(0b00001000);
  });

  it('encodes association in bits 4-5', () => {
    expect(buildScore(0, 0, 2, 0)).toBe(0b00100000);
  });

  it('encodes endorsement in bits 6-7', () => {
    expect(buildScore(0, 0, 0, 2)).toBe(0b10000000);
  });

  it('encodes a mixed score correctly', () => {
    // identity=1, compliance=2, association=3, endorsement=0
    // 01 | 10 | 11 | 00 = 0b00110101 = 0x35
    expect(buildScore(1, 2, 3, 0)).toBe(0b00110101);
  });

  it('clamps values above 3', () => {
    expect(buildScore(5, 0, 0, 0)).toBe(buildScore(3, 0, 0, 0));
  });

  it('clamps negative values to 0', () => {
    expect(buildScore(-1, 0, 0, 0)).toBe(buildScore(0, 0, 0, 0));
  });
});

describe('dimension extractors', () => {
  it('extracts identity from bits 0-1', () => {
    expect(getIdentity(0b00000011)).toBe(3);
    expect(getIdentity(0b00000010)).toBe(2);
    expect(getIdentity(0b00000000)).toBe(0);
  });

  it('extracts compliance from bits 2-3', () => {
    expect(getCompliance(0b00001100)).toBe(3);
    expect(getCompliance(0b00000100)).toBe(1);
  });

  it('extracts association from bits 4-5', () => {
    expect(getAssociation(0b00110000)).toBe(3);
    expect(getAssociation(0b00010000)).toBe(1);
  });

  it('extracts endorsement from bits 6-7', () => {
    expect(getEndorsement(0b11000000)).toBe(3);
    expect(getEndorsement(0b01000000)).toBe(1);
  });

  it('round-trips through buildScore', () => {
    const score = buildScore(1, 2, 3, 0);
    expect(getIdentity(score)).toBe(1);
    expect(getCompliance(score)).toBe(2);
    expect(getAssociation(score)).toBe(3);
    expect(getEndorsement(score)).toBe(0);
  });

  it('round-trips 0xFF', () => {
    expect(getIdentity(0xFF)).toBe(3);
    expect(getCompliance(0xFF)).toBe(3);
    expect(getAssociation(0xFF)).toBe(3);
    expect(getEndorsement(0xFF)).toBe(3);
  });
});

describe('trustScoreColor', () => {
  it('returns gray for null (unscored)', () => {
    expect(trustScoreColor(null)).toBe('#72767d');
  });

  it('returns red for low trust (0-63)', () => {
    expect(trustScoreColor(0)).toBe('#ed4245');
    expect(trustScoreColor(63)).toBe('#ed4245');
  });

  it('returns amber for cautious (64-127)', () => {
    expect(trustScoreColor(64)).toBe('#faa61a');
    expect(trustScoreColor(127)).toBe('#faa61a');
  });

  it('returns green for trusted (128-191)', () => {
    expect(trustScoreColor(128)).toBe('#43b581');
    expect(trustScoreColor(191)).toBe('#43b581');
  });

  it('returns accent blue for highly trusted (192-255)', () => {
    expect(trustScoreColor(192)).toBe('#5865f2');
    expect(trustScoreColor(255)).toBe('#5865f2');
  });
});

describe('trustScoreLabel', () => {
  it('returns "unscored" for null', () => {
    expect(trustScoreLabel(null)).toBe('unscored');
  });

  it('returns "low trust" for 0-63', () => {
    expect(trustScoreLabel(32)).toBe('low trust');
  });

  it('returns "cautious" for 64-127', () => {
    expect(trustScoreLabel(100)).toBe('cautious');
  });

  it('returns "trusted" for 128-191', () => {
    expect(trustScoreLabel(150)).toBe('trusted');
  });

  it('returns "highly trusted" for 192-255', () => {
    expect(trustScoreLabel(255)).toBe('highly trusted');
  });
});
```

**Step 2: Run tests to verify they fail**

Run: `npx vitest run src/lib/trust-score.test.ts`
Expected: FAIL — module not found

**Step 3: Write the implementation**

```typescript
// src/lib/trust-score.ts

export type TrustScore = number; // 0-255, uint8

export interface TrustEdge {
  /** SHA-256 address of the scorer */
  source: string;
  /** SHA-256 address of the scored peer */
  target: string;
  /** 8-bit trust score */
  score: TrustScore;
  /** When this score was last set (unix ms) */
  timestamp: number;
}

/** Dimension labels for display */
export const DIMENSIONS = ['Identity', 'Compliance', 'Association', 'Endorsement'] as const;
export type TrustDimension = (typeof DIMENSIONS)[number];

function clamp02(v: number): number {
  return Math.max(0, Math.min(3, Math.floor(v)));
}

export function buildScore(
  identity: number,
  compliance: number,
  association: number,
  endorsement: number,
): TrustScore {
  return (
    (clamp02(identity) & 0x3) |
    ((clamp02(compliance) & 0x3) << 2) |
    ((clamp02(association) & 0x3) << 4) |
    ((clamp02(endorsement) & 0x3) << 6)
  );
}

export function getIdentity(score: TrustScore): number {
  return score & 0x3;
}

export function getCompliance(score: TrustScore): number {
  return (score >> 2) & 0x3;
}

export function getAssociation(score: TrustScore): number {
  return (score >> 4) & 0x3;
}

export function getEndorsement(score: TrustScore): number {
  return (score >> 6) & 0x3;
}

export function trustScoreColor(score: TrustScore | null): string {
  if (score === null) return '#72767d';
  if (score < 64) return '#ed4245';
  if (score < 128) return '#faa61a';
  if (score < 192) return '#43b581';
  return '#5865f2';
}

export function trustScoreLabel(score: TrustScore | null): string {
  if (score === null) return 'unscored';
  if (score < 64) return 'low trust';
  if (score < 128) return 'cautious';
  if (score < 192) return 'trusted';
  return 'highly trusted';
}
```

**Step 4: Run tests to verify they pass**

Run: `npx vitest run src/lib/trust-score.test.ts`
Expected: PASS (all tests)

**Step 5: Commit**

```bash
git add src/lib/trust-score.ts src/lib/trust-score.test.ts
git commit -m "feat: trust score types and bit manipulation helpers"
```

---

### Task 2: Mock Trust Graph Service

**Files:**
- Create: `src/lib/trust-graph-service.ts`
- Create: `src/lib/trust-graph-service.test.ts`

**Ref:** `src/lib/trust-service.ts` for the `TrustLevel` type and resolution pattern.

**Step 1: Write the failing tests**

```typescript
// src/lib/trust-graph-service.test.ts
import { describe, it, expect } from 'vitest';
import { MockTrustGraphService } from './trust-graph-service';
import { getIdentity, buildScore } from './trust-score';

describe('MockTrustGraphService', () => {
  it('initializes with edges between mock peers', () => {
    const svc = new MockTrustGraphService('local-addr', ['peer-a', 'peer-b', 'peer-c']);
    expect(svc.getEdges().length).toBeGreaterThan(0);
  });

  it('stores localAddress', () => {
    const svc = new MockTrustGraphService('local-addr', ['peer-a']);
    expect(svc.localAddress).toBe('local-addr');
  });
});

describe('setScore / directScore', () => {
  it('sets and retrieves a direct score', () => {
    const svc = new MockTrustGraphService('local-addr', ['peer-a']);
    svc.setScore('peer-a', 0xFF);
    expect(svc.directScore('local-addr', 'peer-a')).toBe(0xFF);
  });

  it('updates existing score', () => {
    const svc = new MockTrustGraphService('local-addr', ['peer-a']);
    svc.setScore('peer-a', 0xFF);
    svc.setScore('peer-a', 0x00);
    expect(svc.directScore('local-addr', 'peer-a')).toBe(0x00);
  });

  it('returns null for unscored pair', () => {
    const svc = new MockTrustGraphService('local-addr', ['peer-a']);
    svc.clearAllLocalScores();
    expect(svc.directScore('local-addr', 'peer-a')).toBeNull();
  });
});

describe('clearScore', () => {
  it('removes a score', () => {
    const svc = new MockTrustGraphService('local-addr', ['peer-a']);
    svc.setScore('peer-a', 0xFF);
    svc.clearScore('peer-a');
    expect(svc.directScore('local-addr', 'peer-a')).toBeNull();
  });
});

describe('edgesFrom / edgesTo', () => {
  it('edgesFrom returns all edges from a given source', () => {
    const svc = new MockTrustGraphService('local-addr', ['peer-a', 'peer-b']);
    svc.setScore('peer-a', 0xFF);
    svc.setScore('peer-b', 0x80);
    const edges = svc.edgesFrom('local-addr');
    const targets = edges.map((e) => e.target);
    expect(targets).toContain('peer-a');
    expect(targets).toContain('peer-b');
  });

  it('edgesTo returns all edges pointing to a given target', () => {
    const svc = new MockTrustGraphService('local-addr', ['peer-a', 'peer-b']);
    svc.setScore('peer-a', 0xFF);
    // peer-a may also have mock edges pointing to others
    const edges = svc.edgesTo('peer-a');
    expect(edges.length).toBeGreaterThan(0);
    expect(edges.every((e) => e.target === 'peer-a')).toBe(true);
  });
});

describe('resolveMediaTrust', () => {
  it('returns untrusted for identity 00', () => {
    const svc = new MockTrustGraphService('local-addr', ['peer-a']);
    svc.setScore('peer-a', buildScore(0, 3, 3, 3));
    expect(svc.resolveMediaTrust('peer-a')).toBe('untrusted');
  });

  it('returns untrusted for identity 01', () => {
    const svc = new MockTrustGraphService('local-addr', ['peer-a']);
    svc.setScore('peer-a', buildScore(1, 3, 3, 3));
    expect(svc.resolveMediaTrust('peer-a')).toBe('untrusted');
  });

  it('returns preview for identity 10', () => {
    const svc = new MockTrustGraphService('local-addr', ['peer-a']);
    svc.setScore('peer-a', buildScore(2, 0, 0, 0));
    expect(svc.resolveMediaTrust('peer-a')).toBe('preview');
  });

  it('returns trusted for identity 11', () => {
    const svc = new MockTrustGraphService('local-addr', ['peer-a']);
    svc.setScore('peer-a', buildScore(3, 0, 0, 0));
    expect(svc.resolveMediaTrust('peer-a')).toBe('trusted');
  });

  it('returns null for unscored peer', () => {
    const svc = new MockTrustGraphService('local-addr', ['peer-a']);
    svc.clearAllLocalScores();
    expect(svc.resolveMediaTrust('peer-a')).toBeNull();
  });
});
```

**Step 2: Run tests to verify they fail**

Run: `npx vitest run src/lib/trust-graph-service.test.ts`
Expected: FAIL — module not found

**Step 3: Write the implementation**

```typescript
// src/lib/trust-graph-service.ts
import type { TrustLevel } from './types';
import type { TrustScore, TrustEdge } from './trust-score';
import { getIdentity, buildScore } from './trust-score';

function randomInt(min: number, max: number): number {
  return Math.floor(min + Math.random() * (max - min + 1));
}

export class MockTrustGraphService {
  readonly localAddress: string;
  private edges: TrustEdge[] = [];

  constructor(localAddress: string, peerAddresses: string[]) {
    this.localAddress = localAddress;
    this.initMockEdges(peerAddresses);
  }

  private initMockEdges(peers: string[]): void {
    const allAddresses = [this.localAddress, ...peers];

    for (const source of allAddresses) {
      for (const target of allAddresses) {
        if (source === target) continue;
        // ~60% chance of having a score for any given peer
        if (Math.random() < 0.4) continue;
        this.edges.push({
          source,
          target,
          score: randomInt(0, 255) as TrustScore,
          timestamp: Date.now() - randomInt(0, 7 * 24 * 60 * 60 * 1000),
        });
      }
    }
  }

  setScore(target: string, score: TrustScore): void {
    const existing = this.edges.find(
      (e) => e.source === this.localAddress && e.target === target,
    );
    if (existing) {
      existing.score = score;
      existing.timestamp = Date.now();
    } else {
      this.edges.push({
        source: this.localAddress,
        target,
        score,
        timestamp: Date.now(),
      });
    }
  }

  clearScore(target: string): void {
    this.edges = this.edges.filter(
      (e) => !(e.source === this.localAddress && e.target === target),
    );
  }

  /** Remove all local user's scores (useful for testing) */
  clearAllLocalScores(): void {
    this.edges = this.edges.filter((e) => e.source !== this.localAddress);
  }

  getEdges(): TrustEdge[] {
    return this.edges;
  }

  edgesFrom(address: string): TrustEdge[] {
    return this.edges.filter((e) => e.source === address);
  }

  edgesTo(address: string): TrustEdge[] {
    return this.edges.filter((e) => e.target === address);
  }

  directScore(source: string, target: string): TrustScore | null {
    const edge = this.edges.find((e) => e.source === source && e.target === target);
    return edge ? edge.score : null;
  }

  /**
   * Derive media trust level from the identity dimension of the local
   * user's score for a peer. Returns null if the peer is unscored.
   */
  resolveMediaTrust(peerAddress: string): TrustLevel | null {
    const score = this.directScore(this.localAddress, peerAddress);
    if (score === null) return null;
    const identity = getIdentity(score);
    if (identity <= 1) return 'untrusted';
    if (identity === 2) return 'preview';
    return 'trusted';
  }
}
```

**Step 4: Run tests to verify they pass**

Run: `npx vitest run src/lib/trust-graph-service.test.ts`
Expected: PASS (all tests)

**Step 5: Commit**

```bash
git add src/lib/trust-graph-service.ts src/lib/trust-graph-service.test.ts
git commit -m "feat: mock trust graph service with media trust derivation"
```

---

### Task 3: Integrate Trust Graph into TrustService

**Files:**
- Modify: `src/lib/trust-service.ts`
- Modify: `src/lib/trust-service.test.ts`

**Step 1: Write the failing tests**

Add these tests to the existing file:

```typescript
// Append to src/lib/trust-service.test.ts

import { MockTrustGraphService } from './trust-graph-service';
import { buildScore } from './trust-score';

describe('TrustService with trust graph fallback', () => {
  it('falls back to trust graph when no override exists', () => {
    const graph = new MockTrustGraphService('local', ['peer-1']);
    graph.setScore('peer-1', buildScore(3, 0, 0, 0)); // identity=3 -> trusted
    const svc = new TrustService(graph);
    expect(svc.resolve('peer-1')).toBe('trusted');
  });

  it('per-peer override beats trust graph', () => {
    const graph = new MockTrustGraphService('local', ['peer-1']);
    graph.setScore('peer-1', buildScore(3, 0, 0, 0)); // identity=3 -> trusted
    const svc = new TrustService(graph);
    svc.setPeerTrust('peer-1', 'untrusted');
    expect(svc.resolve('peer-1')).toBe('untrusted');
  });

  it('per-community override beats trust graph', () => {
    const graph = new MockTrustGraphService('local', ['peer-1']);
    graph.setScore('peer-1', buildScore(3, 0, 0, 0));
    const svc = new TrustService(graph);
    svc.setCommunityTrust('comm-1', 'untrusted');
    expect(svc.resolve('peer-1', 'comm-1')).toBe('untrusted');
  });

  it('falls through to global when trust graph returns null', () => {
    const graph = new MockTrustGraphService('local', ['peer-1']);
    graph.clearAllLocalScores();
    const svc = new TrustService(graph);
    expect(svc.resolve('peer-1')).toBe('untrusted');
  });

  it('works without a trust graph (backwards compatible)', () => {
    const svc = new TrustService();
    expect(svc.resolve('peer-1')).toBe('untrusted');
  });
});
```

**Step 2: Run tests to verify they fail**

Run: `npx vitest run src/lib/trust-service.test.ts`
Expected: FAIL — TrustService constructor doesn't accept a graph argument

**Step 3: Update TrustService to accept optional trust graph**

```typescript
// src/lib/trust-service.ts — updated
import type { MediaAttachment, TrustLevel, TrustSettings } from './types';
import type { MockTrustGraphService } from './trust-graph-service';

export class TrustService {
  readonly settings: TrustSettings;
  private loadedAttachments = new Set<string>();
  private trustGraph: MockTrustGraphService | null;

  constructor(trustGraph?: MockTrustGraphService) {
    this.settings = {
      global: 'untrusted',
      perPeer: new Map(),
      perCommunity: new Map(),
    };
    this.trustGraph = trustGraph ?? null;
  }

  resolve(peerAddress: string, communityId?: string): TrustLevel {
    const peerLevel = this.settings.perPeer.get(peerAddress);
    if (peerLevel !== undefined) return peerLevel;

    if (communityId) {
      const commLevel = this.settings.perCommunity.get(communityId);
      if (commLevel !== undefined) return commLevel;
    }

    // Trust graph fallback (new step in resolution chain)
    if (this.trustGraph) {
      const graphLevel = this.trustGraph.resolveMediaTrust(peerAddress);
      if (graphLevel !== null) return graphLevel;
    }

    return this.settings.global;
  }

  // ... rest of the methods unchanged ...
}
```

The full file keeps all existing methods (`setGlobalTrust`, `setPeerTrust`, `clearPeerTrust`, `setCommunityTrust`, `clearCommunityTrust`, `isLoaded`, `markLoaded`, `clearLoaded`, `isGated`) exactly as they are. Only the constructor and `resolve` change.

**Step 4: Run tests to verify they pass**

Run: `npx vitest run src/lib/trust-service.test.ts`
Expected: PASS (all tests, old and new)

**Step 5: Commit**

```bash
git add src/lib/trust-service.ts src/lib/trust-service.test.ts
git commit -m "feat: integrate trust graph fallback into TrustService resolution chain"
```

---

### Task 4: TrustBadge Component

**Files:**
- Create: `src/lib/components/TrustBadge.svelte`
- Create: `src/lib/components/__tests__/TrustBadge.test.ts`

**Step 1: Write the failing tests**

```typescript
// src/lib/components/__tests__/TrustBadge.test.ts
import { render, screen } from '@testing-library/svelte';
import { describe, it, expect } from 'vitest';
import TrustBadge from '../TrustBadge.svelte';

describe('TrustBadge', () => {
  it('renders a span element', () => {
    render(TrustBadge, { props: { score: 128 } });
    const badge = screen.getByRole('img');
    expect(badge).toBeTruthy();
  });

  it('shows gray for unscored (null)', () => {
    render(TrustBadge, { props: { score: null } });
    const badge = screen.getByRole('img');
    expect(badge.style.background).toContain('#72767d');
  });

  it('shows red for low trust', () => {
    render(TrustBadge, { props: { score: 30 } });
    const badge = screen.getByRole('img');
    expect(badge.style.background).toContain('#ed4245');
  });

  it('shows amber for cautious', () => {
    render(TrustBadge, { props: { score: 100 } });
    const badge = screen.getByRole('img');
    expect(badge.style.background).toContain('#faa61a');
  });

  it('shows green for trusted', () => {
    render(TrustBadge, { props: { score: 150 } });
    const badge = screen.getByRole('img');
    expect(badge.style.background).toContain('#43b581');
  });

  it('shows accent blue for highly trusted', () => {
    render(TrustBadge, { props: { score: 200 } });
    const badge = screen.getByRole('img');
    expect(badge.style.background).toContain('#5865f2');
  });

  it('has accessible aria-label', () => {
    render(TrustBadge, { props: { score: 200 } });
    const badge = screen.getByRole('img');
    expect(badge.getAttribute('aria-label')).toBe('highly trusted');
  });

  it('aria-label says unscored for null', () => {
    render(TrustBadge, { props: { score: null } });
    const badge = screen.getByRole('img');
    expect(badge.getAttribute('aria-label')).toBe('unscored');
  });
});
```

**Step 2: Run tests to verify they fail**

Run: `npx vitest run src/lib/components/__tests__/TrustBadge.test.ts`
Expected: FAIL — module not found

**Step 3: Write the component**

```svelte
<!-- src/lib/components/TrustBadge.svelte -->
<script lang="ts">
  import { trustScoreColor, trustScoreLabel } from '../trust-score';
  import type { TrustScore } from '../trust-score';

  let {
    score,
  }: {
    score: TrustScore | null;
  } = $props();

  let color = $derived(trustScoreColor(score));
  let label = $derived(trustScoreLabel(score));
</script>

<span
  class="trust-badge"
  role="img"
  aria-label={label}
  style="background: {color}"
></span>

<style>
  .trust-badge {
    display: inline-block;
    width: 8px;
    height: 8px;
    border-radius: 50%;
    flex-shrink: 0;
    vertical-align: middle;
  }
</style>
```

**Step 4: Run tests to verify they pass**

Run: `npx vitest run src/lib/components/__tests__/TrustBadge.test.ts`
Expected: PASS (all tests)

**Step 5: Commit**

```bash
git add src/lib/components/TrustBadge.svelte src/lib/components/__tests__/TrustBadge.test.ts
git commit -m "feat: TrustBadge component with color-coded trust indication"
```

---

### Task 5: TrustEditor Component

**Files:**
- Create: `src/lib/components/TrustEditor.svelte`
- Create: `src/lib/components/__tests__/TrustEditor.test.ts`

**Step 1: Write the failing tests**

```typescript
// src/lib/components/__tests__/TrustEditor.test.ts
import { render, screen, fireEvent } from '@testing-library/svelte';
import { describe, it, expect, vi } from 'vitest';
import TrustEditor from '../TrustEditor.svelte';
import { buildScore, getIdentity, getCompliance, getAssociation, getEndorsement } from '../../trust-score';

describe('TrustEditor', () => {
  it('renders four radiogroups', () => {
    render(TrustEditor, { props: { score: null, onScoreChange: vi.fn() } });
    const groups = screen.getAllByRole('radiogroup');
    expect(groups.length).toBe(4);
  });

  it('labels radiogroups by dimension', () => {
    render(TrustEditor, { props: { score: null, onScoreChange: vi.fn() } });
    expect(screen.getByRole('radiogroup', { name: /identity/i })).toBeTruthy();
    expect(screen.getByRole('radiogroup', { name: /compliance/i })).toBeTruthy();
    expect(screen.getByRole('radiogroup', { name: /association/i })).toBeTruthy();
    expect(screen.getByRole('radiogroup', { name: /endorsement/i })).toBeTruthy();
  });

  it('renders 4 radio buttons per dimension', () => {
    render(TrustEditor, { props: { score: null, onScoreChange: vi.fn() } });
    const radios = screen.getAllByRole('radio');
    expect(radios.length).toBe(16);
  });

  it('reflects current score in selected radios', () => {
    const score = buildScore(1, 2, 3, 0);
    render(TrustEditor, { props: { score, onScoreChange: vi.fn() } });
    const radios = screen.getAllByRole('radio') as HTMLInputElement[];
    // Check that the correct radios are checked
    const checked = radios.filter((r) => r.checked);
    expect(checked.length).toBe(4);
  });

  it('emits onScoreChange when a radio is clicked', async () => {
    const onScoreChange = vi.fn();
    render(TrustEditor, { props: { score: buildScore(0, 0, 0, 0), onScoreChange } });
    // Click the "3" option for Identity
    const identityGroup = screen.getByRole('radiogroup', { name: /identity/i });
    const radios = identityGroup.querySelectorAll('input[type="radio"]');
    await fireEvent.click(radios[3]); // level 3
    expect(onScoreChange).toHaveBeenCalled();
    const newScore = onScoreChange.mock.calls[0][0];
    expect(getIdentity(newScore)).toBe(3);
  });

  it('displays overall score as hex', () => {
    render(TrustEditor, { props: { score: 0xFF, onScoreChange: vi.fn() } });
    expect(screen.getByText(/0xff/i)).toBeTruthy();
  });

  it('renders a clear button', () => {
    render(TrustEditor, { props: { score: 0xFF, onScoreChange: vi.fn(), onClear: vi.fn() } });
    expect(screen.getByRole('button', { name: /clear/i })).toBeTruthy();
  });

  it('emits onClear when clear button is clicked', async () => {
    const onClear = vi.fn();
    render(TrustEditor, { props: { score: 0xFF, onScoreChange: vi.fn(), onClear } });
    await fireEvent.click(screen.getByRole('button', { name: /clear/i }));
    expect(onClear).toHaveBeenCalled();
  });

  it('shows all radios unchecked when score is null', () => {
    render(TrustEditor, { props: { score: null, onScoreChange: vi.fn() } });
    const radios = screen.getAllByRole('radio') as HTMLInputElement[];
    const checked = radios.filter((r) => r.checked);
    expect(checked.length).toBe(0);
  });
});
```

**Step 2: Run tests to verify they fail**

Run: `npx vitest run src/lib/components/__tests__/TrustEditor.test.ts`
Expected: FAIL — module not found

**Step 3: Write the component**

```svelte
<!-- src/lib/components/TrustEditor.svelte -->
<script lang="ts">
  import type { TrustScore } from '../trust-score';
  import {
    DIMENSIONS,
    buildScore,
    getIdentity,
    getCompliance,
    getAssociation,
    getEndorsement,
  } from '../trust-score';

  let {
    score,
    onScoreChange,
    onClear,
  }: {
    score: TrustScore | null;
    onScoreChange: (score: TrustScore) => void;
    onClear?: () => void;
  } = $props();

  const LEVELS = [0, 1, 2, 3];

  const extractors = [getIdentity, getCompliance, getAssociation, getEndorsement];

  let dimensions = $derived(
    score !== null
      ? [getIdentity(score), getCompliance(score), getAssociation(score), getEndorsement(score)]
      : [null, null, null, null],
  );

  function handleChange(dimIndex: number, level: number) {
    const current = dimensions.map((d) => d ?? 0);
    current[dimIndex] = level;
    onScoreChange(buildScore(current[0], current[1], current[2], current[3]));
  }

  let hexDisplay = $derived(
    score !== null ? `0x${score.toString(16).padStart(2, '0')}` : '--',
  );

  let fractionDisplay = $derived(
    score !== null ? `${score}/255` : '',
  );
</script>

<div class="trust-editor">
  <h3 class="editor-heading">Trust</h3>

  {#each DIMENSIONS as dim, i}
    <fieldset
      class="dimension"
      role="radiogroup"
      aria-label={dim}
    >
      <legend class="dimension-label">{dim}</legend>
      <div class="level-options">
        {#each LEVELS as level}
          <label class="level-option">
            <input
              type="radio"
              name="trust-{dim}"
              value={level}
              checked={dimensions[i] === level}
              onclick={() => handleChange(i, level)}
            />
            <span class="level-value">{level}</span>
          </label>
        {/each}
      </div>
    </fieldset>
  {/each}

  <div class="score-footer">
    <span class="overall-score">
      Overall: {hexDisplay}
      {#if fractionDisplay}
        <span class="fraction">({fractionDisplay})</span>
      {/if}
    </span>
    {#if onClear}
      <button
        class="clear-button"
        onclick={() => onClear?.()}
        aria-label="Clear trust score"
      >
        Clear
      </button>
    {/if}
  </div>
</div>

<style>
  .trust-editor {
    padding: 12px;
    font-size: 13px;
    color: var(--text-primary, #dcddde);
  }

  .editor-heading {
    margin: 0 0 12px;
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--text-secondary, #b9bbbe);
  }

  .dimension {
    border: none;
    padding: 0;
    margin: 0 0 8px;
  }

  .dimension-label {
    font-size: 12px;
    font-weight: 500;
    color: var(--text-secondary, #b9bbbe);
    margin-bottom: 4px;
  }

  .level-options {
    display: flex;
    gap: 4px;
  }

  .level-option {
    display: flex;
    align-items: center;
    justify-content: center;
    cursor: pointer;
  }

  .level-option input[type='radio'] {
    position: absolute;
    opacity: 0;
    width: 0;
    height: 0;
  }

  .level-value {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 32px;
    height: 28px;
    border: 1px solid var(--bg-tertiary, #40444b);
    border-radius: 4px;
    background: var(--bg-secondary, #2f3136);
    color: var(--text-secondary, #b9bbbe);
    font-size: 12px;
    font-weight: 500;
    transition: background 0.1s, border-color 0.1s;
  }

  .level-option input[type='radio']:checked + .level-value {
    background: var(--accent, #5865f2);
    border-color: var(--accent, #5865f2);
    color: #ffffff;
  }

  .level-option input[type='radio']:focus-visible + .level-value {
    outline: 2px solid var(--accent, #5865f2);
    outline-offset: 1px;
  }

  .level-value:hover {
    background: var(--bg-tertiary, #40444b);
  }

  .score-footer {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-top: 12px;
    padding-top: 8px;
    border-top: 1px solid var(--bg-tertiary, #40444b);
  }

  .overall-score {
    font-family: monospace;
    font-size: 12px;
    color: var(--text-secondary, #b9bbbe);
  }

  .fraction {
    color: var(--text-muted, #72767d);
  }

  .clear-button {
    padding: 4px 12px;
    border: 1px solid var(--bg-tertiary, #40444b);
    border-radius: 4px;
    background: var(--bg-secondary, #2f3136);
    color: var(--text-secondary, #b9bbbe);
    font: inherit;
    font-size: 12px;
    cursor: pointer;
  }

  .clear-button:hover {
    background: var(--bg-tertiary, #40444b);
  }

  .clear-button:focus-visible {
    outline: 2px solid var(--accent, #5865f2);
    outline-offset: 1px;
  }
</style>
```

**Step 4: Run tests to verify they pass**

Run: `npx vitest run src/lib/components/__tests__/TrustEditor.test.ts`
Expected: PASS (all tests)

**Step 5: Commit**

```bash
git add src/lib/components/TrustEditor.svelte src/lib/components/__tests__/TrustEditor.test.ts
git commit -m "feat: TrustEditor component with 4-dimension radiogroup scoring"
```

---

### Task 6: TrustOverview Component

**Files:**
- Create: `src/lib/components/TrustOverview.svelte`
- Create: `src/lib/components/__tests__/TrustOverview.test.ts`

**Ref:** `src/lib/components/DataTable.svelte` for the sortable table pattern (sortable headers with `aria-sort`, keyed `{#each}`, `role="grid"`).

**Step 1: Write the failing tests**

```typescript
// src/lib/components/__tests__/TrustOverview.test.ts
import { render, screen, fireEvent } from '@testing-library/svelte';
import { describe, it, expect, vi } from 'vitest';
import TrustOverview from '../TrustOverview.svelte';
import { buildScore } from '../../trust-score';
import type { TrustEdge } from '../../trust-score';

function makeEdges(): TrustEdge[] {
  return [
    { source: 'local', target: 'peer-a', score: buildScore(3, 2, 1, 0), timestamp: Date.now() },
    { source: 'local', target: 'peer-b', score: buildScore(0, 1, 2, 3), timestamp: Date.now() },
    { source: 'peer-a', target: 'local', score: buildScore(2, 2, 2, 2), timestamp: Date.now() },
  ];
}

interface PeerInfo {
  address: string;
  displayName: string;
}

const peers: PeerInfo[] = [
  { address: 'peer-a', displayName: 'Alpha' },
  { address: 'peer-b', displayName: 'Bravo' },
];

describe('TrustOverview', () => {
  it('renders a grid element', () => {
    render(TrustOverview, {
      props: { localAddress: 'local', peers, edges: makeEdges() },
    });
    expect(screen.getByRole('grid')).toBeTruthy();
  });

  it('renders a row per peer', () => {
    render(TrustOverview, {
      props: { localAddress: 'local', peers, edges: makeEdges() },
    });
    const rows = screen.getAllByRole('row');
    // header + 2 data rows
    expect(rows.length).toBe(3);
  });

  it('displays peer names', () => {
    render(TrustOverview, {
      props: { localAddress: 'local', peers, edges: makeEdges() },
    });
    expect(screen.getByText('Alpha')).toBeTruthy();
    expect(screen.getByText('Bravo')).toBeTruthy();
  });

  it('shows my score for each peer', () => {
    render(TrustOverview, {
      props: { localAddress: 'local', peers, edges: makeEdges() },
    });
    // Alpha: buildScore(3,2,1,0) = 0x1b = 27... let's check hex display
    // Actually just check that scores appear as hex
    const hexValues = screen.getAllByText(/0x[0-9a-f]{2}/i);
    expect(hexValues.length).toBeGreaterThan(0);
  });

  it('shows their score for me when available', () => {
    render(TrustOverview, {
      props: { localAddress: 'local', peers, edges: makeEdges() },
    });
    // peer-a scored local, peer-b did not
    // Check that at least one em-dash exists (peer-b has no score for us)
    const rows = screen.getAllByRole('row');
    const bravoRow = rows.find((r) => r.textContent?.includes('Bravo'));
    expect(bravoRow?.textContent).toContain('\u2014');
  });

  it('has sortable column headers with aria-sort', () => {
    render(TrustOverview, {
      props: { localAddress: 'local', peers, edges: makeEdges() },
    });
    const headers = screen.getAllByRole('columnheader');
    expect(headers.length).toBeGreaterThan(0);
    const sorted = headers.find(
      (h) => h.getAttribute('aria-sort') === 'ascending',
    );
    expect(sorted).toBeTruthy();
  });

  it('sorts by my score when header is clicked', async () => {
    render(TrustOverview, {
      props: { localAddress: 'local', peers, edges: makeEdges() },
    });
    const btn = screen.getByRole('button', { name: /sort by my score/i });
    await fireEvent.click(btn);
    const rows = screen.getAllByRole('row');
    // After sorting ascending by my score, Bravo (lower) should be first
    expect(rows[1].textContent).toContain('Bravo');
  });
});
```

**Step 2: Run tests to verify they fail**

Run: `npx vitest run src/lib/components/__tests__/TrustOverview.test.ts`
Expected: FAIL — module not found

**Step 3: Write the component**

```svelte
<!-- src/lib/components/TrustOverview.svelte -->
<script lang="ts">
  import type { TrustScore, TrustEdge } from '../trust-score';
  import {
    getIdentity,
    getCompliance,
    getAssociation,
    getEndorsement,
    trustScoreColor,
  } from '../trust-score';
  import TrustBadge from './TrustBadge.svelte';

  interface PeerInfo {
    address: string;
    displayName: string;
  }

  type SortKey = 'name' | 'myScore' | 'theirScore' | 'identity' | 'compliance' | 'association' | 'endorsement';

  let {
    localAddress,
    peers,
    edges,
  }: {
    localAddress: string;
    peers: PeerInfo[];
    edges: TrustEdge[];
  } = $props();

  let sortKey: SortKey = $state('name');
  let sortAsc: boolean = $state(true);

  interface PeerRow {
    address: string;
    displayName: string;
    myScore: TrustScore | null;
    theirScore: TrustScore | null;
  }

  let rows = $derived.by(() => {
    const myEdges = new Map(
      edges.filter((e) => e.source === localAddress).map((e) => [e.target, e.score]),
    );
    const theirEdges = new Map(
      edges.filter((e) => e.target === localAddress).map((e) => [e.source, e.score]),
    );
    return peers.map((p) => ({
      address: p.address,
      displayName: p.displayName,
      myScore: myEdges.get(p.address) ?? null,
      theirScore: theirEdges.get(p.address) ?? null,
    }));
  });

  let sortedRows = $derived.by(() => {
    const copy = [...rows];
    copy.sort((a, b) => {
      let cmp = 0;
      switch (sortKey) {
        case 'name':
          cmp = a.displayName.localeCompare(b.displayName);
          break;
        case 'myScore':
          cmp = (a.myScore ?? -1) - (b.myScore ?? -1);
          break;
        case 'theirScore':
          cmp = (a.theirScore ?? -1) - (b.theirScore ?? -1);
          break;
        case 'identity':
          cmp = (a.myScore !== null ? getIdentity(a.myScore) : -1) -
                (b.myScore !== null ? getIdentity(b.myScore) : -1);
          break;
        case 'compliance':
          cmp = (a.myScore !== null ? getCompliance(a.myScore) : -1) -
                (b.myScore !== null ? getCompliance(b.myScore) : -1);
          break;
        case 'association':
          cmp = (a.myScore !== null ? getAssociation(a.myScore) : -1) -
                (b.myScore !== null ? getAssociation(b.myScore) : -1);
          break;
        case 'endorsement':
          cmp = (a.myScore !== null ? getEndorsement(a.myScore) : -1) -
                (b.myScore !== null ? getEndorsement(b.myScore) : -1);
          break;
      }
      return sortAsc ? cmp : -cmp;
    });
    return copy;
  });

  function toggleSort(key: SortKey) {
    if (sortKey === key) {
      sortAsc = !sortAsc;
    } else {
      sortKey = key;
      sortAsc = true;
    }
  }

  function sortIndicator(key: SortKey): string {
    if (sortKey !== key) return '';
    return sortAsc ? ' \u25B2' : ' \u25BC';
  }

  function formatScore(score: TrustScore | null): string {
    if (score === null) return '\u2014';
    return `0x${score.toString(16).padStart(2, '0')}`;
  }

  function dimValue(score: TrustScore | null, extractor: (s: TrustScore) => number): string {
    if (score === null) return '\u2014';
    return String(extractor(score));
  }

  const columns: { key: SortKey; label: string }[] = [
    { key: 'name', label: 'Name' },
    { key: 'myScore', label: 'My Score' },
    { key: 'theirScore', label: 'Their Score' },
    { key: 'identity', label: 'Id' },
    { key: 'compliance', label: 'Comp' },
    { key: 'association', label: 'Assoc' },
    { key: 'endorsement', label: 'Endorse' },
  ];
</script>

<table class="trust-table" role="grid">
  <thead>
    <tr>
      {#each columns as col}
        <th scope="col" aria-sort={sortKey === col.key ? (sortAsc ? 'ascending' : 'descending') : 'none'}>
          <button
            class="sort-button"
            onclick={() => toggleSort(col.key)}
            aria-label="Sort by {col.label}"
          >
            {col.label}{sortIndicator(col.key)}
          </button>
        </th>
      {/each}
    </tr>
  </thead>
  <tbody>
    {#each sortedRows as row (row.address)}
      <tr class="trust-row">
        <td class="name-cell">
          <TrustBadge score={row.myScore} />
          {row.displayName}
        </td>
        <td style="color: {trustScoreColor(row.myScore)}">{formatScore(row.myScore)}</td>
        <td style="color: {trustScoreColor(row.theirScore)}">{formatScore(row.theirScore)}</td>
        <td>{dimValue(row.myScore, getIdentity)}</td>
        <td>{dimValue(row.myScore, getCompliance)}</td>
        <td>{dimValue(row.myScore, getAssociation)}</td>
        <td>{dimValue(row.myScore, getEndorsement)}</td>
      </tr>
    {/each}
  </tbody>
</table>

<style>
  .trust-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 13px;
    color: var(--text-primary, #dcddde);
    background: var(--bg-primary, #1e1f22);
  }

  thead {
    background: var(--bg-secondary, #2f3136);
    position: sticky;
    top: 0;
    z-index: 1;
  }

  th {
    text-align: left;
    padding: 0;
    border-bottom: 1px solid var(--bg-tertiary, #40444b);
    font-weight: 600;
    color: var(--text-secondary, #b9bbbe);
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }

  .sort-button {
    display: flex;
    align-items: center;
    gap: 4px;
    width: 100%;
    padding: 8px 12px;
    border: none;
    background: none;
    color: inherit;
    font: inherit;
    font-weight: 600;
    cursor: pointer;
    text-align: left;
    white-space: nowrap;
  }

  .sort-button:hover {
    color: var(--text-primary, #dcddde);
    background: var(--bg-tertiary, #40444b);
  }

  .sort-button:focus-visible {
    outline: 2px solid var(--accent, #5865f2);
    outline-offset: -2px;
    border-radius: 2px;
  }

  .trust-row {
    border-bottom: 1px solid var(--bg-secondary, #2f3136);
  }

  .trust-row:hover {
    background: var(--bg-secondary, #2f3136);
  }

  td {
    padding: 6px 12px;
    white-space: nowrap;
    color: var(--text-primary, #dcddde);
  }

  .name-cell {
    font-weight: 500;
    display: flex;
    align-items: center;
    gap: 8px;
  }
</style>
```

**Step 4: Run tests to verify they pass**

Run: `npx vitest run src/lib/components/__tests__/TrustOverview.test.ts`
Expected: PASS (all tests)

**Step 5: Run full test suite and build**

Run: `npx vitest run && npm run build`
Expected: All tests pass, build succeeds

**Step 6: Commit**

```bash
git add src/lib/components/TrustOverview.svelte src/lib/components/__tests__/TrustOverview.test.ts
git commit -m "feat: TrustOverview sortable table with bidirectional scores"
```
