# Profile System Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add SVG identicon avatars, profile data model with CID-ready fields, profile popover, and notification sound override chain to the Harmony client.

**Architecture:** Extend the existing `Peer` type with a `Profile` interface containing status text, avatar CIDs, and sound CIDs. Generate deterministic SVG identicons from peer addresses. Add a profile popover triggered by avatar clicks. Extend `NotificationService` with a `resolveSoundCid()` method implementing the 4-tier override chain.

**Tech Stack:** Svelte 5 (runes), TypeScript, vitest + @testing-library/svelte, jsdom

---

**Important context for the implementer:**

- **Repo:** `/Users/zeblith/work/zeblithic/harmony-client`
- **Branch:** `jake-client-profile-system` (already created, must be checked out before any work)
- **Test command:** `npx vitest run`
- **Build command:** `npm run build`
- **Svelte 5 runes:** `$state()`, `$derived()`, `$derived.by()`, `$props()`, `$effect()`, `onclick={handler}` (NOT `on:click`)
- **Design doc:** `/Users/zeblith/work/zeblithic/harmony/docs/plans/2026-03-05-profile-system-design.md`

### Task 1: Profile Type and Mock Data

**Files:**
- Modify: `src/lib/types.ts`
- Modify: `src/lib/mock-data.ts`

**Step 1: Add Profile type to types.ts**

Add after the existing `Peer` interface (line 5):

```typescript
export interface SoundOverrides {
  quiet?: string;
  standard?: string;
  loud?: string;
}

export interface Profile extends Peer {
  statusText?: string;
  avatarCid?: string;
  avatarMiniCid?: string;
  notificationSounds?: SoundOverrides;
}
```

**Step 2: Update mock-data.ts to use Profile and add a profile store**

Change the `peers` array type from `Peer[]` to `Profile[]` and add status text to some peers. Also export a `profileStore` map:

```typescript
import type { Peer, Profile, Message, NavNode } from './types';

export const peers: Profile[] = [
  { address: 'a1b2c3d4', displayName: 'Alice', statusText: 'Working on transport layer' },
  { address: 'e5f6g7h8', displayName: 'Bob' },
  { address: 'i9j0k1l2', displayName: 'Carol', statusText: 'AFK until tomorrow' },
  { address: 'm3n4o5p6', displayName: 'Dave', statusText: 'Reviewing PRs' },
];
```

After the `navNodes` array, add:

```typescript
export const profileStore = new Map<string, Profile>(
  [...peers, { address: 'q7r8s9t0', displayName: 'Eve', statusText: 'Lurking' } as Profile]
    .map((p) => [p.address, p])
);
```

Remove the `avatarUrl: undefined` from each peer entry since `Profile` doesn't require it (it inherits the optional `avatarUrl` from `Peer`).

**Step 3: Run tests to verify nothing breaks**

Run: `npx vitest run`
Expected: All 81 tests pass (Profile extends Peer, so all existing Peer usage still works)

**Step 4: Run build to verify types**

Run: `npm run build`
Expected: Clean build

**Step 5: Commit**

```bash
git add src/lib/types.ts src/lib/mock-data.ts
git commit -m "feat: add Profile type with CID-ready fields and profile store"
```

---

### Task 2: Identicon Generator

**Files:**
- Create: `src/lib/identicon.ts`
- Create: `src/lib/identicon.test.ts`

**Step 1: Write the failing tests**

Create `src/lib/identicon.test.ts`:

```typescript
import { describe, it, expect } from 'vitest';
import { generateIdenticon } from './identicon';

describe('generateIdenticon', () => {
  it('returns an SVG string', () => {
    const svg = generateIdenticon('a1b2c3d4');
    expect(svg).toContain('<svg');
    expect(svg).toContain('</svg>');
  });

  it('is deterministic — same address produces same output', () => {
    const svg1 = generateIdenticon('a1b2c3d4');
    const svg2 = generateIdenticon('a1b2c3d4');
    expect(svg1).toBe(svg2);
  });

  it('produces different output for different addresses', () => {
    const svg1 = generateIdenticon('a1b2c3d4');
    const svg2 = generateIdenticon('e5f6g7h8');
    expect(svg1).not.toBe(svg2);
  });

  it('generates a symmetric 5x5 grid pattern', () => {
    const svg = generateIdenticon('a1b2c3d4');
    // Extract all rect elements
    const rects = svg.match(/<rect /g);
    // Should have at least some filled cells (not all empty, not all filled)
    expect(rects).toBeTruthy();
    expect(rects!.length).toBeGreaterThan(0);
    expect(rects!.length).toBeLessThanOrEqual(25);
  });

  it('produces horizontally symmetric patterns', () => {
    const svg = generateIdenticon('test-address');
    // Extract rect x positions and y positions
    const rectMatches = [...svg.matchAll(/x="(\d+)" y="(\d+)"/g)];
    const cells = rectMatches.map((m) => ({ x: Number(m[1]), y: Number(m[2]) }));

    // For each filled cell at column c, there should be a matching cell at column (4-c)
    // Grid cell size is gridSize = size/5, so x = col * gridSize
    // We check symmetry: if (x, y) exists, then (maxX - x, y) should also exist
    if (cells.length > 0) {
      const maxX = Math.max(...cells.map((c) => c.x));
      for (const cell of cells) {
        const mirrorX = maxX - cell.x;
        const hasMirror = cells.some((c) => c.x === mirrorX && c.y === cell.y);
        expect(hasMirror).toBe(true);
      }
    }
  });

  it('respects custom size parameter', () => {
    const svg = generateIdenticon('a1b2c3d4', 100);
    expect(svg).toContain('width="100"');
    expect(svg).toContain('height="100"');
  });

  it('uses default size of 64', () => {
    const svg = generateIdenticon('a1b2c3d4');
    expect(svg).toContain('width="64"');
    expect(svg).toContain('height="64"');
  });
});
```

**Step 2: Run tests to verify they fail**

Run: `npx vitest run src/lib/identicon.test.ts`
Expected: FAIL — module not found

**Step 3: Implement identicon.ts**

Create `src/lib/identicon.ts`:

```typescript
/**
 * Generate a deterministic SVG identicon from a peer address string.
 * Produces a 5x5 symmetric grid pattern with a foreground color derived from the address hash.
 */
export function generateIdenticon(address: string, size = 64): string {
  const hash = hashAddress(address);
  const hue = (hash[0] * 7 + hash[1] * 13) % 360;
  const fg = `hsl(${hue}, 55%, 50%)`;
  const cellSize = size / 5;

  // Build 5x5 grid — columns 0-1 mirror columns 4-3, column 2 is center
  const grid: boolean[][] = [];
  for (let row = 0; row < 5; row++) {
    grid[row] = [];
    for (let col = 0; col < 3; col++) {
      const byteIdx = (row * 3 + col) % hash.length;
      grid[row][col] = hash[byteIdx] % 2 === 0;
      // Mirror: col 0 → col 4, col 1 → col 3
      if (col < 2) {
        grid[row][4 - col] = grid[row][col];
      }
    }
  }

  // Build SVG rects for filled cells
  const rects: string[] = [];
  for (let row = 0; row < 5; row++) {
    for (let col = 0; col < 5; col++) {
      if (grid[row][col]) {
        const x = Math.round(col * cellSize);
        const y = Math.round(row * cellSize);
        const w = Math.round((col + 1) * cellSize) - x;
        const h = Math.round((row + 1) * cellSize) - y;
        rects.push(`<rect x="${x}" y="${y}" width="${w}" height="${h}" fill="${fg}"/>`);
      }
    }
  }

  return `<svg xmlns="http://www.w3.org/2000/svg" width="${size}" height="${size}" viewBox="0 0 ${size} ${size}">${rects.join('')}</svg>`;
}

/** Simple hash function — returns an array of pseudo-random bytes from an address string. */
function hashAddress(address: string): number[] {
  const bytes: number[] = [];
  // Use a simple but effective mixing function
  let h = 0x811c9dc5; // FNV offset basis
  for (let i = 0; i < address.length; i++) {
    h ^= address.charCodeAt(i);
    h = Math.imul(h, 0x01000193); // FNV prime
    h = h >>> 0; // Convert to unsigned 32-bit
  }
  // Generate 15 bytes (enough for 5x3 grid decisions + color)
  for (let i = 0; i < 15; i++) {
    h ^= (h >>> 13);
    h = Math.imul(h, 0x5bd1e995);
    h = h >>> 0;
    bytes.push(h & 0xff);
  }
  return bytes;
}
```

**Step 4: Run tests to verify they pass**

Run: `npx vitest run src/lib/identicon.test.ts`
Expected: All 7 tests pass

**Step 5: Run all tests**

Run: `npx vitest run`
Expected: All tests pass

**Step 6: Commit**

```bash
git add src/lib/identicon.ts src/lib/identicon.test.ts
git commit -m "feat: add deterministic SVG identicon generator"
```

---

### Task 3: Upgrade Avatar Component

**Files:**
- Modify: `src/lib/components/Avatar.svelte`
- Modify: `src/lib/components/__tests__/TextMessage.test.ts` (verify Avatar still renders)

**Step 1: Rewrite Avatar.svelte to use SVG identicon**

Replace the entire contents of `src/lib/components/Avatar.svelte`:

```svelte
<script lang="ts">
  import { generateIdenticon } from '../identicon';

  let { address, size = 24, displayName = '', avatarUrl, onclick }: {
    address: string;
    size?: number;
    displayName?: string;
    avatarUrl?: string;
    onclick?: (e: MouseEvent) => void;
  } = $props();

  let identiconSvg = $derived(generateIdenticon(address, size));
</script>

{#if avatarUrl}
  <button
    class="avatar"
    style="width: {size}px; height: {size}px;"
    title={displayName}
    onclick={(e) => onclick?.(e)}
    type="button"
  >
    <img src={avatarUrl} alt={displayName} width={size} height={size} />
  </button>
{:else}
  <button
    class="avatar"
    style="width: {size}px; height: {size}px;"
    title={displayName}
    onclick={(e) => onclick?.(e)}
    type="button"
  >
    {@html identiconSvg}
  </button>
{/if}

<style>
  .avatar {
    border-radius: 50%;
    overflow: hidden;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
    user-select: none;
    border: none;
    background: none;
    padding: 0;
    cursor: pointer;
  }

  .avatar :global(svg) {
    display: block;
    border-radius: 50%;
  }

  .avatar img {
    display: block;
    border-radius: 50%;
    object-fit: cover;
  }
</style>
```

Key changes:
- Replaced single-letter + hue background with SVG identicon via `{@html}`
- Added `avatarUrl` prop for future image support
- Added `onclick` prop for popover trigger
- Wrapped in `<button>` for accessibility (clickable)
- Circular clip via `border-radius: 50%` + `overflow: hidden`

**Step 2: Run existing tests to verify Avatar still works in TextMessage**

Run: `npx vitest run`
Expected: All tests pass — TextMessage still renders with the Avatar, just the visual output changed

**Step 3: Run build**

Run: `npm run build`
Expected: Clean build

**Step 4: Commit**

```bash
git add src/lib/components/Avatar.svelte
git commit -m "feat: upgrade Avatar to SVG identicon with click handler"
```

---

### Task 4: Profile Popover Component

**Files:**
- Create: `src/lib/components/ProfilePopover.svelte`
- Create: `src/lib/components/__tests__/ProfilePopover.test.ts`

**Step 1: Write the failing tests**

Create `src/lib/components/__tests__/ProfilePopover.test.ts`:

```typescript
import { render, screen, fireEvent } from '@testing-library/svelte';
import { describe, it, expect, vi } from 'vitest';
import ProfilePopover from '../ProfilePopover.svelte';
import type { Profile } from '../../types';

const mockProfile: Profile = {
  address: 'a1b2c3d4',
  displayName: 'Alice',
  statusText: 'Working on transport layer',
};

describe('ProfilePopover', () => {
  it('renders display name and status text', () => {
    render(ProfilePopover, {
      props: {
        profile: mockProfile,
        x: 100,
        y: 100,
        onClose: vi.fn(),
      },
    });
    expect(screen.getByText('Alice')).toBeTruthy();
    expect(screen.getByText('Working on transport layer')).toBeTruthy();
  });

  it('renders truncated peer address', () => {
    render(ProfilePopover, {
      props: {
        profile: mockProfile,
        x: 100,
        y: 100,
        onClose: vi.fn(),
      },
    });
    expect(screen.getByText('a1b2c3d4')).toBeTruthy();
  });

  it('renders sound slot labels', () => {
    render(ProfilePopover, {
      props: {
        profile: mockProfile,
        x: 100,
        y: 100,
        onClose: vi.fn(),
      },
    });
    expect(screen.getByText('Quiet')).toBeTruthy();
    expect(screen.getByText('Standard')).toBeTruthy();
    expect(screen.getByText('Loud')).toBeTruthy();
  });

  it('shows "System default" when no custom sounds set', () => {
    render(ProfilePopover, {
      props: {
        profile: mockProfile,
        x: 100,
        y: 100,
        onClose: vi.fn(),
      },
    });
    const defaults = screen.getAllByText('System default');
    expect(defaults.length).toBe(3);
  });

  it('calls onClose when Escape is pressed', async () => {
    const onClose = vi.fn();
    render(ProfilePopover, {
      props: {
        profile: mockProfile,
        x: 100,
        y: 100,
        onClose,
      },
    });
    await fireEvent.keyDown(window, { key: 'Escape' });
    expect(onClose).toHaveBeenCalled();
  });

  it('handles profile without status text', () => {
    const noStatus: Profile = { address: 'xyz789', displayName: 'Bob' };
    render(ProfilePopover, {
      props: {
        profile: noStatus,
        x: 100,
        y: 100,
        onClose: vi.fn(),
      },
    });
    expect(screen.getByText('Bob')).toBeTruthy();
    // No status text rendered
    expect(screen.queryByText('Working on transport layer')).toBeNull();
  });
});
```

**Step 2: Run tests to verify they fail**

Run: `npx vitest run src/lib/components/__tests__/ProfilePopover.test.ts`
Expected: FAIL — module not found

**Step 3: Implement ProfilePopover.svelte**

Create `src/lib/components/ProfilePopover.svelte`:

```svelte
<script lang="ts">
  import type { Profile } from '../types';
  import Avatar from './Avatar.svelte';

  let { profile, x, y, onClose }: {
    profile: Profile;
    x: number;
    y: number;
    onClose: () => void;
  } = $props();

  const SOUND_LABELS = { quiet: 'Quiet', standard: 'Standard', loud: 'Loud' } as const;

  $effect(() => {
    function onKeyDown(e: KeyboardEvent) {
      if (e.key === 'Escape') onClose();
    }
    function onClickOutside(e: MouseEvent) {
      const target = e.target as HTMLElement;
      if (!target.closest('.profile-popover')) {
        onClose();
      }
    }
    window.addEventListener('keydown', onKeyDown);
    // Delay click listener to avoid the opening click from immediately closing
    const timer = setTimeout(() => {
      window.addEventListener('click', onClickOutside);
    }, 0);
    return () => {
      window.removeEventListener('keydown', onKeyDown);
      window.removeEventListener('click', onClickOutside);
      clearTimeout(timer);
    };
  });
</script>

<div class="profile-popover" style="left: {x}px; top: {y}px;">
  <div class="popover-header">
    <Avatar address={profile.address} displayName={profile.displayName} size={64} />
    <div class="popover-identity">
      <div class="popover-name">{profile.displayName}</div>
      {#if profile.statusText}
        <div class="popover-status">{profile.statusText}</div>
      {/if}
      <div class="popover-address">{profile.address}</div>
    </div>
  </div>
  <div class="popover-sounds">
    <div class="sounds-label">Notification sounds</div>
    {#each (['quiet', 'standard', 'loud'] as const) as slot}
      <div class="sound-row">
        <span class="sound-slot">{SOUND_LABELS[slot]}</span>
        <span class="sound-value">
          {profile.notificationSounds?.[slot] ?? 'System default'}
        </span>
      </div>
    {/each}
  </div>
</div>

<style>
  .profile-popover {
    position: fixed;
    z-index: 100;
    background: var(--bg-tertiary);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 16px;
    min-width: 240px;
    max-width: 300px;
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.3);
  }

  .popover-header {
    display: flex;
    gap: 12px;
    align-items: flex-start;
    margin-bottom: 12px;
  }

  .popover-identity {
    flex: 1;
    min-width: 0;
  }

  .popover-name {
    font-size: 16px;
    font-weight: 700;
    color: var(--text-primary);
  }

  .popover-status {
    font-size: 12px;
    color: var(--text-muted);
    margin-top: 2px;
  }

  .popover-address {
    font-size: 11px;
    color: var(--text-muted);
    font-family: monospace;
    margin-top: 4px;
    overflow: hidden;
    text-overflow: ellipsis;
  }

  .popover-sounds {
    border-top: 1px solid var(--border);
    padding-top: 10px;
  }

  .sounds-label {
    font-size: 11px;
    font-weight: 600;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 6px;
  }

  .sound-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 3px 0;
  }

  .sound-slot {
    font-size: 12px;
    color: var(--text-secondary);
  }

  .sound-value {
    font-size: 12px;
    color: var(--text-muted);
  }
</style>
```

**Step 4: Run tests to verify they pass**

Run: `npx vitest run src/lib/components/__tests__/ProfilePopover.test.ts`
Expected: All 6 tests pass

**Step 5: Run all tests**

Run: `npx vitest run`
Expected: All tests pass

**Step 6: Commit**

```bash
git add src/lib/components/ProfilePopover.svelte src/lib/components/__tests__/ProfilePopover.test.ts
git commit -m "feat: add ProfilePopover component with sound slots display"
```

---

### Task 5: Sound Override Chain in NotificationService

**Files:**
- Modify: `src/lib/types.ts`
- Modify: `src/lib/notification-service.ts`
- Modify: `src/lib/notification-service.test.ts`

**Step 1: Write the failing tests**

Add to the bottom of `src/lib/notification-service.test.ts`, inside a new `describe` block:

```typescript
import type { Profile } from './types';

describe('NotificationService.resolveSoundCid', () => {
  const senderProfile: Profile = {
    address: 'sender-1',
    displayName: 'Alice',
    notificationSounds: {
      quiet: 'cid-whisper',
      standard: 'cid-chime',
      loud: 'cid-siren',
    },
  };

  const noSoundsProfile: Profile = {
    address: 'sender-2',
    displayName: 'Bob',
  };

  it('returns undefined when no sounds configured anywhere', () => {
    const svc = new NotificationService();
    expect(svc.resolveSoundCid('standard', 'peer-1', noSoundsProfile)).toBeUndefined();
  });

  it('returns sender profile sound as fallback (tier 3)', () => {
    const svc = new NotificationService();
    expect(svc.resolveSoundCid('standard', 'peer-1', senderProfile)).toBe('cid-chime');
    expect(svc.resolveSoundCid('quiet', 'peer-1', senderProfile)).toBe('cid-whisper');
    expect(svc.resolveSoundCid('loud', 'peer-1', senderProfile)).toBe('cid-siren');
  });

  it('per-community sound override beats sender profile (tier 2)', () => {
    const svc = new NotificationService();
    svc.setCommunitySoundOverrides('comm-1', { standard: 'cid-community-chime' });
    expect(svc.resolveSoundCid('standard', 'peer-1', senderProfile, 'comm-1')).toBe('cid-community-chime');
    // Other slots still fall through to sender
    expect(svc.resolveSoundCid('loud', 'peer-1', senderProfile, 'comm-1')).toBe('cid-siren');
  });

  it('per-peer sound override beats per-community (tier 1)', () => {
    const svc = new NotificationService();
    svc.setCommunitySoundOverrides('comm-1', { standard: 'cid-community-chime' });
    svc.setPeerSoundOverrides('peer-1', { standard: 'cid-peer-chime' });
    expect(svc.resolveSoundCid('standard', 'peer-1', senderProfile, 'comm-1')).toBe('cid-peer-chime');
  });

  it('clears peer sound overrides', () => {
    const svc = new NotificationService();
    svc.setPeerSoundOverrides('peer-1', { standard: 'cid-custom' });
    svc.clearPeerSoundOverrides('peer-1');
    expect(svc.resolveSoundCid('standard', 'peer-1', senderProfile)).toBe('cid-chime');
  });

  it('clears community sound overrides', () => {
    const svc = new NotificationService();
    svc.setCommunitySoundOverrides('comm-1', { standard: 'cid-custom' });
    svc.clearCommunitySoundOverrides('comm-1');
    expect(svc.resolveSoundCid('standard', 'peer-1', senderProfile, 'comm-1')).toBe('cid-chime');
  });
});
```

**Step 2: Run tests to verify they fail**

Run: `npx vitest run src/lib/notification-service.test.ts`
Expected: FAIL — `resolveSoundCid` is not a function

**Step 3: Implement the sound override chain**

Add to `src/lib/notification-service.ts`:

1. Import `Profile` and `SoundOverrides` from types:
```typescript
import type {
  MessagePriority,
  NotificationAction,
  NotificationPolicy,
  NotificationSettings,
  Profile,
  SoundOverrides,
} from './types';
```

2. Add sound override maps to `NotificationSettings` in `src/lib/types.ts`:
```typescript
export interface NotificationSettings {
  global: NotificationPolicy;
  perCommunity: Map<string, Partial<NotificationPolicy>>;
  perPeer: Map<string, Partial<NotificationPolicy>>;
  perPeerSounds: Map<string, SoundOverrides>;
  perCommunitySounds: Map<string, SoundOverrides>;
}
```

3. Initialize the new maps in the constructor:
```typescript
constructor() {
  this.settings = {
    global: { ...DEFAULT_POLICY },
    perCommunity: new Map(),
    perPeer: new Map(),
    perPeerSounds: new Map(),
    perCommunitySounds: new Map(),
  };
}
```

4. Add methods to `NotificationService`:
```typescript
resolveSoundCid(
  priority: MessagePriority,
  peerAddress: string,
  senderProfile: Profile,
  communityId?: string,
): string | undefined {
  // Tier 1: Per-peer sound override
  const peerSounds = this.settings.perPeerSounds.get(peerAddress);
  if (peerSounds?.[priority]) return peerSounds[priority];

  // Tier 2: Per-community sound override
  if (communityId) {
    const commSounds = this.settings.perCommunitySounds.get(communityId);
    if (commSounds?.[priority]) return commSounds[priority];
  }

  // Tier 3: Sender's profile default
  if (senderProfile.notificationSounds?.[priority]) {
    return senderProfile.notificationSounds[priority];
  }

  // Tier 4: Client global default (undefined = system default)
  return undefined;
}

setPeerSoundOverrides(peerAddress: string, sounds: SoundOverrides): void {
  this.settings.perPeerSounds.set(peerAddress, { ...sounds });
}

clearPeerSoundOverrides(peerAddress: string): void {
  this.settings.perPeerSounds.delete(peerAddress);
}

setCommunitySoundOverrides(communityId: string, sounds: SoundOverrides): void {
  this.settings.perCommunitySounds.set(communityId, { ...sounds });
}

clearCommunitySoundOverrides(communityId: string): void {
  this.settings.perCommunitySounds.delete(communityId);
}
```

**Step 4: Run tests to verify they pass**

Run: `npx vitest run src/lib/notification-service.test.ts`
Expected: All 15 tests pass (9 existing + 6 new)

**Step 5: Run all tests**

Run: `npx vitest run`
Expected: All tests pass

**Step 6: Commit**

```bash
git add src/lib/types.ts src/lib/notification-service.ts src/lib/notification-service.test.ts
git commit -m "feat: add 4-tier sound override chain to NotificationService"
```

---

### Task 6: Wire Popover into App

**Files:**
- Modify: `src/App.svelte`

**Step 1: Add popover state and profile lookup to App.svelte**

Add imports at the top:
```typescript
import ProfilePopover from './lib/components/ProfilePopover.svelte';
import { profileStore } from './lib/mock-data';
import type { Profile } from './lib/types';
```

Add state variables after the existing state declarations:
```typescript
let popoverProfile = $state<Profile | null>(null);
let popoverX = $state(0);
let popoverY = $state(0);

function handleAvatarClick(address: string, event: MouseEvent) {
  const profile = profileStore.get(address);
  if (!profile) return;
  const rect = (event.currentTarget as HTMLElement).getBoundingClientRect();
  popoverX = rect.right + 8;
  popoverY = rect.top;
  popoverProfile = profile;
}

function closePopover() {
  popoverProfile = null;
}
```

**Step 2: Pass the click handler down through components**

Update the Layout snippets to pass `onAvatarClick`:

```svelte
{#snippet textFeed()}
  <TextFeed messages={allMessages} {collapsed} onMediaClick={scrollToMedia} onSend={handleSend} onAvatarClick={handleAvatarClick} />
{/snippet}
{#snippet mediaFeed()}
  <MediaFeed messages={allMessages} onLinkBack={scrollToMessage} onAvatarClick={handleAvatarClick} />
{/snippet}
```

Add the popover after the `</Layout>` closing tag:

```svelte
{#if popoverProfile}
  <ProfilePopover
    profile={popoverProfile}
    x={popoverX}
    y={popoverY}
    onClose={closePopover}
  />
{/if}
```

**Step 3: Thread onAvatarClick through TextFeed → TextMessage**

In `src/lib/components/TextFeed.svelte`, add `onAvatarClick` prop:

```typescript
let { messages, collapsed = false, onMediaClick, onSend, onAvatarClick }: {
  messages: Message[];
  collapsed?: boolean;
  onMediaClick?: (mediaId: string) => void;
  onSend?: (text: string, priority: MessagePriority) => void;
  onAvatarClick?: (address: string, event: MouseEvent) => void;
} = $props();
```

Pass it to TextMessage:
```svelte
<TextMessage message={item.message} {collapsed} {onMediaClick} {onAvatarClick} />
```

And to QuietMessageGroup:
```svelte
<QuietMessageGroup messages={item.messages} {collapsed} {onMediaClick} {onAvatarClick} />
```

In `src/lib/components/QuietMessageGroup.svelte`, add `onAvatarClick` prop and forward it:

```typescript
let { messages, collapsed = false, onMediaClick, onAvatarClick }: {
  messages: Message[];
  collapsed?: boolean;
  onMediaClick?: (mediaId: string) => void;
  onAvatarClick?: (address: string, event: MouseEvent) => void;
} = $props();
```

Forward to TextMessage:
```svelte
<TextMessage {message} {collapsed} {onMediaClick} {onAvatarClick} />
```

In `src/lib/components/TextMessage.svelte`, add `onAvatarClick` prop and wire it to Avatar:

```typescript
let { message, collapsed = false, onMediaClick, onAvatarClick }: {
  message: Message;
  collapsed?: boolean;
  onMediaClick?: (mediaId: string) => void;
  onAvatarClick?: (address: string, event: MouseEvent) => void;
} = $props();
```

Update the Avatar usage:
```svelte
<Avatar
  address={message.sender.address}
  displayName={message.sender.displayName}
  size={24}
  onclick={(e) => onAvatarClick?.(message.sender.address, e)}
/>
```

**Step 4: Thread onAvatarClick through MediaFeed → MediaCard**

In `src/lib/components/MediaFeed.svelte`, add `onAvatarClick` prop:

```typescript
let { messages, onLinkBack, onAvatarClick }: {
  messages: Message[];
  onLinkBack?: (messageId: string) => void;
  onAvatarClick?: (address: string, event: MouseEvent) => void;
} = $props();
```

Pass it to MediaCard:
```svelte
<MediaCard {message} {attachment} {onLinkBack} {onAvatarClick} />
```

In `src/lib/components/MediaCard.svelte`, add `onAvatarClick` prop and wire it to Avatar:

```typescript
let { message, attachment, onLinkBack, onAvatarClick }: {
  message: Message;
  attachment: MediaAttachment;
  onLinkBack?: (messageId: string) => void;
  onAvatarClick?: (address: string, event: MouseEvent) => void;
} = $props();
```

Update the Avatar in the card header:
```svelte
<Avatar
  address={message.sender.address}
  displayName={message.sender.displayName}
  size={20}
  onclick={(e) => { e.stopPropagation(); onAvatarClick?.(message.sender.address, e); }}
/>
```

Note the `e.stopPropagation()` — without it, clicking the avatar would also trigger the card header's `onLinkBack`.

**Step 5: Run tests and build**

Run: `npx vitest run`
Expected: All tests pass

Run: `npm run build`
Expected: Clean build

**Step 6: Commit**

```bash
git add src/App.svelte src/lib/components/TextFeed.svelte src/lib/components/TextMessage.svelte src/lib/components/QuietMessageGroup.svelte src/lib/components/MediaFeed.svelte src/lib/components/MediaCard.svelte
git commit -m "feat: wire profile popover through avatar clicks across all feeds"
```

---

### Task 7: Nav Panel Avatars and Status Text

**Files:**
- Modify: `src/lib/components/NavNodeRow.svelte`

**Step 1: Add Avatar to DM/group-chat nodes**

Import Avatar at the top of `NavNodeRow.svelte`:
```typescript
import Avatar from './Avatar.svelte';
```

In the text/both display mode section (inside `{:else}` after `<!-- Text or both mode -->`), after the type icon and before the node name, add an avatar for DM/group-chat nodes:

```svelte
{#if (node.type === 'dm' || node.type === 'group-chat') && node.peer}
  <Avatar address={node.peer.address} displayName={node.peer.displayName} size={20} />
{/if}
```

Remove the old `dm-avatar` img tags (lines referencing `node.peer?.avatarUrl`) since avatars are now handled by the Avatar component.

For icon display mode, replace the letter initial with an Avatar component for DM/group-chat nodes:

```svelte
{#if displayMode === 'icon'}
  {#if (node.type === 'dm' || node.type === 'group-chat') && node.peer}
    <Avatar address={node.peer.address} displayName={node.peer.displayName} size={32} />
  {:else}
    <span class="icon-cell">
      {node.name.charAt(0).toUpperCase()}
    </span>
  {/if}
{:else}
  <!-- Text or both mode -->
  <span class="type-icon">{typeIcon(node)}</span>
  {#if (node.type === 'dm' || node.type === 'group-chat') && node.peer}
    <Avatar address={node.peer.address} displayName={node.peer.displayName} size={20} />
  {/if}
  <span class="node-name">{node.name}</span>
{/if}
```

**Step 2: Add status text subtitle for DM nodes**

This requires access to the profile store. Add an `onAvatarClick` prop and a `statusText` prop:

```typescript
import { profileStore } from '../mock-data';

// Inside the script, derive status text from the profile store:
let statusText = $derived(
  node.peer ? profileStore.get(node.peer.address)?.statusText : undefined
);
```

After the node name in text/both mode, add a status subtitle:

```svelte
<span class="node-name">
  {node.name}
  {#if statusText}
    <span class="status-text">{statusText}</span>
  {/if}
</span>
```

Add CSS:
```css
.status-text {
  display: block;
  font-size: 11px;
  font-weight: 400;
  color: var(--text-muted);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
```

**Step 3: Run tests and build**

Run: `npx vitest run`
Expected: All tests pass

Run: `npm run build`
Expected: Clean build

**Step 4: Commit**

```bash
git add src/lib/components/NavNodeRow.svelte
git commit -m "feat: add identicon avatars and status text to nav panel DM entries"
```

---

### Task 8: NotificationSettingsPanel Sound Slots

**Files:**
- Modify: `src/lib/components/NotificationSettingsPanel.svelte`

**Step 1: Add sound slot display to peer and community override sections**

In `NotificationSettingsPanel.svelte`, add a placeholder sound slots section after the notification policy dropdowns in each peer/community override section.

For the peers tab, after the `policy-rows` div inside each peer's override section:

```svelte
<div class="sound-slots">
  <div class="slots-label">Custom sounds</div>
  {#each (['quiet', 'standard', 'loud'] as const) as slot}
    <div class="sound-slot-row">
      <span class="slot-name">{PRIORITY_LABELS[slot]}</span>
      <span class="slot-value">System default</span>
    </div>
  {/each}
</div>
```

Add the same for the communities tab.

Add CSS:
```css
.sound-slots {
  margin-top: 8px;
  padding-top: 8px;
  border-top: 1px solid var(--border);
}

.slots-label {
  font-size: 11px;
  font-weight: 600;
  color: var(--text-muted);
  text-transform: uppercase;
  letter-spacing: 0.5px;
  margin-bottom: 4px;
}

.sound-slot-row {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 2px 0;
}

.slot-name {
  font-size: 12px;
  color: var(--text-secondary);
}

.slot-value {
  font-size: 12px;
  color: var(--text-muted);
  font-style: italic;
}
```

**Step 2: Run tests and build**

Run: `npx vitest run`
Expected: All tests pass

Run: `npm run build`
Expected: Clean build

**Step 3: Commit**

```bash
git add src/lib/components/NotificationSettingsPanel.svelte
git commit -m "feat: add sound slot placeholders to notification settings panel"
```

---

### Task 9: Visual Verification

**Files:** None modified — verification only.

**Step 1: Run all tests**

Run: `npx vitest run`
Expected: All tests pass

**Step 2: Run build**

Run: `npm run build`
Expected: Clean build

**Step 3: Start dev server and verify visually**

Run: `npm run dev`

Check the following in the browser:

1. **Text feed:** Each message has a colorful symmetric identicon instead of a plain letter circle
2. **Media cards:** Card headers show identicons before sender names
3. **Nav panel:** DM entries (Alice, Bob, Carol, Eve) show identicons with status text subtitles for those who have them
4. **Popover:** Clicking any avatar opens a popover with large identicon, name, status, address, and sound slots
5. **Different peers have visually distinct identicons** — Alice, Bob, Carol, Dave, Eve should all look different
6. **Identicons are circular** at all sizes (24px in text feed, 20px in cards/nav, 64px in popover)

**Step 4: Verify no regressions**

- Message priority (quiet grouping, loud borders) still works
- Settings panel still opens/closes
- Media pills still link to cards
- Link-back from cards to messages still works
- Quiet group expand/collapse still works

No commit needed — verification only.
