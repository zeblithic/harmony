# Media Trust & Auto-Preview System Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement a security-first media trust system where external media defaults to blocked placeholders with a two-step confirmation to load, configurable per-peer and per-community.

**Architecture:** A `TrustService` class (parallel to the existing `NotificationService`) manages trust settings and per-session loaded-attachment tracking. `MediaFeed` and `TextMessage` query trust state to decide between rendering `UntrustedMediaCard` placeholders or real `MediaCard` content. A `sanitizeHref()` utility hardens all link rendering regardless of trust level.

**Tech Stack:** Svelte 5 (runes), TypeScript, vitest + @testing-library/svelte

**Working directories:**
- harmony-client code: `/Users/zeblith/work/zeblithic/harmony-client`
- Tests: `npx vitest run` (NOT `cargo test`)
- Build: `npm run build`
- Branch: `jake-client-media-trust` (already created in both repos)

---

### Task 1: URL Sanitization Utility

**Files:**
- Create: `src/lib/url-sanitize.ts`
- Create: `src/lib/url-sanitize.test.ts`

**Context:** Before building the trust system, we need a `sanitizeHref()` function that rejects dangerous URL schemes (`javascript:`, `data:`, `vbscript:`). This utility will be used by MediaCard, UntrustedMediaCard, and TextMessage — anywhere an `href` attribute is set from message data.

**Step 1: Write the failing tests**

Create `src/lib/url-sanitize.test.ts`:

```typescript
import { describe, it, expect } from 'vitest';
import { sanitizeHref } from './url-sanitize';

describe('sanitizeHref', () => {
  it('allows https URLs', () => {
    expect(sanitizeHref('https://example.com/path')).toBe('https://example.com/path');
  });

  it('allows http URLs', () => {
    expect(sanitizeHref('http://example.com')).toBe('http://example.com');
  });

  it('allows mailto URLs', () => {
    expect(sanitizeHref('mailto:user@example.com')).toBe('mailto:user@example.com');
  });

  it('rejects javascript: URLs', () => {
    expect(sanitizeHref('javascript:alert(1)')).toBe('');
  });

  it('rejects javascript: with mixed case', () => {
    expect(sanitizeHref('JaVaScRiPt:alert(1)')).toBe('');
  });

  it('rejects data: URLs', () => {
    expect(sanitizeHref('data:text/html,<script>alert(1)</script>')).toBe('');
  });

  it('rejects vbscript: URLs', () => {
    expect(sanitizeHref('vbscript:MsgBox("hi")')).toBe('');
  });

  it('rejects javascript: with leading whitespace', () => {
    expect(sanitizeHref('  javascript:alert(1)')).toBe('');
  });

  it('returns empty string for empty input', () => {
    expect(sanitizeHref('')).toBe('');
  });

  it('returns empty string for undefined input', () => {
    expect(sanitizeHref(undefined)).toBe('');
  });
});
```

**Step 2: Run tests to verify they fail**

Run: `npx vitest run src/lib/url-sanitize.test.ts`
Expected: FAIL — `sanitizeHref` not found

**Step 3: Write minimal implementation**

Create `src/lib/url-sanitize.ts`:

```typescript
const ALLOWED_SCHEMES = ['https:', 'http:', 'mailto:'];

/**
 * Sanitize a URL for use in href attributes.
 * Only allows http:, https:, and mailto: schemes.
 * Returns empty string for dangerous or empty URLs.
 */
export function sanitizeHref(url: string | undefined): string {
  if (!url) return '';
  const trimmed = url.trim();
  if (!trimmed) return '';
  try {
    const parsed = new URL(trimmed);
    return ALLOWED_SCHEMES.includes(parsed.protocol) ? trimmed : '';
  } catch {
    // Relative URLs or malformed — reject to be safe
    return '';
  }
}
```

**Step 4: Run tests to verify they pass**

Run: `npx vitest run src/lib/url-sanitize.test.ts`
Expected: PASS (all 10 tests)

**Step 5: Commit**

```bash
git add src/lib/url-sanitize.ts src/lib/url-sanitize.test.ts
git commit -m "feat: add URL sanitization utility for href attributes"
```

---

### Task 2: TrustService

**Files:**
- Create: `src/lib/trust-service.ts`
- Create: `src/lib/trust-service.test.ts`
- Modify: `src/lib/types.ts` — add `TrustLevel` and `TrustSettings` types

**Context:** The `TrustService` follows the same pattern as `NotificationService` (see `src/lib/notification-service.ts`). It manages a resolution chain: per-peer override > per-community override > global default. It also tracks which attachment IDs have been explicitly loaded this session.

**Step 1: Add types to `src/lib/types.ts`**

After line 68 (after `export type UnreadLevel = ...`), add:

```typescript
export type TrustLevel = 'untrusted' | 'preview' | 'trusted';

export interface TrustSettings {
  global: TrustLevel;
  perPeer: Map<string, TrustLevel>;
  perCommunity: Map<string, TrustLevel>;
}
```

**Step 2: Write the failing tests**

Create `src/lib/trust-service.test.ts`:

```typescript
import { describe, it, expect } from 'vitest';
import { TrustService } from './trust-service';

describe('TrustService', () => {
  it('returns global default (untrusted) when no overrides exist', () => {
    const svc = new TrustService();
    expect(svc.resolve('peer-1')).toBe('untrusted');
  });

  it('respects custom global default', () => {
    const svc = new TrustService();
    svc.setGlobalTrust('trusted');
    expect(svc.resolve('peer-1')).toBe('trusted');
  });

  it('respects per-peer override over global', () => {
    const svc = new TrustService();
    svc.setPeerTrust('peer-1', 'trusted');
    expect(svc.resolve('peer-1')).toBe('trusted');
    expect(svc.resolve('peer-2')).toBe('untrusted');
  });

  it('respects per-community override over global', () => {
    const svc = new TrustService();
    svc.setCommunityTrust('comm-1', 'trusted');
    expect(svc.resolve('peer-1', 'comm-1')).toBe('trusted');
    expect(svc.resolve('peer-1')).toBe('untrusted');
  });

  it('per-peer beats per-community', () => {
    const svc = new TrustService();
    svc.setCommunityTrust('comm-1', 'trusted');
    svc.setPeerTrust('peer-1', 'untrusted');
    expect(svc.resolve('peer-1', 'comm-1')).toBe('untrusted');
  });

  it('clearPeerTrust removes the override', () => {
    const svc = new TrustService();
    svc.setPeerTrust('peer-1', 'trusted');
    svc.clearPeerTrust('peer-1');
    expect(svc.resolve('peer-1')).toBe('untrusted');
  });

  it('clearCommunityTrust removes the override', () => {
    const svc = new TrustService();
    svc.setCommunityTrust('comm-1', 'trusted');
    svc.clearCommunityTrust('comm-1');
    expect(svc.resolve('peer-1', 'comm-1')).toBe('untrusted');
  });
});

describe('TrustService.loadedAttachments', () => {
  it('tracks loaded attachment IDs', () => {
    const svc = new TrustService();
    expect(svc.isLoaded('att-1')).toBe(false);
    svc.markLoaded('att-1');
    expect(svc.isLoaded('att-1')).toBe(true);
    expect(svc.isLoaded('att-2')).toBe(false);
  });

  it('clearLoaded resets all loaded state', () => {
    const svc = new TrustService();
    svc.markLoaded('att-1');
    svc.markLoaded('att-2');
    svc.clearLoaded();
    expect(svc.isLoaded('att-1')).toBe(false);
    expect(svc.isLoaded('att-2')).toBe(false);
  });
});

describe('TrustService.isGated', () => {
  it('gates image attachments', () => {
    expect(TrustService.isGated({ id: '1', type: 'image', url: 'https://example.com/img.png' })).toBe(true);
  });

  it('gates link attachments', () => {
    expect(TrustService.isGated({ id: '2', type: 'link', url: 'https://example.com' })).toBe(true);
  });

  it('does not gate code attachments', () => {
    expect(TrustService.isGated({ id: '3', type: 'code', content: 'console.log("hi")' })).toBe(false);
  });
});
```

**Step 3: Run tests to verify they fail**

Run: `npx vitest run src/lib/trust-service.test.ts`
Expected: FAIL — `TrustService` not found

**Step 4: Write minimal implementation**

Create `src/lib/trust-service.ts`:

```typescript
import type { MediaAttachment, TrustLevel, TrustSettings } from './types';

export class TrustService {
  settings: TrustSettings;
  private loadedAttachments = new Set<string>();

  constructor() {
    this.settings = {
      global: 'untrusted',
      perPeer: new Map(),
      perCommunity: new Map(),
    };
  }

  resolve(peerAddress: string, communityId?: string): TrustLevel {
    const peerLevel = this.settings.perPeer.get(peerAddress);
    if (peerLevel !== undefined) return peerLevel;

    if (communityId) {
      const commLevel = this.settings.perCommunity.get(communityId);
      if (commLevel !== undefined) return commLevel;
    }

    return this.settings.global;
  }

  setGlobalTrust(level: TrustLevel): void {
    this.settings.global = level;
  }

  setPeerTrust(address: string, level: TrustLevel): void {
    this.settings.perPeer.set(address, level);
  }

  clearPeerTrust(address: string): void {
    this.settings.perPeer.delete(address);
  }

  setCommunityTrust(id: string, level: TrustLevel): void {
    this.settings.perCommunity.set(id, level);
  }

  clearCommunityTrust(id: string): void {
    this.settings.perCommunity.delete(id);
  }

  isLoaded(attachmentId: string): boolean {
    return this.loadedAttachments.has(attachmentId);
  }

  markLoaded(attachmentId: string): void {
    this.loadedAttachments.add(attachmentId);
  }

  clearLoaded(): void {
    this.loadedAttachments.clear();
  }

  /**
   * Determines if an attachment type is subject to the trust gate.
   * Images and links make network requests — gated.
   * Code blocks are local content — not gated.
   */
  static isGated(attachment: MediaAttachment): boolean {
    return attachment.type === 'image' || attachment.type === 'link';
  }
}
```

**Step 5: Run tests to verify they pass**

Run: `npx vitest run src/lib/trust-service.test.ts`
Expected: PASS (all 10 tests)

**Step 6: Commit**

```bash
git add src/lib/types.ts src/lib/trust-service.ts src/lib/trust-service.test.ts
git commit -m "feat: add TrustService with resolution chain and loaded tracking"
```

---

### Task 3: UntrustedMediaCard Component

**Files:**
- Create: `src/lib/components/UntrustedMediaCard.svelte`
- Create: `src/lib/components/__tests__/UntrustedMediaCard.test.ts`

**Context:** This component replaces `MediaCard` for gated attachments when the trust level is `untrusted` or `preview`. It shows a placeholder with a two-step load flow: click "Show" → 1-second cooldown → click "Confirm load" → fires `onLoad` callback. No URL/domain is displayed — only the peer name and attachment type.

Reference the existing `MediaCard.svelte` for the card-header pattern (avatar, sender, timestamp, link-back arrow). Use the `Avatar` component from `src/lib/components/Avatar.svelte`.

**Step 1: Write the failing tests**

Create `src/lib/components/__tests__/UntrustedMediaCard.test.ts`:

```typescript
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/svelte';
import UntrustedMediaCard from '../UntrustedMediaCard.svelte';
import type { Message, MediaAttachment } from '../../types';

const mockMessage: Message = {
  id: 'msg-1',
  sender: { address: 'alice-addr', displayName: 'Alice' },
  text: 'Check this out',
  timestamp: 1709654400000,
  media: [],
  priority: 'standard',
};

const mockImageAttachment: MediaAttachment = {
  id: 'att-1',
  type: 'image',
  url: 'https://example.com/photo.jpg',
  title: 'A photo',
};

const mockLinkAttachment: MediaAttachment = {
  id: 'att-2',
  type: 'link',
  url: 'https://example.com/page',
  title: 'A page',
  domain: 'example.com',
};

describe('UntrustedMediaCard', () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it('renders blocked state with attachment type for images', () => {
    render(UntrustedMediaCard, {
      props: { message: mockMessage, attachment: mockImageAttachment },
    });
    expect(screen.getByText(/blocked media/i)).toBeTruthy();
    expect(screen.getByText(/image/i)).toBeTruthy();
    expect(screen.getByText('Alice')).toBeTruthy();
  });

  it('renders blocked state with attachment type for links', () => {
    render(UntrustedMediaCard, {
      props: { message: mockMessage, attachment: mockLinkAttachment },
    });
    expect(screen.getByText(/link/i)).toBeTruthy();
  });

  it('does not display URL or domain', () => {
    render(UntrustedMediaCard, {
      props: { message: mockMessage, attachment: mockLinkAttachment },
    });
    expect(screen.queryByText('example.com')).toBeNull();
    expect(screen.queryByText('https://example.com/page')).toBeNull();
  });

  it('shows Show button in initial state', () => {
    render(UntrustedMediaCard, {
      props: { message: mockMessage, attachment: mockImageAttachment },
    });
    expect(screen.getByRole('button', { name: /show/i })).toBeTruthy();
  });

  it('transitions to disabled Confirm load after clicking Show', async () => {
    render(UntrustedMediaCard, {
      props: { message: mockMessage, attachment: mockImageAttachment },
    });
    await fireEvent.click(screen.getByRole('button', { name: /show/i }));
    const confirmBtn = screen.getByRole('button', { name: /confirm load/i });
    expect(confirmBtn).toBeTruthy();
    expect(confirmBtn.hasAttribute('disabled') || confirmBtn.getAttribute('aria-disabled') === 'true').toBe(true);
  });

  it('enables Confirm load button after 1 second cooldown', async () => {
    render(UntrustedMediaCard, {
      props: { message: mockMessage, attachment: mockImageAttachment },
    });
    await fireEvent.click(screen.getByRole('button', { name: /show/i }));
    vi.advanceTimersByTime(1000);
    // Need to wait for Svelte reactivity
    await vi.advanceTimersByTimeAsync(0);
    const confirmBtn = screen.getByRole('button', { name: /confirm load/i });
    expect(confirmBtn.hasAttribute('disabled')).toBe(false);
    expect(confirmBtn.getAttribute('aria-disabled')).not.toBe('true');
  });

  it('fires onLoad with attachment ID when Confirm load is clicked after cooldown', async () => {
    const onLoad = vi.fn();
    render(UntrustedMediaCard, {
      props: { message: mockMessage, attachment: mockImageAttachment, onLoad },
    });
    await fireEvent.click(screen.getByRole('button', { name: /show/i }));
    vi.advanceTimersByTime(1000);
    await vi.advanceTimersByTimeAsync(0);
    await fireEvent.click(screen.getByRole('button', { name: /confirm load/i }));
    expect(onLoad).toHaveBeenCalledWith('att-1');
  });

  it('does not fire onLoad when Confirm load clicked during cooldown', async () => {
    const onLoad = vi.fn();
    render(UntrustedMediaCard, {
      props: { message: mockMessage, attachment: mockImageAttachment, onLoad },
    });
    await fireEvent.click(screen.getByRole('button', { name: /show/i }));
    // Don't advance time — still in cooldown
    const confirmBtn = screen.getByRole('button', { name: /confirm load/i });
    await fireEvent.click(confirmBtn);
    expect(onLoad).not.toHaveBeenCalled();
  });

  it('has correct aria-label for screen readers', () => {
    render(UntrustedMediaCard, {
      props: { message: mockMessage, attachment: mockImageAttachment },
    });
    const region = screen.getByLabelText(/blocked media.*image.*alice/i);
    expect(region).toBeTruthy();
  });
});
```

**Step 2: Run tests to verify they fail**

Run: `npx vitest run src/lib/components/__tests__/UntrustedMediaCard.test.ts`
Expected: FAIL — component not found

**Step 3: Write the component**

Create `src/lib/components/UntrustedMediaCard.svelte`:

```svelte
<script lang="ts">
  import type { Message, MediaAttachment } from '../types';
  import Avatar from './Avatar.svelte';

  let { message, attachment, onLinkBack, onAvatarClick, onLoad }: {
    message: Message;
    attachment: MediaAttachment;
    onLinkBack?: (messageId: string) => void;
    onAvatarClick?: (address: string, event: MouseEvent) => void;
    onLoad?: (attachmentId: string) => void;
  } = $props();

  let timeStr = $derived(
    new Date(message.timestamp).toLocaleTimeString([], {
      hour: '2-digit',
      minute: '2-digit',
    })
  );

  type LoadState = 'blocked' | 'confirming' | 'cooldown';
  let loadState = $state<LoadState>('blocked');
  let cooldownTimer: ReturnType<typeof setTimeout> | null = null;

  function handleShow() {
    loadState = 'cooldown';
    cooldownTimer = setTimeout(() => {
      loadState = 'confirming';
    }, 1000);
  }

  function handleConfirm() {
    if (loadState !== 'confirming') return;
    onLoad?.(attachment.id);
  }

  function handleCancel() {
    loadState = 'blocked';
    if (cooldownTimer) {
      clearTimeout(cooldownTimer);
      cooldownTimer = null;
    }
  }

  const TYPE_LABELS: Record<string, string> = {
    image: 'image',
    link: 'link',
  };
</script>

<div
  class="untrusted-card"
  id="media-{attachment.id}"
  aria-label="Blocked media, {TYPE_LABELS[attachment.type] ?? attachment.type}, from {message.sender.displayName}"
>
  <div class="card-header" role="button" tabindex="0" onclick={() => onLinkBack?.(message.id)} onkeydown={(e) => { if ((e.key === 'Enter' || e.key === ' ') && !(e.target as HTMLElement).closest('.avatar')) { e.preventDefault(); onLinkBack?.(message.id); } }}>
    <Avatar
      address={message.sender.address}
      displayName={message.sender.displayName}
      avatarUrl={message.sender.avatarUrl}
      size={20}
      onclick={(e) => { e.stopPropagation(); onAvatarClick?.(message.sender.address, e); }}
    />
    <span class="card-sender">{message.sender.displayName}</span>
    <span class="card-time">{timeStr}</span>
    <span class="link-back-icon" title="Jump to message">&#8599;</span>
  </div>

  <div class="card-body">
    <span class="lock-icon">&#128274;</span>
    <span class="blocked-label">Blocked media &mdash; {TYPE_LABELS[attachment.type] ?? attachment.type}</span>
  </div>

  <div class="card-actions">
    {#if loadState === 'blocked'}
      <button class="action-btn" onclick={handleShow}>Show</button>
    {:else if loadState === 'cooldown'}
      <button class="action-btn confirming" disabled aria-disabled="true">Confirm load</button>
      <button class="cancel-btn" onclick={handleCancel}>Cancel</button>
    {:else if loadState === 'confirming'}
      <button class="action-btn confirming" onclick={handleConfirm} aria-live="polite">Confirm load</button>
      <button class="cancel-btn" onclick={handleCancel}>Cancel</button>
    {/if}
  </div>
</div>

<style>
  .untrusted-card {
    background: var(--bg-tertiary);
    border-radius: 8px;
    overflow: hidden;
    scroll-margin-top: 12px;
  }

  .card-header {
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 8px 12px;
    border: none;
    background: none;
    color: var(--text-secondary);
    font-size: 13px;
    cursor: pointer;
    width: 100%;
    text-align: left;
  }

  .card-header:hover {
    background: rgba(255, 255, 255, 0.03);
  }

  .card-sender {
    font-weight: 600;
    color: var(--text-primary);
    font-size: 13px;
  }

  .card-time {
    color: var(--text-muted);
    font-size: 11px;
  }

  .link-back-icon {
    margin-left: auto;
    font-size: 14px;
    color: var(--text-muted);
    opacity: 0;
    transition: opacity 0.15s;
  }

  .card-header:hover .link-back-icon {
    opacity: 1;
  }

  .card-body {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 12px;
    color: var(--text-muted);
    font-size: 13px;
  }

  .lock-icon {
    font-size: 16px;
  }

  .blocked-label {
    font-style: italic;
  }

  .card-actions {
    display: flex;
    gap: 8px;
    padding: 0 12px 12px;
  }

  .action-btn {
    padding: 6px 16px;
    border: 1px solid var(--border);
    border-radius: 4px;
    background: var(--bg-secondary);
    color: var(--text-primary);
    font-size: 12px;
    cursor: pointer;
  }

  .action-btn:hover:not(:disabled) {
    background: var(--bg-primary);
  }

  .action-btn:disabled {
    opacity: 0.4;
    cursor: not-allowed;
  }

  .action-btn.confirming:not(:disabled) {
    border-color: var(--accent);
    color: var(--accent);
  }

  .cancel-btn {
    padding: 6px 12px;
    border: none;
    border-radius: 4px;
    background: none;
    color: var(--text-muted);
    font-size: 12px;
    cursor: pointer;
  }

  .cancel-btn:hover {
    color: var(--text-primary);
  }
</style>
```

**Step 4: Run tests to verify they pass**

Run: `npx vitest run src/lib/components/__tests__/UntrustedMediaCard.test.ts`
Expected: PASS (all 8 tests)

**Step 5: Commit**

```bash
git add src/lib/components/UntrustedMediaCard.svelte src/lib/components/__tests__/UntrustedMediaCard.test.ts
git commit -m "feat: add UntrustedMediaCard with two-step load confirmation"
```

---

### Task 4: Harden MediaCard Security Attributes

**Files:**
- Modify: `src/lib/components/MediaCard.svelte`

**Context:** Regardless of trust level, all rendered media should have security hardening. This task adds `referrerpolicy="no-referrer"` to images, changes `rel="noopener"` to `rel="noopener noreferrer"` on links, adds `crossorigin="anonymous"` to images, and uses `sanitizeHref()` for link hrefs.

**Step 1: Apply changes to `MediaCard.svelte`**

Add import at top of `<script>` block (after the existing imports):

```typescript
import { sanitizeHref } from '../url-sanitize';
```

Replace the image tag (lines 36-41):

```svelte
      <img
        src={attachment.url}
        alt={attachment.title ?? 'image'}
        class="card-image"
        loading="lazy"
        referrerpolicy="no-referrer"
        crossorigin="anonymous"
      />
```

Replace the link tag (line 46):

```svelte
      <a href={sanitizeHref(attachment.url)} class="card-link" target="_blank" rel="noopener noreferrer">
```

**Step 2: Run all tests and build**

Run: `npx vitest run`
Expected: All tests pass

Run: `npm run build`
Expected: Build succeeds

**Step 3: Commit**

```bash
git add src/lib/components/MediaCard.svelte
git commit -m "fix: add security hardening to MediaCard (referrerpolicy, noreferrer, sanitizeHref)"
```

---

### Task 5: Harden TextMessage Inline Embeds

**Files:**
- Modify: `src/lib/components/TextMessage.svelte`

**Context:** The collapsed/mobile mode in `TextMessage` renders inline images and links directly. Apply the same security hardening as `MediaCard`, plus add `sanitizeHref()` to the inline link.

**Step 1: Apply changes to `TextMessage.svelte`**

Add import at top of `<script>` block:

```typescript
import { sanitizeHref } from '../url-sanitize';
```

Replace the inline image tag (line 40):

```svelte
                <img src={attachment.url} alt={attachment.title ?? 'image'} class="inline-image" referrerpolicy="no-referrer" crossorigin="anonymous" />
```

Replace the inline link tag (lines 42-44):

```svelte
                <a href={sanitizeHref(attachment.url)} class="inline-link" target="_blank" rel="noopener noreferrer">
                  {attachment.title ?? attachment.url}
                </a>
```

**Step 2: Run all tests and build**

Run: `npx vitest run`
Expected: All tests pass

Run: `npm run build`
Expected: Build succeeds

**Step 3: Commit**

```bash
git add src/lib/components/TextMessage.svelte
git commit -m "fix: add security hardening to TextMessage inline embeds"
```

---

### Task 6: Wire Trust Gate into MediaFeed

**Files:**
- Modify: `src/lib/components/MediaFeed.svelte`

**Context:** `MediaFeed` needs to decide for each attachment whether to render `MediaCard` (trusted/loaded) or `UntrustedMediaCard` (untrusted). It accepts a `TrustService` instance and a reactive version counter. When a user loads an untrusted attachment, `markLoaded()` is called and the version counter increments, triggering re-render.

**Step 1: Modify `MediaFeed.svelte`**

Replace the entire file content:

```svelte
<script lang="ts">
  import type { Message } from '../types';
  import { TrustService } from '../trust-service';
  import MediaCard from './MediaCard.svelte';
  import UntrustedMediaCard from './UntrustedMediaCard.svelte';

  let { messages, trustService, trustVersion = 0, onLinkBack, onAvatarClick, onTrustChange }: {
    messages: Message[];
    trustService: TrustService;
    trustVersion?: number;
    onLinkBack?: (messageId: string) => void;
    onAvatarClick?: (address: string, event: MouseEvent) => void;
    onTrustChange?: () => void;
  } = $props();

  let mediaItems = $derived(
    messages
      .filter((m) => m.media.length > 0)
      .flatMap((m) => m.media.map((a) => ({ message: m, attachment: a })))
  );

  function shouldBlock(message: Message, attachmentId: string): boolean {
    void trustVersion;
    if (!TrustService.isGated({ id: attachmentId, type: 'image' })) return false;
    const item = mediaItems.find((mi) => mi.attachment.id === attachmentId);
    if (!item) return false;
    if (!TrustService.isGated(item.attachment)) return false;
    if (trustService.isLoaded(attachmentId)) return false;
    const level = trustService.resolve(message.sender.address);
    return level !== 'trusted';
  }

  function handleLoad(attachmentId: string) {
    trustService.markLoaded(attachmentId);
    onTrustChange?.();
  }
</script>

<div class="media-feed">
  {#if mediaItems.length === 0}
    <div class="empty-state">No media yet</div>
  {:else}
    {#each mediaItems as { message, attachment } (attachment.id)}
      {#if TrustService.isGated(attachment) && trustService.resolve(message.sender.address) !== 'trusted' && !trustService.isLoaded(attachment.id)}
        <UntrustedMediaCard {message} {attachment} {onLinkBack} {onAvatarClick} onLoad={handleLoad} />
      {:else}
        <MediaCard {message} {attachment} {onLinkBack} {onAvatarClick} />
      {/if}
    {/each}
  {/if}
</div>

<style>
  .media-feed {
    display: flex;
    flex-direction: column;
    gap: 12px;
  }

  .empty-state {
    color: var(--text-muted);
    text-align: center;
    padding: 40px 20px;
    font-size: 14px;
  }
</style>
```

**Step 2: Run all tests and build**

Run: `npx vitest run`
Expected: All tests pass

Run: `npm run build`
Expected: Build succeeds

**Step 3: Commit**

```bash
git add src/lib/components/MediaFeed.svelte
git commit -m "feat: wire trust gate into MediaFeed to block untrusted media"
```

---

### Task 7: Wire Trust Gate into TextMessage Pills

**Files:**
- Modify: `src/lib/components/TextMessage.svelte`
- Modify: `src/lib/components/TextFeed.svelte`

**Context:** In desktop mode, media pills should show locked/unlocked state. Untrusted gated media shows `🔒 blocked image` instead of the normal pill. In collapsed mode, untrusted inline embeds show a compact inline placeholder instead of rendering the content. The `TrustService` is passed down through `TextFeed` → `TextMessage`.

**Step 1: Modify `TextMessage.svelte`**

Add to the imports and props:

```typescript
import type { Message } from '../types';
import { TrustService } from '../trust-service';
import { sanitizeHref } from '../url-sanitize';
import Avatar from './Avatar.svelte';
```

Update the props destructuring to accept `trustService` and `trustVersion`:

```typescript
let { message, collapsed = false, onMediaClick, onAvatarClick, trustService, trustVersion = 0 }: {
  message: Message;
  collapsed?: boolean;
  onMediaClick?: (mediaId: string) => void;
  onAvatarClick?: (address: string, event: MouseEvent) => void;
  trustService?: TrustService;
  trustVersion?: number;
} = $props();
```

Add a helper function:

```typescript
function isBlocked(attachmentId: string, attachmentType: string): boolean {
  void trustVersion;
  if (!trustService) return false;
  if (attachmentType === 'code') return false;
  const level = trustService.resolve(message.sender.address);
  if (level === 'trusted') return false;
  return !trustService.isLoaded(attachmentId);
}
```

Replace the media indicators section (lines 34-64) with:

```svelte
    {#if message.media.length > 0}
      <div class="media-indicators">
        {#each message.media as attachment (attachment.id)}
          {#if collapsed}
            {#if isBlocked(attachment.id, attachment.type)}
              <div class="inline-embed inline-blocked">
                <span class="blocked-inline-icon">&#128274;</span>
                <span class="blocked-inline-label">Blocked {attachment.type}</span>
              </div>
            {:else}
              <div class="inline-embed">
                {#if attachment.type === 'image'}
                  <img src={attachment.url} alt={attachment.title ?? 'image'} class="inline-image" referrerpolicy="no-referrer" crossorigin="anonymous" />
                {:else if attachment.type === 'link'}
                  <a href={sanitizeHref(attachment.url)} class="inline-link" target="_blank" rel="noopener noreferrer">
                    {attachment.title ?? attachment.url}
                  </a>
                {:else if attachment.type === 'code'}
                  <pre class="inline-code"><code>{attachment.content}</code></pre>
                {/if}
              </div>
            {/if}
          {:else}
            {#if isBlocked(attachment.id, attachment.type)}
              <button
                class="media-pill blocked"
                onclick={() => onMediaClick?.(attachment.id)}
              >
                <span class="pill-icon">&#128274;</span> blocked {attachment.type}
              </button>
            {:else}
              <button
                class="media-pill"
                onclick={() => onMediaClick?.(attachment.id)}
              >
                {#if attachment.type === 'image'}
                  <span class="pill-icon">&#128444;</span> {attachment.title ?? 'image'}
                {:else if attachment.type === 'link'}
                  <span class="pill-icon">&#128279;</span> {attachment.domain ?? 'link'}
                {:else if attachment.type === 'code'}
                  <span class="pill-icon">&lt;/&gt;</span> {attachment.title ?? 'code'}
                {/if}
              </button>
            {/if}
          {/if}
        {/each}
      </div>
    {/if}
```

Add CSS for blocked states (inside the existing `<style>` block):

```css
  .media-pill.blocked {
    color: var(--text-muted);
    font-style: italic;
  }

  .inline-blocked {
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 8px 12px;
    background: var(--bg-tertiary);
    border-radius: 6px;
    color: var(--text-muted);
    font-size: 12px;
    font-style: italic;
  }

  .blocked-inline-icon {
    font-size: 14px;
  }

  .blocked-inline-label {
    font-size: 12px;
  }
```

**Step 2: Modify `TextFeed.svelte`**

Update imports and props to accept and pass through `trustService` and `trustVersion`:

```svelte
<script lang="ts">
  import type { Message, MessagePriority } from '../types';
  import type { TrustService } from '../trust-service';
  import { groupMessages } from '../feed-utils';
  import TextMessage from './TextMessage.svelte';
  import QuietMessageGroup from './QuietMessageGroup.svelte';
  import ComposeBar from './ComposeBar.svelte';

  let { messages, collapsed = false, onMediaClick, onSend, onAvatarClick, trustService, trustVersion = 0 }: {
    messages: Message[];
    collapsed?: boolean;
    onMediaClick?: (mediaId: string) => void;
    onSend?: (text: string, priority: MessagePriority) => void;
    onAvatarClick?: (address: string, event: MouseEvent) => void;
    trustService?: TrustService;
    trustVersion?: number;
  } = $props();

  let feedItems = $derived(groupMessages(messages));
</script>
```

Update the template to pass `trustService` and `trustVersion` to `TextMessage` and `QuietMessageGroup`:

```svelte
        <TextMessage message={item.message} {collapsed} {onMediaClick} {onAvatarClick} {trustService} {trustVersion} />
```

```svelte
        <QuietMessageGroup messages={item.messages} {collapsed} {onMediaClick} {onAvatarClick} {trustService} {trustVersion} />
```

**Step 3: Update `QuietMessageGroup.svelte`**

Read the file first and add the same `trustService` and `trustVersion` props, passing them through to each `TextMessage` it renders.

**Step 4: Run all tests and build**

Run: `npx vitest run`
Expected: All tests pass

Run: `npm run build`
Expected: Build succeeds

**Step 5: Commit**

```bash
git add src/lib/components/TextMessage.svelte src/lib/components/TextFeed.svelte src/lib/components/QuietMessageGroup.svelte
git commit -m "feat: wire trust gate into text feed pills and inline embeds"
```

---

### Task 8: Add Trust Settings to NotificationSettingsPanel

**Files:**
- Modify: `src/lib/components/NotificationSettingsPanel.svelte`

**Context:** Add a "Media Trust" section to each tab of the existing settings panel. The global tab gets a default trust level dropdown. Per-community and per-peer tabs each get a trust level dropdown with a "Use default" option. The `TrustService` is passed as a new prop alongside the existing `NotificationService`.

**Step 1: Modify `NotificationSettingsPanel.svelte`**

Add to the imports:

```typescript
import type { TrustLevel } from '../types';
import type { TrustService } from '../trust-service';
```

Update the props to accept `trustService` and `onTrustChange`:

```typescript
let { service, trustService, peers, communities, onClose, onTrustChange }: {
  service: NotificationService;
  trustService?: TrustService;
  peers: Peer[];
  communities: NavNode[];
  onClose?: () => void;
  onTrustChange?: () => void;
} = $props();
```

Add trust-related constants and helpers after the existing constants:

```typescript
const TRUST_LABELS: Record<TrustLevel, string> = {
  untrusted: 'Untrusted',
  preview: 'Preview (coming soon)',
  trusted: 'Trusted',
};

const TRUST_LEVELS: TrustLevel[] = ['untrusted', 'preview', 'trusted'];

function getGlobalTrust(): TrustLevel {
  void version;
  return trustService?.settings.global ?? 'untrusted';
}

function setGlobalTrust(level: TrustLevel) {
  trustService?.setGlobalTrust(level);
  version++;
  onTrustChange?.();
}

function getPeerTrust(address: string): TrustLevel | undefined {
  void version;
  return trustService?.settings.perPeer.get(address);
}

function setPeerTrust(address: string, level: TrustLevel | '') {
  if (!trustService) return;
  if (level === '') {
    trustService.clearPeerTrust(address);
  } else {
    trustService.setPeerTrust(address, level);
  }
  version++;
  onTrustChange?.();
}

function getCommunityTrust(id: string): TrustLevel | undefined {
  void version;
  return trustService?.settings.perCommunity.get(id);
}

function setCommunityTrust(id: string, level: TrustLevel | '') {
  if (!trustService) return;
  if (level === '') {
    trustService.clearCommunityTrust(id);
  } else {
    trustService.setCommunityTrust(id, level);
  }
  version++;
  onTrustChange?.();
}
```

Add the trust section to the **global tab** (after the existing policy-rows `</div>`, before the `{:else if activeTab === 'communities'}` block):

```svelte
      {#if trustService}
        <div class="trust-section">
          <div class="section-label">Media Trust</div>
          <div class="policy-row">
            <span class="policy-label">Default trust level</span>
            <select
              value={getGlobalTrust()}
              onchange={(e) => setGlobalTrust((e.target as HTMLSelectElement).value as TrustLevel)}
            >
              {#each TRUST_LEVELS as level}
                <option value={level} disabled={level === 'preview'}>{TRUST_LABELS[level]}</option>
              {/each}
            </select>
          </div>
        </div>
      {/if}
```

Add the trust row to each **community** section (after the sound-slots `</div>`, before the closing `</div>` of override-section):

```svelte
          {#if trustService}
            <div class="trust-section">
              <div class="section-label">Media Trust</div>
              <div class="policy-row">
                <span class="policy-label">Trust level</span>
                <select
                  value={getCommunityTrust(comm.id) ?? ''}
                  onchange={(e) => setCommunityTrust(comm.id, (e.target as HTMLSelectElement).value as TrustLevel | '')}
                >
                  <option value="">Use default</option>
                  {#each TRUST_LEVELS as level}
                    <option value={level} disabled={level === 'preview'}>{TRUST_LABELS[level]}</option>
                  {/each}
                </select>
              </div>
            </div>
          {/if}
```

Add the same trust row to each **peer** section (same placement as communities):

```svelte
          {#if trustService}
            <div class="trust-section">
              <div class="section-label">Media Trust</div>
              <div class="policy-row">
                <span class="policy-label">Trust level</span>
                <select
                  value={getPeerTrust(peer.address) ?? ''}
                  onchange={(e) => setPeerTrust(peer.address, (e.target as HTMLSelectElement).value as TrustLevel | '')}
                >
                  <option value="">Use default</option>
                  {#each TRUST_LEVELS as level}
                    <option value={level} disabled={level === 'preview'}>{TRUST_LABELS[level]}</option>
                  {/each}
                </select>
              </div>
            </div>
          {/if}
```

Add CSS for the trust section:

```css
  .trust-section {
    margin-top: 12px;
    padding-top: 8px;
    border-top: 1px solid var(--border);
  }

  .section-label {
    font-size: 11px;
    font-weight: 600;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 6px;
  }
```

**Step 2: Run all tests and build**

Run: `npx vitest run`
Expected: All tests pass

Run: `npm run build`
Expected: Build succeeds

**Step 3: Commit**

```bash
git add src/lib/components/NotificationSettingsPanel.svelte
git commit -m "feat: add trust level settings to notification settings panel"
```

---

### Task 9: Wire Everything Through App.svelte

**Files:**
- Modify: `src/App.svelte`

**Context:** Create a `TrustService` instance in `App.svelte`, add a reactive version counter, and wire it to `MediaFeed`, `TextFeed`, and `NotificationSettingsPanel`. Set up a demo per-peer trust override to show the feature working with mock data.

**Step 1: Modify `App.svelte`**

Add import:

```typescript
import { TrustService } from './lib/trust-service';
```

After the `notificationService` creation (line 43), add:

```typescript
const trustService = new TrustService();
let trustVersion = $state(0);

function handleTrustChange() {
  trustVersion++;
}

// Demo: trust one peer to show the difference
trustService.setPeerTrust('a1b2c3d4', 'trusted');
```

Update the `MediaFeed` usage (line 117):

```svelte
    <MediaFeed messages={allMessages} {trustService} {trustVersion} onLinkBack={scrollToMessage} onAvatarClick={handleAvatarClick} onTrustChange={handleTrustChange} />
```

Update the `TextFeed` usage (line 114):

```svelte
    <TextFeed messages={allMessages} {collapsed} onMediaClick={scrollToMedia} onSend={handleSend} onAvatarClick={handleAvatarClick} {trustService} {trustVersion} />
```

Update the `NotificationSettingsPanel` usage (lines 120-125):

```svelte
    <NotificationSettingsPanel
      service={notificationService}
      {trustService}
      peers={knownPeers}
      {communities}
      onClose={() => { showSettings = false; }}
      onTrustChange={handleTrustChange}
    />
```

**Step 2: Run all tests and build**

Run: `npx vitest run`
Expected: All tests pass

Run: `npm run build`
Expected: Build succeeds

**Step 3: Commit**

```bash
git add src/App.svelte
git commit -m "feat: wire TrustService through App to all media components"
```

---

### Task 10: Final Verification

**Step 1: Run full test suite**

Run: `npx vitest run`
Expected: All tests pass (previous 102 + new tests)

**Step 2: Run build**

Run: `npm run build`
Expected: Clean build, no warnings

**Step 3: Manual smoke test (optional)**

Run: `npm run dev`
Verify:
- Media cards show as blocked placeholders by default
- Two-step load flow works (Show → cooldown → Confirm load)
- Pills in text feed show `🔒 blocked image` for untrusted media
- Settings panel shows trust level dropdowns
- Trusted peer's media renders automatically (mock: `a1b2c3d4`)
- Code blocks always render (not gated)

**Step 4: Commit any fixes, then stop**

If any issues found during verification, fix and commit. Otherwise, all tasks are complete.
