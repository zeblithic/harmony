# Message Priority Levels Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add three-tier message priority (quiet/standard/loud) with sender toggle, quiet message collapsing in the text feed, NotificationService with override chain, and a receiver notification settings panel.

**Architecture:** `MessagePriority` added to the `Message` type. A `NotificationService` class resolves the per-peer > per-community > global override chain. The `ComposeBar` gets a three-state toggle + keyboard shortcuts. `TextFeed` groups consecutive quiet messages into collapsible rows via a `groupMessages()` utility. A slide-out `NotificationSettingsPanel` lets receivers configure policies at three scopes.

**Tech Stack:** Svelte 5 (runes: `$state`, `$derived`, `$props`, `$effect`), TypeScript, vitest + @testing-library/svelte, jsdom

**Repos:** All code changes in `zeblithic/harmony-client` on branch `jake-client-message-priority`. Design doc in `zeblithic/harmony` (already committed).

**Test command:** `npx vitest run` from the harmony-client root.

**Build command:** `npm run build` from the harmony-client root.

---

### Task 1: Types — MessagePriority + NotificationAction + NotificationPolicy

**Files:**
- Modify: `src/lib/types.ts:20-28` (add priority to Message, add new types)

**Context:** The `Message` interface currently has `id`, `sender`, `text`, `timestamp`, `media`. We need to add a `priority` field and define the notification-related types that will be used by the NotificationService and settings panel.

**Step 1: Add types to `src/lib/types.ts`**

After the `Message` interface (line 28), add:

```typescript
export type MessagePriority = 'quiet' | 'standard' | 'loud';

export type NotificationAction = 'silent' | 'dot_only' | 'notify' | 'sound' | 'break_dnd';

export interface NotificationPolicy {
  quiet: NotificationAction;
  standard: NotificationAction;
  loud: NotificationAction;
}

export interface NotificationSettings {
  global: NotificationPolicy;
  perCommunity: Map<string, Partial<NotificationPolicy>>;
  perPeer: Map<string, Partial<NotificationPolicy>>;
}
```

And add `priority` to the `Message` interface:

```typescript
export interface Message {
  id: string;
  sender: Peer;
  text: string;
  /** Unix timestamp in milliseconds */
  timestamp: number;
  /** Empty array for text-only messages */
  media: MediaAttachment[];
  /** Message priority level, defaults to 'standard' */
  priority: MessagePriority;
}
```

**Step 2: Fix all existing test mock messages**

Every test file that creates a `Message` object needs a `priority` field. Add `priority: 'standard'` to:

- `src/lib/components/__tests__/TextMessage.test.ts` — `mockMessage` (line 8) and `mockMessageWithMedia` (line 14)
- `src/lib/components/__tests__/MediaFeed.test.ts` — all three messages in `messagesWithMedia` (lines 6-28) and the `noMedia` message (line 40)

**Step 3: Fix mock data**

In `src/lib/mock-data.ts`, add `priority: 'standard'` to all 15 messages. Then change these specific ones:
- `msg-04` (`priority: 'quiet'`) — "That looks great" — short acknowledgment
- `msg-06` (`priority: 'quiet'`) — "I ran the same test" — informational
- `msg-08` (`priority: 'loud'`) — "Has anyone tested the Reticulum interop?" — attention-seeking
- `msg-10` (`priority: 'quiet'`) — "Perfect. The identity derivation path..." — acknowledgment
- `msg-14` (`priority: 'quiet'`) — "Looks solid" — acknowledgment

**Step 4: Run tests to verify nothing broke**

Run: `npx vitest run`
Expected: All existing tests pass (the only change is adding a required field that we've added to all mock objects).

**Step 5: Run build to verify types compile**

Run: `npm run build`
Expected: Build succeeds.

**Step 6: Commit**

```bash
git add src/lib/types.ts src/lib/mock-data.ts src/lib/components/__tests__/TextMessage.test.ts src/lib/components/__tests__/MediaFeed.test.ts
git commit -m "feat(types): add MessagePriority, NotificationAction, and NotificationPolicy types"
```

---

### Task 2: NotificationService — Core Resolution Logic

**Files:**
- Create: `src/lib/notification-service.ts`
- Create: `src/lib/notification-service.test.ts`

**Context:** The NotificationService resolves the override chain: per-peer > per-community > global. It's a plain TypeScript class with no Svelte dependencies, making it easy to unit test.

**Step 1: Write failing tests in `src/lib/notification-service.test.ts`**

```typescript
import { describe, it, expect } from 'vitest';
import { NotificationService } from './notification-service';
import type { NotificationPolicy } from './types';

const DEFAULT_POLICY: NotificationPolicy = {
  quiet: 'dot_only',
  standard: 'sound',
  loud: 'break_dnd',
};

describe('NotificationService', () => {
  it('returns global defaults when no overrides exist', () => {
    const svc = new NotificationService();
    expect(svc.resolve('quiet', 'peer-1')).toBe('dot_only');
    expect(svc.resolve('standard', 'peer-1')).toBe('sound');
    expect(svc.resolve('loud', 'peer-1')).toBe('break_dnd');
  });

  it('respects per-peer override over global', () => {
    const svc = new NotificationService();
    svc.setPeerPolicy('peer-1', { quiet: 'silent' });
    expect(svc.resolve('quiet', 'peer-1')).toBe('silent');
    // Other levels still use global
    expect(svc.resolve('standard', 'peer-1')).toBe('sound');
  });

  it('respects per-community override over global', () => {
    const svc = new NotificationService();
    svc.setCommunityPolicy('comm-1', { standard: 'notify' });
    expect(svc.resolve('standard', 'peer-1', 'comm-1')).toBe('notify');
    // Without community context, falls back to global
    expect(svc.resolve('standard', 'peer-1')).toBe('sound');
  });

  it('per-peer beats per-community', () => {
    const svc = new NotificationService();
    svc.setCommunityPolicy('comm-1', { loud: 'sound' });
    svc.setPeerPolicy('peer-1', { loud: 'silent' });
    expect(svc.resolve('loud', 'peer-1', 'comm-1')).toBe('silent');
  });

  it('falls through partial peer override to community', () => {
    const svc = new NotificationService();
    svc.setCommunityPolicy('comm-1', { standard: 'notify' });
    svc.setPeerPolicy('peer-1', { quiet: 'silent' });
    // peer has quiet override but not standard, so standard falls to community
    expect(svc.resolve('standard', 'peer-1', 'comm-1')).toBe('notify');
  });

  it('setGlobalPolicy replaces all global defaults', () => {
    const svc = new NotificationService();
    svc.setGlobalPolicy({ quiet: 'silent', standard: 'notify', loud: 'sound' });
    expect(svc.resolve('quiet', 'peer-1')).toBe('silent');
    expect(svc.resolve('standard', 'peer-1')).toBe('notify');
    expect(svc.resolve('loud', 'peer-1')).toBe('sound');
  });

  it('clearPeerPolicy removes the override', () => {
    const svc = new NotificationService();
    svc.setPeerPolicy('peer-1', { quiet: 'silent' });
    svc.clearPeerPolicy('peer-1');
    expect(svc.resolve('quiet', 'peer-1')).toBe('dot_only');
  });

  it('clearCommunityPolicy removes the override', () => {
    const svc = new NotificationService();
    svc.setCommunityPolicy('comm-1', { standard: 'notify' });
    svc.clearCommunityPolicy('comm-1');
    expect(svc.resolve('standard', 'peer-1', 'comm-1')).toBe('sound');
  });

  it('shouldPlaySound returns true for sound and break_dnd', () => {
    const svc = new NotificationService();
    expect(svc.shouldPlaySound('sound')).toBe(true);
    expect(svc.shouldPlaySound('break_dnd')).toBe(true);
    expect(svc.shouldPlaySound('notify')).toBe(false);
    expect(svc.shouldPlaySound('dot_only')).toBe(false);
    expect(svc.shouldPlaySound('silent')).toBe(false);
  });
});
```

**Step 2: Run tests to verify they fail**

Run: `npx vitest run src/lib/notification-service.test.ts`
Expected: FAIL — module `./notification-service` not found.

**Step 3: Implement `src/lib/notification-service.ts`**

```typescript
import type {
  MessagePriority,
  NotificationAction,
  NotificationPolicy,
  NotificationSettings,
} from './types';

const DEFAULT_POLICY: NotificationPolicy = {
  quiet: 'dot_only',
  standard: 'sound',
  loud: 'break_dnd',
};

export class NotificationService {
  settings: NotificationSettings;

  constructor() {
    this.settings = {
      global: { ...DEFAULT_POLICY },
      perCommunity: new Map(),
      perPeer: new Map(),
    };
  }

  resolve(
    priority: MessagePriority,
    peerAddress: string,
    communityId?: string,
  ): NotificationAction {
    // 1. Per-peer override
    const peerPolicy = this.settings.perPeer.get(peerAddress);
    if (peerPolicy && peerPolicy[priority] !== undefined) {
      return peerPolicy[priority]!;
    }

    // 2. Per-community override
    if (communityId) {
      const commPolicy = this.settings.perCommunity.get(communityId);
      if (commPolicy && commPolicy[priority] !== undefined) {
        return commPolicy[priority]!;
      }
    }

    // 3. Global default
    return this.settings.global[priority];
  }

  setGlobalPolicy(policy: NotificationPolicy): void {
    this.settings.global = { ...policy };
  }

  setCommunityPolicy(communityId: string, policy: Partial<NotificationPolicy>): void {
    this.settings.perCommunity.set(communityId, { ...policy });
  }

  setPeerPolicy(peerAddress: string, policy: Partial<NotificationPolicy>): void {
    this.settings.perPeer.set(peerAddress, { ...policy });
  }

  clearCommunityPolicy(communityId: string): void {
    this.settings.perCommunity.delete(communityId);
  }

  clearPeerPolicy(peerAddress: string): void {
    this.settings.perPeer.delete(peerAddress);
  }

  shouldPlaySound(action: NotificationAction): boolean {
    return action === 'sound' || action === 'break_dnd';
  }
}
```

**Step 4: Run tests to verify they pass**

Run: `npx vitest run src/lib/notification-service.test.ts`
Expected: All 9 tests PASS.

**Step 5: Run full test suite**

Run: `npx vitest run`
Expected: All tests pass.

**Step 6: Commit**

```bash
git add src/lib/notification-service.ts src/lib/notification-service.test.ts
git commit -m "feat: add NotificationService with override chain resolution"
```

---

### Task 3: Feed Utils — groupMessages Function

**Files:**
- Create: `src/lib/feed-utils.ts`
- Create: `src/lib/feed-utils.test.ts`

**Context:** The `TextFeed` needs to collapse consecutive quiet messages. This task creates a pure utility function `groupMessages()` that transforms a flat `Message[]` into `FeedItem[]` — either individual messages or groups of consecutive quiet messages.

**Step 1: Write failing tests in `src/lib/feed-utils.test.ts`**

```typescript
import { describe, it, expect } from 'vitest';
import { groupMessages } from './feed-utils';
import type { Message } from './types';

function msg(id: string, priority: 'quiet' | 'standard' | 'loud' = 'standard', sender = 'Alice'): Message {
  return {
    id,
    sender: { address: sender.toLowerCase(), displayName: sender },
    text: `Message ${id}`,
    timestamp: Date.now(),
    media: [],
    priority,
  };
}

describe('groupMessages', () => {
  it('returns empty array for empty input', () => {
    expect(groupMessages([])).toEqual([]);
  });

  it('wraps standard messages as individual items', () => {
    const result = groupMessages([msg('1'), msg('2')]);
    expect(result).toHaveLength(2);
    expect(result[0].kind).toBe('message');
    expect(result[1].kind).toBe('message');
  });

  it('groups consecutive quiet messages', () => {
    const result = groupMessages([
      msg('1', 'quiet', 'Alice'),
      msg('2', 'quiet', 'Bob'),
    ]);
    expect(result).toHaveLength(1);
    expect(result[0].kind).toBe('quiet-group');
    if (result[0].kind === 'quiet-group') {
      expect(result[0].messages).toHaveLength(2);
    }
  });

  it('breaks group on non-quiet message', () => {
    const result = groupMessages([
      msg('1', 'quiet'),
      msg('2', 'standard'),
      msg('3', 'quiet'),
    ]);
    expect(result).toHaveLength(3);
    expect(result[0].kind).toBe('quiet-group');
    expect(result[1].kind).toBe('message');
    expect(result[2].kind).toBe('quiet-group');
  });

  it('wraps loud messages as individual items', () => {
    const result = groupMessages([msg('1', 'loud')]);
    expect(result).toHaveLength(1);
    expect(result[0].kind).toBe('message');
  });

  it('handles single quiet message as a group of one', () => {
    const result = groupMessages([msg('1', 'quiet')]);
    expect(result).toHaveLength(1);
    expect(result[0].kind).toBe('quiet-group');
    if (result[0].kind === 'quiet-group') {
      expect(result[0].messages).toHaveLength(1);
    }
  });

  it('handles all-quiet input as single group', () => {
    const result = groupMessages([
      msg('1', 'quiet'),
      msg('2', 'quiet'),
      msg('3', 'quiet'),
    ]);
    expect(result).toHaveLength(1);
    expect(result[0].kind).toBe('quiet-group');
    if (result[0].kind === 'quiet-group') {
      expect(result[0].messages).toHaveLength(3);
    }
  });

  it('handles mixed priorities correctly', () => {
    const result = groupMessages([
      msg('1', 'standard'),
      msg('2', 'quiet'),
      msg('3', 'quiet'),
      msg('4', 'loud'),
      msg('5', 'quiet'),
      msg('6', 'standard'),
    ]);
    expect(result).toHaveLength(5);
    expect(result[0].kind).toBe('message');    // standard
    expect(result[1].kind).toBe('quiet-group'); // 2 quiet
    expect(result[2].kind).toBe('message');    // loud
    expect(result[3].kind).toBe('quiet-group'); // 1 quiet
    expect(result[4].kind).toBe('message');    // standard
  });
});
```

**Step 2: Run tests to verify they fail**

Run: `npx vitest run src/lib/feed-utils.test.ts`
Expected: FAIL — module `./feed-utils` not found.

**Step 3: Implement `src/lib/feed-utils.ts`**

```typescript
import type { Message } from './types';

export type FeedItem =
  | { kind: 'message'; message: Message }
  | { kind: 'quiet-group'; messages: Message[] };

export function groupMessages(messages: Message[]): FeedItem[] {
  const items: FeedItem[] = [];
  let quietBuffer: Message[] = [];

  function flushQuiet() {
    if (quietBuffer.length > 0) {
      items.push({ kind: 'quiet-group', messages: quietBuffer });
      quietBuffer = [];
    }
  }

  for (const msg of messages) {
    if (msg.priority === 'quiet') {
      quietBuffer.push(msg);
    } else {
      flushQuiet();
      items.push({ kind: 'message', message: msg });
    }
  }

  flushQuiet();
  return items;
}
```

**Step 4: Run tests to verify they pass**

Run: `npx vitest run src/lib/feed-utils.test.ts`
Expected: All 7 tests PASS.

**Step 5: Run full test suite**

Run: `npx vitest run`
Expected: All tests pass.

**Step 6: Commit**

```bash
git add src/lib/feed-utils.ts src/lib/feed-utils.test.ts
git commit -m "feat: add groupMessages utility for quiet message collapsing"
```

---

### Task 4: QuietMessageGroup Component

**Files:**
- Create: `src/lib/components/QuietMessageGroup.svelte`
- Create: `src/lib/components/__tests__/QuietMessageGroup.test.ts`

**Context:** This component renders a collapsed group of quiet messages. It shows a summary line ("🔇 3 quiet messages from Alice, Bob") and expands to show the individual messages with dimmed styling when clicked.

**Step 1: Write failing tests in `src/lib/components/__tests__/QuietMessageGroup.test.ts`**

```typescript
import { render, screen, fireEvent } from '@testing-library/svelte';
import { describe, it, expect } from 'vitest';
import QuietMessageGroup from '../QuietMessageGroup.svelte';
import type { Message } from '../../types';

function makeMsg(id: string, sender: string): Message {
  return {
    id,
    sender: { address: sender.toLowerCase(), displayName: sender },
    text: `Quiet message ${id}`,
    timestamp: Date.now(),
    media: [],
    priority: 'quiet',
  };
}

describe('QuietMessageGroup', () => {
  it('renders collapsed with message count', () => {
    render(QuietMessageGroup, {
      props: { messages: [makeMsg('1', 'Alice'), makeMsg('2', 'Bob')] },
    });
    expect(screen.getByText(/2 quiet messages/)).toBeTruthy();
  });

  it('shows unique sender names in collapsed view', () => {
    render(QuietMessageGroup, {
      props: {
        messages: [
          makeMsg('1', 'Alice'),
          makeMsg('2', 'Alice'),
          makeMsg('3', 'Bob'),
        ],
      },
    });
    expect(screen.getByText(/Alice, Bob/)).toBeTruthy();
  });

  it('does not show individual messages when collapsed', () => {
    render(QuietMessageGroup, {
      props: { messages: [makeMsg('1', 'Alice')] },
    });
    expect(screen.queryByText('Quiet message 1')).toBeNull();
  });

  it('expands to show individual messages on click', async () => {
    render(QuietMessageGroup, {
      props: { messages: [makeMsg('1', 'Alice'), makeMsg('2', 'Bob')] },
    });
    const toggle = screen.getByRole('button');
    await fireEvent.click(toggle);
    expect(screen.getByText('Quiet message 1')).toBeTruthy();
    expect(screen.getByText('Quiet message 2')).toBeTruthy();
  });

  it('collapses again on second click', async () => {
    render(QuietMessageGroup, {
      props: { messages: [makeMsg('1', 'Alice')] },
    });
    const toggle = screen.getByRole('button');
    await fireEvent.click(toggle);
    expect(screen.getByText('Quiet message 1')).toBeTruthy();
    await fireEvent.click(toggle);
    expect(screen.queryByText('Quiet message 1')).toBeNull();
  });

  it('renders expanded messages with dimmed class', async () => {
    const { container } = render(QuietMessageGroup, {
      props: { messages: [makeMsg('1', 'Alice')] },
    });
    const toggle = screen.getByRole('button');
    await fireEvent.click(toggle);
    const dimmed = container.querySelector('.quiet-expanded');
    expect(dimmed).toBeTruthy();
  });
});
```

**Step 2: Run tests to verify they fail**

Run: `npx vitest run src/lib/components/__tests__/QuietMessageGroup.test.ts`
Expected: FAIL — module `../QuietMessageGroup.svelte` not found.

**Step 3: Implement `src/lib/components/QuietMessageGroup.svelte`**

```svelte
<script lang="ts">
  import type { Message } from '../types';
  import TextMessage from './TextMessage.svelte';

  let { messages }: {
    messages: Message[];
  } = $props();

  let expanded = $state(false);

  let senderNames = $derived(
    [...new Set(messages.map((m) => m.sender.displayName))].join(', ')
  );

  let summary = $derived(
    `🔇 ${messages.length} quiet message${messages.length === 1 ? '' : 's'} from ${senderNames}`
  );
</script>

<div class="quiet-group">
  <button class="quiet-toggle" onclick={() => { expanded = !expanded; }}>
    <span class="quiet-summary">{summary}</span>
    <span class="quiet-chevron">{expanded ? '▾' : '▸'}</span>
  </button>
  {#if expanded}
    <div class="quiet-expanded">
      {#each messages as message (message.id)}
        <TextMessage {message} />
      {/each}
    </div>
  {/if}
</div>

<style>
  .quiet-group {
    border-left: 2px solid var(--border);
    margin: 2px 0;
  }

  .quiet-toggle {
    display: flex;
    align-items: center;
    gap: 8px;
    width: 100%;
    padding: 4px 16px;
    border: none;
    background: none;
    color: var(--text-muted);
    font-size: 12px;
    cursor: pointer;
    text-align: left;
  }

  .quiet-toggle:hover {
    background: var(--bg-secondary);
    color: var(--text-secondary);
  }

  .quiet-chevron {
    font-size: 10px;
  }

  .quiet-expanded {
    opacity: 0.6;
  }
</style>
```

**Step 4: Run tests to verify they pass**

Run: `npx vitest run src/lib/components/__tests__/QuietMessageGroup.test.ts`
Expected: All 6 tests PASS.

**Step 5: Run full test suite**

Run: `npx vitest run`
Expected: All tests pass.

**Step 6: Commit**

```bash
git add src/lib/components/QuietMessageGroup.svelte src/lib/components/__tests__/QuietMessageGroup.test.ts
git commit -m "feat: add QuietMessageGroup component for collapsed quiet messages"
```

---

### Task 5: ComposeBar — Priority Toggle + Send

**Files:**
- Modify: `src/lib/components/ComposeBar.svelte`
- Create: `src/lib/components/__tests__/ComposeBar.test.ts`

**Context:** The compose bar currently has a plain text input with no send functionality. We need to add a three-state priority toggle (quiet/standard/loud), keyboard shortcuts (Enter=send at current priority, Ctrl+Enter=send quiet), and an `onSend` callback. The input changes from `<input>` to `<textarea>` for multi-line support.

**Step 1: Write failing tests in `src/lib/components/__tests__/ComposeBar.test.ts`**

```typescript
import { render, screen, fireEvent } from '@testing-library/svelte';
import { describe, it, expect, vi } from 'vitest';
import ComposeBar from '../ComposeBar.svelte';

describe('ComposeBar', () => {
  it('renders with standard priority selected by default', () => {
    const { container } = render(ComposeBar);
    const standardBtn = container.querySelector('[data-priority="standard"]');
    expect(standardBtn?.classList.contains('active')).toBe(true);
  });

  it('changes priority when toggle buttons are clicked', async () => {
    const { container } = render(ComposeBar);
    const quietBtn = container.querySelector('[data-priority="quiet"]')!;
    await fireEvent.click(quietBtn);
    expect(quietBtn.classList.contains('active')).toBe(true);
    const standardBtn = container.querySelector('[data-priority="standard"]');
    expect(standardBtn?.classList.contains('active')).toBe(false);
  });

  it('calls onSend with text and current priority on Enter', async () => {
    const onSend = vi.fn();
    render(ComposeBar, { props: { onSend } });
    const textarea = screen.getByRole('textbox');
    await fireEvent.input(textarea, { target: { value: 'Hello' } });
    await fireEvent.keyDown(textarea, { key: 'Enter' });
    expect(onSend).toHaveBeenCalledWith('Hello', 'standard');
  });

  it('sends as quiet on Ctrl+Enter regardless of toggle state', async () => {
    const onSend = vi.fn();
    render(ComposeBar, { props: { onSend } });
    const textarea = screen.getByRole('textbox');
    await fireEvent.input(textarea, { target: { value: 'Whisper' } });
    await fireEvent.keyDown(textarea, { key: 'Enter', ctrlKey: true });
    expect(onSend).toHaveBeenCalledWith('Whisper', 'quiet');
  });

  it('resets priority to standard after sending', async () => {
    const onSend = vi.fn();
    const { container } = render(ComposeBar, { props: { onSend } });
    // Set to loud
    const loudBtn = container.querySelector('[data-priority="loud"]')!;
    await fireEvent.click(loudBtn);
    // Send
    const textarea = screen.getByRole('textbox');
    await fireEvent.input(textarea, { target: { value: 'Alert!' } });
    await fireEvent.keyDown(textarea, { key: 'Enter' });
    expect(onSend).toHaveBeenCalledWith('Alert!', 'loud');
    // Priority should reset to standard
    const standardBtn = container.querySelector('[data-priority="standard"]');
    expect(standardBtn?.classList.contains('active')).toBe(true);
  });

  it('does not send empty messages', async () => {
    const onSend = vi.fn();
    render(ComposeBar, { props: { onSend } });
    const textarea = screen.getByRole('textbox');
    await fireEvent.keyDown(textarea, { key: 'Enter' });
    expect(onSend).not.toHaveBeenCalled();
  });

  it('clears textarea after sending', async () => {
    const onSend = vi.fn();
    render(ComposeBar, { props: { onSend } });
    const textarea = screen.getByRole('textbox') as HTMLTextAreaElement;
    await fireEvent.input(textarea, { target: { value: 'Hello' } });
    await fireEvent.keyDown(textarea, { key: 'Enter' });
    expect(textarea.value).toBe('');
  });

  it('shows priority label when not standard', async () => {
    const { container } = render(ComposeBar);
    // Default standard — no label
    expect(container.querySelector('.priority-label')).toBeNull();
    // Switch to quiet
    const quietBtn = container.querySelector('[data-priority="quiet"]')!;
    await fireEvent.click(quietBtn);
    expect(screen.getByText('sending quietly')).toBeTruthy();
  });

  it('allows newline with Shift+Enter', async () => {
    const onSend = vi.fn();
    render(ComposeBar, { props: { onSend } });
    const textarea = screen.getByRole('textbox');
    await fireEvent.keyDown(textarea, { key: 'Enter', shiftKey: true });
    expect(onSend).not.toHaveBeenCalled();
  });
});
```

**Step 2: Run tests to verify they fail**

Run: `npx vitest run src/lib/components/__tests__/ComposeBar.test.ts`
Expected: FAIL — tests fail because ComposeBar doesn't have the new features.

**Step 3: Rewrite `src/lib/components/ComposeBar.svelte`**

```svelte
<script lang="ts">
  import type { MessagePriority } from '../types';

  let { onSend, channelName = 'general' }: {
    onSend?: (text: string, priority: MessagePriority) => void;
    channelName?: string;
  } = $props();

  let draft = $state('');
  let priority = $state<MessagePriority>('standard');

  const PRIORITY_ICONS: Record<MessagePriority, string> = {
    quiet: '🔇',
    standard: '🔔',
    loud: '📢',
  };

  const PRIORITY_LABELS: Record<string, string> = {
    quiet: 'sending quietly',
    loud: 'sending loudly',
  };

  function send(overridePriority?: MessagePriority) {
    const text = draft.trim();
    if (!text) return;
    onSend?.(text, overridePriority ?? priority);
    draft = '';
    priority = 'standard';
  }

  function handleKeyDown(e: KeyboardEvent) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      if (e.ctrlKey) {
        send('quiet');
      } else {
        send();
      }
    }
  }
</script>

<div class="compose-bar">
  <div class="priority-toggle">
    {#each (['quiet', 'standard', 'loud'] as MessagePriority[]) as p}
      <button
        class="priority-btn {priority === p ? 'active' : ''}"
        data-priority={p}
        onclick={() => { priority = p; }}
        title={p}
      >{PRIORITY_ICONS[p]}</button>
    {/each}
  </div>
  <div class="compose-input-wrapper">
    <textarea
      class="compose-input"
      placeholder="Message #{channelName}"
      bind:value={draft}
      onkeydown={handleKeyDown}
      rows="1"
    ></textarea>
    {#if priority !== 'standard'}
      <span class="priority-label">{PRIORITY_LABELS[priority]}</span>
    {/if}
  </div>
</div>

<style>
  .compose-bar {
    display: flex;
    align-items: flex-start;
    gap: 8px;
    padding: 12px 16px;
    border-top: 1px solid var(--border);
    flex-shrink: 0;
  }

  .priority-toggle {
    display: flex;
    gap: 2px;
    padding-top: 6px;
  }

  .priority-btn {
    border: none;
    background: none;
    padding: 4px 6px;
    border-radius: 4px;
    cursor: pointer;
    font-size: 14px;
    opacity: 0.4;
    transition: opacity 0.15s;
  }

  .priority-btn:hover {
    opacity: 0.7;
    background: var(--bg-tertiary);
  }

  .priority-btn.active {
    opacity: 1;
    background: var(--bg-tertiary);
  }

  .compose-input-wrapper {
    flex: 1;
    position: relative;
  }

  .compose-input {
    width: 100%;
    padding: 10px 12px;
    border: none;
    border-radius: 8px;
    background: var(--bg-tertiary);
    color: var(--text-primary);
    font-size: 14px;
    outline: none;
    resize: none;
    font-family: inherit;
    line-height: 1.4;
  }

  .compose-input::placeholder {
    color: var(--text-muted);
  }

  .compose-input:focus {
    box-shadow: 0 0 0 2px var(--accent);
  }

  .priority-label {
    position: absolute;
    bottom: -16px;
    left: 12px;
    font-size: 11px;
    color: var(--text-muted);
  }
</style>
```

**Step 4: Run tests to verify they pass**

Run: `npx vitest run src/lib/components/__tests__/ComposeBar.test.ts`
Expected: All 9 tests PASS.

**Step 5: Run full test suite**

Run: `npx vitest run`
Expected: All tests pass.

**Step 6: Commit**

```bash
git add src/lib/components/ComposeBar.svelte src/lib/components/__tests__/ComposeBar.test.ts
git commit -m "feat: add priority toggle and send functionality to ComposeBar"
```

---

### Task 6: TextFeed + TextMessage — Priority-Aware Rendering

**Files:**
- Modify: `src/lib/components/TextFeed.svelte`
- Modify: `src/lib/components/TextMessage.svelte:19` (add loud styling)
- Modify: `src/lib/components/__tests__/TextMessage.test.ts`

**Context:** `TextFeed` needs to use `groupMessages()` to render quiet messages as collapsed groups and standard/loud messages individually. `TextMessage` needs a visual indicator for loud messages (left accent border). The `TextFeed` also needs to pass the `onSend` prop through to `ComposeBar`.

**Step 1: Add loud message test to `src/lib/components/__tests__/TextMessage.test.ts`**

Add these tests after the existing ones:

```typescript
it('renders loud message with accent border class', () => {
  const loudMsg: Message = {
    ...mockMessage,
    id: 'loud-1',
    priority: 'loud',
  };
  const { container } = render(TextMessage, { props: { message: loudMsg } });
  const el = container.querySelector('.text-message');
  expect(el?.classList.contains('loud')).toBe(true);
});

it('does not add loud class to standard messages', () => {
  const { container } = render(TextMessage, { props: { message: mockMessage } });
  const el = container.querySelector('.text-message');
  expect(el?.classList.contains('loud')).toBe(false);
});
```

**Step 2: Run tests to verify they fail**

Run: `npx vitest run src/lib/components/__tests__/TextMessage.test.ts`
Expected: FAIL — `.loud` class not applied.

**Step 3: Update `TextMessage.svelte` to add loud styling**

Add `class:loud` to the root div (line 19):

```svelte
<div class="text-message" class:loud={message.priority === 'loud'} id="msg-{message.id}">
```

Add the CSS at the end of the `<style>` block:

```css
.text-message.loud {
  border-left: 2px solid var(--accent);
  padding-left: 14px;
}

.text-message.loud .sender-name {
  font-weight: 700;
}
```

**Step 4: Run TextMessage tests**

Run: `npx vitest run src/lib/components/__tests__/TextMessage.test.ts`
Expected: All tests PASS.

**Step 5: Update `TextFeed.svelte` to use groupMessages**

Replace the entire `TextFeed.svelte` with:

```svelte
<script lang="ts">
  import type { Message, MessagePriority } from '../types';
  import { groupMessages } from '../feed-utils';
  import TextMessage from './TextMessage.svelte';
  import QuietMessageGroup from './QuietMessageGroup.svelte';
  import ComposeBar from './ComposeBar.svelte';

  let { messages, collapsed = false, onMediaClick, onSend }: {
    messages: Message[];
    collapsed?: boolean;
    onMediaClick?: (mediaId: string) => void;
    onSend?: (text: string, priority: MessagePriority) => void;
  } = $props();

  let feedItems = $derived(groupMessages(messages));
</script>

<div class="text-feed">
  <div class="messages-scroll">
    {#each feedItems as item, i (item.kind === 'message' ? item.message.id : `quiet-${i}`)}
      {#if item.kind === 'message'}
        <TextMessage message={item.message} {collapsed} {onMediaClick} />
      {:else}
        <QuietMessageGroup messages={item.messages} />
      {/if}
    {/each}
  </div>
  <ComposeBar {onSend} />
</div>

<style>
  .text-feed {
    display: flex;
    flex-direction: column;
    height: 100%;
  }

  .messages-scroll {
    flex: 1;
    overflow-y: auto;
    padding: 8px 0;
  }
</style>
```

**Step 6: Run full test suite**

Run: `npx vitest run`
Expected: All tests pass.

**Step 7: Commit**

```bash
git add src/lib/components/TextFeed.svelte src/lib/components/TextMessage.svelte src/lib/components/__tests__/TextMessage.test.ts
git commit -m "feat: priority-aware text feed with quiet collapsing and loud styling"
```

---

### Task 7: NotificationSettingsPanel Component

**Files:**
- Create: `src/lib/components/NotificationSettingsPanel.svelte`
- Create: `src/lib/components/__tests__/NotificationSettingsPanel.test.ts`

**Context:** A slide-out panel with three tabs (Global / Communities / Peers) for configuring notification policies. It receives a `NotificationService` instance and calls its setter methods when the user changes settings. The panel overlays the media feed area.

**Step 1: Write failing tests in `src/lib/components/__tests__/NotificationSettingsPanel.test.ts`**

```typescript
import { render, screen, fireEvent } from '@testing-library/svelte';
import { describe, it, expect, vi } from 'vitest';
import NotificationSettingsPanel from '../NotificationSettingsPanel.svelte';
import { NotificationService } from '../../notification-service';
import type { NavNode, Peer } from '../../types';

const mockPeers: Peer[] = [
  { address: 'alice-addr', displayName: 'Alice' },
  { address: 'bob-addr', displayName: 'Bob' },
];

const mockCommunities: NavNode[] = [
  {
    id: 'work',
    parentId: null,
    type: 'folder',
    name: 'Work',
    expanded: true,
    unreadCount: 0,
    unreadLevel: 'none',
  },
];

describe('NotificationSettingsPanel', () => {
  it('renders with Global tab active by default', () => {
    const svc = new NotificationService();
    render(NotificationSettingsPanel, {
      props: { service: svc, peers: mockPeers, communities: mockCommunities },
    });
    expect(screen.getByText('Global')).toBeTruthy();
    expect(screen.getByText('Quiet messages')).toBeTruthy();
    expect(screen.getByText('Standard messages')).toBeTruthy();
    expect(screen.getByText('Loud messages')).toBeTruthy();
  });

  it('displays current global policy values', () => {
    const svc = new NotificationService();
    render(NotificationSettingsPanel, {
      props: { service: svc, peers: mockPeers, communities: mockCommunities },
    });
    // Check that the select elements show defaults
    const selects = screen.getAllByRole('combobox');
    expect(selects).toHaveLength(3);
  });

  it('switches to Peers tab and shows peer list', async () => {
    const svc = new NotificationService();
    render(NotificationSettingsPanel, {
      props: { service: svc, peers: mockPeers, communities: mockCommunities },
    });
    const peersTab = screen.getByText('Peers');
    await fireEvent.click(peersTab);
    expect(screen.getByText('Alice')).toBeTruthy();
    expect(screen.getByText('Bob')).toBeTruthy();
  });

  it('switches to Communities tab and shows community list', async () => {
    const svc = new NotificationService();
    render(NotificationSettingsPanel, {
      props: { service: svc, peers: mockPeers, communities: mockCommunities },
    });
    const commTab = screen.getByText('Communities');
    await fireEvent.click(commTab);
    expect(screen.getByText('Work')).toBeTruthy();
  });

  it('calls onClose when close button is clicked', async () => {
    const svc = new NotificationService();
    const onClose = vi.fn();
    render(NotificationSettingsPanel, {
      props: { service: svc, peers: mockPeers, communities: mockCommunities, onClose },
    });
    const closeBtn = screen.getByLabelText('Close settings');
    await fireEvent.click(closeBtn);
    expect(onClose).toHaveBeenCalled();
  });
});
```

**Step 2: Run tests to verify they fail**

Run: `npx vitest run src/lib/components/__tests__/NotificationSettingsPanel.test.ts`
Expected: FAIL — module not found.

**Step 3: Implement `src/lib/components/NotificationSettingsPanel.svelte`**

```svelte
<script lang="ts">
  import type { NotificationAction, NotificationPolicy, NavNode, Peer } from '../types';
  import { NotificationService } from '../notification-service';

  let { service, peers, communities, onClose }: {
    service: NotificationService;
    peers: Peer[];
    communities: NavNode[];
    onClose?: () => void;
  } = $props();

  type Tab = 'global' | 'communities' | 'peers';
  let activeTab = $state<Tab>('global');

  const ACTION_LABELS: Record<NotificationAction, string> = {
    silent: 'Muted',
    dot_only: 'Dot only',
    notify: 'Notification',
    sound: 'Sound',
    break_dnd: 'Break DND',
  };

  const ACTIONS: NotificationAction[] = ['silent', 'dot_only', 'notify', 'sound', 'break_dnd'];

  const PRIORITY_LABELS = {
    quiet: 'Quiet messages',
    standard: 'Standard messages',
    loud: 'Loud messages',
  } as const;

  // Force reactivity on settings changes
  let version = $state(0);

  function getGlobal() {
    void version;
    return service.settings.global;
  }

  function setGlobalLevel(level: 'quiet' | 'standard' | 'loud', action: NotificationAction) {
    const current = service.settings.global;
    service.setGlobalPolicy({ ...current, [level]: action });
    version++;
  }

  function getPeerPolicy(address: string) {
    void version;
    return service.settings.perPeer.get(address);
  }

  function setPeerLevel(address: string, level: 'quiet' | 'standard' | 'loud', action: NotificationAction) {
    const current = service.settings.perPeer.get(address) ?? {};
    service.setPeerPolicy(address, { ...current, [level]: action });
    version++;
  }

  function clearPeer(address: string) {
    service.clearPeerPolicy(address);
    version++;
  }

  function getCommunityPolicy(id: string) {
    void version;
    return service.settings.perCommunity.get(id);
  }

  function setCommunityLevel(id: string, level: 'quiet' | 'standard' | 'loud', action: NotificationAction) {
    const current = service.settings.perCommunity.get(id) ?? {};
    service.setCommunityPolicy(id, { ...current, [level]: action });
    version++;
  }

  function clearCommunity(id: string) {
    service.clearCommunityPolicy(id);
    version++;
  }
</script>

<div class="settings-panel">
  <div class="settings-header">
    <h3>Notification Settings</h3>
    <button class="close-btn" aria-label="Close settings" onclick={() => onClose?.()}>✕</button>
  </div>

  <div class="tabs">
    <button class="tab {activeTab === 'global' ? 'active' : ''}" onclick={() => { activeTab = 'global'; }}>Global</button>
    <button class="tab {activeTab === 'communities' ? 'active' : ''}" onclick={() => { activeTab = 'communities'; }}>Communities</button>
    <button class="tab {activeTab === 'peers' ? 'active' : ''}" onclick={() => { activeTab = 'peers'; }}>Peers</button>
  </div>

  <div class="tab-content">
    {#if activeTab === 'global'}
      <div class="policy-rows">
        {#each (['quiet', 'standard', 'loud'] as const) as level}
          <div class="policy-row">
            <span class="policy-label">{PRIORITY_LABELS[level]}</span>
            <select
              value={getGlobal()[level]}
              onchange={(e) => setGlobalLevel(level, (e.target as HTMLSelectElement).value as NotificationAction)}
            >
              {#each ACTIONS as action}
                <option value={action}>{ACTION_LABELS[action]}</option>
              {/each}
            </select>
          </div>
        {/each}
      </div>

    {:else if activeTab === 'communities'}
      {#each communities as comm (comm.id)}
        <div class="override-section">
          <div class="override-header">
            <span class="override-name">{comm.name}</span>
            {#if getCommunityPolicy(comm.id)}
              <button class="reset-btn" onclick={() => clearCommunity(comm.id)}>Reset</button>
            {/if}
          </div>
          <div class="policy-rows">
            {#each (['quiet', 'standard', 'loud'] as const) as level}
              <div class="policy-row">
                <span class="policy-label">{PRIORITY_LABELS[level]}</span>
                <select
                  value={getCommunityPolicy(comm.id)?.[level] ?? ''}
                  onchange={(e) => {
                    const val = (e.target as HTMLSelectElement).value;
                    if (val) setCommunityLevel(comm.id, level, val as NotificationAction);
                  }}
                >
                  <option value="">Using global default</option>
                  {#each ACTIONS as action}
                    <option value={action}>{ACTION_LABELS[action]}</option>
                  {/each}
                </select>
              </div>
            {/each}
          </div>
        </div>
      {/each}

    {:else if activeTab === 'peers'}
      {#each peers as peer (peer.address)}
        <div class="override-section">
          <div class="override-header">
            <span class="override-name">{peer.displayName}</span>
            {#if getPeerPolicy(peer.address)}
              <button class="reset-btn" onclick={() => clearPeer(peer.address)}>Reset</button>
            {/if}
          </div>
          <div class="policy-rows">
            {#each (['quiet', 'standard', 'loud'] as const) as level}
              <div class="policy-row">
                <span class="policy-label">{PRIORITY_LABELS[level]}</span>
                <select
                  value={getPeerPolicy(peer.address)?.[level] ?? ''}
                  onchange={(e) => {
                    const val = (e.target as HTMLSelectElement).value;
                    if (val) setPeerLevel(peer.address, level, val as NotificationAction);
                  }}
                >
                  <option value="">Using default</option>
                  {#each ACTIONS as action}
                    <option value={action}>{ACTION_LABELS[action]}</option>
                  {/each}
                </select>
              </div>
            {/each}
          </div>
        </div>
      {/each}
    {/if}
  </div>
</div>

<style>
  .settings-panel {
    display: flex;
    flex-direction: column;
    height: 100%;
    background: var(--bg-secondary);
  }

  .settings-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 12px 16px;
    border-bottom: 1px solid var(--border);
  }

  .settings-header h3 {
    margin: 0;
    font-size: 14px;
    font-weight: 600;
    color: var(--text-primary);
  }

  .close-btn {
    border: none;
    background: none;
    color: var(--text-muted);
    font-size: 16px;
    cursor: pointer;
    padding: 4px;
  }

  .close-btn:hover {
    color: var(--text-primary);
  }

  .tabs {
    display: flex;
    border-bottom: 1px solid var(--border);
  }

  .tab {
    flex: 1;
    padding: 8px 12px;
    border: none;
    background: none;
    color: var(--text-muted);
    font-size: 13px;
    cursor: pointer;
    border-bottom: 2px solid transparent;
  }

  .tab.active {
    color: var(--text-primary);
    border-bottom-color: var(--accent);
  }

  .tab:hover {
    color: var(--text-secondary);
  }

  .tab-content {
    flex: 1;
    overflow-y: auto;
    padding: 12px 16px;
  }

  .policy-rows {
    display: flex;
    flex-direction: column;
    gap: 8px;
  }

  .policy-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
  }

  .policy-label {
    font-size: 13px;
    color: var(--text-secondary);
  }

  .policy-row select {
    padding: 4px 8px;
    border: 1px solid var(--border);
    border-radius: 4px;
    background: var(--bg-primary);
    color: var(--text-primary);
    font-size: 12px;
  }

  .override-section {
    margin-bottom: 16px;
    padding-bottom: 16px;
    border-bottom: 1px solid var(--border);
  }

  .override-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 8px;
  }

  .override-name {
    font-size: 13px;
    font-weight: 600;
    color: var(--text-primary);
  }

  .reset-btn {
    border: none;
    background: none;
    color: var(--text-muted);
    font-size: 11px;
    cursor: pointer;
    padding: 2px 6px;
  }

  .reset-btn:hover {
    color: var(--accent);
  }
</style>
```

**Step 4: Run tests to verify they pass**

Run: `npx vitest run src/lib/components/__tests__/NotificationSettingsPanel.test.ts`
Expected: All 5 tests PASS.

**Step 5: Run full test suite**

Run: `npx vitest run`
Expected: All tests pass.

**Step 6: Commit**

```bash
git add src/lib/components/NotificationSettingsPanel.svelte src/lib/components/__tests__/NotificationSettingsPanel.test.ts
git commit -m "feat: add NotificationSettingsPanel with global, community, and peer tabs"
```

---

### Task 8: Layout + App Integration

**Files:**
- Modify: `src/lib/components/Layout.svelte`
- Modify: `src/App.svelte`
- Modify: `src/lib/components/NavPanel.svelte:84-92` (add settings gear icon)
- Modify: `src/lib/mock-data.ts` (add mock NotificationService)

**Context:** Wire everything together. `Layout.svelte` gains a settings panel overlay slot. `App.svelte` instantiates the `NotificationService`, handles `onSend`, and manages the settings panel open/close state. The nav header gets a gear icon to open settings.

**Step 1: Add settings gear to `NavPanel.svelte`**

In `NavPanel.svelte`, add an `onSettingsClick` prop and a gear icon button in the nav header:

Add to props (after line 13):
```typescript
onSettingsClick?: () => void;
```

Replace the nav-header div (lines 85-92) with:
```svelte
<div class="nav-header">
  <input
    class="search-input"
    type="text"
    placeholder="Search"
    bind:value={searchQuery}
  />
  <button class="settings-btn" onclick={() => onSettingsClick?.()} aria-label="Notification settings">⚙</button>
</div>
```

Add CSS for the settings button:
```css
.nav-header {
  padding: 8px 12px;
  border-bottom: 1px solid var(--border);
  display: flex;
  gap: 8px;
  align-items: center;
}

.settings-btn {
  border: none;
  background: none;
  color: var(--text-muted);
  font-size: 16px;
  cursor: pointer;
  padding: 4px;
  flex-shrink: 0;
}

.settings-btn:hover {
  color: var(--text-primary);
}
```

**Step 2: Update `Layout.svelte` to support settings panel**

Add an optional `settingsPanel` snippet prop alongside the existing snippets:

```svelte
<script lang="ts">
  import type { Snippet } from 'svelte';

  let { nav, textFeed, mediaFeed, settingsPanel, collapsed = false, showSettings = false }: {
    nav: Snippet;
    textFeed: Snippet;
    mediaFeed: Snippet;
    settingsPanel?: Snippet;
    collapsed?: boolean;
    showSettings?: boolean;
  } = $props();
</script>

<div class="layout" class:collapsed>
  <aside class="nav-area">
    {@render nav()}
  </aside>
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
</div>
```

The CSS stays the same — the settings panel uses the media-area grid slot.

**Step 3: Update `App.svelte` to wire everything together**

```svelte
<script lang="ts">
  import './app.css';
  import Layout from './lib/components/Layout.svelte';
  import NavPanel from './lib/components/NavPanel.svelte';
  import TextFeed from './lib/components/TextFeed.svelte';
  import MediaFeed from './lib/components/MediaFeed.svelte';
  import NotificationSettingsPanel from './lib/components/NotificationSettingsPanel.svelte';
  import { NotificationService } from './lib/notification-service';
  import { messages, navNodes, peers } from './lib/mock-data';
  import type { MessagePriority } from './lib/types';

  let innerWidth = $state(window.innerWidth);
  let collapsed = $derived(innerWidth <= 768);
  let showSettings = $state(false);

  const notificationService = new NotificationService();

  // Mock per-peer override to demonstrate settings
  notificationService.setPeerPolicy('q7r8s9t0', { quiet: 'silent' });

  let allMessages = $state([...messages]);

  function scrollToMedia(mediaId: string) {
    document.getElementById(`media-${mediaId}`)?.scrollIntoView({
      behavior: 'smooth',
      block: 'center',
    });
  }

  function scrollToMessage(messageId: string) {
    const el = document.getElementById(`msg-${messageId}`);
    if (el) {
      el.scrollIntoView({ behavior: 'smooth', block: 'center' });
      el.classList.add('highlight');
      setTimeout(() => el.classList.remove('highlight'), 1500);
    }
  }

  function handleSend(text: string, priority: MessagePriority) {
    const newMsg = {
      id: `msg-${Date.now()}`,
      sender: { address: 'self', displayName: 'You' },
      text,
      timestamp: Date.now(),
      media: [],
      priority,
    };
    allMessages = [...allMessages, newMsg];
  }

  // Extract community nodes (folders) for settings panel
  let communities = $derived(navNodes.filter((n) => n.type === 'folder'));
</script>

<svelte:window bind:innerWidth />

<Layout {collapsed} {showSettings}>
  {#snippet nav()}
    <NavPanel nodes={navNodes} {collapsed} onSettingsClick={() => { showSettings = !showSettings; }} />
  {/snippet}
  {#snippet textFeed()}
    <TextFeed messages={allMessages} {collapsed} onMediaClick={scrollToMedia} onSend={handleSend} />
  {/snippet}
  {#snippet mediaFeed()}
    <MediaFeed messages={allMessages} onLinkBack={scrollToMessage} />
  {/snippet}
  {#snippet settingsPanel()}
    <NotificationSettingsPanel
      service={notificationService}
      peers={[...new Map(allMessages.map((m) => [m.sender.address, m.sender])).values()]}
      {communities}
      onClose={() => { showSettings = false; }}
    />
  {/snippet}
</Layout>

<style>
  :global(.text-message) {
    transition: background 0.3s ease;
  }

  :global(.text-message.highlight) {
    background: rgba(88, 101, 242, 0.15) !important;
  }
</style>
```

**Step 4: Export `peers` from `mock-data.ts`**

The `peers` array is already exported in `mock-data.ts` (line 3: `export const peers`). No change needed.

**Step 5: Run full test suite**

Run: `npx vitest run`
Expected: All tests pass.

**Step 6: Run build**

Run: `npm run build`
Expected: Build succeeds.

**Step 7: Commit**

```bash
git add src/App.svelte src/lib/components/Layout.svelte src/lib/components/NavPanel.svelte
git commit -m "feat: wire priority system into layout with settings panel toggle"
```

---

### Task 9: Visual Verification

**Files:** None — this is a manual check.

**Step 1: Start dev server**

Run: `npm run dev`

**Step 2: Verify in browser**

Check these behaviors:
- ComposeBar shows three priority icons (🔇 🔔 📢) with 🔔 active
- Clicking icons changes the active state
- Typing and pressing Enter sends a message (appears in feed)
- Ctrl+Enter sends as quiet
- "sending quietly" / "sending loudly" labels appear when not standard
- Quiet messages in the feed are collapsed ("🔇 N quiet messages from...")
- Clicking the collapsed row expands to show dimmed messages
- msg-08 ("Has anyone tested...") has a left accent border (loud)
- Gear icon in nav header opens the settings panel (replaces media feed)
- Settings panel has three tabs, dropdowns work
- Close button returns to media feed

**Step 3: Fix any visual issues**

If anything looks wrong, fix it and run tests again.

**Step 4: Commit any fixes**

```bash
git add -u
git commit -m "fix: visual polish for message priority UI"
```

---

### Summary

| Task | Component | Tests |
|------|-----------|-------|
| 1 | Types + mock data | Fix existing tests |
| 2 | NotificationService | 9 unit tests |
| 3 | groupMessages utility | 7 unit tests |
| 4 | QuietMessageGroup | 6 component tests |
| 5 | ComposeBar priority toggle | 9 component tests |
| 6 | TextFeed + TextMessage | 2 component tests |
| 7 | NotificationSettingsPanel | 5 component tests |
| 8 | Layout + App integration | Build verification |
| 9 | Visual verification | Manual check |

**Total: ~38 new tests across 6 test files.**
