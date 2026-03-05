# Thread View & Media Interleaving Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add threaded conversation support with split panel view, chronological media interleaving, floating thread roots, and three display modes (panel/inline/muted).

**Architecture:** Thread state (`openThreadId`, `threadModes`, `pinnedThreadIds`) lives in App.svelte. TextFeed splits internally when a panel-mode thread opens. MediaFeed receives combined messages and tags thread cards. FloatingThreadBar provides quick access to active threads.

**Tech Stack:** Svelte 5 (runes), TypeScript, vitest + @testing-library/svelte, CSS transitions

**Design doc:** `docs/plans/2026-03-05-thread-view-design.md`

**Repo:** `/Users/zeblith/work/zeblithic/harmony-client` — all file paths are relative to this root.

**Test command:** `npx vitest run`
**Build command:** `npm run build`
**Single test file:** `npx vitest run src/path/to/test.ts`

---

### Task 1: Data Model — Add replyTo and ThreadDisplayMode

**Files:**
- Modify: `src/lib/types.ts:53-63`

**Step 1: Add `replyTo` to Message interface and `ThreadDisplayMode` type**

In `src/lib/types.ts`, add `replyTo` to the `Message` interface and add the new type:

```typescript
// After line 62 (priority field), add:
  /** ID of the thread root message this is a reply to */
  replyTo?: string;
```

And after the `Message` interface (after line 63), add:

```typescript
export type ThreadDisplayMode = 'panel' | 'inline' | 'muted';
```

**Step 2: Run tests to verify nothing breaks**

Run: `npx vitest run`
Expected: All 134 tests pass (replyTo is optional, no existing code affected)

**Step 3: Run build to verify types compile**

Run: `npm run build`
Expected: Build succeeds

**Step 4: Commit**

```bash
git add src/lib/types.ts
git commit -m "feat: add replyTo field and ThreadDisplayMode type to Message"
```

---

### Task 2: Thread Helper Functions

**Files:**
- Modify: `src/lib/feed-utils.ts`
- Modify: `src/lib/feed-utils.test.ts`

**Step 1: Write failing tests for thread helpers**

Add to `src/lib/feed-utils.test.ts`:

```typescript
import { groupMessages, getThreadReplies, getThreadRoots, getThreadMeta } from './feed-utils';
import type { Message, Peer } from './types';

// Add after existing msg() helper:
function threadMsg(id: string, replyTo: string, sender = 'Alice', priority: 'quiet' | 'standard' | 'loud' = 'standard'): Message {
  return {
    id,
    sender: { address: sender.toLowerCase(), displayName: sender },
    text: `Reply ${id}`,
    timestamp: Date.now(),
    media: [],
    priority,
    replyTo,
  };
}

describe('getThreadReplies', () => {
  it('returns replies for a given root', () => {
    const messages = [msg('1'), threadMsg('2', '1'), threadMsg('3', '1'), msg('4')];
    const replies = getThreadReplies(messages, '1');
    expect(replies).toHaveLength(2);
    expect(replies[0].id).toBe('2');
    expect(replies[1].id).toBe('3');
  });

  it('returns empty array when no replies exist', () => {
    const messages = [msg('1'), msg('2')];
    expect(getThreadReplies(messages, '1')).toEqual([]);
  });
});

describe('getThreadRoots', () => {
  it('returns message IDs that have at least one reply', () => {
    const messages = [msg('1'), threadMsg('2', '1'), msg('3'), threadMsg('4', '3')];
    const roots = getThreadRoots(messages);
    expect(roots).toEqual(new Set(['1', '3']));
  });

  it('returns empty set when no threads exist', () => {
    const messages = [msg('1'), msg('2')];
    expect(getThreadRoots(messages)).toEqual(new Set());
  });
});

describe('getThreadMeta', () => {
  it('returns count and participants for each thread root', () => {
    const messages = [
      msg('1', 'standard', 'Alice'),
      threadMsg('2', '1', 'Bob'),
      threadMsg('3', '1', 'Carol'),
      threadMsg('4', '1', 'Bob'),
    ];
    const meta = getThreadMeta(messages);
    expect(meta.size).toBe(1);
    const entry = meta.get('1')!;
    expect(entry.count).toBe(3);
    expect(entry.participants).toHaveLength(2);
    expect(entry.participants.map(p => p.displayName)).toContain('Bob');
    expect(entry.participants.map(p => p.displayName)).toContain('Carol');
  });

  it('returns empty map when no threads', () => {
    expect(getThreadMeta([msg('1')])).toEqual(new Map());
  });
});
```

**Step 2: Run tests to verify they fail**

Run: `npx vitest run src/lib/feed-utils.test.ts`
Expected: FAIL — `getThreadReplies`, `getThreadRoots`, `getThreadMeta` not exported

**Step 3: Implement the helper functions**

Add to `src/lib/feed-utils.ts`:

```typescript
import type { Message, Peer } from './types';

// ... existing groupMessages stays unchanged ...

export function getThreadReplies(messages: Message[], rootId: string): Message[] {
  return messages.filter(m => m.replyTo === rootId);
}

export function getThreadRoots(messages: Message[]): Set<string> {
  const roots = new Set<string>();
  for (const m of messages) {
    if (m.replyTo) roots.add(m.replyTo);
  }
  return roots;
}

export interface ThreadMetaEntry {
  count: number;
  participants: Peer[];
}

export function getThreadMeta(messages: Message[]): Map<string, ThreadMetaEntry> {
  const meta = new Map<string, ThreadMetaEntry>();
  const roots = getThreadRoots(messages);
  for (const rootId of roots) {
    const replies = getThreadReplies(messages, rootId);
    const seen = new Set<string>();
    const participants: Peer[] = [];
    for (const r of replies) {
      if (!seen.has(r.sender.address)) {
        seen.add(r.sender.address);
        participants.push(r.sender);
      }
    }
    meta.set(rootId, { count: replies.length, participants });
  }
  return meta;
}
```

Note: Update the existing import line from `import type { Message } from './types';` to `import type { Message, Peer } from './types';`.

**Step 4: Run tests to verify they pass**

Run: `npx vitest run src/lib/feed-utils.test.ts`
Expected: All tests pass

**Step 5: Commit**

```bash
git add src/lib/feed-utils.ts src/lib/feed-utils.test.ts
git commit -m "feat: add thread helper functions — getThreadReplies, getThreadRoots, getThreadMeta"
```

---

### Task 3: Mock Data — Add Thread Conversations

**Files:**
- Modify: `src/lib/mock-data.ts:13-200`

**Step 1: Add thread replies to existing mock messages**

Add 5 thread reply messages to `src/lib/mock-data.ts`. These are replies to existing messages `msg-02` (Bob's PR link) and `msg-08` (Alice's interop question). Insert them chronologically in the `messages` array.

After `msg-02` (the PR link at `base + 5 * 60_000`), add two replies that occur later:

```typescript
  // Thread reply to msg-02 (Bob's PR)
  {
    id: 'msg-02-r1',
    sender: peers[2], // Carol
    text: 'Reviewed — looks good, left a few comments on the error handling.',
    timestamp: base + 8 * 60_000,
    media: [],
    priority: 'standard',
    replyTo: 'msg-02',
  },
  {
    id: 'msg-02-r2',
    sender: peers[0], // Alice
    text: 'Thanks Carol, I will address those today.',
    timestamp: base + 10 * 60_000,
    media: [],
    priority: 'standard',
    replyTo: 'msg-02',
  },
```

After `msg-08` (the interop question at `base + hour`), add three replies:

```typescript
  // Thread reply to msg-08 (interop question)
  {
    id: 'msg-08-r1',
    sender: peers[3], // Dave
    text: 'Running the full suite now, will post results shortly.',
    timestamp: base + hour + 2 * 60_000,
    media: [],
    priority: 'standard',
    replyTo: 'msg-08',
  },
  {
    id: 'msg-08-r2',
    sender: peers[2], // Carol
    text: 'I tested the identity path — byte-identical to Python.',
    timestamp: base + hour + 3 * 60_000,
    media: [
      {
        id: 'media-08',
        type: 'image',
        url: 'https://placehold.co/600x300/313338/f2f3f5?text=Interop+Test+Results',
        title: 'Identity derivation interop results',
      },
    ],
    priority: 'standard',
    replyTo: 'msg-08',
  },
  {
    id: 'msg-08-r3',
    sender: peers[0], // Alice
    text: 'Excellent — that confirms the HKDF path is correct too.',
    timestamp: base + hour + 4 * 60_000,
    media: [],
    priority: 'quiet',
    replyTo: 'msg-08',
  },
```

Insert these in the correct chronological positions within the existing array. The array must remain sorted by timestamp.

**Step 2: Run tests and build**

Run: `npx vitest run && npm run build`
Expected: All pass — mock data is only consumed at runtime

**Step 3: Commit**

```bash
git add src/lib/mock-data.ts
git commit -m "feat: add thread reply messages to mock data"
```

---

### Task 4: ThreadIndicator Component

**Files:**
- Create: `src/lib/components/ThreadIndicator.svelte`
- Create: `src/lib/components/__tests__/ThreadIndicator.test.ts`

**Step 1: Write failing tests**

Create `src/lib/components/__tests__/ThreadIndicator.test.ts`:

```typescript
import { render, screen } from '@testing-library/svelte';
import { describe, it, expect, vi } from 'vitest';
import ThreadIndicator from '../ThreadIndicator.svelte';
import type { Peer } from '../../types';

const participants: Peer[] = [
  { address: 'a', displayName: 'Alice' },
  { address: 'b', displayName: 'Bob' },
];

describe('ThreadIndicator', () => {
  it('shows reply count and participant names', () => {
    render(ThreadIndicator, {
      props: { count: 3, participants, rootId: 'root-1' },
    });
    expect(screen.getByText(/3 replies/)).toBeTruthy();
    expect(screen.getByText(/Alice, Bob/)).toBeTruthy();
  });

  it('calls onOpen when clicked', async () => {
    const onOpen = vi.fn();
    render(ThreadIndicator, {
      props: { count: 2, participants, rootId: 'root-1', onOpen },
    });
    screen.getByRole('button').click();
    expect(onOpen).toHaveBeenCalledWith('root-1');
  });

  it('shows singular reply for count of 1', () => {
    render(ThreadIndicator, {
      props: { count: 1, participants: [participants[0]], rootId: 'root-1' },
    });
    expect(screen.getByText(/1 reply/)).toBeTruthy();
  });

  it('truncates participant names beyond 3', () => {
    const many: Peer[] = [
      { address: 'a', displayName: 'Alice' },
      { address: 'b', displayName: 'Bob' },
      { address: 'c', displayName: 'Carol' },
      { address: 'd', displayName: 'Dave' },
    ];
    render(ThreadIndicator, {
      props: { count: 5, participants: many, rootId: 'root-1' },
    });
    expect(screen.getByText(/\+1 more/)).toBeTruthy();
  });

  it('has correct aria-label', () => {
    render(ThreadIndicator, {
      props: { count: 3, participants, rootId: 'root-1' },
    });
    const btn = screen.getByRole('button');
    expect(btn.getAttribute('aria-label')).toContain('3 replies');
    expect(btn.getAttribute('aria-label')).toContain('Alice');
  });
});
```

**Step 2: Run tests to verify they fail**

Run: `npx vitest run src/lib/components/__tests__/ThreadIndicator.test.ts`
Expected: FAIL — module not found

**Step 3: Implement ThreadIndicator.svelte**

Create `src/lib/components/ThreadIndicator.svelte`:

```svelte
<script lang="ts">
  import type { Peer } from '../types';
  import Avatar from './Avatar.svelte';

  let { count, participants, rootId, isOpen = false, onOpen }: {
    count: number;
    participants: Peer[];
    rootId: string;
    isOpen?: boolean;
    onOpen?: (rootId: string) => void;
  } = $props();

  let nameList = $derived(() => {
    const names = participants.slice(0, 3).map(p => p.displayName);
    const extra = participants.length - 3;
    if (extra > 0) names.push(`+${extra} more`);
    return names.join(', ');
  });

  let replyWord = $derived(count === 1 ? 'reply' : 'replies');

  let ariaLabel = $derived(
    `${count} ${replyWord} from ${participants.map(p => p.displayName).join(', ')}. Open thread.`
  );
</script>

<button
  class="thread-indicator"
  class:open={isOpen}
  onclick={() => onOpen?.(rootId)}
  aria-label={ariaLabel}
  aria-expanded={isOpen}
>
  <span class="thread-avatars">
    {#each participants.slice(0, 3) as participant (participant.address)}
      <Avatar
        address={participant.address}
        displayName={participant.displayName}
        avatarUrl={participant.avatarUrl}
        size={16}
      />
    {/each}
  </span>
  <span class="thread-info">
    💬 {count} {replyWord} · {nameList()}
  </span>
</button>

<style>
  .thread-indicator {
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 4px 8px;
    margin-top: 4px;
    border: none;
    border-radius: 4px;
    background: none;
    color: var(--text-muted);
    font-size: 12px;
    cursor: pointer;
  }

  .thread-indicator:hover {
    background: var(--bg-tertiary);
    color: var(--accent);
  }

  .thread-indicator.open {
    color: var(--accent);
  }

  .thread-avatars {
    display: flex;
    gap: -4px;
  }

  .thread-avatars > :global(.avatar) {
    margin-left: -4px;
  }

  .thread-avatars > :global(.avatar:first-child) {
    margin-left: 0;
  }

  .thread-info {
    white-space: nowrap;
  }
</style>
```

**Step 4: Run tests to verify they pass**

Run: `npx vitest run src/lib/components/__tests__/ThreadIndicator.test.ts`
Expected: All 5 tests pass

**Step 5: Commit**

```bash
git add src/lib/components/ThreadIndicator.svelte src/lib/components/__tests__/ThreadIndicator.test.ts
git commit -m "feat: add ThreadIndicator component with reply count and participant avatars"
```

---

### Task 5: Reply-To Header in TextMessage

**Files:**
- Modify: `src/lib/components/TextMessage.svelte`
- Modify: `src/lib/components/__tests__/TextMessage.test.ts`

**Step 1: Write failing tests for reply-to header**

Add to `src/lib/components/__tests__/TextMessage.test.ts`:

```typescript
it('shows reply-to header when replyTo is set', () => {
  const replyMsg: Message = {
    ...mockMessage,
    id: 'reply-1',
    replyTo: 'parent-1',
  };
  const parentMsg: Message = {
    ...mockMessage,
    id: 'parent-1',
    text: 'This is the parent message with some long text that should be truncated',
  };
  render(TextMessage, {
    props: {
      message: replyMsg,
      allMessages: [parentMsg, replyMsg],
      trustService: trustedService(),
    },
  });
  expect(screen.getByText(/↩/)).toBeTruthy();
  expect(screen.getByText(/Alice/)).toBeTruthy();
});

it('does not show reply-to header when replyTo is not set', () => {
  render(TextMessage, {
    props: { message: mockMessage, trustService: trustedService() },
  });
  expect(screen.queryByText(/↩/)).toBeNull();
});
```

**Step 2: Run tests to verify they fail**

Run: `npx vitest run src/lib/components/__tests__/TextMessage.test.ts`
Expected: FAIL — no reply-to header renders

**Step 3: Implement reply-to header in TextMessage.svelte**

Add `allMessages` and `onScrollToMessage` as optional props to TextMessage:

```typescript
let { message, collapsed = false, onMediaClick, onAvatarClick, trustService, trustVersion = 0, allMessages = [], onScrollToMessage }: {
  message: Message;
  collapsed?: boolean;
  onMediaClick?: (mediaId: string) => void;
  onAvatarClick?: (address: string, event: MouseEvent) => void;
  trustService?: TrustService;
  trustVersion?: number;
  allMessages?: Message[];
  onScrollToMessage?: (messageId: string) => void;
} = $props();
```

Add a derived for the parent message:

```typescript
let parentMessage = $derived(
  message.replyTo ? allMessages.find(m => m.id === message.replyTo) : undefined
);

let parentPreview = $derived(
  parentMessage ? parentMessage.text.slice(0, 50) + (parentMessage.text.length > 50 ? '...' : '') : ''
);
```

In the template, before `<div class="message-text">`, add:

```svelte
    {#if parentMessage}
      <button
        class="reply-to-header"
        onclick={() => onScrollToMessage?.(parentMessage.id)}
        aria-label="In reply to {parentMessage.sender.displayName}: {parentPreview}"
      >
        <span class="reply-to-icon">↩</span>
        <span class="reply-to-sender">{parentMessage.sender.displayName}</span>
        <span class="reply-to-text">{parentPreview}</span>
      </button>
    {/if}
```

Add styles:

```css
  .reply-to-header {
    display: flex;
    align-items: center;
    gap: 4px;
    padding: 2px 0;
    margin-bottom: 2px;
    border: none;
    background: none;
    color: var(--text-muted);
    font-size: 11px;
    cursor: pointer;
    text-align: left;
  }

  .reply-to-header:hover {
    color: var(--accent);
  }

  .reply-to-icon {
    font-size: 12px;
  }

  .reply-to-sender {
    font-weight: 600;
  }

  .reply-to-text {
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    max-width: 200px;
  }
```

**Step 4: Run tests to verify they pass**

Run: `npx vitest run src/lib/components/__tests__/TextMessage.test.ts`
Expected: All tests pass

**Step 5: Run full test suite**

Run: `npx vitest run`
Expected: All tests pass (new props are optional, backward compatible)

**Step 6: Commit**

```bash
git add src/lib/components/TextMessage.svelte src/lib/components/__tests__/TextMessage.test.ts
git commit -m "feat: add reply-to header in TextMessage for inline thread display"
```

---

### Task 6: ThreadView Component

**Files:**
- Create: `src/lib/components/ThreadView.svelte`
- Create: `src/lib/components/__tests__/ThreadView.test.ts`

**Step 1: Write failing tests**

Create `src/lib/components/__tests__/ThreadView.test.ts`:

```typescript
import { render, screen } from '@testing-library/svelte';
import { describe, it, expect, vi } from 'vitest';
import ThreadView from '../ThreadView.svelte';
import { TrustService } from '../../trust-service';
import type { Message } from '../../types';

function trustedService(): TrustService {
  const ts = new TrustService();
  ts.setGlobalTrust('trusted');
  return ts;
}

const rootMsg: Message = {
  id: 'root-1',
  sender: { address: 'alice', displayName: 'Alice' },
  text: 'Check this out',
  timestamp: Date.now(),
  media: [],
  priority: 'standard',
};

const replies: Message[] = [
  {
    id: 'reply-1',
    sender: { address: 'bob', displayName: 'Bob' },
    text: 'Looks great!',
    timestamp: Date.now() + 1000,
    media: [],
    priority: 'standard',
    replyTo: 'root-1',
  },
  {
    id: 'reply-2',
    sender: { address: 'carol', displayName: 'Carol' },
    text: 'Nice work',
    timestamp: Date.now() + 2000,
    media: [],
    priority: 'standard',
    replyTo: 'root-1',
  },
];

describe('ThreadView', () => {
  it('renders thread root message', () => {
    render(ThreadView, {
      props: { rootMessage: rootMsg, replies: [], trustService: trustedService() },
    });
    expect(screen.getByText('Check this out')).toBeTruthy();
    expect(screen.getByText('Alice')).toBeTruthy();
  });

  it('renders reply messages', () => {
    render(ThreadView, {
      props: { rootMessage: rootMsg, replies, trustService: trustedService() },
    });
    expect(screen.getByText('Looks great!')).toBeTruthy();
    expect(screen.getByText('Nice work')).toBeTruthy();
  });

  it('calls onClose when close button is clicked', () => {
    const onClose = vi.fn();
    render(ThreadView, {
      props: { rootMessage: rootMsg, replies: [], onClose, trustService: trustedService() },
    });
    const closeBtn = screen.getByLabelText('Close thread');
    closeBtn.click();
    expect(onClose).toHaveBeenCalled();
  });

  it('has correct aria role and label', () => {
    const { container } = render(ThreadView, {
      props: { rootMessage: rootMsg, replies: [], trustService: trustedService() },
    });
    const section = container.querySelector('[role="complementary"]');
    expect(section).toBeTruthy();
    expect(section?.getAttribute('aria-label')).toContain('Thread');
  });

  it('renders compose bar with reply placeholder', () => {
    render(ThreadView, {
      props: { rootMessage: rootMsg, replies: [], trustService: trustedService() },
    });
    const textarea = screen.getByPlaceholderText(/Reply in thread/);
    expect(textarea).toBeTruthy();
  });
});
```

**Step 2: Run tests to verify they fail**

Run: `npx vitest run src/lib/components/__tests__/ThreadView.test.ts`
Expected: FAIL — module not found

**Step 3: Implement ThreadView.svelte**

Create `src/lib/components/ThreadView.svelte`:

```svelte
<script lang="ts">
  import type { Message, MessagePriority } from '../types';
  import type { TrustService } from '../trust-service';
  import TextMessage from './TextMessage.svelte';
  import ComposeBar from './ComposeBar.svelte';

  let { rootMessage, replies, onClose, onSend, onAvatarClick, trustService, trustVersion = 0 }: {
    rootMessage: Message;
    replies: Message[];
    onClose?: () => void;
    onSend?: (text: string, priority: MessagePriority) => void;
    onAvatarClick?: (address: string, event: MouseEvent) => void;
    trustService?: TrustService;
    trustVersion?: number;
  } = $props();

  let allThreadMessages = $derived([rootMessage, ...replies]);

  let rootPreview = $derived(
    rootMessage.text.slice(0, 40) + (rootMessage.text.length > 40 ? '...' : '')
  );

  function handleKeyDown(e: KeyboardEvent) {
    if (e.key === 'Escape') {
      e.stopPropagation();
      onClose?.();
    }
  }
</script>

<div
  class="thread-view"
  role="complementary"
  aria-label="Thread: {rootPreview}"
  onkeydown={handleKeyDown}
>
  <div class="thread-header">
    <span class="thread-title">🧵 Thread</span>
    <button class="thread-close" onclick={() => onClose?.()} aria-label="Close thread">×</button>
  </div>

  <div class="thread-messages">
    <div class="thread-root">
      <TextMessage message={rootMessage} {onAvatarClick} {trustService} {trustVersion} />
    </div>

    {#each replies as reply (reply.id)}
      <TextMessage message={reply} {onAvatarClick} {trustService} {trustVersion} allMessages={allThreadMessages} />
    {/each}
  </div>

  <ComposeBar {onSend} channelName="Reply in thread" />
</div>

<style>
  .thread-view {
    display: flex;
    flex-direction: column;
    height: 100%;
    border-top: 1px solid var(--border);
  }

  .thread-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 8px 16px;
    border-bottom: 1px solid var(--border);
    flex-shrink: 0;
  }

  .thread-title {
    font-size: 13px;
    font-weight: 600;
    color: var(--text-primary);
  }

  .thread-close {
    border: none;
    background: none;
    color: var(--text-muted);
    font-size: 18px;
    cursor: pointer;
    padding: 2px 6px;
    border-radius: 4px;
  }

  .thread-close:hover {
    background: var(--bg-tertiary);
    color: var(--text-primary);
  }

  .thread-messages {
    flex: 1;
    overflow-y: auto;
    padding: 8px 0;
  }

  .thread-root {
    border-bottom: 1px solid var(--border);
    padding-bottom: 8px;
    margin-bottom: 4px;
    background: rgba(88, 101, 242, 0.04);
  }
</style>
```

**Step 4: Run tests to verify they pass**

Run: `npx vitest run src/lib/components/__tests__/ThreadView.test.ts`
Expected: All 5 tests pass

**Step 5: Commit**

```bash
git add src/lib/components/ThreadView.svelte src/lib/components/__tests__/ThreadView.test.ts
git commit -m "feat: add ThreadView component with pinned root, replies, and compose bar"
```

---

### Task 7: TextFeed Split Layout with Drag Handle

**Files:**
- Modify: `src/lib/components/TextFeed.svelte`

**Step 1: Add thread-related props and split layout**

Replace `src/lib/components/TextFeed.svelte` with:

```svelte
<script lang="ts">
  import type { Message, MessagePriority } from '../types';
  import type { TrustService } from '../trust-service';
  import type { ThreadMetaEntry } from '../feed-utils';
  import { groupMessages } from '../feed-utils';
  import TextMessage from './TextMessage.svelte';
  import QuietMessageGroup from './QuietMessageGroup.svelte';
  import ComposeBar from './ComposeBar.svelte';
  import ThreadView from './ThreadView.svelte';
  import ThreadIndicator from './ThreadIndicator.svelte';

  let {
    messages,
    collapsed = false,
    onMediaClick,
    onSend,
    onAvatarClick,
    trustService,
    trustVersion = 0,
    threadRoot = null,
    threadReplies = [],
    threadMeta = new Map(),
    openThreadId = null,
    onThreadOpen,
    onThreadClose,
    onThreadSend,
    onScrollToMessage,
  }: {
    messages: Message[];
    collapsed?: boolean;
    onMediaClick?: (mediaId: string) => void;
    onSend?: (text: string, priority: MessagePriority) => void;
    onAvatarClick?: (address: string, event: MouseEvent) => void;
    trustService?: TrustService;
    trustVersion?: number;
    threadRoot?: Message | null;
    threadReplies?: Message[];
    threadMeta?: Map<string, ThreadMetaEntry>;
    openThreadId?: string | null;
    onThreadOpen?: (rootId: string) => void;
    onThreadClose?: () => void;
    onThreadSend?: (text: string, priority: MessagePriority) => void;
    onScrollToMessage?: (messageId: string) => void;
  } = $props();

  let feedItems = $derived(groupMessages(messages));

  // Drag handle state
  let splitPercent = $state(60);
  let isDragging = $state(false);
  let containerEl: HTMLDivElement | undefined = $state();

  function handleDragStart(e: MouseEvent) {
    e.preventDefault();
    isDragging = true;
  }

  function handleDragMove(e: MouseEvent) {
    if (!isDragging || !containerEl) return;
    const rect = containerEl.getBoundingClientRect();
    const pct = ((e.clientY - rect.top) / rect.height) * 100;
    splitPercent = Math.max(20, Math.min(80, pct));
  }

  function handleDragEnd() {
    isDragging = false;
  }

  function handleDragKeyDown(e: KeyboardEvent) {
    if (e.key === 'ArrowUp') {
      e.preventDefault();
      splitPercent = Math.max(20, splitPercent - 5);
    } else if (e.key === 'ArrowDown') {
      e.preventDefault();
      splitPercent = Math.min(80, splitPercent + 5);
    }
  }

  let showThreadPanel = $derived(threadRoot !== null && openThreadId !== null);
</script>

<div
  class="text-feed"
  class:dragging={isDragging}
  bind:this={containerEl}
  onmousemove={handleDragMove}
  onmouseup={handleDragEnd}
  onmouseleave={handleDragEnd}
>
  <div class="main-section" style={showThreadPanel ? `flex-basis: ${splitPercent}%` : ''}>
    <div class="messages-scroll">
      {#each feedItems as item (item.kind === 'message' ? item.message.id : `quiet-${item.messages[0].id}`)}
        {#if item.kind === 'message'}
          <TextMessage
            message={item.message}
            {collapsed}
            {onMediaClick}
            {onAvatarClick}
            {trustService}
            {trustVersion}
            allMessages={messages}
            {onScrollToMessage}
          />
          {#if threadMeta.has(item.message.id)}
            {@const meta = threadMeta.get(item.message.id)!}
            <ThreadIndicator
              count={meta.count}
              participants={meta.participants}
              rootId={item.message.id}
              isOpen={openThreadId === item.message.id}
              onOpen={onThreadOpen}
            />
          {/if}
        {:else}
          <QuietMessageGroup messages={item.messages} {collapsed} {onMediaClick} {onAvatarClick} {trustService} {trustVersion} />
        {/if}
      {/each}
    </div>
    <ComposeBar {onSend} />
  </div>

  {#if showThreadPanel && threadRoot}
    <div
      class="drag-handle"
      role="separator"
      aria-orientation="horizontal"
      aria-valuenow={Math.round(splitPercent)}
      aria-label="Resize thread panel"
      tabindex="0"
      onmousedown={handleDragStart}
      onkeydown={handleDragKeyDown}
    ></div>
    <div class="thread-section">
      <ThreadView
        rootMessage={threadRoot}
        replies={threadReplies}
        onClose={onThreadClose}
        onSend={onThreadSend}
        {onAvatarClick}
        {trustService}
        {trustVersion}
      />
    </div>
  {/if}
</div>

<style>
  .text-feed {
    display: flex;
    flex-direction: column;
    height: 100%;
  }

  .main-section {
    display: flex;
    flex-direction: column;
    min-height: 0;
    flex: 1;
  }

  .messages-scroll {
    flex: 1;
    overflow-y: auto;
    padding: 8px 0;
  }

  .drag-handle {
    height: 4px;
    background: var(--border);
    cursor: row-resize;
    flex-shrink: 0;
    transition: background 0.15s;
  }

  .drag-handle:hover,
  .dragging .drag-handle {
    background: var(--accent);
  }

  .thread-section {
    flex: 1;
    min-height: 0;
    overflow: hidden;
  }

  .text-feed.dragging {
    user-select: none;
  }
</style>
```

**Step 2: Run tests to verify nothing breaks**

Run: `npx vitest run`
Expected: All existing tests pass (new props are optional with defaults)

**Step 3: Run build**

Run: `npm run build`
Expected: Build succeeds

**Step 4: Commit**

```bash
git add src/lib/components/TextFeed.svelte
git commit -m "feat: add split layout with drag handle to TextFeed for thread panel"
```

---

### Task 8: MediaFeed Thread Card Styling

**Files:**
- Modify: `src/lib/components/MediaFeed.svelte`
- Modify: `src/lib/components/__tests__/MediaFeed.test.ts`

**Step 1: Write failing test for thread card class**

Add to `src/lib/components/__tests__/MediaFeed.test.ts`:

```typescript
it('applies thread-card class to messages in threadMessageIds', () => {
  const trustService = new TrustService();
  trustService.setGlobalTrust('trusted');
  const threadIds = new Set(['img-1']);
  const { container } = render(MediaFeed, {
    props: {
      messages: messagesWithMedia,
      trustService,
      threadMessageIds: threadIds,
    },
  });
  const threadCards = container.querySelectorAll('.thread-card');
  expect(threadCards.length).toBeGreaterThan(0);
});

it('does not apply thread-card class when threadMessageIds is empty', () => {
  const trustService = new TrustService();
  trustService.setGlobalTrust('trusted');
  const { container } = render(MediaFeed, {
    props: {
      messages: messagesWithMedia,
      trustService,
    },
  });
  const threadCards = container.querySelectorAll('.thread-card');
  expect(threadCards.length).toBe(0);
});
```

Note: The test checks for `threadMessageIds` containing *attachment IDs* — but actually the design says thread cards are identified by **message ID**. The `threadMessageIds` set contains message IDs, and we need to check if the message owning the attachment is in the set. Update the test accordingly — use the message ID `msg-1` not attachment ID `img-1`:

```typescript
const threadIds = new Set(['msg-1']);
```

**Step 2: Run test to verify it fails**

Run: `npx vitest run src/lib/components/__tests__/MediaFeed.test.ts`
Expected: FAIL — no `.thread-card` class

**Step 3: Add threadMessageIds prop and wrapper to MediaFeed.svelte**

Add `threadMessageIds` prop:

```typescript
let { messages, trustService, trustVersion = 0, onLinkBack, onAvatarClick, onTrustChange, threadMessageIds = new Set() }: {
  // ... existing props ...
  threadMessageIds?: Set<string>;
} = $props();
```

In the template, wrap each card with a conditional thread-card div:

```svelte
{#each mediaItems as { message, attachment } (attachment.id)}
  <div class={threadMessageIds.has(message.id) ? 'thread-card' : ''}>
    <!-- communityId not yet available on Message; per-community trust will apply once content transport provides context -->
    {#if isTrustGated(attachment, message.sender.address)}
      <UntrustedMediaCard {message} {attachment} {onLinkBack} {onAvatarClick} onLoad={handleLoad} />
    {:else}
      <MediaCard {message} {attachment} {onLinkBack} {onAvatarClick} />
    {/if}
    {#if threadMessageIds.has(message.id)}
      <span class="thread-tag">🧵 in thread</span>
    {/if}
  </div>
{/each}
```

Add styles:

```css
  .thread-card {
    border-left: 3px solid var(--accent);
    border-radius: 0 8px 8px 0;
    position: relative;
  }

  .thread-tag {
    position: absolute;
    top: 8px;
    right: 12px;
    font-size: 11px;
    color: var(--accent);
    opacity: 0.7;
  }
```

**Step 4: Run tests to verify they pass**

Run: `npx vitest run src/lib/components/__tests__/MediaFeed.test.ts`
Expected: All tests pass

**Step 5: Commit**

```bash
git add src/lib/components/MediaFeed.svelte src/lib/components/__tests__/MediaFeed.test.ts
git commit -m "feat: add thread card styling to MediaFeed with border and tag"
```

---

### Task 9: App.svelte Thread State Wiring

**Files:**
- Modify: `src/App.svelte`

**Step 1: Add thread state and derived values**

Add after the existing `trustVersion` state (line 46):

```typescript
import type { MessagePriority, Profile, ThreadDisplayMode } from './lib/types';
import { getThreadMeta } from './lib/feed-utils';

// Thread state
let openThreadId = $state<string | null>(null);
let threadModes = $state<Map<string, ThreadDisplayMode>>(new Map());
let pinnedThreadIds = $state<Set<string>>(new Set());

// Thread derivations
let threadMeta = $derived(getThreadMeta(allMessages));

let threadRoot = $derived(
  openThreadId
    ? allMessages.find(m => m.id === openThreadId) ?? null
    : null
);

let threadReplies = $derived(
  openThreadId
    ? allMessages.filter(m => m.replyTo === openThreadId)
    : []
);

let threadMessageIds = $derived(
  openThreadId
    ? new Set(threadReplies.map(m => m.id))
    : new Set<string>()
);

// Main feed: exclude replies for panel/muted threads, keep inline
let mainFeedMessages = $derived(
  allMessages.filter(m => {
    if (!m.replyTo) return true;
    const mode = threadModes.get(m.replyTo) ?? 'panel';
    return mode === 'inline';
  })
);

// Media feed: main + open thread replies (exclude muted)
let mediaMessages = $derived.by(() => {
  const base = allMessages.filter(m => {
    if (!m.replyTo) return true;
    const mode = threadModes.get(m.replyTo) ?? 'panel';
    if (mode === 'muted') return false;
    if (mode === 'inline') return true;
    // panel mode: only include if this thread is open
    return m.replyTo === openThreadId;
  });
  return base;
});

function handleThreadOpen(rootId: string) {
  openThreadId = rootId;
}

function handleThreadClose() {
  openThreadId = null;
}

function handleThreadSend(text: string, priority: MessagePriority) {
  if (!openThreadId) return;
  const newMsg = {
    id: `msg-${Date.now()}`,
    sender: { address: 'self', displayName: 'You' },
    text,
    timestamp: Date.now(),
    media: [],
    priority,
    replyTo: openThreadId,
  };
  allMessages = [...allMessages, newMsg];
}
```

**Step 2: Update the template snippets**

Update the `textFeed` snippet to pass thread props:

```svelte
{#snippet textFeed()}
  <TextFeed
    messages={mainFeedMessages}
    {collapsed}
    onMediaClick={scrollToMedia}
    onSend={handleSend}
    onAvatarClick={handleAvatarClick}
    {trustService}
    {trustVersion}
    {threadRoot}
    {threadReplies}
    {threadMeta}
    {openThreadId}
    onThreadOpen={handleThreadOpen}
    onThreadClose={handleThreadClose}
    onThreadSend={handleThreadSend}
    onScrollToMessage={scrollToMessage}
  />
{/snippet}
```

Update the `mediaFeed` snippet to pass combined messages and thread IDs:

```svelte
{#snippet mediaFeed()}
  <MediaFeed
    messages={mediaMessages}
    {trustService}
    {trustVersion}
    onLinkBack={scrollToMessage}
    onAvatarClick={handleAvatarClick}
    onTrustChange={handleTrustChange}
    {threadMessageIds}
  />
{/snippet}
```

**Step 3: Run tests and build**

Run: `npx vitest run && npm run build`
Expected: All tests pass, build succeeds

**Step 4: Commit**

```bash
git add src/App.svelte
git commit -m "feat: wire thread state management through App to TextFeed and MediaFeed"
```

---

### Task 10: FloatingThreadBar Component

**Files:**
- Create: `src/lib/components/FloatingThreadBar.svelte`
- Create: `src/lib/components/__tests__/FloatingThreadBar.test.ts`

**Step 1: Write failing tests**

Create `src/lib/components/__tests__/FloatingThreadBar.test.ts`:

```typescript
import { render, screen } from '@testing-library/svelte';
import { describe, it, expect, vi } from 'vitest';
import FloatingThreadBar from '../FloatingThreadBar.svelte';
import type { Peer } from '../../types';
import type { ThreadMetaEntry } from '../../feed-utils';

const alice: Peer = { address: 'a', displayName: 'Alice' };
const bob: Peer = { address: 'b', displayName: 'Bob' };

function makeMeta(entries: [string, number, Peer[]][]): Map<string, ThreadMetaEntry> {
  return new Map(entries.map(([id, count, participants]) => [id, { count, participants }]));
}

const rootMessages = new Map<string, { sender: string; text: string }>([
  ['root-1', { sender: 'Alice', text: 'Check this out' }],
  ['root-2', { sender: 'Bob', text: 'API design discussion' }],
]);

describe('FloatingThreadBar', () => {
  it('renders nothing when no threads qualify', () => {
    const { container } = render(FloatingThreadBar, {
      props: {
        threadMeta: new Map(),
        pinnedThreadIds: new Set(),
        visibleThreadIds: new Set(),
        rootMessages,
      },
    });
    expect(container.querySelector('.floating-thread-bar')).toBeNull();
  });

  it('shows pinned threads regardless of visibility', () => {
    const meta = makeMeta([['root-1', 3, [alice, bob]]]);
    render(FloatingThreadBar, {
      props: {
        threadMeta: meta,
        pinnedThreadIds: new Set(['root-1']),
        visibleThreadIds: new Set(['root-1']), // visible but pinned
        rootMessages,
      },
    });
    expect(screen.getByText(/Alice/)).toBeTruthy();
    expect(screen.getByText(/📌/)).toBeTruthy();
  });

  it('shows auto-floated threads when scrolled out of view', () => {
    const meta = makeMeta([['root-1', 2, [alice]]]);
    render(FloatingThreadBar, {
      props: {
        threadMeta: meta,
        pinnedThreadIds: new Set(),
        visibleThreadIds: new Set(), // not visible
        rootMessages,
      },
    });
    expect(screen.getByText(/Alice/)).toBeTruthy();
  });

  it('does not show auto-floated threads that are visible', () => {
    const meta = makeMeta([['root-1', 2, [alice]]]);
    const { container } = render(FloatingThreadBar, {
      props: {
        threadMeta: meta,
        pinnedThreadIds: new Set(),
        visibleThreadIds: new Set(['root-1']), // visible
        rootMessages,
      },
    });
    expect(container.querySelector('.floating-thread-bar')).toBeNull();
  });

  it('calls onThreadOpen when a thread entry is clicked', () => {
    const onThreadOpen = vi.fn();
    const meta = makeMeta([['root-1', 2, [alice]]]);
    render(FloatingThreadBar, {
      props: {
        threadMeta: meta,
        pinnedThreadIds: new Set(),
        visibleThreadIds: new Set(),
        rootMessages,
        onThreadOpen,
      },
    });
    screen.getByRole('button').click();
    expect(onThreadOpen).toHaveBeenCalledWith('root-1');
  });
});
```

**Step 2: Run tests to verify they fail**

Run: `npx vitest run src/lib/components/__tests__/FloatingThreadBar.test.ts`
Expected: FAIL — module not found

**Step 3: Implement FloatingThreadBar.svelte**

Create `src/lib/components/FloatingThreadBar.svelte`:

```svelte
<script lang="ts">
  import type { ThreadMetaEntry } from '../feed-utils';

  let { threadMeta, pinnedThreadIds, visibleThreadIds, rootMessages, openThreadId = null, onThreadOpen }: {
    threadMeta: Map<string, ThreadMetaEntry>;
    pinnedThreadIds: Set<string>;
    visibleThreadIds: Set<string>;
    rootMessages: Map<string, { sender: string; text: string }>;
    openThreadId?: string | null;
    onThreadOpen?: (rootId: string) => void;
  } = $props();

  let entries = $derived.by(() => {
    const result: { id: string; sender: string; text: string; count: number; pinned: boolean }[] = [];
    const seen = new Set<string>();

    // Pinned threads always show
    for (const id of pinnedThreadIds) {
      const meta = threadMeta.get(id);
      const root = rootMessages.get(id);
      if (meta && root) {
        result.push({ id, sender: root.sender, text: root.text, count: meta.count, pinned: true });
        seen.add(id);
      }
    }

    // Auto-float: threads scrolled out of view, max 3
    let autoCount = 0;
    for (const [id, meta] of threadMeta) {
      if (autoCount >= 3) break;
      if (seen.has(id)) continue;
      if (visibleThreadIds.has(id)) continue;
      const root = rootMessages.get(id);
      if (!root) continue;
      result.push({ id, sender: root.sender, text: root.text, count: meta.count, pinned: false });
      autoCount++;
    }

    return result;
  });

  let hasEntries = $derived(entries.length > 0);
</script>

{#if hasEntries}
  <div class="floating-thread-bar">
    {#each entries as entry (entry.id)}
      <button
        class="thread-entry"
        class:active={openThreadId === entry.id}
        onclick={() => onThreadOpen?.(entry.id)}
      >
        {#if entry.pinned}<span class="pin-icon">📌</span>{/if}
        <span class="entry-sender">{entry.sender}</span>
        <span class="entry-text">{entry.text.slice(0, 30)}{entry.text.length > 30 ? '...' : ''}</span>
        <span class="entry-count">({entry.count})</span>
      </button>
    {/each}
  </div>
{/if}

<style>
  .floating-thread-bar {
    display: flex;
    gap: 4px;
    padding: 6px 12px;
    border-bottom: 1px solid var(--border);
    background: var(--bg-secondary);
    flex-shrink: 0;
    overflow-x: auto;
  }

  .thread-entry {
    display: flex;
    align-items: center;
    gap: 4px;
    padding: 4px 8px;
    border: 1px solid var(--border);
    border-radius: 4px;
    background: var(--bg-tertiary);
    color: var(--text-secondary);
    font-size: 11px;
    cursor: pointer;
    white-space: nowrap;
    flex-shrink: 0;
  }

  .thread-entry:hover {
    border-color: var(--accent);
    color: var(--accent);
  }

  .thread-entry.active {
    border-color: var(--accent);
    background: rgba(88, 101, 242, 0.1);
    color: var(--accent);
  }

  .entry-sender {
    font-weight: 600;
  }

  .entry-text {
    color: var(--text-muted);
    max-width: 120px;
    overflow: hidden;
    text-overflow: ellipsis;
  }

  .entry-count {
    color: var(--text-muted);
    font-size: 10px;
  }

  .pin-icon {
    font-size: 10px;
  }
</style>
```

**Step 4: Run tests to verify they pass**

Run: `npx vitest run src/lib/components/__tests__/FloatingThreadBar.test.ts`
Expected: All 5 tests pass

**Step 5: Commit**

```bash
git add src/lib/components/FloatingThreadBar.svelte src/lib/components/__tests__/FloatingThreadBar.test.ts
git commit -m "feat: add FloatingThreadBar for quick access to active threads"
```

---

### Task 11: Wire FloatingThreadBar into TextFeed

**Files:**
- Modify: `src/lib/components/TextFeed.svelte`
- Modify: `src/App.svelte`

**Step 1: Add FloatingThreadBar to TextFeed**

In `TextFeed.svelte`, import FloatingThreadBar and add the necessary props:

```typescript
import FloatingThreadBar from './FloatingThreadBar.svelte';
```

Add props for the floating bar:

```typescript
pinnedThreadIds = new Set(),
visibleThreadIds = new Set(),
```

Add the `rootMessages` derivation inside the script:

```typescript
let rootMessages = $derived.by(() => {
  const map = new Map<string, { sender: string; text: string }>();
  for (const [id, _meta] of threadMeta) {
    const rootMsg = messages.find(m => m.id === id);
    if (rootMsg) {
      map.set(id, { sender: rootMsg.sender.displayName, text: rootMsg.text });
    }
  }
  return map;
});
```

In the template, add `FloatingThreadBar` before the `messages-scroll` div inside the `main-section`:

```svelte
<div class="main-section" style={showThreadPanel ? `flex-basis: ${splitPercent}%` : ''}>
  <FloatingThreadBar
    {threadMeta}
    {pinnedThreadIds}
    {visibleThreadIds}
    {rootMessages}
    {openThreadId}
    onThreadOpen={onThreadOpen}
  />
  <div class="messages-scroll">
    <!-- ... existing content ... -->
  </div>
  <ComposeBar {onSend} />
</div>
```

**Step 2: Pass pinnedThreadIds from App.svelte**

In App.svelte, add `pinnedThreadIds` to the TextFeed snippet props:

```svelte
{pinnedThreadIds}
```

Note: `visibleThreadIds` requires IntersectionObserver which we will add in the next task. For now, pass an empty set — the bar will show all out-of-view threads (which in a test/dev context means all threads, since we can't observe scroll position yet).

**Step 3: Run tests and build**

Run: `npx vitest run && npm run build`
Expected: All pass

**Step 4: Commit**

```bash
git add src/lib/components/TextFeed.svelte src/App.svelte
git commit -m "feat: wire FloatingThreadBar into TextFeed layout"
```

---

### Task 12: Thread Media Card Exit Animation

**Files:**
- Modify: `src/lib/components/MediaFeed.svelte`

**Step 1: Add CSS transition classes for thread card exit**

Add to MediaFeed.svelte styles:

```css
  .thread-card {
    border-left: 3px solid var(--accent);
    border-radius: 0 8px 8px 0;
    position: relative;
    transition: opacity 0.2s ease, max-height 0.3s ease;
    overflow: hidden;
  }

  .thread-card.exiting {
    opacity: 0;
    max-height: 0 !important;
    padding: 0;
    margin: 0;
  }

  @media (prefers-reduced-motion: reduce) {
    .thread-card {
      transition: none;
    }
    .thread-card.exiting {
      display: none;
    }
  }
```

The exiting class will be managed by the parent — when `threadMessageIds` changes from containing items to empty, the parent component applies `.exiting` before removing items from the DOM. For the MVP, the simpler approach is: when `threadMessageIds` becomes empty, thread cards simply disappear. The CSS transition on `.thread-card` handles the visual (opacity on the card). This is sufficient for now — the fade happens naturally as the card is removed.

**Step 2: Run tests and build**

Run: `npx vitest run && npm run build`
Expected: All pass

**Step 3: Commit**

```bash
git add src/lib/components/MediaFeed.svelte
git commit -m "feat: add CSS transition for thread media card exit animation"
```

---

### Task 13: Keyboard Navigation

**Files:**
- Modify: `src/lib/components/TextFeed.svelte`

**Step 1: Add Esc handler at the TextFeed level**

The ThreadView already handles `Esc` internally. Add a global keyboard handler on TextFeed for `Esc` when the thread is open (as a fallback if focus is in the main section):

In TextFeed.svelte, add to the text-feed div:

```svelte
<svelte:window onkeydown={handleGlobalKeyDown} />
```

Add the handler in the script:

```typescript
function handleGlobalKeyDown(e: KeyboardEvent) {
  if (e.key === 'Escape' && showThreadPanel) {
    e.preventDefault();
    onThreadClose?.();
  }
}
```

Note: This is deliberately simple. The `Tab` override to bounce between compose bars is a polish item — standard browser Tab behavior already moves focus correctly between the two compose bars since they're in DOM order.

**Step 2: Run tests and build**

Run: `npx vitest run && npm run build`
Expected: All pass

**Step 3: Commit**

```bash
git add src/lib/components/TextFeed.svelte
git commit -m "feat: add Esc keyboard shortcut to close thread panel"
```

---

### Task 14: Final Integration Test

**Files:**
- Modify: `src/lib/components/__tests__/MediaFeed.test.ts` (if needed)

**Step 1: Run full test suite**

Run: `npx vitest run`
Expected: All tests pass

**Step 2: Run build**

Run: `npm run build`
Expected: Build succeeds with no errors

**Step 3: Verify mock data renders threads**

This is a manual verification step. The mock data from Task 3 should show:
- Thread indicators on `msg-02` (2 replies from Carol, Alice)
- Thread indicators on `msg-08` (3 replies from Dave, Carol, Alice)
- Clicking an indicator should open the thread panel
- Thread media (from `msg-08-r2`) should appear in the media feed with thread styling

**Step 4: Commit any remaining fixes**

If any integration issues are found during the final test run, fix and commit them.

```bash
git add -A
git commit -m "fix: final integration fixes for thread view"
```
