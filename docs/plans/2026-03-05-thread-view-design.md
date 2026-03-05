# Thread View & Media Interleaving Design

**Goal:** Add threaded conversation support to the Harmony client with a split
panel view, chronological media interleaving, and floating thread roots for
quick navigation.

**Bead:** harmony-46zr

**Architecture:** Thread state lives in App.svelte as `openThreadId` and
`threadModes`. TextFeed splits internally when a thread panel opens (no layout
changes). MediaFeed receives combined messages and tags thread cards visually.
Floating thread bar at the top of the main feed provides quick access to active
threads.

---

## 1. Data Model

Add an optional `replyTo` field to `Message`:

```typescript
export interface Message {
  // ...existing fields...
  replyTo?: string;  // message ID of the thread root
}
```

Threads are derived, not stored. A thread root is any message with at least one
reply. All replies point at the original root (flat threading, not nested).

Helper functions:
- `getThreadReplies(messages, rootId)` — filter replies for a given root
- `getThreadRoots(messages)` — messages that have at least one reply

---

## 2. Thread Display Modes

Three modes per thread root, stored as `Map<string, ThreadDisplayMode>`:

```typescript
type ThreadDisplayMode = 'panel' | 'inline' | 'muted';
```

| Mode       | Main feed                                      | Thread panel           | Media feed                                |
|----------- |----------------------------------------------- |----------------------- |------------------------------------------ |
| **panel**  | Hides replies, shows reply indicator on root   | Shows replies when open | Interleaves thread media when panel open  |
| **inline** | Shows replies in chronological position with reply-to header | N/A          | Thread media appears naturally            |
| **muted**  | Hides replies, shows muted indicator on root   | N/A                    | Thread media hidden                       |

Default mode (not in map) is `panel`.

---

## 3. Thread State Management

Thread state lives in App.svelte:

```typescript
let openThreadId = $state<string | null>(null);
let threadModes = $state<Map<string, ThreadDisplayMode>>(new Map());
let pinnedThreadIds = $state<Set<string>>(new Set());
```

Derived values:

```typescript
let threadMessages = $derived(
  openThreadId
    ? allMessages.filter(m => m.replyTo === openThreadId)
    : []
);

let threadRoot = $derived(
  openThreadId
    ? allMessages.find(m => m.id === openThreadId) ?? null
    : null
);

// Main feed: include replies for 'inline' threads only
let mainFeedMessages = $derived(
  allMessages.filter(m => {
    if (!m.replyTo) return true;
    const mode = threadModes.get(m.replyTo) ?? 'panel';
    return mode === 'inline';
  })
);

// Media feed: main + open thread's replies, exclude muted
let mediaMessages = $derived(
  openThreadId
    ? [...mainFeedMessages, ...threadMessages]
        .filter(m => {
          if (!m.replyTo) return true;
          return (threadModes.get(m.replyTo) ?? 'panel') !== 'muted';
        })
    : mainFeedMessages
);
```

---

## 4. TextFeed Split Layout

When a panel-mode thread is open, TextFeed splits into two flex sections:

```
┌─────────────────────────┐
│ FloatingThreadBar        │  ← only if threads qualify
├─────────────────────────┤
│ Main Feed (scrollable)  │  ← flex-basis from drag state
│ [Main ComposeBar]       │     labeled "# general" or similar
├─── drag handle (4px) ───┤
│ Thread View (scrollable)│  ← flex: 1
│ [Thread ComposeBar]     │     labeled "Reply in thread"
└─────────────────────────┘
```

- Default split: 60% main / 40% thread
- Drag handle: `mousedown` → `mousemove` → `mouseup`, adjusts flex-basis
- Both sections scroll independently
- When no thread panel is open, single section (unchanged)

### Compose context awareness

- Main compose bar: subtle label showing channel name
- Thread compose bar: labeled "Reply in thread"
- Active compose bar (focused) gets a left border accent highlight
- No ambiguity about where a message will land

### ThreadView component

New component `ThreadView.svelte`:
- Pinned thread root message at top (highlighted background)
- Thread replies rendered with TextMessage
- Close button (×) in header
- Own ComposeBar (sends with `replyTo` set to thread root)

---

## 5. Media Interleaving

MediaFeed receives the combined message array. Thread media cards slot into the
timeline by timestamp — true chronological mix, no grouping.

### Thread card styling

- 3px left border in accent color
- "🧵 in thread" tag in card header between sender name and timestamp
- Same MediaCard / UntrustedMediaCard components, conditional `.thread-card` CSS

### Animation on thread close

- Thread cards get `.thread-exiting` class
- CSS transition: `opacity 0.2s, max-height 0.3s`
- After transition, cards removed from DOM
- `prefers-reduced-motion: reduce` → instant removal

MediaFeed receives `threadMessageIds: Set<string>` prop to identify which cards
are thread cards.

---

## 6. Thread Discovery & Reply Indicators

### ThreadIndicator component

Shown below messages that have replies in the main feed:

```
Alice: check this out
  💬 3 replies · Carol, Dave
```

- Reply count + participant names (max 3, then "+N more")
- Participant mini avatars (16px), max 3 stacked
- Click opens thread panel
- Unread replies: bold text, highlighted count
- Context menu for mode switching (panel/inline/muted)

### Inline reply-to header

For messages with `replyTo` set (in inline mode), a small clickable header:

```
↩ Alice: "check this out..."
Bob: here's what I found
```

- Truncated to ~50 chars
- Click scrolls to parent message (existing scrollIntoView pattern)
- `aria-label="In reply to Alice: check this out"`

### Thread metadata derivation

App.svelte computes:

```typescript
let threadMeta = $derived(
  new Map(
    allMessages
      .filter(m => !m.replyTo)
      .filter(m => allMessages.some(r => r.replyTo === m.id))
      .map(m => {
        const replies = allMessages.filter(r => r.replyTo === m.id);
        const participants = [...new Set(replies.map(r => r.sender))];
        return [m.id, { count: replies.length, participants }];
      })
  )
);
```

---

## 7. Floating Thread Bar

Compact bar at the top of the main feed for quick thread access:

```
🧵 Alice: "check this out" (3) │ 📌 Bob: "API design" (7) │ Carol: "bug..." (2)
```

### Two sources

1. **Auto-floated (max 3):** Most recent thread roots with active replies that
   have scrolled out of view. Uses IntersectionObserver on thread root messages.
   Sorted by latest reply timestamp.

2. **User-pinned (unlimited):** Pinned via toggle on ThreadIndicator or thread
   panel header. Shown with 📌 icon. Always visible regardless of scroll.

### Behavior

- Each entry: compact button with icon + sender + truncated text + reply count
- Click opens that thread panel
- Currently open thread highlighted in bar
- Bar hidden when no threads qualify (zero overhead)

### Component

`FloatingThreadBar.svelte`:
- Props: `threadMeta`, `pinnedThreadIds`, `visibleThreadIds`, `openThreadId`,
  `onThreadOpen`
- Derives auto-floated: has replies, not visible, sorted by latest reply, take 3
- Merges with pinned, deduplicates

### Visibility tracking

TextFeed registers IntersectionObserver on thread root messages, maintains
`visibleThreadIds: Set<string>`. Only observed on messages with entries in
`threadMeta` (lightweight).

---

## 8. Keyboard Navigation & Accessibility

### Keyboard shortcuts

- `Esc` — closes thread panel, focus returns to main feed
- `Tab` — toggles focus between main and thread compose bars (when panel open)
- Arrow keys (Up/Down) on drag handle — adjust divider ±5%

### Accessibility

- Thread section: `role="complementary"`,
  `aria-label="Thread: {root text truncated}"`
- Drag handle: `role="separator"`, `aria-orientation="horizontal"`,
  `aria-valuenow` for split percentage, keyboard adjustable
- Thread indicator: `aria-label="N replies from X, Y. Open thread."`,
  `aria-expanded` reflecting panel state
- Reply-to header: `aria-label="In reply to Sender: text"`
- Thread media cards: "in thread" is visible text, no extra aria needed
- `prefers-reduced-motion: reduce` — skip all animations

---

## 9. Testing Strategy

### Unit tests (vitest + @testing-library/svelte)

- **Data helpers:** getThreadReplies, getThreadRoots, thread metadata derivation
- **ThreadView:** Renders root + replies, compose sends with replyTo, close fires callback
- **ThreadIndicator:** Reply count, participant names, click opens thread, aria
- **FloatingThreadBar:** Auto-float when scrolled out, pinned always shown, highlights open thread, empty state
- **TextMessage reply header:** Renders when replyTo set, click scrolls to parent
- **MediaFeed thread cards:** .thread-card class applied correctly
- **TextFeed split:** Single section when no thread, splits when thread active
- **Display modes:** Inline keeps replies in feed, muted filters out, panel hides from main

### Not tested

- Drag handle pixel positions (manual)
- CSS animations (visual)
- IntersectionObserver (mock callback logic only, jsdom limitation)

### Mock data

Extend existing mock messages with 2-3 thread conversations.
