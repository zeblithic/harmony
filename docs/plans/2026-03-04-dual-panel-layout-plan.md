# Dual-Panel Layout Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Scaffold the harmony-client Tauri v2 + Svelte 5 app and implement the foundational dual-panel layout (text feed + media feed) with responsive collapse.

**Architecture:** Standard Tauri v2 project layout with Svelte 5 frontend using runes for reactive state. CSS Grid powers the three-panel layout (nav, text feed, media feed) with a media query breakpoint at 768px for responsive collapse. Mock data drives the UI — no daemon integration yet.

**Tech Stack:** Tauri v2, Svelte 5 (runes), Vite, TypeScript, Vitest + @testing-library/svelte

**Target repo:** `zeblithic/harmony-client` (NOT the harmony repo — design docs live in harmony, code lives in harmony-client)

**Design doc:** `docs/plans/2026-03-04-dual-panel-layout-design.md` (in harmony repo)

**Note on project layout:** The approved design doc shows `crates/harmony-app/` but Tauri v2 convention uses `src-tauri/`. This plan uses the standard Tauri v2 layout (`src-tauri/`) to avoid fighting the framework. The `harmony-daemon` crate from the client design doc will be added as a workspace member later when daemon integration begins.

---

### Task 1: Scaffold Tauri v2 + Svelte 5 project

**Files:**
- Create: `harmony-client/package.json`
- Create: `harmony-client/vite.config.ts`
- Create: `harmony-client/tsconfig.json`
- Create: `harmony-client/svelte.config.js`
- Create: `harmony-client/src-tauri/Cargo.toml`
- Create: `harmony-client/src-tauri/tauri.conf.json`
- Create: `harmony-client/src-tauri/src/main.rs`
- Create: `harmony-client/src-tauri/src/lib.rs`
- Create: `harmony-client/src-tauri/build.rs`
- Create: `harmony-client/src/main.ts`
- Create: `harmony-client/src/app.html`
- Create: `harmony-client/src/App.svelte`
- Create: `harmony-client/src/vite-env.d.ts`

**Step 1: Initialize the Svelte + Vite frontend**

```bash
cd /Users/zeblith/work/zeblithic/harmony-client
npm create vite@latest . -- --template svelte-ts
```

If prompted about existing files, choose to overwrite (the repo only has docs + README + LICENSE — we'll restore those after).

**Step 2: Restore docs and license that may have been overwritten**

```bash
cd /Users/zeblith/work/zeblithic/harmony-client
git checkout -- docs/ LICENSE README.md 2>/dev/null || true
```

**Step 3: Install Svelte dependencies**

```bash
cd /Users/zeblith/work/zeblithic/harmony-client
npm install
```

**Step 4: Initialize Tauri v2 backend**

```bash
cd /Users/zeblith/work/zeblithic/harmony-client
npm install -D @tauri-apps/cli@latest
npx tauri init
```

When prompted:
- App name: `harmony-client`
- Window title: `Harmony`
- Web assets location: `../dist`
- Dev server URL: `http://localhost:5173`
- Frontend dev command: `npm run dev`
- Frontend build command: `npm run build`

**Step 5: Install Tauri API package**

```bash
cd /Users/zeblith/work/zeblithic/harmony-client
npm install @tauri-apps/api
```

**Step 6: Verify the scaffold builds**

```bash
cd /Users/zeblith/work/zeblithic/harmony-client
npm run dev &
sleep 3
kill %1
```

Expected: Vite dev server starts on port 5173 without errors.

```bash
cd /Users/zeblith/work/zeblithic/harmony-client/src-tauri
cargo check
```

Expected: Cargo check passes (Tauri Rust side compiles).

**Step 7: Commit**

```bash
cd /Users/zeblith/work/zeblithic/harmony-client
git add -A
git commit -m "feat: scaffold Tauri v2 + Svelte 5 + Vite project"
```

---

### Task 2: Dark theme and global styles

**Files:**
- Create: `harmony-client/src/app.css`

**Step 1: Write the global stylesheet with CSS custom properties**

```css
/* harmony-client/src/app.css */

:root {
  --bg-primary: #1e1f22;
  --bg-secondary: #2b2d31;
  --bg-tertiary: #313338;
  --text-primary: #f2f3f5;
  --text-secondary: #b5bac1;
  --text-muted: #949ba4;
  --accent: #5865f2;
  --accent-hover: #4752c4;
  --border: #3f4147;
  --avatar-size-micro: 24px;
  --avatar-size-mini: 20px;
  --nav-width: 240px;
  --nav-width-collapsed: 56px;
  --breakpoint-collapse: 768px;

  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen,
    Ubuntu, Cantarell, sans-serif;
  font-size: 14px;
  line-height: 1.4;
  color: var(--text-primary);
  background-color: var(--bg-primary);
}

*,
*::before,
*::after {
  box-sizing: border-box;
  margin: 0;
  padding: 0;
}

html, body {
  height: 100%;
  overflow: hidden;
}

::-webkit-scrollbar {
  width: 8px;
}

::-webkit-scrollbar-track {
  background: transparent;
}

::-webkit-scrollbar-thumb {
  background: var(--bg-tertiary);
  border-radius: 4px;
}

::-webkit-scrollbar-thumb:hover {
  background: var(--text-muted);
}
```

**Step 2: Import the stylesheet in the app entry point**

Modify `harmony-client/src/App.svelte`:

```svelte
<script lang="ts">
  import './app.css';
</script>

<main>
  <h1>Harmony</h1>
</main>
```

**Step 3: Verify dark theme renders**

```bash
cd /Users/zeblith/work/zeblithic/harmony-client
npm run dev
```

Open http://localhost:5173 — should show dark background (#1e1f22) with light text.

**Step 4: Commit**

```bash
cd /Users/zeblith/work/zeblithic/harmony-client
git add src/app.css src/App.svelte
git commit -m "feat: add dark theme and global CSS custom properties"
```

---

### Task 3: TypeScript types

**Files:**
- Create: `harmony-client/src/lib/types.ts`

**Step 1: Write the type definitions**

```typescript
// harmony-client/src/lib/types.ts

export interface Peer {
  address: string;
  displayName: string;
  avatarUrl?: string;
}

export interface MediaAttachment {
  id: string;
  type: 'image' | 'link' | 'code';
  /** URL for images and links */
  url?: string;
  /** OG title or filename */
  title?: string;
  /** Extracted domain for link indicators (e.g. "github.com") */
  domain?: string;
  /** Source code for code blocks */
  content?: string;
}

export interface Message {
  id: string;
  sender: Peer;
  text: string;
  /** Unix timestamp in milliseconds */
  timestamp: number;
  /** Empty array for text-only messages */
  media: MediaAttachment[];
}

export interface Channel {
  id: string;
  name: string;
  unreadCount: number;
}
```

**Step 2: Commit**

```bash
cd /Users/zeblith/work/zeblithic/harmony-client
git add src/lib/types.ts
git commit -m "feat: add TypeScript types for Message, Peer, MediaAttachment"
```

---

### Task 4: Mock data

**Files:**
- Create: `harmony-client/src/lib/mock-data.ts`

**Step 1: Write the mock data module**

```typescript
// harmony-client/src/lib/mock-data.ts

import type { Peer, Message, Channel } from './types';

export const peers: Peer[] = [
  { address: 'a1b2c3d4', displayName: 'Alice', avatarUrl: undefined },
  { address: 'e5f6g7h8', displayName: 'Bob', avatarUrl: undefined },
  { address: 'i9j0k1l2', displayName: 'Carol', avatarUrl: undefined },
  { address: 'm3n4o5p6', displayName: 'Dave', avatarUrl: undefined },
];

const hour = 3600_000;
const base = Date.now() - 4 * hour;

export const messages: Message[] = [
  {
    id: 'msg-01',
    sender: peers[0],
    text: 'Hey everyone, just pushed the new transport layer changes.',
    timestamp: base,
    media: [],
  },
  {
    id: 'msg-02',
    sender: peers[1],
    text: 'Nice! Here is the PR for review.',
    timestamp: base + 5 * 60_000,
    media: [
      {
        id: 'media-01',
        type: 'link',
        url: 'https://github.com/zeblithic/harmony/pull/35',
        title: 'feat: durable workflow execution engine',
        domain: 'github.com',
      },
    ],
  },
  {
    id: 'msg-03',
    sender: peers[2],
    text: 'Looking at the benchmarks, throughput is up 3x on the routing tier.',
    timestamp: base + 12 * 60_000,
    media: [
      {
        id: 'media-02',
        type: 'image',
        url: 'https://placehold.co/600x400/313338/f2f3f5?text=Benchmark+Chart',
        title: 'Routing throughput comparison',
      },
    ],
  },
  {
    id: 'msg-04',
    sender: peers[0],
    text: 'That looks great. The adaptive fuel scaling really helped.',
    timestamp: base + 15 * 60_000,
    media: [],
  },
  {
    id: 'msg-05',
    sender: peers[3],
    text: 'Here is the config I used for the starvation test:',
    timestamp: base + 20 * 60_000,
    media: [
      {
        id: 'media-03',
        type: 'code',
        title: 'tier_schedule.toml',
        content: `[tier_schedule]
router_max_per_tick = 10
storage_max_per_tick = 5
starvation_threshold = 8

[adaptive_compute]
high_water = 50
floor_fraction = 0.1`,
      },
    ],
  },
  {
    id: 'msg-06',
    sender: peers[1],
    text: 'I ran the same test with the W-TinyLFU cache enabled.',
    timestamp: base + 30 * 60_000,
    media: [],
  },
  {
    id: 'msg-07',
    sender: peers[2],
    text: 'Check out the cache hit rates — much better with the frequency sketch.',
    timestamp: base + 35 * 60_000,
    media: [
      {
        id: 'media-04',
        type: 'image',
        url: 'https://placehold.co/600x300/313338/f2f3f5?text=Cache+Hit+Rates',
        title: 'W-TinyLFU cache hit rate over time',
      },
    ],
  },
  {
    id: 'msg-08',
    sender: peers[0],
    text: 'Has anyone tested the Reticulum interop with the latest packet format changes?',
    timestamp: base + hour,
    media: [],
  },
  {
    id: 'msg-09',
    sender: peers[3],
    text: 'Yes, all 14 cross-language tests pass. Here is the test output.',
    timestamp: base + hour + 5 * 60_000,
    media: [
      {
        id: 'media-05',
        type: 'link',
        url: 'https://github.com/zeblithic/harmony/actions/runs/123456',
        title: 'CI: All interop tests passing',
        domain: 'github.com',
      },
    ],
  },
  {
    id: 'msg-10',
    sender: peers[1],
    text: 'Perfect. The identity derivation path is byte-identical to Python Reticulum now.',
    timestamp: base + hour + 10 * 60_000,
    media: [],
  },
  {
    id: 'msg-11',
    sender: peers[2],
    text: 'I documented the address derivation flow:',
    timestamp: base + hour + 20 * 60_000,
    media: [
      {
        id: 'media-06',
        type: 'code',
        title: 'address_derivation.rs',
        content: `// Address = SHA256(X25519_pub || Ed25519_pub)[:16]
let mut hasher = Sha256::new();
hasher.update(x25519_public.as_bytes());
hasher.update(ed25519_public.as_bytes());
let hash = hasher.finalize();
let address: [u8; 16] = hash[..16]
    .try_into()
    .expect("SHA256 output is 32 bytes");`,
      },
    ],
  },
  {
    id: 'msg-12',
    sender: peers[0],
    text: 'Clean. Next up is the Zenoh pub/sub integration for presence.',
    timestamp: base + 2 * hour,
    media: [],
  },
  {
    id: 'msg-13',
    sender: peers[3],
    text: 'I have a draft of the liveliness token flow.',
    timestamp: base + 2 * hour + 15 * 60_000,
    media: [
      {
        id: 'media-07',
        type: 'link',
        url: 'https://github.com/zeblithic/harmony/wiki/Liveliness-Tokens',
        title: 'Liveliness Token Design',
        domain: 'github.com',
      },
    ],
  },
  {
    id: 'msg-14',
    sender: peers[1],
    text: 'Looks solid. The key expression hierarchy makes sense for our namespace.',
    timestamp: base + 2 * hour + 25 * 60_000,
    media: [],
  },
  {
    id: 'msg-15',
    sender: peers[2],
    text: 'Agreed. Let us get this merged and start on the voice engine next.',
    timestamp: base + 3 * hour,
    media: [],
  },
];

export const channels: Channel[] = [
  { id: 'ch-general', name: 'general', unreadCount: 3 },
  { id: 'ch-crypto', name: 'crypto', unreadCount: 0 },
  { id: 'ch-transport', name: 'transport', unreadCount: 1 },
  { id: 'ch-voice', name: 'voice', unreadCount: 0 },
  { id: 'ch-testing', name: 'testing', unreadCount: 5 },
];
```

**Step 2: Commit**

```bash
cd /Users/zeblith/work/zeblithic/harmony-client
git add src/lib/mock-data.ts
git commit -m "feat: add mock data for messages, peers, and channels"
```

---

### Task 5: Layout shell with CSS Grid

**Files:**
- Create: `harmony-client/src/lib/components/Layout.svelte`
- Modify: `harmony-client/src/App.svelte`

**Step 1: Create the Layout component**

```svelte
<!-- harmony-client/src/lib/components/Layout.svelte -->
<script lang="ts">
  import type { Snippet } from 'svelte';

  let { nav, textFeed, mediaFeed }: {
    nav: Snippet;
    textFeed: Snippet;
    mediaFeed: Snippet;
  } = $props();

  let innerWidth = $state(window.innerWidth);
  let collapsed = $derived(innerWidth <= 768);
</script>

<svelte:window bind:innerWidth />

<div class="layout" class:collapsed>
  <aside class="nav-area">
    {@render nav()}
  </aside>
  <main class="text-area">
    {@render textFeed()}
  </main>
  {#if !collapsed}
    <section class="media-area">
      {@render mediaFeed()}
    </section>
  {/if}
</div>

<style>
  .layout {
    display: grid;
    grid-template-columns: var(--nav-width) 1fr 1fr;
    grid-template-areas: "nav text media";
    height: 100vh;
    overflow: hidden;
  }

  .layout.collapsed {
    grid-template-columns: var(--nav-width-collapsed) 1fr;
    grid-template-areas: "nav text";
  }

  .nav-area {
    grid-area: nav;
    background: var(--bg-secondary);
    border-right: 1px solid var(--border);
    overflow-y: auto;
  }

  .text-area {
    grid-area: text;
    background: var(--bg-primary);
    overflow-y: auto;
    display: flex;
    flex-direction: column;
  }

  .media-area {
    grid-area: media;
    background: var(--bg-secondary);
    border-left: 1px solid var(--border);
    overflow-y: auto;
    padding: 12px;
  }
</style>
```

**Step 2: Wire Layout into App.svelte**

```svelte
<!-- harmony-client/src/App.svelte -->
<script lang="ts">
  import './app.css';
  import Layout from './lib/components/Layout.svelte';
</script>

<Layout>
  {#snippet nav()}
    <div style="padding: 12px; color: var(--text-secondary);">Nav</div>
  {/snippet}
  {#snippet textFeed()}
    <div style="padding: 12px;">Text Feed</div>
  {/snippet}
  {#snippet mediaFeed()}
    <div style="padding: 12px;">Media Feed</div>
  {/snippet}
</Layout>
```

**Step 3: Verify the three-panel layout renders**

```bash
cd /Users/zeblith/work/zeblithic/harmony-client
npm run dev
```

Open http://localhost:5173 — should show three columns: dark nav sidebar (240px), main area ("Text Feed"), right panel ("Media Feed"). Resize the window below 768px — the media panel should disappear and the nav should narrow to 56px.

**Step 4: Commit**

```bash
cd /Users/zeblith/work/zeblithic/harmony-client
git add src/lib/components/Layout.svelte src/App.svelte
git commit -m "feat: add CSS Grid layout shell with responsive collapse"
```

---

### Task 6: NavPanel component

**Files:**
- Create: `harmony-client/src/lib/components/NavPanel.svelte`
- Modify: `harmony-client/src/App.svelte`

**Step 1: Create NavPanel**

```svelte
<!-- harmony-client/src/lib/components/NavPanel.svelte -->
<script lang="ts">
  import type { Channel } from '../types';

  let { channels, collapsed = false }: {
    channels: Channel[];
    collapsed: boolean;
  } = $props();
</script>

<div class="nav-panel">
  <div class="nav-header">
    {#if !collapsed}
      <h2>Harmony Dev</h2>
    {:else}
      <span class="nav-icon">H</span>
    {/if}
  </div>
  <nav class="channel-list">
    {#each channels as channel (channel.id)}
      <button class="channel-item" class:has-unread={channel.unreadCount > 0}>
        <span class="channel-hash">#</span>
        {#if !collapsed}
          <span class="channel-name">{channel.name}</span>
          {#if channel.unreadCount > 0}
            <span class="unread-badge">{channel.unreadCount}</span>
          {/if}
        {/if}
      </button>
    {/each}
  </nav>
</div>

<style>
  .nav-panel {
    display: flex;
    flex-direction: column;
    height: 100%;
  }

  .nav-header {
    padding: 12px 16px;
    border-bottom: 1px solid var(--border);
    min-height: 48px;
    display: flex;
    align-items: center;
  }

  .nav-header h2 {
    font-size: 15px;
    font-weight: 600;
    color: var(--text-primary);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }

  .nav-icon {
    font-size: 18px;
    font-weight: 700;
    color: var(--text-primary);
    width: 100%;
    text-align: center;
  }

  .channel-list {
    flex: 1;
    padding: 8px 0;
    overflow-y: auto;
  }

  .channel-item {
    display: flex;
    align-items: center;
    gap: 6px;
    width: 100%;
    padding: 6px 16px;
    border: none;
    background: none;
    color: var(--text-muted);
    font-size: 14px;
    cursor: pointer;
    text-align: left;
  }

  .channel-item:hover {
    background: var(--bg-tertiary);
    color: var(--text-primary);
  }

  .channel-item.has-unread {
    color: var(--text-primary);
    font-weight: 600;
  }

  .channel-hash {
    color: var(--text-muted);
    font-size: 16px;
    flex-shrink: 0;
  }

  .channel-name {
    flex: 1;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }

  .unread-badge {
    background: var(--accent);
    color: white;
    font-size: 11px;
    font-weight: 700;
    padding: 1px 6px;
    border-radius: 8px;
    flex-shrink: 0;
  }
</style>
```

**Step 2: Wire NavPanel into App.svelte**

Replace the nav snippet in `App.svelte`:

```svelte
<!-- harmony-client/src/App.svelte -->
<script lang="ts">
  import './app.css';
  import Layout from './lib/components/Layout.svelte';
  import NavPanel from './lib/components/NavPanel.svelte';
  import { channels } from './lib/mock-data';

  let innerWidth = $state(window.innerWidth);
  let collapsed = $derived(innerWidth <= 768);
</script>

<svelte:window bind:innerWidth />

<Layout>
  {#snippet nav()}
    <NavPanel {channels} {collapsed} />
  {/snippet}
  {#snippet textFeed()}
    <div style="padding: 12px;">Text Feed</div>
  {/snippet}
  {#snippet mediaFeed()}
    <div style="padding: 12px;">Media Feed</div>
  {/snippet}
</Layout>
```

**Step 3: Verify NavPanel renders with channel list**

```bash
cd /Users/zeblith/work/zeblithic/harmony-client
npm run dev
```

Expected: Left sidebar shows "Harmony Dev" header and channel list (#general, #crypto, etc.) with unread badges. Collapsing below 768px shows only `#` symbols.

**Step 4: Commit**

```bash
cd /Users/zeblith/work/zeblithic/harmony-client
git add src/lib/components/NavPanel.svelte src/App.svelte
git commit -m "feat: add NavPanel component with channel list and collapse"
```

---

### Task 7: Identicon avatar component

**Files:**
- Create: `harmony-client/src/lib/components/Avatar.svelte`

**Step 1: Create a deterministic identicon avatar**

This generates a simple color from the peer's address hash, used as a fallback when no avatar URL is set.

```svelte
<!-- harmony-client/src/lib/components/Avatar.svelte -->
<script lang="ts">
  let { address, size = 24, displayName = '' }: {
    address: string;
    size?: number;
    displayName?: string;
  } = $props();

  // Simple hash-to-color: pick a hue from the address string
  let hue = $derived(
    address.split('').reduce((acc, ch) => acc + ch.charCodeAt(0), 0) % 360
  );
  let bgColor = $derived(`hsl(${hue}, 45%, 45%)`);
  let initial = $derived(displayName.charAt(0).toUpperCase() || '?');
</script>

<div
  class="avatar"
  style="width: {size}px; height: {size}px; background: {bgColor}; font-size: {Math.round(size * 0.45)}px;"
  title={displayName}
>
  {initial}
</div>

<style>
  .avatar {
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    color: white;
    font-weight: 600;
    flex-shrink: 0;
    user-select: none;
  }
</style>
```

**Step 2: Commit**

```bash
cd /Users/zeblith/work/zeblithic/harmony-client
git add src/lib/components/Avatar.svelte
git commit -m "feat: add Avatar component with deterministic identicon"
```

---

### Task 8: TextMessage component

**Files:**
- Create: `harmony-client/src/lib/components/TextMessage.svelte`

**Step 1: Create the TextMessage component**

```svelte
<!-- harmony-client/src/lib/components/TextMessage.svelte -->
<script lang="ts">
  import type { Message } from '../types';
  import Avatar from './Avatar.svelte';

  let { message, collapsed = false, onMediaClick }: {
    message: Message;
    collapsed?: boolean;
    onMediaClick?: (mediaId: string) => void;
  } = $props();

  let timeStr = $derived(
    new Date(message.timestamp).toLocaleTimeString([], {
      hour: '2-digit',
      minute: '2-digit',
    })
  );
</script>

<div class="text-message" id="msg-{message.id}">
  <Avatar
    address={message.sender.address}
    displayName={message.sender.displayName}
    size={24}
  />
  <div class="message-content">
    <div class="message-header">
      <span class="sender-name">{message.sender.displayName}</span>
      <span class="timestamp">{timeStr}</span>
    </div>
    <div class="message-text">{message.text}</div>
    {#if message.media.length > 0}
      <div class="media-indicators">
        {#each message.media as attachment (attachment.id)}
          {#if collapsed}
            <!-- Inline embed in collapsed mode -->
            <div class="inline-embed">
              {#if attachment.type === 'image'}
                <img src={attachment.url} alt={attachment.title ?? 'image'} class="inline-image" />
              {:else if attachment.type === 'link'}
                <a href={attachment.url} class="inline-link" target="_blank" rel="noopener">
                  {attachment.title ?? attachment.url}
                </a>
              {:else if attachment.type === 'code'}
                <pre class="inline-code"><code>{attachment.content}</code></pre>
              {/if}
            </div>
          {:else}
            <!-- Indicator pill in full mode -->
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
        {/each}
      </div>
    {/if}
  </div>
</div>

<style>
  .text-message {
    display: flex;
    gap: 12px;
    padding: 4px 16px;
    scroll-margin-top: 8px;
  }

  .text-message:hover {
    background: var(--bg-secondary);
  }

  .message-content {
    flex: 1;
    min-width: 0;
  }

  .message-header {
    display: flex;
    align-items: baseline;
    gap: 8px;
  }

  .sender-name {
    font-weight: 600;
    font-size: 14px;
    color: var(--text-primary);
  }

  .timestamp {
    font-size: 11px;
    color: var(--text-muted);
  }

  .message-text {
    color: var(--text-secondary);
    font-size: 14px;
    word-wrap: break-word;
  }

  .media-indicators {
    display: flex;
    flex-wrap: wrap;
    gap: 4px;
    margin-top: 4px;
  }

  .media-pill {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    padding: 2px 8px;
    border: none;
    border-radius: 10px;
    background: var(--bg-tertiary);
    color: var(--text-muted);
    font-size: 12px;
    cursor: pointer;
  }

  .media-pill:hover {
    color: var(--accent);
    background: var(--bg-secondary);
  }

  .pill-icon {
    font-size: 12px;
  }

  .inline-embed {
    margin-top: 8px;
  }

  .inline-image {
    max-width: 100%;
    max-height: 300px;
    border-radius: 8px;
  }

  .inline-link {
    color: var(--accent);
    text-decoration: none;
  }

  .inline-link:hover {
    text-decoration: underline;
  }

  .inline-code {
    background: var(--bg-tertiary);
    padding: 8px 12px;
    border-radius: 6px;
    font-size: 13px;
    overflow-x: auto;
    color: var(--text-secondary);
  }
</style>
```

**Step 2: Commit**

```bash
cd /Users/zeblith/work/zeblithic/harmony-client
git add src/lib/components/TextMessage.svelte
git commit -m "feat: add TextMessage component with media indicators and inline embeds"
```

---

### Task 9: ComposeBar and TextFeed components

**Files:**
- Create: `harmony-client/src/lib/components/ComposeBar.svelte`
- Create: `harmony-client/src/lib/components/TextFeed.svelte`

**Step 1: Create ComposeBar (input stub)**

```svelte
<!-- harmony-client/src/lib/components/ComposeBar.svelte -->
<script lang="ts">
  let draft = $state('');
</script>

<div class="compose-bar">
  <input
    type="text"
    class="compose-input"
    placeholder="Message #general"
    bind:value={draft}
  />
</div>

<style>
  .compose-bar {
    padding: 12px 16px;
    border-top: 1px solid var(--border);
    flex-shrink: 0;
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
  }

  .compose-input::placeholder {
    color: var(--text-muted);
  }

  .compose-input:focus {
    box-shadow: 0 0 0 2px var(--accent);
  }
</style>
```

**Step 2: Create TextFeed**

```svelte
<!-- harmony-client/src/lib/components/TextFeed.svelte -->
<script lang="ts">
  import type { Message } from '../types';
  import TextMessage from './TextMessage.svelte';
  import ComposeBar from './ComposeBar.svelte';

  let { messages, collapsed = false, onMediaClick }: {
    messages: Message[];
    collapsed?: boolean;
    onMediaClick?: (mediaId: string) => void;
  } = $props();
</script>

<div class="text-feed">
  <div class="messages-scroll">
    {#each messages as message (message.id)}
      <TextMessage {message} {collapsed} {onMediaClick} />
    {/each}
  </div>
  <ComposeBar />
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

**Step 3: Commit**

```bash
cd /Users/zeblith/work/zeblithic/harmony-client
git add src/lib/components/ComposeBar.svelte src/lib/components/TextFeed.svelte
git commit -m "feat: add TextFeed and ComposeBar components"
```

---

### Task 10: MediaCard and MediaFeed components

**Files:**
- Create: `harmony-client/src/lib/components/MediaCard.svelte`
- Create: `harmony-client/src/lib/components/MediaFeed.svelte`

**Step 1: Create MediaCard**

```svelte
<!-- harmony-client/src/lib/components/MediaCard.svelte -->
<script lang="ts">
  import type { Message, MediaAttachment } from '../types';
  import Avatar from './Avatar.svelte';

  let { message, attachment, onLinkBack }: {
    message: Message;
    attachment: MediaAttachment;
    onLinkBack?: (messageId: string) => void;
  } = $props();

  let timeStr = $derived(
    new Date(message.timestamp).toLocaleTimeString([], {
      hour: '2-digit',
      minute: '2-digit',
    })
  );
</script>

<div class="media-card" id="media-{attachment.id}">
  <button class="card-header" onclick={() => onLinkBack?.(message.id)}>
    <Avatar
      address={message.sender.address}
      displayName={message.sender.displayName}
      size={20}
    />
    <span class="card-sender">{message.sender.displayName}</span>
    <span class="card-time">{timeStr}</span>
    <span class="link-back-icon" title="Jump to message">&#8599;</span>
  </button>

  <div class="card-content">
    {#if attachment.type === 'image'}
      <img
        src={attachment.url}
        alt={attachment.title ?? 'image'}
        class="card-image"
        loading="lazy"
      />
      {#if attachment.title}
        <p class="card-caption">{attachment.title}</p>
      {/if}
    {:else if attachment.type === 'link'}
      <a href={attachment.url} class="card-link" target="_blank" rel="noopener">
        <div class="link-preview">
          <div class="link-title">{attachment.title ?? attachment.url}</div>
          {#if attachment.domain}
            <div class="link-domain">{attachment.domain}</div>
          {/if}
        </div>
      </a>
    {:else if attachment.type === 'code'}
      <div class="code-block">
        {#if attachment.title}
          <div class="code-filename">{attachment.title}</div>
        {/if}
        <pre><code>{attachment.content}</code></pre>
      </div>
    {/if}
  </div>
</div>

<style>
  .media-card {
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

  .card-content {
    padding: 0 12px 12px;
  }

  .card-image {
    width: 100%;
    border-radius: 4px;
    display: block;
  }

  .card-caption {
    margin-top: 6px;
    font-size: 12px;
    color: var(--text-muted);
  }

  .card-link {
    display: block;
    text-decoration: none;
    color: inherit;
  }

  .link-preview {
    border-left: 3px solid var(--accent);
    padding: 8px 12px;
    border-radius: 0 4px 4px 0;
    background: rgba(0, 0, 0, 0.15);
  }

  .link-title {
    color: var(--accent);
    font-size: 14px;
    font-weight: 500;
  }

  .link-domain {
    color: var(--text-muted);
    font-size: 12px;
    margin-top: 2px;
  }

  .card-link:hover .link-title {
    text-decoration: underline;
  }

  .code-block {
    background: rgba(0, 0, 0, 0.2);
    border-radius: 4px;
    overflow: hidden;
  }

  .code-filename {
    padding: 6px 12px;
    font-size: 12px;
    color: var(--text-muted);
    border-bottom: 1px solid var(--border);
    font-family: monospace;
  }

  .code-block pre {
    padding: 12px;
    margin: 0;
    font-size: 13px;
    line-height: 1.5;
    overflow-x: auto;
    color: var(--text-secondary);
  }
</style>
```

**Step 2: Create MediaFeed**

```svelte
<!-- harmony-client/src/lib/components/MediaFeed.svelte -->
<script lang="ts">
  import type { Message } from '../types';
  import MediaCard from './MediaCard.svelte';

  let { messages, onLinkBack }: {
    messages: Message[];
    onLinkBack?: (messageId: string) => void;
  } = $props();

  // Flatten messages into a list of (message, attachment) pairs, chronologically
  let mediaItems = $derived(
    messages
      .filter((m) => m.media.length > 0)
      .flatMap((m) => m.media.map((a) => ({ message: m, attachment: a })))
  );
</script>

<div class="media-feed">
  {#if mediaItems.length === 0}
    <div class="empty-state">No media yet</div>
  {:else}
    {#each mediaItems as { message, attachment } (attachment.id)}
      <MediaCard {message} {attachment} {onLinkBack} />
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

**Step 3: Commit**

```bash
cd /Users/zeblith/work/zeblithic/harmony-client
git add src/lib/components/MediaCard.svelte src/lib/components/MediaFeed.svelte
git commit -m "feat: add MediaCard and MediaFeed components"
```

---

### Task 11: Wire everything together in App.svelte

**Files:**
- Modify: `harmony-client/src/App.svelte`

**Step 1: Update App.svelte to compose all components**

```svelte
<!-- harmony-client/src/App.svelte -->
<script lang="ts">
  import './app.css';
  import Layout from './lib/components/Layout.svelte';
  import NavPanel from './lib/components/NavPanel.svelte';
  import TextFeed from './lib/components/TextFeed.svelte';
  import MediaFeed from './lib/components/MediaFeed.svelte';
  import { messages, channels } from './lib/mock-data';

  let innerWidth = $state(window.innerWidth);
  let collapsed = $derived(innerWidth <= 768);

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
</script>

<svelte:window bind:innerWidth />

<Layout>
  {#snippet nav()}
    <NavPanel {channels} {collapsed} />
  {/snippet}
  {#snippet textFeed()}
    <TextFeed {messages} {collapsed} onMediaClick={scrollToMedia} />
  {/snippet}
  {#snippet mediaFeed()}
    <MediaFeed {messages} onLinkBack={scrollToMessage} />
  {/snippet}
</Layout>

<style>
  :global(.text-message.highlight) {
    background: rgba(88, 101, 242, 0.15) !important;
    transition: background 0.3s ease;
  }
</style>
```

**Step 2: Verify the full app**

```bash
cd /Users/zeblith/work/zeblithic/harmony-client
npm run dev
```

Expected:
- Three-panel layout with nav, text feed, and media feed
- Text feed shows 15 messages with avatars, names, timestamps
- Messages with media show indicator pills (image/link/code icons)
- Media feed shows 7 cards (images, links, code blocks)
- Clicking a media pill in the text feed scrolls to the card in the media panel
- Clicking a card header scrolls to and highlights the message in the text feed
- Resize below 768px: media panel disappears, indicators become inline embeds, nav collapses

**Step 3: Commit**

```bash
cd /Users/zeblith/work/zeblithic/harmony-client
git add src/App.svelte
git commit -m "feat: wire dual-panel layout with link-back anchoring"
```

---

### Task 12: Set up testing infrastructure

**Files:**
- Create: `harmony-client/src/lib/components/__tests__/TextMessage.test.ts`
- Modify: `harmony-client/package.json` (add test deps)
- Create: `harmony-client/vitest.config.ts`

**Step 1: Install test dependencies**

```bash
cd /Users/zeblith/work/zeblithic/harmony-client
npm install -D vitest @testing-library/svelte @testing-library/jest-dom jsdom
```

**Step 2: Create vitest config**

```typescript
// harmony-client/vitest.config.ts
import { defineConfig } from 'vitest/config';
import { svelte } from '@sveltejs/vite-plugin-svelte';

export default defineConfig({
  plugins: [svelte({ hot: !process.env.VITEST })],
  test: {
    environment: 'jsdom',
    globals: true,
    include: ['src/**/*.test.ts'],
  },
});
```

**Step 3: Write TextMessage component tests**

```typescript
// harmony-client/src/lib/components/__tests__/TextMessage.test.ts
import { render, screen } from '@testing-library/svelte';
import { describe, it, expect, vi } from 'vitest';
import TextMessage from '../TextMessage.svelte';
import type { Message } from '../../types';

const mockMessage: Message = {
  id: 'test-1',
  sender: { address: 'abc123', displayName: 'Alice', avatarUrl: undefined },
  text: 'Hello world',
  timestamp: new Date('2026-03-04T12:00:00Z').getTime(),
  media: [],
};

const mockMessageWithMedia: Message = {
  id: 'test-2',
  sender: { address: 'def456', displayName: 'Bob', avatarUrl: undefined },
  text: 'Check this out',
  timestamp: new Date('2026-03-04T12:05:00Z').getTime(),
  media: [
    {
      id: 'media-1',
      type: 'link',
      url: 'https://example.com',
      title: 'Example Site',
      domain: 'example.com',
    },
  ],
};

describe('TextMessage', () => {
  it('renders sender name and message text', () => {
    render(TextMessage, { props: { message: mockMessage } });
    expect(screen.getByText('Alice')).toBeTruthy();
    expect(screen.getByText('Hello world')).toBeTruthy();
  });

  it('shows media indicator pill when not collapsed', () => {
    render(TextMessage, {
      props: { message: mockMessageWithMedia, collapsed: false },
    });
    expect(screen.getByText('example.com')).toBeTruthy();
  });

  it('shows inline embed when collapsed', () => {
    render(TextMessage, {
      props: { message: mockMessageWithMedia, collapsed: true },
    });
    expect(screen.getByText('Example Site')).toBeTruthy();
  });

  it('calls onMediaClick when pill is clicked', async () => {
    const onClick = vi.fn();
    render(TextMessage, {
      props: {
        message: mockMessageWithMedia,
        collapsed: false,
        onMediaClick: onClick,
      },
    });
    const pill = screen.getByText('example.com').closest('button');
    pill?.click();
    expect(onClick).toHaveBeenCalledWith('media-1');
  });
});
```

**Step 4: Run the tests**

```bash
cd /Users/zeblith/work/zeblithic/harmony-client
npx vitest run
```

Expected: 4 tests pass.

**Step 5: Commit**

```bash
cd /Users/zeblith/work/zeblithic/harmony-client
git add vitest.config.ts src/lib/components/__tests__/ package.json package-lock.json
git commit -m "test: add TextMessage component tests with vitest"
```

---

### Task 13: MediaFeed tests

**Files:**
- Create: `harmony-client/src/lib/components/__tests__/MediaFeed.test.ts`

**Step 1: Write MediaFeed component tests**

```typescript
// harmony-client/src/lib/components/__tests__/MediaFeed.test.ts
import { render, screen } from '@testing-library/svelte';
import { describe, it, expect } from 'vitest';
import MediaFeed from '../MediaFeed.svelte';
import type { Message } from '../../types';

const messagesWithMedia: Message[] = [
  {
    id: 'msg-1',
    sender: { address: 'abc', displayName: 'Alice' },
    text: 'Image here',
    timestamp: Date.now(),
    media: [{ id: 'img-1', type: 'image', url: 'https://example.com/img.png', title: 'Screenshot' }],
  },
  {
    id: 'msg-2',
    sender: { address: 'def', displayName: 'Bob' },
    text: 'No media',
    timestamp: Date.now(),
    media: [],
  },
  {
    id: 'msg-3',
    sender: { address: 'ghi', displayName: 'Carol' },
    text: 'Link share',
    timestamp: Date.now(),
    media: [{ id: 'link-1', type: 'link', url: 'https://example.com', title: 'Example', domain: 'example.com' }],
  },
];

describe('MediaFeed', () => {
  it('renders only messages that have media', () => {
    render(MediaFeed, { props: { messages: messagesWithMedia } });
    expect(screen.getByText('Alice')).toBeTruthy();
    expect(screen.getByText('Carol')).toBeTruthy();
    expect(screen.queryByText('Bob')).toBeNull();
  });

  it('shows empty state when no media exists', () => {
    const noMedia: Message[] = [
      { id: 'm1', sender: { address: 'x', displayName: 'X' }, text: 'hi', timestamp: Date.now(), media: [] },
    ];
    render(MediaFeed, { props: { messages: noMedia } });
    expect(screen.getByText('No media yet')).toBeTruthy();
  });
});
```

**Step 2: Run the tests**

```bash
cd /Users/zeblith/work/zeblithic/harmony-client
npx vitest run
```

Expected: 6 tests pass (4 TextMessage + 2 MediaFeed).

**Step 3: Commit**

```bash
cd /Users/zeblith/work/zeblithic/harmony-client
git add src/lib/components/__tests__/MediaFeed.test.ts
git commit -m "test: add MediaFeed component tests"
```

---

### Task 14: Final verification and cleanup

**Step 1: Run all tests**

```bash
cd /Users/zeblith/work/zeblithic/harmony-client
npx vitest run
```

Expected: All 6+ tests pass.

**Step 2: Run the Tauri Rust build check**

```bash
cd /Users/zeblith/work/zeblithic/harmony-client/src-tauri
cargo check
```

Expected: Compiles without errors.

**Step 3: Verify the app visually**

```bash
cd /Users/zeblith/work/zeblithic/harmony-client
npm run dev
```

Check all requirements from the design doc:
- [ ] Three-panel layout: nav (240px) | text feed | media feed
- [ ] Dark theme with Discord-familiar colors
- [ ] Compact text messages with 24px avatars, bold names, muted timestamps
- [ ] Media indicator pills on messages with attachments
- [ ] Media cards (image, link, code) in the right panel
- [ ] Link-back: clicking media pill scrolls to card, clicking card header scrolls to + highlights message
- [ ] Responsive: below 768px, media panel hidden, inline embeds shown, nav collapses to 56px icons

**Step 4: Commit any final adjustments**

```bash
cd /Users/zeblith/work/zeblithic/harmony-client
git add -A
git commit -m "chore: final cleanup and verification"
```
