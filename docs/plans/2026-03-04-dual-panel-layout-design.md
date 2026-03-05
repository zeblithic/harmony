# Dual-Panel Layout Design

**Bead:** harmony-g2ud
**Date:** 2026-03-04
**Status:** Approved

## Problem

The harmony-client repo has design docs but no code. The foundational dual-panel layout (text feed + media feed) is the shell that all other UX features build on. We need to scaffold the Tauri + Svelte 5 app and implement the dual-panel layout with responsive collapse.

## Decisions

- **Repo:** `zeblithic/harmony-client` (separate from harmony core)
- **Stack:** Tauri v2 + Svelte 5 (runes) + Vite + TypeScript
- **Layout:** CSS Grid with named areas (nav, text-feed, media-feed)
- **Data:** Mock/static data — no daemon integration yet
- **Responsive:** Include responsive collapse at 768px breakpoint

## Project Structure

```
harmony-client/
├── Cargo.toml                     # Workspace: [harmony-app]
├── crates/
│   └── harmony-app/
│       ├── Cargo.toml             # tauri, serde
│       └── src/
│           ├── main.rs            # Tauri bootstrap
│           └── commands.rs        # Tauri IPC stubs
├── ui/
│   ├── package.json
│   ├── vite.config.ts
│   ├── tsconfig.json
│   ├── src/
│   │   ├── app.html
│   │   ├── app.css                # Dark theme, CSS custom properties
│   │   ├── App.svelte
│   │   └── lib/
│   │       ├── components/        # TextMessage, MediaCard, ComposeBar
│   │       ├── stores/            # Runes-based state
│   │       ├── types.ts           # Message, MediaAttachment, Peer
│   │       └── mock-data.ts       # ~15 sample messages
│   └── views/                     # Layout, NavPanel, TextFeed, MediaFeed
├── tauri.conf.json
└── docs/plans/
```

## Component Architecture

```
App.svelte
└── Layout.svelte              ← CSS Grid shell
    ├── NavPanel.svelte        ← Placeholder sidebar
    ├── TextFeed.svelte        ← Compact message list
    │   ├── TextMessage.svelte ← Message row + inline media indicators
    │   └── ComposeBar.svelte  ← Input stub
    └── MediaFeed.svelte       ← Stacked cards
        └── MediaCard.svelte   ← Image/link/code card + link-back
```

### Layout Grid

```css
.layout {
  display: grid;
  grid-template-columns: 240px 1fr 1fr;
  grid-template-areas: "nav text media";
  height: 100vh;
}

@media (max-width: 768px) {
  .layout {
    grid-template-columns: 56px 1fr;
    grid-template-areas: "nav text";
  }
}
```

In collapsed mode, MediaFeed is hidden and TextMessage renders inline embeds instead of indicator chips.

## Data Model

```typescript
interface Message {
  id: string;
  sender: Peer;
  text: string;
  timestamp: number;
  media: MediaAttachment[];
}

interface MediaAttachment {
  id: string;
  type: 'image' | 'link' | 'code';
  url?: string;
  title?: string;
  domain?: string;
  content?: string;
}

interface Peer {
  address: string;
  displayName: string;
  avatarUrl?: string;
}
```

## Link-Back Anchoring

Each MediaCard stores the originating `message.id`. Clicking the card header scrolls the text feed to that message and highlights it. Each TextMessage with media shows a clickable indicator that scrolls to the corresponding media card.

## Visual Design

Dark theme (Discord-familiar):

```css
:root {
  --bg-primary: #1e1f22;
  --bg-secondary: #2b2d31;
  --bg-tertiary: #313338;
  --text-primary: #f2f3f5;
  --text-secondary: #b5bac1;
  --text-muted: #949ba4;
  --accent: #5865f2;
  --border: #3f4147;
}
```

- Text feed: 24px avatar, bold name, muted timestamp, compact rows (~36px)
- Media indicators: small pills in muted color, clickable
- Media cards: rounded corners, tertiary background, image fills width
- Panels scroll independently; link-back anchors provide cross-panel navigation

## Responsive Behavior

- `> 768px`: Full three-panel (nav 240px | text 1fr | media 1fr)
- `<= 768px`: Nav collapses to 56px icons, media panel hidden, inline embeds in text feed

## Testing

1. **Component rendering** — Vitest + @testing-library/svelte
2. **Responsive behavior** — Verify collapsed state toggles and conditional rendering
3. **Link-back anchoring** — Verify scroll dispatch on click
4. **Tauri build** — `cargo build` for the Rust bootstrap
