# Flat Navigation Hierarchy with Color Banding — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the flat channel list with a recursive tree navigation using zero-indentation color banding, bracket markers, three display modes, and configurable per-folder ordering.

**Architecture:** Flat `NavNode[]` array with `parentId` references rendered recursively. Each row shows stacked color border strips for depth. Three display modes (text/icon/both) and three sort orders (activity/pinned/alphabetical) configurable per-folder. All work happens in the `harmony-client` repo at `/Users/zeblith/work/zeblithic/harmony-client`.

**Tech Stack:** Svelte 5 (runes: `$state`, `$derived`, `$props`), TypeScript, vitest + @testing-library/svelte, jsdom

**Important conventions:**
- Svelte 5 uses `onclick={handler}` (NOT `on:click`)
- Props via `$props()`, state via `$state()`, derived via `$derived()`
- Tests need `resolve: { conditions: ['browser'] }` in vitest.config.ts (already configured)
- Run all commands from `/Users/zeblith/work/zeblithic/harmony-client`

---

### Task 1: Update types and add NavNode

**Files:**
- Modify: `src/lib/types.ts`

**Step 1: Add the new types to types.ts**

Add these types after the existing `Channel` interface (keep `Channel` for now — we'll remove it in Task 7):

```ts
export type NavNodeType = 'folder' | 'channel' | 'dm' | 'group-chat';
export type DisplayMode = 'text' | 'icon' | 'both';
export type SortOrder = 'activity' | 'pinned' | 'alphabetical';
export type UnreadLevel = 'none' | 'quiet' | 'standard' | 'loud';

export interface NavNode {
  id: string;
  parentId: string | null;
  type: NavNodeType;
  name: string;
  icon?: string;

  colorIndex?: number;
  expanded: boolean;

  displayMode?: DisplayMode;
  sortOrder?: SortOrder;

  unreadCount: number;
  unreadLevel: UnreadLevel;
  lastActivity?: number;

  peer?: Peer;
}
```

**Step 2: Verify the build still passes**

Run: `cd /Users/zeblith/work/zeblithic/harmony-client && npx vite build`
Expected: Build succeeds (new types are additive, nothing consumes them yet)

**Step 3: Commit**

```bash
cd /Users/zeblith/work/zeblithic/harmony-client
git add src/lib/types.ts
git commit -m "feat(nav): add NavNode type and related enums"
```

---

### Task 2: Create nav-utils.ts with color assignment and sorting helpers

**Files:**
- Create: `src/lib/nav-utils.ts`
- Create: `src/lib/nav-utils.test.ts`

**Step 1: Write the failing tests**

Create `src/lib/nav-utils.test.ts`:

```ts
import { describe, it, expect } from 'vitest';
import {
  getChildNodes,
  getNodeColor,
  sortNodes,
  getInheritedDisplayMode,
  getInheritedSortOrder,
} from './nav-utils';
import type { NavNode } from './types';

const NAV_PALETTE = ['#43b581', '#5865f2', '#9b59b6', '#e67e22'];

const nodes: NavNode[] = [
  { id: 'work', parentId: null, type: 'folder', name: 'Work', expanded: true, unreadCount: 0, unreadLevel: 'none' },
  { id: 'harmony', parentId: 'work', type: 'folder', name: 'Harmony Dev', expanded: true, unreadCount: 0, unreadLevel: 'none' },
  { id: 'ch-general', parentId: 'harmony', type: 'channel', name: 'general', expanded: false, unreadCount: 3, unreadLevel: 'standard', lastActivity: 1000 },
  { id: 'ch-crypto', parentId: 'harmony', type: 'channel', name: 'crypto', expanded: false, unreadCount: 0, unreadLevel: 'none', lastActivity: 500 },
  { id: 'friends', parentId: null, type: 'folder', name: 'Friends', expanded: true, unreadCount: 0, unreadLevel: 'none' },
  { id: 'eve', parentId: null, type: 'dm', name: 'Eve', expanded: false, unreadCount: 0, unreadLevel: 'none' },
];

describe('getChildNodes', () => {
  it('returns top-level nodes when parentId is null', () => {
    const children = getChildNodes(nodes, null);
    expect(children.map((n) => n.id)).toEqual(['work', 'friends', 'eve']);
  });

  it('returns children of a specific parent', () => {
    const children = getChildNodes(nodes, 'harmony');
    expect(children.map((n) => n.id)).toEqual(['ch-general', 'ch-crypto']);
  });

  it('returns empty array for leaf nodes', () => {
    const children = getChildNodes(nodes, 'ch-general');
    expect(children).toEqual([]);
  });
});

describe('getNodeColor', () => {
  it('assigns first top-level folder color index 0', () => {
    expect(getNodeColor(nodes, 'work')).toBe(0);
  });

  it('assigns second top-level folder color index 1', () => {
    expect(getNodeColor(nodes, 'friends')).toBe(1);
  });

  it('assigns nested folder avoiding parent color', () => {
    const color = getNodeColor(nodes, 'harmony');
    expect(color).not.toBe(0); // parent 'work' is 0
    expect(color).toBeGreaterThanOrEqual(0);
    expect(color).toBeLessThanOrEqual(3);
  });

  it('returns -1 for non-folder nodes', () => {
    expect(getNodeColor(nodes, 'ch-general')).toBe(-1);
  });
});

describe('sortNodes', () => {
  it('sorts by activity (most recent first)', () => {
    const children = getChildNodes(nodes, 'harmony');
    const sorted = sortNodes(children, 'activity');
    expect(sorted.map((n) => n.id)).toEqual(['ch-general', 'ch-crypto']);
  });

  it('sorts alphabetically', () => {
    const children = getChildNodes(nodes, 'harmony');
    const sorted = sortNodes(children, 'alphabetical');
    expect(sorted.map((n) => n.id)).toEqual(['ch-crypto', 'ch-general']);
  });

  it('preserves original order for pinned', () => {
    const children = getChildNodes(nodes, 'harmony');
    const sorted = sortNodes(children, 'pinned');
    expect(sorted.map((n) => n.id)).toEqual(['ch-general', 'ch-crypto']);
  });
});

describe('getInheritedDisplayMode', () => {
  it('returns the node displayMode if set', () => {
    const nodesWithMode: NavNode[] = [
      { ...nodes[0], displayMode: 'icon' },
      ...nodes.slice(1),
    ];
    expect(getInheritedDisplayMode(nodesWithMode, 'work')).toBe('icon');
  });

  it('inherits from parent if not set', () => {
    const nodesWithMode: NavNode[] = [
      { ...nodes[0], displayMode: 'both' },
      ...nodes.slice(1),
    ];
    expect(getInheritedDisplayMode(nodesWithMode, 'harmony')).toBe('both');
  });

  it('falls back to text if no ancestor has it set', () => {
    expect(getInheritedDisplayMode(nodes, 'harmony')).toBe('text');
  });
});

describe('getInheritedSortOrder', () => {
  it('returns the node sortOrder if set', () => {
    const nodesWithOrder: NavNode[] = [
      { ...nodes[0], sortOrder: 'alphabetical' },
      ...nodes.slice(1),
    ];
    expect(getInheritedSortOrder(nodesWithOrder, 'work')).toBe('alphabetical');
  });

  it('falls back to activity if no ancestor has it set', () => {
    expect(getInheritedSortOrder(nodes, 'harmony')).toBe('activity');
  });
});
```

**Step 2: Run tests to verify they fail**

Run: `cd /Users/zeblith/work/zeblithic/harmony-client && npx vitest run src/lib/nav-utils.test.ts`
Expected: FAIL — module `./nav-utils` not found

**Step 3: Write the implementation**

Create `src/lib/nav-utils.ts`:

```ts
import type { NavNode, DisplayMode, SortOrder } from './types';

export const NAV_PALETTE = ['#43b581', '#5865f2', '#9b59b6', '#e67e22'] as const;

/** Get direct children of a parent (null = top-level) */
export function getChildNodes(nodes: NavNode[], parentId: string | null): NavNode[] {
  return nodes.filter((n) => n.parentId === parentId);
}

/** Find a node by ID */
export function findNode(nodes: NavNode[], id: string): NavNode | undefined {
  return nodes.find((n) => n.id === id);
}

/** Get the depth of a node (0 = top-level) */
export function getNodeDepth(nodes: NavNode[], nodeId: string): number {
  let depth = 0;
  let current = findNode(nodes, nodeId);
  while (current?.parentId) {
    depth++;
    current = findNode(nodes, current.parentId);
  }
  return depth;
}

/**
 * Get the color index for a folder node.
 * Returns -1 for non-folder nodes (they inherit parent color for band rendering).
 *
 * Algorithm:
 * - Top-level folders: palette index = sibling position % 4
 * - Nested folders: pick from the 3 colors != parent's color, cycling by sibling position
 */
export function getNodeColor(nodes: NavNode[], nodeId: string): number {
  const node = findNode(nodes, nodeId);
  if (!node || node.type !== 'folder') return -1;

  // Use explicit override if set
  if (node.colorIndex !== undefined) return node.colorIndex;

  const siblings = getChildNodes(nodes, node.parentId).filter((n) => n.type === 'folder');
  const siblingIndex = siblings.findIndex((n) => n.id === nodeId);

  if (node.parentId === null) {
    // Top-level: simple rotation
    return siblingIndex % 4;
  }

  // Nested: avoid parent's color
  const parentColor = getNodeColor(nodes, node.parentId);
  const available = [0, 1, 2, 3].filter((c) => c !== parentColor);
  return available[siblingIndex % available.length];
}

/**
 * Get the color ancestry for rendering stacked border strips.
 * Returns array of color indices from outermost ancestor to the node's parent folder.
 */
export function getColorAncestry(nodes: NavNode[], nodeId: string): number[] {
  const colors: number[] = [];
  let current = findNode(nodes, nodeId);

  // Walk up to root, collecting folder colors
  while (current?.parentId) {
    const parent = findNode(nodes, current.parentId);
    if (parent?.type === 'folder') {
      colors.unshift(getNodeColor(nodes, parent.id));
    }
    current = parent;
  }

  return colors;
}

/** Sort nodes according to the given sort order */
export function sortNodes(nodes: NavNode[], order: SortOrder): NavNode[] {
  const copy = [...nodes];
  switch (order) {
    case 'activity':
      return copy.sort((a, b) => (b.lastActivity ?? 0) - (a.lastActivity ?? 0));
    case 'alphabetical':
      return copy.sort((a, b) => a.name.localeCompare(b.name, undefined, { sensitivity: 'base' }));
    case 'pinned':
      return copy; // preserve original order
  }
}

/** Get the effective display mode for a node, inheriting from ancestors */
export function getInheritedDisplayMode(nodes: NavNode[], nodeId: string): DisplayMode {
  let current = findNode(nodes, nodeId);
  while (current) {
    if (current.displayMode) return current.displayMode;
    current = current.parentId ? findNode(nodes, current.parentId) : undefined;
  }
  return 'text';
}

/** Get the effective sort order for a node, inheriting from ancestors */
export function getInheritedSortOrder(nodes: NavNode[], nodeId: string): SortOrder {
  let current = findNode(nodes, nodeId);
  while (current) {
    if (current.sortOrder) return current.sortOrder;
    current = current.parentId ? findNode(nodes, current.parentId) : undefined;
  }
  return 'activity';
}
```

**Step 4: Run tests to verify they pass**

Run: `cd /Users/zeblith/work/zeblithic/harmony-client && npx vitest run src/lib/nav-utils.test.ts`
Expected: All 11 tests PASS

**Step 5: Commit**

```bash
cd /Users/zeblith/work/zeblithic/harmony-client
git add src/lib/nav-utils.ts src/lib/nav-utils.test.ts
git commit -m "feat(nav): color assignment, sorting, and inheritance utils"
```

---

### Task 3: Create NavNodeRow component

**Files:**
- Create: `src/lib/components/NavNodeRow.svelte`
- Create: `src/lib/components/__tests__/NavNodeRow.test.ts`

**Step 1: Write the failing tests**

Create `src/lib/components/__tests__/NavNodeRow.test.ts`:

```ts
import { render, screen } from '@testing-library/svelte';
import { describe, it, expect, vi } from 'vitest';
import NavNodeRow from '../NavNodeRow.svelte';
import type { NavNode } from '../../types';

const baseNode: NavNode = {
  id: 'ch-general',
  parentId: 'harmony',
  type: 'channel',
  name: 'general',
  expanded: false,
  unreadCount: 0,
  unreadLevel: 'none',
};

describe('NavNodeRow', () => {
  it('renders channel name with hash prefix', () => {
    render(NavNodeRow, {
      props: { node: baseNode, colorAncestry: [], displayMode: 'text', isLastChild: false },
    });
    expect(screen.getByText('general')).toBeTruthy();
    expect(screen.getByText('#')).toBeTruthy();
  });

  it('renders color bands matching ancestry depth', () => {
    const { container } = render(NavNodeRow, {
      props: {
        node: baseNode,
        colorAncestry: [0, 1],
        displayMode: 'text',
        isLastChild: false,
      },
    });
    const bands = container.querySelectorAll('.color-band');
    expect(bands.length).toBe(2);
  });

  it('renders folder with bracket marker', () => {
    const folderNode: NavNode = {
      ...baseNode,
      id: 'work',
      parentId: null,
      type: 'folder',
      name: 'Work',
      expanded: true,
    };
    render(NavNodeRow, {
      props: { node: folderNode, colorAncestry: [], displayMode: 'text', isLastChild: false },
    });
    expect(screen.getByText('Work')).toBeTruthy();
  });

  it('shows standard unread badge with count', () => {
    const unreadNode: NavNode = { ...baseNode, unreadCount: 5, unreadLevel: 'standard' };
    render(NavNodeRow, {
      props: { node: unreadNode, colorAncestry: [], displayMode: 'text', isLastChild: false },
    });
    expect(screen.getByText('5')).toBeTruthy();
  });

  it('shows loud unread badge with pulsing class', () => {
    const loudNode: NavNode = { ...baseNode, unreadCount: 2, unreadLevel: 'loud' };
    const { container } = render(NavNodeRow, {
      props: { node: loudNode, colorAncestry: [], displayMode: 'text', isLastChild: false },
    });
    const badge = container.querySelector('.unread-badge.loud');
    expect(badge).toBeTruthy();
    expect(badge?.textContent?.trim()).toBe('2');
  });

  it('shows quiet unread dot', () => {
    const quietNode: NavNode = { ...baseNode, unreadCount: 1, unreadLevel: 'quiet' };
    const { container } = render(NavNodeRow, {
      props: { node: quietNode, colorAncestry: [], displayMode: 'text', isLastChild: false },
    });
    const dot = container.querySelector('.unread-dot');
    expect(dot).toBeTruthy();
  });

  it('renders close bracket when isLastChild and parent is folder', () => {
    const { container } = render(NavNodeRow, {
      props: { node: baseNode, colorAncestry: [0], displayMode: 'text', isLastChild: true },
    });
    const bracket = container.querySelector('.bracket-close');
    expect(bracket).toBeTruthy();
  });
});
```

**Step 2: Run tests to verify they fail**

Run: `cd /Users/zeblith/work/zeblithic/harmony-client && npx vitest run src/lib/components/__tests__/NavNodeRow.test.ts`
Expected: FAIL — module not found

**Step 3: Write the component**

Create `src/lib/components/NavNodeRow.svelte`:

```svelte
<script lang="ts">
  import type { NavNode, DisplayMode } from '../types';
  import { NAV_PALETTE } from '../nav-utils';
  import Avatar from './Avatar.svelte';

  let {
    node,
    colorAncestry,
    displayMode = 'text',
    isLastChild = false,
    onToggle,
    onClick,
  }: {
    node: NavNode;
    colorAncestry: number[];
    displayMode?: DisplayMode;
    isLastChild?: boolean;
    onToggle?: (nodeId: string) => void;
    onClick?: (nodeId: string) => void;
  } = $props();

  let typeIcon = $derived(
    node.type === 'channel' ? '#' :
    node.type === 'dm' ? '@' :
    node.type === 'group-chat' ? '@' :
    node.expanded ? '▾' : '▸'
  );

  let showName = $derived(displayMode !== 'icon');
  let showIcon = $derived(displayMode !== 'text' || node.type === 'folder');

  function handleClick() {
    if (node.type === 'folder') {
      onToggle?.(node.id);
    } else {
      onClick?.(node.id);
    }
  }
</script>

<button
  class="nav-row"
  class:has-unread={node.unreadLevel !== 'none'}
  class:folder={node.type === 'folder'}
  onclick={handleClick}
>
  <div class="bands">
    {#each colorAncestry as colorIdx (colorIdx + '-' + colorAncestry.indexOf(colorIdx))}
      <span
        class="color-band"
        style="left: {colorAncestry.indexOf(colorIdx) * 4}px; background: {NAV_PALETTE[colorIdx]}"
      ></span>
    {/each}
  </div>

  <div class="row-content" style="padding-left: {colorAncestry.length * 4 + 8}px">
    {#if displayMode === 'icon' && (node.type === 'dm' || node.type === 'group-chat') && node.peer}
      <Avatar address={node.peer.address} displayName={node.peer.displayName} size={32} />
    {:else if displayMode === 'icon'}
      <span class="node-icon-grid">{node.icon ?? node.name.charAt(0).toUpperCase()}</span>
    {:else}
      <span class="type-icon">{typeIcon}</span>
      {#if showName}
        <span class="node-name">{node.name}</span>
      {/if}
    {/if}

    {#if displayMode === 'both' && (node.type === 'dm' || node.type === 'group-chat') && node.peer}
      <Avatar address={node.peer.address} displayName={node.peer.displayName} size={20} />
      <span class="node-name">{node.name}</span>
    {/if}

    {#if node.unreadLevel === 'standard' && node.unreadCount > 0}
      <span class="unread-badge">{node.unreadCount}</span>
    {:else if node.unreadLevel === 'loud' && node.unreadCount > 0}
      <span class="unread-badge loud">{node.unreadCount}</span>
    {:else if node.unreadLevel === 'quiet' && node.unreadCount > 0}
      <span class="unread-dot"></span>
    {/if}
  </div>

  {#if node.type === 'folder' && node.expanded}
    <span class="bracket bracket-open">&#9484;</span>
  {/if}
  {#if isLastChild && colorAncestry.length > 0}
    <span class="bracket bracket-close">&#9496;</span>
  {/if}
</button>

<style>
  .nav-row {
    display: flex;
    align-items: center;
    position: relative;
    width: 100%;
    height: 28px;
    padding: 0;
    border: none;
    background: none;
    color: var(--text-muted);
    font-size: 14px;
    cursor: pointer;
    text-align: left;
  }

  .nav-row:hover {
    background: var(--bg-tertiary);
    color: var(--text-primary);
  }

  .nav-row.has-unread {
    color: var(--text-primary);
    font-weight: 600;
  }

  .bands {
    position: absolute;
    left: 0;
    top: 0;
    bottom: 0;
    pointer-events: none;
  }

  .color-band {
    position: absolute;
    top: 0;
    bottom: 0;
    width: 4px;
  }

  .row-content {
    display: flex;
    align-items: center;
    gap: 6px;
    flex: 1;
    min-width: 0;
    padding-right: 24px;
  }

  .type-icon {
    color: var(--text-muted);
    font-size: 14px;
    flex-shrink: 0;
    width: 16px;
    text-align: center;
  }

  .node-name {
    flex: 1;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }

  .node-icon-grid {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 32px;
    height: 32px;
    border-radius: 8px;
    background: var(--bg-tertiary);
    font-size: 14px;
    font-weight: 600;
    color: var(--text-primary);
    flex-shrink: 0;
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

  .unread-badge.loud {
    animation: pulse 1.5s ease-in-out infinite;
  }

  @keyframes pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.8; transform: scale(1.1); }
  }

  .unread-dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: var(--text-muted);
    flex-shrink: 0;
    opacity: 0;
    transition: opacity 0.2s ease;
  }

  .nav-row:hover .unread-dot {
    opacity: 1;
  }

  .bracket {
    position: absolute;
    right: 8px;
    color: var(--text-muted);
    font-size: 12px;
    line-height: 28px;
    pointer-events: none;
  }
</style>
```

**Step 4: Run tests to verify they pass**

Run: `cd /Users/zeblith/work/zeblithic/harmony-client && npx vitest run src/lib/components/__tests__/NavNodeRow.test.ts`
Expected: All 7 tests PASS

**Step 5: Commit**

```bash
cd /Users/zeblith/work/zeblithic/harmony-client
git add src/lib/components/NavNodeRow.svelte src/lib/components/__tests__/NavNodeRow.test.ts
git commit -m "feat(nav): NavNodeRow component with color bands, brackets, and unread indicators"
```

---

### Task 4: Create NavTree recursive renderer

**Files:**
- Create: `src/lib/components/NavTree.svelte`
- Create: `src/lib/components/__tests__/NavTree.test.ts`

**Step 1: Write the failing tests**

Create `src/lib/components/__tests__/NavTree.test.ts`:

```ts
import { render, screen } from '@testing-library/svelte';
import { describe, it, expect } from 'vitest';
import NavTree from '../NavTree.svelte';
import type { NavNode } from '../../types';

const nodes: NavNode[] = [
  { id: 'work', parentId: null, type: 'folder', name: 'Work', expanded: true, unreadCount: 0, unreadLevel: 'none', sortOrder: 'alphabetical' },
  { id: 'ch-crypto', parentId: 'work', type: 'channel', name: 'crypto', expanded: false, unreadCount: 0, unreadLevel: 'none', lastActivity: 500 },
  { id: 'ch-general', parentId: 'work', type: 'channel', name: 'general', expanded: false, unreadCount: 3, unreadLevel: 'standard', lastActivity: 1000 },
  { id: 'friends', parentId: null, type: 'folder', name: 'Friends', expanded: false, unreadCount: 0, unreadLevel: 'none' },
  { id: 'eve', parentId: null, type: 'dm', name: 'Eve', expanded: false, unreadCount: 0, unreadLevel: 'none' },
];

describe('NavTree', () => {
  it('renders top-level nodes', () => {
    render(NavTree, { props: { nodes, parentId: null } });
    expect(screen.getByText('Work')).toBeTruthy();
    expect(screen.getByText('Friends')).toBeTruthy();
    expect(screen.getByText('Eve')).toBeTruthy();
  });

  it('renders children of expanded folder', () => {
    render(NavTree, { props: { nodes, parentId: null } });
    // Work is expanded and sorted alphabetically
    expect(screen.getByText('crypto')).toBeTruthy();
    expect(screen.getByText('general')).toBeTruthy();
  });

  it('does not render children of collapsed folder', () => {
    render(NavTree, { props: { nodes, parentId: null } });
    // Friends is collapsed — no children to show anyway, but verify folder renders
    expect(screen.getByText('Friends')).toBeTruthy();
  });

  it('respects alphabetical sort order', () => {
    const { container } = render(NavTree, { props: { nodes, parentId: null } });
    const names = Array.from(container.querySelectorAll('.node-name'))
      .map((el) => el.textContent?.trim());
    // Under Work (alphabetical): crypto before general
    const cryptoIdx = names.indexOf('crypto');
    const generalIdx = names.indexOf('general');
    expect(cryptoIdx).toBeLessThan(generalIdx);
  });
});
```

**Step 2: Run tests to verify they fail**

Run: `cd /Users/zeblith/work/zeblithic/harmony-client && npx vitest run src/lib/components/__tests__/NavTree.test.ts`
Expected: FAIL — module not found

**Step 3: Write the component**

Create `src/lib/components/NavTree.svelte`:

```svelte
<script lang="ts">
  import type { NavNode } from '../types';
  import {
    getChildNodes,
    getNodeColor,
    getColorAncestry,
    sortNodes,
    getInheritedSortOrder,
    getInheritedDisplayMode,
  } from '../nav-utils';
  import NavNodeRow from './NavNodeRow.svelte';

  let {
    nodes,
    parentId = null,
    onToggle,
    onClick,
  }: {
    nodes: NavNode[];
    parentId?: string | null;
    onToggle?: (nodeId: string) => void;
    onClick?: (nodeId: string) => void;
  } = $props();

  let children = $derived(() => {
    const raw = getChildNodes(nodes, parentId);
    const order = parentId ? getInheritedSortOrder(nodes, parentId) : 'activity';
    return sortNodes(raw, order);
  });
</script>

{#each children() as node, i (node.id)}
  {@const ancestry = getColorAncestry(nodes, node.id)}
  {@const displayMode = node.parentId ? getInheritedDisplayMode(nodes, node.parentId) : 'text'}
  {@const isLastChild = i === children().length - 1 && ancestry.length > 0}

  <NavNodeRow
    {node}
    colorAncestry={ancestry}
    {displayMode}
    {isLastChild}
    {onToggle}
    {onClick}
  />

  {#if node.type === 'folder' && node.expanded}
    <svelte:self {nodes} parentId={node.id} {onToggle} {onClick} />
  {/if}
{/each}
```

**Step 4: Run tests to verify they pass**

Run: `cd /Users/zeblith/work/zeblithic/harmony-client && npx vitest run src/lib/components/__tests__/NavTree.test.ts`
Expected: All 4 tests PASS

**Step 5: Commit**

```bash
cd /Users/zeblith/work/zeblithic/harmony-client
git add src/lib/components/NavTree.svelte src/lib/components/__tests__/NavTree.test.ts
git commit -m "feat(nav): NavTree recursive renderer with sorting and color ancestry"
```

---

### Task 5: Rewrite NavPanel to use NavTree

**Files:**
- Modify: `src/lib/components/NavPanel.svelte`
- Create: `src/lib/components/__tests__/NavPanel.test.ts`

**Step 1: Write the failing tests**

Create `src/lib/components/__tests__/NavPanel.test.ts`:

```ts
import { render, screen } from '@testing-library/svelte';
import { describe, it, expect } from 'vitest';
import NavPanel from '../NavPanel.svelte';
import type { NavNode } from '../../types';

const nodes: NavNode[] = [
  { id: 'work', parentId: null, type: 'folder', name: 'Work', expanded: true, unreadCount: 0, unreadLevel: 'none' },
  { id: 'ch-general', parentId: 'work', type: 'channel', name: 'general', expanded: false, unreadCount: 3, unreadLevel: 'standard' },
  { id: 'friends', parentId: null, type: 'folder', name: 'Friends', expanded: true, unreadCount: 0, unreadLevel: 'none' },
  { id: 'dm-bob', parentId: 'friends', type: 'dm', name: 'Bob', expanded: false, unreadCount: 0, unreadLevel: 'none' },
  { id: 'eve', parentId: null, type: 'dm', name: 'Eve', expanded: false, unreadCount: 0, unreadLevel: 'none' },
];

describe('NavPanel', () => {
  it('renders the tree when not collapsed', () => {
    render(NavPanel, { props: { nodes, collapsed: false } });
    expect(screen.getByText('Work')).toBeTruthy();
    expect(screen.getByText('general')).toBeTruthy();
    expect(screen.getByText('Friends')).toBeTruthy();
    expect(screen.getByText('Bob')).toBeTruthy();
    expect(screen.getByText('Eve')).toBeTruthy();
  });

  it('shows only top-level icons when collapsed', () => {
    render(NavPanel, { props: { nodes, collapsed: true } });
    // Should show first letters of top-level nodes
    expect(screen.getByText('W')).toBeTruthy();
    expect(screen.getByText('F')).toBeTruthy();
    expect(screen.getByText('E')).toBeTruthy();
    // Should NOT show nested items
    expect(screen.queryByText('general')).toBeNull();
    expect(screen.queryByText('Bob')).toBeNull();
  });

  it('filters nodes by search query', async () => {
    render(NavPanel, { props: { nodes, collapsed: false } });
    const searchInput = screen.getByPlaceholderText('Search');
    // Simulate typing
    await searchInput.focus();
    // @ts-ignore
    searchInput.value = 'gen';
    searchInput.dispatchEvent(new Event('input', { bubbles: true }));
    // Should show matching node and its ancestors
    expect(screen.getByText('general')).toBeTruthy();
  });
});
```

**Step 2: Run tests to verify they fail**

Run: `cd /Users/zeblith/work/zeblithic/harmony-client && npx vitest run src/lib/components/__tests__/NavPanel.test.ts`
Expected: FAIL — NavPanel still expects `channels` prop, not `nodes`

**Step 3: Rewrite NavPanel.svelte**

Replace the entire content of `src/lib/components/NavPanel.svelte`:

```svelte
<script lang="ts">
  import type { NavNode } from '../types';
  import NavTree from './NavTree.svelte';

  let { nodes, collapsed = false, onNodeClick }: {
    nodes: NavNode[];
    collapsed: boolean;
    onNodeClick?: (nodeId: string) => void;
  } = $props();

  let navNodes = $state(nodes);
  let searchQuery = $state('');

  // Update navNodes when prop changes
  $effect(() => {
    navNodes = [...nodes];
  });

  let filteredNodes = $derived(() => {
    if (!searchQuery.trim()) return navNodes;
    const query = searchQuery.toLowerCase();

    // Find matching nodes and their ancestors
    const matchIds = new Set<string>();
    for (const node of navNodes) {
      if (node.name.toLowerCase().includes(query)) {
        matchIds.add(node.id);
        // Add all ancestors
        let current = node;
        while (current.parentId) {
          matchIds.add(current.parentId);
          const parent = navNodes.find((n) => n.id === current!.parentId);
          if (!parent) break;
          current = parent;
        }
      }
    }

    return navNodes
      .filter((n) => matchIds.has(n.id))
      .map((n) => n.type === 'folder' ? { ...n, expanded: true } : n);
  });

  let topLevelNodes = $derived(
    navNodes.filter((n) => n.parentId === null)
  );

  function toggleFolder(nodeId: string) {
    navNodes = navNodes.map((n) =>
      n.id === nodeId ? { ...n, expanded: !n.expanded } : n
    );
  }
</script>

<div class="nav-panel">
  {#if collapsed}
    <div class="collapsed-icons">
      {#each topLevelNodes as node (node.id)}
        <button
          class="collapsed-icon"
          class:has-unread={node.unreadLevel !== 'none' || node.unreadCount > 0}
          onclick={() => onNodeClick?.(node.id)}
          title={node.name}
        >
          {node.icon ?? node.name.charAt(0).toUpperCase()}
        </button>
      {/each}
    </div>
  {:else}
    <div class="search-bar">
      <input
        type="text"
        placeholder="Search"
        bind:value={searchQuery}
      />
    </div>
    <nav class="tree-container">
      <NavTree
        nodes={filteredNodes()}
        parentId={null}
        onToggle={toggleFolder}
        onClick={onNodeClick}
      />
    </nav>
  {/if}
</div>

<style>
  .nav-panel {
    display: flex;
    flex-direction: column;
    height: 100%;
  }

  .search-bar {
    padding: 8px 12px;
    border-bottom: 1px solid var(--border);
  }

  .search-bar input {
    width: 100%;
    padding: 6px 10px;
    border: none;
    border-radius: 4px;
    background: var(--bg-tertiary);
    color: var(--text-primary);
    font-size: 13px;
    outline: none;
  }

  .search-bar input::placeholder {
    color: var(--text-muted);
  }

  .search-bar input:focus {
    box-shadow: 0 0 0 1px var(--accent);
  }

  .tree-container {
    flex: 1;
    padding: 4px 0;
    overflow-y: auto;
  }

  .collapsed-icons {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 4px;
    padding: 8px 0;
  }

  .collapsed-icon {
    width: 40px;
    height: 40px;
    border: none;
    border-radius: 8px;
    background: var(--bg-tertiary);
    color: var(--text-muted);
    font-size: 16px;
    font-weight: 600;
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: center;
  }

  .collapsed-icon:hover {
    background: var(--accent);
    color: white;
  }

  .collapsed-icon.has-unread {
    color: var(--text-primary);
  }
</style>
```

**Step 4: Run tests to verify they pass**

Run: `cd /Users/zeblith/work/zeblithic/harmony-client && npx vitest run src/lib/components/__tests__/NavPanel.test.ts`
Expected: All 3 tests PASS

**Step 5: Commit**

```bash
cd /Users/zeblith/work/zeblithic/harmony-client
git add src/lib/components/NavPanel.svelte src/lib/components/__tests__/NavPanel.test.ts
git commit -m "feat(nav): rewrite NavPanel with tree structure, search, and collapsed mode"
```

---

### Task 6: Update mock data with tree structure

**Files:**
- Modify: `src/lib/mock-data.ts`

**Step 1: Replace the channels export with navNodes**

Keep the existing `peers` and `messages` exports unchanged. Replace the `channels` export at the bottom of the file with:

```ts
export const navNodes: NavNode[] = [
  // Work folder (top-level, green)
  {
    id: 'work',
    parentId: null,
    type: 'folder',
    name: 'Work',
    expanded: true,
    unreadCount: 0,
    unreadLevel: 'none',
    sortOrder: 'activity',
    lastActivity: Date.now(),
  },
  // Harmony Dev (nested in Work, blue)
  {
    id: 'harmony-dev',
    parentId: 'work',
    type: 'folder',
    name: 'Harmony Dev',
    expanded: true,
    unreadCount: 0,
    unreadLevel: 'none',
    sortOrder: 'activity',
    lastActivity: Date.now(),
  },
  {
    id: 'ch-general',
    parentId: 'harmony-dev',
    type: 'channel',
    name: 'general',
    expanded: false,
    unreadCount: 3,
    unreadLevel: 'standard',
    lastActivity: Date.now() - 5 * 60_000,
  },
  {
    id: 'ch-crypto',
    parentId: 'harmony-dev',
    type: 'channel',
    name: 'crypto',
    expanded: false,
    unreadCount: 0,
    unreadLevel: 'none',
    lastActivity: Date.now() - 2 * 3600_000,
  },
  {
    id: 'ch-transport',
    parentId: 'harmony-dev',
    type: 'channel',
    name: 'transport',
    expanded: false,
    unreadCount: 1,
    unreadLevel: 'quiet',
    lastActivity: Date.now() - 30 * 60_000,
  },
  // IPFS Crew (nested in Work, purple)
  {
    id: 'ipfs-crew',
    parentId: 'work',
    type: 'folder',
    name: 'IPFS Crew',
    expanded: true,
    unreadCount: 0,
    unreadLevel: 'none',
    sortOrder: 'alphabetical',
    lastActivity: Date.now() - 3600_000,
  },
  {
    id: 'ch-mesh',
    parentId: 'ipfs-crew',
    type: 'channel',
    name: 'mesh',
    expanded: false,
    unreadCount: 0,
    unreadLevel: 'none',
    lastActivity: Date.now() - 4 * 3600_000,
  },
  {
    id: 'ch-routing',
    parentId: 'ipfs-crew',
    type: 'channel',
    name: 'routing',
    expanded: false,
    unreadCount: 2,
    unreadLevel: 'loud',
    lastActivity: Date.now() - 10 * 60_000,
  },
  // Alice DM (inside Work)
  {
    id: 'dm-alice',
    parentId: 'work',
    type: 'dm',
    name: 'Alice',
    expanded: false,
    unreadCount: 0,
    unreadLevel: 'none',
    peer: peers[0],
    lastActivity: Date.now() - 2 * 3600_000,
  },
  // Friends folder (top-level, blue)
  {
    id: 'friends',
    parentId: null,
    type: 'folder',
    name: 'Friends',
    expanded: true,
    unreadCount: 0,
    unreadLevel: 'none',
    sortOrder: 'pinned',
    lastActivity: Date.now() - 3600_000,
  },
  {
    id: 'dm-bob',
    parentId: 'friends',
    type: 'dm',
    name: 'Bob',
    expanded: false,
    unreadCount: 0,
    unreadLevel: 'none',
    peer: peers[1],
    lastActivity: Date.now() - 5 * 3600_000,
  },
  {
    id: 'dm-carol',
    parentId: 'friends',
    type: 'dm',
    name: 'Carol',
    expanded: false,
    unreadCount: 1,
    unreadLevel: 'standard',
    peer: peers[2],
    lastActivity: Date.now() - 45 * 60_000,
  },
  // Eve DM (top-level, no folder)
  {
    id: 'dm-eve',
    parentId: null,
    type: 'dm',
    name: 'Eve',
    expanded: false,
    unreadCount: 0,
    unreadLevel: 'none',
    peer: { address: 'q7r8s9t0', displayName: 'Eve' },
    lastActivity: Date.now() - 8 * 3600_000,
  },
];
```

Also add the import at the top of the file:

```ts
import type { Peer, Message, NavNode } from './types';
```

**Step 2: Verify build**

Run: `cd /Users/zeblith/work/zeblithic/harmony-client && npx vite build`
Expected: Build succeeds (nothing consumes `navNodes` yet, and `channels` is still exported)

**Step 3: Commit**

```bash
cd /Users/zeblith/work/zeblithic/harmony-client
git add src/lib/mock-data.ts
git commit -m "feat(nav): add tree-structured navNodes mock data"
```

---

### Task 7: Wire App.svelte to use NavNode tree

**Files:**
- Modify: `src/App.svelte`
- Modify: `src/lib/mock-data.ts` (remove old `channels` export)

**Step 1: Update App.svelte import and NavPanel props**

In `src/App.svelte`, change the import from:

```ts
import { messages, channels } from './lib/mock-data';
```

to:

```ts
import { messages, navNodes } from './lib/mock-data';
```

And change the NavPanel snippet from:

```svelte
<NavPanel {channels} {collapsed} />
```

to:

```svelte
<NavPanel nodes={navNodes} {collapsed} />
```

**Step 2: Remove old channels export from mock-data.ts**

Delete the `channels` export from `src/lib/mock-data.ts` (the `Channel[]` array at the bottom). Keep the `Channel` interface in `types.ts` for now in case tests reference it — but remove the `channels` mock data.

**Step 3: Remove old Channel import from mock-data.ts**

Update the import in `src/lib/mock-data.ts` from:

```ts
import type { Peer, Message, Channel } from './types';
```

to:

```ts
import type { Peer, Message, NavNode } from './types';
```

(If `Channel` is no longer imported anywhere after this, remove it from `types.ts` too.)

**Step 4: Verify build and all tests pass**

Run: `cd /Users/zeblith/work/zeblithic/harmony-client && npx vite build && npx vitest run`
Expected: Build succeeds, all tests pass

**Step 5: Commit**

```bash
cd /Users/zeblith/work/zeblithic/harmony-client
git add src/App.svelte src/lib/mock-data.ts src/lib/types.ts
git commit -m "feat(nav): wire App to use tree-structured navigation"
```

---

### Task 8: Add display mode toggle to folders

**Files:**
- Modify: `src/lib/components/NavNodeRow.svelte`

**Step 1: Add the display mode toggle**

Add an `onDisplayModeChange` callback prop to `NavNodeRow`:

```ts
onDisplayModeChange?: (nodeId: string, mode: DisplayMode) => void;
```

For folder rows, add a toggle button that appears on hover and cycles through text → icon → both → text:

```svelte
{#if node.type === 'folder'}
  <button
    class="mode-toggle"
    onclick|stopPropagation={() => {
      const modes: DisplayMode[] = ['text', 'icon', 'both'];
      const currentMode = displayMode;
      const nextIdx = (modes.indexOf(currentMode) + 1) % modes.length;
      onDisplayModeChange?.(node.id, modes[nextIdx]);
    }}
    title="Display: {displayMode}"
  >
    {displayMode === 'text' ? '☰' : displayMode === 'icon' ? '⊞' : '☰⊞'}
  </button>
{/if}
```

**Important:** In Svelte 5, event modifiers like `|stopPropagation` are not supported on `onclick`. Instead use:

```svelte
onclick={(e) => {
  e.stopPropagation();
  const modes: DisplayMode[] = ['text', 'icon', 'both'];
  const nextIdx = (modes.indexOf(displayMode) + 1) % modes.length;
  onDisplayModeChange?.(node.id, modes[nextIdx]);
}}
```

Add CSS for the toggle:

```css
.mode-toggle {
  position: absolute;
  right: 24px;
  border: none;
  background: none;
  color: var(--text-muted);
  font-size: 12px;
  cursor: pointer;
  opacity: 0;
  transition: opacity 0.15s ease;
  padding: 2px 4px;
}

.nav-row:hover .mode-toggle {
  opacity: 1;
}

.mode-toggle:hover {
  color: var(--text-primary);
}
```

**Step 2: Wire the callback through NavTree and NavPanel**

Add `onDisplayModeChange` prop to NavTree (pass through like `onToggle`).

In NavPanel, add a handler:

```ts
function changeDisplayMode(nodeId: string, mode: DisplayMode) {
  navNodes = navNodes.map((n) =>
    n.id === nodeId ? { ...n, displayMode: mode } : n
  );
}
```

Pass it to NavTree as `onDisplayModeChange={changeDisplayMode}`.

**Step 3: Verify build and tests**

Run: `cd /Users/zeblith/work/zeblithic/harmony-client && npx vite build && npx vitest run`
Expected: Build succeeds, all tests pass

**Step 4: Commit**

```bash
cd /Users/zeblith/work/zeblithic/harmony-client
git add src/lib/components/NavNodeRow.svelte src/lib/components/NavTree.svelte src/lib/components/NavPanel.svelte
git commit -m "feat(nav): display mode toggle for folders (text/icon/both)"
```

---

### Task 9: Add sort order dropdown to folders

**Files:**
- Modify: `src/lib/components/NavNodeRow.svelte`
- Modify: `src/lib/components/NavTree.svelte`
- Modify: `src/lib/components/NavPanel.svelte`

**Step 1: Add sort order popover to NavNodeRow**

Add an `onSortOrderChange` callback prop. For folder rows, add a small dropdown that appears on right-click or via a gear icon on hover:

```ts
onSortOrderChange?: (nodeId: string, order: SortOrder) => void;
```

Add a simple popover state and UI:

```svelte
<script lang="ts">
  // ... existing code ...
  let showSortMenu = $state(false);
</script>

<!-- Inside the nav-row button, add a sort trigger for folders -->
{#if node.type === 'folder' && node.sortOrder}
  <button
    class="sort-trigger"
    onclick={(e) => { e.stopPropagation(); showSortMenu = !showSortMenu; }}
    title="Sort: {node.sortOrder ?? 'activity'}"
  >
    ↕
  </button>
{/if}

<!-- Sort menu popover (outside the button) -->
{#if showSortMenu}
  <div class="sort-menu">
    {#each ['activity', 'pinned', 'alphabetical'] as order}
      <button
        class="sort-option"
        class:active={node.sortOrder === order}
        onclick={(e) => {
          e.stopPropagation();
          onSortOrderChange?.(node.id, order as SortOrder);
          showSortMenu = false;
        }}
      >
        {order === 'activity' ? '🕐 Activity' : order === 'pinned' ? '📌 Pinned' : '🔤 A-Z'}
      </button>
    {/each}
  </div>
{/if}
```

CSS for sort menu:

```css
.sort-trigger {
  position: absolute;
  right: 36px;
  border: none;
  background: none;
  color: var(--text-muted);
  font-size: 11px;
  cursor: pointer;
  opacity: 0;
  transition: opacity 0.15s ease;
  padding: 2px 4px;
}

.nav-row:hover .sort-trigger {
  opacity: 1;
}

.sort-menu {
  position: absolute;
  right: 8px;
  top: 28px;
  background: var(--bg-tertiary);
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 4px;
  z-index: 10;
  display: flex;
  flex-direction: column;
  gap: 2px;
  min-width: 120px;
}

.sort-option {
  border: none;
  background: none;
  color: var(--text-secondary);
  padding: 6px 8px;
  border-radius: 4px;
  cursor: pointer;
  font-size: 12px;
  text-align: left;
}

.sort-option:hover {
  background: var(--bg-secondary);
  color: var(--text-primary);
}

.sort-option.active {
  color: var(--accent);
}
```

**Step 2: Wire through NavTree and NavPanel**

Pass `onSortOrderChange` through NavTree to NavNodeRow.

In NavPanel, add handler:

```ts
function changeSortOrder(nodeId: string, order: SortOrder) {
  navNodes = navNodes.map((n) =>
    n.id === nodeId ? { ...n, sortOrder: order } : n
  );
}
```

**Step 3: Verify build and tests**

Run: `cd /Users/zeblith/work/zeblithic/harmony-client && npx vite build && npx vitest run`
Expected: Build succeeds, all tests pass

**Step 4: Commit**

```bash
cd /Users/zeblith/work/zeblithic/harmony-client
git add src/lib/components/NavNodeRow.svelte src/lib/components/NavTree.svelte src/lib/components/NavPanel.svelte
git commit -m "feat(nav): sort order dropdown for folders (activity/pinned/alphabetical)"
```

---

### Task 10: Final integration test and visual verification

**Files:**
- No new files — verification only

**Step 1: Run all tests**

Run: `cd /Users/zeblith/work/zeblithic/harmony-client && npx vitest run`
Expected: All tests pass (nav-utils: 11, NavNodeRow: 7, NavTree: 4, NavPanel: 3, TextMessage: 4, MediaFeed: 2 = 31 total)

**Step 2: Build verification**

Run: `cd /Users/zeblith/work/zeblithic/harmony-client && npx vite build`
Expected: Build succeeds

**Step 3: Visual verification with dev server**

Run: `cd /Users/zeblith/work/zeblithic/harmony-client && npx vite --port 5199 &`

Use Playwright to take a screenshot and verify:
- Color bands visible on left edge of nav items
- Bracket markers (┌/┘) visible on folder open/close
- Unread indicators: standard badge on #general, quiet dot on #transport, pulsing badge on #routing
- Nested folders properly indented via color bands
- Tree collapses when folders are toggled

Kill the dev server after verification.

**Step 4: Commit any final adjustments**

If visual inspection reveals CSS tweaks needed, fix and commit.
