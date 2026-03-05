# Flat Navigation Hierarchy with Color Banding — Design

**Goal:** Replace the flat channel list in harmony-client's NavPanel with a recursive tree navigation using zero-indentation depth visualization via color banding and bracket markers.

**Architecture:** Flat `NavNode[]` array with `parentId` references. Components recursively render by filtering children. Color assignment computed at render time from sibling position and parent color. Three display modes and three sort orders configurable per-folder.

**Tech Stack:** Svelte 5 (runes), TypeScript, vitest + @testing-library/svelte

---

## 1. Data Model

The flat `Channel` type is replaced with `NavNode`:

```ts
type NavNodeType = 'folder' | 'channel' | 'dm' | 'group-chat';
type DisplayMode = 'text' | 'icon' | 'both';
type SortOrder = 'activity' | 'pinned' | 'alphabetical';
type UnreadLevel = 'none' | 'quiet' | 'standard' | 'loud';

interface NavNode {
  id: string;
  parentId: string | null;    // null = top-level
  type: NavNodeType;
  name: string;
  icon?: string;              // emoji or letter for icon grid mode

  // Tree display
  colorIndex?: number;        // 0-3, auto-assigned if not set
  expanded: boolean;          // folder open/closed state

  // Folder-specific
  displayMode?: DisplayMode;  // per-folder display toggle
  sortOrder?: SortOrder;      // per-folder ordering

  // Unread state
  unreadCount: number;
  unreadLevel: UnreadLevel;   // highest priority unread

  // Ordering
  lastActivity?: number;      // timestamp for activity-first sort

  // For DMs/group-chats
  peer?: Peer;                // avatar source for icon mode
}
```

Key decisions:
- `colorIndex` is optional — auto-computed from sibling position when not explicitly set
- `expanded` controls folder collapse state
- `displayMode` and `sortOrder` are per-folder, inherited from nearest ancestor if not set (fallback: `text` and `activity`)
- `unreadLevel` tracks the highest-priority unread for indicator styling

---

## 2. Color Banding System

Four colors rotating by sibling position, avoiding the parent's color.

**Palette:**
```
['#43b581', '#5865f2', '#9b59b6', '#e67e22']
 (green,     blue,      purple,    orange)
```

**Assignment algorithm:**
1. Top-level folders: palette order by position (0, 1, 2, 3, 0, ...)
2. Nested folders: pick from the 3 colors that aren't the parent's color, cycling by sibling position
3. Channels/DMs/group-chats inherit their parent folder's color (no own band)

**Rendering:**
- Each folder at depth `d` adds a 4px left border strip in its assigned color
- Strips stack: a channel 3 levels deep has 3 stacked strips (12px total)
- Each node renders `depth` number of `<span>` strips, each offset 4px further right
- Max practical depth ~8 = 32px (one avatar width)

**Bracket markers** (colorblind accessibility):
- Folder open: `┌` on the right edge of the folder header row
- Folder close: `┘` on the right edge of the last visible child
- Rendered as muted text (`var(--text-muted)`)

---

## 3. Component Architecture

```
NavPanel.svelte          — shell: search bar + tree container + collapsed mode
  NavTree.svelte         — recursive renderer for a given parentId
    NavNodeRow.svelte    — single row: bands + icon/name + unread + bracket
```

**NavPanel** owns the full `NavNode[]` array, collapsed flag, and search filter. When collapsed, renders only top-level nodes as icons (first letter or emoji).

**NavTree** is the recursive engine:
1. Filters nodes where `node.parentId === parentId`
2. Sorts by parent folder's `sortOrder`
3. Renders each as `NavNodeRow`
4. For expanded folders, recurses with the folder's `id`

**NavNodeRow** renders one row:
- Left: stacked color band strips (one per ancestor folder)
- Center: type icon (`#` channel, `@` DM, folder icon) + name + unread indicator
- Right: bracket marker if folder opener or last child
- Display mode determines content (text/icon/both)

**Unread indicators** (three styles):
- `quiet`: small dot, visible on hover only (opacity transition)
- `standard`: bold name + count badge (current style)
- `loud`: pulsing badge with `@keyframes` animation

---

## 4. Display Modes & Ordering

**Display modes** (per-folder toggle, defaults to `text`):
- **Text list**: 28px rows. Type icon + name + unread.
- **Icon grid**: 32x32 cells in flexbox wrap. First letter/emoji or avatar. Unread dot overlay.
- **Both**: 32px rows with icon left, name right.

A subtle toggle button in each folder header cycles modes (appears on hover).

**Ordering** (per-folder, defaults to `activity`):
- **Activity-first**: Sort by `lastActivity` timestamp descending.
- **Pinned**: Static array order.
- **Alphabetical**: Case-insensitive name sort.

A dropdown/popover on the folder header selects ordering mode.

**Inheritance**: Folders without explicit `displayMode`/`sortOrder` inherit from nearest ancestor, falling back to `text`/`activity`.

---

## 5. Mock Data & Testing

**Mock tree:**
```
Work                          (folder, green)
  Harmony Dev                 (folder, blue)
    # general                 (channel, 3 unread standard)
    # crypto                  (channel, 0 unread)
    # transport               (channel, 1 unread quiet)
  IPFS Crew                   (folder, purple)
    # mesh                    (channel, 0 unread)
    # routing                 (channel, 2 unread loud)
  Alice                       (dm, in Work folder)
Friends                       (folder, blue)
  Bob                         (dm, 0 unread)
  Carol                       (dm, 1 unread standard)
Eve                           (dm, top-level, no folder)
```

Exercises: 2 nesting levels, all 4 colors, all 3 unread levels, DMs inside and outside folders.

**Tests** (~8-10):
- NavNodeRow: color bands match depth, bracket markers render for folders, correct unread indicator per level
- NavTree: correct sort order per mode, expanded/collapsed state, display mode toggling
- NavPanel: collapsed mode shows top-level icons only, search filters nodes by name

---

## Scope Decisions

- **No community hubs** — everything is folders, channels, or DMs. Hub node type added when Zenoh integration lands.
- **Collapsed nav** — top-level icons only (quick-jump bar), no tree depth visible.
- **All three unread styles** stubbed with mock data, even though priority system (harmony-pzem) isn't built yet.
- **Full vision scope** — tree + color banding + brackets + all display modes + all ordering modes.
