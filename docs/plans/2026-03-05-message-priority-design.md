# Message Priority Levels — Design

Bead: `harmony-pzem`

Three-tier message priority system (quiet / standard / loud) with sender-side
UX, receiver notification controls, and a NotificationService that resolves the
full override chain.

---

## Data Model

### New types in `types.ts`

```typescript
type MessagePriority = 'quiet' | 'standard' | 'loud';

type NotificationAction = 'silent' | 'dot_only' | 'notify' | 'sound' | 'break_dnd';

interface NotificationPolicy {
  quiet: NotificationAction;     // default: 'dot_only'
  standard: NotificationAction;  // default: 'sound'
  loud: NotificationAction;      // default: 'break_dnd'
}

interface NotificationSettings {
  global: NotificationPolicy;
  perCommunity: Map<string, Partial<NotificationPolicy>>;  // keyed by nav node ID
  perPeer: Map<string, Partial<NotificationPolicy>>;       // keyed by peer address
}
```

### Message extension

Add `priority: MessagePriority` to the existing `Message` interface. Defaults to
`'standard'`.

---

## NotificationService

A class in `src/lib/notification-service.ts` that owns the settings state and
resolves the override chain.

```typescript
class NotificationService {
  settings: NotificationSettings;

  resolve(priority: MessagePriority, peerAddress: string, communityId?: string): NotificationAction;

  setGlobalPolicy(policy: NotificationPolicy): void;
  setCommunityPolicy(communityId: string, policy: Partial<NotificationPolicy>): void;
  setPeerPolicy(peerAddress: string, policy: Partial<NotificationPolicy>): void;
  clearCommunityPolicy(communityId: string): void;
  clearPeerPolicy(peerAddress: string): void;

  shouldPlaySound(action: NotificationAction): boolean;
}
```

### Resolution chain

1. Check `perPeer[peerAddress]` for the given priority level
2. If not set, check `perCommunity[communityId]` for that level
3. If not set, fall back to `global[priority]`

Sound playback is stubbed — `shouldPlaySound` returns a boolean but no audio is
played. Actual playback deferred to the profile system bead.

---

## ComposeBar — Priority Toggle + Keyboard Shortcuts

### Visual

A three-state toggle button group to the left of the input: `🔇` (quiet) `🔔`
(standard) `📢` (loud). Active state highlighted with accent color. Default is
always `standard`.

### Interaction

- Click any icon to set priority for the next message
- `Enter` sends at current priority (defaults to standard)
- `Ctrl+Enter` sends as quiet regardless of toggle state
- Priority resets to `standard` after every send
- No keyboard shortcut for loud — intentional friction per design doc

### Changes

- New prop: `onSend?: (text: string, priority: MessagePriority) => void`
- Input changes from `<input>` to `<textarea>` (Shift+Enter for newline)
- Subtle label below input when not standard: "sending quietly" / "sending loudly"

---

## TextFeed — Quiet Message Collapsing

### Grouping

Messages are processed into display items before rendering. Consecutive quiet
messages from any sender merge into a collapsed group. Non-quiet messages break
the group.

```typescript
type FeedItem =
  | { kind: 'message'; message: Message }
  | { kind: 'quiet-group'; messages: Message[] };
```

A `groupMessages(messages: Message[]): FeedItem[]` function in
`src/lib/feed-utils.ts` handles grouping — pure function, easy to test.

### Collapsed row

```
🔇 3 quiet messages from Alice, Bob  [▸ expand]
```

- Muted text color, smaller font
- Click to expand inline (reveals messages with 0.6 opacity)
- Click again to re-collapse

### New component

`QuietMessageGroup.svelte` — takes `messages: Message[]`, manages its own
expand/collapse state.

### Loud messages

Standard messages render normally. Loud messages get a 2px left accent border
and slightly bolder sender name.

---

## Notification Settings Panel

A slide-out panel that overlays the media feed area. Accessible from a gear icon
in the nav header.

### Tabs

**Global:** Three rows (quiet / standard / loud), each with a dropdown for
`NotificationAction`.

```
Quiet messages:     [dot_only ▾]
Standard messages:  [sound ▾]
Loud messages:      [break_dnd ▾]
```

**Communities:** Lists folder-type nav nodes. Expand to configure per-priority
overrides. Unset levels show "Using global default". Reset button clears
override.

**Peers:** Lists known peers by display name. Same expand-to-configure pattern.
Unset levels show "Using community/global default". Reset button clears
override.

### Action labels

- `silent` → "Muted"
- `dot_only` → "Dot only"
- `notify` → "Notification"
- `sound` → "Sound"
- `break_dnd` → "Break DND"

### Component

`NotificationSettingsPanel.svelte` — receives `NotificationService`, reads
settings, calls setter methods on change. Open/close state owned by
`Layout.svelte`.

---

## Mock Data & Integration

- Add `priority` field to existing mock messages. Most stay `'standard'`. Short
  acknowledgments become `'quiet'` (msg-04, msg-06, msg-10, msg-14). One
  attention-seeking message becomes `'loud'` (msg-08).
- Default `NotificationService` with sensible defaults. One mock per-peer
  override (Eve → quiet: silent) to demonstrate the settings panel.
- `Layout.svelte` owns settings panel state and `NotificationService` instance,
  threading them to `TextFeed`, `ComposeBar`, and `NotificationSettingsPanel`.
- Nav `unreadLevel` stays as hardcoded mock data. Deriving from message
  priorities requires per-channel message storage not yet built.

---

## Testing

### Unit tests (pure functions)

- `notification-service.test.ts` — resolve chain, defaults, partial overrides,
  clearing policies
- `feed-utils.test.ts` — consecutive quiet grouping, non-quiet breaks groups,
  single quiet, mixed priorities, empty input, all-quiet

### Component tests (vitest + @testing-library/svelte)

- `ComposeBar.test.ts` — default priority, toggle changes priority, Enter sends,
  Ctrl+Enter sends quiet, priority resets after send, no loud shortcut
- `QuietMessageGroup.test.ts` — collapsed rendering, expand/collapse toggle,
  dimmed class on expanded messages
- `TextFeed.test.ts` (extend) — quiet groups render collapsed, loud messages
  have accent border, standard messages normal
- `NotificationSettingsPanel.test.ts` — displays global settings, dropdown calls
  setter, community/peer tabs, reset clears override

### Not tested

- Actual sound playback (stubbed)
- DND interaction (OS-level)
- Real notification push (no backend)
