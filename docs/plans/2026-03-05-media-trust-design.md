# Media Trust & Auto-Preview System Design

Security-first media handling for the Harmony client. Nothing renders
automatically unless the user has opted in.

Implements bead `harmony-2fvs`. See also section 5 of
`harmony-client/docs/plans/2026-03-03-chat-ux-vision-design.md`.

---

## Data Model

### Types

```typescript
type TrustLevel = 'untrusted' | 'preview' | 'trusted';

interface TrustSettings {
  global: TrustLevel;                       // default: 'untrusted'
  perPeer: Map<string, TrustLevel>;         // keyed by peer address
  perCommunity: Map<string, TrustLevel>;    // keyed by community/folder id
}
```

### TrustService

A plain class parallel to `NotificationService`:

```typescript
class TrustService {
  settings: TrustSettings;

  // Resolution chain: per-peer > per-community > global
  resolve(peerAddress: string, communityId?: string): TrustLevel;

  // Setters
  setPeerTrust(address: string, level: TrustLevel): void;
  setCommunityTrust(id: string, level: TrustLevel): void;
  setGlobalTrust(level: TrustLevel): void;

  // Per-session loaded attachment tracking
  private loadedAttachments: Set<string>;
  isLoaded(attachmentId: string): boolean;
  markLoaded(attachmentId: string): void;
}
```

### Trust gate scope

**Gated** (subject to trust level):
- External URLs (images, links) — these make network requests to arbitrary
  servers, leaking the user's IP and potentially loading tracking pixels or
  malicious payloads.

**Bypasses gate** (always renders):
- Plain text.
- Code blocks — `attachment.content` is already in the message, no fetch.
- CID-referenced content — content-addressed means integrity-verified before
  rendering.
- Sender avatars — identicons are locally generated; future `avatarUrl` will
  be CID-based.

---

## Trust Levels

| Level | Behavior | Scope |
|---|---|---|
| **Untrusted** (default) | Placeholder card, two-step click to load one-time | This PR |
| **Preview** | Skeleton only — behaves identically to Untrusted | Data model + UI toggle present, marked "coming soon". OG metadata fetching is future work. |
| **Trusted** | Full auto-preview — images render, links expand | This PR |

### Resolution chain (highest priority wins)

```
1. Per-peer override       → use it
2. Per-community override  → use it
3. Global default          → untrusted
```

Peers and communities with no override fall through to global. "Use default"
in the UI means no override is stored.

---

## Placeholder Card (UntrustedMediaCard)

For untrusted media, a new `UntrustedMediaCard` component replaces the
normal `MediaCard`.

### Visual

```
┌──────────────────────────────────┐
│  Alice · 2:34 PM            ↗   │
│                                  │
│  🔒  Blocked media — image       │
│                                  │
│  [ Show ]                        │
│                                  │
└──────────────────────────────────┘
```

- Same card-header as `MediaCard` (avatar, sender, timestamp, link-back).
- Body: lock icon, "Blocked media", attachment type (image/link).
- **No URL or domain displayed** — prevents social engineering.
- Single "Show" button.

### Two-step load flow

1. User clicks **Show** → button text changes to **"Confirm load"** with a
   1-second disabled cooldown (grayed out, no pointer events).
2. After cooldown, user clicks **"Confirm load"** → attachment ID added to
   `TrustService.loadedAttachments`, component re-renders as real `MediaCard`.
3. If user clicks away or waits without confirming, state resets to "Show".

The cooldown prevents accidental double-tap from bypassing the confirmation.

### Per-attachment scope

Each placeholder is independent. Loading one image from a peer does NOT
load their other media. Users control exactly what gets fetched.

---

## Text Feed Integration

### Desktop mode (pills)

For untrusted media, pills change appearance:
- Normal: `🖼 Routing comparison`
- Untrusted: `🔒 blocked image`

Clicking an untrusted pill scrolls to the placeholder card in the media
feed (same scroll-to behavior as today).

### Collapsed/mobile mode (inline embeds)

Untrusted inline embeds show as inline placeholder blocks instead of
rendering `<img src>` directly. Same two-step load flow, inline.

---

## Settings UI

Extend the existing `NotificationSettingsPanel` with a "Media Trust"
section.

### Global

```
Media Trust
  Default trust level:  [Untrusted ▾]
```

Dropdown options: Untrusted / Preview (coming soon) / Trusted.

### Per-peer tab

```
Alice
  Notifications: ...existing...
  Media trust:   [Use default ▾]
```

Dropdown options: Use default / Untrusted / Preview (coming soon) / Trusted.

### Per-community tab

Same pattern as per-peer.

"Use default" means no override — falls through to global default.

---

## Security Hardening

Applied to ALL media rendering regardless of trust level:

- `referrerpolicy="no-referrer"` on all `<img>` tags.
- `rel="noopener noreferrer"` on all `<a>` tags (currently only `noopener`).
- Sanitize `href` values: reject `javascript:`, `data:`, `vbscript:` schemes.
  Only allow `http:`, `https:`, and `mailto:`.
- `crossorigin="anonymous"` on `<img>` tags to prevent cookies being sent
  with image requests.

A `sanitizeHref()` utility function handles scheme validation.

---

## File Plan

### New files

| File | Purpose |
|---|---|
| `src/lib/trust-service.ts` | TrustService class |
| `src/lib/trust-service.test.ts` | Unit tests for resolution chain + loaded tracking |
| `src/lib/components/UntrustedMediaCard.svelte` | Placeholder card with two-step load |
| `src/lib/components/__tests__/UntrustedMediaCard.test.ts` | Component tests |
| `src/lib/url-sanitize.ts` | `sanitizeHref()` scheme allowlist |
| `src/lib/url-sanitize.test.ts` | Tests for scheme validation |

### Modified files

| File | Change |
|---|---|
| `src/lib/types.ts` | Add `TrustLevel`, `TrustSettings` |
| `src/lib/components/MediaFeed.svelte` | Accept `trustService`, gate rendering |
| `src/lib/components/MediaCard.svelte` | Add security attributes, sanitize hrefs |
| `src/lib/components/TextMessage.svelte` | Show locked pills, harden inline embeds |
| `src/lib/components/NotificationSettingsPanel.svelte` | Trust level dropdowns |
| `src/App.svelte` | Create `TrustService`, wire through |

---

## Out of Scope (YAGNI)

- OG metadata fetching (Preview level is skeleton only).
- Media proxy / relay server.
- CSP configuration (Tauri-level concern, separate bead).
- Blocklist / ban feature.
- "Trust all" bulk action.

---

## Accessibility

Accessibility is a design requirement for Harmony, not a nice-to-have.

- All interactive elements use semantic HTML (`<button>`, not `<div>` with
  role hacks) where possible.
- Keyboard navigation: all placeholder cards and confirmation buttons are
  focusable and operable via Enter and Space.
- Screen reader: placeholder cards announce "Blocked media, image, from
  Alice. Press Enter to show." via `aria-label`.
- The 1-second cooldown is communicated visually (grayed button) and via
  `aria-disabled` + `aria-live` announcement when the button becomes active.
