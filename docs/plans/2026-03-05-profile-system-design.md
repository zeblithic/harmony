# Profile System Design

Bead: `harmony-m6dp` — Client UX: profile system with CID-referenced avatars and notification sounds

## Goal

Add visual identity (SVG identicon avatars) and sonic identity infrastructure (notification sound override chain) to the Harmony client prototype. Establishes the Profile data model with CID-ready fields for future CAS integration.

## Scope

**In scope:**
- Profile data model with CID-ready avatar and sound fields
- SVG-based deterministic identicon generation from peer address
- Profile popover on avatar click (display name, status text, identicon, sound slots)
- Sound override chain: per-peer > per-community > sender default > client default
- Status text field on profiles
- Avatars in text feed, media feed headers, and nav panel DM entries

**Out of scope:**
- Actual image upload or CAS storage (no real CIDs yet)
- Audio playback (data model only for sounds)
- Profile editing UI (profiles are mock data)

## Architecture

### 1. Profile Data Model

New `Profile` type extends the existing `Peer` interface:

```typescript
export interface Profile extends Peer {
  statusText?: string;              // max 128 chars
  avatarCid?: string;               // Future: CID for 1024x1024 source
  avatarMiniCid?: string;           // Future: CID for 256x256 mini
  notificationSounds?: {
    quiet?: string;                 // Future: CID for quiet sound file
    standard?: string;              // Future: CID for standard sound file
    loud?: string;                  // Future: CID for loud sound file
  };
}
```

CID fields are optional and undefined in this prototype. The identicon is always generated from the address as the fallback.

Profile store: `Map<string, Profile>` keyed by peer address, populated from mock data.

Messages continue to carry `sender: Peer`. The app looks up the full `Profile` from the store when extended info is needed.

### 2. SVG Identicon Generation

Pure function in `identicon.ts`:

- Hash the peer address to get deterministic bytes
- Derive foreground color: HSL with hue from hash, fixed saturation/lightness
- Generate 5x5 grid pattern: only 3 columns computed (columns 0-1 mirror columns 4-3, column 2 is center) for symmetric "face-like" patterns
- Each cell on/off based on hash bits
- Render as inline SVG `<rect>` elements

The symmetry makes identicons recognizable and memorable. At 24px micro size each grid cell is ~4px, still readable.

### 3. Avatar Component Upgrade

`Avatar.svelte` keeps the same prop interface (`address`, `size`, `displayName`):

- Adds optional `avatarUrl` prop for future CID-resolved images
- If `avatarUrl` set: renders `<img>` with circular clip
- If not: renders SVG identicon inside circular clip
- Works at all sizes: 20px (nav), 24px (text feed), 64px (popover)

### 4. Profile Popover

New `ProfilePopover.svelte` component:

- Trigger: clicking any Avatar anywhere in the app
- Content: large identicon (64px), display name, status text, truncated peer address (monospace), sound slot display ("System default" for each)
- Positioning: absolute, near the click target (right or below)
- Dismiss: click outside or Escape
- Rendered once at app level, receives target profile + position as props

### 5. Sound Override Chain

Extends `NotificationService` with `resolveSoundCid()`:

```
1. Receiver's per-peer sound override    → check first
2. Receiver's per-community sound override → check second
3. Sender's profile default sound        → check third
4. Client's global default               → undefined (system default)
```

New type:

```typescript
export interface SoundOverrides {
  quiet?: string;    // CID
  standard?: string;
  loud?: string;
}
```

Per-peer and per-community overrides gain optional `soundOverrides` alongside existing notification policy.

No actual audio playback — just the resolution chain returning which CID would be played.

### 6. Integration Points

**Nav panel DM entries:** 20px Avatar + status text subtitle under peer name.

**MediaFeed card headers:** 20px Avatar before sender name.

**NotificationSettingsPanel:** Each peer/community section gains a "Custom sounds" area showing "System default" for each slot (placeholder for future audio).

## Testing

- `identicon.ts`: deterministic output, symmetry, different addresses produce different patterns
- `ProfilePopover`: render, dismiss on Escape, display of all profile fields
- `NotificationService.resolveSoundCid()`: 4-tier override chain resolution
- Existing Avatar tests updated for identicon rendering
