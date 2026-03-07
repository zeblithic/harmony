# Harmony Browser Tauri Shell Design

**Goal:** Scaffold the `zeblithic/harmony-browser` Tauri v2 + Svelte 5 desktop app as a lean MVP: address bar, content pane with markdown rendering, and trust badges. No subscriptions, no privacy panel, no navigation history.

**Architecture:** Thin IPC bridge — Tauri commands wrap `BrowserCore::handle_event()`, perform any I/O the core requests (fetch content), then return serialized render results to the Svelte frontend. The frontend is a pure renderer with zero business logic.

**Tech Stack:** Tauri v2, Svelte 5 (runes), Vite, vitest, TypeScript, pulldown-cmark (Rust-side markdown rendering)

## Key Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| MVP scope | Address bar + content pane + trust badge | Lean; subscriptions and privacy panel deferred |
| Rust integration | Git dependency on harmony-browser crate | Same pattern as harmony-client; stable |
| Markdown rendering | Rust-side via pulldown-cmark | Keeps rendering logic testable in Rust, smaller JS bundle |
| Navigation history | Not in MVP | Pure UI state, easy to add later |
| Trust-gated content | Text placeholder + approve button | Simple, accessible; blurred previews deferred |
| Content source | Embedded test fixtures in Tauri crate | No network needed for MVP demo |

## Project Structure

```
harmony-browser/
├── package.json
├── vite.config.ts
├── vitest.config.ts
├── svelte.config.js
├── tsconfig.json
├── index.html
├── src/
│   ├── main.ts
│   ├── App.svelte
│   ├── app.css
│   └── lib/
│       ├── components/
│       │   ├── AddressBar.svelte
│       │   ├── ContentPane.svelte
│       │   └── TrustBadge.svelte
│       ├── browser-service.ts
│       ├── browser-service.test.ts
│       └── types.ts
├── src-tauri/
│   ├── Cargo.toml
│   ├── tauri.conf.json
│   ├── build.rs
│   └── src/
│       ├── main.rs
│       ├── lib.rs
│       └── fixtures.rs
└── LICENSE
```

## IPC Layer

The Tauri Rust side holds `BrowserCore` in managed state behind a `Mutex`. Two commands:

### `navigate(input: String) -> Result<ActionResponse, String>`

1. Parse input via `BrowseTarget::parse(&input)`
2. Feed `BrowserEvent::Navigate(target)` into `BrowserCore`
3. Handle resulting actions:
   - `FetchContent { cid }` — look up content (from fixtures in MVP), feed `ContentFetched` back into core
   - `QueryNamed { key_expr }` — resolve name to CID (from fixtures), then fetch as above
   - `Render(resolved)` — convert to `ActionResponse`
4. For markdown content, render to HTML via `pulldown-cmark` before returning
5. Return `ActionResponse` to frontend

### `approve_content(cid_hex: String) -> Result<ActionResponse, String>`

1. Parse CID from hex
2. Feed `BrowserEvent::ApproveContent { cid }` into core (marks as approved, emits `FetchContent`)
3. Fetch and render as above
4. Return `ActionResponse` with `trust_level: "full_trust"`

### ActionResponse

```rust
#[derive(Serialize)]
struct ActionResponse {
    cid: String,          // hex-encoded ContentId
    mime: String,         // "markdown" | "plain_text" | "image_png" | ...
    content_html: String, // pre-rendered HTML for markdown, raw text for plain
    trust_level: String,  // "full_trust" | "preview" | "untrusted" | "unknown"
}
```

## Frontend Components

### AddressBar

- Text input, Enter-to-navigate
- Calls `browserService.navigate(input)`, passes result to ContentPane
- Shows trust badge inline (left of input)
- Displays errors inline below the input
- Supports input formats: `hmy:<hex>`, `wiki/name`, `~presence/**`

### ContentPane

Renders `ActionResponse` based on MIME and trust level:

- **Markdown:** Renders `content_html` in a sandboxed `<article>` element
- **Plain text:** Wraps in `<pre>`
- **Trust-gated (untrusted/unknown):** Shows text placeholder "Content blocked -- author unknown/untrusted" with "Approve & Load" button
- **Empty state:** "Enter an address to browse the Harmony network"
- **Error state:** Shows error message from navigate failure

### TrustBadge

Visual indicator with four states:

| Trust Level | Visual | aria-label |
|---|---|---|
| full_trust | Green dot | "Fully trusted" |
| preview | Yellow dot | "Preview trust" |
| untrusted | Red dot | "Untrusted" |
| unknown | Gray dot | "Unknown author" |

## Content Fixtures

Embedded in `src-tauri/src/fixtures.rs` for the MVP demo. The `navigate` command resolves known paths to pre-built content:

| Path | Content | Trust Level |
|---|---|---|
| `wiki/hello` | Markdown document with headings and links | unknown (no author) |
| `wiki/trust-demo` | Markdown from a "known" author | full_trust |
| `plain/example` | Plain text blob | unknown |

Unknown paths return an error. Fixtures exercise the full pipeline: parse, BrowserCore event handling, markdown rendering, trust-level response.

## Testing Strategy

### Rust (cargo test)

- Tauri command unit tests: verify `navigate("hmy:...")` returns correct `ActionResponse` shape
- Markdown rendering: pulldown-cmark conversion (input markdown -> expected HTML)
- Error handling: invalid CID hex, empty input, unknown paths
- Fixture content: each fixture path returns expected MIME and trust level

### Frontend (vitest + @testing-library/svelte)

- `browser-service.test.ts` — Mock `@tauri-apps/api` invoke, verify correct command names and argument shapes
- `AddressBar` — Renders input, Enter triggers navigate, displays error state
- `ContentPane` — Renders markdown HTML, renders plain text in `<pre>`, shows trust-gated placeholder with approve button, shows empty state
- `TrustBadge` — Renders correct color/label for each trust level, has accessible `aria-label`

No E2E tests in MVP.

## Future Phases

| Phase | Scope |
|-------|-------|
| 3 | Subscription sidebar, live update cards, Tauri event streaming |
| 4 | Privacy panel, connection display, inline trust editing |
| 5 | WASM app hosting (requires UCANs) |
| 6 | Native renderer (Xilem/Makepad behind renderer trait) |
