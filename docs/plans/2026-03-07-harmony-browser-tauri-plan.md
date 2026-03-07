# Harmony Browser Tauri Shell Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Scaffold the `zeblithic/harmony-browser` Tauri v2 + Svelte 5 desktop app with address bar, markdown content pane, and trust badges.

**Architecture:** Thin IPC bridge — Tauri commands wrap `BrowserCore::handle_event()`, perform I/O (content fetch from fixtures), render markdown to HTML via pulldown-cmark, and return serialized results. The Svelte frontend is a pure renderer with zero business logic.

**Tech Stack:** Tauri v2, Svelte 5 (runes), Vite, vitest, TypeScript, pulldown-cmark, harmony-browser crate (git dep)

**Repo:** `/Users/zeblith/work/zeblithic/harmony-browser` — clean slate, just a LICENSE file.

**Reference:** `/Users/zeblith/work/zeblithic/harmony-client` — same tech stack, use as template for config files.

---

### Task 1: Scaffold the Tauri + Svelte project

**Files:**
- Create: `package.json`
- Create: `vite.config.ts`
- Create: `vitest.config.ts`
- Create: `svelte.config.js`
- Create: `tsconfig.json`
- Create: `index.html`
- Create: `src/main.ts`
- Create: `src/App.svelte`
- Create: `src/app.css`
- Create: `src/vite-env.d.ts`
- Create: `src-tauri/Cargo.toml`
- Create: `src-tauri/tauri.conf.json`
- Create: `src-tauri/build.rs`
- Create: `src-tauri/src/main.rs`
- Create: `src-tauri/src/lib.rs`

**Context:** This is a greenfield scaffold. Every file must be created from scratch. The repo currently contains only a LICENSE file. All work happens in `/Users/zeblith/work/zeblithic/harmony-browser`.

**Step 1: Create package.json**

```json
{
  "name": "harmony-browser",
  "version": "0.1.0",
  "description": "Content-addressed, trust-aware browser for the Harmony network",
  "type": "module",
  "scripts": {
    "dev": "vite",
    "build": "vite build",
    "preview": "vite preview",
    "test": "vitest run",
    "tauri": "tauri"
  },
  "license": "MIT",
  "devDependencies": {
    "@sveltejs/vite-plugin-svelte": "^6.2.4",
    "@tauri-apps/cli": "^2.10.1",
    "@testing-library/jest-dom": "^6.9.1",
    "@testing-library/svelte": "^5.3.1",
    "jsdom": "^28.1.0",
    "svelte": "^5.53.7",
    "typescript": "^5.9.3",
    "vite": "^7.3.1",
    "vitest": "^4.0.18"
  },
  "dependencies": {
    "@tauri-apps/api": "^2.10.1"
  }
}
```

**Step 2: Create vite.config.ts**

```typescript
import { defineConfig } from 'vite';
import { svelte } from '@sveltejs/vite-plugin-svelte';

export default defineConfig({
  plugins: [svelte()],
  clearScreen: false,
  server: {
    port: 5173,
    strictPort: true,
  },
  envPrefix: ['VITE_', 'TAURI_'],
  build: {
    target: 'esnext',
    minify: !process.env.TAURI_DEBUG ? 'esbuild' : false,
    sourcemap: !!process.env.TAURI_DEBUG,
  },
});
```

**Step 3: Create vitest.config.ts**

```typescript
import { defineConfig } from 'vitest/config';
import { svelte } from '@sveltejs/vite-plugin-svelte';

export default defineConfig({
  plugins: [svelte({ hot: !process.env.VITEST })],
  resolve: {
    conditions: ['browser'],
  },
  test: {
    environment: 'jsdom',
    globals: true,
    include: ['src/**/*.test.ts'],
  },
});
```

**Step 4: Create svelte.config.js**

```javascript
import { vitePreprocess } from '@sveltejs/vite-plugin-svelte';

export default {
  preprocess: vitePreprocess(),
};
```

**Step 5: Create tsconfig.json**

```json
{
  "compilerOptions": {
    "target": "ESNext",
    "useDefineForClassFields": true,
    "module": "ESNext",
    "resolveJsonModule": true,
    "allowJs": true,
    "checkJs": true,
    "isolatedModules": true,
    "moduleDetection": "force",
    "moduleResolution": "bundler",
    "strict": true,
    "noUnusedLocals": true,
    "noUnusedParameters": true,
    "noFallthroughCasesInSwitch": true,
    "verbatimModuleSyntax": true
  },
  "include": ["src/**/*.ts", "src/**/*.svelte"]
}
```

**Step 6: Create index.html**

```html
<!doctype html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Harmony Browser</title>
  </head>
  <body>
    <div id="app"></div>
    <script type="module" src="/src/main.ts"></script>
  </body>
</html>
```

**Step 7: Create src/vite-env.d.ts**

```typescript
/// <reference types="svelte" />
/// <reference types="vite/client" />
```

**Step 8: Create src/main.ts**

```typescript
import App from './App.svelte';
import { mount } from 'svelte';

const app = mount(App, {
  target: document.getElementById('app')!,
});

export default app;
```

**Step 9: Create src/App.svelte**

A minimal placeholder that proves the app renders:

```svelte
<script lang="ts">
</script>

<main>
  <h1>Harmony Browser</h1>
  <p>Content-addressed, trust-aware browsing.</p>
</main>

<style>
  main {
    font-family: system-ui, -apple-system, sans-serif;
    max-width: 960px;
    margin: 0 auto;
    padding: 2rem;
  }
</style>
```

**Step 10: Create src/app.css**

```css
:root {
  font-family: system-ui, -apple-system, sans-serif;
  color: #213547;
  background-color: #ffffff;
}

@media (prefers-color-scheme: dark) {
  :root {
    color: #e0e0e0;
    background-color: #1a1a2e;
  }
}

*,
*::before,
*::after {
  box-sizing: border-box;
}

body {
  margin: 0;
  min-height: 100vh;
}
```

**Step 11: Create src-tauri/build.rs**

```rust
fn main() {
    tauri_build::build();
}
```

**Step 12: Create src-tauri/Cargo.toml**

```toml
[package]
name = "harmony-browser-app"
version = "0.1.0"
description = "Harmony Browser — content-addressed, trust-aware browsing"
authors = ["zeblithic"]
edition = "2021"
rust-version = "1.75"

[build-dependencies]
tauri-build = { version = "2", features = [] }

[dependencies]
tauri = { version = "2", features = [] }
serde = { version = "1", features = ["derive"] }
serde_json = "1"
harmony-browser = { git = "https://github.com/zeblithic/harmony.git", package = "harmony-browser" }
harmony-content = { git = "https://github.com/zeblithic/harmony.git", package = "harmony-content" }
pulldown-cmark = "0.12"
```

Note: `harmony-content` is needed for `ContentId` and `BlobStore`/`BundleBuilder` used in fixtures.

**Step 13: Create src-tauri/tauri.conf.json**

```json
{
  "$schema": "https://raw.githubusercontent.com/tauri-apps/tauri/dev/crates/tauri-cli/config.schema.json",
  "productName": "Harmony Browser",
  "version": "0.1.0",
  "identifier": "net.zeblith.harmony-browser",
  "build": {
    "beforeDevCommand": "npm run dev",
    "beforeBuildCommand": "npm run build",
    "devUrl": "http://localhost:5173",
    "frontendDist": "../dist"
  },
  "app": {
    "windows": [
      {
        "title": "Harmony Browser",
        "width": 1024,
        "height": 768,
        "minWidth": 400,
        "minHeight": 300
      }
    ]
  }
}
```

**Step 14: Create src-tauri/src/main.rs**

```rust
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

fn main() {
    harmony_browser_app::run();
}
```

**Step 15: Create src-tauri/src/lib.rs**

Minimal bootstrap with no commands yet:

```rust
pub fn run() {
    tauri::Builder::default()
        .run(tauri::generate_context!())
        .expect("error while running harmony browser");
}
```

**Step 16: Install npm dependencies**

Run: `cd /Users/zeblith/work/zeblithic/harmony-browser && npm install`
Expected: `node_modules` created, `package-lock.json` generated.

**Step 17: Add .gitignore**

Create `.gitignore`:

```
node_modules/
dist/
src-tauri/target/
```

**Step 18: Verify the frontend builds**

Run: `cd /Users/zeblith/work/zeblithic/harmony-browser && npm run build`
Expected: `dist/` directory created with bundled assets.

**Step 19: Verify the Rust side compiles**

Run: `cd /Users/zeblith/work/zeblithic/harmony-browser/src-tauri && cargo check`
Expected: Compiles without errors (downloads harmony-browser git dep).

**Step 20: Commit**

```bash
cd /Users/zeblith/work/zeblithic/harmony-browser
git add -A
git commit -m "feat: scaffold Tauri v2 + Svelte 5 project

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 2: TypeScript types and browser service

**Files:**
- Create: `src/lib/types.ts`
- Create: `src/lib/browser-service.ts`
- Create: `src/lib/browser-service.test.ts`

**Context:** These TypeScript types mirror the Rust `ActionResponse` struct. The browser service wraps `@tauri-apps/api/core` invoke calls so components don't import Tauri directly. Tests mock the invoke function.

**Step 1: Write the failing test for browser-service**

Create `src/lib/browser-service.test.ts`:

```typescript
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { navigate, approveContent } from './browser-service';
import type { ActionResponse } from './types';

// Mock the Tauri invoke API
vi.mock('@tauri-apps/api/core', () => ({
  invoke: vi.fn(),
}));

import { invoke } from '@tauri-apps/api/core';
const mockInvoke = vi.mocked(invoke);

describe('browser-service', () => {
  beforeEach(() => {
    mockInvoke.mockReset();
  });

  describe('navigate', () => {
    it('calls invoke with navigate command and input', async () => {
      const response: ActionResponse = {
        cid: 'abc123',
        mime: 'markdown',
        content_html: '<p>hello</p>',
        trust_level: 'unknown',
      };
      mockInvoke.mockResolvedValue(response);

      const result = await navigate('wiki/hello');

      expect(mockInvoke).toHaveBeenCalledWith('navigate', { input: 'wiki/hello' });
      expect(result).toEqual(response);
    });

    it('propagates errors from invoke', async () => {
      mockInvoke.mockRejectedValue('Invalid CID');

      await expect(navigate('hmy:bad')).rejects.toBe('Invalid CID');
    });
  });

  describe('approveContent', () => {
    it('calls invoke with approve_content command and cid_hex', async () => {
      const response: ActionResponse = {
        cid: 'abc123',
        mime: 'markdown',
        content_html: '<p>hello</p>',
        trust_level: 'full_trust',
      };
      mockInvoke.mockResolvedValue(response);

      const result = await approveContent('abc123');

      expect(mockInvoke).toHaveBeenCalledWith('approve_content', { cidHex: 'abc123' });
      expect(result).toEqual(response);
    });
  });
});
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/zeblith/work/zeblithic/harmony-browser && npx vitest run`
Expected: FAIL — modules not found.

**Step 3: Create types.ts**

Create `src/lib/types.ts`:

```typescript
export interface ActionResponse {
  cid: string;
  mime: string;
  content_html: string;
  trust_level: 'full_trust' | 'preview' | 'untrusted' | 'unknown';
}
```

**Step 4: Create browser-service.ts**

Create `src/lib/browser-service.ts`:

```typescript
import { invoke } from '@tauri-apps/api/core';
import type { ActionResponse } from './types';

export async function navigate(input: string): Promise<ActionResponse> {
  return invoke<ActionResponse>('navigate', { input });
}

export async function approveContent(cidHex: string): Promise<ActionResponse> {
  return invoke<ActionResponse>('approve_content', { cidHex });
}
```

**Step 5: Run tests to verify they pass**

Run: `cd /Users/zeblith/work/zeblithic/harmony-browser && npx vitest run`
Expected: 3 tests PASS.

**Step 6: Commit**

```bash
cd /Users/zeblith/work/zeblithic/harmony-browser
git add src/lib/types.ts src/lib/browser-service.ts src/lib/browser-service.test.ts
git commit -m "feat: TypeScript types and browser service with Tauri invoke wrapper

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 3: TrustBadge component

**Files:**
- Create: `src/lib/components/TrustBadge.svelte`
- Create: `src/lib/components/__tests__/TrustBadge.test.ts`

**Context:** A small visual indicator showing trust level as a colored dot with accessible label. Four states: full_trust (green), preview (yellow), untrusted (red), unknown (gray). Uses Svelte 5 runes (`$props()`).

**Step 1: Write the failing test**

Create `src/lib/components/__tests__/TrustBadge.test.ts`:

```typescript
import { render, screen } from '@testing-library/svelte';
import { describe, it, expect } from 'vitest';
import TrustBadge from '../TrustBadge.svelte';

describe('TrustBadge', () => {
  it('renders green dot for full_trust', () => {
    render(TrustBadge, { props: { level: 'full_trust' } });
    const badge = screen.getByRole('img', { name: 'Fully trusted' });
    expect(badge).toBeTruthy();
    expect(badge.style.backgroundColor).toBe('var(--trust-full)');
  });

  it('renders yellow dot for preview', () => {
    render(TrustBadge, { props: { level: 'preview' } });
    const badge = screen.getByRole('img', { name: 'Preview trust' });
    expect(badge).toBeTruthy();
    expect(badge.style.backgroundColor).toBe('var(--trust-preview)');
  });

  it('renders red dot for untrusted', () => {
    render(TrustBadge, { props: { level: 'untrusted' } });
    const badge = screen.getByRole('img', { name: 'Untrusted' });
    expect(badge).toBeTruthy();
    expect(badge.style.backgroundColor).toBe('var(--trust-untrusted)');
  });

  it('renders gray dot for unknown', () => {
    render(TrustBadge, { props: { level: 'unknown' } });
    const badge = screen.getByRole('img', { name: 'Unknown author' });
    expect(badge).toBeTruthy();
    expect(badge.style.backgroundColor).toBe('var(--trust-unknown)');
  });
});
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/zeblith/work/zeblithic/harmony-browser && npx vitest run`
Expected: FAIL — component not found.

**Step 3: Implement TrustBadge.svelte**

Create `src/lib/components/TrustBadge.svelte`:

```svelte
<script lang="ts">
  type TrustLevel = 'full_trust' | 'preview' | 'untrusted' | 'unknown';

  const LABELS: Record<TrustLevel, string> = {
    full_trust: 'Fully trusted',
    preview: 'Preview trust',
    untrusted: 'Untrusted',
    unknown: 'Unknown author',
  };

  const COLORS: Record<TrustLevel, string> = {
    full_trust: 'var(--trust-full)',
    preview: 'var(--trust-preview)',
    untrusted: 'var(--trust-untrusted)',
    unknown: 'var(--trust-unknown)',
  };

  let { level }: { level: TrustLevel } = $props();
</script>

<span
  role="img"
  aria-label={LABELS[level]}
  class="trust-badge"
  style:background-color={COLORS[level]}
></span>

<style>
  .trust-badge {
    display: inline-block;
    width: 12px;
    height: 12px;
    border-radius: 50%;
    flex-shrink: 0;
  }
</style>
```

Also add the CSS custom properties to `src/app.css`:

```css
:root {
  --trust-full: #22c55e;
  --trust-preview: #eab308;
  --trust-untrusted: #ef4444;
  --trust-unknown: #9ca3af;
}
```

**Step 4: Run tests to verify they pass**

Run: `cd /Users/zeblith/work/zeblithic/harmony-browser && npx vitest run`
Expected: 4 TrustBadge tests + 3 browser-service tests PASS.

**Step 5: Commit**

```bash
cd /Users/zeblith/work/zeblithic/harmony-browser
git add src/lib/components/TrustBadge.svelte src/lib/components/__tests__/TrustBadge.test.ts src/app.css
git commit -m "feat: TrustBadge component with accessible labels and trust colors

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 4: ContentPane component

**Files:**
- Create: `src/lib/components/ContentPane.svelte`
- Create: `src/lib/components/__tests__/ContentPane.test.ts`

**Context:** Renders content based on MIME type and trust level. Four states: markdown (rendered HTML), plain text (`<pre>`), trust-gated (placeholder + approve button), and empty (welcome message). The component receives an `ActionResponse` or `null` and an error string or `null`.

**Step 1: Write the failing test**

Create `src/lib/components/__tests__/ContentPane.test.ts`:

```typescript
import { render, screen } from '@testing-library/svelte';
import { describe, it, expect } from 'vitest';
import ContentPane from '../ContentPane.svelte';
import type { ActionResponse } from '../../types';

describe('ContentPane', () => {
  it('shows empty state when no content', () => {
    render(ContentPane, { props: { content: null, error: null } });
    expect(screen.getByText('Enter an address to browse the Harmony network')).toBeTruthy();
  });

  it('shows error message', () => {
    render(ContentPane, { props: { content: null, error: 'Content not found' } });
    expect(screen.getByRole('alert')).toBeTruthy();
    expect(screen.getByText('Content not found')).toBeTruthy();
  });

  it('renders markdown content as HTML', () => {
    const content: ActionResponse = {
      cid: 'abc123',
      mime: 'markdown',
      content_html: '<h1>Hello World</h1>\n<p>Some content</p>\n',
      trust_level: 'unknown',
    };
    render(ContentPane, { props: { content, error: null } });
    const article = screen.getByRole('article');
    expect(article).toBeTruthy();
    expect(article.innerHTML).toContain('<h1>Hello World</h1>');
  });

  it('renders plain text in a pre element', () => {
    const content: ActionResponse = {
      cid: 'abc123',
      mime: 'plain_text',
      content_html: 'just plain text',
      trust_level: 'unknown',
    };
    render(ContentPane, { props: { content, error: null } });
    const pre = document.querySelector('pre');
    expect(pre).toBeTruthy();
    expect(pre!.textContent).toBe('just plain text');
  });

  it('shows trust-gated placeholder for untrusted content', () => {
    const content: ActionResponse = {
      cid: 'abc123',
      mime: 'markdown',
      content_html: '<p>secret stuff</p>',
      trust_level: 'untrusted',
    };
    render(ContentPane, { props: { content, error: null } });
    expect(screen.getByText(/Content blocked/)).toBeTruthy();
    expect(screen.getByRole('button', { name: /Approve/i })).toBeTruthy();
  });

  it('renders full content for full_trust', () => {
    const content: ActionResponse = {
      cid: 'abc123',
      mime: 'markdown',
      content_html: '<p>trusted content</p>',
      trust_level: 'full_trust',
    };
    render(ContentPane, { props: { content, error: null } });
    const article = screen.getByRole('article');
    expect(article.innerHTML).toContain('<p>trusted content</p>');
  });

  it('renders full content for unknown trust (text is safe)', () => {
    const content: ActionResponse = {
      cid: 'abc123',
      mime: 'markdown',
      content_html: '<p>unknown author content</p>',
      trust_level: 'unknown',
    };
    render(ContentPane, { props: { content, error: null } });
    const article = screen.getByRole('article');
    expect(article.innerHTML).toContain('<p>unknown author content</p>');
  });
});
```

**Design note:** In the MVP, text content (markdown, plain text) renders regardless of trust level because text is safe. Only `untrusted` trust level gates content behind approval — this matches the design doc where `Preview` shows text but gates media. Since we have no media in the MVP, `unknown` and `preview` both show text freely, and only `untrusted` blocks.

**Step 2: Run test to verify it fails**

Run: `cd /Users/zeblith/work/zeblithic/harmony-browser && npx vitest run`
Expected: FAIL — component not found.

**Step 3: Implement ContentPane.svelte**

Create `src/lib/components/ContentPane.svelte`:

```svelte
<script lang="ts">
  import type { ActionResponse } from '../types';
  import TrustBadge from './TrustBadge.svelte';

  let {
    content,
    error,
    onapprove,
  }: {
    content: ActionResponse | null;
    error: string | null;
    onapprove?: (cidHex: string) => void;
  } = $props();

  function handleApprove() {
    if (content && onapprove) {
      onapprove(content.cid);
    }
  }
</script>

<section class="content-pane">
  {#if error}
    <div role="alert" class="error">{error}</div>
  {:else if !content}
    <p class="empty">Enter an address to browse the Harmony network</p>
  {:else if content.trust_level === 'untrusted'}
    <div class="gated">
      <TrustBadge level={content.trust_level} />
      <p>Content blocked — author untrusted</p>
      <button onclick={handleApprove}>Approve & Load</button>
    </div>
  {:else if content.mime === 'plain_text'}
    <pre>{content.content_html}</pre>
  {:else}
    <article>{@html content.content_html}</article>
  {/if}
</section>

<style>
  .content-pane {
    padding: 1rem;
    flex: 1;
    overflow-y: auto;
  }

  .empty {
    color: var(--trust-unknown);
    text-align: center;
    margin-top: 4rem;
  }

  .error {
    color: var(--trust-untrusted);
    padding: 1rem;
    border: 1px solid var(--trust-untrusted);
    border-radius: 4px;
  }

  .gated {
    text-align: center;
    margin-top: 4rem;
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 1rem;
  }

  .gated button {
    padding: 0.5rem 1rem;
    border-radius: 4px;
    border: 1px solid currentColor;
    background: transparent;
    cursor: pointer;
  }

  article {
    line-height: 1.6;
  }

  pre {
    white-space: pre-wrap;
    word-break: break-word;
    background: #f5f5f5;
    padding: 1rem;
    border-radius: 4px;
  }

  @media (prefers-color-scheme: dark) {
    pre {
      background: #2a2a3e;
    }
  }
</style>
```

**Step 4: Run tests to verify they pass**

Run: `cd /Users/zeblith/work/zeblithic/harmony-browser && npx vitest run`
Expected: 7 ContentPane tests + prior tests PASS.

**Step 5: Commit**

```bash
cd /Users/zeblith/work/zeblithic/harmony-browser
git add src/lib/components/ContentPane.svelte src/lib/components/__tests__/ContentPane.test.ts
git commit -m "feat: ContentPane component with markdown, plain text, and trust-gated rendering

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 5: AddressBar component

**Files:**
- Create: `src/lib/components/AddressBar.svelte`
- Create: `src/lib/components/__tests__/AddressBar.test.ts`

**Context:** Text input with Enter-to-navigate. Shows a TrustBadge for the current content's trust level. Fires an `onnavigate` callback with the input value.

**Step 1: Write the failing test**

Create `src/lib/components/__tests__/AddressBar.test.ts`:

```typescript
import { render, screen, fireEvent } from '@testing-library/svelte';
import { describe, it, expect, vi } from 'vitest';
import AddressBar from '../AddressBar.svelte';

describe('AddressBar', () => {
  it('renders a text input', () => {
    render(AddressBar, { props: { trustLevel: null, onnavigate: vi.fn() } });
    expect(screen.getByRole('textbox')).toBeTruthy();
  });

  it('has accessible label', () => {
    render(AddressBar, { props: { trustLevel: null, onnavigate: vi.fn() } });
    expect(screen.getByLabelText('Address')).toBeTruthy();
  });

  it('calls onnavigate with input value on Enter', async () => {
    const onnavigate = vi.fn();
    render(AddressBar, { props: { trustLevel: null, onnavigate } });
    const input = screen.getByRole('textbox');
    await fireEvent.input(input, { target: { value: 'wiki/hello' } });
    await fireEvent.keyDown(input, { key: 'Enter' });
    expect(onnavigate).toHaveBeenCalledWith('wiki/hello');
  });

  it('does not call onnavigate on non-Enter keys', async () => {
    const onnavigate = vi.fn();
    render(AddressBar, { props: { trustLevel: null, onnavigate } });
    const input = screen.getByRole('textbox');
    await fireEvent.input(input, { target: { value: 'wiki/hello' } });
    await fireEvent.keyDown(input, { key: 'a' });
    expect(onnavigate).not.toHaveBeenCalled();
  });

  it('shows TrustBadge when trustLevel is provided', () => {
    render(AddressBar, { props: { trustLevel: 'full_trust', onnavigate: vi.fn() } });
    expect(screen.getByRole('img', { name: 'Fully trusted' })).toBeTruthy();
  });

  it('does not show TrustBadge when trustLevel is null', () => {
    render(AddressBar, { props: { trustLevel: null, onnavigate: vi.fn() } });
    expect(screen.queryByRole('img')).toBeNull();
  });
});
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/zeblith/work/zeblithic/harmony-browser && npx vitest run`
Expected: FAIL — component not found.

**Step 3: Implement AddressBar.svelte**

Create `src/lib/components/AddressBar.svelte`:

```svelte
<script lang="ts">
  import TrustBadge from './TrustBadge.svelte';

  type TrustLevel = 'full_trust' | 'preview' | 'untrusted' | 'unknown';

  let {
    trustLevel,
    onnavigate,
  }: {
    trustLevel: TrustLevel | null;
    onnavigate: (input: string) => void;
  } = $props();

  let inputValue = $state('');

  function handleKeyDown(e: KeyboardEvent) {
    if (e.key === 'Enter') {
      onnavigate(inputValue);
    }
  }
</script>

<nav class="address-bar">
  {#if trustLevel}
    <TrustBadge level={trustLevel} />
  {/if}
  <label>
    <span class="sr-only">Address</span>
    <input
      type="text"
      bind:value={inputValue}
      onkeydown={handleKeyDown}
      placeholder="hmy:... or wiki/topic or ~presence/**"
      aria-label="Address"
    />
  </label>
</nav>

<style>
  .address-bar {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.5rem 1rem;
    border-bottom: 1px solid #e0e0e0;
  }

  label {
    flex: 1;
  }

  input {
    width: 100%;
    padding: 0.5rem;
    font-family: monospace;
    font-size: 0.9rem;
    border: 1px solid #ccc;
    border-radius: 4px;
    background: inherit;
    color: inherit;
  }

  .sr-only {
    position: absolute;
    width: 1px;
    height: 1px;
    padding: 0;
    margin: -1px;
    overflow: hidden;
    clip: rect(0, 0, 0, 0);
    white-space: nowrap;
    border: 0;
  }

  @media (prefers-color-scheme: dark) {
    .address-bar {
      border-bottom-color: #333;
    }

    input {
      border-color: #555;
    }
  }
</style>
```

**Step 4: Run tests to verify they pass**

Run: `cd /Users/zeblith/work/zeblithic/harmony-browser && npx vitest run`
Expected: 6 AddressBar tests + prior tests PASS.

**Step 5: Commit**

```bash
cd /Users/zeblith/work/zeblithic/harmony-browser
git add src/lib/components/AddressBar.svelte src/lib/components/__tests__/AddressBar.test.ts
git commit -m "feat: AddressBar component with Enter-to-navigate and TrustBadge

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 6: Wire up App.svelte

**Files:**
- Modify: `src/App.svelte`

**Context:** Connect AddressBar and ContentPane with browser-service. App.svelte is the composition root — it holds the current content state and wires the navigate/approve callbacks.

**Step 1: Update App.svelte**

```svelte
<script lang="ts">
  import './app.css';
  import AddressBar from './lib/components/AddressBar.svelte';
  import ContentPane from './lib/components/ContentPane.svelte';
  import { navigate, approveContent } from './lib/browser-service';
  import type { ActionResponse } from './lib/types';

  let content = $state<ActionResponse | null>(null);
  let error = $state<string | null>(null);

  async function handleNavigate(input: string) {
    error = null;
    content = null;
    try {
      content = await navigate(input);
    } catch (e) {
      error = String(e);
    }
  }

  async function handleApprove(cidHex: string) {
    error = null;
    try {
      content = await approveContent(cidHex);
    } catch (e) {
      error = String(e);
    }
  }
</script>

<main>
  <AddressBar
    trustLevel={content?.trust_level ?? null}
    onnavigate={handleNavigate}
  />
  <ContentPane {content} {error} onapprove={handleApprove} />
</main>

<style>
  main {
    display: flex;
    flex-direction: column;
    height: 100vh;
  }
</style>
```

**Step 2: Verify the frontend still builds**

Run: `cd /Users/zeblith/work/zeblithic/harmony-browser && npm run build`
Expected: Builds without errors.

**Step 3: Verify all tests still pass**

Run: `cd /Users/zeblith/work/zeblithic/harmony-browser && npx vitest run`
Expected: All tests pass.

**Step 4: Commit**

```bash
cd /Users/zeblith/work/zeblithic/harmony-browser
git add src/App.svelte
git commit -m "feat: wire App.svelte with AddressBar, ContentPane, and browser-service

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 7: Rust content fixtures

**Files:**
- Create: `src-tauri/src/fixtures.rs`

**Context:** Embedded test content for the MVP demo. Builds real content bundles using `harmony-content` APIs so the full pipeline is exercised. Each fixture has a known named path, pre-built bundle bytes, and a trust scenario.

**Step 1: Create fixtures.rs**

Create `src-tauri/src/fixtures.rs`:

```rust
use harmony_content::blob::{BlobStore, MemoryBlobStore};
use harmony_content::bundle::BundleBuilder;
use harmony_content::cid::ContentId;

pub struct Fixture {
    pub cid: ContentId,
    pub data: Vec<u8>,
}

/// Resolve a named path to a fixture.
/// Returns None if the path is unknown.
pub fn resolve_named(key_expr: &str) -> Option<Fixture> {
    match key_expr {
        "harmony/content/wiki/hello" => Some(build_markdown_fixture(
            b"# Hello, Harmony!\n\nWelcome to the decentralized web.\n\n## What is this?\n\nThis is a content-addressed document. Its identity is its hash.\nNo servers, no cookies, no tracking.\n",
        )),
        "harmony/content/wiki/trust-demo" => Some(build_markdown_fixture(
            b"# Trust Demo\n\nThis content comes from a **trusted author**.\n\nImages and media would load automatically at this trust level.\n",
        )),
        "harmony/content/plain/example" => Some(build_plain_fixture(
            b"Just plain bytes.\nNo formatting, no markup.\nContent-addressed and tamper-proof.",
        )),
        _ => None,
    }
}

fn build_markdown_fixture(content: &[u8]) -> Fixture {
    let mut store = MemoryBlobStore::new();
    let blob_cid = store.insert(content).unwrap();
    let mut builder = BundleBuilder::new();
    builder.add(blob_cid);
    builder.with_metadata(content.len() as u64, 1, 1000, *b"text/md\0");
    let (data, cid) = builder.build().unwrap();
    Fixture { cid, data }
}

fn build_plain_fixture(content: &[u8]) -> Fixture {
    let mut store = MemoryBlobStore::new();
    let blob_cid = store.insert(content).unwrap();
    let mut builder = BundleBuilder::new();
    builder.add(blob_cid);
    builder.with_metadata(content.len() as u64, 1, 1000, *b"text/pln");
    let (data, cid) = builder.build().unwrap();
    Fixture { cid, data }
}
```

**Step 2: Add `mod fixtures;` to lib.rs**

Add to `src-tauri/src/lib.rs`:

```rust
mod fixtures;
```

**Step 3: Verify Rust compiles**

Run: `cd /Users/zeblith/work/zeblithic/harmony-browser/src-tauri && cargo check`
Expected: Compiles without errors.

**Step 4: Commit**

```bash
cd /Users/zeblith/work/zeblithic/harmony-browser
git add src-tauri/src/fixtures.rs src-tauri/src/lib.rs
git commit -m "feat: embedded content fixtures for MVP demo

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 8: Tauri commands (navigate + approve)

**Files:**
- Modify: `src-tauri/src/lib.rs`
- Modify: `src-tauri/Cargo.toml` (add `std::sync::Mutex`)

**Context:** The two Tauri commands that bridge `BrowserCore` to the frontend. The `navigate` command parses input, runs it through the core state machine, handles the resulting actions (fetch from fixtures), renders markdown to HTML, and returns an `ActionResponse`. The `approve_content` command marks a CID as approved and re-fetches.

**Step 1: Update lib.rs with full command implementations**

Replace `src-tauri/src/lib.rs`:

```rust
use std::sync::Mutex;

use harmony_browser::{BrowserAction, BrowserCore, BrowserEvent, BrowseTarget, MimeHint, ResolvedContent};
use serde::Serialize;
use tauri::State;

mod fixtures;

#[derive(Debug, Clone, Serialize)]
pub struct ActionResponse {
    cid: String,
    mime: String,
    content_html: String,
    trust_level: String,
}

fn mime_to_string(mime: &MimeHint) -> String {
    match mime {
        MimeHint::Markdown => "markdown".into(),
        MimeHint::PlainText => "plain_text".into(),
        MimeHint::Image(_) => "image".into(),
        MimeHint::HarmonyApp => "harmony_app".into(),
        MimeHint::Unknown(_) => "unknown".into(),
    }
}

fn trust_to_string(trust: &harmony_browser::TrustDecision) -> String {
    match trust {
        harmony_browser::TrustDecision::FullTrust => "full_trust".into(),
        harmony_browser::TrustDecision::Preview => "preview".into(),
        harmony_browser::TrustDecision::Untrusted => "untrusted".into(),
        harmony_browser::TrustDecision::Unknown => "unknown".into(),
    }
}

fn render_markdown(md: &str) -> String {
    let parser = pulldown_cmark::Parser::new(md);
    let mut html = String::new();
    pulldown_cmark::html::push_html(&mut html, parser);
    html
}

fn resolve_render_action(action: BrowserAction) -> Option<ActionResponse> {
    match action {
        BrowserAction::Render(ResolvedContent::Static {
            cid, mime, data, trust_level, ..
        }) => {
            let content_html = match mime {
                MimeHint::Markdown => {
                    let text = String::from_utf8_lossy(&data);
                    render_markdown(&text)
                }
                _ => String::from_utf8_lossy(&data).into_owned(),
            };
            Some(ActionResponse {
                cid: hex::encode(cid.as_bytes()),
                mime: mime_to_string(&mime),
                content_html,
                trust_level: trust_to_string(&trust_level),
            })
        }
        _ => None,
    }
}

#[tauri::command]
fn navigate(state: State<'_, Mutex<BrowserCore>>, input: String) -> Result<ActionResponse, String> {
    let target = BrowseTarget::parse(&input).map_err(|e| e.to_string())?;
    let mut core = state.lock().unwrap();
    let actions = core.handle_event(BrowserEvent::Navigate(target));

    for action in actions {
        match action {
            BrowserAction::FetchContent { cid } => {
                // Look up in fixtures by CID
                // For MVP, we resolve named paths first, so this handles direct CID navigation
                let hex_cid = hex::encode(cid.as_bytes());
                return Err(format!("Direct CID fetch not yet supported: {}", hex_cid));
            }
            BrowserAction::QueryNamed { key_expr } => {
                // Resolve named path from fixtures
                let fixture = fixtures::resolve_named(&key_expr)
                    .ok_or_else(|| format!("Not found: {}", key_expr))?;

                // Feed the content back into the core
                let render_actions = core.handle_event(BrowserEvent::ContentFetched {
                    cid: fixture.cid,
                    data: fixture.data,
                });

                for ra in render_actions {
                    if let Some(response) = resolve_render_action(ra) {
                        return Ok(response);
                    }
                }
                return Err("Content resolved but could not render".into());
            }
            _ => {}
        }
    }

    Err("No actionable result".into())
}

#[tauri::command]
fn approve_content(
    state: State<'_, Mutex<BrowserCore>>,
    cid_hex: String,
) -> Result<ActionResponse, String> {
    let bytes = hex::decode(&cid_hex).map_err(|e| format!("Invalid hex: {}", e))?;
    if bytes.len() != 32 {
        return Err(format!("Expected 32 bytes, got {}", bytes.len()));
    }
    let mut arr = [0u8; 32];
    arr.copy_from_slice(&bytes);
    let cid = harmony_content::cid::ContentId::from_bytes(arr);

    let mut core = state.lock().unwrap();
    let _actions = core.handle_event(BrowserEvent::ApproveContent { cid });

    // For MVP, re-navigate would be needed. Return a simple acknowledgment.
    Err("Approve noted — re-navigate to see updated trust level".into())
}

pub fn run() {
    tauri::Builder::default()
        .manage(Mutex::new(BrowserCore::new()))
        .invoke_handler(tauri::generate_handler![navigate, approve_content])
        .run(tauri::generate_context!())
        .expect("error while running harmony browser");
}
```

**Step 2: Add hex dependency to Cargo.toml**

Add `hex = "0.4"` to `[dependencies]` in `src-tauri/Cargo.toml`.

**Step 3: Verify Rust compiles**

Run: `cd /Users/zeblith/work/zeblithic/harmony-browser/src-tauri && cargo check`
Expected: Compiles without errors.

**Step 4: Verify Rust tests pass**

Run: `cd /Users/zeblith/work/zeblithic/harmony-browser/src-tauri && cargo test`
Expected: Compiles and any tests pass.

**Step 5: Commit**

```bash
cd /Users/zeblith/work/zeblithic/harmony-browser
git add src-tauri/src/lib.rs src-tauri/Cargo.toml
git commit -m "feat: Tauri navigate and approve_content commands with fixture resolution

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 9: Rust-side unit tests

**Files:**
- Modify: `src-tauri/src/lib.rs` (add `#[cfg(test)]` module)

**Context:** Test the markdown rendering and the conversion functions independently from Tauri managed state.

**Step 1: Add tests module to lib.rs**

Append to `src-tauri/src/lib.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn render_markdown_basic() {
        let html = render_markdown("# Hello\n\nWorld");
        assert!(html.contains("<h1>Hello</h1>"));
        assert!(html.contains("<p>World</p>"));
    }

    #[test]
    fn render_markdown_with_bold() {
        let html = render_markdown("This is **bold** text");
        assert!(html.contains("<strong>bold</strong>"));
    }

    #[test]
    fn render_markdown_with_link() {
        let html = render_markdown("[click](hmy:abc123)");
        assert!(html.contains("href=\"hmy:abc123\""));
    }

    #[test]
    fn mime_to_string_values() {
        assert_eq!(mime_to_string(&MimeHint::Markdown), "markdown");
        assert_eq!(mime_to_string(&MimeHint::PlainText), "plain_text");
    }

    #[test]
    fn trust_to_string_values() {
        assert_eq!(trust_to_string(&harmony_browser::TrustDecision::FullTrust), "full_trust");
        assert_eq!(trust_to_string(&harmony_browser::TrustDecision::Unknown), "unknown");
    }

    #[test]
    fn fixtures_resolve_known_paths() {
        assert!(fixtures::resolve_named("harmony/content/wiki/hello").is_some());
        assert!(fixtures::resolve_named("harmony/content/wiki/trust-demo").is_some());
        assert!(fixtures::resolve_named("harmony/content/plain/example").is_some());
    }

    #[test]
    fn fixtures_unknown_path_returns_none() {
        assert!(fixtures::resolve_named("harmony/content/nonexistent").is_none());
    }
}
```

**Step 2: Run Rust tests**

Run: `cd /Users/zeblith/work/zeblithic/harmony-browser/src-tauri && cargo test`
Expected: 7 tests pass.

**Step 3: Commit**

```bash
cd /Users/zeblith/work/zeblithic/harmony-browser
git add src-tauri/src/lib.rs
git commit -m "test: Rust-side unit tests for markdown rendering, type conversion, and fixtures

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 10: Final verification and push

**Files:** None new — verification only.

**Step 1: Run all frontend tests**

Run: `cd /Users/zeblith/work/zeblithic/harmony-browser && npx vitest run`
Expected: All tests pass (browser-service: 3, TrustBadge: 4, ContentPane: 7, AddressBar: 6 = 20 tests).

**Step 2: Run all Rust tests**

Run: `cd /Users/zeblith/work/zeblithic/harmony-browser/src-tauri && cargo test`
Expected: 7 tests pass.

**Step 3: Run Rust clippy**

Run: `cd /Users/zeblith/work/zeblithic/harmony-browser/src-tauri && cargo clippy`
Expected: Zero warnings.

**Step 4: Verify frontend build**

Run: `cd /Users/zeblith/work/zeblithic/harmony-browser && npm run build`
Expected: Builds successfully.

**Step 5: Push to origin**

```bash
cd /Users/zeblith/work/zeblithic/harmony-browser
git push -u origin main
```
