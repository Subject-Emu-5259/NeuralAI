# Neural Browser — Implementation Plan

## Goal
Replace the current Browser tab (Remote Mode: Chromium via `/api/browser/*`) with a real,
from-scratch web browser engine built inside NeuralAI, seeded by browser.engineering + Brubeck.
Eventual spin-out: standalone **Neural Browser** app.

## Phase 0 — DONE (offline in-page engine)
Files in `from-scratch/web_ui/static/js/engine/`:
- `html.js` — HTML tokenizer + DOM tree + UA stylesheet (BE ch.2–3, Brubeck ch.1–2)
- `css.js` — CSS tokenizer + cascade + selector match + color resolve (BE ch.5, Brubeck ch.3–4)
- `layout.js` — block + inline layout, real text measure, link hit-testing (BE ch.6, Brubeck ch.5–6)
- `paint.js` — canvas raster (BE ch.7, Brubeck ch.7)
- `engine.js` — `NeuralEngine.run(html, {width})` → laid-out root with `__links`
- `neural-browser.js` — controller for `#tab-browser`: Engine Mode toggle, URL/HTML input,
  zoom, click-to-follow-links. Remote Mode (`browser.js`) is UNTOUCHED.

Wired into `templates/index.html`: `#tab-browser` markup + 6 engine `<script>` tags.
Verified in node (jsdom + canvas): demo doc renders, inline link detected at correct coords.

## Phase 1 — Fetch real URLs, render in-engine
- Add `/api/browser/engine?url=` to `webui_service.py`: server-side fetch + HTML sanitize (strip
  scripts/event handlers for safety) → return raw HTML. Client runs it through the engine.
- Support `<img>` decode (draw rect/placeholder; later real decode), basic `<a>` history (push URL).
- Engine Mode becomes the default for "render this page offline".

## Phase 2 — Real networking + navigation
- In-browser fetch with CORS handling; HTTP status/headers (BE ch.1–2).
- Multi-tab engine state; back/forward stack (BE ch.10); virtual scrolling for tall pages.
- Re-layout on zoom (already wired to `_nbZoom`).

## Phase 3 — Scripts + forms
- Sandboxed JS runtime (restricted `eval` or QuickJS) + DOM API surface (BE ch.6, Brubeck ch.8).
- Forms: input/submit → navigation (BE ch.12).

## Phase 4 — Security + effects
- Origin isolation, sandbox, CSP-lite (BE ch.11). Visual effects: opacity/transform/compositing (BE ch.13).

## Phase 5 — Spin-out
- Extract `engine/` into a standalone **Neural Browser** package: own window, address bar,
  devtools, settings. Reuse the same 5-stage pipeline.

## Guardrails (from AGENTS.md)
- Do NOT break the Google-style chat UI or live `/api/browser/*` Remote endpoints.
- Engine Mode is additive until Phase 2+ security lands; Remote stays default for real browsing.
- Keep UA stylesheet minimal/correct; preserve link hit-testing.

## Verification
- Node harness (`/tmp/dbg2.js` style) confirms layout height > 0 and links detected.
- Live: open `https://neuralai-web-ui-deandrewharris.zocomputer.io` → Browser tab → Engine Mode
  checked → demo renders; click the browser.engineering link opens it in a new tab.
