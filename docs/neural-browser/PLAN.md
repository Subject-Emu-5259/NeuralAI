# Neural Browser — Implementation Plan (replace the NeuralAI Browser tab)

Goal: Replace the current **screenshot + link-rectangle** Browser tab with a **real, working
in-app browser** that can later become a standalone **Neural Browser** application.

Current state (verified 2026-07-19):
- `from-scratch/web_ui/templates/index.html` → Browser tab markup at lines ~3578–3660 (tab
  strip, omni bar, bookmarks, screenshot `<img id="browserScreenshot">`, placeholder).
- `from-scratch/web_ui/static/js/browser.js` → drives Playwright backend; receives PNG +
  `_linkRects` and overlays clickable regions. This is a *remote view*, not a real browser.
- Backend `tools/web_browser.py` + `services/webui_service.py` routes (`/api/browser/*`):
  `health, tabs, new, close-tab, select, navigate, action, run(SSE), close`. Per-user
  Playwright session already works.

## Strategy: Hybrid (real now, engine later)

Phase 1 ships a genuinely interactive browser using the existing Chromium backend but
rendered through a **live proxy iframe** instead of a screenshot. Phase 2 adds a from-scratch
Python engine (seed for the standalone app) as a "Neural Engine" mode.

---

## PHASE 1 — Real interactive browser (replaces screenshot UI)

### 1.1 Backend: proxy renderer
- Add `GET /api/browser/proxy` in `services/webui_service.py`:
  - Query `?url=`, `?tab=<id>`, `?mode=document|screenshot`.
  - For `mode=document`: use the per-user Playwright page to `goto(url)`, then return the
    page's live HTML wrapped so assets load through the same proxy (`/api/browser/asset?url=`).
    Set `X-Frame-Options` neutralized, `CSP` rewritten to allow the proxy origin.
  - Persist cookies per tab so logins/forms survive.
  - Block `javascript:`/`data:` targets; enforce same-origin isolation per tab session.
- Add `GET /api/browser/asset?url=` to proxy images/css/js (so the iframe renders fully).
- Keep `/api/browser/navigate`, `/action`, `/run` for the AI-Mirror mode.

### 1.2 Frontend: real browser chrome
Rewrite `browser.js` + the Browser tab markup in `index.html`:
- Replace `<img id="browserScreenshot">` with an `<iframe id="browserFrame">` pointing at
  `/api/browser/proxy?tab=<id>&url=<url>`.
- Address bar (`browserGo`): resolve URL, update iframe `src`, push history.
- Tabs: each tab = its own proxy session id; `new`/`select`/`close-tab` map to backend.
- Back/Forward/Reload/Home: call backend history per tab, then reload iframe.
- Zoom: `iframe.style.zoom` or `transform: scale()` + scroll adjust.
- Bookmarks + History: keep `localStorage` keys, now also reflect real navigations.
- Link overlay rects are REMOVED — the iframe is natively clickable/scrollable.
- Status: lock icon from `https`/`proxied` state, current URL/title from `postMessage` the
  backend injects into the framed page.

### 1.3 Verification
- `curl` the proxy for a static site, confirm HTML returns and assets proxy.
- Live: `https://neuralai-web-ui-deandrewharris.zocomputer.io` → Browser tab → navigate to
  a JS-heavy site (e.g. a SPA), confirm it renders and is clickable (not a PNG).
- Confirm AI-Mirror (`/api/browser/run` SSE) still works alongside the iframe.

Static edits (html/css/js) go live WITHOUT restart. Only restart `neuralai-web-ui` if
`webui_service.py` changes.

---

## PHASE 2 — From-scratch "Neural Engine" (standalone-app seed)

Build a Python browser engine from the extracted chapters (`EXTRACTED_LEARNINGS.md` §3):
1. `engine/http.py` — URL parse, socket GET, redirects, chunked, cookies.
2. `engine/html.py` — tokenizer + tree builder (void els, implicit tags).
3. `engine/css.py` — tokenizer, selector match, cascade/specificity, inheritance.
4. `engine/layout.py` — block + inline layout, boxes, x/y/w/h.
5. `engine/paint.py` — paint commands → PNG (or canvas) for simple pages.
6. Wire as `mode=neural` in `/api/browser/proxy`: backend renders the page server-side with
   the from-scratch engine and returns the painted image + a link map (or an HTML/SVG
   representation). This proves the dependency-free engine and seeds the standalone Neural
   Browser.

Caveat (honest): Phase 2 engine will render **static** pages only until a JS runtime
(QuickJS/Duktape) is integrated. That's a Phase 3 item, not Phase 1/2.

---

## PHASE 3 (future) — Standalone Neural Browser app
- Extract `engine/` + `browser.js` chrome into a standalone Electron/Tauri or Zo Site app.
- Add JS runtime, forms, history bfcache, embeds (iframes/video) per the chapter map.
- Reuse the same `web_browser.py` backend interface so it's one codebase, two deployments.

---

## Files touched
- `services/webui_service.py` — add `/api/browser/proxy`, `/api/browser/asset`.
- `tools/web_browser.py` — add per-tab cookie/doc proxy helpers (extend `BrowserSession`).
- `from-scratch/web_ui/templates/index.html` — replace screenshot markup with iframe + chrome.
- `from-scratch/web_ui/static/js/browser.js` — rewrite as real browser controller.
- `from-scratch/web_ui/static/css/main.css` — add `.browser-frame`, proxy styles.
- `engine/*.py` (Phase 2) — new from-scratch engine modules.
- `docs/neural-browser/EXTRACTED_LEARNINGS.md` — this plan's source notes.

## Open decision for DeAndrew
Confirm **Hybrid** (Phase 1 Chromium-backed live browser + Phase 2 engine seed) vs **Pure
from-scratch engine only**. Hybrid is recommended because it delivers a "100% real working"
browser today while still building the standalone-app engine you want.
