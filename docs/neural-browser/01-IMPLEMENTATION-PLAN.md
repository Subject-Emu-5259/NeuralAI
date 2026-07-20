# Neural Browser — Implementation Plan
*Replaces the NeuralAI Browser tab (currently a remote screenshot view) with a real working in-app browser. Approved 2026-07-19.*

## Goal
A browser tab in NeuralAI where web pages are **actually rendered, clickable, scrollable, and scriptable** — not a screenshot. Designed so it can later become a standalone **Neural Browser** app.

## Architecture decision: Hybrid (real engine + live surface)
- **Engine (keep):** `tools/web_browser.py` (Playwright + Chromium, gVisor-hardened: `--no-sandbox --disable-dev-shm-usage`). Already does navigate / click / scroll / fill / JS / forms / multi-tab.
- **UI (replace):** swap the `#browserScreenshot` JPEG + link-rect overlay for a **live rendering surface**:
  - **Primary: same-origin proxy `<iframe>`** — new backend route fetches the target page, rewrites absolute URLs to the proxy, and serves it inside a sandboxed `<iframe>` in the NeuralAI tab. Pages become natively interactive (real clicks, typing, scrolling). This is the "100% real browser" experience.
  - **Fallback: streamed canvas + input forwarding** — for sites that break under proxying (CSP, X-Frame-Options), fall back to the existing Playwright screenshot stream with coordinated mouse/keyboard forwarding (already partially built).
  - **AI Mirror:** keep `/api/browser/run` SSE; user watches AI steps, can Take Over.

## Why not a from-scratch JS engine (Brubeck-style)
A pure in-browser engine lacks JavaScript, forms, real text, navigation, and security without years of work. The sources confirm even the reference book delegates JS to an interpreter. Hybrid gives a *real* browser now; the Brubeck pipeline is retained as a **DOM/CSS inspector learning mode**, not the primary surface.

## Build phases

### Phase 0 — Proxy backend (the core)
- New `services/webui_service.py` routes (token_required, same auth as existing `/api/browser/*`):
  - `GET /api/browser/proxy?url=<target>` → fetch target via Chromium or `requests`, inline/rewrite asset URLs to `/api/browser/proxy?url=<abs>`, strip frame-busting, set `X-Frame-Options: ALLOWALL` on our response, `Content-Security-Policy` relaxed for the frame.
  - `POST /api/browser/proxy/action` → forward click/scroll/type to the live Chromium tab (reuse `BrowserManager.action`).
- Reuse `BrowserManager` tabs so the proxy iframe and AI Mirror share session state.

### Phase 1 — Live UI surface
- In `templates/index.html` Browser tab: replace `#browserScreenshot` block with `<iframe id="browserFrame" sandbox="allow-scripts allow-same-origin allow-forms allow-popups">` pointing at `/api/browser/proxy?url=...`.
- `static/js/browser.js`: `browserGo(url)` sets `iframe.src = /api/browser/proxy?url=` + encoded; keep tab strip, bookmarks (`localStorage` key `neuralBrowserBookmarks`), zoom, back/forward (drive Chromium + `iframe` reload).
- Keep multi-tab: each tab = one Chromium page + one proxy iframe.

### Phase 2 — Interaction parity
- Click/type inside iframe works natively (sandbox allows it). For proxy-breaking sites, switch that tab to canvas-fallback automatically.
- Back/forward/reload wired to `BrowserManager` + iframe sync.
- Bookmarks, history (per tab), zoom already in `browser.js` — repoint to live surface.

### Phase 3 — AI Mirror + Take Over
- `/api/browser/run` SSE stays. When user clicks Take Over on a Mirror session, switch that tab to the live proxy iframe (session already in Chromium).

### Phase 4 — Extract standalone Neural Browser (future)
- Copy `browser.js` + proxy routes + `tools/web_browser.py` into `Projects/NeuralBrowser` as a standalone Zo Site/Service. Same engine, independent UI.

## Files touched
- `services/webui_service.py` — add `/api/browser/proxy` (+ action). Restart `neuralai-web-ui` (`svc_1cHl6qlp4_g`) after.
- `templates/index.html` — Browser tab markup: swap screenshot img for proxy iframe.
- `static/js/browser.js` — `browserGo`/action wired to proxy; preserve all `/api/browser/*` calls.
- `static/css/main.css` — iframe container styles (light/dark safe; do NOT reintroduce the black banner bug).
- New: `docs/neural-browser/00-SOURCES-SYNTHESIS.md` (done), this plan.

## Verification (per USER rule: confirm before claiming fixed)
- `curl` the proxy route for 3 sites (a static page, a JS-heavy page, a form page) → expect 200 + HTML containing rewritten URLs.
- Live UI: navigate to a URL in the Browser tab → page is interactive (can click a link, type in a box, scroll). Screenshot is NOT the only view.
- Regression guard: all existing `/api/browser/*` endpoints still respond; AI Mirror SSE still streams.
- Confirm no black surfaces in light mode (the prior bug must not return).

## Risks / notes
- Some sites (banking, Google login) block framing / detect automation — canvas-fallback + clear "site unsupported in live mode" notice handles these.
- Proxying arbitrary web through our origin has security implications; scope CSP, strip secrets, and keep `token_required` on proxy so only the owner can use it.
