# Neural Browser — Extracted Learnings from browser.engineering + Brubeck Toy Engine

Source corpus (saved as `.md` in this folder):
- **browser.engineering** (Powell & Reitz, *Web Browser Engineering*) — 16 chapters: `intro, http, html, layout, styles, text, chrome, scripts, forms, history, embeds, visual-effects, animations, accessibility, scheduling, invalidation, security, graphics, glossary`.
- **limpet.net toy layout engine** (Matt Brubeck, 2014) — 7 parts: tokenize CSS, parse CSS, style tree, layout boxes, block layout, paint.

These two sources were extracted and read in full on 2026-07-19 to plan replacing the
NeuralAI **Browser** tab with a real, working browser that can grow into a standalone
**Neural Browser** application.

---

## 1. What a "real browser" is made of (browser.engineering chapter map)

A production browser has these subsystems. The toy engine covers a subset.

| Subsystem | browser.engineering | Brubeck toy | Notes for Neural Browser |
|---|---|---|---|
| **Networking / HTTP** | `http.html` | — | URL parse, socket, request/response, redirects, caching headers |
| **HTML parsing** | `html.html` | — | Tokenizer → tree (handles broken HTML, implicit tags) |
| **Layout / reflow** | `layout.html` | parts 5–6 | Box model, inline vs block, recursion |
| **Style / CSS** | `styles.html` | parts 2–4 | Selector matching, cascade, specificity, inheritance |
| **Text shaping** | `text.html` | — | Fonts, line breaking, glyph runs |
| **Chrome / UI** | `chrome.html` | — | Tabs, address bar, back/forward, bookmarks |
| **Scripting** | `scripts.html` | — | JS execution, DOM APIs, event loop |
| **Forms** | `forms.html` | — | Inputs, submission, state |
| **History** | `history.html` | — | Session history, bfcache |
| **Embeds** | `embeds.html` | — | iframes, images, video |
| **Paint / raster** | `graphics.html`, `visual-effects.html` | part 7 | Actual pixels, transforms, opacity |
| **Security** | `security.html` | — | Same-origin, mixed content, sandboxes |

---

## 2. The two viable engine strategies

### Strategy A — From-scratch browser engine (Brubeck path)
A Python/JS engine that: parses HTML → builds DOM, parses CSS → styles, computes layout
boxes, paints to canvas. The toy engine is ~500 lines of Python and renders real pages
(basic CSS, block/inline layout, painting). **Pros:** full control, educational, no deps,
portable to a standalone app. **Cons:** cannot run JavaScript, cannot do forms/auth, no
video/iframes, CSS coverage is small, slow on big pages. This is what the user literally
described ("build your own browser engine"), and it's the seed for a standalone Neural
Browser — but on its own it is NOT "100% real working" for modern web.

### Strategy B — Real Chromium backend + true in-page render surface (recommended bridge)
Keep the existing Playwright/Chromium backend (it already does HTTP, JS, CSS, forms,
video — everything a real browser needs) but replace the **screenshot + link-rect** UI
with a real rendering surface inside NeuralAI:
- Server-side: route page HTML/CSS/JS through the Chromium backend, expose the live page
  via a **proxy iframe** (`/api/browser/proxy?url=...`) so the in-page `<iframe>` shows the
  *actual* rendered, interactive page (clickable, scrollable, scriptable), not a PNG.
- Client-side: `browser.js` becomes a real browser controller — address bar, tabs,
  back/forward, reload, zoom, bookmarks, history, all driving the proxy.
- Security: enforce same-origin proxying, sanitize `X-Frame-Options`/`CSP`, block
  `javascript:` and `data:` navigations, isolate per-tab sessions.
**Pros:** genuinely "100% real working" today, supports JS/forms/auth, reuses the engine
the user already runs. **Cons:** depends on Chromium process; not a from-scratch engine.

### Recommended path (hybrid)
Ship **Strategy B first** (real working browser in NeuralAI, replacing the screenshot UI),
then **layer Strategy A as "Neural Engine mode"** — a from-scratch Python engine (from the
Brubeck/browser.engineering chapters) that can render simple pages *without* Chromium,
proving the standalone Neural Browser vision and giving a dependency-free fallback. Both
share the same `browser.js` chrome (tabs/address bar) and the same `web_browser.py` backend
interface, so they're one product, two engines.

---

## 3. Key implementation details pulled from the sources

### HTTP (`http.html`)
- Parse URL into scheme/host/port/path/query.
- Use `socket` + `send` for the request; read response headers until `\r\n\r\n`.
- Handle `301/302` redirects (follow `Location`, cap recursion).
- Parse `Content-Type`, `Content-Length`, chunked transfer (`Transfer-Encoding: chunked`).
- Store cookies for session continuity (forms/auth need this).

### HTML (`html.html`)
- Tokenizer: emit `OpenTag`/`CloseTag`/`Char`/`EOF`.
- Tree builder: implicit `<html>/<body>`, close p on block, reconstruct active formatting
  elements (the "adoption agency" algorithm is optional for a toy).
- Handle void elements (`<img>, <br>, <input>`), self-closing.

### CSS + Style (`styles.html`, Brubeck 2–4)
- Tokenize CSS into rules + declarations.
- Selector matching: tag, `#id`, `.class`, descendant/child combinators.
- Cascade: later rule wins; specificity `(id, class, tag)`; `!important` overrides.
- Inheritance: unset properties inherit from parent (e.g. `color`, `font`).
- Compute *used* values: keywords → pixels, percentages → parent-relative.

### Layout (`layout.html`, Brubeck 5–6)
- `BlockLayout` recurses over children, stacks vertically.
- `InlineLayout` (Brubeck "boxes") wraps text into lines, splits on width.
- Each layout node has `x, y, w, h`; `layout()` computes from parent's content box.
- Inputs: `font-style, font-size, font-weight` affect inline/line height.

### Paint (`graphics.html`, Brubeck 7)
- `PaintCommand` list (rect, text, line).
- Draw background, borders, then text glyphs at `(x, y + baseline)`.
- A real impl uses a canvas/`tkinter`/`skia`; the toy uses `tkinter`.

### Chrome (`chrome.html`)
- Tabs = separate sessions/states; address bar resolves URL; back/forward = history stack.
- Bookmarks persisted locally.

### Scripts (`scripts.html`) — the hard part
- A from-scratch engine would need a JS runtime (Duktape/QuickJS or a WASM V8). Without it,
  only Strategy B (Chromium) gives real JS. This is the deciding factor for "100% working".

### Security (`security.html`)
- Same-origin policy: scripts from origin A can't read DOM of origin B.
- When proxying via iframe, the proxy must strip/neutralize `document.cookie` leakage and
  block nav to `javascript:`/`data:`.

---

## 4. Decision needed from DeAndrew

Two product directions, both documented above:

1. **Hybrid (recommended):** Real Chromium-backed browser in NeuralAI NOW (replaces the
   screenshot UI with a live proxy iframe), plus a from-scratch "Neural Engine" mode built
   from these chapters as a dependency-free fallback / standalone-app seed.
2. **Pure from-scratch engine (Brubeck-style):** Build the Python engine only. Honest caveat:
   it will render static pages but **cannot run JS/forms/auth** — not "100% real working"
   for modern sites until a JS runtime is added.

The implementation plan below assumes **Hybrid** and sequences the work so a real browser
ships first, then the engine seed follows.
