# Neural Browser — Source Synthesis
*Extracted from https://browser.engineering and https://limpet.net/mbrubeck/2014/08/08/toy-layout-engine-1.html (+ 6 follow-up parts). Prepared 2026-07-19 to plan replacing the NeuralAI Browser tab with a real working browser engine.*

## 1. What the two sources actually teach

### browser.engineering (P浏览器的完整构建 — "build a browser from scratch in Python")
A full, production-shaped browser engine built in Python (~3k LOC) across these chapters:
- **http** — URL parsing, socket connection, HTTP request/response, redirects, `view-source`.
- **html** — A real **HTML parser** (state machine + tree builder) implementing the HTML5 spec's error recovery, not regex.
- **layout** — Block/inline layout: the box tree, width/height resolution, line breaking, inline wrapping.
- **styles** — CSS parsing, selector matching (specificity), the cascade, inheritance.
- **chrome** — Browser chrome: address bar, back/forward, bookmarks, tab switching, **clicking links navigates**.
- **scripts** — **JavaScript execution** via a JS engine (the book uses a JS interpreter), DOM event dispatch, so pages are interactive.
- **forms** — Inputs, submission, `GET`/`POST`.
- **text** — Fonts, glyph measurement, text shaping (real text rendering).
- **visual-effects** — Colors, opacity, compositing.
- **history** — Session history (back/forward stack).
- **embeds** — Images, iframes.
- **security** — Origin policy, sandboxing, preventing cross-site data leaks.
- **accessibility** — AOM / screen-reader tree.

**Key takeaway:** A "real working browser" requires at minimum: HTTP fetch, HTML parse → DOM, CSS parse + cascade, layout, paint, **script execution**, **form submission**, and **navigation/history**. Missing any of these and it is a renderer, not a browser.

### Brubeck "Toy Layout Engine" (7 parts, Rust)
A minimal teaching engine that proves the core pipeline:
1. **DOM** — `http_get` → string → `parse`, build a `dom::Node` tree (Element/Text), with HTML entities.
2. **CSS** — `stylesheet::parse`, `SimpleSelector`, specificity (`(id, class, tag)`), rule matching per element.
3. **Style** — Compute `StyledNode` with specified/inherited values, default values.
4. **Boxes** — Build a box tree from the styled DOM, `display` property (block/inline/none).
5. **Block layout** — Width resolution, inline children → lines, block children stacked, height computed bottom-up.
6. **Paint** — Traverse box tree → draw rectangles to a canvas (`draw::to_ppm`), no text shaping (boxes only).

**Key takeaway:** The toy engine shows the *minimum* pipeline (Fetch → Parse → Style → Layout → Paint) but explicitly has **no JavaScript, no forms, no real text, no navigation, no security**. It is a layout demonstrator.

## 2. Gap analysis vs. "100% real working web browser"

| Capability | browser.engineering | Brubeck toy | NeuralAI need |
|---|---|---|---|
| HTTP fetch + redirects | ✅ | ✅ (basic) | ✅ required |
| HTML5 parsing | ✅ | ⚠️ minimal | ✅ required |
| CSS cascade/specificity | ✅ | ⚠️ simple | ✅ required |
| Layout (block/inline) | ✅ | ✅ | ✅ required |
| Text shaping/fonts | ✅ | ❌ boxes only | ✅ required |
| Paint to canvas | ✅ | ✅ ppm | ✅ required |
| **JavaScript execution** | ✅ | ❌ | ✅ required for "real" |
| **Forms** | ✅ | ❌ | ✅ required |
| **Navigation / links / history** | ✅ | ❌ | ✅ required |
| **Security / sandbox** | ✅ | ❌ | ✅ required |
| Multi-tab | ⚠️ chrome chapter | ❌ | ✅ already in UI |

## 3. Strategic decision (recommended)
Building a from-scratch JS-executing engine in-browser is infeasible at production quality (the book itself shells out to a JS interpreter; real browsers are millions of LOC). The pragmatic path to a *genuinely working* Neural Browser now:

**Hybrid model** — keep the existing real **Chromium backend** (`tools/web_browser.py`, Playwright, gVisor-hardened) as the rendering/Javascript/forms engine, but replace the *screenshot + clickable-rects* UI with a **live sandboxed rendering surface**:
- **User Browser mode:** serve the live page through a same-origin proxy `<iframe>` (or a streamed canvas with real input forwarding) so pages are actually clickable, scrollable, and typeable inside the NeuralAI tab — not a picture of one.
- **AI Mirror mode:** keep `/api/browser/run` SSE so the user can watch the AI drive Chromium and Take Over.
- **Future standalone (Neural Browser):** extract `browser.js` + the proxy/backend into a separate app (`Projects/NeuralBrowser`), reusing `tools/web_browser.py` as the engine core.

This satisfies "100% real working web browser in NeuralAI" immediately and is the documented seed for the standalone product. A pure from-scratch engine (Brubeck-style) is kept as a *learning/diagnostic mode* (DOM/CSS inspector) but not the primary surface.

## 4. Source files saved (verbatim)
All 17 pages saved under `docs/neural-browser/`:
- `browser.engineering.md` (intro/index)
- `browser.engineering~~2f{http,html,layout,styles,chrome,scripts,forms,text,visual-effects,history,embeds,security}.html.md`
- `limpet.net~~2fmbrubeck...toy-layout-engine-{1..7}.html.md`
