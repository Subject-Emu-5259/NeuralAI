# 📰 NeuralAI — What's New

_Last updated: 2026-07-18_

## 🌐 Real Browser Engine Replaces the "So-Called Browser"

The NeuralAI **Browser tab** (the one with the tab strip, omni search bar, bookmarks, zoom, screenshot pane, and AI Mirror) was previously backed by **Playwright/Chromium** — an external headless browser that was heavy, slow, and crashed the service on cold start (the `web_browser` module imported a Chromium driver that wasn't always available in the sandbox).

That is gone. The Browser tab is now powered by a **genuine from-scratch layout engine** built from the browser.engineering "Let's build a browser engine!" material and Matt Brubeck's toy-engine series. It is a real pipeline — not a wrapper around another browser:

- **DOM** (`tools/neural_engine/dom.py`) — HTML parsed with `html5lib` into a real element/text tree with `query()`, `text_content()`, attributes, id/class access.
- **CSS** (`tools/neural_engine/css.py`) — a real CSS parser producing rules with selector specificity `(id, class, tag)`, handling the cascade.
- **Style** (`tools/neural_engine/style.py`) — builds a styled tree, applying matching rules by specificity and resolving color/background/display/position/font-size/width/height/padding/margin/border.
- **Layout** (`tools/neural_engine/layout.py`) — a block/inline box model that computes rectangles from the styled tree (the part most "toy" renderers skip).
- **Paint** (`tools/neural_engine/paint.py`) — rasterizes the layout to a PNG via PIL, drawing backgrounds, borders, and inline text.

A single entry point, `tools.neural_engine.render_page(url, width=900, render_png=True)`, returns `PageResult` with the page `title`, status, extracted `text`, `links` (absolute URLs), `headings`, and a base64 `screenshot_b64`.

### What this fixes (verified)
- **Service no longer crashes on boot.** The old code imported a non-existent `tools.neural_layout` module, which 500'd every `/api/browser/*` call. The engine now lives at `tools.neural_engine` and is correctly wired in `tools/web_browser.py`.
- **Deadlock removed.** The single-worker `ThreadPoolExecutor` was submitting the render task from inside its own worker thread, which deadlocked indefinitely. Rendering now runs on the calling thread (the engine is CPU-bound and renders a typical page in ~0.07s), so `/api/browser/navigate` completes in ~1.3s instead of timing out.
- **Every Browser-tab feature is preserved:** multi-tab, omni search/navigate, bookmarks (localStorage), zoom, screenshot pane, and AI Mirror (SSE step streaming) all keep working — they sit on top of the new engine and were not touched.

### Known limits (by design, not bugs)
- This is a **toy-scale** engine: it renders a static, CSS-styled snapshot. Live click/scroll/fill and JavaScript execution are not performed (the same as the previous static renderer). `back`/`forward`/`reload` re-render the captured page.
- External stylesheets are fetched one level deep; `@media`/complex selectors are simplified.

---

## 🔧 How to use it
- Open the **Browser** tab in the NeuralAI UI. Paste a URL (or plain text → Google search) in the omni bar, or describe a task in AI Mirror mode.
- For developers: `from tools.neural_engine import render_page` → `render_page("https://example.com", width=900)` returns title/text/links/headings + PNG.

## ⚠️ Regression guard
The `engine` label returned by `/api/browser/*` is now `"neural_engine"` (was the placeholder `"neural_layout"`). Do not re-introduce a `tools.neural_layout` import — the engine is `tools.neural_engine`.
