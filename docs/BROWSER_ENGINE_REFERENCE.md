# Browser Engineering Reference — extracted from browser.engineering + Brubeck toy engine

> Purpose: single source of truth for building the real **Neural Browser** that will
> replace the current screenshot-based Browser tab in NeuralAI, and eventually become
> its own standalone application.
>
> Sources:
> - https://browser.engineering — *"Web Browser Engineering"* by Pavel Panchekha & Chris Harrelson (16 chapters, full Python browser "Robinson").
> - https://limpet.net/mbrubeck/2014/08/08/toy-layout-engine-1.html — Matt Brubeck's 7-part *"Let's build a browser engine"* (Rust toy engine).

---

## 0. TL;DR architecture of a real browser

A browser is a pipeline. Each stage consumes the previous stage's output:

```
URL ──► HTTP fetch ──► HTML bytes
   ──► Unicode decode + <meta charset>
   ──► HTML parser ──► DOM tree
   ──► CSS parser ──► stylesheet (cascade)
   ──► Style resolution ──► styled DOM (each node gets computed styles)
   ──► Layout ──► boxes (position + size)
   ──► Paint ──► display list / pixels
   ──► (Scripts run during parse via the event loop; can mutate DOM/CSS)
   ──► User input (clicks, scroll) ──► hit testing on boxes ──► navigation / JS
```

The current NeuralAI Browser tab is **NOT** this pipeline. It is a *remote control*:
Playwright drives Chromium **server-side**, returns a **screenshot PNG + clickable link
rectangles**. That is a remote-desktop view, not an in-app browser. The plan below
replaces it with a genuine browser that renders real DOM/CSS/JS.

---

## 1. browser.engineering — chapter-by-chapter (the production-grade path)

| Ch | Topic | What it teaches | Neural Browser relevance |
|----|-------|-----------------|--------------------------|
| intro | Why build a browser | Mental model: browser = pipeline of stages | Defines our module boundaries |
| http | Networking | `socket` → TLS → HTTP GET; `url` parsing; `get_requests` generator; throttling; caching; `conditional_get` (ETag/Last-Modified) | **Network layer** of Neural Browser |
| html | HTML parsing | Tokenizer (CTokenizer), tree builder, insertion modes, error recovery, `<!doctype>` quirks, `EntityParser`, `view_source` | **DOM parser** |
| layout | Layout engine | `DocumentLayout`, `BlockLayout`, `InlineLayout`, `LineLayout`; `get_ascent`/`get_descent`; width/height/overflow; word wrap; `input`/button layout | **Core renderer** |
| styles | CSS | `CSSParser` (tokens → rules), `compile_selector`; cascade (specificity, `!important`, inline, UA defaults); `style`/`style_paragraph`/`style_node`; inherited vs non-inherited | **Style system** |
| chrome | Browser chrome | Tabs, back/forward **history** (`History` class: `go_back`/`go_forward`/`add`/`current`), navigation, `load` coroutine, bookmarks, URL bar, zoom, cookies bar | **Tab + history UI** (what we already have in UI) |
| scripts | JavaScript | DLite/WASM JS runtime (`JSContext`); `run` coroutine; `document`/`window` global stubs; `document.getElementById`, `document.createElement`, event listeners, `setTimeout`; same-origin checks | **JS execution** |
| forms | Forms & controls | `INPUT`, `TEXTAREA`, `SELECT`; focus/active; key events; form submission (GET/POST); `value`/`checked`; autofill | **Interactive controls** |
| text | Text & fonts | `Font` abstraction (HarfBuzz/FreeType into Skia), `get_font`; `word`/`flush`; `create_word`; `get_font`/font fallback; `draw` glyphs; bidi | **Typography** |
| graphics | Graphics / canvas | `DrawCommand` list; `save`/`restore`; `translate`; `clip`; rasterize (Skia) | **Paint backend** |
| visual-effects | Visual effects | opacity, filters, blend modes, `z-index`, stacking contexts, transforms | Polish |
| animations | Animation | `Animation` class, `play`, `rAF`, transitions | Polish |
| history | History & session | persists across reloads; multi-tab history | Tab state |
| accessibility | A11y | AX tree, focus order, names | Optional |
| embeds | Images & iframes | `IMG` decode (`PNG`/`GIF`); `IFRAME` → nested `Tab`; `<video>`/`<audio>` stubs | **Images + frames** |
| security | Security | **Same-origin policy**, `file:`/`javascript:` URL parsing, mixed-content block, cookie/referer leakage, `X-Frame-Options`, sandbox flags, parse `javascript:` URL safely | **Critical for safety** |

### Key implementation details pulled from the book

**HTTP (`http.py`)**
- `URL` class parses `scheme://host:port/path?query#fragment`; defaults port 80/443; `resolve` for relative links.
- `request` opens `ssl.create_default_context().wrap_socket(socket, server_hostname=host)` for HTTPS; sends `GET {path} HTTP/1.0\r\nHost: {host}\r\n\r\n`; reads until `socket.close()`.
- `condition_get` sends `If-None-Match`/`If-Modified-Since`; server 304 → reuse cache.
- Throttle: cap active connections per host; `MAX_POOL`.

**HTML (`html.py`)**
- `HTMLParser`: `parse()`: `lex()` → tokens → `parse_doctype`/`parse_tag`/`parse_text`; `in_body` insertion mode; `implicit_tags` closes `<p>`/`<li>`; `parse_error` recovers.
- Builds `Element` tree with `children`, `parent`, `tag`, `attributes`, `text`.

**Layout (`layout.py`)**
- `DocumentLayout` → one `BlockLayout` child (the `<html>`/`<body>`).
- `BlockLayout.layout()`: measure `children` heights, set `self.height`; `InlineLayout` wraps words into `LineLayout`s using `word(self, word)` with width budget.
- Coordinate system: `x, y, width, height` per box; `paint()` walks tree → `DisplayList`.
- Recurses; `needs_layout` invalidation flag (see `invalidation`).

**Styles (`css.py` + `layout.py`)**
- `CSSParser`: `parse()` → list of `(selectors, declarations)`; `compile_selector` turns `a b > c:hover` into matcher.
- `style(node)`: cascade order — UA sheet → author sheet (specificity) → `!important` → inline style.
- `inherited` properties (color, font) copy from parent when absent.

**Chrome (`browser.py`, `history.py`)**
- `Tab` wraps `{url, history, document, browser}`. `load(url, payload=None)` fetches, decodes, parses, styles, lays out, paints.
- `Browser` holds `tabs: List[Tab]`, `active_tab`. `History` is a per-tab stack with `go_back`/`go_forward` and a `current` index.
- `click` hit-test: `tree_to_list(layout)` flattens boxes; find box containing `(x,y)`; if it's an `<a>`, navigate to `href`.

**Scripts (`scripts.py`)**
- `JSContext` runs WASM-compiled JS. `runtime` exposes `window`, `document`.
- `document.getElementById(id)` scans DOM; `document.createElement` makes `Element`; event listeners stored per node.
- `run()` coroutine steps the JS event loop interleaved with layout/paint.

**Security (`security.py`)**
- `same_origin(a, b)` compares scheme+host+port.
- Block `javascript:`/`data:` navigation unless explicitly safe; block mixed content (https page loading http subresource); `X-Frame-Options: DENY/SAMEORIGIN` disables iframe embedding.

---

## 2. Brubeck toy engine (Rust) — the minimal real engine

Matt Brubeck's 7-part series builds a **from-scratch** engine in Rust. It proves you can
render real web pages with a tiny, understandable codebase. Stages:

1. **Part 1 — Setup**: `rustc main.rs`; `get_default_stylesheet()`; `Element`/`Node` tree; `parse_html` (recursive descent, no real tokenizer yet — simplified).
2. **Part 2 — Styles**: `parse_css` → `Stylesheet{rules}`; `match_rule`/`matching_simple_selector` (tag/id/class); `specified_values` map; `cascade` picks winning declaration by **specificity** (inline > id > class > tag) then **source order**.
3. **Part 3 — CSS values**: `Value` enum (`Keyword`, `Length`, `Color`); `parse_value`; color parsing `#rrggbb`; `to_px`.
4. **Part 4 — Style tree**: `style_tree` recurses DOM → `StyledNode` with inherited properties (`color`, `font`) and UA default stylesheet.
5. **Part 5 — Boxes**: `layout_tree` → `LayoutBox` (`Block`/`Inline`/`Anonymous`); `build_box_tree`; `BlockContainer`/`BlockLevel`/`InlineLevel`; `get_style` helpers; `layout` computes `dimensions {x,y,width,height}` with `padding/border/margin` (`get_padding` etc.).
6. **Part 6 — Block layout**: `layout_block_children` (each child `x = cx`, `y = cy`, then `cy += child.height`); `calculate_block_width` (solve `width + margin = container - padding - border` with auto rules); `calculate_block_height` sums children; `layout_inline_children` → `LineBox`es with word wrapping.
7. **Part 7 — Painting**: `paint(&self, display_list)` → `background`/`outline`/`borders`/`text`; `DisplayCommand::{SolidColor, DrawText}`; `TextPain`/`FontCollection` via `fontkit`; `to_px` for font sizes; rasterize to PNG.

**Takeaway for Neural Browser:** the Brubeck series is the *teaching scaffold*. We can port
its exact pipeline (parse → style → box → block layout → paint) to JavaScript/TypeScript so
the engine runs **in the browser tab itself** (no server Chromium needed). That is the
"100% real working web browser" built from scratch.

---

## 3. What "real working browser" requires (checklist)

| Capability | Toy engine (Brubeck/port) | Production (browser.engineering/Chromium) |
|------------|---------------------------|-------------------------------------------|
| Fetch URL + TLS | ❌ (needs add) | ✅ HTTP chapter |
| Parse HTML → DOM | ✅ simplified | ✅ full tokenizer |
| CSS cascade | ✅ basic | ✅ full |
| Block + inline layout | ✅ | ✅ + flex/grid later |
| Paint to screen | ✅ PNG | ✅ Skia/canvas |
| Click links / navigate | ⚠️ manual | ✅ hit-test |
| Run JavaScript | ❌ | ✅ (scripts chapter) |
| Forms/inputs | ❌ | ✅ |
| Images/iframes | ❌ | ✅ |
| Back/forward history | ❌ | ✅ |
| Security (SOP) | ❌ | ✅ critical |
| Scroll/zoom | UI only | ✅ |

---

## 4. Decision for Neural Browser

Two buildable targets (not mutually exclusive — phase order matters):

- **Target A — In-app from-scratch engine (Brubeck port to TS):** Runs entirely client-side
  in the NeuralAI Browser tab. Renders real HTML/CSS with our own layout code. No external
  browser. This is the "100% real working web browser" you build yourself and is the seed of
  the standalone **Neural Browser** app. Limitation: JS execution & forms need added work.

- **Target B — Real Chromium, DOM-based (not screenshot):** Keep Chromium (Playwright) but
  return **serialized DOM + computed styles + text**, not a PNG. The tab renders that DOM in a
  sandboxed `<iframe>`/`shadow DOM`. Gives a *genuinely complete* browser now (JS, forms,
  images all work) and is the faster path to a usable product. Same-origin/security handled by
  Chromium. Later, extract the controller + a minimal UI shell as the standalone Neural Browser.

**Recommendation (in plan):** Ship **Target B first** (real browser, usable in days, no
regressions to current UI), then **Target A as the long-term Neural Browser engine** (from-
scratch, portable, ownable IP). The plan below sequences both.
