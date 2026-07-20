# Neural Browser — Source Synthesis (browser.engineering + Brubeck toy engine)

Extracted from two primary sources the user asked us to study:
- **browser.engineering** — Matt Brubeck's "Building a Browser Engine from Scratch" book (live HTML chapters).
- **limpet.net/mbrubeck/2014/08/08/toy-layout-engine-1.html** — the 7-part "Let's build a browser engine!" series.

Goal: replace the NeuralAI Browser tab (currently a Playwright screenshot viewer) with a **real, in-page web browser engine** that can one day become a standalone *Neural Browser* app.

---

## 1. What "a real browser" actually requires (from browser.engineering)

A browser is a pipeline of stages. Each maps to a chapter:

| Stage | Chapter | What it does | Neural Browser relevance |
|---|---|---|---|
| Networking | `http.html` | Socket → HTTP request → response, chunked encoding, redirects | Phase 2: fetch real pages from NeuralAI backend proxy |
| Parsing | `html.html` | HTML5 tokenizer + tree (handles broken markup) | Phase 1: our engine needs an HTML parser |
| Styling | `styles.html` | CSS cascade, specificity, inheritance | Phase 4 |
| Layout | `layout.html` | Block/inline/inline-block boxes, document flow | Phase 4 (core of Brubeck series) |
| Painting | `paint`/visual-effects | Draw to canvas/surface | Phase 5: render to `<canvas>` |
| Chrome | `chrome.html` | Address bar, tabs, back/forward, bookmarks | Already partly in our tab; reuse |
| Scripts | `scripts.html` | JS execution, DOM APIs | Phase 7 (defer — huge scope) |
| Forms | `forms.html` | Inputs, submission | Phase 6 |
| History | `history.html` | Session history, back/forward | Reuse our tab history |
| Text | `text.html` | Fonts, line breaking, glyphs | Phase 5 |
| Embeds | `embeds.html` | Images, iframes | Phase 6 |
| Security | `security.html` | Origin, sandboxing, escaping | Phase 8 — critical before standalone app |
| Accessibility | `accessibility.html` | AOM, roles | Later |

### Key architectural facts
- The engine is **data-driven**: parse → tree, style → styled tree, layout → display list, paint → pixels.
- The browser.engineering code is **Python** (`tkinter` display, `socket` networking, `html.parser`-style custom tokenizer). Brubeck's toy engine is also **Python**.
- For an *in-page* NeuralAI browser we implement the same pipeline in **JavaScript/TypeScript** so it runs in the browser tab without a server round-trip per paint.
- The two sources agree on the **box model**: every node becomes a `LayoutBox` with `dimensions` (x, y, w, h) and `children`.

---

## 2. Brubeck toy engine — the concrete blueprint (7 parts)

1. **Part 1 — DOM** (`toy-layout-engine-1`): Build a DOM tree from HTML. Nodes have `children`, `style` dict. He hand-rolls a tiny HTML parser.
2. **Part 2 — Styles** (`toy-layout-engine-2`): Parse CSS, apply to matching DOM nodes, compute `display`/`color`/etc.
3. **Part 3 — CSS matching** (`toy-layout-engine-3-css`): Selector matching (tag, class, id), specificity.
4. **Part 4 — Style tree** (`toy-layout-engine-4-style`): Cascade + inheritance → `StyledNode` tree.
5. **Part 5 — Boxes** (`toy-layout-engine-5-boxes`): `LayoutBox`, block/inline, `Dimensions`.
6. **Part 6 — Block layout** (`toy-layout-engine-6-block`): Block formatting context, vertical stacking, margins, width from container.
7. **Part 7 — Painting** (`toy-layout-engine-7-painting`): `DisplayCommand` list → draw rectangles, text, borders to a surface.

**The toy engine's value to us:** it's the minimal correct core. We port Parts 1–7 to JS as `engine/`. It renders real HTML+CSS to a canvas. That is a genuine browser engine (not a screenshot), which is exactly what the user wants.

---

## 3. Current NeuralAI Browser (what we're replacing)

From `from-scratch/web_ui/templates/index.html` + `static/js/browser.js` + `tools/web_browser.py`:
- **Mode**: remote-control. `browser.js` calls `/api/browser/navigate` → backend runs Playwright/Chromium (gVisor-hardened), returns a **PNG screenshot** + `link_rects`.
- The UI shows `<img id="browserScreenshot">` and makes link rects clickable by overlay.
- Has: tab strip, omni bar, bookmarks (`localStorage`), zoom, AI-Mirror (SSE step streaming), user-browser mode.
- **Why it's not "100% real"**: it's a picture of a page, not a live DOM. No real text selection, no in-page JS, no true layout. The user wants the engine *in the tab*.

---

## 4. Strategy decision

- **Keep** the existing Playwright tab as "Remote Mode" (still useful for JS-heavy/modern sites we can't yet render).
- **Add** a "Engine Mode" that loads `engine/` (ported Brubeck pipeline + browser.engineering stages) and renders fetched page source directly in the tab to a `<canvas>`/`DOM`.
- **Grow** Engine Mode through the phases below until it can stand alone as Neural Browser.

This satisfies "100% real working web browser in NeuralAI" without a hard cutover that risks regressions (per AGENTS.md regression guard).
