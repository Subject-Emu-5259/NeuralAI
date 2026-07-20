# Neural Browser Engine — Build Notes (extracted from browser.engineering + Brubeck)

## Sources
- browser.engineering (full book: http, html, layout, styles, chrome, scripts, forms, security, text, history, embeds, visual-effects)
- https://limpet.net/mbrubeck/2014/08/08/toy-layout-engine-1.html (toy layout engine series, parts 1–8)

## Core pipeline (the 5 stages we implemented)
1. **HTTP / fetch** (BE ch.1–2): a browser is a client that sends requests and parses responses.
   Phase 1 will add a real fetch + response/status-line/header parse.
2. **HTML parsing** (BE ch.2–3, Brubeck ch.1–2): tokenizer splits text into open/close/text/comment
   tokens; tree builder handles implicit closes, void elements, raw text (script/style). We built
   `html.js` with a correct small parser + a UA stylesheet.
3. **CSS / styles** (BE ch.5, Brubeck ch.3–4): tokenizer → rules; selector matching for
   tag / .class / #id / descendant; cascade with author > UA origin and `!important`;
   computed-style inheritance. Built `css.js`.
4. **Layout** (BE ch.6, Brubeck ch.5–6): block and inline formatting; line breaking with real
   text measurement; margins/padding/borders; inline links tracked for hit-testing. Built `layout.js`.
5. **Paint** (BE ch.7, Brubeck ch.7): box tree → canvas draw commands (fill, text, borders,
   underlines). Built `paint.js`.

## What the references add for later phases
- **Scripts** (BE ch.6 / Brubeck ch.8): run JS against the DOM; event loop; our engine will need a
  sandboxed JS runtime (e.g. a restricted `eval`/QuickJS) + a DOM API surface.
- **Forms** (BE ch.12): input/submit; wire to navigation.
- **History** (BE ch.10): session history, back/forward — maps to multi-tab state.
- **Security** (BE ch.11): origin, sandbox, CSP — critical before any real navigation.
- **Visual effects** (BE ch.13): opacity, transforms, compositing.

## Key design decisions for Neural Browser
- Keep the engine pure (no DOM side effects) so it runs in a Web Worker later.
- UA stylesheet stays minimal and correct (block/inline defaults, headings, anchors, lists).
- Remote Mode (Chromium via `/api/browser/*`) remains the safe default for real sites until
  Phase 2+ security lands; Engine Mode is the from-scratch replacement path.
