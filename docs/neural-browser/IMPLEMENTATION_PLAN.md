# Neural Browser — Implementation Plan

Replaces the NeuralAI Browser tab's screenshot viewer with a **real in-page browser engine**, grown from browser.engineering + Brubeck's toy engine, and ultimately spun out as a standalone *Neural Browser* app.

**Status: APPROVED (2026-07-19).** Work begins at Phase 0.

---

## Architecture

```
NeuralAI Browser tab
├── Remote Mode  (existing) — Playwright screenshot + link rects  [/api/browser/*]
└── Engine Mode  (new)      — engine/ renders fetched HTML/CSS in-tab
      engine/
        parser.js     (HTML tokenizer + DOM tree)        [Brubeck P1]
        css.js        (CSS parse + selector match)       [Brubeck P2-P3]
        style.js      (cascade + inheritance → styled tree) [Brubeck P4]
        layout.js     (LayoutBox, block layout)          [Brubeck P5-P6]
        paint.js      (display list → canvas)            [Brubeck P7]
        browser.js    (navigate, history, bookmarks, tabs) [browser.engineering chrome/history]
        net.js        (fetch via /api/browser/fetch proxy) [browser.engineering http]
```

Engine Mode fetches page *source* through a backend proxy (`/api/browser/fetch`) that returns raw HTML (server-side fetch, CORS-safe), then parses + lays out + paints client-side. No per-paint server call.

---

## Phases (each ends verified in the live tab)

- **Phase 0 — Scaffold + toggle.** Add `Engine Mode` toggle to the Browser tab. Create `engine/` skeleton. Render a hard-coded test HTML to canvas via the pipeline. No backend change. *Verification: toggle shows a rendered box + text.*
- **Phase 1 — HTML parser.** Port Brubeck P1: tokenizer → DOM tree (handle broken markup, void elements, text nodes). *Verify: real fetched HTML builds a tree.*
- **Phase 2 — Networking.** Add `/api/browser/fetch` (server-side GET, returns raw HTML + content-type). `net.js` uses it. *Verify: navigate to example.com renders.*
- **Phase 3 — CSS parse + match.** Port Brubeck P2–P3: parse `<style>`/inline CSS, tag/class/id selectors, specificity. *Verify: colored headings.*
- **Phase 4 — Style tree + layout.** Port Brubeck P4–P6: cascade, inheritance, block/inline boxes, document flow, margins, width. *Verify: multi-paragraph article lays out correctly.*
- **Phase 5 — Painting + text.** Port Brubeck P7 + browser.engineering `text.html`: draw rects, borders, text glyphs, basic fonts/line-break. *Verify: readable page, not just blocks.*
- **Phase 6 — Forms + embeds + links.** `forms.html`/`embeds.html`: clickable links (real navigation, not rects), images, inputs. *Verify: clicking a link navigates; images show.*
- **Phase 7 — Scripts (deferred).** `scripts.html`: minimal JS/DOM. Largest scope — implement only if needed for target sites; can stay Remote Mode for JS-heavy pages.
- **Phase 8 — Security + standalone.** `security.html`: origin sandbox, HTML escaping, no `javascript:`/data exfil. Then extract `engine/` + chrome into a standalone **Neural Browser** app (own service/build).

---

## Constraints (from AGENTS.md)
- `from-scratch/web_ui` is the LIVE UI; do not redesign layout. Add Engine Mode as a toggle, preserve existing Remote Mode + all `/api/browser/*` calls.
- Static edits (html/css/js) go live WITHOUT service restart. Only restart `neuralai-web-ui` (`svc_1cHl6qlp4_g`) if `services/webui_service.py` changes.
- Keep `/api/browser/health|navigate|action|run|close` intact (Remote Mode depends on them).

## First deliverable (this session)
Phase 0: Engine Mode toggle + `engine/` skeleton that paints a test document to canvas, verifiable in the live tab.
