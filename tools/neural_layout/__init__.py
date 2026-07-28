"""NeuralLayout — a from-scratch HTML/CSS rendering engine.

Replaces the previous Playwright/Chromium "browser" backing with a real
layout engine built from first principles, while the NeuralAI Browser UI
(tab strip, omni bar, bookmarks, screenshot pane, AI Mirror) stays intact.

Pipeline (mbrubeck toy-engine series + browser.engineering):
  HTML string -> DOM (dom.py)
             -> CSS parse + cascade (css.py, default styles)
             -> layout tree: boxes, block + inline/text flow (layout.py)
             -> display list -> raster (paint.py: PNG via PIL, SVG fallback)

Public API:
  render_html(html: str, width: int = 760) -> RenderResult
  RenderResult.text        full extracted text
  RenderResult.markdown    simplified markdown
  RenderResult.png_path    path to raster (if PIL available)
  RenderResult.svg         debug SVG string
  RenderResult.title       <title> if present
"""
from __future__ import annotations
import os
import uuid
from dataclasses import dataclass, field
from typing import Optional

from .dom import parse_html, Element, TextNode
from .css import CSSStyleSheet, parse_stylesheet, parse_inline_style, DEFAULT_STYLES  # noqa
from .layout import layout, Box  # noqa
from . import paint as _paint

AUDIO_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "from-scratch", "web_ui", "static", "render")
try:
    os.makedirs(AUDIO_DIR, exist_ok=True)
except Exception:
    pass


@dataclass
class RenderResult:
    text: str = ""
    markdown: str = ""
    png_path: Optional[str] = None
    svg: str = ""
    title: str = ""
    width: int = 0
    height: int = 0


def _extract_text(el) -> str:
    if isinstance(el, TextNode):
        return el.text
    out = []
    if isinstance(el, Element):
        if el.tag in ("script", "style"):
            return ""
        for c in el.children:
            out.append(_extract_text(c))
    return "".join(out)


def _to_markdown(el, depth: int = 0) -> str:
    if isinstance(el, TextNode):
        return el.text
    if not isinstance(el, Element):
        return ""
    if el.tag in ("script", "style"):
        return ""
    lines = []
    pad = "  " * depth
    if el.tag in ("h1", "h2", "h3", "h4"):
        level = int(el.tag[1])
        lines.append("#" * level + " " + _extract_text(el).strip())
    elif el.tag == "p":
        lines.append(_extract_text(el).strip())
    elif el.tag in ("li",):
        lines.append(pad + "- " + _extract_text(el).strip())
    elif el.tag in ("blockquote",):
        lines.append("> " + _extract_text(el).strip())
    elif el.tag == "a":
        lines.append(f"[{_extract_text(el).strip()}]({el.attrs.get('href','')})")
    elif el.tag in ("hr",):
        lines.append("---")
    else:
        for c in el.children:
            lines.append(_to_markdown(c, depth))
    return "\n".join(l for l in lines if l) + "\n"


def render_html(html: str, width: int = 760) -> RenderResult:
    root = parse_html(html)
    sheet = parse_stylesheet(DEFAULT_STYLES)
    # also parse any <style> blocks in the doc
    def collect_styles(n):
        if isinstance(n, Element) and n.tag == "style" and n.children:
            txt = n.children[0].text if isinstance(n.children[0], TextNode) else ""
            if txt:
                sheet.rules.extend(parse_stylesheet(txt).rules)
        for c in (n.children if isinstance(n, Element) else []):
            collect_styles(c)
    collect_styles(root)

    box = layout(root, sheet, width=width)
    height = int(box.h) + 24

    # raster
    png_path = None
    fname = f"{uuid.uuid4().hex}.png"
    full = os.path.join(AUDIO_DIR, fname)
    if _paint.render_png(box, width, height, full):
        png_path = full

    svg = _paint.render_svg(box, width, height)

    title = ""
    def find_title(n):
        nonlocal title
        if isinstance(n, Element) and n.tag == "title" and n.children:
            title = n.children[0].text if isinstance(n.children[0], TextNode) else ""
        for c in (n.children if isinstance(n, Element) else []):
            find_title(c)
    find_title(root)

    md = _to_markdown(root).strip()
    txt = _extract_text(root).strip()

    return RenderResult(text=txt, markdown=md, png_path=png_path,
                        svg=svg, title=title, width=width, height=height)


__all__ = ["render_html", "RenderResult", "parse_html"]
