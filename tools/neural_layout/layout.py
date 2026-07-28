"""NeuralLayout — layout engine.

Implements the mbrubeck toy-engine Part 5 (Boxes), Part 6 (Block layout) and
browser.engineering "Laying Out Pages". Pipeline:
  styled DOM (Element + computed style map) -> layout tree of Boxes
  - block boxes stack vertically inside their containing block
  - inline text runs wrap and stack horizontally, then vertically
  - anonymous inline boxes wrap runs of text inside block contexts
The root box is the viewport; its `.h` is the document height.

Not standards-complete (no flex/grid/float); tolerant by design.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .dom import Element, Node, Text, parse_html
from .css import CSSStyleSheet, get_styles, parse_stylesheet, DEFAULT_STYLES


# --- Box model -------------------------------------------------------------

@dataclass
class Box:
    kind: str = "block"          # 'block' | 'text'
    x: float = 0.0
    y: float = 0.0
    w: float = 0.0
    h: float = 0.0
    text: str = ""               # for kind == 'text'
    color: Optional[str] = None
    font_size: float = 16.0
    bold: bool = False
    bg: Optional[str] = None
    children: List["Box"] = field(default_factory=list)

    def add(self, b: "Box") -> "Box":
        self.children.append(b)
        return b


# --- style helpers ----------------------------------------------------------

def _px(value: Any, default: float, font_size: float) -> float:
    if value is None:
        return default
    s = str(value).strip().lower()
    if s in ("auto", ""):
        return default
    if s.endswith("px"):
        try:
            return float(s[:-2])
        except ValueError:
            return default
    if s.endswith("em"):
        try:
            return float(s[:-2]) * font_size
        except ValueError:
            return default
    if s.endswith("%"):
        return default  # percentage handled by caller
    try:
        return float(s)
    except ValueError:
        return default


def _display(style: Dict[str, Any]) -> str:
    d = str(style.get("display", "")).lower()
    if d in ("none",):
        return "none"
    if d in ("inline", "inline-block"):
        return "inline"
    return "block"


# --- layout ----------------------------------------------------------------

LINE_HEIGHT = 1.35
TOP_MARGIN = 12.0


def layout(root: Element, sheet: CSSStyleSheet = None, width: int = 760) -> Box:
    if sheet is None:
        sheet = parse_stylesheet(DEFAULT_STYLES)
    viewport = Box(kind="block", x=0, y=0, w=width, h=0)
    body = _find_body(root) or root
    _layout_block(body, sheet, viewport, width, 0.0, TOP_MARGIN)
    viewport.h = _col_height(viewport.children, TOP_MARGIN)
    return viewport


def _find_body(node: Node) -> Optional[Element]:
    if isinstance(node, Element) and node.tag in ("body", "html"):
        return node
    if isinstance(node, Element):
        for c in node.children:
            r = _find_body(c)
            if r:
                return r
    return None


def _col_height(children: List[Box], top: float) -> float:
    bottom = top
    for c in children:
        bottom = max(bottom, c.y + c.h)
    return bottom


def _layout_block(el: Element, sheet: CSSStyleSheet, parent_box: Box,
                  avail_w: float, x: float, y: float) -> Box:
    style = get_styles(el, sheet)
    if _display(style) == "none":
        return Box(kind="block")  # omitted

    margin = _px(style.get("margin"), 0, 16)
    padding = _px(style.get("padding"), 0, 16)
    bg = style.get("background") or style.get("background-color")

    box = parent_box.add(Box(kind="block", x=x + margin, y=y + margin,
                             w=avail_w - 2 * margin, h=0, bg=bg))

    if el.tag in ("h1", "h2", "h3", "h4", "h5", "h6"):
        try:
            lvl = int(el.tag[1])
        except ValueError:
            lvl = 1
        font_size = 28 - (lvl - 1) * 3
        weight = "bold"
        top_gap = font_size * 0.8
        bot_gap = font_size * 0.4
    elif el.tag == "li":
        font_size = 16
        weight = "normal"
        top_gap = 2
        bot_gap = 2
    else:
        font_size = 16
        weight = "bold" if el.tag in ("b", "strong") else "normal"
        top_gap = 6
        bot_gap = 6

    cy = box.y + top_gap
    content_w = box.w - 2 * padding

    # gather styled children (skip script/style/head)
    kids = [c for c in el.children if isinstance(c, (Element, Text))]
    kids = [c for c in kids if not (isinstance(c, Element) and c.tag in ("script", "style", "head", "meta", "link"))]

    # Split into block-level vs inline runs; wrap inline runs into anon boxes.
    anon_runs: List[List[Node]] = []
    block_kids: List[Node] = []
    for c in kids:
        if isinstance(c, Text):
            if anon_runs and isinstance(anon_runs[-1], list) and anon_runs[-1] and isinstance(anon_runs[-1][-1], Text):
                anon_runs[-1].append(c)
            else:
                anon_runs.append([c])
        elif isinstance(c, Element) and _display(get_styles(c, sheet)) == "inline":
            if anon_runs and isinstance(anon_runs[-1], list):
                anon_runs[-1].append(c)
            else:
                anon_runs.append([c])
        else:
            block_kids.append(c)
            anon_runs.append(None)  # marker for "block here"

    # Render interleaved
    ai = 0
    for item in (kids if False else _interleave(anon_runs, block_kids)):
        if item is None:
            continue
        if isinstance(item, list):  # inline anon run
            line_box = _layout_inline(item, sheet, box.x + padding, cy,
                                      content_w, font_size, weight == "bold")
            cy = line_box.y + line_box.h
            box.add(line_box)
        else:  # block element
            sub = _layout_block(item, sheet, box, content_w,
                               box.x + padding, cy)
            cy = sub.y + sub.h + bot_gap
            # already added inside _layout_block via parent_box.add

    box.h = (cy - box.y) + bot_gap
    return box


def _interleave(anon_runs, block_kids):
    out = []
    bi = 0
    for run in anon_runs:
        if run is None:
            out.append(block_kids[bi] if bi < len(block_kids) else None)
            bi += 1
        else:
            out.append(run)
    while bi < len(block_kids):
        out.append(block_kids[bi])
        bi += 1
    return out


def _layout_inline(run: List[Node], sheet: CSSStyleSheet, x: float, y: float,
                   max_w: float, base_fs: float, base_bold: bool) -> Box:
    """Layout a sequence of inline text/element nodes into wrapped lines."""
    box = Box(kind="block", x=x, y=y, w=max_w, h=0)
    tokens = _tokenize_run(run, sheet, base_fs, base_bold)
    cur_x = x
    cur_y = y
    line_h = base_fs * LINE_HEIGHT
    for tok in tokens:
        tw = tok["w"]
        if cur_x + tw > x + max_w and tok["text"].strip():
            cur_x = x
            cur_y += line_h
        if tok["text"].strip() or tok["text"] == " ":
            tbox = Box(kind="text", x=cur_x, y=cur_y, w=tw, h=line_h,
                       text=tok["text"], color=tok["color"],
                       font_size=tok["fs"], bold=tok["bold"])
            box.add(tbox)
        cur_x += tw
    box.h = (cur_y - y) + line_h
    return box


def _tokenize_run(run, sheet, base_fs, base_bold):
    out = []
    for node in run:
        if isinstance(node, Text):
            style = {"color": None}
            fs = base_fs
            bold = base_bold
            out.extend(_split_words(node.text, fs, bold, None))
        elif isinstance(node, Element):
            style = get_styles(node, sheet)
            fs = _px(style.get("font-size"), base_fs, base_fs)
            bold = base_bold or (node.tag in ("b", "strong"))
            color = style.get("color")
            if node.tag == "br":
                out.append({"text": "\n", "w": 0, "fs": fs, "bold": bold, "color": color})
                continue
            text = _inline_text(node)
            out.extend(_split_words(text, fs, bold, color))
    return out


def _inline_text(el: Element) -> str:
    parts = []
    for c in el.children:
        if isinstance(c, Text):
            parts.append(c.text)
        elif isinstance(c, Element):
            parts.append(_inline_text(c))
    return "".join(parts)


def _split_words(text: str, fs: float, bold: bool, color) -> List[dict]:
    # approximate: average glyph width ~0.55em for proportional fonts
    avg = fs * 0.55
    res = []
    # collapse whitespace, treat spaces as breakable
    for word in text.replace("\n", " ").split(" "):
        if word == "":
            res.append({"text": " ", "w": avg, "fs": fs, "bold": bold, "color": color})
            continue
        res.append({"text": word, "w": max(avg, len(word) * avg),
                    "fs": fs, "bold": bold, "color": color})
        res.append({"text": " ", "w": avg, "fs": fs, "bold": bold, "color": color})
    # drop trailing space artifact
    if res and res[-1]["text"] == " ":
        res.pop()
    return res


__all__ = ["layout", "Box", "DEFAULT_STYLES"]
