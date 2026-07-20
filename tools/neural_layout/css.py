"""NeuralLayout CSS — parser, selector model, and specificity.

Implements the mbrubeck toy-engine Part 3 (CSS parsing) + Part 4 (Style)
and the browser.engineering "Applying Author Styles" chapter:
- Parse a simplified CSS subset (selectors + declarations).
- Represent selectors as (tag, #id, .classes) for matching.
- Compute specificity (a,b,c) and sort so higher specificity wins.
- Cascade: author rules then inline style attribute (most specific).

Not standards-complete; tolerant of unrecognized input (skips it) per
mbrubeck Part 3 guidance.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class Declaration:
    name: str
    value: str


@dataclass
class Rule:
    selectors: List["Selector"]
    declarations: List[Declaration]


@dataclass
class Selector:
    tag: Optional[str] = None
    id: Optional[str] = None
    classes: List[str] = field(default_factory=list)
    # compound (space-separated) descendant chain, most-specific last
    chain: List["Selector"] = field(default_factory=list)

    def specificity(self) -> Tuple[int, int, int]:
        a = 1 if self.id else 0
        b = len(self.classes)
        c = 1 if self.tag else 0
        return (a, b, c)

    def matches(self, el) -> bool:  # el: Element
        if self.tag and self.tag != "*" and el.tag != self.tag:
            return False
        if self.id and el.id != self.id:
            return False
        el_classes = set(el.get_classes())
        for c in self.classes:
            if c not in el_classes:
                return False
        return True


@dataclass
class CSSStyleSheet:
    rules: List[Rule] = field(default_factory=list)


def _tokenize_value(value: str) -> str:
    return value.strip().rstrip(";").strip()


def parse_declarations(block: str) -> List[Declaration]:
    """Parse the body of a rule: `prop: val; prop2: val2;`."""
    out: List[Declaration] = []
    for part in block.split(";"):
        part = part.strip()
        if not part or ":" not in part:
            continue
        name, _, val = part.partition(":")
        name = name.strip().lower()
        val = val.strip()
        if name and val:
            out.append(Declaration(name=name, value=val))
    return out


def _parse_selector(text: str) -> Selector:
    text = text.strip()
    sel = Selector()
    # handle compound (descendant) by taking the last simple selector for match
    simples = [s for s in text.split() if s]
    primary = simples[-1] if simples else "*"
    i = 0
    buf = ""
    for ch in primary:
        if ch == "#":
            if buf:
                sel.tag = buf if sel.tag is None else sel.tag
            buf = ""
            i = 1
        elif ch == ".":
            if buf:
                sel.tag = buf if sel.tag is None else sel.tag
            buf = ""
            i = 2
        else:
            buf += ch
    if i == 1:
        sel.id = buf
    elif i == 2:
        if buf:
            sel.classes.append(buf)
    else:
        sel.tag = buf if buf else None
    return sel


def parse_stylesheet(css: str) -> CSSStyleSheet:
    """Parse a full CSS string into a stylesheet of rules."""
    sheet = CSSStyleSheet()
    i = 0
    n = len(css)
    while i < n:
        # find next rule opener
        if css[i] == "/" and i + 1 < n and css[i + 1] == "*":
            end = css.find("*/", i + 2)
            i = end + 2 if end != -1 else n
            continue
        open_brace = css.find("{", i)
        if open_brace == -1:
            break
        pre = css[i:open_brace]
        close_brace = css.find("}", open_brace)
        if close_brace == -1:
            break
        body = css[open_brace + 1:close_brace]
        # split selectors on commas
        for sel_text in pre.split(","):
            sel_text = sel_text.strip()
            if not sel_text:
                continue
            sel = _parse_selector(sel_text)
            if sel is None:
                continue
            decls = parse_declarations(body)
            if decls:
                sheet.rules.append(Rule(selectors=[sel], declarations=decls))
        i = close_brace + 1
    return sheet


def parse_inline_style(style_attr: str) -> dict:
    out = {}
    for d in parse_declarations(style_attr):
        out[d.name] = d.value
    return out


def style_for_element(el, sheet: CSSStyleSheet) -> dict:
    """Compute the cascaded style dict for one element.

    Author rules (sorted by specificity) then inline style override.
    """
    # gather matching rules with specificity
    matched = []
    for rule in sheet.rules:
        for sel in rule.selectors:
            if sel.matches(el):  # type: ignore[arg-type]
                matched.append((sel.specificity(), rule))
                break
    matched.sort(key=lambda x: x[0])
    style: dict = {}
    for _, rule in matched:
        for d in rule.declarations:
            style[d.name] = d.value
    # inline style is most specific
    if el.inline_style:
        for k, v in el.inline_style.items():
            style[k] = v
    return style


# Default author-equivalent stylesheet (browser.engineering "user styles" subset).
# Provides sane defaults so pages render even with no author CSS.
DEFAULT_STYLES = """
* { font-size: 16px; color: #111111; }
body { display: block; margin: 8px; }
h1 { display: block; font-size: 32px; font-weight: bold; margin: 8px 0; }
h2 { display: block; font-size: 24px; font-weight: bold; margin: 8px 0; }
h3 { display: block; font-size: 20px; font-weight: bold; margin: 8px 0; }
h4 { display: block; font-size: 18px; font-weight: bold; margin: 8px 0; }
h5, h6 { display: block; font-weight: bold; margin: 8px 0; }
p { display: block; margin: 8px 0; }
blockquote { display: block; margin: 8px 16px; font-style: italic; }
ul, ol { display: block; margin: 8px 0; padding-left: 32px; }
li { display: list-item; }
a { display: inline; color: #0000ee; }
em, i { font-style: italic; }
strong, b { font-weight: bold; }
hr { display: block; border-top: 1px solid #999; margin: 8px 0; height: 0; }
img { display: inline; }
code, pre { font-family: monospace; }
pre { display: block; white-space: pre; margin: 8px 0; }
table { display: block; }
tr { display: block; }
td, th { display: inline; padding: 4px; }
script, style { display: none; }
"""


# Backwards-friendly alias used by layout.py / paint.py.
def get_styles(el, sheet):
    return style_for_element(el, sheet)
