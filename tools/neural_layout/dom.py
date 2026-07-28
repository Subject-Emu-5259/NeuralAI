"""NeuralLayout DOM — HTML parsing into a tree of Element/Text nodes.

Implements the browser.engineering / mbrubeck "Constructing an HTML Tree"
model: a Document root containing Element and Text nodes. We use
BeautifulSoup as the tokenizer/parser front-end (tolerant of real-world
HTML) and normalize its output into our own lightweight Node classes so the
rest of the engine (CSS, layout, paint) is fully self-contained and does
not depend on bs4 internals.

Reference basis:
- browser.engineering/chapters/html.html (HTML parsing -> tree)
- limpet.net/mbrubeck toy engine Part 2 (DOM data structures)
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional

from bs4 import BeautifulSoup


class Node:
    """Base node. Children are Nodes; parent is a Node or None."""
    parent: Optional["Element"] = None

    def __init__(self) -> None:
        self.children: List["Node"] = []
        self.parent = None
        self.tag: str = "#node"

    def add_child(self, child: "Node") -> "Node":
        child.parent = self if isinstance(self, Element) else None
        self.children.append(child)
        return child


@dataclass
class Text:
    """A run of textual content."""
    text: str
    parent: Optional["Element"] = None
    tag: str = "#text"

    def __repr__(self) -> str:  # pragma: no cover
        return f"Text({self.text[:20]!r})"


VALID_TAGS = {
    "html", "head", "body", "title", "meta", "link", "style", "script",
    "div", "span", "p", "a", "img", "h1", "h2", "h3", "h4", "h5", "h6",
    "ul", "ol", "li", "table", "tr", "td", "th", "blockquote", "pre", "code",
    "b", "i", "strong", "em", "small", "br", "hr", "button", "input",
    "form", "label", "header", "footer", "nav", "section", "article",
    "main", "aside", "figure", "figcaption", "video", "audio", "source",
    "iframe", "img", "svg", "path", "picture", "details", "summary",
    "select", "option", "textarea", "time", "mark", "sub", "sup", "u", "s",
}


@dataclass
class Element(Node):
    """An element node with a tag, attributes, and children."""
    tag: str = "div"
    attributes: dict = field(default_factory=dict)
    children: List[Node] = field(default_factory=list)
    parent: Optional["Element"] = None
    # computed style (filled by style module)
    style: dict = field(default_factory=dict)
    # inline style declarations parsed from the style="" attribute
    inline_style: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.children is None:
            self.children = []

    @property
    def id(self) -> Optional[str]:
        return self.attributes.get("id")

    def get_classes(self) -> List[str]:
        c = self.attributes.get("class", "")
        if isinstance(c, list):
            c = " ".join(c)
        return c.split()

    def add_child(self, child: Node) -> Node:  # type: ignore[override]
        child.parent = self
        self.children.append(child)
        return child

    def text_content(self) -> str:
        out = []
        for ch in self.children:
            if isinstance(ch, Text):
                out.append(ch.text)
            elif isinstance(ch, Element):
                out.append(ch.text_content())
        return "".join(out)

    def __repr__(self) -> str:  # pragma: no cover
        return f"Element<{self.tag}>"


class Document(Element):
    """Root of the document tree."""
    def __init__(self) -> None:
        super().__init__()
        self.tag = "#document"
        self.stylesheets: List["CSSStyleSheet"] = []

    def __repr__(self) -> str:  # pragma: no cover
        return "Document()"


def _convert(bs_node, parent: Node) -> Optional[Node]:
    from bs4.element import Tag, NavigableString
    if isinstance(bs_node, NavigableString):
        txt = str(bs_node)
        # collapse pure whitespace between block elements to nothing
        if txt.strip() == "":
            return None
        t = Text(text=txt)
        t.parent = parent if isinstance(parent, Element) else None
        return t
    if isinstance(bs_node, Tag):
        tag = bs_node.name
        if not isinstance(tag, str):
            tag = "div"
        attrs = {}
        for k, v in bs_node.attrs.items():
            if isinstance(v, list):
                v = " ".join(str(x) for x in v)
            attrs[str(k)] = v
        el = Element(tag=tag.lower(), attributes=attrs)
        el.parent = parent if isinstance(parent, Element) else None
        for child in bs_node.children:
            node = _convert(child, el)
            if node is not None:
                el.children.append(node)
        return el
    return None


def parse_html(html: str) -> Document:
    """Parse an HTML string into a NeuralLayout Document tree."""
    soup = BeautifulSoup(html, "html.parser")
    doc = Document()
    for child in soup.children:
        node = _convert(child, doc)
        if node is not None:
            doc.children.append(node)
    return doc


def find_element(node: Node, tag: str) -> Optional[Element]:
    if isinstance(node, Element) and node.tag == tag:
        return node
    for ch in node.children:
        res = find_element(ch, tag)
        if res is not None:
            return res
    return None


def iter_elements(node: Node):
    if isinstance(node, Element):
        yield node
    for ch in node.children:
        yield from iter_elements(ch)


# Public alias used by __init__ / css / render callers.
TextNode = Text
