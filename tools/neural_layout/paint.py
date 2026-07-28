"""NeuralLayout — rasterization (paint).

Implements mbrubeck toy-engine Part 7 (Painting 101): traverse the layout
tree, build a display list (rectangles + text), and rasterize to PNG via PIL
(with an SVG fallback so the engine still works where PIL/image deps are
absent). The web UI shows the PNG; the SVG is surfaced as a debug artifact
(the old browser also returned screenshots, so UI parity is preserved).
"""
from __future__ import annotations

import base64
from typing import List, Optional

from .layout import Box


def _color(v: Optional[str], default: str = "#111111") -> str:
    if not v:
        return default
    v = str(v).strip()
    if v.lower() in ("transparent", "none", "auto", ""):
        return default
    return v


def _paint_png(box: Box, img, draw):
    # background for block boxes
    if box.kind == "block" and box.bg:
        c = _color(box.bg, "#ffffff")
        if c.lower() != "#ffffff":
            try:
                draw.rectangle([box.x, box.y, box.x + box.w, box.y + box.h], fill=c)
            except Exception:
                pass

    # text runs
    if box.kind == "text" and box.text:
        try:
            draw.text((box.x, box.y), box.text, fill=_color(box.color))
        except Exception:
            pass

    for ch in box.children:
        _paint_png(ch, img, draw)


def render_png(box: Box, width: int, height: int, out_path: str) -> bool:
    try:
        from PIL import Image, ImageDraw
    except Exception:
        return False
    try:
        img = Image.new("RGB", (max(1, int(width)), max(1, int(height))), "#ffffff")
        draw = ImageDraw.Draw(img)
        _paint_png(box, img, draw)
        img.save(out_path, "PNG")
        return True
    except Exception:
        return False


def _paint_svg(box: Box, parts: List[str]):
    if box.kind == "block" and box.bg:
        c = _color(box.bg, "#ffffff")
        if c.lower() != "#ffffff":
            parts.append(
                f'<rect x="{box.x:.0f}" y="{box.y:.0f}" width="{box.w:.0f}" '
                f'height="{box.h:.0f}" fill="{c}"/>'
            )
    if box.kind == "text" and box.text:
        parts.append(
            f'<text x="{box.x:.0f}" y="{box.y + box.h * 0.8:.0f}" '
            f'font-size="{box.font_size:.0f}" fill="{_color(box.color)}">'
            f'{_esc(box.text)}</text>'
        )
    for ch in box.children:
        _paint_svg(ch, parts)


def _esc(s: str) -> str:
    return (s or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def render_svg(box: Box, width: int, height: int) -> str:
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff"/>',
    ]
    _paint_svg(box, parts)
    parts.append("</svg>")
    return "\n".join(parts)


def render_base64_svg(box: Box, width: int, height: int) -> str:
    svg = render_svg(box, width, height)
    return "data:image/svg+xml;base64," + base64.b64encode(svg.encode()).decode()


__all__ = ["render_png", "render_svg", "render_base64_svg"]
