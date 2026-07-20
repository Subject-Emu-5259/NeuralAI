# tools/web_browser.py
# REAL embedded web browser built on Playwright (Chromium).
# Reference build: the browser.engineering "build your own browser" pipeline
# (fetch -> parse -> DOM -> CSSOM -> layout -> paint) plus Playwright's
# Python Page/Context API for genuine multi-tab, navigation, clicks, scroll,
# and live screenshots — hardened for the gVisor sandbox.
#
# Per-user BrowserManager owns ONE Chromium browser instance and a list of
# tabs (Playwright pages). The Flask UI calls these endpoints; each page's
# work runs on a dedicated owner thread (_run) so Playwright stays thread-safe.

from __future__ import annotations

import concurrent.futures

import threading
import queue
import base64
import io
import re
from typing import Any, Dict, List, Optional

from playwright.sync_api import sync_playwright, Browser, BrowserContext, Page


_MANAGERS: Dict[str, "BrowserManager"] = {}
_LOCK = threading.Lock()

_LAUNCH_ARGS = [
    "--no-sandbox",
    "--disable-setuid-sandbox",
    "--disable-dev-shm-usage",
    "--disable-gpu",
    "--single-process",
    "--no-zygote",
]
_DEFAULT_VIEWPORT = {"width": 1280, "height": 800}
_HOME = "https://www.google.com"


def _to_data_url(buf: bytes, fmt: str = "jpeg") -> str:
    return f"data:image/{fmt};base64," + base64.b64encode(buf).decode("ascii")


class BrowserManager:
    """Thread-actor browser manager.

    All Playwright/Chromium operations run on ONE dedicated background thread
    (the "browser thread"). Flask serves requests on multiple threads
    (app.run(threaded=True)), but Chromium under gVisor must be driven from a
    single thread, otherwise you get "cannot switch to a different thread (which
    has exited)". Request threads submit a callable + args to a queue and block
    on a result future until the browser thread finishes it.
    """

    def __init__(self, sid: str):
        self.sid = sid
        self._q: "queue.Queue" = queue.Queue()
        self._started = threading.Event()
        self._ready = threading.Event()
        self._thread = threading.Thread(target=self._loop, name=f"bm-{sid}", daemon=True)
        self._thread.start()
        # Wait briefly for the browser to come up so the first navigate is fast.
        self._ready.wait(timeout=30)

    def _loop(self):
        """Owns Chromium. Processes submitted jobs sequentially."""
        p = None
        browser = None
        ctx = None
        tabs: List[Page] = []
        active = [0]

        def new_page() -> Page:
            page = ctx.new_page()
            page.set_viewport_size(_DEFAULT_VIEWPORT)
            return page

        def add_tab(url: str = _HOME) -> int:
            page = new_page()
            tabs.append(page)
            idx = len(tabs) - 1
            active[0] = idx
            try:
                page.goto(url, wait_until="domcontentloaded", timeout=30000)
            except Exception:
                pass
            return idx

        def snapshot(i: int) -> Dict[str, Any]:
            if i < 0 or i >= len(tabs):
                return {"index": i, "title": "", "url": "", "screenshot": "", "links": []}
            page = tabs[i]
            try:
                buf = page.screenshot(type="jpeg", quality=72, full_page=False)
                shot = _to_data_url(buf, "jpeg")
            except Exception:
                shot = ""
            try:
                title = page.title()
                url = page.url
            except Exception:
                title, url = "", ""
            try:
                raw = page.eval_on_selector_all(
                    "a[href]",
                    "els => els.map(e => ({href: e.href, text: (e.innerText||'').trim().slice(0,120)}))",
                )
                seen = set()
                links = []
                for r in raw:
                    h = r.get("href", "")
                    if not h or h.startswith(("javascript:", "mailto:", "tel:")):
                        continue
                    if h in seen:
                        continue
                    seen.add(h)
                    links.append({"href": h, "text": r.get("text", "") or h})
                links = links[:200]
            except Exception:
                links = []
            rects = []
            try:
                rects = page.evaluate(
                    "() => Array.from(document.querySelectorAll('a[href]')).map(a => { const r = a.getBoundingClientRect(); return {x: r.x, y: r.y, w: r.width, h: r.height, href: a.href}; }).filter(b => b.w > 0 && b.h > 0)"
                ) or []
            except Exception:
                rects = []
            return {
                "index": i,
                "title": title,
                "url": url,
                "screenshot": shot,
                "links": links,
                "link_rects": rects,
            }

        try:
            p = sync_playwright().start()
            browser = p.chromium.launch(args=_LAUNCH_ARGS)
            ctx = browser.new_context(
                viewport=_DEFAULT_VIEWPORT,
                user_agent=(
                    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/124 Safari/537.36"
                ),
            )
            add_tab(_HOME)
            self._ready.set()
        except Exception as e:
            self._ready.set()
            self._fatal = str(e)
            return

        while True:
            item = self._q.get()
            if item is None:
                break
            fn, args, kwargs, future = item
            try:
                # Bind the helpers into the submitted function's closure space
                # by passing them explicitly.
                result = fn(
                    *args,
                    **kwargs,
                    _helpers={"add_tab": add_tab, "snapshot": snapshot, "tabs": tabs, "get_active": lambda: active[0], "set_active": lambda v: active.__setitem__(0, v)},
                )
                future.set_result(result)
            except Exception as e:
                future.set_exception(e)

    def _submit(self, fn, *args, **kwargs):
        if not self._ready.is_set():
            raise RuntimeError(getattr(self, "_fatal", "browser not ready"))
        fut: "concurrent.futures.Future" = concurrent.futures.Future()
        self._q.put((fn, args, kwargs, fut))
        return fut.result(timeout=90)

    # -- public API (runs on the request thread, submits to browser thread) --
    def list_tabs(self) -> List[Dict[str, Any]]:
        def _do(_helpers):
            return [_helpers["snapshot"](i) for i in range(len(_helpers["tabs"]))]
        return self._submit(_do)

    def new_tab(self, url: str = _HOME) -> Dict[str, Any]:
        def _do(_helpers):
            idx = _helpers["add_tab"](url)
            return _helpers["snapshot"](idx)
        return self._submit(_do)

    def close_tab(self, idx: int) -> List[Dict[str, Any]]:
        def _do(_helpers):
            tabs = _helpers["tabs"]
            if idx < 0 or idx >= len(tabs):
                return [_helpers["snapshot"](i) for i in range(len(tabs))]
            try:
                tabs[idx].close()
            except Exception:
                pass
            tabs.pop(idx)
            if not tabs:
                _helpers["add_tab"](_HOME)
            return [_helpers["snapshot"](i) for i in range(len(tabs))]
        return self._submit(_do)

    def select(self, idx: int) -> Dict[str, Any]:
        def _do(_helpers):
            tabs = _helpers["tabs"]
            if 0 <= idx < len(tabs):
                _helpers["set_active"](idx)
            return _helpers["snapshot"](_helpers["get_active"]())
        return self._submit(_do)

    def navigate(self, url: str, tab_id: Optional[int] = None) -> Dict[str, Any]:
        def _do(_helpers):
            tabs = _helpers["tabs"]
            i = int(_helpers["get_active"]()) if tab_id is None else int(tab_id)
            if i < 0 or i >= len(tabs):
                return {"success": False, "error": "no such tab"}
            u = url if url.startswith(("http://", "https://")) else "https://" + url
            try:
                tabs[i].goto(u, wait_until="domcontentloaded", timeout=30000)
                _helpers["set_active"](i)
                return _helpers["snapshot"](i)
            except Exception as e:
                return {"success": False, "error": str(e), "url": u}
        # normalize active to int for safety
        return self._submit(_do)

    def action(self, action: str, extra: Optional[Dict[str, Any]] = None, tab_id: Optional[int] = None) -> Dict[str, Any]:
        def _do(_helpers):
            tabs = _helpers["tabs"]
            i = _helpers["get_active"]() if tab_id is None else int(tab_id)
            if i < 0 or i >= len(tabs):
                return {"success": False, "error": "no such tab"}
            opts = extra or {}
            page = tabs[i]
            try:
                if action == "back":
                    try: page.go_back(timeout=15000)
                    except Exception: pass
                elif action == "forward":
                    try: page.go_forward(timeout=15000)
                    except Exception: pass
                elif action == "reload":
                    try: page.reload(wait_until="domcontentloaded", timeout=30000)
                    except Exception: pass
                elif action in ("scroll_up", "scroll_down"):
                    amt = int(opts.get("amount", 800))
                    page.mouse.wheel(0, -amt if action == "scroll_up" else amt)
                    page.wait_for_timeout(300)
                elif action == "click":
                    x = float(opts.get("x", 0)); y = float(opts.get("y", 0))
                    page.mouse.click(x, y)
                    page.wait_for_timeout(300)
                _helpers["set_active"](i)
                return _helpers["snapshot"](i)
            except Exception as e:
                return {"success": False, "error": str(e)}
        return self._submit(_do)

    def run(self, url: str, steps: Optional[List[str]] = None) -> str:
        nav = self.navigate(url)
        if not nav.get("success"):
            return f"[NAV_FAIL] {nav.get('error', 'unknown')}"
        steps = steps or []
        out: List[str] = []
        for s in steps:
            low = (s or "").lower()
            try:
                if low.startswith("click"):
                    m = re.search(r"https?://\S+|[\w.-]+\.[a-z]{2,}(/\S*)?", s)
                    target = m.group(0) if m else ""
                    if target:
                        r = self.click(target)
                        out.append(f"clicked {target}" if r.get("success") else f"click failed: {r.get('error')}")
                elif low.startswith("scroll"):
                    r = self.action("scroll_down")
                    out.append("scrolled" if r.get("success") else "scroll failed")
                else:
                    r = self.action(s)
                    out.append(f"did: {s}" if r.get("success") else f"failed: {r.get('error')}")
            except Exception as e:
                out.append(f"step error: {e}")
        snap = self.tab_snapshot()
        out.append(f"url={snap.get('url')} title={snap.get('title')}")
        return "\n".join(out)

    def click(self, href: str, idx: Optional[int] = None) -> Dict[str, Any]:
        def _do(_helpers):
            tabs = _helpers["tabs"]
            i = _helpers["get_active"]() if idx is None else idx
            if i < 0 or i >= len(tabs):
                return {"success": False, "error": "no such tab"}
            page = tabs[i]
            target = href
            if not href.startswith(("http://", "https://")):
                from urllib.parse import urljoin
                target = urljoin(page.url or _HOME, href)
            try:
                page.goto(target, wait_until="domcontentloaded", timeout=30000)
                _helpers["set_active"](i)
                return _helpers["snapshot"](i)
            except Exception as e:
                return {"success": False, "error": str(e), "url": href}
        return self._submit(_do)

    def set_zoom(self, level: int, idx: Optional[int] = None) -> Dict[str, Any]:
        def _do(_helpers):
            tabs = _helpers["tabs"]
            i = _helpers["get_active"]() if idx is None else idx
            if 0 <= i < len(tabs):
                try:
                    tabs[i].evaluate(f"document.body.style.zoom = {level/100}")
                except Exception:
                    pass
            return _helpers["snapshot"](i)
        return self._submit(_do)

    def tab_snapshot(self, idx: Optional[int] = None) -> Dict[str, Any]:
        def _do(_helpers):
            i = _helpers["get_active"]() if idx is None else int(idx)
            return _helpers["snapshot"](i)
        return self._submit(_do)

    def tabs(self) -> List[Dict[str, Any]]:
        return self.list_tabs()

    @property
    def active(self) -> int:
        def _do(_helpers):
            return _helpers["get_active"]()
        return self._submit(_do)

    def get_tab(self, idx: Optional[int] = None) -> Dict[str, Any]:
        return self.tab_snapshot(idx)

    def close(self):
        try:
            self._q.put(None)
        except Exception:
            pass

def get_manager(sid: str = "default") -> BrowserManager:
    with _LOCK:
        mgr = _MANAGERS.get(sid)
        if mgr is None:
            mgr = BrowserManager(sid)
            _MANAGERS[sid] = mgr
        return mgr


def close_session(sid: str = "default"):
    with _LOCK:
        mgr = _MANAGERS.pop(sid, None)
    if mgr:
        try:
            mgr.close()
        except Exception:
            pass


def close_manager(sid: str = "default"):
    """Alias used by the /api/browser/close route."""
    close_session(sid)
