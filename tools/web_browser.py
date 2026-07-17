# tools/web_browser.py
# Agentic browsing with Playwright (headless Chromium)
# Supports: navigate, click links by text, fill forms, extract text.
#
# Thread-safety note: a Playwright `sync_playwright()` object is bound to the
# OS thread that created it. Flask serves each /api/tool request on a worker
# thread, so we MUST create AND drive the browser from the SAME thread.
# We achieve this with a single dedicated executor thread per session that
# owns the Playwright/Browser/Page objects and runs every action on itself.
from __future__ import annotations

import re
import threading
import concurrent.futures
from typing import Any, Dict, List, Optional

from playwright.sync_api import sync_playwright, Browser, Page


_SESSIONS: Dict[str, "BrowserSession"] = {}
_LOCK = threading.Lock()


class BrowserSession:
    def __init__(self, session_id: str):
        self.session_id = session_id
        # Dedicated thread that owns the Playwright event loop.
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix=f"pw-{session_id}"
        )
        fut = self._executor.submit(self._bootstrap)
        fut.result(timeout=60)  # raise bootstrap errors to caller

    def _bootstrap(self):
        self.pw = sync_playwright().start()
        self.browser: Browser = self.pw.chromium.launch(headless=True)
        self.context = self.browser.new_context(
            user_agent=(
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/124 Safari/537.36"
            )
        )
        self.page: Page = self.context.new_page()
        self.page.set_default_timeout(20000)

    def _run(self, fn, *args, **kwargs):
        # Always execute on the owning thread.
        return self._executor.submit(fn, *args, **kwargs).result(timeout=120)

    def run(self, url: str, steps: List[str]) -> str:
        return self._run(self._run_impl, url, steps)

    def _run_impl(self, url: str, steps: List[str]) -> str:
        try:
            self.page.goto(url, wait_until="domcontentloaded", timeout=30000)
            self.page.wait_for_timeout(1500)
            log = [f"🌐 Loaded: {self.page.url}"]
            log.append(f"📄 Title: {self.page.title()}")

            if not steps:
                text = self._extract_text()
                log.append(f"\n--- Page text (first 1500 chars) ---\n{text[:1500]}")
                return "\n".join(log)

            for step in steps:
                out = self._do_step(step)
                if out:
                    log.append(out)
            log.append(f"\n--- Current page: {self.page.url} ---")
            log.append(self._extract_text()[:1500])
            return "\n".join(log)
        except Exception as e:
            return f"❌ Browse run error: {e}"

    def _do_step(self, step: str) -> str:
        s = step.lower()
        m = re_click(s)
        if m:
            try:
                self.page.get_by_text(m, exact=False).first.click()
                self.page.wait_for_timeout(1200)
                return f"🖱️ Clicked: {m} → now at {self.page.url}"
            except Exception as e:
                return f"⚠️ Click failed for '{m}': {e}"
        m = re_fill(s)
        if m:
            sel, val = m
            try:
                self.page.fill(sel, val, timeout=8000)
                return f"✍️ Filled '{sel}' with '{val}'"
            except Exception as e:
                try:
                    self.page.get_by_placeholder(sel, exact=False).fill(val)
                    return f"✍️ Filled placeholder '{sel}' with '{val}'"
                except Exception as e2:
                    return f"⚠️ Fill failed for '{sel}': {e2}"
        m = re_go(s)
        if m:
            self.page.goto(m, wait_until="domcontentloaded")
            self.page.wait_for_timeout(1200)
            return f"➡️ Navigated to {m}"
        if "scroll down" in s:
            self.page.mouse.wheel(0, 1200)
            self.page.wait_for_timeout(600)
            return "🔽 Scrolled down"
        if "scroll up" in s:
            self.page.mouse.wheel(0, -1200)
            self.page.wait_for_timeout(600)
            return "🔼 Scrolled up"
        if any(k in s for k in ["type ", "search ", "enter "]):
            val = step.split(maxsplit=1)[-1] if " " in step else step
            try:
                self.page.keyboard.type(val)
                self.page.keyboard.press("Enter")
                self.page.wait_for_timeout(1500)
                return f"⌨️ Typed '{val}' and pressed Enter → {self.page.url}"
            except Exception as e:
                return f"⚠️ Type failed: {e}"
        return f"⏭️ Unrecognized step skipped: {step}"

    def _extract_text(self) -> str:
        try:
            return self.page.inner_text()
        except Exception:
            return ""

    def close(self):
        try:
            self._executor.submit(self._close_impl).result(timeout=30)
        except Exception:
            pass
        self._executor.shutdown(wait=False)

    def _close_impl(self):
        try:
            self.context.close()
            self.browser.close()
            self.pw.stop()
        except Exception:
            pass


def re_click(s: str):
    m = re.search(r"click (?:on |the )?['\"]?([^'\"]+)['\"]?", s)
    return m.group(1) if m else None


def re_fill(s: str):
    m = re.search(r"fill (.+?) with (.+)", s)
    return (m.group(1).strip(), m.group(2).strip()) if m else None


def re_go(s: str):
    m = re.search(r"go (https?://\S+)", s)
    return m.group(1) if m else None


def get_session(session_id: str = "default") -> BrowserSession:
    with _LOCK:
        sess = _SESSIONS.get(session_id)
        if sess is None:
            sess = BrowserSession(session_id)
            _SESSIONS[session_id] = sess
        return sess


def close_session(session_id: str = "default"):
    with _LOCK:
        sess = _SESSIONS.pop(session_id, None)
    if sess:
        sess.close()
