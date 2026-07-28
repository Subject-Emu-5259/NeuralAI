"""
NeuralBrowser Live Proxy — stealth iframe proxy for public browsing.

Strips X-Frame-Options / CSP frame-ancestors so arbitrary public sites can be
embedded in a live <iframe> with a real cursor and instant clicks. Rewrites
relative asset URLs so images/css/js load through this proxy. Blocks private
IPs / localhost (no SSRF to Zo internals). Public, no auth.

Run: uvicorn browser_proxy:app --host 0.0.0.0 --port 8080
"""
import ipaddress
import re
import urllib.parse
from fastapi import FastAPI, Request, Response, HTTPException
from fastapi.responses import Response as Resp, PlainTextResponse
import httpx

app = FastAPI()
client = httpx.AsyncClient(follow_redirects=True, timeout=20.0, verify=False)

STRIP_HEADERS = {
    "x-frame-options",
    "content-security-policy",
    "content-security-policy-report-only",
    "frame-options",
}
HOP_BY_HOP = {
    "content-length", "transfer-encoding", "connection", "keep-alive",
    "proxy-authenticate", "proxy-authorization", "te", "trailers", "upgrade",
}

_BLOCK_PRIVATE = True


def _is_private_host(host: str) -> bool:
    host = host.strip().lower()
    if host in ("localhost", "0.0.0.0", "::1"):
        return True
    # strip brackets for ipv6
    if host.startswith("[") and host.endswith("]"):
        host = host[1:-1]
    try:
        ip = ipaddress.ip_address(host)
        return ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast
    except ValueError:
        return False


def _safe_url(raw: str) -> str:
    if not raw:
        raise HTTPException(400, "missing url")
    if len(raw) > 4096:
        raise HTTPException(414, "url too long")
    try:
        p = urllib.parse.urlparse(raw)
    except Exception:
        raise HTTPException(400, "bad url")
    if p.scheme not in ("http", "https"):
        raise HTTPException(400, "only http/https allowed")
    if _BLOCK_PRIVATE and _is_private_host(p.hostname or ""):
        raise HTTPException(403, "blocked: private/loopback target")
    return raw


def _rewrite_html(html: str, base_url: str, proxy_root: str) -> str:
    """Rewrite relative/absolute URLs in HTML so assets flow through the proxy.
    base_url: the real target page URL. proxy_root: this proxy's origin.
    """
    base_parsed = urllib.parse.urlparse(base_url)
    proxy_origin = proxy_root.rstrip("/")

    def to_proxy(u: str) -> str:
        u = u.strip()
        if not u or u.startswith(("#", "data:", "javascript:", "mailto:", "tel:")):
            return u
        if u.startswith("//"):
            u = base_parsed.scheme + ":" + u
        if u.startswith("/"):
            absu = f"{base_parsed.scheme}://{base_parsed.netloc}{u}"
        elif u.startswith("http://") or u.startswith("https://"):
            absu = u
        else:
            absu = urllib.parse.urljoin(base_url, u)
        return f"{proxy_origin}/?url={urllib.parse.quote(absu, safe='')}"

    # href / src
    html = re.sub(r'(href\s*=\s*")([^"]*)(")', lambda m: m.group(1) + to_proxy(m.group(2)) + m.group(3), html, flags=re.IGNORECASE)
    html = re.sub(r"(href\s*=\s*')([^']*)(')", lambda m: m.group(1) + to_proxy(m.group(2)) + m.group(3), html, flags=re.IGNORECASE)
    html = re.sub(r'(src\s*=\s*")([^"]*)(")', lambda m: m.group(1) + to_proxy(m.group(2)) + m.group(3), html, flags=re.IGNORECASE)
    html = re.sub(r"(src\s*=\s*')([^']*)(')", lambda m: m.group(1) + to_proxy(m.group(2)) + m.group(3), html, flags=re.IGNORECASE)
    # CSS url() inside style attrs / tags
    html = re.sub(r'(url\s*\(\s*["\']?)([^"\')\s]+)(["\']?\s*\))', lambda m: m.group(1) + to_proxy(m.group(2)) + m.group(3), html, flags=re.IGNORECASE)
    # <base> would break our relative rewrites; neutralize it.
    html = re.sub(r"<base[^>]*>", "", html, flags=re.IGNORECASE)
    # CSP meta tags
    html = re.sub(r'<meta[^>]*http-equiv\s*=\s*["\']?content-security-policy["\']?[^>]*>', "", html, flags=re.IGNORECASE)
    return html


@app.get("/healthz")
async def healthz():
    return {"ok": True}


@app.get("/")
async def proxy(request: Request, url: str = ""):
    if not url:
        return PlainTextResponse(
            "NeuralBrowser Live Proxy. Use /?url=<target> to browse a public site.",
            status_code=200,
        )
    target = _safe_url(url)
    # Build forward headers (strip proxy-specific)
    fwd = {k: v for k, v in request.headers.items() if k.lower() not in ("host", "connection", "accept-encoding")}
    try:
        r = await client.get(target, headers=fwd)
    except httpx.HTTPError as e:
        raise HTTPException(502, f"upstream error: {e}")

    ctype = (r.headers.get("content-type") or "text/plain").lower()
    body = r.content

    if "text/html" in ctype:
        proxy_root = str(request.base_url).rstrip("/")
        body = _rewrite_html(body.decode("utf-8", "replace"), target, proxy_root).encode("utf-8")
        ctype = "text/html; charset=utf-8"

    out = Resp(content=body, status_code=r.status_code)
    for k, v in r.headers.items():
        lk = k.lower()
        if lk in STRIP_HEADERS or lk in HOP_BY_HOP:
            continue
        out.headers[k] = v
    # Hard guarantee: never let the frame be blocked.
    out.headers["X-Frame-Options"] = "ALLOWALL"
    out.headers["Content-Security-Policy"] = "frame-ancestors *"
    # Tell browsers we meant to allow embedding.
    out.headers["Access-Control-Allow-Origin"] = "*"
    out.headers["Content-Type"] = ctype
    return out
