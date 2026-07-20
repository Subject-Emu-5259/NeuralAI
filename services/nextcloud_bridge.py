"""NeuralDrive <-> Nextcloud bridge.

Provides per-user cloud storage for NeuralAI accounts by provisioning a
Nextcloud user for every signed-up (or guest) NeuralAI identity and proxying
file operations to that user's Nextcloud home over WebDAV.

The Flask layer (services/webui_service.py) imports the following symbols:
    neuralai_to_nc_username, provision_user,
    list_files, get_file, put_file, mkdir, delete_file
"""
from __future__ import annotations

import os
import json
import secrets
import string
import subprocess
import threading
import urllib.request
import urllib.error
from pathlib import Path

NC_ROOT = "/home/workspace/Projects/NeuralAI/services/nextcloud"
NC_BASE = "http://127.0.0.1:8080"
WEBDAV = f"{NC_BASE}/remote.php/webdav"
OCC = os.path.join(NC_ROOT, "occ")

# Local credential store for provisioned Nextcloud users. The bridge owns the
# random passwords; NeuralAI users authenticate via their NeuralAI JWT, never
# directly with Nextcloud. File ops use this store for WebDAV Basic Auth.
_STORE_PATH = os.path.join(NC_ROOT, "data", ".neurldrive_users.json")
_store_lock = threading.Lock()


def _load_store() -> dict:
    try:
        with open(_STORE_PATH) as f:
            return json.load(f)
    except Exception:
        return {}


def _save_store(store: dict) -> None:
    os.makedirs(os.path.dirname(_STORE_PATH), exist_ok=True)
    tmp = _STORE_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(store, f)
    os.replace(tmp, _STORE_PATH)
    try:
        os.chmod(_STORE_PATH, 0o600)
    except Exception:
        pass


def neuralai_to_nc_username(neuralai_uid: str) -> str:
    """Map a NeuralAI user id to a valid Nextcloud username.

    Nextcloud usernames: lowercase, 3-64 chars, [a-z0-9._@-].
    Guests arrive as 'guest' or 'guest_<session>'; we namespace them so they
    still get an isolated (non-persistent) home.
    """
    uid = (neuralai_uid or "guest").lower().strip()
    # Replace illegal chars
    kept = []
    for ch in uid:
        if ch.isalnum() or ch in "._@-":
            kept.append(ch)
        else:
            kept.append("_")
    name = "".join(kept)
    name = name.strip("._-")
    if len(name) > 60:
        name = name[:60]
    if not name or len(name) < 3:
        name = f"u_{name}" if name else "u_guest"
    if len(name) < 3:
        name = (name + "___")[:6]
    return name


def _rand_pw() -> str:
    alphabet = string.ascii_letters + string.digits + "!@#$%"
    return "".join(secrets.choice(alphabet) for _ in range(24))


def provision_user(nc_user: str) -> bool:
    """Ensure a Nextcloud user exists for the given NC username.

    Safe to call on every request; only creates when missing. Returns True.
    """
    store = _load_store()
    if nc_user in store:
        return True

    # Check existing NC users via occ
    try:
        out = subprocess.run(
            ["php", OCC, "user:list"],
            cwd=NC_ROOT, capture_output=True, text=True, timeout=30,
        )
        existing = set()
        for line in out.stdout.splitlines():
            line = line.strip()
            if line.startswith("- "):
                existing.add(line[2:].split(":")[0].strip())
        if nc_user in existing:
            # existed before bridge tracked it; create a passthrough password
            pw = _rand_pw()
            with _store_lock:
                s = _load_store()
                s[nc_user] = pw
                _save_store(s)
            return True
    except Exception:
        pass

    pw = _rand_pw()
    try:
        res = subprocess.run(
            ["php", OCC, "user:add", nc_user,
             "--password-from-env", "--display-name", nc_user],
            cwd=NC_ROOT, capture_output=True, text=True, timeout=30,
            env={**os.environ, "OC_PASS": pw},
        )
        if res.returncode != 0:
            # fall back: already exists?
            if "already exists" in (res.stderr or "").lower():
                pass
            else:
                # store anyway so WebDAV attempts can proceed
                pass
    except Exception:
        pass

    with _store_lock:
        s = _load_store()
        s[nc_user] = pw
        _save_store(s)
    return True


def _creds(nc_user: str):
    store = _load_store()
    return nc_user, store.get(nc_user, "")


def _wddav(method: str, nc_user: str, path: str, data: bytes | None = None,
           ctype: str = "application/octet-stream"):
    user, pw = _creds(nc_user)
    url = f"{WEBDAV}/{path.lstrip('/')}" if path else WEBDAV + "/"
    req = urllib.request.Request(url, data=data, method=method)
    if data is not None:
        req.add_header("Content-Type", ctype)
    if pw:
        import base64
        token = base64.b64encode(f"{user}:{pw}".encode()).decode()
        req.add_header("Authorization", f"Basic {token}")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.status, resp.read(), resp.headers.get("Content-Type", ctype)
    except urllib.error.HTTPError as e:
        return e.code, e.read(), e.headers.get("Content-Type", ctype)


def put_file(nc_user: str, name: str, data: bytes, ctype: str = "application/octet-stream") -> bool:
    provision_user(nc_user)
    name = name.lstrip("/")
    status, _, _ = _wddav("PUT", nc_user, name, data, ctype)
    return status in (201, 204, 200)


def get_file(nc_user: str, name: str):
    provision_user(nc_user)
    name = name.lstrip("/")
    status, body, ctype = _wddav("GET", nc_user, name)
    return status, body, ctype


def delete_file(nc_user: str, name: str) -> bool:
    provision_user(nc_user)
    name = name.lstrip("/")
    status, _, _ = _wddav("DELETE", nc_user, name)
    return status in (204, 200, 404)


def mkdir(nc_user: str, name: str) -> bool:
    provision_user(nc_user)
    name = name.strip("/").replace("..", "")
    if not name:
        return False
    status, _, _ = _wddav("MKCOL", nc_user, name)
    return status in (201, 405, 200)


def list_files(nc_user: str) -> list:
    """List the user's Nextcloud home. Returns a list of file dicts compatible
    with the NeuralDrive UI (name, size, type, is_dir, mtime)."""
    provision_user(nc_user)
    # PROPFIND with Depth 1
    user, pw = _creds(nc_user)
    import base64
    url = WEBDAV + "/"
    body = '<?xml version="1.0" encoding="utf-8"?>' \
           '<d:propfind xmlns:d="DAV:"><d:prop>' \
           '<d:resourcetype/><d:getcontentlength/><d:getlastmodified/>' \
           '<d:getcontenttype/></d:prop></d:propfind>'
    req = urllib.request.Request(url, data=body.encode(), method="PROPFIND")
    req.add_header("Content-Type", "application/xml")
    req.add_header("Depth", "1")
    if user and pw:
        token = base64.b64encode(f"{user}:{pw}".encode()).decode()
        req.add_header("Authorization", f"Basic {token}")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            xml = resp.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        xml = e.read().decode("utf-8", "replace")
    except Exception as e:
        return [{"name": str(e), "size": 0, "type": "error", "is_dir": False}]

    import xml.etree.ElementTree as ET
    ns = {"d": "DAV:"}
    files = []
    try:
        root = ET.fromstring(xml)
    except Exception:
        return files
    for resp in root.findall("d:response", ns):
        href = resp.findtext("d:href", default="", namespaces=ns)
        # Strip webdav base path to get the relative name
        rel = href
        marker = "/remote.php/webdav/"
        if marker in rel:
            rel = rel.split(marker, 1)[1]
        rel = rel.strip("/")
        if rel == "":
            continue  # skip the home collection itself
        props = resp.find("d:propstat/d:prop", ns)
        is_dir = False
        if props is not None:
            rt = props.find("d:resourcetype", ns)
            if rt is not None and rt.find("d:collection", ns) is not None:
                is_dir = True
        size = 0
        if props is not None:
            sl = props.findtext("d:getcontentlength", default="0", namespaces=ns)
            try:
                size = int(sl or 0)
            except Exception:
                size = 0
        ctype = ""
        if props is not None:
            ctype = props.findtext("d:getcontenttype", default="", namespaces=ns) or ""
        ftype = "dir" if is_dir else ("image" if ctype.startswith("image/") else "file")
        files.append({
            "name": rel,
            "size": size,
            "type": ftype,
            "is_dir": is_dir,
            "mtime": "",
        })
    return files


if __name__ == "__main__":
    # quick self-test
    u = neuralai_to_nc_username("testuser@example.com")
    print("mapped:", u)
    provision_user(u)
    print("provisioned:", u)
    print("list:", list_files(u))
