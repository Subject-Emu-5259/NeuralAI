# tools/memory_graph.py
# Phase 8: Knowledge Graph (local DuckDB) + Supermemory sync (long-term cross-project memory)
import os
import json
import sqlite3
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional

# Local knowledge graph backed by DuckDB (fast, file-based, no server)
try:
    import duckdb
    _DB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "memory_graph.duckdb")
    _con = duckdb.connect(_DB)
    _con.execute("""
        CREATE TABLE IF NOT EXISTS nodes (
            id VARCHAR PRIMARY KEY,
            label VARCHAR,
            type VARCHAR,
            props JSON,
            created_at TIMESTAMP DEFAULT now()
        )
    """)
    _con.execute("""
        CREATE TABLE IF NOT EXISTS edges (
            src VARCHAR,
            dst VARCHAR,
            rel VARCHAR,
            props JSON,
            created_at TIMESTAMP DEFAULT now()
        )
    """)
    _USE_DUCKDB = True
except Exception:
    # Fallback to sqlite if duckdb unavailable
    _DB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "memory_graph.db")
    _con = sqlite3.connect(_DB)
    _USE_DUCKDB = False


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def add_node(label: str, node_type: str = "entity", props: Optional[Dict] = None, node_id: Optional[str] = None) -> str:
    node_id = node_id or f"{node_type}:{label.lower().replace(' ', '_')}"
    props = props or {}
    if _USE_DUCKDB:
        _con.execute(
            "INSERT OR REPLACE INTO nodes (id, label, type, props, created_at) VALUES (?,?,?,?,?)",
            [node_id, label, node_type, json.dumps(props), _now()],
        )
    else:
        _con.execute(
            "INSERT OR REPLACE INTO nodes (id, label, type, props, created_at) VALUES (?,?,?,?,?)",
            (node_id, label, node_type, json.dumps(props), _now()),
        )
    return node_id


def add_edge(src: str, dst: str, rel: str, props: Optional[Dict] = None) -> None:
    props = props or {}
    if _USE_DUCKDB:
        _con.execute(
            "INSERT INTO edges (src, dst, rel, props, created_at) VALUES (?,?,?,?,?)",
            [src, dst, rel, json.dumps(props), _now()],
        )
    else:
        _con.execute(
            "INSERT INTO edges (src, dst, rel, props, created_at) VALUES (?,?,?,?,?)",
            (src, dst, rel, json.dumps(props), _now()),
        )


def search_nodes(query: str, limit: int = 10) -> List[Dict[str, Any]]:
    like = f"%{query.lower()}%"
    if _USE_DUCKDB:
        rows = _con.execute(
            "SELECT id, label, type, props FROM nodes WHERE lower(label) LIKE ? OR lower(type) LIKE ? ORDER BY created_at DESC LIMIT ?",
            [like, like, limit],
        ).fetchall()
    else:
        cur = _con.execute(
            "SELECT id, label, type, props FROM nodes WHERE lower(label) LIKE ? OR lower(type) LIKE ? ORDER BY created_at DESC LIMIT ?",
            (like, like, limit),
        )
        rows = cur.fetchall()
    out = []
    for r in rows:
        out.append({"id": r[0], "label": r[1], "type": r[2], "props": json.loads(r[3]) if r[3] else {}})
    return out


def neighbors(node_id: str, limit: int = 20) -> List[Dict[str, Any]]:
    if _USE_DUCKDB:
        rows = _con.execute(
            "SELECT src, dst, rel, props FROM edges WHERE src = ? OR dst = ? ORDER BY created_at DESC LIMIT ?",
            [node_id, node_id, limit],
        ).fetchall()
    else:
        cur = _con.execute(
            "SELECT src, dst, rel, props FROM edges WHERE src = ? OR dst = ? ORDER BY created_at DESC LIMIT ?",
            (node_id, node_id, limit),
        )
        rows = cur.fetchall()
    out = []
    for r in rows:
        out.append({"src": r[0], "dst": r[1], "rel": r[2], "props": json.loads(r[3]) if r[3] else {}})
    return out


def stats() -> Dict[str, int]:
    if _USE_DUCKDB:
        n = _con.execute("SELECT count(*) FROM nodes").fetchone()[0]
        e = _con.execute("SELECT count(*) FROM edges").fetchone()[0]
    else:
        n = _con.execute("SELECT count(*) FROM nodes").fetchone()[0]
        e = _con.execute("SELECT count(*) FROM edges").fetchone()[0]
    return {"nodes": n, "edges": e, "backend": "duckdb" if _USE_DUCKDB else "sqlite"}


# --- Supermemory long-term sync ---
_SUPERMEMORY_URL = "https://api.supermemory.ai/v1/memories"


def _sm_key() -> str:
    return os.environ.get("SUPERMEMORY_API_KEY", "")


def supermemory_save(content: str, tags: Optional[List[str]] = None, category: str = "neuralai") -> Dict[str, Any]:
    """Persist a fact/decision to Supermemory (long-term, cross-project)."""
    import urllib.request
    import urllib.error
    key = _sm_key()
    if not key:
        return {"success": False, "error": "SUPERMEMORY_API_KEY not set"}
    payload = {"content": content, "tags": tags or [], "category": category}
    req = urllib.request.Request(
        _SUPERMEMORY_URL,
        data=json.dumps(payload).encode(),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return {"success": True, "data": json.loads(resp.read().decode())}
    except urllib.error.HTTPError as e:
        return {"success": False, "error": f"Supermemory HTTP {e.code}: {e.read().decode()[:300]}"}
    except Exception as e:
        return {"success": False, "error": str(e)}


def supermemory_search(query: str, limit: int = 5) -> Dict[str, Any]:
    """Recall from Supermemory long-term memory."""
    import urllib.request
    import urllib.error
    key = _sm_key()
    if not key:
        return {"success": False, "error": "SUPERMEMORY_API_KEY not set"}
    url = f"{_SUPERMEMORY_URL}/search?q={urllib.parse.quote(query)}&limit={limit}"
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {key}"}, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return {"success": True, "data": json.loads(resp.read().decode())}
    except urllib.error.HTTPError as e:
        return {"success": False, "error": f"Supermemory HTTP {e.code}: {e.read().decode()[:300]}"}
    except Exception as e:
        return {"success": False, "error": str(e)}
