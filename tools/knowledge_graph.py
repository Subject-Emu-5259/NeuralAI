# tools/knowledge_graph.py
# Phase 8: Neural Knowledge Graph + Supermemory sync.
# Local graph store: DuckDB (file at /home/.z/workspaces/neuralai_memory/kg.db).
# Supermemory mirror: best-effort push to api.supermemory.ai when SUPERMEMORY_API_KEY is set.

import os
import json
import time
import urllib.request
import urllib.error
from datetime import datetime, timezone

DB_PATH = os.environ.get(
    "NEURAL_KG_PATH",
    "/home/.z/workspaces/neuralai_memory/kg.db",
)
SUPERMEMORY_BASE = "https://api.supermemory.ai"
SUPERMEMORY_KEY = os.environ.get("SUPERMEMORY_API_KEY", "")


def _db():
    import duckdb
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    con = duckdb.connect(DB_PATH)
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS memories (
            id VARCHAR PRIMARY KEY,
            content VARCHAR,
            summary VARCHAR,
            entity VARCHAR,
            rel VARCHAR,
            obj VARCHAR,
            ts DOUBLE,
            src VARCHAR
        )
        """
    )
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS edges (
            id VARCHAR PRIMARY KEY,
            subj VARCHAR,
            rel VARCHAR,
            obj VARCHAR,
            ts DOUBLE
        )
        """
    )
    return con


def _now() -> float:
    return datetime.now(timezone.utc).timestamp()


def _sm_push(content: str, summary: str):
    if not SUPERMEMORY_KEY:
        return
    try:
        payload = json.dumps({"content": content, "summary": summary}).encode()
        req = urllib.request.Request(
            f"{SUPERMEMORY_BASE}/memory",
            data=payload,
            headers={
                "Authorization": f"Bearer {SUPERMEMORY_KEY}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=8) as r:
            return r.status
    except Exception:
        return None


def save_memory(content: str, tags: str = "", relation_to: str = "", relation: str = "", src: str = "chat") -> dict:
    """Persist a memory node + optional edge. Mirrors to Supermemory when key present."""
    import hashlib
    content = (content or "").strip()
    if not content:
        return {"success": False, "error": "empty content"}
    entity = tags or ""
    mid = hashlib.sha1(f"{content}{_now()}".encode()).hexdigest()[:16]
    summary = content[:140]
    con = _db()
    con.execute(
        "INSERT INTO memories VALUES (?,?,?,?,?,?,?,?)",
        [mid, content, summary, entity, relation, relation_to, _now(), src],
    )
    if entity and relation and relation_to:
        eid = hashlib.sha1(f"{entity}{relation}{relation_to}".encode()).hexdigest()[:16]
        con.execute(
            "INSERT OR REPLACE INTO edges VALUES (?,?,?,?,?)",
            [eid, entity, relation, relation_to, _now()],
        )
    con.close()
    _sm_push(content, summary)
    return {"success": True, "id": mid, "remote": bool(_sm_push(content, summary)), "output": f"💾 Saved memory [{mid}]" + (f" → ({entity})-{relation}->({relation_to})" if entity else "")}


def extract_and_store(text: str, src: str = "chat") -> dict:
    """Lightweight extractor: store whole passage as a memory node (entity/rel left to later passes)."""
    return save_memory(text, src=src)


def search_memory(query: str, limit: int = 5) -> dict:
    con = _db()
    rows = con.execute(
        """
        SELECT id, content, entity, rel, obj, ts FROM memories
        ORDER BY ts DESC LIMIT ?
        """,
        [limit],
    ).fetchall()
    con.close()
    hits = [
        {"id": r[0], "content": r[1], "entity": r[2], "rel": r[3], "obj": r[4], "ts": r[5]}
        for r in rows
        if not query or query.lower() in (r[1] or "").lower()
    ]
    return {"success": True, "output": f"🔎 {len(hits)} memories", "data": {"memories": hits}}


def recall(query: str, limit: int = 5, q: str = "") -> dict:
    if not query and q:
        query = q
    res = search_memory(query, limit)
    items = res.get("data", {}).get("memories", [])
    if not items:
        return {"success": True, "output": "🧠 No matching memories yet. Use /remember <text> to teach me."}
    out = "🧠 Recalled:\n" + "\n".join(f"• {m['content']}" for m in items)
    return {"success": True, "output": out, "data": res["data"]}


def get_graph(entity: str = "", node_id: str = "") -> dict:
    if not entity and node_id:
        entity = node_id
    con = _db()
    if entity:
        edges = con.execute(
            "SELECT subj, rel, obj FROM edges WHERE subj=? OR obj=? ORDER BY ts DESC",
            [entity, entity],
        ).fetchall()
    else:
        edges = con.execute("SELECT subj, rel, obj FROM edges ORDER BY ts DESC LIMIT 200").fetchall()
    nodes = con.execute("SELECT DISTINCT entity FROM memories WHERE entity <> ''").fetchall()
    con.close()
    edge_list = [{"subj": e[0], "rel": e[1], "obj": e[2]} for e in edges]
    return {
        "success": True,
        "output": f"🕸️ Graph: {len(nodes)} entities, {len(edge_list)} edges"
        + (f" (focus: {entity})" if entity else ""),
        "data": {"entities": [n[0] for n in nodes], "edges": edge_list},
    }
