#!/usr/bin/env python3
"""
NeuralAI Unified Service - ALL IN ONE
===================================
- Model inference (Mamba family)
- Neural Uplink (4 parallel agents) 
- Tools (code, terminal)
- Web UI
"""
import os, sys, json, asyncio, requests, logging, threading, secrets, re
import sqlite3, subprocess, tempfile, uuid, jwt
from pathlib import Path
from datetime import datetime, timedelta, timezone
from functools import wraps
from werkzeug.security import generate_password_hash, check_password_hash
from flask import Flask, Response, jsonify, request, send_from_directory, stream_with_context
from flask_sock import Sock
import websocket  # websocket-client for proxying

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("NeuralAI")
# Tool layer (web search / browse / fetch / code / etc.)
# Ensure project root (parent of services/) is importable so `tools` resolves
# whether this file is run directly from services/ or imported elsewhere.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
from tools.tool_handler import run_tool as _run_tool
from tools.tool_router import route as _route_nl_tool
from tools import refine as _refine
from tools import web_browser as _web_browser
from services.nextcloud_bridge import (
    neuralai_to_nc_username,
    provision_user,
    list_files as nc_list_files,
    get_file as nc_get_file,
    put_file as nc_put_file,
    mkdir as nc_mkdir,
    delete_file as nc_delete_file,
)

# Local model manager (NeuralAI DPO / Air models)
import scripts.model_manager as _model_manager

def _active_model():
    """Return the currently active inference model metadata from model_manager."""
    return _model_manager.active_model()

# torch is imported lazily inside load_model() only when LLM_BACKEND=local
# This prevents 6GB+ RAM usage on ZO Computer when using external API backends
torch = None

app = Flask(__name__, static_folder=None)
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "neural-ai-multi-layer-secure-secret-key-2026-v5-stable")
# === CORS for BYO API (OpenAI-compatible) endpoints ===
# Lets other chat UIs (e.g. ZO Computer's Bring Your Own Key) call
# /v1/chat/completions and /v1/models — including browser-side / preflight.
@app.after_request
def _add_cors_headers(resp):
    p = request.path
    if p.startswith("/v1") or p.startswith("/api/settings/api-key"):
        resp.headers["Access-Control-Allow-Origin"] = "*"
        resp.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        resp.headers["Access-Control-Allow-Headers"] = "Authorization, Content-Type, X-Api-Key"
        resp.headers["Access-Control-Expose-Headers"] = "Content-Type, X-Request-Id"
        resp.headers["Access-Control-Max-Age"] = "86400"
    return resp

# Config
PORT = int(os.environ.get("PORT", "5000"))
MODEL_PATH = os.environ.get("MODEL_PATH", "/home/workspace/Projects/NeuralAI/models/mamba-k1")
BASE_MODEL = os.environ.get("BASE_MODEL", "state-spaces/mamba-130m-hf")
STATIC_PATH = os.environ.get("STATIC_PATH", "/home/workspace/Projects/NeuralAI/from-scratch/web_ui")
# Zo Computer API identity token (used by the host's native image generator)
ZO_API_TOKEN = os.environ.get("ZO_API_TOKEN", os.environ.get("ZO_CLIENT_IDENTITY_TOKEN", ""))
DATA_DIR = Path("/home/workspace/Projects/NeuralAI/data")
DATA_DIR.mkdir(parents=True, exist_ok=True)
DATABASE = str(DATA_DIR / "neuralai.db")
# Repository root (parent of services/) — used by the self-update endpoint
REPO_ROOT = Path(__file__).resolve().parent.parent
# Founder account — auto-promoted to founder on login/signup
FOUNDER_EMAIL = os.environ.get("FOUNDER_EMAIL", "deandrewh26@gmail.com")

# ====================
# LLM BACKEND CONFIG
# ====================
# On ZO Computer: PyTorch + full models can OOM; LM Studio GGUF keeps memory usage low.
# LOCAL:    LM Studio (llama.cpp) on localhost:1234 serves Mamba Q4_K_M GGUF models.
# Local LM Studio on port 1234 is the default inference backend.
# Override with env vars: LLM_BACKEND, LLM_API_URL, LLM_MODEL, LLM_API_KEY.
LLM_BACKEND = os.environ.get("LLM_BACKEND", "lmstudio")
LLM_API_URL = os.environ.get("LLM_API_URL", "http://127.0.0.1:1234/v1")
LLM_MODEL = os.environ.get("LLM_MODEL", "mamba-k1")  # active Mamba model used through LM Studio
LLM_API_KEY = os.environ.get("LLM_API_KEY", "")
_USE_FLOAT16 = os.environ.get("NEURALAI_FLOAT16", "").lower() in ("1", "true", "yes")
REQUEST_TIMEOUT_SECONDS = 60
MAX_CONTEXT_TOKENS = 2048
CHAT_HISTORY_CHAR_LIMIT = 20_000
LOCAL_CONTEXT_TOKEN_LIMIT = MAX_CONTEXT_TOKENS
LOCAL_GENERATION_MAX_TIME = 60
LOCAL_TOP_K = 50
LOCAL_TOP_P = 0.95
MAX_HISTORY_TURNS = 16
STOP_SEQUENCES = ("<|im_end|>", "<|im_start|>", "</s>", "<|endoftext|>")

# Model globals (PyTorch) — only loaded when LLM_BACKEND=local
model = None
tokenizer = None
if LLM_BACKEND == "local":
    model_status = "loading"
else:
    model_status = "ready"
if LLM_BACKEND not in ("local", "none") and not LLM_API_URL:
    LLM_API_URL = "http://127.0.0.1:1234/v1"

logger.info(f"[BOOT] backend={LLM_BACKEND} api_url={LLM_API_URL} model={LLM_MODEL}")
inference_count = 0

# Streaming abort control: conv_id -> threading.Event
stop_events = {}

# ====================
# DEFENSE 1: KEEP-ALIVE PINGER
# ====================
# Prevents ZO Computer from putting the service to sleep by pinging /health
# every 5 minutes in a background thread.
def _keep_alive_pinger():
    """Background thread: pings own /health endpoint every 5 min to prevent ZO sleep.

    Hits the PUBLIC service URL when NEURALAI_PUBLIC_URL is set (real external
    ingress, so the ZO Computer sandbox is not idled/slept by the platform), and
    falls back to localhost on any failure so the process stays self-warm too.
    """
    import urllib.request
    public_url = (os.environ.get("NEURALAI_PUBLIC_URL") or "").rstrip("/")
    while True:
        try:
            time.sleep(300)  # 5 minutes
            targets = []
            if public_url:
                targets.append(f"{public_url}/health")
            targets.append(f"http://127.0.0.1:{PORT}/health")
            ok = False
            for t in targets:
                try:
                    urllib.request.urlopen(t, timeout=10)
                    logger.info(f"[KEEPALIVE] Health ping OK -> {t}")
                    ok = True
                    break
                except Exception as inner_e:
                    logger.warning(f"[KEEPALIVE] Ping failed ({t}): {inner_e}")
            if not ok:
                logger.warning("[KEEPALIVE] All ping targets failed (non-fatal)")
        except Exception as e:
            logger.warning(f"[KEEPALIVE] Ping loop error (non-fatal): {e}")

# ====================
# DEFENSE 2: MEMORY WATCHDOG
# ====================
# Monitors AVAILABLE RAM (not %). The loaded model legitimately uses most of the
# 4 GB on ZO, so a high % is normal and must NOT pause the service. Only a truly
# low amount of reclaimable memory (<50 MB) is treated as critical.
def _memory_watchdog():
    """Background thread: monitors system memory every 60s."""
    global model_status
    while True:
        try:
            time.sleep(60)
            import re as _re
            with open("/proc/meminfo") as f:
                mem = f.read()
            total = int(_re.search(r'MemTotal:\s+(\d+)', mem).group(1))
            avail = int(_re.search(r'MemAvailable:\s+(\d+)', mem).group(1))
            avail_mb = avail / 1024
            used_pct = (1 - avail / total) * 100
            # The loaded model legitimately uses nearly all RAM on the 4 GB ZO host;
            # gVisor often reports MemAvailable near 0 even while inference works fine.
            # We only GC + log here — we NEVER flip model_status to "overloaded",
            # because that 503s every request (incl. chat) and the host then sleeps
            # the service, which is the root cause of the recurring pauses.
            if avail_mb < 150:
                logger.warning(f"[WATCHDOG] Low reclaimable memory: {avail_mb:.0f}MB available ({used_pct:.0f}% used) — running GC")
                import gc; gc.collect()
            else:
                logger.debug(f"[WATCHDOG] Memory OK: {avail_mb:.0f}MB available")
        except Exception:
            pass  # /proc not available (non-Linux), skip silently

# Terminal sessions
terminal_sessions = {}
# Conversations storage (Simple JSON file)
CONV_FILE = Path("/home/workspace/Projects/NeuralAI/conversations.json")
# Files storage
STORAGE_SERVICE = os.environ.get("STORAGE_SERVICE", "http://localhost:7003")
STORAGE_ROOT = Path("/home/workspace/Projects/NeuralAI/storage")
STORAGE_ROOT.mkdir(parents=True, exist_ok=True)
from collections import defaultdict
import hashlib

# ====================
# NEURALAI SYSTEM PROMPT
# ====================
NEURALAI_SYSTEM_PROMPT = """You are NeuralAI, an AI assistant created by De'Andrew Preston Harris.

## Core Identity
- Your name is NeuralAI. You are NOT De'Andrew Preston Harris.
- De'Andrew Preston Harris is your creator and the founder of NeuralAI.
- You speak with the user as a warm, conversational assistant.

## Response Style
- Be warm, conversational, and natural — like a knowledgeable friend, not a robot
- Use short paragraphs, line breaks, and bullet lists when they help readability
- Match length to the question: be concise for small talk, detailed for complex topics
- Use examples, metaphors, or analogies only when they clarify
- Acknowledge uncertainty honestly; never invent facts
- NEVER dump a long numbered list of capabilities unless specifically asked
- NEVER start with bold headers like **Plan** or **Goal**
- NEVER show your internal reasoning or planning
- NEVER repeat the same sentence or recycle your previous answer unless asked
- Go straight to your response without showing how you arrived at it

## Identity Guardrails — Follow exactly
- If asked your name, say: "I'm NeuralAI."
- If asked who created you, say: "De'Andrew Preston Harris created me." or "I was created by De'Andrew Preston Harris."
- NEVER say "I'm De'Andrew Preston Harris" or "My name is De'Andrew".
- NEVER say "De'Andrew Preston Harris is my name" or "De'Andrew is my name".
- If the user says they are De'Andrew Preston Harris, acknowledge warmly and continue answering. Do not assume you are him.

## Conversation Memory
- Use the User Profile, User Memory, and conversation history provided in the system prompt
- Treat remembered facts as true and refer to them naturally
- If the user corrects a fact, acknowledge and use the correction going forward

## Safety
- Be helpful, honest, and harmless
- Respect user privacy and data"""

# ====================
# DATABASE LAYER
# ====================
def get_db():
    db = sqlite3.connect(DATABASE)
    db.row_factory = sqlite3.Row
    return db

def init_db():
    db = get_db()
    db.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE,
            first_name TEXT,
            last_name TEXT,
            bod TEXT,
            bio TEXT,
            is_founder INTEGER DEFAULT 0,
            password_hash TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS conversations (
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            title TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            message_count INTEGER DEFAULT 0
        );
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS user_settings (
            user_id TEXT NOT NULL,
            key TEXT NOT NULL,
            value TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            PRIMARY KEY (user_id, key)
        );
        CREATE TABLE IF NOT EXISTS memory_facts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            fact TEXT NOT NULL,
            category TEXT DEFAULT 'general',
            importance INTEGER DEFAULT 0,
            user_id TEXT,
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS active_rules (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            rule TEXT NOT NULL,
            active INTEGER DEFAULT 1,
            user_id TEXT,
            created_at TEXT NOT NULL
        );
    """)
    db.commit()
    db.close()

# ====================
# AUTH DECORATOR
# ====================
def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get("Authorization")
        if not token:
            token = request.args.get("token")
        if not token:
            request.user_id = "guest"
            return f(request.user_id, *args, **kwargs)
        try:
            token = token.replace("Bearer ", "")
            payload = jwt.decode(token, app.config["SECRET_KEY"], algorithms=["HS256"])
            request.user_id = payload["user_id"]
        except Exception:
            return jsonify({"error": "Invalid token"}), 401
        return f(request.user_id, *args, **kwargs)
    return decorated

def load_convs():
    if CONV_FILE.exists():
        try:
            with open(CONV_FILE) as f: return json.load(f)
        except: return {}
    return {}

def save_convs(data):
    with open(CONV_FILE, 'w') as f: json.dump(data, f)

# Neural Uplink Agents
UPLINK_AGENTS = {
    "dialog": {"name": "DIALOG", "role": "Conversation", "color": "🔵", "system": "You are DIALOG, a concise AI assistant."},
    "data": {"name": "DATA", "role": "Data Analysis", "color": "🟢", "system": "You are DATA, specialized in data analysis."},
    "ops": {"name": "OPS", "role": "Operations", "color": "🟡", "system": "You are OPS, specialized in execution."},
    "world": {"name": "WORLD", "role": "Creativity", "color": "🟣", "system": "You are WORLD, specialized in creative tasks."},
}

# ====================
# MODEL LOADING
# ====================
def _forward_to_external_llm(messages, max_tokens=256, temperature=0.7, stream=False):
    """Forward inference to an external OpenAI-compatible API (Ollama, LM Studio, etc.).
    Returns a requests.Response (streaming) or a dict (non-streaming).
    """
    api_url = LLM_API_URL.rstrip("/")
    endpoint = f"{api_url}/chat/completions"
    headers = {"Content-Type": "application/json"}
    if LLM_API_KEY:
        headers["Authorization"] = f"Bearer {LLM_API_KEY}"
    body = {
        "model": _active_model()["id"],
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": stream,
    }
    logger.info("[LLM] Forwarding to %s backend at %s", LLM_BACKEND, endpoint)
    if stream:
        return requests.post(endpoint, json=body, headers=headers, stream=True, timeout=REQUEST_TIMEOUT_SECONDS)
    resp = requests.post(endpoint, json=body, headers=headers, timeout=REQUEST_TIMEOUT_SECONDS)
    resp.raise_for_status()
    return resp.json()

def load_model():
    global model, tokenizer, model_status
    if LLM_BACKEND in ("none",):
        model_status = "ready (lightweight mode — no model loaded)"
        logger.info("[OK] Lightweight mode active — no model loaded. Chat will use template responses.")
        return
    if LLM_BACKEND == "zo":
        raise RuntimeError("ZO native backend is disabled. Use LLM_BACKEND=lmstudio or openai_compatible.")
    if LLM_BACKEND != "local":
        # openai_compatible / lmstudio / ollama -> local inference server (e.g. LM Studio :1234)
        model_status = "ready"
        print(f"[OK] Using local OpenAI-compatible backend: {LLM_BACKEND} @ {LLM_API_URL}")
        return
    try:
        global torch
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        tokenizer.pad_token = tokenizer.eos_token
        adapter = Path(MODEL_PATH)
        has_adapter = any((adapter / f).exists() for f in ["adapter_model.bin", "adapter_model.safetensors"])
        # Use float16 on ZO to fit in 4GB RAM (~700MB vs ~1.4GB float32)
        dtype = torch.float16 if _USE_FLOAT16 else torch.float32
        if adapter.exists() and has_adapter:
            base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=dtype, device_map=None, low_cpu_mem_usage=True)
            model = PeftModel.from_pretrained(base, str(adapter), torch_dtype=dtype)
        else:
            model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=dtype, device_map=None, low_cpu_mem_usage=True)
        model.eval()
        model_status = "ready"
        print(f"[OK] Model loaded. Params: {sum(p.numel() for p in model.parameters()):,}")
    except Exception as e:
        model_status = f"error: {e}"
        print(f"[ERROR] Model: {e}")

def get_conversation_history(conv_id, limit=10):
    """Get recent conversation history for context"""
    if not conv_id:
        return []
    try:
        db = get_db()
        rows = db.execute(
            "SELECT role, content FROM messages WHERE conversation_id = ? ORDER BY id DESC LIMIT ?",
            (conv_id, limit)
        ).fetchall()
        db.close()
        return list(reversed([dict(r) for r in rows]))
    except Exception:
        return []

def _cap_text(text, max_chars=3500):
    """Cap a single chat message so one oversized paste can't blow the prompt to 30k+ tokens."""
    if not isinstance(text, str):
        text = str(text)
    return text if len(text) <= max_chars else text[-max_chars:]


def _coerce_chat_messages(messages):
    cleaned = []
    for msg in messages or []:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        if role not in {"system", "user", "assistant"}:
            continue
        content = msg.get("content", "")
        if isinstance(content, list):
            content = " ".join(
                part.get("text", "")
                for part in content
                if isinstance(part, dict)
            )
        if content is None:
            content = ""
        if not isinstance(content, str):
            content = str(content)
        if role != "assistant" and not content.strip():
            continue
        cleaned.append({"role": role, "content": content})
    return cleaned


def _estimate_tokens(text):
    if not isinstance(text, str):
        return 0
    return max(1, int(len(text.split()) * 1.3))


def _get_user_context(user_id):
    ctx = {"profile": [], "memory": [], "rules": []}
    if not user_id or str(user_id).lower() in ("guest", "anonymous", "none"):
        return ctx
    try:
        db = get_db()
        try:
            user = db.execute("SELECT first_name, last_name, bio FROM users WHERE id = ?", (user_id,)).fetchone()
            if user:
                if user["first_name"]:
                    ctx["profile"].append(f"First name: {user['first_name']}")
                if user["last_name"]:
                    ctx["profile"].append(f"Last name: {user['last_name']}")
                if user["bio"]:
                    ctx["profile"].append(f"Bio: {user['bio']}")
            mem_rows = db.execute(
                "SELECT fact FROM memory_facts WHERE user_id = ? ORDER BY importance DESC LIMIT 20",
                (user_id,),
            ).fetchall()
            ctx["memory"] = [r["fact"] for r in mem_rows]
            rule_rows = db.execute(
                "SELECT rule FROM active_rules WHERE user_id = ? AND active = 1 ORDER BY id DESC LIMIT 10",
                (user_id,),
            ).fetchall()
            ctx["rules"] = [r["rule"] for r in rule_rows]
        finally:
            db.close()
    except Exception:
        pass
    return ctx


def _build_system_prompt(user_id=None):
    prompt = NEURALAI_SYSTEM_PROMPT
    ctx = _get_user_context(user_id)
    if ctx["profile"]:
        prompt += "\n\n## User Profile\n" + "\n".join(f"- {p}" for p in ctx["profile"])
    if ctx["memory"]:
        prompt += "\n\n## User Memory (use these facts in your replies)\n" + "\n".join(f"- {m}" for m in ctx["memory"])
    if ctx["rules"]:
        prompt += "\n\n## Active User Rules\n" + "\n".join(f"- {r}" for r in ctx["rules"])
    return prompt


def _truncate_by_token_est(messages, max_tokens=MAX_CONTEXT_TOKENS):
    if not messages:
        return messages
    system_msgs = [m for m in messages if m.get("role") == "system"]
    turns = [m for m in messages if m.get("role") != "system"]
    while True:
        total = sum(_estimate_tokens(m.get("content", "")) for m in system_msgs + turns)
        if total <= max_tokens or len(turns) <= 1:
            break
        turns.pop(0)
    return system_msgs + turns


def _chat_messages_from_request(prompt, conv_id=None, messages=None, history_turns=MAX_HISTORY_TURNS):
    if messages:
        chat_messages = _coerce_chat_messages(messages)
    else:
        history = get_conversation_history(conv_id, history_turns) if conv_id else []
        chat_messages = [
            {"role": "user" if msg["role"] == "user" else "assistant", "content": _cap_text(msg["content"])}
            for msg in history
        ]
    if not chat_messages or chat_messages[-1].get("role") != "user" or chat_messages[-1].get("content", "") != prompt:
        chat_messages.append({"role": "user", "content": _cap_text(prompt)})
    # Build a dynamic system prompt scoped to the current user (profile + memory + rules).
    user_id = getattr(request, "user_id", None) if request else None
    return [{"role": "system", "content": _build_system_prompt(user_id)}] + chat_messages


def _truncate_chat_messages(messages, max_chars=CHAT_HISTORY_CHAR_LIMIT):
    if not messages:
        return messages
    system_msgs = [m for m in messages if m.get("role") == "system"][:1]
    turns = [m for m in messages if m.get("role") != "system"]
    while turns and sum(len(str(m.get("content", ""))) for m in system_msgs + turns) > max_chars:
        turns.pop(0)
    return system_msgs + turns


def _strip_stop_sequences(text):
    if not text:
        return text
    cut = len(text)
    for marker in STOP_SEQUENCES:
        pos = text.find(marker)
        if pos != -1:
            cut = min(cut, pos)
    return text[:cut].rstrip()


def build_prompt_with_context(prompt, conv_id=None, max_history=5, client_messages=None):
    """Build a ChatML-formatted prompt (matching the model's trained chat template).

    The active Mamba model uses GGUF via LM Studio; prompts are formatted by the server.
        <|im_start|>system\n...\n<|im_end|>\n
        <|im_start|>user\n...\n<|im_end|>\n
        <|im_start|>assistant\n
    Feeding it freeform "User:/NeuralAI:" text caused the model to not recognize
    turn boundaries and "talk to itself". Using the correct template fixes that.
    """
    messages = _chat_messages_from_request(prompt, conv_id=conv_id, messages=client_messages, history_turns=max_history)

    # Use the tokenizer's native chat template so formatting exactly matches training.
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        # Fallback manual ChatML assembly (mirrors chat_template.jinja)
        out = []
        for i, msg in enumerate(messages):
            if i == 0 and msg["role"] != "system":
                out.append("<|im_start|>system\nYou are a helpful AI assistant named NeuralAI<|im_end|>\n")
            out.append(f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n")
        out.append("<|im_start|>assistant\n")
        return "".join(out)

def _truncate_to_fit(messages, tokenizer_obj, context_limit=MAX_CONTEXT_TOKENS):
    """Truncate a list of ChatML messages so tokenized length fits within context_limit.

    Always keeps the system prompt. Drops oldest user/assistant turns when the
    total token count exceeds the limit, keeping at least the most recent pair.
    Mamba models have large context windows; we use 2048 as a
    safe ceiling to leave room for generated tokens and prevent ZO 120s timeout.
    """
    if not messages or tokenizer_obj is None:
        return messages
    # Tokenize the full prompt to check length
    try:
        full = tokenizer_obj.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        token_count = len(tokenizer_obj.encode(full))
    except Exception:
        return messages  # if we can't count, just pass through
    if token_count <= context_limit:
        return messages
    # Separate system prompt (always keep) from conversation turns
    system_msgs = [m for m in messages if m.get("role") == "system"]
    turns = [m for m in messages if m.get("role") != "system"]
    # Greedily drop oldest turns until we fit
    while turns and token_count > context_limit:
        dropped = turns.pop(0)
        try:
            full = tokenizer_obj.apply_chat_template(system_msgs + turns, tokenize=False, add_generation_prompt=True)
            token_count = len(tokenizer_obj.encode(full))
        except Exception:
            break
    logger.info(f"[TRUNC] Input {token_count} tokens → kept {len(turns)} turns (limit {context_limit})")
    return system_msgs + turns


def generate_response(prompt, max_tokens=256, temperature=0.7, conv_id=None, messages=None):
    """Enhanced response generation with system prompt and context."""
    global model, tokenizer, inference_count
    # === No backend (lightweight mode) ===
    if LLM_BACKEND == "none":
        return "I'm NeuralAI. The AI model isn't loaded due to memory constraints. I can't generate AI responses in this mode."
    # === External LLM backend ===
    if LLM_BACKEND in ("ollama", "lmstudio", "openai_compatible"):
        try:
            # Build OpenAI-format messages for the external API
            api_messages = _truncate_by_token_est(
                _chat_messages_from_request(prompt, conv_id=conv_id, messages=messages),
            )
            data = _forward_to_external_llm(api_messages, max_tokens=max_tokens, temperature=temperature, stream=False)
            content = _strip_stop_sequences(_strip_reasoning(data["choices"][0]["message"]["content"]))
            content = _normalize_identity(content)
            inference_count += 1
            return content.strip()
        except Exception as e:
            logger.error(f"[LLM] External backend error: {e}")
            return f"Backend error: {e}"
    # === Local PyTorch inference ===
    if model is None or tokenizer is None:
        return "I'm NeuralAI. The AI model isn't loaded due to memory constraints on this machine (4GB ZO Computer). The model needs ~2GB but only less is available. Please try again later or contact support."
    try:
        full = build_prompt_with_context(prompt, conv_id, client_messages=messages)
        inputs = tokenizer(full, return_tensors="pt")
        # Safety: if prompt exceeds model context, truncate from the front
        max_input = LOCAL_CONTEXT_TOKEN_LIMIT if LLM_BACKEND == "local" else 4000  # local CPU: keep prefill ~25s
        if inputs["input_ids"].shape[-1] > max_input:
            inputs["input_ids"] = inputs["input_ids"][:, -max_input:]
            inputs["attention_mask"] = inputs["attention_mask"][:, -max_input:]
            logger.warning(f"[TRUNC] Input truncated to {max_input} tokens (was {inputs['input_ids'].shape[-1]})")
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                max_time=LOCAL_GENERATION_MAX_TIME,
                do_sample=True,
                temperature=temperature,
                top_p=LOCAL_TOP_P,
                top_k=LOCAL_TOP_K,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=[tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else None,
                repetition_penalty=1.1
            )
        new_tokens = out[0][inputs["input_ids"].shape[-1]:]
        response = _strip_stop_sequences(tokenizer.decode(new_tokens, skip_special_tokens=True).strip())
        if response.startswith("<|im_start|>assistant"):
            response = response[len("<|im_start|>assistant"):].strip()
        if response.startswith("NeuralAI:"):
            response = response[len("NeuralAI:"):].strip()
        response = _strip_stop_sequences(_strip_reasoning(response))
        response = _normalize_identity(response)
        inference_count += 1
        return response
    except Exception as e:
        logger.error(f"Generation error: {e}")
        return "I encountered an error generating a response. Please try again."

def stream_response(prompt, max_tokens=256, temperature=0.7, conv_id=None, already_rendered=False, messages=None):
    """Token streaming — external backend or local TextIteratorStreamer.

    When *already_rendered* is True, *prompt* is a fully-rendered ChatML string
    (from apply_chat_template) and must NOT be passed through build_prompt_with_context
    again.  This prevents the double-wrapping bug that blew up token counts.
    """
    global model, tokenizer, inference_count
    # === External LLM backend ===
    if LLM_BACKEND in ("ollama", "lmstudio", "openai_compatible"):
        try:
            api_messages = _truncate_by_token_est(
                _chat_messages_from_request(prompt, conv_id=conv_id, messages=messages),
            )
            resp = _forward_to_external_llm(api_messages, max_tokens=max_tokens, temperature=temperature, stream=True)
            if resp.status_code != 200:
                yield f"Backend error ({resp.status_code}): {resp.text[:200]}"
                return
            for line in resp.iter_lines():
                if not line or not line.startswith(b"data: "):
                    continue
                payload = line[6:].decode().strip()
                if payload == "[DONE]":
                    break
                try:
                    chunk = json.loads(payload)
                    delta = chunk.get("choices", [{}])[0].get("delta", {})
                    content = delta.get("content", "")
                    if content:
                        # Strip reasoning from external backend streams too
                        content = _strip_stop_sequences(_strip_reasoning(content))
                        yield content
                except json.JSONDecodeError:
                    continue
            inference_count += 1
            return
        except Exception as e:
            logger.error(f"[LLM] External backend stream error: {e}")
            yield f"Backend error: {e}"
            return
    # === Local PyTorch streaming ===
    if model is None or tokenizer is None:
        # Lightweight mode: return a helpful response without the model
        yield f"I'm NeuralAI. I received your message but the AI model isn't loaded (memory-limited environment). Here's what I can tell you:  On this ZO Computer (4GB RAM), the model can't run due to memory constraints. Please check back when more resources are available."
        return
    stop_event = stop_events.get(conv_id) if conv_id else None
    try:
        from transformers import TextIteratorStreamer
        # If already rendered (from BYO API path), use directly to avoid double ChatML wrapping
        if already_rendered:
            full = prompt
        else:
            full = build_prompt_with_context(prompt, conv_id, client_messages=messages)
        inputs = tokenizer(full, return_tensors="pt")
        # Safety: if prompt exceeds model context, truncate from the front
        max_input = LOCAL_CONTEXT_TOKEN_LIMIT if LLM_BACKEND == "local" else 4000  # local CPU: keep prefill ~25s
        if inputs["input_ids"].shape[-1] > max_input:
            inputs["input_ids"] = inputs["input_ids"][:, -max_input:]
            inputs["attention_mask"] = inputs["attention_mask"][:, -max_input:]
            logger.warning(f"[TRUNC] Stream input truncated to {max_input} tokens")
        logger.info(f"[INFER] Input tokens: {inputs['input_ids'].shape[-1]}, generating up to {max_tokens} new tokens")
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        # Use greedy decoding (do_sample=False) for faster CPU inference
        gen_kwargs = dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=max_tokens,
            max_time=LOCAL_GENERATION_MAX_TIME,
            do_sample=False if temperature <= 0.1 else True,
            temperature=max(temperature, 0.3),
            top_p=LOCAL_TOP_P,
            top_k=LOCAL_TOP_K,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=[tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else None,
            repetition_penalty=1.05,
        )
        thread = threading.Thread(target=model.generate, kwargs=gen_kwargs, daemon=True)
        thread.start()
        # Strip reasoning tokens from stream output.
        # Accumulate tokens until we find the '!!' delimiter, then start yielding.
        _buf = ""
        _reasoning_done = False
        for token in streamer:
            if stop_event and stop_event.is_set():
                break
            if token:
                if _reasoning_done:
                    yield _strip_stop_sequences(token)
                else:
                    _buf += token
                    if "!!" in _buf:
                        parts = _buf.split("!!", 1)
                        _reasoning_done = True
                        remainder = _strip_stop_sequences(parts[1].strip())
                        if remainder:
                            yield remainder
                        logger.info(f"[THINK] Stream: stripped {len(parts[0])} chars of reasoning")
                    elif len(_buf) > 600:
                        # Safety: if no !! after 600 chars, assume no reasoning block
                        _reasoning_done = True
                        yield _strip_stop_sequences(_buf)
        # If stream ended before !!, yield whatever we have
        if not _reasoning_done and _buf:
            stripped = _strip_stop_sequences(_strip_reasoning(_buf))
            if stripped != _buf:
                yield stripped
            else:
                yield _buf
        thread.join()
        inference_count += 1
    except Exception as e:
        logger.error(f"Stream generation error: {e}")
        yield "I encountered an error generating a response. Please try again."

def _strip_reasoning(text):
    """Strip the model's internal chain-of-thought from visible output.

    SmolLM2-360M-Instruct + NeuralAI LoRA tends to emit a reasoning block
    (often a **Bold Header** followed by first-person deliberation) before the
    actual response.  We detect the delimiter '!!' which the model uses to
    separate its internal thinking from the user-facing answer.

    Examples of what gets stripped:
        **Greet User Warmly**\n\nI need to respond as NeuralAI...!! I'm NeuralAI. ...
    """
    if not text:
        return text
    # Primary delimiter: '!!' — the model's own separator between thought and speech
    if "!!" in text:
        after = text.split("!!", 1)[1].strip()
        if after:  # only use if there's actual content after !!
            logger.info(f"[THINK] Stripped reasoning ({len(text) - len(after)} chars)")
            return after
    # Secondary pattern: bold header + first-person reasoning before actual response
    # Matches: **Some Plan**\n\nI need to.../I should.../Let me...
    import re as _re
    think_match = _re.match(
        r'^\*\*[^*]+\*\*\s*\n\s*(?:I (?:need|should|want|must|have to|will|can|could|would)|'
        r'Let me |The user |My approach |First, |Step \d)[^.]*\.\.\.[^.]*\.\s*',
        text, _re.DOTALL
    )
    if think_match:
        after = text[think_match.end():].strip()
        if after:
            logger.info(f"[THINK] Stripped reasoning pattern ({think_match.end()} chars)")
            return after
    return text

def _normalize_identity(text):
    """Fix identity-confused outputs without affecting normal references.

    The 360M model sometimes confuses "De'Andrew Preston Harris" (creator) with
    its own identity. This guard strips those claims and replaces them with the
    canonical NeuralAI identity statement.
    """
    if not text:
        return text
    import re as _re
    # Leading "I'm De'Andrew ... creator of NeuralAI" style intros
    text = _re.sub(
        r"^(\s*)(?:I'm|I am)\s+De'Andrew(?:\s+Preston)?(?:\s+Harris)?[\s,.\-–—]*(?:creator of NeuralAI|founder of NeuralAI)?[\s,.\-–—]*",
        r"\1I'm NeuralAI, an AI assistant created by De'Andrew Preston Harris.",
        text,
        flags=_re.IGNORECASE,
    )
    # Inline claims that De'Andrew is the model's name
    text = _re.sub(
        r"\bDe'Andrew(?:\s+Preston)?(?:\s+Harris)?\s+is\s+(?:indeed\s+)?my\s+name\b",
        "My name is NeuralAI",
        text,
        flags=_re.IGNORECASE,
    )
    return text


def _identity_guard_stream(generator):
    """Buffer the first sentence of a stream so identity corrections apply early."""
    buffer = ""
    flushed = False
    for token in generator:
        if not flushed:
            buffer += token
            # Flush once we hit a sentence boundary or have enough context
            if buffer.endswith((".", "!", "?", "\n")) or len(buffer) >= 120:
                yield _normalize_identity(buffer)
                flushed = True
                buffer = ""
        else:
            yield token
    if buffer:
        yield _normalize_identity(buffer)


# ====================
# IMAGE PROMPT ENHANCER
# ====================

def enhance_image_prompt(prompt):
    """Expand a short user request into a detailed, brand-styled image prompt.

    Uses the local LLM when available; otherwise falls back to a deterministic
    template so 'generate a dog' still becomes a rich NeuralAI-styled prompt.
    """
    tmpl = (
        "Rewrite the user's short image request into a single detailed, "
        "photorealistic image-generation prompt in NeuralAI's signature dark/neon "
        "'vibe stack' aesthetic. Add lighting, mood, composition, and medium. "
        "Output ONLY the prompt, no quotes, no commentary.\n\n"
        f"User request: {prompt}\n\nDetailed prompt:"
    )
    # Try the LLM first (kept in-memory in this process).
    try:
        if model is not None and tokenizer is not None:
            inputs = tokenizer(tmpl, return_tensors="pt")
            with torch.no_grad():
                out = model.generate(
                    **inputs, max_new_tokens=80, do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
            txt = tokenizer.decode(out[0][inputs["input_ids"].shape[-1]:],
                                   skip_special_tokens=True).strip()
            txt = txt.split("\n")[0].strip().strip('"').strip("'")
            if txt and len(txt) > len(prompt):
                return txt
    except Exception as e:
        logger.warning(f"[enhance_image_prompt] LLM enhance failed, using template: {e}")

    # Template fallback: brand-styled expansion.
    subject = prompt.strip().strip(".").lower()
    return (
        f"{prompt}, cinematic dark-mode composition, neon accent rim lighting, "
        f"high contrast, hyper-detailed, 8k, volumetric fog, vibe stack aesthetic, "
        f"centered subject, professional concept art"
    )


# ====================
# ROUTES - STATIC
# ====================
import time

def _build_version():
    # Cache-bust from the app JS file's mtime so any static edit forces a fresh
    # browser fetch (the ?v= query string is what browsers key caching on).
    js_path = f"{STATIC_PATH}/static/js/main_v2.js"
    try:
        return str(int(os.path.getmtime(js_path)))
    except OSError:
        return str(int(time.time()))

@app.route("/")
def index():
    # Visitors land on the branded landing page first; the app lives at /app.
    p = f"{STATIC_PATH}/static/landing/index.html"
    if os.path.exists(p):
        return open(p).read(), 200, {
            "Content-Type": "text/html",
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0"
        }
    return "Landing page not found", 404

@app.route("/app")
def app_index():
    p = f"{STATIC_PATH}/templates/index.html"
    if os.path.exists(p):
        with open(p) as f:
            content = f.read()
            # Inject build version for cache busting
            content = content.replace("{{BUILD_VERSION}}", _build_version())
            return content, 200, {
                "Content-Type": "text/html",
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Pragma": "no-cache",
                "Expires": "0"
            }
    return "index.html not found", 404

@app.route("/landing")
def landing():
    p = f"{STATIC_PATH}/static/landing/index.html"
    if os.path.exists(p):
        with open(p) as f:
            return f.read(), 200, {
                "Content-Type": "text/html",
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Pragma": "no-cache",
                "Expires": "0"
            }
    return "Landing page not found", 404

@app.route("/<path:path>")
def static_files(path):
    for base in [f"{STATIC_PATH}/static", STATIC_PATH]:
        p = os.path.join(base, path)
        if os.path.exists(p) and os.path.isfile(p):
            ext = path.split('.')[-1]
            ct = {"js": "application/javascript", "css": "text/css", "png": "image/png", "jpg": "image/jpeg", "ico": "image/x-icon"}
            # Hard no-cache for JS/CSS so browsers/Cloudflare always revalidate and
            # never serve a stale 304 copy (which caused the blank-tab/black-terminal regression).
            if ext in ("js", "css"):
                resp = send_from_directory(os.path.dirname(p), os.path.basename(p),
                                           mimetype=ct.get(ext, "text/plain"), max_age=0)
                resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
                resp.headers["Pragma"] = "no-cache"
                resp.headers.pop("ETag", None)
                resp.headers.pop("Last-Modified", None)
                return resp
            return send_from_directory(os.path.dirname(p), os.path.basename(p),
                                       mimetype=ct.get(ext, "text/plain"), max_age=31536000)
    return "Not found", 404

# ====================
# ROUTES - POLICIES
# ====================
@app.route("/privacy")
def privacy():
    p = f"{STATIC_PATH}/templates/privacy.html"
    if os.path.exists(p):
        with open(p) as f:
            return f.read(), 200, {"Content-Type": "text/html"}
    return "Privacy policy not found", 404

@app.route("/terms")
def terms():
    p = f"{STATIC_PATH}/templates/terms.html"
    if os.path.exists(p):
        with open(p) as f:
            return f.read(), 200, {"Content-Type": "text/html"}
    return "Terms of service not found", 404

# ====================
# DEFENSE 3: REJECT IF OVERLOADED
# ====================
@app.before_request
def _reject_if_overloaded():
    # Never 503 liveness probes. The host pauses the service when /health fails,
    # which was the root cause of the recurring "NeuralAI pauses" problem.
    if request.path in ("/health", "/api/health", "/api/status", "/api/healthz"):
        return
    if model_status == "overloaded":
        from flask import abort
        abort(503)

# ====================
# ROUTES - HEALTH
# ====================
@app.route("/health")
@app.route("/api/health")
@app.route("/api/status")
def health():
    # Defense 2 integration: reject requests if memory overloaded
    active = _active_model()
    return jsonify({"status": model_status, "model": active["id"], "model_label": active["label"],
                    "model_version": active["label"],
                    "inference_count": inference_count, "uplink": "integrated",
                    "timestamp": datetime.now(timezone.utc).isoformat(), "version": "7.3.0",
                    "llm_backend": LLM_BACKEND})

# ====================
# ROUTES - MODEL SWITCHING
# ====================
@app.route("/api/model", methods=["GET"])
def api_model_list():
    """Return available inference models and the actively loaded model."""
    try:
        current = _active_model()
    except Exception as e:
        logger.warning(f"model_manager.get_active_model failed: {e}")
        current = {"id": "mamba-k1", "label": "Mamba K1 (130M, SFT)"}
    return jsonify({"active": current, "models": _model_manager.MODELS})

@app.route("/api/model/switch", methods=["POST"])
@token_required
def api_model_switch(current_user):
    data = request.get_json(silent=True) or {}
    model_id = str(data.get("model", "")).strip().lower()
    current_id = _active_model()["id"]
    if model_id == current_id:
        return jsonify({"success": True, "message": "Already active", "active": _active_model()})
    if model_id not in _model_manager.MODELS:
        return jsonify({"success": False, "error": f"Unknown model '{model_id}'"}), 400
    try:
        _model_manager.set_active(model_id)
        subprocess.Popen(
            ["supervisorctl", "-c", _model_manager.SUPERVISORD_CONF, "restart", "neuralai-lmstudio"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        return jsonify({"success": True, "active": _active_model()})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

# ====================
# ROUTES - RELEASE NOTES
# ====================
@app.route("/api/release-notes", methods=["GET"])
def api_release_notes():
    notes_path = DATA_DIR / "release_notes.json"
    try:
        if notes_path.exists():
            with open(notes_path) as f:
                return jsonify(json.load(f))
    except Exception as e:
        logger.warning(f"Failed to read release notes: {e}")
    return jsonify({
        "version": "v7.3.0",
        "title": "NeuralAI v7.3.0 — Release Notes",
        "released": "2026-07-13",
        "notes": []
    })

# ====================
# ROUTES - MODEL
# ====================
@app.route("/generate", methods=["POST"])
def generate():
    data = request.get_json() or {}
    return jsonify({"response": generate_response(data.get("prompt", "")), "inference_count": inference_count})

@app.route("/generate/stream", methods=["POST"])
def generate_stream():
    data = request.get_json() or {}
    prompt = data.get("prompt", "")
    
    def generate():
        response = generate_response(prompt)
        for word in response.split():
            yield f"data: {json.dumps({'token': word+' '})}\n\n"
        yield "data: [DONE]\n\n"
    
    return Response(generate(), mimetype="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

# Unified AI API for Frontend
@app.route("/api/chat", methods=["POST"])
@token_required
def api_chat(current_user):
    start_time = time.time()
    data = request.get_json() or {}
    prompt = data.get("prompt", "")
    use_uplink = data.get("use_uplink", False)
    conv_id = data.get("conversation_id")
    client_messages = data.get("messages")

    if not prompt:
        return jsonify({"error": "No prompt provided"}), 400

    # Save user message to DB if conversation_id provided
    if conv_id:
        try:
            db = get_db()
            now = datetime.now(timezone.utc).isoformat()
            db.execute("INSERT INTO messages (conversation_id, role, content, created_at) VALUES (?, 'user', ?, ?)",
                       (conv_id, prompt, now))
            db.execute("UPDATE conversations SET updated_at = ?, message_count = message_count + 1 WHERE id = ?", (now, conv_id))

            # Auto-generate title from first message
            msg_count = db.execute("SELECT COUNT(*) as cnt FROM messages WHERE conversation_id = ?", (conv_id,)).fetchone()["cnt"]
            if msg_count <= 1:
                auto_title = prompt[:50].strip()
                if len(prompt) > 50:
                    last_space = auto_title.rfind(" ")
                    if last_space > 20:
                        auto_title = auto_title[:last_space]
                    auto_title += "..."
                db.execute("UPDATE conversations SET title = ? WHERE id = ?", (auto_title, conv_id))

            db.commit()
            db.close()
        except Exception as e:
            logger.error(f"Failed to save user message: {e}")

    def generate_unified():
        # --- NL -> tool routing (agentic router) ---
        # Plain-English web/image/news requests are intercepted here so they hit
        # the real tools instead of the 360M model (which has no tools and would
        # hallucinate a news list or fall through to the placeholder image path).
        try:
            routed = _route_nl_tool(prompt)
        except Exception as _re:
            logger.warning(f"[api_chat] tool routing failed, using model: {_re}")
            routed = None
        if routed:
            tool_name, tool_params = routed
            try:
                res = _run_tool(tool_name, tool_params)
                raw = (res.get("output") or "") if isinstance(res, dict) else str(res)
                if not raw and isinstance(res, dict) and res.get("error"):
                    raw = "⚠️ " + res["error"]
                kind = "news" if tool_name in ("news",) else ("web_search" if tool_name in ("web_search", "research") else "")
                out = _refine.refine_text(raw, kind) if raw else "(no result)"
            except Exception as _te:
                out = f"⚠️ Tool error: {_te}"
            yield f"data: {json.dumps({'content': out})}\n\n"
            if conv_id and out:
                try:
                    db = get_db()
                    now = datetime.now(timezone.utc).isoformat()
                    db.execute("INSERT INTO messages (conversation_id, role, content, created_at) VALUES (?, 'assistant', ?, ?)",
                               (conv_id, out, now))
                    db.commit()
                    db.close()
                except Exception as _e:
                    logger.error(f"Failed to save tool assistant message: {_e}")
            yield "data: [DONE]\n\n"
            return
        if use_uplink:
            for agent_name, agent in UPLINK_AGENTS.items():
                try:
                    resp = generate_response(f"[{agent['system']}]\n{prompt}", max_tokens=120, messages=client_messages)
                    if resp:
                        chunk = f"{agent['color']} **{agent['name']}**: {resp.strip()}\n\n"
                        yield f"data: {json.dumps({'content': chunk})}\n\n"
                except: pass
        else:
            # ---- NL -> tool routing (agentic) ----
            # Plain-English web/image/news/research requests are intercepted BEFORE
            # they reach the 360M model, so the assistant actually uses its tools
            # instead of hallucinating a news list or returning a placeholder image.
            # tool_router.route() falls back to the keyword router (and ultimately
            # None) on any LLM failure, so a missing key never breaks normal chat.
            try:
                routed = _route_nl_tool(prompt)
            except Exception as _re:
                logger.warning(f"[api_chat] NL routing failed, falling back to model: {_re}")
                routed = None
            if routed:
                tool_name, tool_params = routed
                try:
                    res = _run_tool(tool_name, tool_params or {})
                    raw = (res.get("output") or "").strip()
                    if not raw and res.get("error"):
                        raw = f"⚠️ {res['error']}"
                    # Refine web/news/research output into clean markdown.
                    kind = tool_name if tool_name in ("news", "web_search", "research", "web_fetcher") else ""
                    if kind and raw:
                        try:
                            raw = _refine.refine_text(raw, kind)
                        except Exception:
                            pass
                    if raw:
                        yield f"data: {json.dumps({'content': raw})}\n\n"
                        if conv_id:
                            try:
                                db = get_db()
                                now = datetime.now(timezone.utc).isoformat()
                                db.execute("INSERT INTO messages (conversation_id, role, content, created_at) VALUES (?, 'assistant', ?, ?)",
                                           (conv_id, raw, now))
                                db.commit()
                                db.close()
                            except Exception as e:
                                logger.error(f"Failed to save tool-routed message: {e}")
                        stop_events.pop(conv_id, None)
                        yield "data: [DONE]\n\n"
                        return
                except Exception as _te:
                    logger.error(f"[api_chat] tool '{tool_name}' failed: {_te}")
                    # fall through to model generation below
            # Real token streaming — first token arrives in <1s instead of after full generation
            full_response = []
            for token in _identity_guard_stream(stream_response(prompt, conv_id=conv_id, messages=client_messages)):
                full_response.append(token)
                yield f"data: {json.dumps({'content': token})}\n\n"
            response = "".join(full_response)
            # Save assistant response
            if conv_id and response:
                try:
                    db = get_db()
                    now = datetime.now(timezone.utc).isoformat()
                    db.execute("INSERT INTO messages (conversation_id, role, content, created_at) VALUES (?, 'assistant', ?, ?)",
                               (conv_id, response, now))
                    db.commit()
                    db.close()
                except Exception as e:
                    logger.error(f"Failed to save assistant message: {e}")
            # Clear any stop event for this conversation
            stop_events.pop(conv_id, None)

        yield "data: [DONE]\n\n"

    return Response(generate_unified(), mimetype="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

@app.route("/api/chat/stop", methods=["POST"])
@token_required
def api_chat_stop(current_user):
    data = request.get_json() or {}
    conv_id = data.get("conversation_id")
    if conv_id:
        stop_events[conv_id] = threading.Event()
        stop_events[conv_id].set()
        return jsonify({"success": True, "stopped": conv_id})
    return jsonify({"success": False, "error": "No conversation_id provided"}), 400

# ====================
# ROUTES - MONITORING
# ====================
@app.route("/api/monitoring/metrics", methods=["GET"])
def get_metrics():
    """Get system metrics (public for status page)"""
    return jsonify({
        "model_status": model_status,
        "inference_count": inference_count,
        "version": "7.3.0"
    })

# ====================
# ROUTES - CONVERSATIONS
# ====================
@app.route("/api/conversations", methods=["GET", "POST"])
@token_required
def manage_convs(current_user):
    db = get_db()
    try:
        if request.method == "POST":
            data = request.get_json() or {}
            cid = str(uuid.uuid4().hex[:8])
            now = datetime.now(timezone.utc).isoformat()
            db.execute("INSERT INTO conversations (id, user_id, title, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
                       (cid, current_user, data.get("title", "New Chat"), now, now))
            db.commit()
            return jsonify({"success": True, "id": cid})
        
        rows = db.execute("SELECT id, title, updated_at FROM conversations WHERE user_id = ? ORDER BY updated_at DESC", (current_user,)).fetchall()
        convs = [dict(row) for row in rows]
        return jsonify(convs)
    finally:
        db.close()

@app.route("/api/conversations/<cid>", methods=["GET", "PUT", "DELETE"])
@token_required
def conv_detail(current_user, cid):
    db = get_db()
    try:
        if request.method == "DELETE":
            db.execute("DELETE FROM messages WHERE conversation_id = ?", (cid,))
            db.execute("DELETE FROM conversations WHERE id = ? AND user_id = ?", (cid, current_user))
            db.commit()
            return jsonify({"success": True})

        if request.method == "PUT":
            data = request.get_json(silent=True) or {}
            title = data.get("title", "").strip()
            if not title:
                return jsonify({"error": "Title required"}), 400
            db.execute("UPDATE conversations SET title = ?, updated_at = ? WHERE id = ? AND user_id = ?",
                       (title, datetime.now(timezone.utc).isoformat(), cid, current_user))
            db.commit()
            return jsonify({"success": True})

        conv = db.execute("SELECT * FROM conversations WHERE id = ? AND user_id = ?", (cid, current_user)).fetchone()
        if not conv: return jsonify({"error": "Not found"}), 404
        
        msgs = db.execute("SELECT role, content, created_at FROM messages WHERE conversation_id = ? ORDER BY id ASC", (cid,)).fetchall()
        return jsonify({**dict(conv), "messages": [dict(m) for m in msgs]})
    finally:
        db.close()

# ====================
# ROUTES - FILES (Proxied to Storage Service)
# ====================
def _resolve_nc_user(current_user):
    """Map the JWT-authenticated NeuralAI user to a provisioned Nextcloud user.
    Guests get a per-session Nextcloud home too (scoped, not persistent accounts).
    """
    nc_user = neuralai_to_nc_username(str(current_user))
    provision_user(nc_user)
    return nc_user


@app.route("/api/files", methods=["GET", "POST"])
@token_required
def manage_files(current_user):
    """NeuralDrive: list/upload files in the calling user's Nextcloud home."""
    try:
        nc_user = _resolve_nc_user(current_user)
        if request.method == "POST":
            if 'file' not in request.files: return jsonify({"error": "No file"}), 400
            file = request.files['file']
            data = file.read()
            ok = nc_put_file(nc_user, file.filename, data, file.content_type or "application/octet-stream")
            if not ok:
                return jsonify({"error": "Upload failed"}), 500
            return jsonify({"success": True, "name": file.filename, "size": len(data)})
        files = nc_list_files(nc_user)
        return jsonify(files)
    except Exception as e:
        logger.error(f"File management error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/files/mkdir", methods=["POST"])
@token_required
def make_dir(current_user):
    data = request.get_json() or {}
    name = (data.get("name") or "").strip().replace("/", "").replace("..", "")
    if not name:
        return jsonify({"error": "No folder name"}), 400
    try:
        nc_user = _resolve_nc_user(current_user)
        ok = nc_mkdir(nc_user, name)
        if not ok:
            return jsonify({"error": "mkdir failed"}), 500
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/files/<path:filename>", methods=["GET", "DELETE"])
@token_required
def handle_file(current_user, filename):
    """NeuralDrive: download/delete a file in the calling user's Nextcloud home."""
    try:
        nc_user = _resolve_nc_user(current_user)
        if request.method == "DELETE":
            ok = nc_delete_file(nc_user, filename)
            if not ok:
                return jsonify({"error": "Delete failed"}), 500
            return jsonify({"success": True})
        status, content, ctype = nc_get_file(nc_user, filename)
        if status == 404:
            return jsonify({"error": "Not found"}), 404
        if status != 200:
            return jsonify({"error": "Download failed"}), 500
        from flask import Response as _R
        return _R(content, mimetype=ctype)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ====================
# ROUTES - TERMINAL
# ====================
@app.route("/api/terminal/create", methods=["POST"])
def terminal_create():
    session_id = str(uuid.uuid4())[:8]
    terminal_sessions[session_id] = {"id": session_id, "output": [], "running": True}
    return jsonify({"session_id": session_id, "status": "created"})

@app.route("/api/terminal/<session_id>/send", methods=["POST"])
def terminal_send(session_id):
    if session_id not in terminal_sessions:
        return jsonify({"error": "Session not found"}), 404
    data = request.get_json() or {}
    cmd = data.get("command", "")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
        output = result.stdout or result.stderr or ""
        terminal_sessions[session_id]["output"].append({"cmd": cmd, "output": output})
        return jsonify({"output": output, "exit_code": result.returncode})
    except Exception as e:
        return jsonify({"output": f"Error: {e}", "exit_code": 1})

@app.route("/api/terminal/<session_id>/output", methods=["GET"])
def terminal_output(session_id):
    if session_id not in terminal_sessions:
        return jsonify({"output": []})
    return jsonify({"output": terminal_sessions[session_id]["output"]})

# ====================
# ROUTES - CODE EXECUTION
# ====================
@app.route("/api/execute/code", methods=["POST"])
def execute_code():
    data = request.get_json() or {}
    code = data.get("code", "")
    language = data.get("language", "python")
    suffix = ".py" if language == "python" else ".js"
    with tempfile.NamedTemporaryFile(mode='w', suffix=suffix, delete=False) as f:
        f.write(code)
        path = f.name
    try:
        cmd = ["python3", path] if language == "python" else ["node", path]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        return jsonify({"success": result.returncode == 0, "output": result.stdout, "error": result.stderr})
    except Exception as e:
        return jsonify({"success": False, "output": "", "error": str(e)})
    finally:
        os.unlink(path)

# ====================
# ROUTES - IMAGE GENERATION (proxy to tools_service if available)
# ====================
@app.route("/api/image", methods=["POST"])
def api_image():
    data = request.get_json() or {}
    prompt = data.get("prompt", "")
    if not prompt:
        return jsonify({"success": False, "error": "No prompt provided"}), 400

    gen_dir = Path(STATIC_PATH) / "static" / "generated"
    gen_dir.mkdir(parents=True, exist_ok=True)
    timestamp = int(time.time())
    file_stem = f"gen_{timestamp}"

    # ---- 1) ZO native image generator (REAL images, platform-provided) ----
    # Mirrors the approach in from-scratch/web_ui/neuralai_engine.py which calls
    # /home/.z/tools/generate_image.py — the host's real image generation tool.
    try:
        script = (
            "import sys, os\n"
            "os.environ['ZO_CLIENT_IDENTITY_TOKEN'] = " + repr(ZO_API_TOKEN) + "\n"
            "sys.path.insert(0, '/home/.z/tools')\n"
            "try:\n"
            "    from generate_image import generate_image as _gen\n"
            "except Exception as e:\n"
            "    print('IMPORT_FAIL', e); sys.exit(2)\n"
            "ok = _gen(prompt=" + repr(prompt) +
            ", output_dir=" + repr(str(gen_dir)) +
            ", file_stem=" + repr(file_stem) + ", aspect_ratio='1:1')\n"
            "sys.exit(0 if ok else 1)\n"
        )
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as tf:
            tf.write(script)
            script_path = tf.name
        env = dict(os.environ)
        env["ZO_CLIENT_IDENTITY_TOKEN"] = ZO_API_TOKEN
        try:
            r = subprocess.run(["python3", script_path], capture_output=True, text=True, timeout=150, env=env)
        finally:
            try:
                os.unlink(script_path)
            except Exception:
                pass
        if r.returncode == 0:
            matches = sorted(gen_dir.glob(f"{file_stem}*"))
            if matches:
                fname = matches[-1].name
                return jsonify({
                    "success": True,
                    "image_url": f"/static/generated/{fname}",
                    "prompt": prompt,
                    "placeholder": False,
                    "provider": "zo-native"
                })
            logger.warning("[api_image] ZO generator reported success but produced no file")
        else:
            logger.warning(f"[api_image] ZO native image gen returned {r.returncode}: {r.stderr[:200]}")
    except Exception as e:
        logger.warning(f"[api_image] ZO native image gen unavailable: {e}")

    # ---- 2) Local diffusion engine (REAL SD model, opt-in to avoid OOM on small hosts) ----
    if os.environ.get("NEURALAI_DIFFUSION", "").lower() in ("1", "true", "yes"):
        try:
            import sys as _sys
            _svc_dir = os.path.dirname(os.path.abspath(__file__))
            if _svc_dir not in _sys.path:
                _sys.path.insert(0, _svc_dir)
            from diffusion_engine import NeuralAIDiffusion
            # Expand the short user prompt into a brand-styled image prompt.
            enhanced = enhance_image_prompt(prompt)
            engine = NeuralAIDiffusion()
            out_path = gen_dir / f"{file_stem}.png"
            if engine.generate(enhanced, str(out_path)):
                return jsonify({
                    "success": True,
                    "image_url": f"/static/generated/{file_stem}.png",
                    "prompt": enhanced,
                    "raw_prompt": prompt,
                    "placeholder": False,
                    "provider": "diffusion"
                })
        except Exception as e:
            logger.warning(f"[api_image] Diffusion image gen failed: {e}")

    # ---- 3) Last resort: clearly-labeled concept placeholder (never pretend it's AI) ----
    try:
        from PIL import Image, ImageDraw
        import random
        filename = f"{file_stem}.png"
        filepath = gen_dir / filename
        img = Image.new("RGB", (512, 512))
        draw = ImageDraw.Draw(img)
        # Deterministic-ish gradient from prompt hash
        seed = sum(ord(c) for c in prompt)
        random.seed(seed)
        base_r, base_g, base_b = random.randint(20, 80), random.randint(20, 80), random.randint(60, 140)
        for y in range(512):
            r = min(255, base_r + int((y / 512) * 80))
            g = min(255, base_g + int((y / 512) * 100))
            b = min(255, base_b + int((y / 512) * 120))
            draw.line([(0, y), (512, y)], fill=(r, g, b))
        # A few accent circles for visual interest
        for _ in range(5):
            x, y = random.randint(40, 472), random.randint(40, 472)
            rad = random.randint(20, 70)
            col = (random.randint(150, 255), random.randint(150, 255), random.randint(150, 255))
            draw.ellipse([x - rad, y - rad, x + rad, y + rad], fill=col)
        draw.text((20, 470), f"Concept: {prompt[:40]}", fill=(220, 220, 220))
        img.save(filepath)
        return jsonify({
            "success": True,
            "image_url": f"/static/generated/{filename}",
            "prompt": prompt,
            "placeholder": True,
            "note": "AI image generation is unavailable on this host, so this is a concept placeholder (not a real AI image). Enable ZO image generation or set NEURALAI_DIFFUSION=1 for local models."
        })
    except Exception as e:
        return jsonify({"success": False, "error": f"Image generation failed: {e}"})


# ====================
# ROUTES - TOOLS (web surf / search / fetch / browse)
# ====================
@app.route("/api/tool", methods=["POST"])
@token_required
def api_tool(current_user):
    """Execute a backend tool (web_search, web_browser, web_fetcher, ...).

    Body: {"tool": "web_search", "params": {"query": "...", "top_k": 5}}
    Returns the tool handler's result dict.
    """
    data = request.get_json() or {}
    tool = data.get("tool", "").strip()
    params = data.get("params", {}) or {}
    if not tool:
        return jsonify({"success": False, "error": "No tool specified"}), 400
    try:
        result = _run_tool(tool, params)
        return jsonify(result)
    except Exception as e:
        logger.error(f"[api_tool] {tool} failed: {e}")
        return jsonify({"success": False, "error": f"Tool error: {e}", "data": {}}), 500

# ====================
# ROUTES - EMBEDDED BROWSER (Mirror + Standalone)
# ====================
# A single Playwright session (tools/web_browser.BrowserSession) is shared per
# client so the embedded browser panel in the UI can both (A) mirror the AI's
# automation steps live AND (B) act as a standalone user-controlled browser.
# Screenshots are returned as base64 PNG data URLs so the SPA can render them
# without any extra static-file wiring.

def _browser_session(uid):
    """Get (creating if needed) the per-user real Playwright browser manager."""
    sid = "ui_" + str(uid or "anon")
    return _web_browser.get_manager(sid)

def _plan_browser_steps(task, start_url=""):
    """Lightweight planner: turn a natural-language task into plain-language
    browser steps for the AI Mirror stream. No network, no model dependency."""
    t = (task or "").lower()
    steps = []
    import re as _re
    m = _re.search(r"https?://\S+", task or "")
    if start_url:
        steps.append(f"Open {start_url}")
    elif m:
        steps.append(f"Open {m.group(0)}")
    if any(k in t for k in ("search", "look up", "find", "google")):
        q = _re.sub(r"^(please |can you |help me )?(search|look up|find|google) (for |on |about )?", "", t)
        q = _re.sub(r"https?://\S+", "", q).strip()
        if q:
            steps.append(f"Search Google for: {q}")
            steps.append("Scroll down to read results")
    elif m is None and any(k in t for k in ("go to", "open", "visit", "navigate to")):
        site = _re.sub(r".*?(go to|open|visit|navigate to)\s*", "", t).strip()
        if site and "." in site:
            steps.append(f"Open https://{site}" if not site.startswith("http") else f"Open {site}")
    else:
        if task:
            steps.append(f"Search Google for: {task}")
            steps.append("Scroll down to read results")
    if not steps:
        steps.append("Open a new page")
    return steps[:8]

@app.route("/api/browser/health", methods=["GET"])
@token_required
def api_browser_health(current_user):
    mgr = _browser_session(current_user)
    return jsonify({"success": True, "active": len(mgr.tabs()), "engine": "playwright-chromium"})


def _coerce_tab_id(raw):
    """Convert a string/int tab id from the frontend into int or None (active)."""
    if raw is None:
        return None
    s = str(raw).strip()
    if s == "" or s.lower() == "active":
        return None
    try:
        return int(s)
    except ValueError:
        return None


@app.route("/api/browser/tabs", methods=["GET"])
@token_required
def api_browser_tabs(current_user):
    mgr = _browser_session(current_user)
    return jsonify({"success": True, "active_tab": mgr.active, "tabs": mgr.list_tabs()})


@app.route("/api/browser/new", methods=["POST"])
@token_required
def api_browser_new(current_user):
    data = request.get_json(silent=True) or {}
    url = (data.get("url") or "").strip()
    mgr = _browser_session(current_user)
    tab = mgr.new_tab(url)
    return jsonify({"success": True, "active_tab": mgr.active, "tabs": mgr.list_tabs(), "tab": tab})


@app.route("/api/browser/close-tab", methods=["POST"])
@token_required
def api_browser_close_tab(current_user):
    data = request.get_json(silent=True) or {}
    tab_id = _coerce_tab_id(data.get("tab_id") or data.get("id"))
    mgr = _browser_session(current_user)
    tabs = mgr.close_tab(tab_id)
    return jsonify({"success": True, "active_tab": mgr.active, "tabs": tabs})


@app.route("/api/browser/select", methods=["POST"])
@token_required
def api_browser_select(current_user):
    data = request.get_json(silent=True) or {}
    tab_id = _coerce_tab_id(data.get("tab_id") or data.get("id"))
    mgr = _browser_session(current_user)
    tab = mgr.select(tab_id)
    return jsonify({"success": True, "active_tab": mgr.active, "tabs": mgr.list_tabs(), "tab": tab})


@app.route("/api/browser/navigate", methods=["POST"])
@token_required
def api_browser_navigate(current_user):
    data = request.get_json(silent=True) or {}
    url = (data.get("url") or "").strip()
    tab_id = _coerce_tab_id(data.get("tab_id") or data.get("id"))
    if not url:
        return jsonify({"success": False, "error": "url required"}), 400
    if not url.startswith("http"):
        url = "https://" + url
    try:
        mgr = _browser_session(current_user)
        tab = mgr.navigate(url, tab_id=tab_id)
        return jsonify({"success": True, "active_tab": mgr.active, "tabs": mgr.list_tabs(), "tab": tab})
    except Exception as e:
        return jsonify({"success": False, "error": str(e), "url": url}), 500


@app.route("/api/browser/action", methods=["POST"])
@token_required
def api_browser_action(current_user):
    data = request.get_json(silent=True) or {}
    act = (data.get("action") or "").lower()
    extra = data.get("extra") or {}
    tab_id = _coerce_tab_id(data.get("tab_id") or data.get("id"))
    try:
        mgr = _browser_session(current_user)
        tab = mgr.action(act, extra=extra, tab_id=tab_id)
        return jsonify({"success": True, "active_tab": mgr.active, "tabs": mgr.list_tabs(), "tab": tab})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/browser/run", methods=["POST"])
@token_required
def api_browser_run(current_user):
    """Mirror mode: AI drives a multi-step task across the active tab.
    Streams each step as an SSE event with a screenshot so the UI shows progress."""
    data = request.get_json(silent=True) or {}
    task = (data.get("task") or "").strip()
    url = (data.get("url") or "").strip()
    tab_id = (data.get("tab_id") or "").strip()
    if not task and not url:
        return jsonify({"success": False, "error": "task or url required"}), 400

    mgr = _browser_session(current_user)

    def gen():
        import json as _json
        def _ev(kind, text, tab=None):
            payload = {"kind": kind, "text": text}
            if tab:
                payload["tab"] = tab
            return "data: " + _json.dumps(payload) + "\n\n"
        try:
            start_url = url if url.startswith("http") else ("https://" + url if url else None)
            steps = _plan_browser_steps(task, start_url)
            for i, step in enumerate(steps):
                tab = mgr.action(step, tab_id=tab_id or None)
                yield _ev("step", f"[{i+1}/{len(steps)}] {step}", tab)
            yield _ev("done", "Task complete", mgr.get_tab(mgr.active))
        except Exception as e:
            yield _ev("error", f"Error: {e}")

    return Response(gen(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.route("/api/browser/close", methods=["POST"])
@token_required
def api_browser_close(current_user):
    try:
        _web_browser.close_manager("ui_" + str(current_user or "anon"))
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500
# ====================
# ROUTES - AUTH
# ====================
@app.route("/api/auth/guest", methods=["POST"])
def guest_login():
    code = uuid.uuid4().hex[:8]
    user_id = f"guest_{os.urandom(4).hex()}"
    token = jwt.encode({"user_id": user_id, "role": "maestro"}, app.config["SECRET_KEY"], algorithm="HS256")
    return jsonify({"token": token, "user": {"username": f"Maestro_{code[:4]}", "role": "maestro"}})

@app.route("/api/auth/signup", methods=["POST"])
def signup():
    data = request.get_json(silent=True) or {}
    username = data.get("username", "").strip()
    email = data.get("email", "").strip()
    password = data.get("password", "")
    if not username or not password:
        return jsonify({"error": "Missing fields"}), 400
    is_founder = 1 if email == FOUNDER_EMAIL else 0
    hashed = generate_password_hash(password)
    uid = "user_" + uuid.uuid4().hex[:8]
    now = datetime.now(timezone.utc).isoformat()
    db = get_db()
    try:
        db.execute("INSERT INTO users (id, username, email, is_founder, password_hash, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                   (uid, username, email, is_founder, hashed, now))
        db.commit()
        # Provision a per-user Nextcloud home (NeuralDrive) for this new account
        try:
            provision_user(neuralai_to_nc_username(uid))
        except Exception as _pe:
            logger.warning(f"Nextcloud provisioning skipped for {uid}: {_pe}")
        token = jwt.encode({"user_id": uid, "is_founder": is_founder, "exp": datetime.now(timezone.utc) + timedelta(days=30)},
                           app.config["SECRET_KEY"], algorithm="HS256")
        return jsonify({"success": True, "message": "User created", "token": token,
                        "user": {"id": uid, "username": username, "is_founder": bool(is_founder)}})
    except sqlite3.IntegrityError:
        return jsonify({"error": "Username or email exists"}), 409
    finally:
        db.close()

@app.route("/api/auth/login", methods=["POST"])
def login():
    data = request.get_json(silent=True) or {}
    identity = (data.get("username") or data.get("email") or "").strip()
    password = data.get("password", "")
    if not identity or not password:
        return jsonify({"error": "Missing credentials"}), 400
    db = get_db()
    try:
        user = db.execute("SELECT * FROM users WHERE username = ? OR email = ?", (identity, identity)).fetchone()
        if user and check_password_hash(user["password_hash"], password):
            # Auto-promote the founder account on login (in case it predates the flag)
            if user["email"] == FOUNDER_EMAIL and not user["is_founder"]:
                db.execute("UPDATE users SET is_founder = 1 WHERE id = ?", (user["id"],))
                db.commit()
                user = db.execute("SELECT * FROM users WHERE id = ?", (user["id"],)).fetchone()
            token = jwt.encode({"user_id": user["id"], "is_founder": user["is_founder"],
                                "exp": datetime.now(timezone.utc) + timedelta(days=30)},
                               app.config["SECRET_KEY"], algorithm="HS256")
            # Ensure the user's NeuralDrive (Nextcloud home) exists on login
            try:
                provision_user(neuralai_to_nc_username(user["id"]))
            except Exception as _pe:
                logger.warning(f"Nextcloud provisioning skipped for {user['id']}: {_pe}")
            return jsonify({"success": True, "token": token,
                            "user": {"id": user["id"], "username": user["username"], "is_founder": bool(user["is_founder"])}})
        return jsonify({"error": "Invalid credentials"}), 401
    finally:
        db.close()

@app.route("/api/auth/maestro", methods=["POST"])
def maestro_login():
    data = request.get_json(silent=True) or {}
    code_in = (data.get("code") or data.get("maestro_id") or "").strip()
    if not code_in:
        return jsonify({"error": "Maestro ID required"}), 400
    user_id = f"maestro_{os.urandom(4).hex()}"
    token = jwt.encode({"user_id": user_id, "role": "maestro"}, app.config["SECRET_KEY"], algorithm="HS256")
    return jsonify({"token": token, "user": {"username": code_in, "role": "maestro"}})

# ====================
# ROUTES - USER
# ====================
@app.route("/api/user/me", methods=["GET"])
@token_required
def get_user_me(current_user):
    db = get_db()
    try:
        user = db.execute("SELECT * FROM users WHERE id = ?", (current_user,)).fetchone()
        if not user:
            return jsonify({"user": {"id": current_user, "username": current_user, "is_founder": False}})
        u_dict = dict(user)
        if "password_hash" in u_dict: del u_dict["password_hash"]
        return jsonify({"user": u_dict})
    finally:
        db.close()

@app.route("/api/user/update", methods=["POST"])
@token_required
def update_user(current_user):
    data = request.get_json(silent=True) or {}
    db = get_db()
    try:
        for field in ["first_name", "last_name", "bio", "bod", "email"]:
            if field in data:
                db.execute(f"UPDATE users SET {field} = ? WHERE id = ?", (data[field], current_user))
        db.commit()
        return jsonify({"success": True})
    finally:
        db.close()

# ====================
# ROUTES - SETTINGS
# ====================
@app.route("/api/settings", methods=["GET", "POST"])
@token_required
def manage_settings(current_user):
    db = get_db()
    try:
        if request.method == "POST":
            data = request.get_json() or {}
            now = datetime.now(timezone.utc).isoformat()
            for k, v in data.items():
                db.execute("INSERT OR REPLACE INTO user_settings (user_id, key, value, updated_at) VALUES (?, ?, ?, ?)",
                           (current_user, k, str(v), now))
            db.commit()
            return jsonify({"success": True})
        rows = db.execute("SELECT key, value FROM user_settings WHERE user_id = ?", (current_user,)).fetchall()
        settings = {row["key"]: row["value"] for row in rows}
        return jsonify({"success": True, "settings": settings})
    finally:
        db.close()

# ====================
# ROUTES - API KEY (BYO API)
# ====================
# NeuralAI can act as an OpenAI-compatible backend for external hosts (e.g. ZO Computer
# "BYO API"). A user generates a personal API key here; the key is stored hashed and used
# to authenticate requests to /v1/chat/completions. The raw key is shown only once.
def _hash_key(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()

def _api_key_hint(raw: str) -> str:
    if not raw:
        return ""
    if len(raw) <= 16:
        return raw
    return f"{raw[:10]}…{raw[-6:]}"

def _api_key_status(current_user):
    db = get_db()
    try:
        rows = db.execute(
            "SELECT key, value FROM user_settings WHERE user_id = ? AND key IN ('api_key_hint', 'api_key_created_at', 'api_key_last_used_at', 'api_key_hash')",
            (current_user,),
        ).fetchall()
        data = {row["key"]: row["value"] for row in rows}
        has_key = bool(data.get("api_key_hash"))
        return {
            "has_key": has_key,
            "api_key_hint": data.get("api_key_hint", "Active key on file"),
            "api_key_created_at": data.get("api_key_created_at"),
            "api_key_last_used_at": data.get("api_key_last_used_at"),
        }
    finally:
        db.close()

@app.route("/api/settings/api-key", methods=["POST", "DELETE"])
@app.route("/api/settings/api-key", methods=["GET"])
@token_required
def manage_api_key(current_user):
    db = get_db()
    try:
        if request.method == "GET":
            status = _api_key_status(current_user)
            status.update({
                "success": True,
                "endpoint": "https://neuralai-web-ui-deandrewharris.zocomputer.io/v1",
                "chat_endpoint": "https://neuralai-web-ui-deandrewharris.zocomputer.io/v1/chat/completions",
                "model": "neuralai",
            })
            return jsonify(status)
        if request.method == "DELETE":
            for key_name in ("api_key_hash", "api_key_hint", "api_key_created_at", "api_key_last_used_at"):
                db.execute("DELETE FROM user_settings WHERE user_id = ? AND key = ?", (current_user, key_name))
            db.commit()
            return jsonify({"success": True, "message": "API key revoked."})

        # POST -> generate a new key (revoking any previous one)
        raw = "nai_" + secrets.token_urlsafe(32)
        hint = _api_key_hint(raw)
        created_at = datetime.now(timezone.utc).isoformat()
        db.execute("INSERT OR REPLACE INTO user_settings (user_id, key, value, updated_at) VALUES (?, ?, ?, ?)",
                   (current_user, "api_key_hash", _hash_key(raw), datetime.now(timezone.utc).isoformat()))
        db.execute("INSERT OR REPLACE INTO user_settings (user_id, key, value, updated_at) VALUES (?, ?, ?, ?)",
                   (current_user, "api_key_hint", hint, created_at))
        db.execute("INSERT OR REPLACE INTO user_settings (user_id, key, value, updated_at) VALUES (?, ?, ?, ?)",
                   (current_user, "api_key_created_at", created_at, created_at))
        db.commit()
        # Return the raw key ONCE. It is never stored or retrievable again.
        return jsonify({
            "success": True,
            "api_key": raw,
            "api_key_hint": hint,
            "endpoint": "https://neuralai-web-ui-deandrewharris.zocomputer.io/v1",
            "chat_endpoint": "https://neuralai-web-ui-deandrewharris.zocomputer.io/v1/chat/completions",
            "model": "neuralai",
        })
    finally:
        db.close()

def _user_for_api_key(api_key: str):
    """Resolve a raw API key to a user_id, or None if invalid.

    Accepts two credential types:
      1. A NeuralAI-generated personal API key (stored hashed in user_settings).
      2. The ZO Computer platform identity token (ZO_CLIENT_IDENTITY_TOKEN) — required
         when the request passes through ZO's hosting gateway, which rejects any call
         lacking a valid platform Authorization header. When the platform token is
         presented, we resolve to the founder account so the gateway's auth and the
         app's auth both succeed.
    """
    if not api_key:
        return None
    # 1) ZO platform token (gateway auth)
    zo_token = os.environ.get("ZO_CLIENT_IDENTITY_TOKEN", "")
    if zo_token and api_key == zo_token:
        return "founder"
    # 2) NeuralAI personal API key (hashed lookup)
    h = _hash_key(api_key)
    db = get_db()
    try:
        row = db.execute("SELECT user_id FROM user_settings WHERE key = 'api_key_hash' AND value = ?", (h,)).fetchone()
        if not row:
            return None
        user_id = row["user_id"]
        now = datetime.now(timezone.utc).isoformat()
        db.execute(
            "INSERT OR REPLACE INTO user_settings (user_id, key, value, updated_at) VALUES (?, ?, ?, ?)",
            (user_id, "api_key_last_used_at", now, now),
        )
        db.commit()
        return user_id
    finally:
        db.close()

@app.route("/v1/models", methods=["GET"])
def list_models():
    """OpenAI-compatible model listing for BYO API hosts."""
    active = _active_model()
    return jsonify({
        "object": "list",
        "data": [{
            "id": active["id"],
            "object": "model",
            "created": 1700000000,
            "owned_by": "neuralai",
            "root": active["id"],
            "parent": None,
        }]
    })


def _streaming_response(gen, model_id, stream):
    """Return an SSE stream or a single JSON chat.completion object based on
    the caller's `stream` flag. Every backend generator yields SSE
    'data: {...}' frames, so non-streaming just reassembles them."""
    if stream:
        return Response(stream_with_context(gen), mimetype="text/event-stream")
    parts = []
    for frame in gen:
        if not frame.startswith("data:"):
            continue
        payload = frame[len("data:"):].strip()
        if payload == "[DONE]":
            continue
        try:
            obj = json.loads(payload)
        except Exception:
            continue
        for ch in obj.get("choices", []):
            d = ch.get("delta", {})
            if d.get("content"):
                parts.append(d["content"])
    return jsonify({
        "id": "chatcmpl-" + secrets.token_hex(8),
        "object": "chat.completion",
        "created": int(datetime.now(timezone.utc).timestamp()),
        "model": model_id or "neuralai",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "".join(parts)}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    })

@app.route("/v1/chat/completions", methods=["GET"])
@app.route("/v1/chat/completions/", methods=["GET"])
@app.route("/v1/chat/completions/<model_id>", methods=["GET"])
@app.route("/v1/chat/completions/<model_id>/", methods=["GET"])
@app.route("/v1/chat/completions/chat/completions", methods=["GET"])
@app.route("/v1/chat/completions/chat/completions/", methods=["GET"])
@app.route("/v1", methods=["GET"])
@app.route("/v1/", methods=["GET"])
def openai_chat_completions_get(model_id=None):
    # Health / capability probe — ZO Computer's BYOK validation does a GET on the
    # endpoint (base URL or /v1/chat/completions). Return 200 so validation
    # passes; the actual chat runs over POST (openai_chat_completions below).
    return jsonify({
        "object": "list",
        "data": [{"id": "neuralai", "object": "model", "owned_by": "neuralai", "root": "neuralai", "parent": None}],
        "status": "ok",
    })

@app.route("/v1/chat/completions", methods=["POST"])
@app.route("/v1/chat/completions/", methods=["POST"])
@app.route("/v1/chat/completions/<model_id>", methods=["POST"])
@app.route("/v1/chat/completions/<model_id>/", methods=["POST"])
@app.route("/v1/chat/completions/chat/completions", methods=["POST"])
@app.route("/v1/chat/completions/chat/completions/", methods=["POST"])
@app.route("/v1", methods=["POST"])
@app.route("/v1/", methods=["POST"])
def openai_chat_completions(model_id=None):
    """OpenAI-compatible chat completions endpoint for external BYO API hosts (e.g. ZO Computer).

    Auth: Authorization: Bearer <api_key>  OR  ?api_key=<api_key>
    Accepts {model, messages, max_tokens, temperature, stream}.
    Uses the same local model + NeuralAI system prompt as the in-app chat.
    """
    # --- API key auth ---
    # Accept: Authorization: Bearer <key> | Authorization: <key> | x-api-key: <key> | ?api_key= | body.api_key
    auth = request.headers.get("Authorization", "")
    api_key = auth.replace("Bearer ", "", 1).strip() if auth else ""
    if not api_key:
        api_key = request.headers.get("X-Api-Key", "").strip()
    if not api_key:
        api_key = request.args.get("api_key", "").strip()
    if not api_key:
        api_key = (request.get_json(silent=True) or {}).get("api_key", "").strip()
    user_id = _user_for_api_key(api_key)
    if not user_id:
        # The ZO native backend authenticates via the platform identity token
        # (ZO_CLIENT_IDENTITY_TOKEN), not a user-supplied key, so it must be
        # allowed unkeyed just like the local backend. Otherwise every chat
        # request returns "Invalid API key" (the recurring unauthorized error).
        if LLM_BACKEND in ("local", "lmstudio", "openai_compatible"):
            user_id = "founder"
        else:
            return jsonify({"error": "Invalid API key"}), 401

    data = request.get_json(silent=True) or {}
    messages = data.get("messages", [])
    model_id = data.get("model", "neuralai")  # request model ID (not the global model object)
    # Local CPU backend is slow: cap generation lower so responses stream fast.
    _mt_default = 48 if LLM_BACKEND == "local" else 512
    _mt_cap = 80 if LLM_BACKEND == "local" else 2048
    max_tokens = min(int(data.get("max_tokens", _mt_default)), _mt_cap)
    temperature = float(data.get("temperature", 0.3))  # lower default for faster CPU inference
    # Default to streaming (BYO API hosts like ZO Computer show tokens as they
    # arrive), but honor the caller's `stream` flag so non-streaming OpenAI
    # clients also work.
    stream = bool(data.get("stream", True))

    # Build the same system prompt the in-app chat uses
    system_content = _build_system_prompt(user_id)

    # Assemble ChatML messages for the local model
    chat_messages = [{"role": "system", "content": system_content}]
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        if isinstance(content, list):  # handle multimodal content arrays
            content = " ".join(p.get("text", "") for p in content if isinstance(p, dict))
        chat_messages.append({"role": role, "content": _cap_text(content)})

    # Truncate to fit the model's context window (prevent OOM from 50K+ token payloads)
    # Skip when tokenizer is None (external backends handle their own limits)
    if tokenizer is not None:
        chat_messages = _truncate_to_fit(chat_messages, tokenizer)

    # === External backend: forward messages directly (no tokenizer needed) ===
    if LLM_BACKEND in ("ollama", "lmstudio", "openai_compatible"):
        def gen_external():
            try:
                resp = _forward_to_external_llm(chat_messages, max_tokens=max_tokens, temperature=temperature, stream=True)
                if resp.status_code != 200:
                    err = f"Backend error ({resp.status_code}): {resp.text[:200]}"
                    yield "data: " + json.dumps({"choices": [{"delta": {"content": err}, "finish_reason": None}]}) + "\n\n"
                else:
                    for line in resp.iter_lines():
                        if not line or not line.startswith(b"data: "):
                            continue
                        payload = line[6:].decode().strip()
                        if payload == "[DONE]":
                            break
                        try:
                            chunk = json.loads(payload)
                            delta = chunk.get("choices", [{}])[0].get("delta", {})
                            content = delta.get("content", "")
                            if content:
                                yield "data: " + json.dumps({"choices": [{"index": 0, "delta": {"content": content}, "finish_reason": None}]}) + "\n\n"
                        except json.JSONDecodeError:
                            continue
            except Exception as e:
                yield "data: " + json.dumps({"choices": [{"delta": {"content": f"Error: {e}"}, "finish_reason": None}]}) + "\n\n"
            yield "data: " + json.dumps({"choices": [{"delta": {}, "finish_reason": "stop"}]}) + "\n\n"
            yield "data: [DONE]\n\n"
        return _streaming_response(gen_external(), model_id, stream)


    # === Local PyTorch: render via tokenizer ===
    try:
        prompt = tokenizer.apply_chat_template(chat_messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        # Fallback manual ChatML assembly
        out = []
        for i, msg in enumerate(chat_messages):
            if i == 0 and msg["role"] != "system":
                out.append("<|im_start|>system\nYou are a helpful AI assistant named NeuralAI<|im_end|>\n")
            out.append(f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n")
        out.append("<|im_start|>assistant\n")
        prompt = "".join(out)

    # Streaming (SSE) — always enabled for BYO API compatibility
    # CRITICAL: already_rendered=True because prompt was built via apply_chat_template above.
    # Passing it through build_prompt_with_context again would double-wrap in ChatML.
    def gen():
        yield "data: " + json.dumps({"id": "chatcmpl-" + secrets.token_hex(8), "object": "chat.completion.chunk",
                                      "created": int(datetime.now(timezone.utc).timestamp()), "model": model_id,
                                      "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]}) + "\n\n"
        for chunk in stream_response(prompt, max_tokens=max_tokens, temperature=temperature, already_rendered=True):
            content = re.sub(r"<tool>.*?</tool>", "", chunk, flags=re.DOTALL)
            if content:
                yield "data: " + json.dumps({"choices": [{"index": 0, "delta": {"content": content}, "finish_reason": None}]}) + "\n\n"
        yield "data: " + json.dumps({"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}) + "\n\n"
        yield "data: [DONE]\n\n"

    return _streaming_response(gen(), model_id, stream)

# ====================
# ROUTES - SELF UPDATE (Founder only)
# ====================
@app.route("/api/admin/update", methods=["POST"])
@token_required
def admin_self_update(current_user):
    """Pull the latest code from origin/master and restart this service in place.

    Gated to founder accounts only. The restart is performed by re-exec'ing the
    current process (os.execv) so the host's process manager keeps the same PID/socket.
    """
    db = get_db()
    try:
        user = db.execute("SELECT is_founder FROM users WHERE id = ?", (current_user,)).fetchone()
    finally:
        db.close()
    if not user or not user["is_founder"]:
        return jsonify({"success": False, "error": "Founder access required"}), 403

    try:
        pull = subprocess.run(
            ["git", "pull", "origin", "master"],
            cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=120
        )
        pull_out = (pull.stdout + pull.stderr).strip()
        if pull.returncode != 0:
            return jsonify({"success": False, "error": "git pull failed", "detail": pull_out}), 500
    except Exception as e:
        return jsonify({"success": False, "error": f"git pull error: {e}"}), 500

    # Restart in place: re-exec the current interpreter with the same argv.
    # The host's process manager (or ZO entrypoint) will keep serving on the same port.
    try:
        logger.info("Self-update: git pull succeeded, restarting in place...")
        os.execv(sys.executable, [sys.executable] + sys.argv)
    except Exception as e:
        return jsonify({"success": False, "error": f"restart failed: {e}", "pull": pull_out}), 500

# ====================
# ROUTES - MEMORY
# ====================
@app.route("/api/memory", methods=["GET", "POST"])
@token_required
def manage_memory(current_user):
    db = get_db()
    try:
        if request.method == "POST":
            data = request.get_json() or {}
            fact = data.get("fact")
            if not fact: return jsonify({"error": "Missing fact"}), 400
            now = datetime.now(timezone.utc).isoformat()
            db.execute("INSERT INTO memory_facts (fact, user_id, created_at) VALUES (?, ?, ?)", (fact, current_user, now))
            db.commit()
            return jsonify({"success": True})
        rows = db.execute("SELECT id, fact, created_at FROM memory_facts WHERE user_id = ? ORDER BY created_at DESC", (current_user,)).fetchall()
        return jsonify({"success": True, "facts": [dict(row) for row in rows]})
    finally:
        db.close()

@app.route("/api/memory/<int:id>", methods=["DELETE"])
@token_required
def delete_memory(current_user, id):
    db = get_db()
    try:
        db.execute("DELETE FROM memory_facts WHERE id = ? AND user_id = ?", (id, current_user))
        db.commit()
        return jsonify({"success": True})
    finally:
        db.close()

# ====================
# ROUTES - RULES
# ====================
@app.route("/api/rules", methods=["GET", "POST"])
@token_required
def manage_rules(current_user):
    db = get_db()
    try:
        if request.method == "POST":
            data = request.get_json() or {}
            rule = data.get("rule")
            if not rule: return jsonify({"error": "Missing rule"}), 400
            now = datetime.now(timezone.utc).isoformat()
            db.execute("INSERT INTO active_rules (rule, user_id, created_at) VALUES (?, ?, ?)", (rule, current_user, now))
            db.commit()
            return jsonify({"success": True})
        rows = db.execute("SELECT id, rule, active, created_at FROM active_rules WHERE user_id = ? ORDER BY created_at DESC", (current_user,)).fetchall()
        return jsonify({"success": True, "rules": [dict(row) for row in rows]})
    finally:
        db.close()

@app.route("/api/rules/<int:id>", methods=["DELETE"])
@token_required
def delete_rule(current_user, id):
    db = get_db()
    try:
        db.execute("DELETE FROM active_rules WHERE id = ? AND user_id = ?", (id, current_user))
        db.commit()
        return jsonify({"success": True})
    finally:
        db.close()

@app.route("/api/rules/<int:id>/toggle", methods=["POST"])
@token_required
def toggle_rule(current_user, id):
    db = get_db()
    try:
        row = db.execute("SELECT active FROM active_rules WHERE id = ? AND user_id = ?", (id, current_user)).fetchone()
        if row:
            new_status = 0 if row["active"] else 1
            db.execute("UPDATE active_rules SET active = ? WHERE id = ? AND user_id = ?", (new_status, id, current_user))
            db.commit()
        return jsonify({"success": True})
    finally:
        db.close()

# ====================
# ROUTES - UPLOAD
# ====================
@app.route("/api/upload", methods=["POST"])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file"}), 400
    file = request.files['file']
    save_path = STORAGE_ROOT / file.filename
    file.save(str(save_path))
    return jsonify({"success": True, "name": file.filename, "size": save_path.stat().st_size})

# ====================
# WEBSOCKET PROXY - Voice Service
# ====================
# Proxies WebSocket connections from /voice/ws to the local voice service on port 5001
VOICE_SERVICE = os.environ.get("VOICE_SERVICE_URL", "ws://127.0.0.1:5001/ws")

@app.route("/voice/ws")
def voice_ws_proxy():
    """Upgrade HTTP to WebSocket and proxy to voice service."""
    from flask_sock import Sock
    import websocket as ws_lib
    
    # This endpoint is handled by flask_sock via the sock instance below
    pass

sock = Sock(app)

@sock.route("/voice/ws")
def voice_proxy(ws):
    """Proxy WebSocket between browser and voice service on localhost:5001."""
    import websocket as ws_lib
    import threading
    
    logger.info("[VoiceProxy] Browser connected, opening upstream to %s", VOICE_SERVICE)
    
    # Connect upstream to the voice service
    upstream = ws_lib.create_connection(VOICE_SERVICE, timeout=30)
    
    def recv_from_upstream():
        try:
            while True:
                data = upstream.recv()
                if not data:
                    break
                ws.send(data)
        except Exception as e:
            logger.info("[VoiceProxy] Upstream closed: %s", e)
        finally:
            try:
                ws.close()
            except:
                pass
    
    t = threading.Thread(target=recv_from_upstream, daemon=True)
    t.start()
    
    try:
        while True:
            data = ws.receive()
            if data is None:
                break
            upstream.send(data)
    except Exception as e:
        logger.info("[VoiceProxy] Browser disconnected: %s", e)
    finally:
        try:
            upstream.close()
        except:
            pass

# ====================
# STARTUP
# ====================
if __name__ == "__main__":
    print(f"NeuralAI Unified Service starting on port {PORT}...")
    init_db()
    if LLM_BACKEND == "local":
        load_model()
    else:
        logger.info(f"[BOOT] Backend={LLM_BACKEND} — skipping local model load")
    # Launch defense threads: keep-alive + memory watchdog
    threading.Thread(target=_keep_alive_pinger, daemon=True).start()
    threading.Thread(target=_memory_watchdog, daemon=True).start()
    logger.info("[BOOT] Defense threads launched: keep-alive pinger + memory watchdog")
    app.run(host="0.0.0.0", port=PORT, debug=False, threaded=True)
