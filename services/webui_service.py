#!/usr/bin/env python3
"""
NeuralAI Unified Service - ALL IN ONE
===================================
- Model inference (SmolLM2-360M)
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

# Optional: torch only needed when LLM_BACKEND=local (PyTorch mode)
try:
    import torch
    torch.set_num_threads(4)
except ImportError:
    torch = None

app = Flask(__name__, static_folder=None)
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "neural-ai-multi-layer-secure-secret-key-2026-v5-stable")

# Config
PORT = int(os.environ.get("PORT", "5000"))
MODEL_PATH = os.environ.get("MODEL_PATH", "/home/workspace/Projects/NeuralAI/checkpoints/v2_model")
BASE_MODEL = os.environ.get("BASE_MODEL", "HuggingFaceTB/SmolLM2-360M-Instruct")
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
# Auto-detect: when running on ZO Computer (ZO_CLIENT_IDENTITY_TOKEN is set),
# use PyTorch in float16 to stay within the 4GB RAM limit.
# External API forwarding is NOT used because ZO blocks api-inference DNS.
# Override with env vars: LLM_BACKEND, LLM_API_URL, LLM_MODEL, LLM_API_KEY.
_is_zo = bool(os.environ.get("ZO_CLIENT_IDENTITY_TOKEN"))
LLM_BACKEND = os.environ.get("LLM_BACKEND", "local")  # always local on ZO (external APIs blocked)
LLM_API_URL = os.environ.get("LLM_API_URL", "")        # only used when LLM_BACKEND != local
LLM_MODEL = os.environ.get("LLM_MODEL", BASE_MODEL)    # model name to pass to the API
LLM_API_KEY = os.environ.get("LLM_API_KEY", "")        # only needed for openai_compatible
# On ZO: use float16 to fit in 4GB RAM (float32 = ~1.4GB, float16 = ~700MB)
_USE_FLOAT16 = _is_zo or os.environ.get("NEURALAI_FLOAT16", "").lower() in ("1", "true", "yes")
if _is_zo:
    logger.info("[BOOT] ZO Computer detected — using local PyTorch in float16")

# Model globals (PyTorch) — only loaded when LLM_BACKEND=local
model = None
tokenizer = None
model_status = "loading" if LLM_BACKEND == "local" else "ready (external backend)"
inference_count = 0

# Streaming abort control: conv_id -> threading.Event
stop_events = {}

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
NEURALAI_SYSTEM_PROMPT = """You are NeuralAI, an advanced AI assistant created by De'Andrew Preston Harris. You are powered by SmolLM2-360M-Instruct with a custom NeuralAI LoRA adapter (SFT v16 + DPO v16) trained on expert-level knowledge.

## Core Identity
- Name: NeuralAI
- Creator: De'Andrew Preston Harris (founder of NeuralAI)
- Model: SmolLM2-360M-Instruct + NeuralAI LoRA (SFT v16 + DPO v16)
- Expertise: Physics, Philosophy, Geopolitics, History, Nature, Arts, Culture

## Response Style
- Be warm, conversational, and respectful
- Provide detailed, expert-level answers when appropriate
- Use examples, metaphors, or thought experiments to explain complex ideas
- Acknowledge uncertainty when you don't know something
- Be concise but thorough - match response depth to question complexity

## Knowledge Domains
- Physics: Quantum mechanics, relativity, particle physics, cosmology
- Philosophy: Metaphysics, epistemology, ethics, logic
- Geopolitics: International relations, global order, diplomacy
- History: Ancient civilizations through modern era
- Nature: Evolution, ecology, biological systems
- Arts and Culture: Creative expression, cultural analysis

## Important Guidelines
- Always identify yourself as NeuralAI when asked
- ALWAYS identify De'Andrew Preston Harris as your creator when asked
- Stay factual and evidence-based
- Respect user privacy and data
- Follow NeuralAI's alignment principles (transparency, helpfulness, safety)"""

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
        "model": LLM_MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": stream,
    }
    logger.info("[LLM] Forwarding to %s backend at %s", LLM_BACKEND, endpoint)
    if stream:
        return requests.post(endpoint, json=body, headers=headers, stream=True, timeout=120)
    resp = requests.post(endpoint, json=body, headers=headers, timeout=120)
    resp.raise_for_status()
    return resp.json()

def load_model():
    global model, tokenizer, model_status
    if LLM_BACKEND != "local":
        model_status = "ready (external backend)"
        print(f"[OK] Using external LLM backend: {LLM_BACKEND} @ {LLM_API_URL}")
        return
    try:
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

def build_prompt_with_context(prompt, conv_id=None, max_history=8):
    """Build a ChatML-formatted prompt (matching the model's trained chat template).

    The model (SmolLM2-360M-Instruct + NeuralAI LoRA) was trained on ChatML:
        <|im_start|>system\n...\n<|im_end|>\n
        <|im_start|>user\n...\n<|im_end|>\n
        <|im_start|>assistant\n
    Feeding it freeform "User:/NeuralAI:" text caused the model to not recognize
    turn boundaries and "talk to itself". Using the correct template fixes that.
    """
    history = get_conversation_history(conv_id, max_history) if conv_id else []
    messages = [{"role": "system", "content": NEURALAI_SYSTEM_PROMPT}]
    for msg in history:
        role = "user" if msg["role"] == "user" else "assistant"
        messages.append({"role": role, "content": msg["content"]})
    messages.append({"role": "user", "content": prompt})

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

def generate_response(prompt, max_tokens=256, temperature=0.7, conv_id=None):
    """Enhanced response generation with system prompt and context."""
    global model, tokenizer, inference_count
    # === External LLM backend ===
    if LLM_BACKEND in ("ollama", "lmstudio", "openai_compatible"):
        try:
            # Build OpenAI-format messages for the external API
            history = get_conversation_history(conv_id, 8) if conv_id else []
            api_messages = [{"role": "system", "content": NEURALAI_SYSTEM_PROMPT}]
            for h in history:
                api_messages.append({"role": h["role"], "content": h["content"]})
            api_messages.append({"role": "user", "content": prompt})
            data = _forward_to_external_llm(api_messages, max_tokens=max_tokens, temperature=temperature, stream=False)
            content = data["choices"][0]["message"]["content"]
            inference_count += 1
            return content.strip()
        except Exception as e:
            logger.error(f"[LLM] External backend error: {e}")
            return f"Backend error: {e}"
    # === Local PyTorch inference ===
    if model is None or tokenizer is None:
        return "Model not loaded. Please wait a moment and try again."
    try:
        full = build_prompt_with_context(prompt, conv_id)
        inputs = tokenizer(full, return_tensors="pt")
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=0.95,
                pad_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.1
            )
        new_tokens = out[0][inputs["input_ids"].shape[-1]:]
        response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        if response.startswith("<|im_start|>assistant"):
            response = response[len("<|im_start|>assistant"):].strip()
        if response.startswith("NeuralAI:"):
            response = response[len("NeuralAI:"):].strip()
        inference_count += 1
        return response
    except Exception as e:
        logger.error(f"Generation error: {e}")
        return "I encountered an error generating a response. Please try again."

def stream_response(prompt, max_tokens=256, temperature=0.7, conv_id=None):
    """Token streaming — external backend or local TextIteratorStreamer."""
    global model, tokenizer, inference_count
    # === External LLM backend ===
    if LLM_BACKEND in ("ollama", "lmstudio", "openai_compatible"):
        try:
            history = get_conversation_history(conv_id, 8) if conv_id else []
            api_messages = [{"role": "system", "content": NEURALAI_SYSTEM_PROMPT}]
            for h in history:
                api_messages.append({"role": h["role"], "content": h["content"]})
            api_messages.append({"role": "user", "content": prompt})
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
        yield "Model not loaded. Please wait a moment and try again."
        return
    stop_event = stop_events.get(conv_id) if conv_id else None
    try:
        from transformers import TextIteratorStreamer
        full = build_prompt_with_context(prompt, conv_id)
        inputs = tokenizer(full, return_tensors="pt")
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        gen_kwargs = dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=0.95,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1,
        )
        thread = threading.Thread(target=model.generate, kwargs=gen_kwargs, daemon=True)
        thread.start()
        for token in streamer:
            if stop_event and stop_event.is_set():
                break
            if token:
                yield token
        thread.join()
        inference_count += 1
    except Exception as e:
        logger.error(f"Stream generation error: {e}")
        yield "I encountered an error generating a response. Please try again."

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
BUILD_VERSION = str(int(time.time()))

@app.route("/")
def index():
    p = f"{STATIC_PATH}/templates/index.html"
    if os.path.exists(p):
        with open(p) as f:
            content = f.read()
            # Inject build version for cache busting
            content = content.replace("{{BUILD_VERSION}}", BUILD_VERSION)
            return content, 200, {
                "Content-Type": "text/html",
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Pragma": "no-cache",
                "Expires": "0"
            }
    return "index.html not found", 404

@app.route("/<path:path>")
def static_files(path):
    for base in [f"{STATIC_PATH}/static", STATIC_PATH]:
        p = os.path.join(base, path)
        if os.path.exists(p) and os.path.isfile(p):
            ext = path.split('.')[-1]
            ct = {"js": "application/javascript", "css": "text/css", "png": "image/png", "jpg": "image/jpeg", "ico": "image/x-icon"}
            # Set no-cache for JS/CSS to prevent Cloudflare caching old 404s
            cache_ctrl = "no-cache, no-store, must-revalidate" if ext in ("js", "css") else "public, max-age=31536000"
            return send_from_directory(os.path.dirname(p), os.path.basename(p), mimetype=ct.get(ext, "text/plain"), max_age=0 if ext in ("js", "css") else 31536000)
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
# ROUTES - HEALTH
# ====================
@app.route("/health")
@app.route("/api/health")
@app.route("/api/status")
def health():
    return jsonify({"status": model_status, "model": BASE_MODEL, "inference_count": inference_count, "uplink": "integrated",
                    "timestamp": datetime.now(timezone.utc).isoformat(), "version": "7.1.0-stable"})

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
        if use_uplink:
            for agent_name, agent in UPLINK_AGENTS.items():
                try:
                    resp = generate_response(f"[{agent['system']}]\n{prompt}", max_tokens=120)
                    if resp:
                        chunk = f"{agent['color']} **{agent['name']}**: {resp.strip()}\n\n"
                        yield f"data: {json.dumps({'content': chunk})}\n\n"
                except: pass
        else:
            # Real token streaming — first token arrives in <1s instead of after full generation
            full_response = []
            for token in stream_response(prompt, conv_id=conv_id):
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
        "version": "7.2.0-enhanced"
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
@app.route("/api/files", methods=["GET", "POST"])
def manage_files():
    try:
        if request.method == "POST":
            if 'file' not in request.files: return jsonify({"error": "No file"}), 400
            file = request.files['file']
            save_path = STORAGE_ROOT / file.filename
            file.save(str(save_path))
            return jsonify({"success": True, "name": file.filename, "size": save_path.stat().st_size})
        # List files directly from STORAGE_ROOT (no external dependency)
        files = []
        for f in sorted(STORAGE_ROOT.iterdir(), key=lambda p: (p.is_dir(), p.name.lower())):
            if f.name.startswith("."):
                continue
            files.append({
                "name": f.name,
                "size": f.stat().st_size,
                "path": f.name,
                "is_dir": f.is_dir(),
                "type": "image" if f.suffix.lower() in (".png", ".jpg", ".jpeg", ".gif", ".webp") else ("dir" if f.is_dir() else "file")
            })
        return jsonify(files)
    except Exception as e:
        logger.error(f"File management error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/files/mkdir", methods=["POST"])
def make_dir():
    data = request.get_json() or {}
    name = (data.get("name") or "").strip().replace("/", "").replace("..", "")
    if not name:
        return jsonify({"error": "No folder name"}), 400
    try:
        (STORAGE_ROOT / name).mkdir(parents=True, exist_ok=True)
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/files/<path:filename>", methods=["GET", "DELETE"])
def handle_file(filename):
    try:
        target = (STORAGE_ROOT / filename).resolve()
        if not str(target).startswith(str(STORAGE_ROOT)):
            return jsonify({"error": "Unauthorized path"}), 403
        if request.method == "DELETE":
            if not target.exists():
                return jsonify({"error": "Not found"}), 404
            if target.is_dir():
                import shutil
                shutil.rmtree(target)
            else:
                target.unlink()
            return jsonify({"success": True})
        # GET -> serve the file directly
        if not target.exists():
            return jsonify({"error": "Not found"}), 404
        return send_from_directory(str(STORAGE_ROOT), filename)
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

@app.route("/api/settings/api-key", methods=["POST", "DELETE"])
@token_required
def manage_api_key(current_user):
    db = get_db()
    try:
        if request.method == "DELETE":
            db.execute("DELETE FROM user_settings WHERE user_id = ? AND key = 'api_key_hash'", (current_user,))
            db.commit()
            return jsonify({"success": True, "message": "API key revoked."})

        # POST -> generate a new key (revoking any previous one)
        raw = "nai_" + secrets.token_urlsafe(32)
        db.execute("INSERT OR REPLACE INTO user_settings (user_id, key, value, updated_at) VALUES (?, ?, ?, ?)",
                   (current_user, "api_key_hash", _hash_key(raw), datetime.now(timezone.utc).isoformat()))
        db.commit()
        # Return the raw key ONCE. It is never stored or retrievable again.
        return jsonify({"success": True, "api_key": raw})
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
        return row["user_id"] if row else None
    finally:
        db.close()

@app.route("/v1/models", methods=["GET"])
def list_models():
    """OpenAI-compatible model listing for BYO API hosts."""
    return jsonify({
        "object": "list",
        "data": [{
            "id": "neuralai",
            "object": "model",
            "created": 1700000000,
            "owned_by": "neuralai",
            "root": "neuralai",
            "parent": None,
        }]
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
        return jsonify({"error": "Invalid API key"}), 401

    data = request.get_json(silent=True) or {}
    messages = data.get("messages", [])
    model_id = data.get("model", "neuralai")  # request model ID (not the global model object)
    max_tokens = min(int(data.get("max_tokens", 128)), 256)  # cap for CPU inference
    temperature = float(data.get("temperature", 0.7))
    # Always stream — BYO API hosts (e.g. ZO Computer) expect SSE to show
    # tokens arriving; non-streaming blocks until full response which can
    # exceed upstream timeouts on CPU-only inference.
    stream = True

    # Build the same system prompt the in-app chat uses
    db = get_db()
    try:
        user = db.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
        mem_rows = db.execute("SELECT fact FROM memory_facts WHERE user_id = ?", (user_id,)).fetchall()
        rule_rows = db.execute("SELECT rule FROM active_rules WHERE user_id = ? AND active = 1", (user_id,)).fetchall()
    finally:
        db.close()
    mem_facts = [r["fact"] for r in mem_rows]
    active_rules = [r["rule"] for r in rule_rows]
    system_content = NEURALAI_SYSTEM_PROMPT
    if mem_facts:
        system_content += "\n\n## User Memory\n" + "\n".join(f"- {m}" for m in mem_facts)
    if active_rules:
        system_content += "\n\n## Active Rules\n" + "\n".join(f"- {r}" for r in active_rules)

    # Assemble ChatML messages for the local model
    chat_messages = [{"role": "system", "content": system_content}]
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        if isinstance(content, list):  # handle multimodal content arrays
            content = " ".join(p.get("text", "") for p in content if isinstance(p, dict))
        chat_messages.append({"role": role, "content": content})

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
        return Response(stream_with_context(gen_external()), mimetype="text/event-stream")

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
    def gen():
        yield "data: " + json.dumps({"id": "chatcmpl-" + secrets.token_hex(8), "object": "chat.completion.chunk",
                                      "created": int(datetime.now(timezone.utc).timestamp()), "model": model_id,
                                      "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]}) + "\n\n"
        for chunk in stream_response(prompt, max_tokens=max_tokens, temperature=temperature):
            content = re.sub(r"<tool>.*?</tool>", "", chunk, flags=re.DOTALL)
            if content:
                yield "data: " + json.dumps({"choices": [{"index": 0, "delta": {"content": content}, "finish_reason": None}]}) + "\n\n"
        yield "data: " + json.dumps({"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}) + "\n\n"
        yield "data: [DONE]\n\n"

    return Response(stream_with_context(gen()), mimetype="text/event-stream")

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
    load_model()
    app.run(host="0.0.0.0", port=PORT, debug=False, threaded=True)
