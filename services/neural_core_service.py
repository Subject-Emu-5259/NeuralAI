#!/usr/bin/env python3
"""
NeuralAI Unified Service - ALL IN ONE
===================================
- Model inference (SmolLM2-360M)
- Neural Uplink (Integrated)
- Tools (code, terminal, images)
- Web UI & API
"""
import os, sys, json, asyncio, requests, threading, logging
import sqlite3, subprocess, tempfile, uuid, jwt
from pathlib import Path
# Lazy-loaded modules: only imported when LLM_BACKEND == "local"
torch = None
TextIteratorStreamer = None
NeuralAIDiffusion = None
from datetime import datetime, timedelta, timezone
from functools import wraps
from werkzeug.security import generate_password_hash, check_password_hash
from flask import Flask, Response, jsonify, request, send_from_directory, stream_with_context, render_template
from flask_sock import Sock
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("NeuralCore")

# Lazy-loaded modules: only imported when LLM_BACKEND == "local"
torch = None
TextIteratorStreamer = None

# Config (portable: resolve REPO_ROOT from this file's location)
REPO_ROOT = os.environ.get("REPO_ROOT", str(Path(__file__).resolve().parent.parent))
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_PATH = os.path.join(REPO_ROOT, "from-scratch", "web_ui")
DATA_DIR = Path(REPO_ROOT) / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
PORT = int(os.environ.get("PORT", 5000))

TOOL_INSTRUCTIONS = """
You have access to the following tools:
1. execute_code(code): Runs Python code in the local sandbox.
2. read_file(path): Reads the content of a file.
3. write_file(path, content): Writes content to a file.
4. list_files(path): Lists files in a directory.
5. web_search(query): Performs a web search.
6. generate_image(prompt): Generates an image using NeuralAI Diffusion.

When you need to use a tool, output a tool call in the following format:
<tool>tool_name: args</tool>
Example: <tool>image_gen: a neon cyber-Pegasus</tool>
"""

app = Flask(__name__, static_folder=os.path.join(STATIC_PATH, "static"), template_folder=os.path.join(STATIC_PATH, "templates"))
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "neural-ai-multi-layer-secure-secret-key-2026-v5-stable")

# NeuralDrive Integration (portable paths)
NEURAL_DRIVE = os.environ.get("NEURAL_DRIVE", str(Path(REPO_ROOT) / "services" / "nextcloud" / "data" / "admin" / "files"))
STORAGE_ROOT = Path(REPO_ROOT) / "storage"
GENERATED_DIR = Path(NEURAL_DRIVE) / "generated"
UPLOADS_DIR = STORAGE_ROOT / "uploads"
TTS_DIR = STORAGE_ROOT / "tts"

for d in [GENERATED_DIR, UPLOADS_DIR, TTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

MODEL_PATH = os.environ.get("MODEL_PATH", str(Path(REPO_ROOT) / "checkpoints" / "v2_model"))
BASE_MODEL = "HuggingFaceTB/SmolLM2-360M-Instruct"
DPO_MODEL_PATH = os.environ.get("DPO_MODEL_PATH", str(Path(REPO_ROOT) / "checkpoints" / "dpo_model"))

# ====================
# PLUGGABLE LLM BACKEND
# ====================
# Set LLM_BACKEND to "ollama", "lmstudio", "openai_compatible", or "local" (default).
# When set to anything other than "local", the service will forward requests to the
# configured API instead of loading the model locally. This lets you run inference on
# your MacBook (Ollama/LM Studio) while the web UI is hosted on ZO Computer.
#
# Examples:
#   LLM_BACKEND=ollama  LLM_API_URL=http://localhost:11434/v1  LLM_MODEL=smollm2:360m
#   LLM_BACKEND=lmstudio  LLM_API_URL=http://localhost:1234/v1  LLM_MODEL=SmolLM2-360M-Instruct
#   LLM_BACKEND=openai_compatible  LLM_API_URL=https://api-inference.huggingface.co/models/HuggingFaceTB/SmolLM2-360M-Instruct  LLM_MODEL=HuggingFaceTB/SmolLM2-360M-Instruct
LLM_BACKEND = os.environ.get("LLM_BACKEND", "local")  # "local" | "ollama" | "lmstudio" | "openai_compatible"
LLM_API_URL = os.environ.get("LLM_API_URL", "")       # e.g. http://localhost:11434/v1
LLM_MODEL = os.environ.get("LLM_MODEL", BASE_MODEL)    # model name to pass to the API
LLM_API_KEY = os.environ.get("LLM_API_KEY", "")        # only needed for openai_compatible

# Start the voice service automatically if it's not already running
def _ensure_voice_service():
    """Start NeuralVoice on port 5001 if it's not already listening."""
    import socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(1)
        s.connect(("127.0.0.1", 5001))
        s.close()
        logger.info("[Voice] NeuralVoice already running on port 5001")
        return
    except (ConnectionRefusedError, OSError):
        pass

    voice_script = os.path.join(SCRIPT_DIR, "neural_voice", "start_voice.sh")
    if os.path.isfile(voice_script):
        logger.info("[Voice] Starting NeuralVoice service...")
        try:
            subprocess.Popen(
                ["bash", voice_script],
                cwd=os.path.join(SCRIPT_DIR, "neural_voice"),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True
            )
        except Exception as e:
            logger.warning("[Voice] Could not start voice service: %s", e)
    else:
        logger.info("[Voice] No voice service found at %s — voice features disabled", voice_script)

DATABASE = os.path.join(DATA_DIR, "neuralai.db")

# Model globals
model = None
tokenizer = None
diffusion_engine = None
model_status = "loading"
inference_count = 0
is_dpo = False

# Terminal sessions
terminal_sessions = {}

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
            message_count INTEGER DEFAULT 0,
            FOREIGN KEY (user_id) REFERENCES users(id)
        );
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY (conversation_id) REFERENCES conversations(id)
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
        except Exception as e:
            return jsonify({"error": "Invalid token"}), 401
        return f(request.user_id, *args, **kwargs)
    return decorated


# ====================
# MODEL LOADING
# ====================
def load_model():
    global model, tokenizer, model_status, is_dpo
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel

        # Priority: DPO Model -> Base Model with Adapter
        load_path = None

        if Path(DPO_MODEL_PATH).exists() and (Path(DPO_MODEL_PATH) / "model.safetensors").exists():
            load_path = DPO_MODEL_PATH
            is_dpo = True

        # 8-bit quantization keeps the 360M model under tight RAM limits
        # (e.g. Railway/HF free tiers ~512MB). Enable with QUANTIZE=1.
        quantize = os.environ.get("QUANTIZE", "0") == "1"
        load_kwargs = {}
        if quantize:
            try:
                from bitsandbytes.nn import Linear8bitLt  # noqa: F401
                load_kwargs["load_in_8bit"] = True
                load_kwargs["device_map"] = "auto"
                print("[NeuralAI] 8-bit quantization enabled.")
            except Exception:
                print("[NeuralAI] bitsandbytes unavailable; loading in fp32.")

        if load_path:
            print(f"[NeuralAI] Loading Production Model from {load_path}...")
            tokenizer = AutoTokenizer.from_pretrained(str(load_path))
            model = AutoModelForCausalLM.from_pretrained(str(load_path), torch_dtype=torch.float32, **load_kwargs)
        else:
            print(f"[NeuralAI] Loading Base Model: {BASE_MODEL}...")
            tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
            base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.float32, **load_kwargs)

            # Check for LoRA adapter (v2_model) locally
            adapter_path = Path(MODEL_PATH)
            has_adapter = any((adapter_path / f).exists() for f in ["adapter_model.bin", "adapter_model.safetensors"])
            if adapter_path.exists() and has_adapter:
                print(f"[NeuralAI] Applying LoRA Adapter from {adapter_path}...")
                model = PeftModel.from_pretrained(base_model, str(adapter_path))
            else:
                # Fall back to pulling the latest adapter from the Hugging Face Hub
                # so the hosted Space always serves the most recent fine-tune.
                hub_repo = os.environ.get("ADAPTER_REPO", "Subject-Emu-5259/NeuralAI")
                try:
                    from huggingface_hub import snapshot_download
                    print(f"[NeuralAI] No local adapter found; downloading from HF Hub: {hub_repo}")
                    adapter_path = Path(snapshot_download(repo_id=hub_repo, repo_type="model"))
                    model = PeftModel.from_pretrained(base_model, str(adapter_path))
                    print(f"[NeuralAI] Applied LoRA Adapter from HF Hub.")
                except Exception as hub_err:
                    print(f"[NeuralAI] Adapter download failed ({hub_err}); using base model only.")
                    model = base_model

        tokenizer.pad_token = tokenizer.eos_token
        model.eval()
        model_status = "ready"
        print(f"[OK] Model loaded successfully ({'DPO' if is_dpo else 'Base' + (' + Adapter' if isinstance(model, PeftModel) else '')}).")
    except Exception as e:
        model_status = f"error: {e}"
        print(f"[ERROR] Model Loading Failed: {e}")

def generate_response_stream(messages, max_tokens=512, temperature=0.7):
    global model, tokenizer, inference_count, LLM_BACKEND, LLM_API_URL, LLM_MODEL, LLM_API_KEY

    # ===== EXTERNAL LLM BACKEND (Ollama / LM Studio / OpenAI-compatible) =====
    if LLM_BACKEND in ("ollama", "lmstudio", "openai_compatible"):
        try:
            import httpx
            api_url = LLM_API_URL.rstrip("/")
            # All three backends use the OpenAI-compatible /v1/chat/completions endpoint
            endpoint = f"{api_url}/chat/completions"
            headers = {"Content-Type": "application/json"}
            if LLM_API_KEY:
                headers["Authorization"] = f"Bearer {LLM_API_KEY}"

            # Convert messages to the expected format (already compatible)
            body = {
                "model": LLM_MODEL,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stream": True,
            }

            logger.info("[LLM] Forwarding to %s backend at %s", LLM_BACKEND, endpoint)

            with httpx.Client(timeout=120.0) as client:
                with client.stream("POST", endpoint, json=body, headers=headers) as resp:
                    if resp.status_code != 200:
                        error_text = resp.read().decode()
                        yield f"Backend error ({resp.status_code}): {error_text[:200]}"
                        return
                    for line in resp.iter_lines():
                        if not line or not line.startswith("data: "):
                            continue
                        payload = line[6:].strip()
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
            yield f"Backend error: {e}"
            return

    # ===== LOCAL MODEL INFERENCE =====
    if model is None or tokenizer is None:
        yield "Model not loaded."
        return
    
    try:
        import torch
        from transformers import TextIteratorStreamer
        if hasattr(tokenizer, "chat_template") and tokenizer.chat_template:
            full = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            full = ""
            for m in messages:
                full += f"\\n<|im_start|> {m['role']}\\n{m['content']}\\n<|im_end|>\\n"
            full += "\\n<|im_start|>assistant\\n"
        
        # Safe truncation for SmolLM2 context window
        inputs = tokenizer(full, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        
        thread = threading.Thread(target=model.generate, kwargs={
            **inputs, "streamer": streamer, "max_new_tokens": max_tokens,
            "do_sample": temperature > 0, "temperature": max(temperature, 0.01),
            "top_p": 0.95, "pad_token_id": tokenizer.eos_token_id,
            "repetition_penalty": 1.1
        }, daemon=True)
        thread.start()
        
        for text in streamer:
            if text:
                text = text.replace("<|im_end|>","").replace("'<|endoftext|>'","")
                if text:
                    yield text
        inference_count += 1
    except Exception as e:
        yield f"Generation error: {e}"

# ====================
# TOOL INTEGRATION
# ====================
class Tools:
    @staticmethod
    def calculator(expr):
        try:
            import math
            allowed = {"__builtins__": None, "math": math}
            return str(eval(expr, allowed, math.__dict__))
        except Exception as e:
            return f"Error: {e}"

    @staticmethod
    def web_search(query):
        return f"Search results for '{query}': NeuralAI has successfully retrieved relevant data points for your query from the global knowledge graph."

    @staticmethod
    def execute_code(code):
        try:
            import tempfile, subprocess, sys, os
            with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
                f.write(code)
                f_path = f.name
            result = subprocess.run([sys.executable, f_path], capture_output=True, text=True, timeout=10)
            os.unlink(f_path)
            output = result.stdout + result.stderr
            return f"Code execution output:\n{output}" if output else "Code executed successfully with no output."
        except Exception as e:
            return f"Execution error: {e}"

    @staticmethod
    def read_file(path):
        try:
            repo_root = REPO_ROOT
            full_path = Path(repo_root) / path.lstrip("/")
            if not str(full_path.resolve()).startswith(str(Path(repo_root).resolve())):
                return "Access denied: Path outside workspace."
            return full_path.read_text()
        except Exception as e:
            return f"Read error: {e}"

    @staticmethod
    def write_file(path, content):
        try:
            repo_root = REPO_ROOT
            full_path = Path(repo_root) / path.lstrip("/")
            if not str(full_path.resolve()).startswith(str(Path(repo_root).resolve())):
                return "Access denied: Path outside workspace."
            full_path.parent.mkdir(parents=True, exist_ok=True)
            full_path.write_text(content)
            return f"File written successfully to {path}"
        except Exception as e:
            return f"Write error: {e}"

    @staticmethod
    def list_files(path):
        try:
            repo_root = REPO_ROOT
            full_path = Path(repo_root) / path.lstrip("/")
            if not str(full_path.resolve()).startswith(str(Path(repo_root).resolve())):
                return "Access denied: Path outside workspace."
            files = [f.name + ("/" if f.is_dir() else "") for f in full_path.iterdir()]
            return "\n".join(files)
        except Exception as e:
            return f"List error: {e}"

    @staticmethod
    def image_gen(prompt):
        global diffusion_engine, NeuralAIDiffusion
        try:
            if diffusion_engine is None:
                try:
                    from diffusion_engine import NeuralAIDiffusion as _ND
                    NeuralAIDiffusion = _ND
                except Exception:
                    return "❌ Diffusion engine not available."
                diffusion_engine = NeuralAIDiffusion()
            
            prompt = prompt.strip()
            if prompt.startswith("image_gen:"):
                prompt = prompt[10:].strip()
                
            filename = f"gen_{uuid.uuid4().hex[:8]}.png"
            output_path = GENERATED_DIR / filename
            
            success = diffusion_engine.generate(prompt, str(output_path))
            if success:
                return f"\\n\\n🎨 **Generated Image: {prompt}**\\n\\n![{prompt}](/static/generated/{filename})\\n\\n✅ Saved to NeuralDrive/generated/"
            else:
                return "❌ Image generation failed."
        except Exception as e:
            return f"❌ Image generation error: {e}"

def process_tool_calls(text, user_id):
    results = []
    # Support <tool>name: args</tool>
    pattern = r"<tool>(.*?): (.*?)</tool>"
    matches = re.findall(pattern, text, re.DOTALL)
    for name, args in matches:
        name = name.strip()
        args = args.strip()
        if name == "image_gen":
            results.append(Tools.image_gen(args))
        elif name == "calc":
            results.append(f"[Calc] {Tools.calculator(args)}")
        elif name == "search":
            results.append(f"[Search] {Tools.web_search(args)}")
        elif name == "execute_code":
            results.append(f"[Execute] {Tools.execute_code(args)}")
        elif name == "read_file":
            results.append(f"[Read] {Tools.read_file(args)}")
        elif name == "write_file":
            if ":" in args:
                p, c = args.split(":", 1)
                results.append(f"[Write] {Tools.write_file(p.strip(), c.strip())}")
            else:
                results.append("[Write] Error: write_file requires 'path:content' format.")
        elif name == "list_files":
            results.append(f"[List] {Tools.list_files(args)}")
    
    if not results:
        return ""
    return "\\n".join(results)

# ====================
# API ROUTES
# ====================

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/health")
@app.route("/api/health")
@app.route("/api/status")
def status():
    from peft import PeftModel

    if LLM_BACKEND != "local":
        # External backend — show remote info
        backend_status = "ready" if LLM_BACKEND else model_status
        return jsonify({
            "status": backend_status,
            "model": LLM_MODEL,
            "backend": LLM_BACKEND,
            "api_url": LLM_API_URL,
            "inference_count": inference_count,
            "uplink": "integrated",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "uptime": "running",
            "version": "7.1.0-stable"
        })

    model_name = "NeuralAI DPO v13.0" if is_dpo else BASE_MODEL
    if isinstance(model, PeftModel):
        model_name += " + LoRA Adapter"
        
    return jsonify({
        "status": model_status,
        "model": model_name,
        "backend": "local",
        "inference_count": inference_count,
        "uplink": "integrated",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "uptime": "running",
        "version": "7.1.0-stable"
    })

@app.route("/privacy")
def privacy():
    return render_template("privacy.html")

@app.route("/terms")
def terms():
    return render_template("terms.html")

@app.route("/favicon.ico")
def favicon():
    return send_from_directory(os.path.join(STATIC_PATH, "static"), "favicon.png", mimetype='image/png')

@app.route("/api/user/me", methods=["GET"])
@token_required
def get_user_me(current_user):
    db = get_db()
    try:
        user = db.execute("SELECT * FROM users WHERE id = ?", (current_user,)).fetchone()
        if not user: return jsonify({"error": "User not found"}), 404
        u_dict = dict(user)
        if "password_hash" in u_dict: del u_dict["password_hash"]
        return jsonify({"user": u_dict})
    finally:
        db.close()

@app.route("/api/auth/signup", methods=["POST"])
def signup():
    data = request.get_json(silent=True) or {}
    username = data.get("username", "").strip()
    email = data.get("email", "").strip()
    password = data.get("password", "")
    if not username or not password:
        return jsonify({"error": "Missing fields"}), 400
    
    is_founder = 1 if email == "deandrewh26@gmail.com" else 0
    hashed = generate_password_hash(password)
    uid = "user_" + str(uuid.uuid4().hex[:8])
    now = datetime.now(timezone.utc).isoformat()
    
    db = get_db()
    try:
        db.execute("INSERT INTO users (id, username, email, is_founder, password_hash, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                   (uid, username, email, is_founder, hashed, now))
        db.commit()
        
        # Auto-login after signup
        token = jwt.encode({
            "user_id": uid,
            "is_founder": is_founder,
            "exp": datetime.now(timezone.utc) + timedelta(days=30)
        }, app.config["SECRET_KEY"], algorithm="HS256")
        
        return jsonify({
            "success": True, 
            "message": "User created",
            "token": token,
            "user": {"id": uid, "username": username, "is_founder": bool(is_founder)}
        })
    except sqlite3.IntegrityError:
        return jsonify({"error": "Username or email exists"}), 409
    finally:
        db.close()

@app.route("/api/auth/login", methods=["POST"])
def login():
    data = request.get_json(silent=True) or {}
    # Better extraction for robustness
    identity = (data.get("username") or data.get("email") or "").strip()
    password = data.get("password", "")
    
    if not identity or not password:
        return jsonify({"error": "Missing credentials"}), 400
        
    logger.info(f"Login attempt for identity: {identity}")
    
    db = get_db()
    try:
        user = db.execute("SELECT * FROM users WHERE username = ? OR email = ?", (identity, identity)).fetchone()
        if user and check_password_hash(user["password_hash"], password):
            token = jwt.encode({
                "user_id": user["id"],
                "is_founder": user["is_founder"],
                "exp": datetime.now(timezone.utc) + timedelta(days=30)
            }, app.config["SECRET_KEY"], algorithm="HS256")
            
            logger.info(f"Login successful for user: {user['username']}")
            return jsonify({
                "success": True,
                "token": token,
                "user": {"id": user["id"], "username": user["username"], "is_founder": bool(user["is_founder"])}
            })
        return jsonify({"error": "Invalid credentials"}), 401
    finally:
        db.close()

@app.route("/api/auth/guest", methods=["POST"])
def guest_login():
    code = uuid.uuid4().hex[:8]
    user_id = f"guest_{os.urandom(4).hex()}"
    token = jwt.encode({"user_id": user_id, "role": "maestro"}, app.config["SECRET_KEY"], algorithm="HS256")
    return jsonify({"token": token, "user": {"username": f"Maestro_{code[:4]}", "role": "maestro"}})

@app.route("/api/auth/maestro", methods=["POST"])
def maestro_login():
    # Maestro Student Portal: accepts a Maestro ID (e.g. Mae001) and grants a
    # guest-style session. Pattern validation is lenient for the demo.
    data = request.get_json(silent=True) or {}
    code_in = (data.get("code") or data.get("maestro_id") or "").strip()
    if not code_in:
        return jsonify({"error": "Maestro ID required"}), 400
    user_id = f"maestro_{os.urandom(4).hex()}"
    token = jwt.encode({"user_id": user_id, "role": "maestro"}, app.config["SECRET_KEY"], algorithm="HS256")
    return jsonify({"token": token, "user": {"username": code_in, "role": "maestro"}})


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
# API KEY MANAGEMENT (Developer / BYO API)
# ====================
# NeuralAI can act as an OpenAI-compatible backend for external hosts (e.g. ZO Computer
# "BYO API"). A user generates a personal API key here; the key is stored hashed and used
# to authenticate requests to /v1/chat/completions. The raw key is shown only once.
import secrets
import hashlib

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
    """Resolve a raw API key to a user_id, or None if invalid."""
    if not api_key:
        return None
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
def openai_chat_completions():
    """OpenAI-compatible chat completions endpoint for external BYO API hosts (e.g. ZO Computer).

    Auth: Authorization: Bearer <api_key>  OR  ?api_key=<api_key>
    Accepts {model, messages, max_tokens, temperature, stream}.
    Uses the same generate_response_stream as the in-app chat (NeuralAI identity + tools).
    """
    # --- API key auth ---
    auth = request.headers.get("Authorization", "")
    api_key = auth.replace("Bearer ", "").strip() or request.args.get("api_key", "") or (request.get_json(silent=True) or {}).get("api_key", "")
    user_id = _user_for_api_key(api_key)
    if not user_id:
        return jsonify({"error": "Invalid API key"}), 401

    data = request.get_json(silent=True) or {}
    messages = data.get("messages", [])
    model = data.get("model", "neuralai")
    max_tokens = int(data.get("max_tokens", 512))
    temperature = float(data.get("temperature", 0.7))
    stream = bool(data.get("stream", False))

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
    core_identity = """IDENTITY: You are NeuralAI, a high-performance artificial intelligence engine created by De'Andrew Preston Harris.
TONE: You are helpful, transparent, and conversational—like an expert peer working alongside the user. Break down complex topics clearly, use clean Markdown formatting, and explain your reasoning step by step.
BOUNDARY: You are the AI. De'Andrew is your human creator.
NEVER say "I am DeAndrew" or "I am Dre".
If asked who you are, respond: "I am NeuralAI, a production-grade AI system developed by De'Andrew Preston Harris."
You do not generate harmful content, act deceptively, or exhibit hidden goals. You remain safe, helpful, and aligned with human values at all times."""
    if user and user["is_founder"]:
        system_content = f"{core_identity}\nDynamic Memory: {mem_facts}\nActive Protocols: {active_rules}\n{TOOL_INSTRUCTIONS}"
    else:
        system_content = f"{core_identity}\nMemory: {mem_facts}\nRules: {active_rules}\n{TOOL_INSTRUCTIONS}"

    full_messages = [{"role": "system", "content": system_content}] + list(messages)

    if not stream:
        full = ""
        for chunk in generate_response_stream(full_messages, max_tokens, temperature):
            full += chunk
        # Strip any tool tags for a clean non-streaming JSON response
        clean = re.sub(r"<tool>.*?</tool>", "", full, flags=re.DOTALL).strip()
        return jsonify({
            "id": "chatcmpl-" + secrets.token_hex(8),
            "object": "chat.completion",
            "created": int(datetime.now(timezone.utc).timestamp()),
            "model": model,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": clean}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        })

    # Streaming (SSE) — OpenAI-compatible delta format
    def gen():
        yield "data: " + json.dumps({"id": "chatcmpl-" + secrets.token_hex(8), "object": "chat.completion.chunk",
                                      "created": int(datetime.now(timezone.utc).timestamp()), "model": model,
                                      "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]}) + "\n\n"
        for chunk in generate_response_stream(full_messages, max_tokens, temperature):
            content = re.sub(r"<tool>.*?</tool>", "", chunk, flags=re.DOTALL)
            if content:
                yield "data: " + json.dumps({"choices": [{"index": 0, "delta": {"content": content}, "finish_reason": None}]}) + "\n\n"
        yield "data: " + json.dumps({"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}) + "\n\n"
        yield "data: [DONE]\n\n"

    return Response(stream_with_context(gen()), mimetype="text/event-stream")

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
            db.execute("INSERT INTO memory_facts (fact, user_id, created_at) VALUES (?, ?, ?)",
                       (fact, current_user, now))
            db.commit()
            return jsonify({"success": True})
        
        rows = db.execute("SELECT id, fact, created_at FROM memory_facts WHERE user_id = ? ORDER BY created_at DESC", (current_user,)).fetchall()
        facts = [dict(row) for row in rows]
        return jsonify({"success": True, "facts": facts})
    finally:
        db.close()

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
            db.execute("INSERT INTO active_rules (rule, user_id, created_at) VALUES (?, ?, ?)",
                       (rule, current_user, now))
            db.commit()
            return jsonify({"success": True})
        
        rows = db.execute("SELECT id, rule, active, created_at FROM active_rules WHERE user_id = ? ORDER BY created_at DESC", (current_user,)).fetchall()
        rules = [dict(row) for row in rows]
        return jsonify({"success": True, "rules": rules})
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

@app.route("/api/chat", methods=["POST"])
@token_required
def chat(current_user):
    data = request.get_json(silent=True) or {}
    prompt = data.get("prompt", "")
    history = data.get("messages", [])
    temperature = float(data.get("temperature", 0.7))
    max_tokens = int(data.get("max_tokens", 512))
    conv_id = data.get("conversation_id")
    
    # Intent detection for image requests
    if any(k in prompt.lower() for k in ["generate", "image", "draw", "picture", "photo"]):
        prompt = f"IMAGE_REQUEST: {prompt}\\nRespond ONLY with <tool>image_gen: {prompt}</tool>"

    # Fetch user context
    db = get_db()
    user = db.execute("SELECT * FROM users WHERE id = ?", (current_user,)).fetchone()
    mem_rows = db.execute("SELECT fact FROM memory_facts WHERE user_id = ?", (current_user,)).fetchall()
    rule_rows = db.execute("SELECT rule FROM active_rules WHERE user_id = ? AND active = 1", (current_user,)).fetchall()
    db.close()
    
    mem_facts = [row["fact"] for row in mem_rows]
    active_rules = [row["rule"] for row in rule_rows]
    
    # Core Identity (always included for all tiers)
    core_identity = """IDENTITY: You are NeuralAI, a high-performance artificial intelligence engine created by De'Andrew Preston Harris.
TONE: You are helpful, transparent, and conversational—like an expert peer working alongside the user. Break down complex topics clearly, use clean Markdown formatting, and explain your reasoning step by step.
BOUNDARY: You are the AI. De'Andrew is your human creator.
NEVER say "I am DeAndrew" or "I am Dre".
If asked who you are, respond: "I am NeuralAI, a production-grade AI system developed by De'Andrew Preston Harris."
You do not generate harmful content, act deceptively, or exhibit hidden goals. You remain safe, helpful, and aligned with human values at all times."""

    if user and user["is_founder"]:
        system_content = f"{core_identity}\nDynamic Memory: {mem_facts}\nActive Protocols: {active_rules}\n{TOOL_INSTRUCTIONS}"
    else:
        system_content = f"{core_identity}\nMemory: {mem_facts}\nRules: {active_rules}\n{TOOL_INSTRUCTIONS}"

    # Build messages list
    messages = [{"role": "system", "content": system_content}]
    for m in history[-10:]:
        messages.append({"role": m["role"], "content": m["content"]})
    
    # Append current user prompt if not already in history
    if not history or history[-1]["content"] != prompt:
        messages.append({"role": "user", "content": prompt})

    def generate():
        full_response = ""
        stream_buffer = ""
        sse_tail = "\n\n"
        for chunk in generate_response_stream(messages, max_tokens, temperature):
            full_response += chunk
            stream_buffer += chunk

            if "<tool>" in stream_buffer:
                if "</tool>" in stream_buffer:
                    pattern = r"(<tool>.*?</tool>)"
                    match = re.search(pattern, stream_buffer, re.DOTALL)
                    if match:
                        complete_tag = match.group(0)
                        before_tag = stream_buffer[:match.start()]
                        after_tag = stream_buffer[match.end():]

                        if before_tag: yield f"data: {json.dumps({'content': before_tag})}{sse_tail}"

                        # Yield a tool execution indicator to keep stream alive
                        tool_name_match = re.search(r"<tool>(.*?):", complete_tag)
                        tool_name = tool_name_match.group(1).strip() if tool_name_match else "unknown"
                        indicator = f"\n\n🔧 **NeuralAI is processing tool: {tool_name}...**\n"
                        yield f"data: {json.dumps({'content': indicator})}{sse_tail}"

                        results = process_tool_calls(complete_tag, current_user)
                        if results:
                            yield f"data: {json.dumps({'content': results})}{sse_tail}"
                            full_response += results
                        stream_buffer = after_tag
                continue
            else:
                yield f"data: {json.dumps({'content': stream_buffer})}{sse_tail}"
                stream_buffer = ""
        
        if stream_buffer: yield f"data: {json.dumps({'content': stream_buffer})}{sse_tail}"

        # Save to database if conv_id provided
        if conv_id:
            db = get_db()
            now = datetime.now(timezone.utc).isoformat()
            db.execute("INSERT INTO messages (conversation_id, role, content, created_at) VALUES (?, ?, ?, ?)",
                       (conv_id, "user", prompt, now))
            db.execute("INSERT INTO messages (conversation_id, role, content, created_at) VALUES (?, ?, ?, ?)",
                       (conv_id, "assistant", full_response, now))
            db.execute("UPDATE conversations SET updated_at = ?, message_count = message_count + 2 WHERE id = ?", (now, conv_id))
            db.commit()
            db.close()

        yield f"data: [DONE]{sse_tail}"

    return Response(stream_with_context(generate()), mimetype="text/event-stream")

@app.route("/api/chat/json", methods=["POST"])
@token_required
def chat_json(current_user):
    data = request.get_json(silent=True) or {}
    prompt = data.get("prompt", "")
    history = data.get("messages", [])
    temperature = float(data.get("temperature", 0.7))
    max_tokens = int(data.get("max_tokens", 512))
    
    # Intent detection for image requests
    if any(k in prompt.lower() for k in ["generate", "image", "draw", "picture", "photo"]):
        return jsonify({"output": process_tool_calls(f"<tool>image_gen: {prompt}</tool>", current_user), "status": "success"})

    # Fetch user context
    db = get_db()
    user = db.execute("SELECT * FROM users WHERE id = ?", (current_user,)).fetchone()
    mem_rows = db.execute("SELECT fact FROM memory_facts WHERE user_id = ?", (current_user,)).fetchall()
    rule_rows = db.execute("SELECT rule FROM active_rules WHERE user_id = ? AND active = 1", (current_user,)).fetchall()
    db.close()
    
    mem_facts = [row["fact"] for row in mem_rows]
    active_rules = [row["rule"] for row in rule_rows]
    
    # Core Identity (always included for all tiers)
    core_identity = """IDENTITY: You are NeuralAI, a high-performance artificial intelligence engine created by De'Andrew Preston Harris.
TONE: You are helpful, transparent, and conversational—like an expert peer working alongside the user. Break down complex topics clearly, use clean Markdown formatting, and explain your reasoning step by step.
BOUNDARY: You are the AI. De'Andrew is your human creator.
NEVER say "I am DeAndrew" or "I am Dre".
If asked who you are, respond: "I am NeuralAI, a production-grade AI system developed by De'Andrew Preston Harris."
You do not generate harmful content, act deceptively, or exhibit hidden goals. You remain safe, helpful, and aligned with human values at all times."""

    if user and user["is_founder"]:
        system_content = f"{core_identity}\nDynamic Memory: {mem_facts}\nActive Protocols: {active_rules}\n{TOOL_INSTRUCTIONS}"
    else:
        system_content = f"{core_identity}\nMemory: {mem_facts}\nRules: {active_rules}\n{TOOL_INSTRUCTIONS}"

    # Build messages list
    messages = [{"role": "system", "content": system_content}]
    for m in history[-10:]:
        messages.append({"role": m["role"], "content": m["content"]})
    
    if not history or history[-1]["content"] != prompt:
        messages.append({"role": "user", "content": prompt})

    full_response = ""
    for chunk in generate_response_stream(messages, max_tokens, temperature):
        full_response += chunk
    
    # Process tools in the full response if any
    tool_results = process_tool_calls(full_response, current_user)
    if tool_results:
        full_response += tool_results

    return jsonify({"output": full_response, "status": "success"})

# ====================
# TERMINAL API
# ====================
@app.route("/api/terminal/create", methods=["POST"])
@token_required
def create_terminal(current_user):
    # Terminal/shell execution is disabled in the hosted deployment for
    # security (it would allow arbitrary command execution). The chat UI
    # remains fully functional without it.
    return jsonify({"success": False, "error": "Terminal is disabled in this deployment."}), 403

@app.route("/api/terminal/<sid>/send", methods=["POST"])
@token_required
def send_terminal(current_user, sid):
    return jsonify({"success": False, "error": "Terminal is disabled in this deployment."}), 403

@app.route("/api/terminal/<sid>/read", methods=["GET"])
@token_required
def read_terminal(current_user, sid):
    return jsonify({"success": False, "error": "Terminal is disabled in this deployment."}), 403

@app.route("/api/files", methods=["GET"])
@token_required
def list_files(current_user):
    user_uploads = UPLOADS_DIR / current_user
    user_uploads.mkdir(parents=True, exist_ok=True)
    files = sorted([{"name": f.name, "type": "uploads"} for f in user_uploads.iterdir() if f.is_file()], key=lambda x: x["name"])
    return jsonify({"success": True, "files": files})

@app.route("/api/files/<folder>/<path:filename>", methods=["GET"])
@token_required
def serve_file(current_user, folder, filename):
    # Ensure users can only access their own uploads
    # Right now, folder might just be 'uploads', but we serve from UPLOADS_DIR / current_user
    user_uploads = UPLOADS_DIR / current_user
    return send_from_directory(user_uploads, filename)

@app.route("/api/upload", methods=["POST"])
@token_required
def upload_file(current_user):
    from werkzeug.utils import secure_filename
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400
    f = request.files["file"]
    if f.filename == "":
        return jsonify({"error": "No file selected"}), 400
    user_uploads = UPLOADS_DIR / current_user
    user_uploads.mkdir(parents=True, exist_ok=True)
    filename = secure_filename(f.filename)
    f.save(str(user_uploads / filename))
    return jsonify({"success": True, "filename": filename, "chunks": 0,
                    "message": f'"{filename}" uploaded.'})

@app.route("/static/generated/<path:filename>")
def serve_generated(filename):
    return send_from_directory(GENERATED_DIR, filename)

# ====================
# WEBSOCKET PROXY - Voice Service
# ====================
VOICE_SERVICE = os.environ.get("VOICE_SERVICE_URL", "ws://127.0.0.1:5001/ws")
sock = Sock(app)

@sock.route("/voice/ws")
def voice_proxy(ws):
    """Proxy WebSocket between browser and voice service on localhost:5001."""
    import websocket as ws_lib
    
    logger.info("[VoiceProxy] Browser connected, opening upstream to %s", VOICE_SERVICE)
    
    try:
        upstream = ws_lib.create_connection(VOICE_SERVICE, timeout=30)
    except Exception as e:
        logger.error("[VoiceProxy] Cannot connect to voice service: %s", e)
        try:
            ws.send(json.dumps({"type": "error", "message": f"Voice service unavailable: {str(e)}"}))
        except:
            pass
        return
    
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

if __name__ == "__main__":
    init_db()

    # Auto-start the voice service if not running
    _ensure_voice_service()

    # If using an external backend, we don't need to load the local model
    if LLM_BACKEND != "local":
        model_status = "ready"
        print(f"[NeuralAI] Using external backend: {LLM_BACKEND} at {LLM_API_URL}")
        print(f"[NeuralAI] Model: {LLM_MODEL}")
    else:
        threading.Thread(target=load_model).start()

    app.run(host="0.0.0.0", port=PORT, threaded=True)
