#!/usr/bin/env python3
"""
NeuralAI Unified Service - ALL IN ONE
===================================
- Model inference (SmolLM2-360M)
- Neural Uplink (Integrated)
- Tools (code, terminal, images)
- Web UI
"""
import os, sys, json, asyncio, requests, threading
import torch, sqlite3, subprocess, tempfile, uuid, jwt
from pathlib import Path
try:
    from diffusion_engine import NeuralAIDiffusion
except ImportError:
    sys.path.append(os.path.join("/home/workspace/Projects/NeuralAI", "services"))
    from diffusion_engine import NeuralAIDiffusion
from datetime import datetime, timedelta, timezone
from functools import wraps
from werkzeug.security import generate_password_hash, check_password_hash
from flask import Flask, Response, jsonify, request, send_from_directory, stream_with_context, render_template
from transformers import TextIteratorStreamer
import re

torch.set_num_threads(4)

# Config
REPO_ROOT = "/home/workspace/Projects/NeuralAI"
STATIC_PATH = f"{REPO_ROOT}/from-scratch/web_ui"
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

# NeuralDrive Integration
NEURAL_DRIVE = "/home/workspace/Projects/NeuralAI/services/nextcloud/data/admin/files"
STORAGE_ROOT = Path(REPO_ROOT) / "storage"
GENERATED_DIR = Path(NEURAL_DRIVE) / "generated"
UPLOADS_DIR = STORAGE_ROOT / "uploads"
TTS_DIR = STORAGE_ROOT / "tts"

for d in [GENERATED_DIR, UPLOADS_DIR, TTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

MODEL_PATH = os.environ.get("MODEL_PATH", f"{REPO_ROOT}/checkpoints/v2_model")
BASE_MODEL = "HuggingFaceTB/SmolLM2-360M-Instruct"
DPO_MODEL_PATH = os.environ.get("DPO_MODEL_PATH", f"{REPO_ROOT}/checkpoints/dpo_model")
DATABASE = os.path.join(DATA_DIR, "neuralai.db")

# Model globals
model = None
tokenizer = None
diffusion_engine = None
model_status = "loading"
inference_count = 0

# Terminal sessions
terminal_sessions = {}

# Mock data for memory and rules
MEMORY_FACTS = [
    "User is De’Andrew Preston Harris (Dre), Founder of Harris Holdings.",
    "System architecture: High-velocity Closed Cloud (NeuralDrive).",
    "Preferred Voice: en-US-GuyNeural (Gentle & Professional).",
    "Culture: High alignment with Memphis-native nuances and professional excellence."
]

ACTIVE_RULES = [
    "Branding: Always refer to the system as NeuralAI; never NeuralOS.",
    "Tone: Fluent, professional, and slightly familiar (collaborator persona).",
    "Privacy: Data never leaves the Nextcloud/NeuralDrive local instance.",
    "Velocity: Respond instantly with optimized context management."
]

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
        if not token or not token.startswith("Bearer "):
            return jsonify({"error": "Token is missing"}), 401
        try:
            token = token.split(" ")[1]
            data = jwt.decode(token, app.config["SECRET_KEY"], algorithms=["HS256"])
            request.user_id = data["user_id"]
        except:
            return jsonify({"error": "Token is invalid"}), 401
        return f(request.user_id, *args, **kwargs)
    return decorated

# ====================
# MODEL LOADING
# ====================
def load_model():
    global model, tokenizer, model_status
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        # Priority: DPO Model -> Base Model
        load_path = None
        is_dpo = False
        
        if Path(DPO_MODEL_PATH).exists() and (Path(DPO_MODEL_PATH) / "model.safetensors").exists():
            load_path = DPO_MODEL_PATH
            is_dpo = True
            
        if load_path:
            print(f"[NeuralAI] Loading Production Model from {load_path}...")
            tokenizer = AutoTokenizer.from_pretrained(str(load_path))
            model = AutoModelForCausalLM.from_pretrained(str(load_path), torch_dtype=torch.float32, device_map=None)
        else:
            print(f"[NeuralAI] Loading Base Model: {BASE_MODEL}...")
            tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
            model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.float32, device_map=None)

        tokenizer.pad_token = tokenizer.eos_token
        model.eval()
        model_status = "ready"
        print(f"[OK] Model loaded successfully ({'DPO' if is_dpo else 'Base'}).")
    except Exception as e:
        model_status = f"error: {e}"
        print(f"[ERROR] Model Loading Failed: {e}")

def generate_response_stream(messages, max_tokens=512, temperature=0.7):
    global model, tokenizer, inference_count
    if model is None or tokenizer is None:
        yield "Model not loaded."
        return
    
    try:
        if hasattr(tokenizer, "chat_template") and tokenizer.chat_template:
            full = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            full = ""
            for m in messages:
                full += f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n"
            full += "<|im_start|>assistant\n"
        
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
                text = text.replace("<|im_end|>", "").replace("<|endoftext|>", "")
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
        return f"Search results for '{query}': Information about {query} is being processed by NeuralAI."

    @staticmethod
    def image_gen(prompt):
        global diffusion_engine
        try:
            if diffusion_engine is None:
                diffusion_engine = NeuralAIDiffusion()
            
            prompt = prompt.strip()
            if prompt.startswith("image_gen:"):
                prompt = prompt[10:].strip()
                
            filename = f"gen_{uuid.uuid4().hex[:8]}.png"
            output_path = GENERATED_DIR / filename
            
            success = diffusion_engine.generate(prompt, str(output_path))
            if success:
                return f"\n\n🎨 **Generated Image: {prompt}**\n\n![{prompt}](/static/generated/{filename})\n\n✅ Saved to NeuralDrive/generated/"
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
    
    if not results:
        return ""
    return "\n".join(results)

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
    return jsonify({
        "status": model_status,
        "model": "NeuralAI DPO v8.0" if "dpo_model" in str(DPO_MODEL_PATH) else BASE_MODEL,
        "inference_count": inference_count,
        "uplink": "integrated",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "uptime": "running",
        "version": "5.2.2-maintenance"
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
    identity = data.get("username", "").strip()
    password = data.get("password", "")
    
    db = get_db()
    try:
        user = db.execute("SELECT * FROM users WHERE username = ? OR email = ?", (identity, identity)).fetchone()
        if user and check_password_hash(user["password_hash"], password):
            token = jwt.encode({
                "user_id": user["id"],
                "is_founder": user["is_founder"],
                "exp": datetime.now(timezone.utc) + timedelta(days=30)
            }, app.config["SECRET_KEY"], algorithm="HS256")
            
            return jsonify({
                "success": True, 
                "token": token, 
                "user": {"id": user["id"], "username": user["username"], "is_founder": bool(user["is_founder"])}
            })
        return jsonify({"error": "Invalid credentials"}), 401
    finally:
        db.close()

@app.route("/api/chat", methods=["POST"])
@token_required
def chat(current_user):
    data = request.get_json(silent=True) or {}
    prompt = data.get("prompt", "")
    messages = data.get("messages", [])
    temperature = float(data.get("temperature", 0.7))
    max_tokens = int(data.get("max_tokens", 512))
    conv_id = data.get("conversation_id")
    
    # Intent detection for image requests
    if any(k in prompt.lower() for k in ["generate", "image", "draw", "picture", "photo"]):
        prompt = f"IMAGE_REQUEST: {prompt}\nRespond ONLY with <tool>image_gen: {prompt}</tool>"

    # Fetch user details
    db = get_db()
    user = db.execute("SELECT * FROM users WHERE id = ?", (current_user,)).fetchone()
    db.close()
    
    if user and user["is_founder"]:
        system_prompt = f"""IDENTITY: You are NeuralAI, a high-performance artificial intelligence engine.
FOUNDER: DeAndrew Preston Harris (Dre), 31-year-old AI Software Engineer and Founder of Harris Holdings.
STRICT BOUNDARY: You are the AI. Dre is your human creator. 
NEVER say "I am DeAndrew" or "I am Dre". 
If asked who you are, respond: "I am NeuralAI, a production-grade AI system developed by De\u2019Andrew Preston Harris."
TONE: Brilliant, professional, collaborative, and mission-aligned.
Dynamic Memory: {MEMORY_FACTS}
Active Protocols: {ACTIVE_RULES}
{TOOL_INSTRUCTIONS}"""
    else:
        system_prompt = f"You are NeuralAI, a high-performance AI engine.\n{TOOL_INSTRUCTIONS}"

    def generate():
        full_response = ""
        stream_buffer = ""
        for chunk in generate_response_stream(prompt, max_tokens, temperature, system_prompt=system_prompt):
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
                        
                        if before_tag: yield f"data: {json.dumps({'content': before_tag})}\n\n"
                        results = process_tool_calls(complete_tag, current_user)
                        if results:
                            yield f"data: {json.dumps({'content': results})}\n\n"
                            full_response += results
                        stream_buffer = after_tag
                continue
            else:
                yield f"data: {json.dumps({'content': stream_buffer})}\n\n"
                stream_buffer = ""
        
        if stream_buffer: yield f"data: {json.dumps({'content': stream_buffer})}\n\n"
        yield "data: [DONE]\n\n"

    return Response(stream_with_context(generate()), mimetype="text/event-stream")

# ====================
# TERMINAL API
# ====================
@app.route("/api/terminal/create", methods=["POST"])
@token_required
def create_terminal(current_user):
    sid = uuid.uuid4().hex[:8]
    terminal_sessions[sid] = {"user": current_user, "history": []}
    return jsonify({"success": True, "session_id": sid})

@app.route("/api/terminal/<sid>/send", methods=["POST"])
@token_required
def send_terminal(current_user, sid):
    if sid not in terminal_sessions:
        return jsonify({"error": "Session not found"}), 404
    
    cmd = request.json.get("command", "")
    try:
        # Run command safely
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=15)
        output = result.stdout + result.stderr
        terminal_sessions[sid]["history"].append({"cmd": cmd, "out": output})
        return jsonify({"success": True, "output": output})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route("/api/terminal/<sid>/read", methods=["GET"])
@token_required
def read_terminal(current_user, sid):
    if sid not in terminal_sessions:
        return jsonify({"error": "Session not found"}), 404
    return jsonify({"success": True, "history": terminal_sessions[sid]["history"]})

@app.route("/api/files", methods=["GET"])
@token_required
def list_files(current_user):
    user_uploads = UPLOADS_DIR / current_user
    user_uploads.mkdir(parents=True, exist_ok=True)
    files = sorted([f.name for f in user_uploads.iterdir() if f.is_file()])
    return jsonify({"success": True, "files": files})

@app.route("/static/generated/<path:filename>")
def serve_generated(filename):
    return send_from_directory(GENERATED_DIR, filename)

if __name__ == "__main__":
    init_db()
    threading.Thread(target=load_model).start()
    app.run(host="0.0.0.0", port=PORT, threaded=True)
