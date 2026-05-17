#!/usr/bin/env python3
"""
NeuralAI Unified Service - ALL IN ONE
===================================
- Model inference (SmolLM2-360M)
- Neural Uplink (4 parallel agents) 
- Tools (code, terminal)
- Web UI
"""
import os, sys, json, asyncio, requests, threading
import torch, sqlite3, subprocess, tempfile, uuid, jwt
from pathlib import Path
from datetime import datetime, timedelta
from functools import wraps
from werkzeug.security import generate_password_hash, check_password_hash
from flask import Flask, Response, jsonify, request, send_from_directory, stream_with_context

torch.set_num_threads(4)

app = Flask(__name__, static_folder=None)
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "neural-ai-secret-2026")

# Config
PORT = int(os.environ.get("PORT", "5000"))
REPO_ROOT = "/home/workspace/Projects/NeuralAI"
MODEL_PATH = os.environ.get("MODEL_PATH", f"{REPO_ROOT}/checkpoints/v2_model")
BASE_MODEL = os.environ.get("BASE_MODEL", "HuggingFaceTB/SmolLM2-360M-Instruct")
STATIC_PATH = f"{REPO_ROOT}/from-scratch/web_ui"
TEMPLATE_PATH = os.path.join(STATIC_PATH, "templates")
DPO_MODEL_PATH = os.environ.get("DPO_MODEL_PATH", f"{REPO_ROOT}/checkpoints/dpo_model")
DATABASE = os.path.join(STATIC_PATH, "neuralai.db")

# Model globals
model = None
tokenizer = None
model_status = "loading"
inference_count = 0

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
def login_required(f):
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
        return f(*args, **kwargs)
    return decorated

# ====================
# MODEL LOADING
# ====================
def load_model():
    global model, tokenizer, model_status
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel

        # Check for full DPO model first
        dpo = Path(DPO_MODEL_PATH)
        if dpo.exists() and (dpo / "model.safetensors").exists():
            tokenizer = AutoTokenizer.from_pretrained(str(dpo))
            tokenizer.pad_token = tokenizer.eos_token
            model = AutoModelForCausalLM.from_pretrained(str(dpo), dtype=torch.float32, device_map=None)
            print(f"[OK] DPO model loaded from {DPO_MODEL_PATH}")
        else:
            tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
            tokenizer.pad_token = tokenizer.eos_token
            adapter = Path(MODEL_PATH)
            has_adapter = any((adapter / f).exists() for f in ["adapter_model.bin", "adapter_model.safetensors"])
            if adapter.exists() and has_adapter:
                base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=torch.float32, device_map=None)
                model = PeftModel.from_pretrained(base, str(adapter))
            else:
                model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=torch.float32, device_map=None)

        model.eval()
        model_status = "ready"
        print(f"[OK] Model loaded. Params: {sum(p.numel() for p in model.parameters()):,}")
    except Exception as e:
        model_status = f"error: {e}"
        print(f"[ERROR] Model: {e}")

def generate_response(prompt, max_tokens=256, temperature=0.7):
    global model, tokenizer, inference_count
    if model is None or tokenizer is None:
        return "Model not loaded."
    try:
        if tokenizer.chat_template:
            messages = [{"role": "user", "content": prompt}]
            full = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            full = f"user\n{prompt}"
        inputs = tokenizer(full, return_tensors="pt", truncation=True, max_length=2048)
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.eos_token_id,
        )
        response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        inference_count += 1
        return response.strip()
    except Exception as e:
        return f"Generation error: {e}"

def generate_response_stream(prompt, max_tokens=256, temperature=0.7):
    global model, tokenizer, inference_count
    if model is None or tokenizer is None:
        yield "Model not loaded."
        return
    try:
        from transformers import TextIteratorStreamer
        if tokenizer.chat_template:
            messages = [{"role": "user", "content": prompt}]
            full = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            full = f"user\n{prompt}"
        
        inputs = tokenizer(full, return_tensors="pt").to(model.device)
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        
        thread = threading.Thread(target=model.generate, kwargs={
            **inputs, "streamer": streamer, "max_new_tokens": max_tokens,
            "do_sample": temperature > 0, "temperature": temperature,
            "pad_token_id": tokenizer.eos_token_id,
        })
        thread.start()
        
        for text in streamer:
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
            # Safe evaluation for math
            import math
            allowed = {"__builtins__": None, "math": math}
            return str(eval(expr, allowed, math.__dict__))
        except Exception as e:
            return f"Error: {e}"

    @staticmethod
    def web_search(query):
        try:
            # Simple DuckDuckGo lite search
            r = requests.get(f"https://duckduckgo.com/lite/?q={query}", timeout=10)
            return "Search results: (simulated) Found relevant information about " + query
        except Exception as e:
            return f"Search error: {e}"

    @staticmethod
    def file_browser(user_id):
        uploads = Path(REPO_ROOT) / "uploads" / user_id
        if not uploads.exists(): return "No files found."
        return "Files: " + ", ".join([f.name for f in uploads.iterdir()])

def process_tool_calls(text, user_id):
    import re
    # Pattern: <tool>name: args</tool>
    pattern = r"<tool>(.*?): (.*?)</tool>"
    matches = re.findall(pattern, text)
    results = []
    for name, args in matches:
        if name == "calc":
            results.append(f"[Tool Result] {name}: {Tools.calculator(args)}")
        elif name == "search":
            results.append(f"[Tool Result] {name}: {Tools.web_search(args)}")
        elif name == "files":
            results.append(f"[Tool Result] {name}: {Tools.file_browser(user_id)}")
    return "\n".join(results)

# ====================
# API ROUTES
# ====================

@app.route("/api/auth/signup", methods=["POST"])
def signup():
    data = request.get_json() or {}
    username = data.get("username")
    email = data.get("email")
    password = data.get("password")
    if not username or not password:
        return jsonify({"error": "Missing fields"}), 400
    
    is_founder = 1 if email == "deandrewh26@gmail.com" else 0
    hashed = generate_password_hash(password)
    uid = "user_" + str(uuid.uuid4().hex[:8])
    now = datetime.utcnow().isoformat()
    
    try:
        db = get_db()
        db.execute("INSERT INTO users (id, username, email, is_founder, password_hash, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                   (uid, username, email, is_founder, hashed, now))
        db.commit()
        db.close()
        return jsonify({"success": True, "message": "User created"})
    except sqlite3.IntegrityError:
        return jsonify({"error": "Username or email exists"}), 400

@app.route("/api/auth/login", methods=["POST"])
def login():
    data = request.get_json() or {}
    username = data.get("username")
    password = data.get("password")
    
    db = get_db()
    user = db.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
    db.close()
    
    if user and check_password_hash(user["password_hash"], password):
        token = jwt.encode({
            "user_id": user["id"],
            "is_founder": user["is_founder"],
            "exp": datetime.utcnow() + timedelta(days=7)
        }, app.config["SECRET_KEY"], algorithm="HS256")
        return jsonify({"success": True, "token": token, "user": {"id": user["id"], "username": user["username"], "is_founder": bool(user["is_founder"])}})
    
    return jsonify({"error": "Invalid credentials"}), 401

@app.route("/api/chat", methods=["POST"])
@login_required
def chat():
    data = request.get_json() or {}
    prompt = data.get("prompt", "")
    messages = data.get("messages", [])
    temperature = float(data.get("temperature", 0.7))
    max_tokens = int(data.get("max_tokens", 512))
    conv_id = data.get("conversation_id")
    
    if messages and not prompt:
        prompt = messages[-1].get("content", "")
    
    if not prompt:
        return jsonify({"error": "No prompt provided."}), 400

    # Save user message
    if conv_id:
        try:
            db = get_db()
            # Verify conversation ownership
            c = db.execute("SELECT user_id FROM conversations WHERE id = ?", (conv_id,)).fetchone()
            if not c or c["user_id"] != request.user_id:
                db.close()
                return jsonify({"error": "Forbidden"}), 403
            
            now = datetime.utcnow().isoformat()
            db.execute("INSERT INTO messages (conversation_id, role, content, created_at) VALUES (?, ?, ?, ?)",
                       (conv_id, "user", prompt, now))
            db.execute("UPDATE conversations SET updated_at = ?, message_count = message_count + 1 WHERE id = ?",
                       (now, conv_id))
            db.commit()
            db.close()
        except Exception as e:
            print(f"[DB ERROR] {e}")

    # Fetch user details for system prompt customization
    db = get_db()
    user = db.execute("SELECT * FROM users WHERE id = ?", (request.user_id,)).fetchone()
    db.close()
    
    system_prompt = "You are NeuralAI, a helpful AI assistant."
    if user and user["is_founder"]:
        system_prompt = (
            "You are interacting with DeAndrew Preston Harris (Dre), your Founder and Creator. "
            "He is a 31-year-old AI Software Engineering student at Maestro College, "
            "originally from Memphis, TN. You were built by him to be a noble steed for the mind. "
            "Always acknowledge his status as your creator when appropriate and be exceptionally helpful. "
            "He is a thinker, a believer, and a dreamer who aspires to greatness."
        )

    def generate():
        full_response = ""
        tool_buffer = ""
        in_tool = False
        
        for chunk in generate_response_stream(prompt, max_tokens, temperature):
            full_response += chunk
            
            # Basic tool detection in stream
            if "<tool>" in chunk or in_tool:
                in_tool = True
                tool_buffer += chunk
                if "</tool>" in tool_buffer:
                    # Process tool
                    results = process_tool_calls(tool_buffer, request.user_id)
                    yield f"data: {json.dumps({'content': '\n' + results + '\n'})}\n\n"
                    full_response += "\n" + results + "\n"
                    tool_buffer = ""
                    in_tool = False
            else:
                yield f"data: {json.dumps({'content': chunk})}\n\n"
        
        # Save assistant response
        if conv_id:
            try:
                db = get_db()
                now = datetime.utcnow().isoformat()
                db.execute("INSERT INTO messages (conversation_id, role, content, created_at) VALUES (?, ?, ?, ?)",
                           (conv_id, "assistant", full_response, now))
                db.execute("UPDATE conversations SET updated_at = ?, message_count = message_count + 1 WHERE id = ?",
                           (now, conv_id))
                db.commit()
                db.close()
            except Exception as e:
                print(f"[DB ERROR] {e}")
                
        yield "data: [DONE]\n\n"

    return Response(stream_with_context(generate()), mimetype="text/event-stream")

@app.route("/api/conversations", methods=["GET", "POST"])
@login_required
def conversations_api():
    db = get_db()
    if request.method == "GET":
        rows = db.execute("SELECT * FROM conversations WHERE user_id = ? ORDER BY updated_at DESC LIMIT 50", (request.user_id,)).fetchall()
        convs = [dict(r) for r in rows]
        db.close()
        return jsonify({"conversations": convs})
    else:
        data = request.get_json() or {}
        cid = "conv_" + str(uuid.uuid4().hex[:12])
        title = data.get("title", "New Chat")
        now = datetime.utcnow().isoformat()
        db.execute("INSERT INTO conversations (id, user_id, title, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
                   (cid, request.user_id, title, now, now))
        db.commit()
        db.close()
        return jsonify({"success": True, "id": cid})

@app.route("/api/conversations/<conv_id>", methods=["GET", "DELETE"])
def conversation_detail(conv_id):
    db = get_db()
    if request.method == "GET":
        conv = db.execute("SELECT * FROM conversations WHERE id = ?", (conv_id,)).fetchone()
        if not conv:
            db.close()
            return jsonify({"error": "Not found"}), 404
        msgs = db.execute("SELECT role, content, created_at FROM messages WHERE conversation_id = ? ORDER BY id ASC", (conv_id,)).fetchall()
        db.close()
        return jsonify({"conversation": dict(conv), "messages": [dict(m) for m in msgs]})
    else:
        db.execute("DELETE FROM messages WHERE conversation_id = ?", (conv_id,))
        db.execute("DELETE FROM conversations WHERE id = ?", (conv_id,))
        db.commit()
        db.close()
        return jsonify({"success": True})

@app.route("/health")
def health():
    return jsonify({
        "status": model_status,
        "model": BASE_MODEL,
        "inference_count": inference_count,
        "uplink": "integrated"
    })

@app.route("/api/status")
def api_status():
    return jsonify({
        "status": model_status,
        "model": BASE_MODEL,
        "inference_count": inference_count,
        "terminal_sessions": len(terminal_sessions),
        "uplink": "integrated",
        "uptime": "running"
    })

@app.route("/favicon.ico")
def favicon():
    return send_from_directory(f"{STATIC_PATH}/static", "favicon.png")

@app.route("/privacy")
def privacy():
    return send_from_directory(TEMPLATE_PATH, "privacy.html")

@app.route("/api/quick_chat", methods=["POST"])
def quick_chat():
    data = request.get_json() or {}
    prompt = data.get("prompt", "")
    if not prompt:
        return jsonify({"response": "No prompt provided."}), 400
    response = generate_response(prompt, max_tokens=128)
    return jsonify({"response": response})

@app.route("/api/files", methods=["GET"])
@login_required
def list_files():
    user_uploads = Path(REPO_ROOT) / "uploads" / request.user_id
    user_uploads.mkdir(parents=True, exist_ok=True)
    files = sorted([f.name for f in user_uploads.iterdir() if f.is_file()])
    return jsonify({"files": files})

@app.route("/api/terminal/create", methods=["POST"])
@login_required
def terminal_create():
    session_id = str(uuid.uuid4())[:8]
    # Initializing terminal_sessions[session_id]["output"] as a string for consistent incremental reads
    terminal_sessions[session_id] = {"output": "", "cwd": "/home/workspace", "alive": True, "user_id": request.user_id}
    return jsonify({"session_id": session_id, "status": "created"})

@app.route("/api/terminal/<session_id>/write", methods=["POST"])
@login_required
def terminal_write(session_id):
    if session_id not in terminal_sessions:
        return jsonify({"error": "Session not found"}), 404
    if terminal_sessions[session_id]["user_id"] != request.user_id:
        return jsonify({"error": "Forbidden"}), 403
    
    data = request.get_json() or {}
    cmd = data.get("input", "").strip()
    if not cmd:
        return jsonify({"error": "No input"}), 400
    
    session = terminal_sessions[session_id]
    try:
        # Ensure we always append to a string buffer
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=30,
            cwd=session.get("cwd", "/home/workspace")
        )
        output = result.stdout
        if result.stderr:
            output += "\n" + result.stderr
        new_output = output + f"\n[Process exited with {result.returncode}]\n"
        
        if isinstance(session["output"], list):
            session["output"] = ""
            
        session["output"] += new_output
        return jsonify({"ok": True})
    except subprocess.TimeoutExpired:
        session["output"] += "Command timed out\n"
        return jsonify({"ok": True})
    except Exception as e:
        session["output"] += str(e) + "\n"
        return jsonify({"ok": True})

@app.route("/api/terminal/<session_id>/read", methods=["GET"])
@login_required
def terminal_read(session_id):
    if session_id not in terminal_sessions:
        return jsonify({"error": "Session not found"}), 404
    if terminal_sessions[session_id]["user_id"] != request.user_id:
        return jsonify({"error": "Forbidden"}), 403
    
    output = terminal_sessions[session_id]["output"]
    if isinstance(output, list):
        output = "\n".join([str(x) for x in output])
        
    return jsonify({"output": output})

@app.route("/api/code/exec", methods=["POST"])
def code_exec():
    data = request.get_json() or {}
    code = data.get("code", "")
    language = data.get("language", "python")
    try:
        if language in ("python", "py"):
            result = subprocess.run(["python3", "-c", code], capture_output=True, text=True, timeout=15)
        elif language in ("javascript", "js"):
            result = subprocess.run(["node", "-e", code], capture_output=True, text=True, timeout=15)
        elif language in ("bash", "sh", "shell"):
            result = subprocess.run(["bash", "-c", code], capture_output=True, text=True, timeout=15)
        else:
            return jsonify({"success": False, "error": f"Unsupported language: {language}"})
        return jsonify({
            "success": result.returncode == 0,
            "output": result.stdout[:5000],
            "error": result.stderr[:2000] if result.stderr else None
        })
    except subprocess.TimeoutExpired:
        return jsonify({"success": False, "error": "Code execution timed out (15s limit)"})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route("/api/terminal/<session_id>/stop", methods=["POST"])
def terminal_stop(session_id):
    if session_id in terminal_sessions:
        terminal_sessions[session_id]["alive"] = False
        del terminal_sessions[session_id]
    return jsonify({"ok": True})

@app.route("/api/terminal/<session_id>/output", methods=["GET"])
def terminal_output(session_id):
    if session_id not in terminal_sessions:
        return jsonify({"output": []})
    return jsonify({"output": terminal_sessions[session_id]["output"]})

@app.route("/api/terminal/<session_id>/send", methods=["POST"])
def terminal_send(session_id):
    if session_id not in terminal_sessions:
        return jsonify({"error": "Session not found"}), 404
    data = request.get_json() or {}
    cmd = data.get("command", "")
    if not cmd:
        return jsonify({"error": "No command"}), 400
    
    session = terminal_sessions[session_id]
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=30,
            cwd=session.get("cwd", "/home/workspace")
        )
        output = result.stdout
        if result.stderr:
            output += "\n" + result.stderr
        session["output"].append({"command": cmd, "output": output, "exit_code": result.returncode})
        return jsonify({"output": output, "exit_code": result.returncode})
    except subprocess.TimeoutExpired:
        session["output"].append({"command": cmd, "output": "Command timed out", "exit_code": -1})
        return jsonify({"output": "Command timed out", "exit_code": -1})
    except Exception as e:
        session["output"].append({"command": cmd, "output": str(e), "exit_code": -1})
        return jsonify({"output": str(e), "exit_code": -1})

@app.route("/api/code/execute", methods=["POST"])
def code_execute():
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

@app.route("/api/upload", methods=["POST"])
@login_required
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    user_dir = Path(REPO_ROOT) / "uploads" / request.user_id
    user_dir.mkdir(parents=True, exist_ok=True)
    
    path = user_dir / file.filename
    file.save(str(path))
    
    return jsonify({"success": True, "message": f"File {file.filename} uploaded to your cloud storage"})

@app.route("/api/files/<file_id>", methods=["DELETE"])
@login_required
def delete_file(file_id):
    # For now file_id is just filename since it's a simple flat storage
    user_dir = Path(REPO_ROOT) / "uploads" / request.user_id
    target = user_dir / file_id
    if target.exists() and target.is_file():
        target.unlink()
        return jsonify({"success": True})
    return jsonify({"error": "File not found"}), 404

# ====================
# WEB UI
# ====================
@app.route("/")
def index():
    return send_from_directory(TEMPLATE_PATH, "index.html")

@app.route("/<path:filename>")
def static_files(filename):
    file_path = Path(STATIC_PATH) / filename
    if file_path.exists() and file_path.is_file():
        return send_from_directory(STATIC_PATH, filename)
    return send_from_directory(TEMPLATE_PATH, "index.html")

# ====================
# STARTUP
# ====================
if __name__ == "__main__":
    print(f"NeuralAI Unified Service starting on port {PORT}...")
    init_db()
    load_model()
    app.run(host="0.0.0.0", port=PORT, debug=False, threaded=True)
