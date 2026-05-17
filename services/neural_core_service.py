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
import torch, sqlite3, subprocess, tempfile, uuid
from pathlib import Path
from datetime import datetime
from flask import Flask, Response, jsonify, request, send_from_directory, stream_with_context

torch.set_num_threads(4)

app = Flask(__name__, static_folder=None)

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
        CREATE TABLE IF NOT EXISTS conversations (
            id TEXT PRIMARY KEY,
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
            created_at TEXT NOT NULL,
            FOREIGN KEY (conversation_id) REFERENCES conversations(id)
        );
    """)
    db.commit()
    db.close()

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
# API ROUTES
# ====================

@app.route("/api/chat", methods=["POST"])
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
            now = datetime.utcnow().isoformat()
            db.execute("INSERT INTO messages (conversation_id, role, content, created_at) VALUES (?, ?, ?, ?)",
                       (conv_id, "user", prompt, now))
            db.execute("UPDATE conversations SET updated_at = ?, message_count = message_count + 1 WHERE id = ?",
                       (now, conv_id))
            db.commit()
            db.close()
        except Exception as e:
            print(f"[DB ERROR] {e}")

    def generate():
        full_response = ""
        for chunk in generate_response_stream(prompt, max_tokens, temperature):
            full_response += chunk
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
def conversations_api():
    db = get_db()
    if request.method == "GET":
        rows = db.execute("SELECT * FROM conversations ORDER BY updated_at DESC LIMIT 50").fetchall()
        convs = [dict(r) for r in rows]
        db.close()
        return jsonify({"conversations": convs})
    else:
        data = request.get_json() or {}
        cid = "conv_" + str(uuid.uuid4().hex[:12])
        title = data.get("title", "New Chat")
        now = datetime.utcnow().isoformat()
        db.execute("INSERT INTO conversations (id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
                   (cid, title, now, now))
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
def list_files():
    uploads = Path(STATIC_PATH) / "uploads"
    uploads.mkdir(exist_ok=True)
    files = sorted([{"name": f.name, "id": str(uuid.uuid4())[:8]} for f in uploads.iterdir() if f.is_file()])
    return jsonify({"files": [f["name"] for f in files], "ids": [f["id"] for f in files]})

@app.route("/api/terminal/create", methods=["POST"])
def terminal_create():
    session_id = str(uuid.uuid4())[:8]
    terminal_sessions[session_id] = {"output": "", "cwd": "/home/workspace", "alive": True}
    return jsonify({"session_id": session_id, "status": "created"})

@app.route("/api/terminal/<session_id>/write", methods=["POST"])
def terminal_write(session_id):
    if session_id not in terminal_sessions:
        return jsonify({"error": "Session not found"}), 404
    data = request.get_json() or {}
    cmd = data.get("input", "").strip()
    if not cmd:
        return jsonify({"error": "No input"}), 400
    
    session = terminal_sessions[session_id]
    try:
        # Simple non-PTY fallback for now, but aligned with UI polling
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=30,
            cwd=session.get("cwd", "/home/workspace")
        )
        output = result.stdout
        if result.stderr:
            output += "\n" + result.stderr
        session["output"] += output + f"\n[Process exited with {result.returncode}]\n"
        return jsonify({"ok": True})
    except subprocess.TimeoutExpired:
        session["output"] += "Command timed out\n"
        return jsonify({"ok": True})
    except Exception as e:
        session["output"] += str(e) + "\n"
        return jsonify({"ok": True})

@app.route("/api/terminal/<session_id>/read", methods=["GET"])
def terminal_read(session_id):
    if session_id not in terminal_sessions:
        return jsonify({"output": "", "alive": False}), 404
    session = terminal_sessions[session_id]
    output = session["output"]
    session["output"] = "" # Clear after read as UI polls
    return jsonify({"output": output, "alive": session["alive"]})

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
