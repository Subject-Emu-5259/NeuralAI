#!/usr/bin/env python3
"""
NeuralAI Unified Service - ALL IN ONE
===================================
- Model inference (SmolLM2-360M)
- Neural Uplink (4 parallel agents) 
- Tools (code, terminal)
- Web UI
"""
import os, sys, json, asyncio, requests
import torch, sqlite3, subprocess, tempfile, uuid
from pathlib import Path
from datetime import datetime
from flask import Flask, Response, jsonify, request, send_from_directory

torch.set_num_threads(4)

app = Flask(__name__, static_folder=None)

# Config
PORT = int(os.environ.get("PORT", "5000"))
MODEL_PATH = os.environ.get("MODEL_PATH", "/home/workspace/Projects/NeuralAI/checkpoints/v2_model")
BASE_MODEL = os.environ.get("BASE_MODEL", "HuggingFaceTB/SmolLM2-360M-Instruct")
STATIC_PATH = "/home/workspace/Projects/NeuralAI/from-scratch/web_ui"

# Model globals
model = None
tokenizer = None
model_status = "loading"
inference_count = 0

# Terminal sessions
terminal_sessions = {}
# Conversations storage (Simple JSON file)
CONV_FILE = Path("/home/workspace/Projects/NeuralAI/conversations.json")
# Files storage
STORAGE_SERVICE = os.environ.get("STORAGE_SERVICE", "http://localhost:7003")
STORAGE_ROOT = Path("/home/workspace/Projects/NeuralAI/storage")
STORAGE_ROOT.mkdir(parents=True, exist_ok=True)

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
def load_model():
    global model, tokenizer, model_status
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        tokenizer.pad_token = tokenizer.eos_token
        adapter = Path(MODEL_PATH)
        has_adapter = any((adapter / f).exists() for f in ["adapter_model.bin", "adapter_model.safetensors"])
        if adapter.exists() and has_adapter:
            base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.float32, device_map=None)
            model = PeftModel.from_pretrained(base, str(adapter))
        else:
            model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.float32, device_map=None)
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
        full = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        inputs = tokenizer(full, return_tensors="pt")
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=True, temperature=temperature, top_p=0.95, pad_token_id=tokenizer.eos_token_id)
        new_tokens = out[0][inputs["input_ids"].shape[-1]:]
        inference_count += 1
        return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    except Exception as e:
        return f"Error: {e}"

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
def health():
    return jsonify({"status": model_status, "model": BASE_MODEL, "inference_count": inference_count, "uplink": "integrated"})

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
def api_chat():
    data = request.get_json() or {}
    prompt = data.get("prompt", "")
    use_uplink = data.get("use_uplink", False)
    
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
            response = generate_response(prompt)
            for word in response.split():
                yield f"data: {json.dumps({'content': word + ' '})}\n\n"
        
        yield "data: [DONE]\n\n"

    return Response(generate_unified(), mimetype="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

# ====================
# ROUTES - CONVERSATIONS
# ====================
@app.route("/api/conversations", methods=["GET", "POST"])
def manage_convs():
    convs = load_convs()
    if request.method == "POST":
        data = request.get_json() or {}
        cid = str(uuid.uuid4())[:8]
        convs[cid] = {"title": data.get("title", "New Chat"), "messages": []}
        save_convs(convs)
        return jsonify({"success": True, "id": cid})
    
    return jsonify([{"id": k, "title": v["title"]} for k, v in convs.items()])

@app.route("/api/conversations/<cid>", methods=["GET", "DELETE"])
def conv_detail(cid):
    convs = load_convs()
    if cid not in convs: return jsonify({"error": "Not found"}), 404
    if request.method == "DELETE":
        del convs[cid]
        save_convs(convs)
        return jsonify({"success": True})
    return jsonify(convs[cid])

# ====================
# ROUTES - FILES (Proxied to Storage Service)
# ====================
@app.route("/api/files", methods=["GET", "POST"])
def manage_files():
    try:
        if request.method == "POST":
            if 'file' not in request.files: return jsonify({"error": "No file"}), 400
            file = request.files['file']
            files = {'file': (file.filename, file.read(), file.content_type)}
            r = requests.post(f"{STORAGE_SERVICE}/api/storage/upload", files=files)
            return jsonify(r.json()), r.status_code
        
        r = requests.get(f"{STORAGE_SERVICE}/api/storage/list")
        if r.status_code == 200:
            data = r.json()
            legacy_files = []
            for item in data.get("items", []):
                legacy_files.append({
                    "name": item["name"],
                    "size": item["size"],
                    "path": item["name"],
                    "is_dir": item["is_dir"]
                })
            return jsonify(legacy_files)
        return jsonify(r.json()), r.status_code
    except Exception as e:
        print(f"[WARN] Storage service down: {e}")
        files = []
        for f in STORAGE_ROOT.iterdir():
            files.append({"name": f.name, "size": f.stat().st_size, "path": f.name})
        return jsonify(files)

@app.route("/api/files/<path:filename>", methods=["GET", "DELETE"])
def handle_file(filename):
    try:
        if request.method == "DELETE":
            r = requests.delete(f"{STORAGE_SERVICE}/api/storage/delete", params={"path": filename})
            return jsonify(r.json()), r.status_code
        
        r = requests.get(f"{STORAGE_SERVICE}/api/storage/download", params={"path": filename}, stream=True)
        return Response(r.iter_content(chunk_size=1024), content_type=r.headers.get('Content-Type'))
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
# STARTUP
# ====================
if __name__ == "__main__":
    print(f"NeuralAI Unified Service starting on port {PORT}...")
    load_model()
    app.run(host="0.0.0.0", port=PORT, debug=False, threaded=True)
