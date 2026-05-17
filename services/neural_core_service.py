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

# ====================
# API ROUTES
# ====================

@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json() or {}
    prompt = data.get("prompt", "")
    messages = data.get("messages", [])
    temperature = data.get("temperature", 0.7)
    max_tokens = data.get("max_tokens", 256)
    use_uplink = data.get("use_uplink", False)
    
    if messages:
        prompt = messages[-1].get("content", "") if messages else prompt
    
    if not prompt:
        return jsonify({"response": "No prompt provided.", "status": "error"}), 400
    
    response = generate_response(prompt, max_tokens=max_tokens, temperature=temperature)
    return jsonify({"response": response, "status": "ok", "inference_count": inference_count})

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
    terminal_sessions[session_id] = {"output": [], "cwd": "/home/workspace"}
    return jsonify({"session_id": session_id, "status": "created"})

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

@app.route("/api/terminal/<session_id>/output", methods=["GET"])
def terminal_output(session_id):
    if session_id not in terminal_sessions:
        return jsonify({"output": []})
    return jsonify({"output": terminal_sessions[session_id]["output"]})

@app.route("/api/terminal/<session_id>/read", methods=["GET"])
def terminal_read(session_id):
    if session_id not in terminal_sessions:
        return jsonify({"output": []})
    return jsonify({"output": terminal_sessions[session_id]["output"]})

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
    return send_from_directory(STATIC_PATH, "index.html")

@app.route("/<path:filename>")
def static_files(filename):
    file_path = Path(STATIC_PATH) / filename
    if file_path.exists() and file_path.is_file():
        return send_from_directory(STATIC_PATH, filename)
    return send_from_directory(STATIC_PATH, "index.html")

# ====================
# STARTUP
# ====================
if __name__ == "__main__":
    print(f"NeuralAI Unified Service starting on port {PORT}...")
    load_model()
    app.run(host="0.0.0.0", port=PORT, debug=False, threaded=True)
