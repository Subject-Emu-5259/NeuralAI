#!/usr/bin/env python3
"""
NeuralAI Web UI Service
- Clean web interface  
- Calls model service for inference
- Calls tools service for execution
- Neural Uplink for multi-agent analysis
"""

import os
import sys
import json
import requests
import sqlite3
from pathlib import Path
from datetime import datetime
from flask import Flask, Response, jsonify, render_template, request, stream_with_context, g

# Configuration
PORT = int(os.environ.get("PORT", "5000"))
MODEL_SERVICE = os.environ.get("MODEL_SERVICE", "http://localhost:7001")
TOOLS_SERVICE = os.environ.get("TOOLS_SERVICE", "http://localhost:7001")
UPLINK_SERVICE = os.environ.get("UPLINK_SERVICE", "http://localhost:8000")
DATABASE = Path("/home/workspace/Projects/NeuralAI/from-scratch/web_ui/neuralai.db")

app = Flask(__name__, 
            template_folder="/home/workspace/Projects/NeuralAI/from-scratch/web_ui/templates",
            static_folder="/home/workspace/Projects/NeuralAI/from-scratch/web_ui/static")
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024


# ====================
# DATABASE
# ====================

def get_db():
    if 'db' not in g:
        g.db = sqlite3.connect(str(DATABASE))
        g.db.row_factory = sqlite3.Row
    return g.db


@app.teardown_appcontext
def close_db(error):
    db = g.pop('db', None)
    if db is not None:
        db.close()


def init_db():
    if not DATABASE.exists():
        DATABASE.parent.mkdir(parents=True, exist_ok=True)
        db = sqlite3.connect(str(DATABASE))
        db.execute("""CREATE TABLE IF NOT EXISTS conversations (
            id TEXT PRIMARY KEY, title TEXT, created_at TEXT, updated_at TEXT, message_count INTEGER DEFAULT 0)""")
        db.execute("""CREATE TABLE IF NOT EXISTS messages (
            id TEXT PRIMARY KEY, conversation_id TEXT, role TEXT, content TEXT, created_at TEXT)""")
        db.commit()
        db.close()


# ====================
# ROUTES  
# ====================

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/health", methods=["GET"])
def health():
    services = {}
    try:
        resp = requests.get("http://localhost:7001/health", timeout=2)
        services["unified"] = resp.json()
    except:
        services["unified"] = {"status": "offline"}
    
    try:
        resp = requests.get(f"{UPLINK_SERVICE}/health", timeout=2)
        services["uplink"] = resp.json()
    except:
        services["uplink"] = {"status": "offline"}
    
    return jsonify({"status": "ok", "version": "4.0-uplink", "services": services})


@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json()
    message = data.get("message", data.get("prompt", "")).strip()
    conv_id = data.get("conversation_id")
    
    if not message:
        return jsonify({"error": "No message provided"}), 400
    
    tool = detect_tool(message)
    if tool:
        return handle_tool(tool, message, data)
    
    if conv_id:
        db = get_db()
        db.execute("INSERT INTO messages (id, conversation_id, role, content, created_at) VALUES (?, ?, ?, ?, ?)",
                   (f"msg_{datetime.now().timestamp()}", conv_id, "user", message, datetime.utcnow().isoformat()))
        db.commit()
    
    def generate():
        full_response = ""
        try:
            resp = requests.post(f"{MODEL_SERVICE}/generate/stream",
                                json={"prompt": message, "max_tokens": 256, "temperature": 0.7},
                                stream=True, timeout=60)
            for line in resp.iter_lines():
                if line:
                    line = line.decode('utf-8')
                    if line.startswith("data: "):
                        data_str = line[6:]
                        if data_str == "[DONE]":
                            break
                        try:
                            d = json.loads(data_str)
                            token = d.get("token", "")
                            if token:
                                full_response += token
                                yield f"data: {json.dumps({'content': token})}\n\n"
                        except:
                            pass
            
            if conv_id and full_response:
                db = get_db()
                now = datetime.utcnow().isoformat()
                db.execute("INSERT INTO messages (id, conversation_id, role, content, created_at) VALUES (?, ?, ?, ?, ?)",
                          (f"msg_{datetime.now().timestamp()}", conv_id, "assistant", full_response, now))
                db.commit()
            yield "data: [DONE]\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'content': f'Error: {str(e)}'})}\n\n"
            yield "data: [DONE]\n\n"
    
    return Response(stream_with_context(generate()), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache"})


# ====================
# TOOL DETECTION
# ====================

def detect_tool(message: str) -> str:
    lower = message.lower()
    
    # Neural Uplink
    if any(k in lower for k in ["uplink", "analyze this", "deep analysis", "multi-agent",
                                 "all agents", "neural uplink", "comprehensive analysis",
                                 "breakdown", "break this down", "all angles"]):
        return "uplink"
    
    # Terminal
    for prefix in ["run ", "execute ", "shell ", "command ", "bash "]:
        if lower.startswith(prefix):
            return "terminal"
    
    # Code
    if any(k in lower for k in ["run this code", "execute this", "run python", "run js"]):
        return "code"
    
    # Image
    if any(k in lower for k in ["create an image", "generate an image", "make an image", "draw an image"]):
        return "image"
    
    return None


def handle_tool(tool: str, message: str, data: dict):
    
    if tool == "uplink":
        def generate():
            try:
                resp = requests.post(f"{UPLINK_SERVICE}/uplink",
                                    json={"prompt": message}, timeout=60)
                result = resp.json()
                if result.get("success"):
                    yield f"data: {json.dumps({'content': result.get('fused', 'No response')})}\n\n"
                else:
                    yield f"data: {json.dumps({'content': 'Uplink error'})}\n\n"
                yield "data: [DONE]\n\n"
            except Exception as e:
                yield f"data: {json.dumps({'content': f'Uplink offline: {str(e)}'})}\n\n"
                yield "data: [DONE]\n\n"
        return Response(generate(), mimetype="text/event-stream", headers={"Cache-Control": "no-cache"})
    
    if tool == "terminal":
        cmd = message
        for prefix in ["run ", "execute ", "shell ", "command ", "bash "]:
            if message.lower().startswith(prefix):
                cmd = message[len(prefix):].strip()
                break
        def generate():
            try:
                resp = requests.post(f"{TOOLS_SERVICE}/execute/shell",
                                    json={"command": cmd}, timeout=30)
                result = resp.json()
                yield f"data: {json.dumps({'content': '```bash\\n'})}\n\n"
                if result.get("success"):
                    yield f"data: {json.dumps({'content': result.get('output', '')})}\n\n"
                else:
                    yield f"data: {json.dumps({'content': result.get('error', 'Error')})}\n\n"
                yield f"data: {json.dumps({'content': '```\\n'})}\n\n"
                yield "data: [DONE]\n\n"
            except Exception as e:
                yield f"data: {json.dumps({'content': str(e)})}\n\n"
                yield "data: [DONE]\n\n"
        return Response(generate(), mimetype="text/event-stream", headers={"Cache-Control": "no-cache"})
    
    if tool == "code":
        code = message
        for trigger in ["run this code:", "execute this code:", "run python:"]:
            if trigger in message.lower():
                code = message[message.lower().find(trigger) + len(trigger):].strip()
                break
        def generate():
            try:
                resp = requests.post(f"{TOOLS_SERVICE}/execute/code",
                                    json={"code": code, "language": "python"}, timeout=30)
                result = resp.json()
                yield f"data: {json.dumps({'content': '```\\n'})}\n\n"
                yield f"data: {json.dumps({'content': result.get('output', result.get('error', 'No output'))})}\n\n"
                yield f"data: {json.dumps({'content': '```\\n'})}\n\n"
                yield "data: [DONE]\n\n"
            except Exception as e:
                yield f"data: {json.dumps({'content': str(e)})}\n\n"
                yield "data: [DONE]\n\n"
        return Response(generate(), mimetype="text/event-stream", headers={"Cache-Control": "no-cache"})
    
    if tool == "image":
        prompt = message
        for trigger in ["create an image of ", "generate an image of ", "image of "]:
            if trigger in message.lower():
                prompt = message[message.lower().find(trigger) + len(trigger):]
                break
        def generate():
            try:
                resp = requests.post(f"{TOOLS_SERVICE}/generate/image",
                                    json={"prompt": prompt}, timeout=120)
                result = resp.json()
                if result.get("success"):
                    yield f"data: {json.dumps({'content': '🎨 Image generated!\\n'})}\n\n"
                    yield "data: " + json.dumps({"content": "![](" + result.get("image_url", "") + ")"}) + "\n\n"
                else:
                    yield f"data: {json.dumps({'content': result.get('error', 'Error')})}\n\n"
                yield "data: [DONE]\n\n"
            except Exception as e:
                yield f"data: {json.dumps({'content': str(e)})}\n\n"
                yield "data: [DONE]\n\n"
        return Response(generate(), mimetype="text/event-stream", headers={"Cache-Control": "no-cache"})
    
    return jsonify({"error": "Unknown tool"}), 400


# ====================
# CONVERSATIONS
# ====================

@app.route("/api/conversations", methods=["GET"])
def list_conversations():
    db = get_db()
    rows = db.execute("SELECT * FROM conversations ORDER BY updated_at DESC LIMIT 50").fetchall()
    return jsonify({"conversations": [dict(r) for r in rows]})


@app.route("/api/conversations/<conv_id>", methods=["GET"])  
def get_conversation(conv_id):
    db = get_db()
    conv = db.execute("SELECT * FROM conversations WHERE id = ?", (conv_id,)).fetchone()
    if not conv:
        return jsonify({"error": "Not found"}), 404
    messages = db.execute("SELECT * FROM messages WHERE conversation_id = ? ORDER BY created_at", (conv_id,)).fetchall()
    return jsonify({"conversation": dict(conv), "messages": [dict(m) for m in messages]})


# ====================
# TERMINAL API
# ====================

# Terminal sessions storage
terminal_sessions = {}

@app.route("/api/terminal/create", methods=["POST"])
def terminal_create():
    """Create a new terminal session."""
    import uuid
    session_id = str(uuid.uuid4())[:8]
    terminal_sessions[session_id] = {
        "created": datetime.utcnow().isoformat(),
        "output": "",
        "running": False
    }
    return jsonify({"session_id": session_id, "status": "created"})

@app.route("/api/terminal/<session_id>/read", methods=["GET"])
def terminal_read(session_id):
    """Read terminal output."""
    session = terminal_sessions.get(session_id, {})
    return jsonify({
        "output": session.get("output", ""),
        "running": session.get("running", False)
    })

@app.route("/api/terminal/<session_id>/write", methods=["POST"])
def terminal_write(session_id):
    """Write command to terminal."""
    data = request.get_json()
    command = data.get("command", "")
    
    if session_id not in terminal_sessions:
        terminal_sessions[session_id] = {"output": "", "running": False}
    
    # Execute command via tools service
    try:
        resp = requests.post(f"{TOOLS_SERVICE}/execute/shell",
                            json={"command": command}, timeout=30)
        result = resp.json()
        
        output = result.get("output", "") or result.get("error", "")
        terminal_sessions[session_id]["output"] += f"$ {command}\n{output}\n"
        
        return jsonify({
            "output": terminal_sessions[session_id]["output"],
            "running": False,
            "success": result.get("success", False)
        })
    except Exception as e:
        return jsonify({"output": str(e), "running": False, "success": False})

@app.route("/api/terminal/<session_id>/stop", methods=["POST"])
def terminal_stop(session_id):
    """Stop terminal session."""
    if session_id in terminal_sessions:
        terminal_sessions[session_id]["running"] = False
    return jsonify({"status": "stopped"})

@app.route("/api/terminal/snippets", methods=["GET"])
def terminal_snippets():
    """Get code snippets."""
    return jsonify({
        "snippets": [
            {"lang": "python", "name": "hello", "code": "print('Hello, World!')"},
            {"lang": "bash", "name": "info", "code": "uname -a"}
        ]
    })

@app.route("/api/terminal/snippets/<lang>/<name>", methods=["GET"])
def terminal_snippet(lang, name):
    """Get specific snippet."""
    snippets = {
        ("python", "hello"): "print('Hello, World!')",
        ("bash", "info"): "uname -a"
    }
    return jsonify({"code": snippets.get((lang, name), "# Not found")})


# ====================
# STARTUP
# ====================

print(f"[WebUI] Port: {PORT}")
print(f"[WebUI] Model: {MODEL_SERVICE}")
print(f"[WebUI] Uplink: {UPLINK_SERVICE}")
init_db()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT, debug=False, threaded=True)
