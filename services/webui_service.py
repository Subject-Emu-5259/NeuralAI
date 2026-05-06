#!/usr/bin/env python3
"""
NeuralAI Web UI Service
- Clean web interface
- Calls model service for inference
- Calls tools service for execution
- No model loading - just API calls
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
TOOLS_SERVICE = os.environ.get("TOOLS_SERVICE", "http://localhost:7002")
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
    """Initialize database if needed."""
    if not DATABASE.exists():
        DATABASE.parent.mkdir(parents=True, exist_ok=True)
        db = sqlite3.connect(str(DATABASE))
        db.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                title TEXT,
                created_at TEXT,
                updated_at TEXT,
                message_count INTEGER DEFAULT 0
            )
        """)
        db.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id TEXT PRIMARY KEY,
                conversation_id TEXT,
                role TEXT,
                content TEXT,
                created_at TEXT
            )
        """)
        db.commit()
        db.close()
        print(f"[WebUI] Database initialized at {DATABASE}")


# ====================
# ROUTES
# ====================

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/health", methods=["GET"])
def health():
    """Check all services."""
    services = {}
    
    # Check model service
    try:
        resp = requests.get(f"{MODEL_SERVICE}/health", timeout=2)
        services["model"] = resp.json()
    except:
        services["model"] = {"status": "offline"}
    
    # Check tools service
    try:
        resp = requests.get(f"{TOOLS_SERVICE}/health", timeout=2)
        services["tools"] = resp.json()
    except:
        services["tools"] = {"status": "offline"}
    
    return jsonify({
        "status": "ok",
        "version": "4.0-microservices",
        "services": services
    })


@app.route("/api/chat", methods=["POST"])
def chat():
    """Main chat endpoint - proxies to model service."""
    data = request.get_json()
    
    message = data.get("message", data.get("prompt", "")).strip()
    conv_id = data.get("conversation_id")
    messages = data.get("messages", [])
    
    if not message:
        return jsonify({"error": "No message provided"}), 400
    
    # Check for tool triggers
    tool = detect_tool(message)
    
    if tool:
        # Route to tools service
        return handle_tool(tool, message, data)
    
    # Save user message
    if conv_id:
        db = get_db()
        now = datetime.utcnow().isoformat()
        db.execute(
            "INSERT INTO messages (id, conversation_id, role, content, created_at) VALUES (?, ?, ?, ?, ?)",
            (f"msg_{datetime.now().timestamp()}", conv_id, "user", message, now)
        )
        db.commit()
    
    # Stream from model service
    def generate():
        full_response = ""
        
        try:
            # Call model service with streaming
            resp = requests.post(
                f"{MODEL_SERVICE}/generate/stream",
                json={"prompt": message, "max_tokens": 256, "temperature": 0.7},
                stream=True,
                timeout=60
            )
            
            for line in resp.iter_lines():
                if line:
                    line = line.decode('utf-8')
                    if line.startswith("data: "):
                        data_str = line[6:]
                        if data_str == "[DONE]":
                            break
                        try:
                            data = json.loads(data_str)
                            token = data.get("token", "")
                            if token:
                                full_response += token
                                yield f"data: {json.dumps({'content': token})}\n\n"
                        except:
                            pass
            
            # Save response
            if conv_id and full_response:
                db = get_db()
                now = datetime.utcnow().isoformat()
                db.execute(
                    "INSERT INTO messages (id, conversation_id, role, content, created_at) VALUES (?, ?, ?, ?, ?)",
                    (f"msg_{datetime.now().timestamp()}", conv_id, "assistant", full_response, now)
                )
                db.execute(
                    "UPDATE conversations SET updated_at = ?, message_count = message_count + 1 WHERE id = ?",
                    (now, conv_id)
                )
                db.commit()
            
            yield "data: [DONE]\n\n"
            
        except requests.exceptions.ConnectionError:
            yield f"data: {json.dumps({'content': 'Model service offline. Please wait for model to load.'})}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'content': f'Error: {str(e)}'})}\n\n"
            yield "data: [DONE]\n\n"
    
    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    )


def detect_tool(message: str) -> str:
    """Detect if message should trigger a tool."""
    lower = message.lower()
    
    # Terminal/shell commands
    for prefix in ["run ", "execute ", "shell ", "command ", "bash "]:
        if lower.startswith(prefix):
            return "terminal"
    
    # Code execution
    if any(k in lower for k in ["run this code", "execute this", "run python", "run js"]):
        return "code"
    
    # Image generation
    if any(k in lower for k in ["create an image", "generate an image", "make an image", "draw an image"]):
        return "image"
    
    return None


def handle_tool(tool: str, message: str, data: dict):
    """Handle tool requests."""
    
    if tool == "terminal":
        # Extract command
        lower = message.lower()
        cmd = message
        for prefix in ["run ", "execute ", "shell ", "command ", "bash "]:
            if lower.startswith(prefix):
                cmd = message[len(prefix):].strip()
                break
        
        def generate():
            try:
                resp = requests.post(
                    f"{TOOLS_SERVICE}/execute/shell",
                    json={"command": cmd},
                    timeout=30
                )
                result = resp.json()
                
                yield f"data: {json.dumps({'content': '```bash\\n'})}\n\n"
                
                if result.get("success"):
                    output = result.get("output", "")
                    for line in output.split("\n"):
                        yield f"data: {json.dumps({'content': line + '\\n'})}\n\n"
                else:
                    yield f"data: {json.dumps({'content': result.get('error', 'Unknown error')})}\n\n"
                
                yield f"data: {json.dumps({'content': '```\\n'})}\n\n"
                yield "data: [DONE]\n\n"
                
            except Exception as e:
                yield f"data: {json.dumps({'content': f'Tools service error: {str(e)}'})}\n\n"
                yield "data: [DONE]\n\n"
        
        return Response(generate(), mimetype="text/event-stream",
                        headers={"Cache-Control": "no-cache"})
    
    if tool == "code":
        # Extract code
        code = message
        for trigger in ["run this code:", "execute this code:", "run python:", "run python code:"]:
            if trigger in message.lower():
                idx = message.lower().find(trigger)
                code = message[idx + len(trigger):].strip()
                break
        
        def generate():
            try:
                resp = requests.post(
                    f"{TOOLS_SERVICE}/execute/code",
                    json={"code": code, "language": "python"},
                    timeout=30
                )
                result = resp.json()
                
                yield f"data: {json.dumps({'content': '```\\n'})}\n\n"
                
                if result.get("success"):
                    output = result.get("output", "(no output)")
                    yield f"data: {json.dumps({'content': output})}\n\n"
                else:
                    yield f"data: {json.dumps({'content': 'Error: ' + result.get('error', 'Unknown error')})}\n\n"
                
                yield f"data: {json.dumps({'content': '\\n```\\n'})}\n\n"
                yield "data: [DONE]\n\n"
                
            except Exception as e:
                yield f"data: {json.dumps({'content': f'Tools service error: {str(e)}'})}\n\n"
                yield "data: [DONE]\n\n"
        
        return Response(generate(), mimetype="text/event-stream",
                        headers={"Cache-Control": "no-cache"})
    
    if tool == "image":
        # Extract prompt
        prompt = message.lower()
        for trigger in ["create an image of ", "generate an image of ", "make an image of ", "draw an image of "]:
            if trigger in prompt:
                prompt = message[message.lower().find(trigger) + len(trigger):]
                break
        
        # Return placeholder for now (image generation uses Zo's built-in)
        def generate():
            yield f"data: {json.dumps({'content': f'🎨 To generate an image, use the Image tool directly. Prompt: {prompt}'})}\n\n"
            yield "data: [DONE]\n\n"
        
        return Response(generate(), mimetype="text/event-stream",
                        headers={"Cache-Control": "no-cache"})
    
    return jsonify({"error": "Unknown tool"}), 400


# ====================
# CONVERSATIONS API
# ====================

@app.route("/api/conversations", methods=["GET"])
def list_conversations():
    """List all conversations."""
    db = get_db()
    rows = db.execute(
        "SELECT * FROM conversations ORDER BY updated_at DESC LIMIT 50"
    ).fetchall()
    
    return jsonify({
        "conversations": [dict(r) for r in rows]
    })


@app.route("/api/conversations/<conv_id>", methods=["GET"])
def get_conversation(conv_id):
    """Get a conversation with messages."""
    db = get_db()
    
    conv = db.execute(
        "SELECT * FROM conversations WHERE id = ?", (conv_id,)
    ).fetchone()
    
    if not conv:
        return jsonify({"error": "Not found"}), 404
    
    messages = db.execute(
        "SELECT * FROM messages WHERE conversation_id = ? ORDER BY created_at",
        (conv_id,)
    ).fetchall()
    
    return jsonify({
        "conversation": dict(conv),
        "messages": [dict(m) for m in messages]
    })


# ====================
# STARTUP
# ====================

print(f"[WebUI] Starting on port {PORT}")
print(f"[WebUI] Model service: {MODEL_SERVICE}")
print(f"[WebUI] Tools service: {TOOLS_SERVICE}")

init_db()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT, debug=False, threaded=True)
