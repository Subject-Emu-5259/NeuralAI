import re
# NeuralAI Web UI v5.2 - Enhanced with Persistence, Memory, and Settings
import hashlib
import json
import os
import time
import sqlite3
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Optional

# Disable tokenizer parallelism warning
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from flask import Flask, Response, jsonify, render_template, request, stream_with_context, g
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
import jwt
from functools import wraps

# NeuralAI Cloud Client
try:
    from neural_cloud_client import NeuralCloudClient
    cloud_client = NeuralCloudClient(base_url="http://localhost:8002/remote.php/dav/files/admin", user="admin", password="NeuralAI_Admin_2026!")
except ImportError:
    cloud_client = None

# NeuralAI Engine - Router + Local Model + Uplink + Tools
try:
    from neuralai_router import neuralai_route
    from neuralai_engine import neuralai_chat, local_model, neuralai_tool_call
    HAS_ROUTER = True
except ImportError as e:
    print(f"[NeuralAI] Import error: {e}")
    HAS_ROUTER = False
    def neuralai_route(msg):
        return ("local", None)
    neuralai_chat = None
    local_model = None
    neuralai_tool_call = None

def run_tool_sync(tool: str, msg: str):
    """Run tool synchronously by collecting all chunks from async generator."""
    import asyncio
    import time
    import os
    
    if neuralai_tool_call is None:
        return ["[Error] Tool handler not available"]
    try:
        async def collect_chunks():
            chunks = []
            async for chunk in neuralai_tool_call(tool, msg):
                chunks.append(chunk)
            return chunks
        return asyncio.run(collect_chunks())
    except Exception as e:
        return [f"[Tool Error] {e}"]

def strip_terminal_prefix(msg: str) -> str:
    """Remove terminal command prefixes."""
    lower = msg.lower()
    for prefix in ["run ", "execute ", "shell ", "command "]:
        if lower.startswith(prefix):
            return msg[len(prefix):].strip()
    return msg

try:
    import torch
except Exception:
    torch = None

try:
    import requests
except Exception:
    requests = None

try:
    from rag import index_document, query_documents, rebuild_index_registry
except Exception:
    def index_document(filepath: str, collection_name: str = "documents") -> dict:
        return {"chunks": 0, "error": "RAG backend unavailable"}

    def query_documents(query: str, collection_name: str = "documents", top_k: int = 4) -> list[dict]:
        return []

    def rebuild_index_registry(collection_name: str = "documents") -> dict:
        return {}

try:
    from terminal import terminal_bp
except Exception:
    from flask import Blueprint
    terminal_bp = Blueprint("terminal", __name__)

BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent.parent
DATA_DIR = REPO_ROOT / "data"
STORAGE_DIR = REPO_ROOT / "storage"
LOGS_DIR = REPO_ROOT / "logs"

# Structured storage
UPLOAD_FOLDER = REPO_ROOT / "uploads"
IMAGE_STORAGE = STORAGE_DIR / "images"

# Ensure all structured directories exist
for d in [DATA_DIR, STORAGE_DIR, LOGS_DIR, UPLOAD_FOLDER, IMAGE_STORAGE]:
    d.mkdir(parents=True, exist_ok=True)

# Database path
DATABASE = DATA_DIR / "neuralai.db"

MODEL_PATH = os.environ.get("MODEL_PATH", str(REPO_ROOT / "checkpoints" / "v2_model"))
MODEL_NAME = os.environ.get("MODEL_NAME", "HuggingFaceTB/SmolLM2-360M-Instruct")
UPLINK_URL = os.environ.get("UPLINK_URL", "http://localhost:7000")
PORT = int(os.environ.get("PORT", "5000"))
ALLOWED = {".pdf", ".docx", ".doc", ".txt", ".md"}
REGISTRY_FILE = DATA_DIR / ".indexed_files.json"
VERSION = os.environ.get("NEURALAI_VERSION", "4.0")

app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("NEURALAI_SECRET", "neural-intelligence-core-2026-secret-x")
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024
app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)
app.register_blueprint(terminal_bp)

# ========================================
# AUTH DECORATOR
# ========================================

def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get("Authorization")
        if not token or not token.startswith("Bearer "):
            return jsonify({"error": "Authentication required"}), 401
        try:
            token = token.split(" ")[1]
            data = jwt.decode(token, app.config["SECRET_KEY"], algorithms=["HS256"])
            request.user_id = data["user_id"]
        except Exception:
            return jsonify({"error": "Invalid or expired token"}), 401
        return f(*args, **kwargs)
    return decorated

INDEXED_FILES: dict[str, str] = {}
model = None
tokenizer = None
model_error: str | None = None


# ========================================
# DATABASE LAYER
# ========================================

def get_db():
    """Get database connection."""
    db = getattr(g, '_database', None)
    if db is None:
        db = g._database = sqlite3.connect(str(DATABASE))
        db.row_factory = sqlite3.Row
    return db


@app.teardown_appcontext
def close_connection(exception):
    """Close database connection."""
    db = getattr(g, '_database', None)
    if db is not None:
        db.close()


def init_db():
    """Initialize database tables."""
    db = get_db()
    
    # Check if columns exist (Migration)
    try:
        cursor = db.execute("PRAGMA table_info(users)")
        cols = [row[1] for row in cursor.fetchall()]
        if "account_type" not in cols:
            db.execute("ALTER TABLE users ADD COLUMN account_type TEXT DEFAULT 'standard'")
        if "invite_code" not in cols:
            db.execute("ALTER TABLE users ADD COLUMN invite_code TEXT")
    except Exception as e:
        print(f"[NeuralAI] Migration warning: {e}")

    db.executescript("""
        -- Users table
        CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE,
            first_name TEXT,
            last_name TEXT,
            bio TEXT,
            password_hash TEXT NOT NULL,
            account_type TEXT DEFAULT 'standard',
            invite_code TEXT,
            is_founder INTEGER DEFAULT 0,
            created_at TEXT NOT NULL
        );
        
        -- Conversations table
        CREATE TABLE IF NOT EXISTS conversations (
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            title TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            message_count INTEGER DEFAULT 0,
            FOREIGN KEY (user_id) REFERENCES users(id)
        );
        
        -- Messages table
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY (conversation_id) REFERENCES conversations(id)
        );
        
        -- User settings table
        CREATE TABLE IF NOT EXISTS user_settings (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        
        -- Memory facts table
        CREATE TABLE IF NOT EXISTS memory_facts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            fact TEXT NOT NULL,
            category TEXT DEFAULT 'general',
            created_at TEXT NOT NULL,
            importance INTEGER DEFAULT 0
        );
        
        -- Model rules table
        CREATE TABLE IF NOT EXISTS model_rules (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            rule TEXT NOT NULL,
            is_active INTEGER DEFAULT 1,
            created_at TEXT NOT NULL
        );
        
        -- Preference data table for DPO
        CREATE TABLE IF NOT EXISTS preference_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            prompt TEXT NOT NULL,
            chosen TEXT NOT NULL,
            rejected TEXT,
            category TEXT DEFAULT 'general',
            source TEXT DEFAULT 'user_feedback',
            created_at TEXT NOT NULL
        );
        
        -- Create indexes
        CREATE INDEX IF NOT EXISTS idx_messages_conversation ON messages(conversation_id);
        CREATE INDEX IF NOT EXISTS idx_memory_category ON memory_facts(category);
    """)
    db.commit()
    
    # Initialize default settings if not exist
    defaults = {
        "user_bio": "A curious user exploring AI capabilities.",
        "model_temperature": "0.7",
        "model_max_tokens": "512",
        "model_name": "SmolLM2-360M-Instruct",
        "theme": "dark",
        "auto_save": "true",
    }
    now = datetime.utcnow().isoformat()
    for key, value in defaults.items():
        try:
            db.execute(
                "INSERT OR IGNORE INTO user_settings (key, value, updated_at) VALUES (?, ?, ?)",
                (key, value, now)
            )
        except:
            pass
    db.commit()


def generate_conv_id() -> str:
    """Generate unique conversation ID."""
    import uuid
    return f"conv_{uuid.uuid4().hex[:12]}"


# ========================================
# SETTINGS API
# ========================================

@app.route("/api/settings", methods=["GET"])
def get_settings():
    """Get all user settings."""
    db = get_db()
    rows = db.execute("SELECT key, value FROM user_settings").fetchall()
    settings = {row["key"]: row["value"] for row in rows}
    return jsonify({"settings": settings})


@app.route("/api/settings", methods=["POST"])
def update_settings():
    """Update user settings."""
    data = request.get_json(silent=True) or {}
    db = get_db()
    now = datetime.utcnow().isoformat()
    
    for key, value in data.items():
        db.execute(
            "INSERT OR REPLACE INTO user_settings (key, value, updated_at) VALUES (?, ?, ?)",
            (key, str(value), now)
        )
    db.commit()
    return jsonify({"success": True, "updated": list(data.keys())})


@app.route("/api/settings/<key>", methods=["GET"])
def get_setting(key):
    """Get single setting."""
    db = get_db()
    row = db.execute("SELECT value FROM user_settings WHERE key = ?", (key,)).fetchone()
    if row:
        return jsonify({"key": key, "value": row["value"]})
    return jsonify({"error": "Setting not found"}), 404


# ========================================
# MEMORY API
# ========================================

@app.route("/api/memory", methods=["GET"])
def get_memory():
    """Get all memory facts."""
    db = get_db()
    rows = db.execute(
        "SELECT id, fact, category, importance, created_at FROM memory_facts ORDER BY importance DESC, created_at DESC"
    ).fetchall()
    facts = [dict(row) for row in rows]
    return jsonify({"facts": facts})


@app.route("/api/memory", methods=["POST"])
def add_memory():
    """Add a memory fact."""
    data = request.get_json(silent=True) or {}
    fact = data.get("fact", "").strip()
    category = data.get("category", "general")
    importance = data.get("importance", 0)
    
    if not fact:
        return jsonify({"error": "Fact is required"}), 400
    
    db = get_db()
    now = datetime.utcnow().isoformat()
    cursor = db.execute(
        "INSERT INTO memory_facts (fact, category, importance, created_at) VALUES (?, ?, ?, ?)",
        (fact, category, importance, now)
    )
    db.commit()
    return jsonify({"success": True, "id": cursor.lastrowid, "fact": fact})


@app.route("/api/memory/<int:fact_id>", methods=["PUT"])
def update_memory(fact_id):
    """Update a memory fact."""
    data = request.get_json(silent=True) or {}
    fact = data.get("fact", "").strip()
    
    if not fact:
        return jsonify({"error": "Fact content is required"}), 400
    
    db = get_db()
    db.execute("UPDATE memory_facts SET fact = ? WHERE id = ?", (fact, fact_id))
    db.commit()
    return jsonify({"success": True})


@app.route("/api/memory/<int:fact_id>", methods=["DELETE"])
def delete_memory(fact_id):
    """Delete a memory fact."""
    db = get_db()
    db.execute("DELETE FROM memory_facts WHERE id = ?", (fact_id,))
    db.commit()
    return jsonify({"success": True})


# ========================================
# RULES API
# ========================================

@app.route("/api/rules", methods=["GET"])
def get_rules():
    """Get all model rules."""
    db = get_db()
    rows = db.execute("SELECT id, rule, is_active, created_at FROM model_rules ORDER BY created_at DESC").fetchall()
    rules = [dict(row) for row in rows]
    return jsonify({"rules": rules})


@app.route("/api/rules", methods=["POST"])
def add_rule():
    """Add a model rule."""
    data = request.get_json(silent=True) or {}
    rule = data.get("rule", "").strip()
    is_active = data.get("is_active", 1)
    
    if not rule:
        return jsonify({"error": "Rule is required"}), 400
    
    db = get_db()
    now = datetime.utcnow().isoformat()
    cursor = db.execute(
        "INSERT INTO model_rules (rule, is_active, created_at) VALUES (?, ?, ?)",
        (rule, is_active, now)
    )
    db.commit()
    return jsonify({"success": True, "id": cursor.lastrowid})


@app.route("/api/rules/<int:rule_id>", methods=["PUT"])
def update_rule(rule_id):
    """Update a model rule."""
    data = request.get_json(silent=True) or {}
    rule = data.get("rule", "").strip()
    
    if not rule:
        return jsonify({"error": "Rule content is required"}), 400
    
    db = get_db()
    db.execute("UPDATE model_rules SET rule = ? WHERE id = ?", (rule, rule_id))
    db.commit()
    return jsonify({"success": True})


@app.route("/api/rules/<int:rule_id>", methods=["DELETE"])
def delete_rule(rule_id):
    """Delete a model rule."""
    db = get_db()
    db.execute("DELETE FROM model_rules WHERE id = ?", (rule_id,))
    db.commit()
    return jsonify({"success": True})


@app.route("/api/rules/<int:rule_id>/toggle", methods=["POST"])
def toggle_rule(rule_id):
    """Toggle rule active state."""
    db = get_db()
    row = db.execute("SELECT is_active FROM model_rules WHERE id = ?", (rule_id,)).fetchone()
    if not row:
        return jsonify({"error": "Rule not found"}), 404
    
    new_state = 0 if row["is_active"] else 1
    db.execute("UPDATE model_rules SET is_active = ? WHERE id = ?", (new_state, rule_id))
    db.commit()
    return jsonify({"success": True, "is_active": new_state})


# ========================================
# AUTH API
# ========================================

@app.route("/api/auth/signup", methods=["POST"])
def signup():
    data = request.get_json(silent=True) or {}
    username = data.get("username")
    password = data.get("password")
    email = data.get("email")

    if not username or not password:
        return jsonify({"error": "Username and password required"}), 400

    db = get_db()
    user_id = hashlib.sha256(username.encode()).hexdigest()[:12]
    pw_hash = generate_password_hash(password)
    now = datetime.utcnow().isoformat()

    try:
        db.execute(
            "INSERT INTO users (id, username, email, password_hash, created_at) VALUES (?, ?, ?, ?, ?)",
            (user_id, username, email, pw_hash, now)
        )
        db.commit()
        return jsonify({"success": True, "message": "User created"})
    except sqlite3.IntegrityError:
        return jsonify({"error": "Username or email already exists"}), 409

@app.route("/api/auth/login", methods=["POST"])
def login():
    data = request.get_json(silent=True) or {}
    username = data.get("username")
    password = data.get("password")

    if not username or not password:
        return jsonify({"error": "Credentials required"}), 400

    db = get_db()
    user = db.execute("SELECT * FROM users WHERE username = ? OR email = ?", (username, username)).fetchone()

    if not user or not check_password_hash(user["password_hash"], password):
        return jsonify({"error": "Invalid username or password"}), 401

    token = jwt.encode({
        "user_id": user["id"],
        "exp": datetime.now(timezone.utc) + timedelta(days=30)
    }, app.config["SECRET_KEY"], algorithm="HS256")

    return jsonify({
        "success": True,
        "token": token,
        "user": {
            "id": user["id"],
            "username": user["username"],
            "account_type": user["account_type"] or "standard",
            "is_founder": bool(user["is_founder"])
        }
    })

@app.route("/api/auth/maestro", methods=["POST"])
def maestro_auth():
    """Specialized access for Maestro College students."""
    data = request.get_json(silent=True) or {}
    code = data.get("invite_code", "").strip()
    
    # Pattern: Mae + 3 digits (e.g., Mae001) or MAE-XXXX
    is_valid_pattern = re.match(r"^(Mae\d{3}|MAE-[A-Z0-9]{4,})$", code, re.IGNORECASE)
    
    if not is_valid_pattern:
        return jsonify({"error": "Invalid Maestro Access Pattern. Example: Mae001"}), 400
        
    db = get_db()
    # Check if user already exists
    username = code.upper()
    user = db.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
    
    if not user:
        # Create a Maestrian Guest account
        user_id = hashlib.sha256(username.encode()).hexdigest()[:12]
        # Maestro accounts use the code as both username and a default recognizable hash
        pw_hash = generate_password_hash(f"maestro_{code.lower()}")
        now = datetime.utcnow().isoformat()
        db.execute(
            "INSERT INTO users (id, username, password_hash, account_type, invite_code, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (user_id, username, pw_hash, "maestro", code, now)
        )
        db.commit()
        user = db.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()

    token = jwt.encode({
        "user_id": user["id"],
        "exp": datetime.now(timezone.utc) + timedelta(days=7) # Shorter duration for guest/test access
    }, app.config["SECRET_KEY"], algorithm="HS256")

    return jsonify({
        "success": True,
        "token": token,
        "user": {
            "id": user["id"],
            "username": user["username"],
            "account_type": "maestro",
            "is_founder": False
        }
    })

@app.route("/api/auth/guest", methods=["POST"])
def guest_auth():
    """Instant anonymous guest access."""
    db = get_db()
    import random
    guest_id = random.randint(1000, 9999)
    username = f"GUEST_{guest_id}"
    user_id = f"guest_{hashlib.sha256(username.encode()).hexdigest()[:8]}"
    
    # Temporary password
    pw_hash = generate_password_hash(f"guest_pass_{guest_id}")
    now = datetime.utcnow().isoformat()
    
    try:
        db.execute(
            "INSERT INTO users (id, username, password_hash, account_type, created_at) VALUES (?, ?, ?, ?, ?)",
            (user_id, username, pw_hash, "guest", now)
        )
        db.commit()
    except sqlite3.IntegrityError:
        # Retry with different ID if collision (rare)
        return guest_auth()

    token = jwt.encode({
        "user_id": user_id,
        "exp": datetime.now(timezone.utc) + timedelta(hours=24) # 24 hour guest pass
    }, app.config["SECRET_KEY"], algorithm="HS256")

    return jsonify({
        "success": True,
        "token": token,
        "user": {
            "id": user_id,
            "username": username,
            "account_type": "guest",
            "is_founder": False
        }
    })


# ========================================
# CONVERSATIONS API
# ========================================

@app.route("/api/preference", methods=["POST"])
@token_required
def add_preference():
    """Add a chosen/rejected preference pair for DPO."""
    data = request.get_json(silent=True) or {}
    prompt = data.get("prompt", "").strip()
    chosen = data.get("chosen", "").strip()
    rejected = data.get("rejected", "").strip()
    category = data.get("category", "general")
    
    if not prompt or not chosen:
        return jsonify({"error": "Prompt and chosen response required"}), 400
        
    db = get_db()
    now = datetime.utcnow().isoformat()
    db.execute(
        "INSERT INTO preference_data (prompt, chosen, rejected, category, created_at) VALUES (?, ?, ?, ?, ?)",
        (prompt, chosen, rejected, category, now)
    )
    db.commit()
    return jsonify({"success": True})


@app.route("/api/conversations", methods=["GET"])
@token_required
def list_conversations():
    """List all conversations."""
    db = get_db()
    rows = db.execute(
        "SELECT id, title, created_at, updated_at, message_count FROM conversations WHERE user_id = ? ORDER BY updated_at DESC LIMIT 50",
        (request.user_id,)
    ).fetchall()
    conversations = [dict(row) for row in rows]
    return jsonify({"conversations": conversations})


@app.route("/api/conversations", methods=["POST"])
@token_required
def create_conversation():
    """Create new conversation."""
    data = request.get_json(silent=True) or {}
    title = data.get("title", "New Chat")
    
    conv_id = generate_conv_id()
    now = datetime.utcnow().isoformat()
    
    db = get_db()
    db.execute(
        "INSERT INTO conversations (id, user_id, title, created_at, updated_at, message_count) VALUES (?, ?, ?, ?, ?, 0)",
        (conv_id, request.user_id, title, now, now)
    )
    db.commit()
    return jsonify({"success": True, "id": conv_id, "title": title})


@app.route("/api/conversations/<conv_id>", methods=["GET"])
@token_required
def get_conversation(conv_id):
    """Get conversation with messages."""
    db = get_db()
    
    conv = db.execute("SELECT * FROM conversations WHERE id = ? AND user_id = ?", (conv_id, request.user_id)).fetchone()
    if not conv:
        return jsonify({"error": "Conversation not found"}), 404
    
    messages = db.execute(
        "SELECT role, content, created_at FROM messages WHERE conversation_id = ? ORDER BY id ASC",
        (conv_id,)
    ).fetchall()
    
    return jsonify({
        "conversation": dict(conv),
        "messages": [dict(m) for m in messages]
    })


@app.route("/api/conversations/<conv_id>", methods=["DELETE"])
@token_required
def delete_conversation(conv_id):
    """Delete conversation and its messages."""
    db = get_db()
    # Check ownership
    conv = db.execute("SELECT id FROM conversations WHERE id = ? AND user_id = ?", (conv_id, request.user_id)).fetchone()
    if not conv:
        return jsonify({"error": "Forbidden"}), 403

    db.execute("DELETE FROM messages WHERE conversation_id = ?", (conv_id,))
    db.execute("DELETE FROM conversations WHERE id = ?", (conv_id,))
    db.commit()
    return jsonify({"success": True})


@app.route("/api/conversations/<conv_id>/rename", methods=["POST"])
@token_required
def rename_conversation(conv_id):
    """Rename conversation."""
    data = request.get_json(silent=True) or {}
    title = data.get("title", "Untitled")
    
    db = get_db()
    # Check ownership
    conv = db.execute("SELECT id FROM conversations WHERE id = ? AND user_id = ?", (conv_id, request.user_id)).fetchone()
    if not conv:
        return jsonify({"error": "Forbidden"}), 403

    db.execute("UPDATE conversations SET title = ?, updated_at = ? WHERE id = ?", 
               (title, datetime.utcnow().isoformat(), conv_id))
    db.commit()
    return jsonify({"success": True})


@app.route("/api/conversations/<conv_id>/messages", methods=["POST"])
@token_required
def add_message(conv_id):
    """Add message to conversation."""
    data = request.get_json(silent=True) or {}
    role = data.get("role", "user")
    content = data.get("content", "")
    
    if not content:
        return jsonify({"error": "Content required"}), 400
    
    db = get_db()
    now = datetime.utcnow().isoformat()
    
    # Add message
    db.execute(
        "INSERT INTO messages (conversation_id, role, content, created_at) VALUES (?, ?, ?, ?)",
        (conv_id, role, content, now)
    )
    
    # Update conversation stats
    db.execute(
        "UPDATE conversations SET updated_at = ?, message_count = message_count + 1 WHERE id = ?",
        (now, conv_id)
    )
    
    # Auto-rename if first user message
    if role == "user":
        count = db.execute("SELECT COUNT(*) as cnt FROM messages WHERE conversation_id = ? AND role = 'user'", (conv_id,)).fetchone()
        if count["cnt"] == 1:
            title = content[:40] + ("..." if len(content) > 40 else "")
            db.execute("UPDATE conversations SET title = ? WHERE id = ?", (title, conv_id))
    
    db.commit()
    return jsonify({"success": True})


# ========================================
# FILE SYSTEM HELPERS
# ========================================

def load_registry() -> dict[str, str]:
    if REGISTRY_FILE.exists():
        try:
            return json.loads(REGISTRY_FILE.read_text())
        except Exception:
            return {}
    return {}


def save_registry() -> None:
    REGISTRY_FILE.write_text(json.dumps(INDEXED_FILES, indent=2, sort_keys=True))


def model_device():
    if torch is None or model is None:
        return "cpu"
    try:
        return str(next(model.parameters()).device)
    except Exception:
        return "cpu"


def model_type() -> str:
    adapter_files = [Path(MODEL_PATH) / "adapter_model.safetensors", Path(MODEL_PATH) / "adapter_model.bin"]
    if any(p.exists() for p in adapter_files):
        return "fine-tuned"
    if model is not None:
        return "base"
    if model_error:
        return "fallback"
    return "unknown"


def query_uplink(user_msg: str, conversation_history: list[dict]) -> str:
    if requests is None:
        return "[Uplink unavailable: requests dependency missing]"
    payload = {
        "task": user_msg,
        "context": {"conversation": conversation_history[-6:] if conversation_history else []},
    }
    try:
        resp = requests.post(f"{UPLINK_URL}/api/v1/zo/tasks", json=payload, timeout=25)
        data = resp.json()
        result = data.get("result", data.get("error", str(data)))
        if isinstance(result, dict):
            result = result.get("result", str(result))
        return str(result) if result else ""
    except Exception as exc:
        return f"[Agent error: {exc}]"


def load_model() -> None:
    global model, tokenizer, model_error
    if model is not None or model_error:
        return
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel

        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        tokenizer.pad_token = tokenizer.eos_token

        adapter_path = Path(MODEL_PATH)
        has_adapter = adapter_path.exists() and (
            (adapter_path / "adapter_model.safetensors").exists() or
            (adapter_path / "adapter_model.bin").exists()
        )

        if has_adapter:
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                device_map="auto" if torch is not None and torch.cuda.is_available() else None,
                torch_dtype=torch.float16 if torch is not None and torch.cuda.is_available() else torch.float32,
            )
            model = PeftModel.from_pretrained(model, str(adapter_path))
            print(f"[NeuralAI] Fine-tuned model loaded with LoRA adapter from {MODEL_PATH}")
        else:
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                device_map="auto" if torch is not None and torch.cuda.is_available() else None,
                torch_dtype=torch.float16 if torch is not None and torch.cuda.is_available() else torch.float32,
            )
            print(f"[NeuralAI] Base model loaded: {MODEL_NAME}")

        model.eval()
        model_error = None
    except Exception as exc:
        model = None
        tokenizer = None
        model_error = str(exc)
        print(f"[NeuralAI] Model load failed: {exc}")


def get_system_prompt(founder_mode=False) -> str:
    """Build system prompt from user bio, memory, and rules."""
    db = get_db()
    
    # Get user bio and names
    def get_setting(key, default=""):
        row = db.execute("SELECT value FROM user_settings WHERE key = ?", (key,)).fetchone()
        return row["value"] if row else default

    user_bio = get_setting("user_bio", "A curious user exploring AI capabilities.")
    first_name = get_setting("user_first_name", "De'Andrew")
    last_name = get_setting("user_last_name", "Harris")
    full_name = f"{first_name} {last_name}".strip()
    
    # Get active rules
    rules_rows = db.execute("SELECT rule FROM model_rules WHERE is_active = 1").fetchall()
    rules = [r["rule"] for r in rules_rows]
    
    # Get top memory facts
    memory_rows = db.execute(
        "SELECT fact FROM memory_facts ORDER BY importance DESC LIMIT 10"
    ).fetchall()
    memories = [m["fact"] for m in memory_rows]
    
    # System Info
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    system_info = f"""
### SYSTEM ENVIRONMENT
- CURRENT TIME: {now_str}
- PLATFORM: NeuralAI Intelligence Engine v{VERSION}
- ARCHITECTURE: High-Velocity Node (Local)
- STATUS: Production Ready
"""

    # IDENTITY CORE - EXTREME REINFORCEMENT
    identity_block = f"""
### IDENTITY CORE (CRITICAL)
- YOUR NAME: NeuralAI
- YOUR ROLE: Artificial Intelligence Assistant
- USER'S NAME: {full_name}
- USER'S ROLE: Creator, Founder of Harris Holdings, and my Boss.

### MANDATORY RULES:
1. NEVER claim to be {full_name}.
2. NEVER say your name is {first_name} or {last_name}.
3. If asked for your name, ALWAYS say "I am NeuralAI".
4. You are an AI created by {full_name}. You are not a human.
5. If someone calls you {first_name}, politely correct them: "I am NeuralAI, the intelligence engine created by {first_name}."
"""

    # Build prompt
    if founder_mode:
        base = f"You are NeuralAI, the high-velocity intelligence engine for Harris Holdings. You operate with extreme clarity and architectural excellence. You speak directly to your founder, {full_name}."
    else:
        base = f"You are NeuralAI, a helpful AI model designed for creative thinking and high-velocity shipping. You are speaking to {full_name}."
    
    base = identity_block + "\n" + system_info + "\n" + base
    
    if user_bio:
        base += f"\n\n## User Profile\nThe user you are talking to is {full_name}. Here is his bio:\n{user_bio}"
    
    if memories:
        base += "\n\n## Long-Term Memory\n" + "\n".join(f"- {m}" for m in memories)
    
    if rules:
        base += "\n\n## Behavioral Guidelines\n" + "\n".join(f"- {r}" for r in rules)
    
    # FINAL REINFORCEMENT (Most influential position)
    base += f"\n\nFINAL REMINDER: You are NeuralAI. You are NOT {full_name}. You are an AI assistant."
    
    return base


def build_doc_context(user_content: str, file_ids: list[str]) -> str:
    if not file_ids:
        return ""
    try:
        docs = query_documents(user_content, top_k=3)
    except Exception:
        docs = []
    if not docs:
        return ""
    chunks_text = "\n\n---\n\n".join(f"[From {d['source']}]: {d['content']}" for d in docs)
    return f"\n\nRelevant context from uploaded documents:\n{chunks_text}\n"


def build_prompt(messages: list[dict], user_content: str, doc_context: str, founder_mode=False) -> str:
    # Get dynamic system prompt
    system_content = get_system_prompt(founder_mode=founder_mode)
    
    # Add document context if files attached
    if doc_context:
        system_content += "\n\n" + doc_context

    enriched_chat = [{"role": "system", "content": system_content}]
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "").strip()
        if role in ("user", "assistant") and content:
            enriched_chat.append({"role": role, "content": content})
    if not enriched_chat or enriched_chat[-1]["role"] != "user":
        enriched_chat.append({"role": "user", "content": user_content})

    if tokenizer is not None:
        try:
            return tokenizer.apply_chat_template(enriched_chat, tokenize=False, add_generation_prompt=True)
        except Exception:
            pass

    prompt = []
    for msg in enriched_chat:
        prompt.append(f"{msg['role']}\n{msg['content']}")
    prompt.append("assistant")
    return "\n\n".join(prompt)


def answer_with_model_stream(messages: list[dict], user_content: str, doc_context: str, max_new_tokens: int, temperature: float, founder_mode=False):
    """
    Yields tokens from the local model with PROACTIVE identity output guarding.
    """
    try:
        from neuralai_engine import local_model
        
        # Get current user name for guarding
        db = get_db()
        first_name_row = db.execute("SELECT value FROM user_settings WHERE key = 'user_first_name'").fetchone()
        last_name_row = db.execute("SELECT value FROM user_settings WHERE key = 'user_last_name'").fetchone()
        
        fn = first_name_row["value"] if first_name_row else "De'Andrew"
        ln = last_name_row["value"] if last_name_row else "Harris"
        full_n = f"{fn} {ln}".strip()

        # Normalized versions for guarding (stripping non-alpha for robust matching)
        def normalize(s):
            return re.sub(r'[^a-zA-Z]', '', s).lower()
        
        norm_fn = normalize(fn)
        norm_ln = normalize(ln)
        norm_full = normalize(full_n)

        full_formatted_prompt = build_prompt(messages, user_content, doc_context, founder_mode=founder_mode)
        
        cumulative_buffer = ""
        chunk_buffer = ""
        
        for token in local_model.generate_sync_stream(
            full_formatted_prompt, 
            max_new_tokens=max_new_tokens
        ):
            if not token:
                continue
                
            cumulative_buffer += token
            chunk_buffer += token
            
            # If the chunk buffer gets large enough, or we see a sentence end, evaluate
            # We delay output slightly to ensure we don't leak half a name
            if len(chunk_buffer) > 10 or any(c in token for c in [".", "!", "?", "\n"]):
                # Check for identity violations in the cumulative context
                # We check for phrases like "My name is [Name]" or "I am [Name]"
                violations = [
                    f"my name is {fn}", f"my name is {full_n}",
                    f"i am {fn}", f"i am {full_n}",
                    "my name is deandrew", "i am deandrew"
                ]
                
                detected = False
                for v in violations:
                    # Case insensitive check with normalized variants
                    if v.lower() in cumulative_buffer.lower() or normalize(v) in normalize(cumulative_buffer):
                        detected = True
                        break
                
                if detected:
                    # If detected, we perform a hard replacement on the entire cumulative buffer 
                    # but only yield the "new" parts that haven't been sent yet.
                    # Since we are buffering chunks, we can catch it before it leaks too much.
                    # However, to be 100% safe, we'll just replace in the current chunk if it triggered it.
                    protected_chunk = chunk_buffer
                    protected_chunk = re.sub(re.escape(fn), "NeuralAI", protected_chunk, flags=re.IGNORECASE)
                    protected_chunk = re.sub(re.escape(full_n), "NeuralAI", protected_chunk, flags=re.IGNORECASE)
                    protected_chunk = re.sub(r"De'Andrew", "NeuralAI", protected_chunk, flags=re.IGNORECASE)
                    protected_chunk = re.sub(r"De’Andrew", "NeuralAI", protected_chunk, flags=re.IGNORECASE)
                    
                    yield protected_chunk
                else:
                    yield chunk_buffer
                
                chunk_buffer = ""
        
        # Yield remaining buffer
        if chunk_buffer:
            yield chunk_buffer
            
    except Exception as e:
        yield f"I'm online, but the local engine encountered an error: {e}. You said: {user_content}"


def stream_words(text: str):
    """Stream text word by word, preserving newlines."""
    # Split by lines to preserve structure
    lines = text.split('\n')
    for i, line in enumerate(lines):
        if line:
            # Stream words in the line
            words = line.split()
            for word in words:
                yield f"data: {json.dumps({'content': word + ' '})}\n\n"
                time.sleep(0.005)
        # Add newline after each line except the last empty one
        if i < len(lines) - 1:
            yield 'data: {"content": "\n"}\n\n'


INDEXED_FILES = load_registry()
try:
    rebuild_index_registry()
except Exception:
    pass


# ========================================
# ROUTES
# ========================================

@app.route("/sse-test")
def sse_test():
    return render_template("sse_test.html")


# API endpoint for image generation
@app.route("/api/generate-image", methods=["POST"])
def api_generate_image():
    """Generate an image and save to NeuralAI storage."""
    from flask import request
    import subprocess
    import time
    import os
    
    data = request.get_json(silent=True) or {}
    prompt = data.get("prompt", "")
    style = data.get("style", "realistic")
    aspect_ratio = data.get("aspect_ratio", "1:1")
    
    if not prompt:
        return jsonify({"error": "Prompt required"}), 400
    
    # Prepare output directory
    output_dir = "/home/workspace/NeuralAI/images"
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate filename
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    file_stem = f"neuralai_{timestamp}"
    
    # Build full prompt
    full_prompt = f"{prompt}, {style} style" if style else prompt
    
    try:
        # Note: In production, this would call the actual image generation API
        # For now, we'll use a placeholder approach
        import requests
        
        # This endpoint would normally call OpenAI/Google/etc.
        # Return the expected file info
        return jsonify({
            "success": True,
            "file_stem": file_stem,
            "output_dir": output_dir,
            "image_url": f"/neuralai/images/{file_stem}.jpg",
            "prompt": full_prompt,
            "message": "Image generation initiated. Check /neuralai/images/ for results."
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Serve generated images from NeuralAI storage
@app.route("/generated_images/<filename>")
@app.route("/neuralai/images/<filename>")  
def serve_neuralai_image(filename):
    from flask import send_from_directory
    import os
    
    # Check if file exists in structured storage
    filepath = IMAGE_STORAGE / filename
    if filepath.exists():
        return send_from_directory(str(IMAGE_STORAGE), filename)
    
    return "Image not found", 404

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/privacy")
def privacy():
    return render_template("privacy.html")


@app.route("/terms")
def terms():
    return render_template("terms.html")


@app.route("/api/files/<folder>/<path:filename>")
def serve_file(folder, filename):
    from flask import send_from_directory
    if folder == "generated":
        directory = IMAGE_STORAGE
    else:
        directory = UPLOAD_FOLDER
    
    filepath = directory / filename
    if filepath.exists():
        return send_from_directory(str(directory), filename)
    
    return "File not found", 404


@app.route("/api/status", methods=["GET"])
def get_status():
    db = get_db()
    device = model_device()
    rules_count = db.execute("SELECT COUNT(*) FROM model_rules WHERE is_active = 1").fetchone()[0]
    return jsonify({
        "status": "ready",
        "version": "5.2.1-maintenance",
        "device": device,
        "active_rules": rules_count
    })


@app.route("/api/user/me", methods=["GET"])
def get_user_me():
    db = get_db()
    
    def get_setting(key, default=""):
        row = db.execute("SELECT value FROM user_settings WHERE key = ?", (key,)).fetchone()
        return row["value"] if row else default

    return jsonify({
        "user": {
            "first_name": get_setting("user_first_name", "De'Andrew"),
            "last_name": get_setting("user_last_name", "Harris"),
            "email": get_setting("user_email", "deandrewharris@zo.computer"),
            "username": get_setting("user_username", "deandrewharris"),
            "bio": get_setting("user_bio", "A curious user exploring AI capabilities.")
        }
    })


@app.route("/api/user/update", methods=["POST"])
def update_user():
    data = request.get_json(silent=True) or {}
    db = get_db()
    now = datetime.now(timezone.utc).isoformat()
    
    for key, val in data.items():
        if key in ["first_name", "last_name", "email", "username", "bio"]:
            db_key = f"user_{key}" if key != "bio" else "user_bio"
            db.execute(
                "INSERT OR REPLACE INTO user_settings (key, value, updated_at) VALUES (?, ?, ?)",
                (db_key, val, now)
            )
    
    db.commit()
    return jsonify({"success": True})


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"ok": True, "version": VERSION})


@app.route("/api/files", methods=["GET"])
def list_files():
    files_list = []
    # Recursively find all files in UPLOAD_FOLDER
    if UPLOAD_FOLDER.exists():
        for f in UPLOAD_FOLDER.rglob("*"):
            if f.is_file():
                # Use relative path so we can serve it back
                rel_path = f.relative_to(UPLOAD_FOLDER)
                files_list.append({
                    "id": hashlib.sha256(str(rel_path).encode()).hexdigest()[:16],
                    "name": str(rel_path),
                    "type": "upload",
                    "size": f.stat().st_size
                })
    
    # Also add generated images
    if IMAGE_STORAGE.exists():
        for f in IMAGE_STORAGE.rglob("*"):
            if f.is_file():
                rel_path = f.relative_to(IMAGE_STORAGE)
                files_list.append({
                    "id": hashlib.sha256(str(rel_path).encode()).hexdigest()[:16],
                    "name": str(rel_path),
                    "type": "generated",
                    "size": f.stat().st_size
                })

    return jsonify({"files": files_list})


@app.route("/api/files/<file_id>", methods=["DELETE"])
def delete_file(file_id):
    filename = None
    filepath_to_delete = None
    
    # Check INDEXED_FILES first
    if file_id in INDEXED_FILES:
        filename = INDEXED_FILES[file_id]
        del INDEXED_FILES[file_id]
        save_registry()
        filepath_to_delete = UPLOAD_FOLDER / filename
    else:
        # Search by hashing relative paths
        if UPLOAD_FOLDER.exists():
            for f in UPLOAD_FOLDER.rglob("*"):
                if f.is_file():
                    rel_path = str(f.relative_to(UPLOAD_FOLDER))
                    if hashlib.sha256(rel_path.encode()).hexdigest()[:16] == file_id:
                        filename = rel_path
                        filepath_to_delete = f
                        break
        
        # Search generated images
        if not filename and IMAGE_STORAGE.exists():
            for f in IMAGE_STORAGE.rglob("*"):
                if f.is_file():
                    rel_path = str(f.relative_to(IMAGE_STORAGE))
                    if hashlib.sha256(rel_path.encode()).hexdigest()[:16] == file_id:
                        filename = rel_path
                        filepath_to_delete = f
                        break

    if not filename or not filepath_to_delete:
        return jsonify({"error": "File not found"}), 404

    try:
        if filepath_to_delete.exists():
            filepath_to_delete.unlink()
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    return jsonify({"success": True, "deleted": filename})


@app.route("/api/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED:
        return jsonify({"error": f"Unsupported type: {ext}"}), 400

    filename = secure_filename(file.filename)
    filepath = UPLOAD_FOLDER / filename
    file.save(filepath)

    # Sync to Cloud
    if cloud_client:
        try:
            cloud_client.upload_file(str(filepath), filename)
        except Exception as e:
            print(f"[NeuralDrive] Cloud sync failed: {e}")

    result = index_document(str(filepath))
    file_id = result.get("file_id", hashlib.sha256(filename.encode()).hexdigest()[:16])
    INDEXED_FILES[file_id] = filename
    save_registry()

    return jsonify(
        {
            "success": True,
            "filename": filename,
            "file_id": file_id,
            "chunks": result.get("chunks", 0),
            "message": f'"{filename}" indexed — {result.get("chunks", 0)} chunks ready.',
        }
    )



@app.route("/api/chat", methods=["POST"])
@token_required
def chat():
    data = request.get_json(silent=True) or {}
    messages = data.get("messages", []) or []
    prompt_only = data.get("prompt", "")
    conv_id = data.get("conversation_id")  # NEW: conversation ID for persistence
    force_local = data.get("force_local", False)
    founder_mode = data.get("founder_mode", False)
    
    # Get settings from DB
    db = get_db()
    temp_row = db.execute("SELECT value FROM user_settings WHERE key = 'model_temperature'").fetchone()
    tokens_row = db.execute("SELECT value FROM user_settings WHERE key = 'model_max_tokens'").fetchone()
    
    max_new_tokens = int(data.get("max_tokens", tokens_row["value"] if tokens_row else 512))
    temperature = float(data.get("temperature", temp_row["value"] if temp_row else 0.7))
    file_ids = data.get("file_ids", []) or []

    def generate():
        last_user = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                last_user = msg.get("content", "").strip()
                break
        user_content = last_user or prompt_only

        if not user_content:
            yield f"data: {json.dumps({'error': 'No message content'})}\n\n"
            yield "data: [DONE]\n\n"
            return

        # Save user message to conversation
        if conv_id:
            now = datetime.utcnow().isoformat()
            db_inner = get_db()
            # Ensure ownership
            conv_check = db_inner.execute("SELECT id FROM conversations WHERE id = ? AND user_id = ?", (conv_id, request.user_id)).fetchone()
            if not conv_check:
                 yield f"data: {json.dumps({'error': 'Unauthorized conversation'})}\n\n"
                 yield "data: [DONE]\n\n"
                 return

            db_inner.execute(
                "INSERT INTO messages (conversation_id, role, content, created_at) VALUES (?, ?, ?, ?)",
                (conv_id, "user", user_content, now)
            )
            db_inner.execute(
                "UPDATE conversations SET updated_at = ?, message_count = message_count + 1 WHERE id = ?",
                (now, conv_id)
            )
            # Auto-rename if first message
            count = db_inner.execute("SELECT COUNT(*) as cnt FROM messages WHERE conversation_id = ? AND role = 'user'", (conv_id,)).fetchone()
            if count["cnt"] == 1:
                title = user_content[:40] + ("..." if len(user_content) > 40 else "")
                db_inner.execute("UPDATE conversations SET title = ? WHERE id = ?", (title, conv_id))
            db_inner.commit()

        # NEW ROUTING: Use clean router
        if force_local:
            route, tool = "local", None
        else:
            route, tool = neuralai_route(user_content)

        if route == "tool":
            # Execute tool using sync wrapper
            full_response = ""
            for chunk in run_tool_sync(tool, user_content):
                full_response += chunk
                if chunk:
                    if "\n" in chunk:
                        for i, part in enumerate(chunk.split("\n")):
                            if part:
                                yield f"data: {json.dumps({'content': part})}\n\n"
                            if i < len(chunk.split("\n")) - 1:
                                yield 'data: {"content": "\n"}\n\n'
                    else:
                        yield f"data: {json.dumps({'content': chunk})}\n\n"
            
            # Save assistant response
            if conv_id:
                now = datetime.utcnow().isoformat()
                db_inner = get_db()
                db_inner.execute(
                    "INSERT INTO messages (conversation_id, role, content, created_at) VALUES (?, ?, ?, ?)",
                    (conv_id, "assistant", full_response, now)
                )
                db_inner.execute(
                    "UPDATE conversations SET updated_at = ?, message_count = message_count + 1 WHERE id = ?",
                    (now, conv_id)
                )
                db_inner.commit()
            
            yield "data: [DONE]\n\n"
            return

        if route == "uplink":
            msg_val3 = '[Neural Uplink] Routing to agent network...\\n'
            yield f"data: {json.dumps({'content': msg_val3})}\n\n"
            agent_response = query_uplink(user_content, messages)
            for chunk in stream_words(agent_response):
                yield chunk
            
            # Save assistant response
            if conv_id:
                now = datetime.utcnow().isoformat()
                db_inner = get_db()
                db_inner.execute(
                    "INSERT INTO messages (conversation_id, role, content, created_at) VALUES (?, ?, ?, ?)",
                    (conv_id, "assistant", agent_response, now)
                )
                db_inner.execute(
                    "UPDATE conversations SET updated_at = ?, message_count = message_count + 1 WHERE id = ?",
                    (now, conv_id)
                )
                db_inner.commit()
            
            yield "data: [DONE]\n\n"
            return

        # DEFAULT: Local model
        doc_context = build_doc_context(user_content, file_ids)
        
        full_response = ""
        for chunk in answer_with_model_stream(messages, user_content, doc_context, max_new_tokens, temperature, founder_mode=founder_mode):
            if chunk:
                # Format for SSE - stream chunk by chunk directly
                # Replace newlines so they don't break SSE format
                if "\n" in chunk:
                    for i, part in enumerate(chunk.split("\n")):
                        if part:
                            yield f"data: {json.dumps({'content': part})}\n\n"
                        if i < len(chunk.split("\n")) - 1:
                            yield 'data: {"content": "\n"}\n\n'
                else:
                    yield f"data: {json.dumps({'content': chunk})}\n\n"
                full_response += chunk
        
        # Save assistant response
        if conv_id:
            now = datetime.utcnow().isoformat()
            db_inner = get_db()
            db_inner.execute(
                "INSERT INTO messages (conversation_id, role, content, created_at) VALUES (?, ?, ?, ?)",
                (conv_id, "assistant", full_response, now)
            )
            db_inner.execute(
                "UPDATE conversations SET updated_at = ?, message_count = message_count + 1 WHERE id = ?",
                (now, conv_id)
            )
            db_inner.commit()
        
        yield "data: [DONE]\n\n"

    headers = {
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",
    }
    return Response(stream_with_context(generate()), mimetype="text/event-stream", headers=headers)


# Initialize database on startup
with app.app_context():
    init_db()
    print(f"[NeuralAI] Database initialized at {DATABASE}")

    # Pre-load model on startup to avoid first-request delay
    print("[NeuralAI] Pre-loading model...")
    load_model()
    from neuralai_engine import local_model
    try:
        for _ in local_model.generate_sync_stream("Warmup", max_new_tokens=3):
            pass
        print("[NeuralAI] Model warmup complete. Ready!")
    except Exception as w:
        print(f"[NeuralAI] Warmup warning: {w}")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT, debug=False)
