import os
import shutil
import json
from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory
from datetime import datetime

app = Flask(__name__)

# Config
STORAGE_ROOT = Path("/home/workspace/Projects/NeuralAI/storage")
STORAGE_ROOT.mkdir(parents=True, exist_ok=True)
PORT = int(os.environ.get("STORAGE_PORT", "7003"))

def get_file_info(path: Path):
    stat = path.stat()
    return {
        "name": path.name,
        "is_dir": path.is_dir(),
        "size": stat.st_size if path.is_file() else 0,
        "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
        "extension": path.suffix[1:] if path.is_file() else ""
    }

@app.route("/api/storage/list", methods=["GET"])
def list_storage():
    rel_path = request.args.get("path", "")
    target = (STORAGE_ROOT / rel_path).resolve()
    
    # Security check: ensure path is within STORAGE_ROOT
    if not str(target).startswith(str(STORAGE_ROOT)):
        return jsonify({"error": "Unauthorized path"}), 403
    
    if not target.exists():
        return jsonify({"error": "Path not found"}), 404
    
    items = []
    for item in target.iterdir():
        if item.name.startswith("."): continue
        items.append(get_file_info(item))
    
    # Sort: directories first, then alphabetical
    items.sort(key=lambda x: (not x["is_dir"], x["name"].lower()))
    
    return jsonify({
        "path": rel_path,
        "items": items
    })

@app.route("/api/storage/upload", methods=["POST"])
def upload_file():
    rel_path = request.args.get("path", "")
    target_dir = (STORAGE_ROOT / rel_path).resolve()
    
    if not str(target_dir).startswith(str(STORAGE_ROOT)):
        return jsonify({"error": "Unauthorized path"}), 403
    
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    target_dir.mkdir(parents=True, exist_ok=True)
    file_path = target_dir / file.filename
    file.save(str(file_path))
    
    return jsonify({"success": True, "info": get_file_info(file_path)})

@app.route("/api/storage/mkdir", methods=["POST"])
def make_dir():
    data = request.get_json() or {}
    rel_path = data.get("path", "")
    target = (STORAGE_ROOT / rel_path).resolve()
    
    if not str(target).startswith(str(STORAGE_ROOT)):
        return jsonify({"error": "Unauthorized path"}), 403
    
    target.mkdir(parents=True, exist_ok=True)
    return jsonify({"success": True})

@app.route("/api/storage/delete", methods=["DELETE"])
def delete_item():
    rel_path = request.args.get("path", "")
    target = (STORAGE_ROOT / rel_path).resolve()
    
    if not str(target).startswith(str(STORAGE_ROOT)):
        return jsonify({"error": "Unauthorized path"}), 403
    
    if not target.exists():
        return jsonify({"error": "Not found"}), 404
    
    if target.is_dir():
        shutil.rmtree(target)
    else:
        target.unlink()
        
    return jsonify({"success": True})

@app.route("/api/storage/download", methods=["GET"])
def download_file():
    rel_path = request.args.get("path", "")
    target = (STORAGE_ROOT / rel_path).resolve()
    
    if not str(target).startswith(str(STORAGE_ROOT)) or target.is_dir():
        return jsonify({"error": "Unauthorized path"}), 403
    
    return send_from_directory(target.parent, target.name)

@app.route("/api/storage/search", methods=["GET"])
def search_files():
    query = request.args.get("q", "").lower()
    if not query: return jsonify([])
    
    results = []
    for p in STORAGE_ROOT.rglob("*"):
        if p.name.startswith("."): continue
        if query in p.name.lower():
            results.append({
                "name": p.name,
                "path": str(p.relative_to(STORAGE_ROOT)),
                "is_dir": p.is_dir()
            })
    return jsonify(results[:50])

@app.route("/health")
def health():
    return jsonify({"status": "ready", "service": "NeuralStorage"})

if __name__ == "__main__":
    print(f"NeuralAI Storage Service starting on port {PORT}...")
    app.run(host="0.0.0.0", port=PORT, debug=False, threaded=True)
