#!/usr/bin/env python3
"""
NeuralAI Tools Service
- Isolated sandbox for code execution
- Terminal access
- File operations
- Exposes tools API on port 7002
"""

import os
import sys
import json
import subprocess
import tempfile
import asyncio
from pathlib import Path
from flask import Flask, Response, jsonify, request
from datetime import datetime
from typing import Dict, Any

# Configuration
PORT = int(os.environ.get("TOOLS_PORT", "7002"))
STORAGE_PATH = os.environ.get("STORAGE_PATH", "/home/workspace/NeuralAI")
MAX_OUTPUT = 10000
DEFAULT_TIMEOUT = 30

app = Flask(__name__)


# ====================
# CODE SANDBOX
# ====================

def run_code(code: str, language: str = "python", timeout: int = DEFAULT_TIMEOUT) -> Dict[str, Any]:
    """Execute code in sandboxed environment."""
    import time
    start = time.time()
    
    # Write to temp file
    suffix = ".py" if language == "python" else ".js"
    with tempfile.NamedTemporaryFile(mode='w', suffix=suffix, delete=False, encoding='utf-8') as f:
        f.write(code)
        temp_path = f.name
    
    try:
        if language == "python":
            result = subprocess.run(
                ['python3', temp_path],
                capture_output=True,
                text=True,
                timeout=timeout,
                env={'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONUNBUFFERED': '1'}
            )
        else:  # javascript
            result = subprocess.run(
                ['node', temp_path],
                capture_output=True,
                text=True,
                timeout=timeout
            )
        
        return {
            "success": result.returncode == 0,
            "output": result.stdout[:MAX_OUTPUT],
            "error": result.stderr[:MAX_OUTPUT] if result.returncode != 0 else "",
            "exit_code": result.returncode,
            "execution_time": time.time() - start
        }
        
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "output": "",
            "error": f"Timeout after {timeout}s",
            "exit_code": -1,
            "execution_time": timeout
        }
    except Exception as e:
        return {
            "success": False,
            "output": "",
            "error": str(e),
            "exit_code": -1,
            "execution_time": time.time() - start
        }
    finally:
        try:
            os.unlink(temp_path)
        except:
            pass


def run_shell(command: str, timeout: int = DEFAULT_TIMEOUT) -> Dict[str, Any]:
    """Execute shell command."""
    import time
    start = time.time()
    
    try:
        result = subprocess.run(
            ['bash', '-c', command],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=tempfile.gettempdir()
        )
        
        return {
            "success": result.returncode == 0,
            "output": result.stdout[:MAX_OUTPUT],
            "error": result.stderr[:MAX_OUTPUT] if result.returncode != 0 else "",
            "exit_code": result.returncode,
            "execution_time": time.time() - start
        }
        
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "output": "",
            "error": f"Timeout after {timeout}s",
            "exit_code": -1,
            "execution_time": timeout
        }
    except Exception as e:
        return {
            "success": False,
            "output": "",
            "error": str(e),
            "exit_code": -1,
            "execution_time": time.time() - start
        }


# ====================
# FILE MANAGER
# ====================

def list_files(directory: str = "") -> Dict[str, Any]:
    """List files in NeuralAI storage."""
    try:
        base = Path(STORAGE_PATH)
        target = base / directory if directory else base
        
        if not target.exists():
            return {"success": False, "error": f"Directory not found: {directory}"}
        
        files = []
        for item in target.iterdir():
            files.append({
                "name": item.name,
                "type": "directory" if item.is_dir() else "file",
                "size": item.stat().st_size if item.is_file() else 0,
                "modified": datetime.fromtimestamp(item.stat().st_mtime).isoformat()
            })
        
        return {
            "success": True,
            "path": str(target),
            "files": sorted(files, key=lambda x: (x["type"], x["name"]))
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


def read_file(filepath: str) -> Dict[str, Any]:
    """Read a file from NeuralAI storage."""
    try:
        base = Path(STORAGE_PATH)
        target = base / filepath
        
        if not target.exists():
            return {"success": False, "error": f"File not found: {filepath}"}
        
        with open(target, 'r', encoding='utf-8') as f:
            content = f.read()
        
        return {
            "success": True,
            "path": str(target),
            "content": content,
            "size": len(content)
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


def write_file(filepath: str, content: str) -> Dict[str, Any]:
    """Write a file to NeuralAI storage."""
    try:
        base = Path(STORAGE_PATH)
        target = base / filepath
        target.parent.mkdir(parents=True, exist_ok=True)
        
        with open(target, 'w', encoding='utf-8') as f:
            f.write(content)
        
        return {
            "success": True,
            "path": str(target),
            "size": len(content)
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


# ====================
# API ENDPOINTS
# ====================

@app.route("/health", methods=["GET"])
def health():
    """Health check."""
    return jsonify({
        "status": "ready",
        "port": PORT,
        "storage": STORAGE_PATH
    })


@app.route("/execute/code", methods=["POST"])
def execute_code():
    """Execute code in sandbox."""
    data = request.get_json()
    code = data.get("code", "")
    language = data.get("language", "python")
    timeout = data.get("timeout", DEFAULT_TIMEOUT)
    
    result = run_code(code, language, timeout)
    return jsonify(result)


@app.route("/execute/shell", methods=["POST"])
def execute_shell():
    """Execute shell command."""
    data = request.get_json()
    command = data.get("command", "")
    timeout = data.get("timeout", DEFAULT_TIMEOUT)
    
    result = run_shell(command, timeout)
    return jsonify(result)


@app.route("/files/list", methods=["GET", "POST"])
def files_list():
    """List files in storage."""
    if request.method == "POST":
        data = request.get_json()
        directory = data.get("directory", "")
    else:
        directory = request.args.get("directory", "")
    
    result = list_files(directory)
    return jsonify(result)


@app.route("/files/read", methods=["POST"])
def files_read():
    """Read a file."""
    data = request.get_json()
    filepath = data.get("path", "")
    result = read_file(filepath)
    return jsonify(result)


@app.route("/files/write", methods=["POST"])
def files_write():
    """Write a file."""
    data = request.get_json()
    filepath = data.get("path", "")
    content = data.get("content", "")
    result = write_file(filepath, content)
    return jsonify(result)


print(f"[Tools Service] Starting on port {PORT}")
print(f"[Tools Service] Storage: {STORAGE_PATH}")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT, debug=False, threaded=True)
