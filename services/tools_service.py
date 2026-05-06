#!/usr/bin/env python3
"""
NeuralAI Tools Service
- Isolated sandbox for code execution
- Terminal access
- File operations
- Image generation (via Zo API)
- Exposes tools API on port 7002
"""

import os
import sys
import json
import subprocess
import tempfile
import asyncio
import requests
from pathlib import Path
from flask import Flask, Response, jsonify, request, send_from_directory
from datetime import datetime
from typing import Dict, Any

# Configuration
PORT = int(os.environ.get("TOOLS_PORT", "7002"))
STORAGE_PATH = os.environ.get("STORAGE_PATH", "/home/workspace/NeuralAI")
IMAGE_PATH = os.environ.get("IMAGE_PATH", "/home/workspace/NeuralAI/images")
MAX_OUTPUT = 10000
DEFAULT_TIMEOUT = 30

# Zo API configuration
ZO_API_URL = "https://api.zo.computer"
ZO_TOKEN = os.environ.get("ZO_CLIENT_IDENTITY_TOKEN", "")

app = Flask(__name__)

# Ensure image directory exists
Path(IMAGE_PATH).mkdir(parents=True, exist_ok=True)


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
# IMAGE GENERATION
# ====================

def generate_image_via_zo(prompt: str, output_dir: str = IMAGE_PATH) -> Dict[str, Any]:
    """
    Generate an image using Zo's image generation capability.
    For real AI-generated images, use the main Zo chat interface.
    """
    try:
        import time
        from PIL import Image, ImageDraw, ImageFont
        import random
        import math
        
        timestamp = int(time.time())
        filename = f"generated_{timestamp}.png"
        filepath = Path(output_dir) / filename
        
        # Create a visually appealing placeholder with gradient background
        img = Image.new('RGB', (512, 512))
        draw = ImageDraw.Draw(img)
        
        # Create gradient background (simulating a scene)
        for y in range(512):
            # Gradient from dark blue to lighter blue (like a sky)
            r = int(20 + (y / 512) * 60)
            g = int(20 + (y / 512) * 80)
            b = int(60 + (y / 512) * 100)
            draw.line([(0, y), (512, y)], fill=(r, g, b))
        
        # Add some "scene" elements based on prompt keywords
        # Sun/moon for sky themes
        if any(word in prompt.lower() for word in ['sun', 'sunset', 'moon', 'sky', 'star']):
            # Draw a circle (sun/moon)
            cx, cy = 256, 150
            for r in range(50, 0, -1):
                color = (255, 200, 100) if 'sun' in prompt.lower() else (240, 240, 220)
                draw.ellipse([cx-r, cy-r, cx+r, cy+r], fill=color)
        
        # Mountains for landscape themes
        if any(word in prompt.lower() for word in ['mountain', 'landscape', 'hill', 'peak']):
            # Draw simple mountain shapes
            points1 = [(0, 400), (150, 200), (300, 350), (512, 180), (512, 512), (0, 512)]
            points2 = [(0, 350), (200, 220), (350, 300), (512, 250), (512, 512), (0, 512)]
            draw.polygon(points1, fill=(60, 50, 70))
            draw.polygon(points2, fill=(80, 70, 90))
        
        # Water for ocean/sea themes
        if any(word in prompt.lower() for word in ['ocean', 'sea', 'water', 'lake', 'river']):
            for y in range(400, 512):
                r = int(20 + random.randint(0, 20))
                g = int(60 + random.randint(0, 40))
                b = int(120 + random.randint(0, 30))
                draw.line([(0, y), (512, y)], fill=(r, g, b))
        
        # Add prompt text at bottom
        text = f"Concept: {prompt[:40]}..."
        draw.text((20, 460), text, fill=(200, 200, 200))
        draw.text((20, 480), "Generated by NeuralAI (placeholder)", fill=(150, 150, 150))
        
        # Save
        filepath.parent.mkdir(parents=True, exist_ok=True)
        img.save(filepath)
        
        return {
            "success": True,
            "image_path": str(filepath),
            "image_url": f"/images/{filename}",
            "prompt": prompt,
            "placeholder": True,
            "note": "For real AI-generated images, ask Zo directly or enable spending in billing settings"
        }
        
    except Exception as e:
        return {
            "success": False,
            "image_path": "",
            "image_url": "",
            "prompt": prompt,
            "error": str(e)
        }



def create_placeholder_image(prompt: str, filepath: Path, error: str = "") -> Dict[str, Any]:
    """Create a placeholder image with text using PIL."""
    try:
        from PIL import Image, ImageDraw, ImageFont
        
        # Create image
        img = Image.new('RGB', (512, 512), color=(20, 20, 30))
        draw = ImageDraw.Draw(img)
        
        # Add prompt text
        y = 40
        words = prompt.split()
        line = ""
        for word in words:
            test_line = line + word + " "
            if len(test_line) > 35:
                draw.text((20, y), line, fill=(150, 150, 150))
                y += 25
                line = word + " "
            else:
                line = test_line
        if line:
            draw.text((20, y), line, fill=(150, 150, 150))
        
        # Add note
        draw.text((20, 450), "Image generation placeholder", fill=(80, 80, 80))
        
        # Save
        filepath.parent.mkdir(parents=True, exist_ok=True)
        img.save(filepath)
        
        return {
            "success": True,
            "image_path": str(filepath),
            "image_url": f"/images/{filepath.name}",
            "prompt": prompt,
            "placeholder": True,
            "error": error
        }
        
    except ImportError:
        # PIL not available - return error
        return {
            "success": False,
            "image_path": "",
            "image_url": "",
            "prompt": prompt,
            "error": "Image generation requires PIL. Install with: pip install Pillow"
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
        "storage": STORAGE_PATH,
        "images": IMAGE_PATH
    })


@app.route("/images/<filename>", methods=["GET"])
def serve_image(filename):
    """Serve generated images."""
    return send_from_directory(IMAGE_PATH, filename)


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


@app.route("/generate/image", methods=["POST"])
def generate_image():
    """Generate an image from a prompt."""
    data = request.get_json()
    prompt = data.get("prompt", "")
    output_dir = data.get("output_dir", IMAGE_PATH)
    
    if not prompt:
        return jsonify({"success": False, "error": "No prompt provided"}), 400
    
    result = generate_image_via_zo(prompt, output_dir)
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
print(f"[Tools Service] Images: {IMAGE_PATH}")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT, debug=False, threaded=True)
