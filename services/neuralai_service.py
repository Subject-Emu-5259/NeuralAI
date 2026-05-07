#!/usr/bin/env python3
"""
NeuralAI Unified Service
- Model inference (port 7001)
- Tools execution (port 7002)
- All in one process, shared memory
"""

import os
import sys
import json
import torch
import subprocess
import tempfile
import asyncio
from pathlib import Path
from datetime import datetime
from flask import Flask, Response, jsonify, request
from typing import Dict, Any

# CPU optimization
torch.set_num_threads(4)

# Configuration
PORT = int(os.environ.get("NEURAL_PORT", "7001"))
MODEL_PATH = os.environ.get("MODEL_PATH", "/home/workspace/Projects/NeuralAI/checkpoints/v2_model")
BASE_MODEL = os.environ.get("BASE_MODEL", "HuggingFaceTB/SmolLM2-360M-Instruct")
STORAGE_PATH = os.environ.get("STORAGE_PATH", "/home/workspace/NeuralAI")

app = Flask(__name__)

# Shared state
model = None
tokenizer = None
model_status = "loading"
inference_count = 0

# ====================
# MODEL LOADING
# ====================

def load_model():
    """Load model once on startup."""
    global model, tokenizer, model_status
    
    print(f"[NeuralAI] Loading model from {MODEL_PATH}")
    
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
        
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        tokenizer.pad_token = tokenizer.eos_token
        
        adapter_path = Path(MODEL_PATH)
        adapter_bin = adapter_path / "adapter_model.bin"
        adapter_safetensors = adapter_path / "adapter_model.safetensors"
        
        if adapter_path.exists() and (adapter_bin.exists() or adapter_safetensors.exists()):
            print(f"[NeuralAI] Loading with LoRA adapter...")
            base = AutoModelForCausalLM.from_pretrained(
                BASE_MODEL,
                torch_dtype=torch.float32,
                device_map=None,
                low_cpu_mem_usage=True
            )
            model = PeftModel.from_pretrained(base, str(adapter_path))
        else:
            print(f"[NeuralAI] Loading base model...")
            model = AutoModelForCausalLM.from_pretrained(
                BASE_MODEL,
                torch_dtype=torch.float32,
                device_map=None,
                low_cpu_mem_usage=True
            )
        
        model.eval()
        model_status = "ready"
        print(f"[NeuralAI] ✓ Model ready! Parameters: {sum(p.numel() for p in model.parameters()):,}")
        
    except Exception as e:
        import traceback
        model_status = "error"
        print(f"[NeuralAI] ✗ Failed: {e}")
        traceback.print_exc()


# ====================
# MODEL ENDPOINTS
# ====================

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": model_status,
        "inference_count": inference_count,
        "model": BASE_MODEL,
        "storage": STORAGE_PATH,
        "images": f"{STORAGE_PATH}/images",
        "port": PORT
    })


@app.route("/status", methods=["GET"])
def status():
    return jsonify({
        "status": model_status,
        "model_loaded": model is not None,
        "tokenizer_loaded": tokenizer is not None,
        "inference_count": inference_count,
        "model_path": MODEL_PATH,
        "storage": STORAGE_PATH
    })


@app.route("/generate", methods=["POST"])
def generate():
    """Generate text (non-streaming)."""
    global inference_count
    
    if model is None or tokenizer is None:
        return jsonify({"error": "Model not loaded"}), 503
    
    try:
        data = request.get_json()
        prompt = data.get("prompt", data.get("message", ""))
        max_tokens = data.get("max_tokens", 256)
        temperature = data.get("temperature", 0.7)
        
        if not prompt:
            return jsonify({"error": "No prompt provided"}), 400
        
        full_prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        inputs = tokenizer(full_prompt, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=0.95,
                pad_token_id=tokenizer.eos_token_id
            )
        
        new_tokens = outputs[0][inputs["input_ids"].shape[-1]:]
        response = tokenizer.decode(new_tokens, skip_special_tokens=True)
        
        inference_count += 1
        return jsonify({"response": response, "tokens": len(new_tokens)})
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/generate/stream", methods=["POST"])
def generate_stream():
    """Generate text with streaming."""
    global inference_count
    
    if model is None or tokenizer is None:
        return jsonify({"error": "Model not loaded"}), 503
    
    try:
        from transformers import TextIteratorStreamer
        import threading
        
        data = request.get_json()
        prompt = data.get("prompt", data.get("message", ""))
        max_tokens = data.get("max_tokens", 256)
        temperature = data.get("temperature", 0.7)
        
        full_prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        inputs = tokenizer(full_prompt, return_tensors="pt")
        
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        
        thread = threading.Thread(target=model.generate, kwargs=dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=0.95,
            pad_token_id=tokenizer.eos_token_id
        ))
        thread.start()
        
        def generate():
            for token in streamer:
                yield f"data: {json.dumps({'token': token})}\n\n"
            yield "data: [DONE]\n\n"
        
        inference_count += 1
        return Response(generate(), mimetype="text/event-stream",
                       headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ====================
# TOOLS ENDPOINTS
# ====================

@app.route("/execute/code", methods=["POST"])
def execute_code():
    """Execute code in sandbox."""
    data = request.get_json()
    code = data.get("code", "")
    language = data.get("language", "python")
    timeout = data.get("timeout", 30)
    
    import time
    start = time.time()
    
    suffix = ".py" if language == "python" else ".js"
    with tempfile.NamedTemporaryFile(mode='w', suffix=suffix, delete=False, encoding='utf-8') as f:
        f.write(code)
        temp_path = f.name
    
    try:
        if language == "python":
            result = subprocess.run(['python3', temp_path], capture_output=True, text=True, timeout=timeout)
        else:
            result = subprocess.run(['node', temp_path], capture_output=True, text=True, timeout=timeout)
        
        return jsonify({
            "success": result.returncode == 0,
            "output": result.stdout[:10000],
            "error": result.stderr[:10000] if result.returncode != 0 else "",
            "exit_code": result.returncode,
            "execution_time": time.time() - start
        })
    except subprocess.TimeoutExpired:
        return jsonify({"success": False, "error": f"Timeout after {timeout}s", "exit_code": -1})
    except Exception as e:
        return jsonify({"success": False, "error": str(e), "exit_code": -1})
    finally:
        try:
            os.unlink(temp_path)
        except:
            pass


@app.route("/execute/shell", methods=["POST"])
def execute_shell():
    """Execute shell command."""
    data = request.get_json()
    command = data.get("command", "")
    timeout = data.get("timeout", 30)
    
    import time
    start = time.time()
    
    try:
        result = subprocess.run(['bash', '-c', command], capture_output=True, text=True, timeout=timeout)
        return jsonify({
            "success": result.returncode == 0,
            "output": result.stdout[:10000],
            "error": result.stderr[:10000] if result.returncode != 0 else "",
            "exit_code": result.returncode,
            "execution_time": time.time() - start
        })
    except subprocess.TimeoutExpired:
        return jsonify({"success": False, "error": f"Timeout after {timeout}s", "exit_code": -1})
    except Exception as e:
        return jsonify({"success": False, "error": str(e), "exit_code": -1})


@app.route("/generate/image", methods=["POST"])
def generate_image():
    """Generate placeholder image."""
    data = request.get_json()
    prompt = data.get("prompt", "concept")
    
    try:
        from PIL import Image, ImageDraw, ImageFont
        
        # Create image directory
        image_dir = Path(STORAGE_PATH) / "images"
        image_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"generated_{timestamp}.jpg"
        filepath = image_dir / filename
        
        # Create image
        img = Image.new('RGB', (512, 512), color=(20, 25, 45))
        draw = ImageDraw.Draw(img)
        
        # Add gradient
        for i in range(512):
            draw.line([(0, i), (512, i)], fill=(20 + i//10, 25 + i//15, 45 + i//20))
        
        # Add sun/moon for sky themes
        if any(w in prompt.lower() for w in ['sun', 'moon', 'sky', 'sunset', 'sunrise', 'night']):
            draw.ellipse([200, 80, 312, 192], fill=(255, 200, 100) if 'sun' in prompt.lower() else (220, 220, 240))
        
        # Add mountains for landscape themes
        if any(w in prompt.lower() for w in ['mountain', 'landscape', 'hill', 'valley']):
            draw.polygon([(0, 380), (150, 280), (300, 350), (512, 300), (512, 512), (0, 512)], 
                        fill=(30, 25, 50))
        
        # Add text
        text = prompt[:40] + "..." if len(prompt) > 40 else prompt
        draw.text((20, 460), f"Concept: {text}", fill=(180, 180, 200))
        draw.text((20, 485), "Generated by NeuralAI", fill=(100, 100, 120))
        
        img.save(filepath, quality=85)
        
        return jsonify({
            "success": True,
            "image_path": str(filepath),
            "image_url": f"/images/{filename}",
            "prompt": prompt,
            "placeholder": True
        })
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


@app.route("/files/list", methods=["GET"])
def files_list():
    """List files in storage."""
    directory = request.args.get("directory", "")
    try:
        base = Path(STORAGE_PATH)
        target = base / directory if directory else base
        
        if not target.exists():
            return jsonify({"success": False, "error": "Not found"})
        
        files = []
        for item in target.iterdir():
            files.append({
                "name": item.name,
                "type": "directory" if item.is_dir() else "file",
                "size": item.stat().st_size if item.is_file() else 0
            })
        
        return jsonify({"success": True, "files": files, "path": str(target)})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


# Load model on startup
print(f"[NeuralAI] Starting unified service on port {PORT}")
load_model()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT, debug=False, threaded=True)
