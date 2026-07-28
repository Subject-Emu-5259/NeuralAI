#!/usr/bin/env python3
"""
NeuralAI Model Service
- Loads model once on startup
- Keeps in memory
- Exposes inference API on port 7001
- Handles both sync and streaming responses
"""

import os
import sys
import json
import torch
from pathlib import Path
from flask import Flask, Response, jsonify, request
from datetime import datetime

# CPU optimization
torch.set_num_threads(4)

# Configuration
PORT = int(os.environ.get("MODEL_PORT", "7001"))
MODEL_PATH = os.environ.get("MODEL_PATH", "/home/workspace/Projects/NeuralAI/checkpoints/v2_model")
BASE_MODEL = os.environ.get("BASE_MODEL", "HuggingFaceTB/SmolLM2-360M-Instruct")

app = Flask(__name__)

# Global model state
model = None
tokenizer = None
model_status = "loading"
model_error = None
inference_count = 0


def load_model():
    """Load model once on startup."""
    global model, tokenizer, model_status, model_error
    
    print(f"[Model Service] Loading model from {MODEL_PATH}")
    print(f"[Model Service] Base model: {BASE_MODEL}")
    
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        tokenizer.pad_token = tokenizer.eos_token
        
        # Check for adapter
        adapter_path = Path(MODEL_PATH)
        adapter_bin = adapter_path / "adapter_model.bin"
        adapter_safetensors = adapter_path / "adapter_model.safetensors"
        
        if adapter_path.exists() and (adapter_bin.exists() or adapter_safetensors.exists()):
            print(f"[Model Service] Loading with LoRA adapter...")
            base = AutoModelForCausalLM.from_pretrained(
                BASE_MODEL,
                torch_dtype=torch.float32,
                device_map=None,
                low_cpu_mem_usage=True
            )
            model = PeftModel.from_pretrained(base, str(adapter_path))
            print(f"[Model Service] LoRA adapter loaded!")
        else:
            print(f"[Model Service] Loading base model only...")
            model = AutoModelForCausalLM.from_pretrained(
                BASE_MODEL,
                torch_dtype=torch.float32,
                device_map=None,
                low_cpu_mem_usage=True
            )
        
        model.eval()
        model_status = "ready"
        model_error = None
        
        params = sum(p.numel() for p in model.parameters())
        print(f"[Model Service] ✓ Model ready! Parameters: {params:,}")
        print(f"[Model Service] Listening on port {PORT}")
        
    except Exception as e:
        import traceback
        model_status = "error"
        model_error = str(e)
        print(f"[Model Service] ✗ Failed to load model: {e}")
        traceback.print_exc()


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({
        "status": model_status,
        "error": model_error,
        "inference_count": inference_count,
        "model": BASE_MODEL,
        "port": PORT
    })


@app.route("/status", methods=["GET"])
def status():
    """Detailed status endpoint."""
    return jsonify({
        "status": model_status,
        "error": model_error,
        "inference_count": inference_count,
        "model_loaded": model is not None,
        "tokenizer_loaded": tokenizer is not None,
        "model_path": MODEL_PATH,
        "base_model": BASE_MODEL,
        "device": "cpu",
        "threads": 4
    })


@app.route("/generate", methods=["POST"])
def generate():
    """Generate text response (non-streaming)."""
    global inference_count
    
    if model is None or tokenizer is None:
        return jsonify({"error": "Model not loaded", "status": model_status}), 503
    
    try:
        data = request.get_json()
        prompt = data.get("prompt", "")
        max_tokens = data.get("max_tokens", 256)
        temperature = data.get("temperature", 0.7)
        
        if not prompt:
            return jsonify({"error": "No prompt provided"}), 400
        
        # Build full prompt with chat template
        if not prompt.startswith("<|im_start|>"):
            full_prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        else:
            full_prompt = prompt
        
        # Tokenize
        inputs = tokenizer(full_prompt, return_tensors="pt")
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=0.95,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decode only new tokens
        new_tokens = outputs[0][inputs["input_ids"].shape[-1]:]
        response = tokenizer.decode(new_tokens, skip_special_tokens=True)
        
        inference_count += 1
        
        return jsonify({
            "response": response,
            "tokens_generated": len(new_tokens),
            "inference_count": inference_count
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/generate/stream", methods=["POST"])
def generate_stream():
    """Generate text response with streaming."""
    global inference_count
    
    if model is None or tokenizer is None:
        return jsonify({"error": "Model not loaded"}), 503
    
    try:
        from transformers import TextIteratorStreamer
        import threading
        
        data = request.get_json()
        prompt = data.get("prompt", "")
        max_tokens = data.get("max_tokens", 256)
        temperature = data.get("temperature", 0.7)
        
        if not prompt:
            return jsonify({"error": "No prompt provided"}), 400
        
        # Build full prompt
        if not prompt.startswith("<|im_start|>"):
            full_prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        else:
            full_prompt = prompt
        
        inputs = tokenizer(full_prompt, return_tensors="pt")
        
        # Create streamer
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        
        # Run generation in thread
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
        
        return Response(
            generate(),
            mimetype="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
        )
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Load model on startup
print(f"[Model Service] Starting...")
print(f"[Model Service] Port: {PORT}")
load_model()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT, debug=False, threaded=True)
