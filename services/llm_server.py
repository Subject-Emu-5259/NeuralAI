#!/usr/bin/env python3
"""
Lightweight LLM inference server using ctransformers (GGUF models).
Runs on port 1234 with OpenAI-compatible /v1/chat/completions API.
Designed for ZO Computer's 4GB RAM constraint.
"""
import json
import os
import sys
import time
import uuid
from http.server import HTTPServer, BaseHTTPRequestHandler
from threading import Thread

MODEL_PATH = os.environ.get("LLM_MODEL_PATH", os.path.expanduser("~/models/smollm2-360m-instruct-q4_k_m.gguf"))
MODEL_TYPE = os.environ.get("LLM_MODEL_TYPE", "llama")
PORT = int(os.environ.get("LLM_PORT", "1234"))
CONTEXT_LENGTH = int(os.environ.get("LLM_CONTEXT_LENGTH", "2048"))
MAX_TOKENS = int(os.environ.get("LLM_MAX_TOKENS", "512"))

llm = None

def load_model():
    global llm
    try:
        from ctransformers import AutoModelForCausalLM
        print(f"[llm_server] Loading model from {MODEL_PATH} ...")
        llm = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            model_type=MODEL_TYPE,
            max_new_tokens=MAX_TOKENS,
            context_length=CONTEXT_LENGTH,
        )
        print(f"[llm_server] Model loaded successfully. Context={CONTEXT_LENGTH}, MaxTokens={MAX_TOKENS}")
        return True
    except Exception as e:
        print(f"[llm_server] FAILED to load model: {e}", file=sys.stderr)
        return False


def generate(messages, temperature=0.7, max_tokens=None):
    """Generate a response from messages array (OpenAI format)."""
    global llm
    if llm is None:
        return "Error: Model not loaded"

    # Build a simple prompt from messages
    prompt = ""
    system_msg = None
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        if role == "system":
            system_msg = content
            prompt += f"<|system|>\n{content}\n"
        elif role == "user":
            prompt += f"<|user|>\n{content}\n"
        elif role == "assistant":
            prompt += f"<|assistant|>\n{content}\n"

    if system_msg is None:
        prompt = f"<|system|>\nYou are a helpful assistant.\n" + prompt
    prompt += "<|assistant|>\n"

    tokens = max_tokens or MAX_TOKENS
    result = llm(prompt, max_new_tokens=tokens, temperature=temperature, stop=["</s>", "<|end|>"])
    return result.strip() if isinstance(result, str) else str(result)


class LLMHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/v1/models" or self.path == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            resp = {
                "object": "list",
                "data": [
                    {
                        "id": "smollm2-360m-instruct",
                        "object": "model",
                        "created": int(time.time()),
                        "owned_by": "neuralai",
                    }
                ],
            }
            if self.path == "/health":
                resp = {"status": "ok", "model": MODEL_PATH, "loaded": llm is not None}
            self.wfile.write(json.dumps(resp).encode())
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length) if content_length > 0 else b""

        if self.path == "/v1/chat/completions":
            try:
                data = json.loads(body)
                messages = data.get("messages", [])
                temperature = data.get("temperature", 0.7)
                max_tokens = data.get("max_tokens", MAX_TOKENS)
                stream = data.get("stream", False)

                response_text = generate(messages, temperature=temperature, max_tokens=max_tokens)

                if stream:
                    # Streaming SSE response
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Cache-Control", "no-cache")
                    self.send_header("Connection", "keep-alive")
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.end_headers()

                    chat_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
                    # Send the full response as one chunk
                    chunk = {
                        "id": chat_id,
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": data.get("model", "smollm2-360m-instruct"),
                        "choices": [{
                            "index": 0,
                            "delta": {"content": response_text},
                            "finish_reason": None,
                        }],
                    }
                    self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                    # Send finish
                    done_chunk = {
                        "id": chat_id,
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": data.get("model", "smollm2-360m-instruct"),
                        "choices": [{
                            "index": 0,
                            "delta": {},
                            "finish_reason": "stop",
                        }],
                    }
                    self.wfile.write(f"data: {json.dumps(done_chunk)}\n\n".encode())
                    self.wfile.write(b"data: [DONE]\n\n")
                else:
                    # Non-streaming JSON response
                    resp = {
                        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
                        "object": "chat.completion",
                        "created": int(time.time()),
                        "model": data.get("model", "smollm2-360m-instruct"),
                        "choices": [{
                            "index": 0,
                            "message": {"role": "assistant", "content": response_text},
                            "finish_reason": "stop",
                        }],
                        "usage": {
                            "prompt_tokens": 0,
                            "completion_tokens": 0,
                            "total_tokens": 0,
                        },
                    }
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.end_headers()
                    self.wfile.write(json.dumps(resp).encode())

            except Exception as e:
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                error_resp = {"error": {"message": str(e), "type": "server_error"}}
                self.wfile.write(json.dumps(error_resp).encode())
        else:
            self.send_response(404)
            self.end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.end_headers()

    def log_message(self, format, *args):
        print(f"[llm_server] {args[0]}")


def main():
    print(f"[llm_server] Starting on port {PORT}...")
    print(f"[llm_server] Model path: {MODEL_PATH}")
    print(f"[llm_server] Model type: {MODEL_TYPE}")

    if not os.path.exists(MODEL_PATH):
        print(f"[llm_server] ERROR: Model file not found at {MODEL_PATH}", file=sys.stderr)
        print(f"[llm_server] Download a GGUF model first!", file=sys.stderr)
        sys.exit(1)

    if not load_model():
        print("[llm_server] Failed to load model, exiting.", file=sys.stderr)
        sys.exit(1)

    server = HTTPServer(("0.0.0.0", PORT), LLMHandler)
    print(f"[llm_server] ✅ Server running on http://0.0.0.0:{PORT}")
    print(f"[llm_server] API: POST http://localhost:{PORT}/v1/chat/completions")
    server.serve_forever()


if __name__ == "__main__":
    main()
