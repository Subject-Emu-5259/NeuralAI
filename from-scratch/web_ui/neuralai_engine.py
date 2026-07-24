# neuralai_engine.py - NeuralAI Engine v2.0
# Local model + Tools (terminal, code, images) + Streaming

import asyncio
import torch
from typing import AsyncGenerator, Dict, Any, List, Tuple
import aiohttp
import asyncio.subprocess as asp
import os
import sys
from pathlib import Path
import json
import time
import subprocess

# CPU optimization
torch.set_num_threads(4)

# Import tools
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

try:
    from tools.code_sandbox import CodeSandbox
    from tools.file_manager import FileManager
    from tools.web_fetcher import WebFetcher
    from tools.db_connector import DatabaseConnector
    from tools.git_assistant import GitAssistant
    
    code_sandbox = CodeSandbox()
    file_manager = FileManager()
    web_fetcher = WebFetcher()
    db_connector = DatabaseConnector()
    git_assistant = GitAssistant()
except ImportError as e:
    print(f"[NeuralAI Engine] Import Error: {e}")
    code_sandbox = None
    file_manager = None
    web_fetcher = None
    db_connector = None
    git_assistant = None

# Uplink ports
UPLINK_BASE = "http://localhost"
DIALOG_PORT = 7101
DATA_PORT = 7102
OPS_PORT = 7103
WORLD_PORT = 7104

# Model globals
model = None
tokenizer = None
model_error = None


def load_local_model():
    global model, tokenizer, model_error
    if model is not None or model_error:
        return
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
        
        base_model = "HuggingFaceTB/SmolLM2-360M-Instruct"
        adapter_path = Path(__file__).resolve().parent.parent.parent / "checkpoints" / "v2_model"
        
        tokenizer = AutoTokenizer.from_pretrained(base_model)
        tokenizer.pad_token = tokenizer.eos_token
        
        base = AutoModelForCausalLM.from_pretrained(
            base_model, torch_dtype=torch.float32, device_map=None, low_cpu_mem_usage=True
        )
        
        adapter_file = adapter_path / "adapter_model.safetensors"
        if adapter_path.exists() and adapter_file.exists():
            model = PeftModel.from_pretrained(base, str(adapter_path))
        else:
            model = base
        
        model.eval()
        model_error = None
        print("[NeuralAI] Model loaded")
    except Exception as e:
        model = tokenizer = None
        model_error = str(e)
        print(f"[NeuralAI] Model error: {e}")


class LocalModel:
    def generate_sync_stream(self, prompt: str, max_new_tokens: int = 512, temperature: float = 0.7, repetition_penalty: float = 1.3):
        load_local_model()
        if model is None or tokenizer is None:
            for ch in "[Model] Not loaded":
                yield ch
            return
        try:
            from transformers import TextIteratorStreamer
            import threading
            
            full_prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
            inputs = tokenizer(full_prompt, return_tensors="pt")
            streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
            
            thread = threading.Thread(target=model.generate, kwargs={
                **inputs, "streamer": streamer, "max_new_tokens": max_new_tokens,
                "do_sample": True, "temperature": temperature, "top_p": 0.95,
                "repetition_penalty": repetition_penalty,
                "pad_token_id": tokenizer.eos_token_id
            })
            thread.start()
            for text in streamer:
                yield text
        except Exception as e:
            yield f"[Error] {e}"

    async def generate(self, prompt: str, max_new_tokens: int = 256):
        load_local_model()
        if model is None or tokenizer is None:
            for ch in "[Model] Not loaded":
                yield ch
            return
        try:
            full_prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
            inputs = tokenizer(full_prompt, return_tensors="pt")
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=max_new_tokens,
                    do_sample=True, temperature=0.7, top_p=0.95, pad_token_id=tokenizer.eos_token_id)
            text = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
            for ch in text:
                yield ch
        except Exception as e:
            for ch in f"[Error] {e}":
                yield ch


local_model = LocalModel()


def neuralai_route(msg: str) -> Tuple[str, str | None]:
    try:
        from neuralai_router import neuralai_route as _route
        return _route(msg)
    except:
        lower = msg.lower()
        if any(k in lower for k in ["research", "analyze", "debug"]):
            return ("uplink", None)
        return ("local", None)


async def neuralai_local(prompt: str):
    async for token in local_model.generate(prompt):
        yield token


async def neuralai_uplink(prompt: str) -> str:
    async with aiohttp.ClientSession() as session:
        tasks = [
            session.post(f"{UPLINK_BASE}:{p}/task", json={"goal": prompt}, timeout=aiohttp.ClientTimeout(total=30))
            for p in [DIALOG_PORT, DATA_PORT, OPS_PORT, WORLD_PORT]
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
    return "[Uplink] Processing..."


async def neuralai_tool_call(tool: str, msg: str):
    from neuralai_router import extract_tool_params
    params = extract_tool_params(msg, tool)
    
    # Image generation
    if tool == "image_gen":
        prompt = params.get("prompt", msg)
        style = params.get("style", "realistic")
        aspect = params.get("aspect_ratio", "1:1")
        
        yield f"🎨 **Generating: {prompt}**\n\n"
        
        output_dir = "/home/workspace/NeuralAI/images"
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        file_stem = f"neuralai_{timestamp}"
        full_prompt = f"{prompt}, {style} style" if style else prompt
        
        try:
            result = subprocess.run([
                "python3", "-c",
                f'''
import sys
sys.path.insert(0, "/home/.z/tools")
from generate_image import generate_image as gen
r = gen(prompt="{full_prompt.replace(chr(34), chr(92)+chr(34))}", file_stem="{file_stem}", output_dir="{output_dir}", aspect_ratio="{aspect}")
print("OK" if r else "FAIL")
'''
            ], capture_output=True, text=True, timeout=120)
            
            if "OK" in result.stdout:
                yield f"![{prompt}](/neuralai/images/{file_stem}_1.png)\n\n"
                yield f"✅ Saved to `/NeuralAI/images/`\n"
            else:
                yield "❌ Generation failed\n"
        except Exception as e:
            yield f"❌ Error: {e}\n"
        return
    
    # Terminal
    if tool == "terminal":
        cmd = msg
        for p in ["run ", "execute ", "shell "]:
            if msg.lower().startswith(p):
                cmd = msg[len(p):]
                break
        yield "```bash\n"
        proc = await asp.create_subprocess_shell(cmd, stdout=asp.PIPE, stderr=asp.PIPE)
        while True:
            line = await proc.stdout.readline()
            if not line: break
            yield line.decode()
        yield "```\n"
        return
    
    # Code execution
    if tool == "code_exec" and code_sandbox:
        code = params.get("code", msg)
        yield "[Sandbox] Running...\n```"
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, code_sandbox.run_python, code)
        yield result.get("output", result.get("error", "No output"))
        yield "\n```\n"
        return
    
    # Code generation
    if tool == "code_gen" and code_sandbox:
        yield "[NeuralAI] Writing code...\n"
        code_text = ""
        async for c in local_model.generate(f"Write Python for: {msg}", max_new_tokens=512):
            code_text += c
        import re
        m = re.search(r"```python\s*([\s\S]*?)```", code_text)
        if m:
            code = m.group(1).strip()
            yield f"```python\n{code}\n```\n"
            result = await asyncio.get_event_loop().run_in_executor(None, code_sandbox.run_python, code)
            yield "Output:\n```\n" + (result.get("output") or result.get("error")) + "\n```\n"
        return
    
    # File manager
    if tool == "file_manager" and file_manager:
        query = params.get("query", msg)
        result = await asyncio.get_event_loop().run_in_executor(None, file_manager.search, query)
        if result.get("success"):
            for r in result.get("results", [])[:5]:
                yield f"- {r['path']}\n"
        return
    
    # Git
    if tool == "git" and git_assistant:
        git_assistant.repo_path = Path("/home/workspace/Projects/NeuralAI")
        result = await asyncio.get_event_loop().run_in_executor(None, git_assistant.status)
        if result.get("success"):
            yield f"Branch: {result['branch']}\n"
        return
    
    yield f"[Tool] {tool} pending"


async def stream_text(text: str):
    for ch in text:
        yield ch


async def neuralai_chat(msg: str):
    route, tool = neuralai_route(msg)
    if route == "local":
        async for t in neuralai_local(msg):
            yield t
    elif route == "uplink":
        yield "[Uplink] Connecting...\n"
        resp = await neuralai_uplink(msg)
        async for t in stream_text(resp):
            yield t
    elif route == "tool":
        async for t in neuralai_tool_call(tool, msg):
            yield t
    else:
        async for t in stream_text(f"[NeuralAI] {msg}"):
            yield t


# Warmup
if os.environ.get("NEURALAI_WARMUP", "true") != "false":
    try:
        load_local_model()
    except:
        pass
