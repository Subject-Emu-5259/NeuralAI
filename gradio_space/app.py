# NeuralAI v2 Gradio Space
# This Space
# Requires HF PRO for Gradio Spaces

import os
import torch
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from huggingface_hub import snapshot_download

# --- WORKAROUND: gradio-client 1.1.x-1.3.0 crashes on every / request ----------
# Gradio 4.x emits a JSON schema with `additionalProperties: true` (a bool).
# gradio_client.utils.get_type() does `if "const" in schema:` which raises
# `TypeError: argument of type 'bool' is not iterable` when schema is a bool,
# so the / route returns HTTP 500 and the healthcheck fails. Patch get_type
# to treat any non-dict schema as "Any" so the API info builds cleanly.
import gradio_client.utils as _gcu

_orig_json_schema = _gcu._json_schema_to_python_type  # capture real impl

def _safe_json_schema(schema, defs):
    # Gradio 4.x emits `additionalProperties: true` (a bool). gradio_client
    # 1.1.x-1.3.0 then calls _json_schema_to_python_type(True) which crashes
    # ('bool' object has no attribute 'get'). Treat any non-dict schema as Any.
    if not isinstance(schema, dict):
        return "Any"
    return _orig_json_schema(schema, defs)

_gcu._json_schema_to_python_type = _safe_json_schema
# ------------------------------------------------------------------------------

# Configuration
BASE_MODEL = "HuggingFaceTB/SmolLM2-360M-Instruct"
ADAPTER_REPO = "Subject-Emu-5259/NeuralAI"
HF_TOKEN = os.environ.get("HF_TOKEN")  # optional; repo is public

# Lazy-load the model so the server binds its port immediately
# (Render's port scanner times out if loading blocks startup).
tokenizer = None
model = None

def load_model():
    global tokenizer, model
    if model is not None:
        return
    print("Loading base model...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    # 8-bit quantization keeps the 360M model under the 512MB free-tier RAM
    # limit, but it REQUIRES CUDA + bitsandbytes. Railway (and other CPU-only
    # hosts) have no GPU, so fall back to a plain fp32 CPU load there.
    use_8bit = torch.cuda.is_available()
    print(f"CUDA available: {torch.cuda.is_available()} -> 8-bit={use_8bit}", flush=True)
    if use_8bit:
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            load_in_8bit=True,
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            torch_dtype="auto",
            trust_remote_code=True,
        ).to("cpu")
    # Always pull the LATEST adapter from the Hub on each (re)start so a fresh
    # `train_dpo.py --push` is reflected without a rebuild. snapshot_download
    # checks the Hub for updates and only re-downloads if the adapter changed.
    print(f"Fetching latest LoRA adapter from {ADAPTER_REPO}...", flush=True)
    adapter_path = snapshot_download(
        repo_id=ADAPTER_REPO,
        repo_type="model",
        token=HF_TOKEN,
        local_dir="/tmp/neuralai_adapter",
    )
    print("Loading LoRA adapter...", flush=True)
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    print("Model ready!", flush=True)

SYSTEM_PROMPT = (
    "You are NeuralAI v2, a helpful, concise assistant. "
    "Answer in plain text. Use short paragraphs. No markdown unless asked."
)

def format_prompt(history, user_message):
    """Format conversation history into a prompt."""
    prompt = SYSTEM_PROMPT + "\n\n"
    for user, assistant in history:
        prompt += f"User: {user}\nAssistant: {assistant}\n"
    prompt += f"User: {user_message}\nAssistant:"
    return prompt

def chat(message, history):
    """Generate response from the model."""
    load_model()
    prompt = format_prompt(history, message)
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Decode only the new tokens
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return response.strip()

# Gradio interface
# Gradio 6.0 moved `theme`/`css` from gr.Blocks() to launch() and renamed
# `server_port` -> `port`. Support both 4.x and 6.x via version detection so
# the container binds $PORT regardless of which major is installed.
_GV = tuple(int(p) for p in gr.__version__.split(".")[:2])
_BLOCKS_KWARGS = {"title": "NeuralAI v2 Chat"}
_LAUNCH_THEME_CSS = {}
if _GV >= (5, 0):
    _LAUNCH_THEME_CSS = {
        "theme": gr.themes.Soft(),
        "css": """
        .gradio-container { max-width: 900px !important; }
        .chat-message { font-size: 15px; }
        """,
    }
else:
    _BLOCKS_KWARGS["theme"] = gr.themes.Soft()
    _BLOCKS_KWARGS["css"] = """
    .gradio-container { max-width: 900px !important; }
    .chat-message { font-size: 15px; }
    """

with gr.Blocks(**_BLOCKS_KWARGS) as demo:
    gr.Markdown("""
    # 🧠 NeuralAI v2 — Chat Demo
    
    **LoRA adapter** for **SmolLM2-360M-Instruct** (DPO-aligned)
    
    This Space runs the **fine-tuned adapter** locally with PEFT merging — no Inference API needed.
    """)
    
    chatbot = gr.Chatbot(
        label="NeuralAI v2",
        height=500,
        bubble_full_width=False,
    )
    
    with gr.Row():
        msg = gr.Textbox(
            placeholder="Ask NeuralAI anything...",
            scale=9,
            container=False,
            show_label=False,
        )
        send = gr.Button("Send", variant="primary", scale=1)
    
    clear = gr.Button("Clear", variant="secondary")
    
    gr.Markdown("""
    ---
    **Model:** `Subject-Emu-5259/NeuralAI` (LoRA, rank 16, α 32)  
    **Base:** `HuggingFaceTB/SmolLM2-360M-Instruct`  
    **Training:** LoRA + DPO, 3 epochs, 363 samples  
    **Author:** De'Andrew P. Harris
    """)
    
    def respond(message, chat_history):
        if not message.strip():
            return "", chat_history
        bot_message = chat(message, chat_history)
        chat_history.append((message, bot_message))
        return "", chat_history
    
    msg.submit(respond, [msg, chatbot], [msg, chatbot])
    send.click(respond, [msg, chatbot], [msg, chatbot])
    clear.click(lambda: None, None, chatbot, queue=False)

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 7860))
    # Gradio 6.x renamed server_port -> port. Use the right kwarg per version
    # so the host's assigned $PORT is actually bound (otherwise it defaults to
    # 7860 and the port scanner finds nothing).
    _launch_kwargs = {
        "server_name": "0.0.0.0",
        "share": False,
        "show_error": True,
        **_LAUNCH_THEME_CSS,
    }
    if _GV >= (5, 0):
        _launch_kwargs["port"] = port
    else:
        _launch_kwargs["server_port"] = port
    demo.launch(**_launch_kwargs)