# NeuralAI v2 Gradio Space
# This Space
# Requires HF PRO for Gradio Spaces

import os
import torch
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Configuration
BASE_MODEL = "HuggingFaceTB/SmolLM2-360M-Instruct"
ADAPTER_REPO = "Subject-Emu-5259/NeuralAI"

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
    # 8-bit quantization keeps the 360M model under the 512MB free-tier RAM limit
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        load_in_8bit=True,
        device_map="auto",
        trust_remote_code=True,
    )
    print("Loading LoRA adapter...", flush=True)
    model = PeftModel.from_pretrained(model, ADAPTER_REPO)
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
with gr.Blocks(
    title="NeuralAI v2 Chat",
    theme=gr.themes.Soft(),
    css="""
    .gradio-container { max-width: 900px !important; }
    .chat-message { font-size: 15px; }
    """
) as demo:
    gr.Markdown("""
    # 🧠 NeuralAI v2 — Chat Demo
    
    **LoRA adapter** for **SmolLM2-360M-Instruct** (DPO-aligned)
    
    This Space runs the **fine-tuned adapter** locally with PEFT merging — no Inference API needed.
    """)
    
    chatbot = gr.Chatbot(
        label="NeuralAI v2",
        height=500,
        show_copy_button=True,
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
    # Gradio 6.x renamed server_port -> port; use `port` so Render's assigned
    # $PORT is actually bound (otherwise it defaults to 7860 and Render's
    # port scanner finds nothing).
    demo.launch(
        server_name="0.0.0.0",
        port=port,
        share=False,
        show_error=True,
    )