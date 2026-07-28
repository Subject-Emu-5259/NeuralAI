import sys

with open('/home/workspace/Projects/NeuralAI/services/neural_core_service.py', 'r') as f:
    content = f.read()

# Replace generate_response_stream signature and logic
old_stream = '''def generate_response_stream(prompt, max_tokens=256, temperature=0.7, system_prompt=None):
    global model, tokenizer, inference_count
    if model is None or tokenizer is None:
        yield "Model not loaded."
        return
    
    if system_prompt is None:
        system_prompt = DEFAULT_SYSTEM_PROMPT
        
    try:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": prompt})
        
        if tokenizer.chat_template:
            full = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            full = f"system\\n{system_prompt}\\nuser\\n{prompt}" if system_prompt else f"user\\n{prompt}"'''

new_stream = '''def generate_response_stream(prompt, max_tokens=256, temperature=0.7, system_prompt=None, history=None):
    global model, tokenizer, inference_count
    if model is None or tokenizer is None:
        yield "Model not loaded."
        return
    
    if system_prompt is None:
        system_prompt = DEFAULT_SYSTEM_PROMPT
        
    try:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        if history:
            messages.extend(history)
        else:
            messages.append({"role": "user", "content": prompt})
        
        if tokenizer.chat_template:
            full = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            full = ""
            for msg in messages:
                full += f"<|im_start|>{msg['role']}\\n{msg['content']}<|im_end|>\\n"
            full += "<|im_start|>assistant\\n"'''

if old_stream not in content:
    print("Failed to find old stream logic!")
    sys.exit(1)

content = content.replace(old_stream, new_stream)

# Replace the chat endpoint call
old_call = 'for chunk in generate_response_stream(prompt, max_tokens, temperature, system_prompt=system_prompt):'
new_call = 'for chunk in generate_response_stream(prompt, max_tokens, temperature, system_prompt=system_prompt, history=messages):'

if old_call not in content:
    print("Failed to find chat endpoint call!")
    sys.exit(1)

content = content.replace(old_call, new_call)

with open('/home/workspace/Projects/NeuralAI/services/neural_core_service.py', 'w') as f:
    f.write(content)

print("Successfully patched.")
