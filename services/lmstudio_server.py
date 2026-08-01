#!/usr/bin/env python3
"""NeuralAI inference server with a vocabulary-friendly chat format.

This custom wrapper replaces the bare-llama_cpp.server CLI so we can register
a chat format that matches how the NeuralAI Mamba models were trained, then
run llama_cpp.server with that format.
"""
import sys

# Register the custom chat format BEFORE importing Llama.
import llama_cpp.llama_chat_format as chat_fmt

def format_neuralai_intel(messages, **kwargs):
    """No-special-token formatter.

    Works with GPT-NeoX / Mamba tokenizers that only have `<|endoftext|>` as a
    special token.  The model learned that after the last `### Assistant:`
    line it should emit the response.
    """
    system = ""
    for m in messages:
        if m.get("role") == "system":
            system = m.get("content", "")
    prompt_parts = []
    if system:
        prompt_parts.append(f"### System:\n{system}")
    for m in messages:
        role = m.get("role", "")
        content = m.get("content", "")
        if role == "user":
            prompt_parts.append(f"### User:\n{content}")
        elif role == "assistant" and content:
            # Include prior assistant turns in context without a separator
            prompt_parts.append(f"### Assistant:\n{content}")
    prompt_parts.append("### Assistant:")
    prompt = "\n".join(prompt_parts)
    return chat_fmt.ChatFormatterResponse(prompt=prompt, stop=["### User:", "### System:"])

chat_fmt.register_chat_format("neuralai-intel")(format_neuralai_intel)

from llama_cpp.server.__main__ import main

if __name__ == "__main__":
    sys.argv[0] = "llama_cpp.server"
    main()
