"""
Mamba SSM Chat Formatter - NeuralAI's Mamba model prompt engineering.
Provides chat template formatting specifically tuned for state-space models.

Unlike Transformer models, Mamba SSM models need tightly-structured prompts
with clear separators to maintain coherence across the linear state pass.

Template format: <|user|>\\n{message}\\n<|assistant|>\\n
"""

MAMBA_CHAT_TEMPLATE = "<|user|>\n{message}\n<|assistant|>\n"
MAMBA_SYSTEM_PREFIX = "<|system|>\n{system}\n"
MAMBA_MULTITURN_TEMPLATE = "<|user|>\n{message}\n<|assistant|>\n{response}\n"

def detect_mamba_model(model_type):
    """Check if the active model is a Mamba SSM architecture."""
    return isinstance(model_type, str) and model_type.lower() == "mamba"

class MambaChatFormatter:
    """Formats messages into Mamba-compatible prompt strings."""

    def __init__(self, system_prompt=None):
        self.system_prompt = system_prompt

    def format_single(self, message: str) -> str:
        """Format a single-turn conversation."""
        prompt = MAMBA_CHAT_TEMPLATE.format(message=message.strip())
        if self.system_prompt:
            prompt = MAMBA_SYSTEM_PREFIX.format(system=self.system_prompt.strip()) + prompt
        return prompt

    def format_multi(self, messages: list) -> str:
        """
        Format a multi-turn conversation history.
        messages: list of dicts with 'role' and 'content'
        """
        parts = []
        if self.system_prompt:
            parts.append(MAMBA_SYSTEM_PREFIX.format(system=self.system_prompt.strip()))
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "").strip()
            if role == "user":
                parts.append(f"<|user|>\n{content}\n")
            elif role == "assistant":
                parts.append(f"<|assistant|>\n{content}\n")
        if not parts or parts[-1].startswith("<|assistant|>"):
            parts.append("<|assistant|>\n")
        return "".join(parts)
