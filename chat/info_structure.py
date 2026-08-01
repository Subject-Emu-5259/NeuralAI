"""
Information Structure module for NeuralAI chat responses.
Generates structured info cards for model responses - Mamba architecture
details, training provenance, and response metadata.
"""

import json
from datetime import datetime

class InfoCard:
    """Structured response metadata card."""

    def __init__(self, model_id: str, model_label: str, architecture: str = "",
                 params: str = "", training: str = "", ownership: str = ""):
        self.model_id = model_id
        self.model_label = model_label
        self.architecture = architecture
        self.params = params
        self.training = training
        self.ownership = ownership
        self.timestamp = datetime.now().isoformat()

    def to_dict(self):
        return {
            "model_id": self.model_id,
            "model_label": self.model_label,
            "architecture": self.architecture,
            "params": self.params,
            "training": self.training,
            "ownership": self.ownership,
            "timestamp": self.timestamp,
        }

    def to_html(self):
        """Render the info card as HTML for the web UI."""
        badge = "🧠 Mamba SSM" if "mamba" in self.architecture.lower() else "🤖 Neural Model"
        return (
            f'<div class="mamba-info-card" data-model="{self.model_id}">'
            f'<div class="mamba-badge">{badge} · {self.params}</div>'
            f'<div class="mamba-detail">{self.model_label}</div>'
            f'<div class="mamba-meta">Architecture: {self.architecture}</div>'
            f'<div class="mamba-meta">Training: {self.training}</div>'
            f'<div class="mamba-owner">{self.ownership}</div>'
            f'</div>'
        )

def format_model_card(model_info: dict) -> str:
    """Build an HTML info card from a model dict."""
    card = InfoCard(
        model_id=model_info.get("id", ""),
        model_label=model_info.get("label", ""),
        architecture=model_info.get("architecture", ""),
        params=model_info.get("params", ""),
        training=model_info.get("training", ""),
        ownership=model_info.get("ownership", ""),
    )
    return card.to_html()

def build_mamba_info_html(model_info: dict) -> str:
    """Build an info card specifically for Mamba models."""
    if model_info.get("type", "").lower() != "mamba":
        return ""
    return format_model_card(model_info)
