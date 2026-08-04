#!/usr/bin/env python3
"""Manage the active GGUF model for the NeuralAI inference server."""
import argparse
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_DIR = os.path.join(ROOT, "config")
CONFIG_PATH = os.path.join(CONFIG_DIR, "active_model.json")
SUPERVISORD_CONF = "/etc/zo/supervisord-user.conf"

# Mamba family chat-format notes:
# All three models share a GPTNeoXTokenizer whose only special token is
# <|endoftext|>.  Legacy llama-2/chatml templates use tokens such as </s> or
# <|im_start|> which are NOT in the tokenizer vocabulary, which is why the
# models were producing token-soup.  The "intel" chat handler uses plain-text
# markers (### System:, ### User:, ### Assistant:) and trains/prompts with
# only existing tokens, so it is the consistent format for the whole family.
MODELS = {
    "mamba-k1": {
        "id": "mamba-k1",
        "label": "Mamba K1 (130M, SFT v3)",
        "path": "/home/workspace/Projects/NeuralAI/models/k1/current/gguf/NeuralAI-Mamba-K1-v3.Q4_K_M.gguf",
        "hf_repo": "Subject-Emu-5259/NeuralAI-Mamba-K1",
        "params": "130M",
        "type": "mamba",
        "architecture": "Mamba SSM (state-spaces/mamba-130m-hf)",
        "training": "SFT LoRA v3 on single-turn UltraChat (intel prompt format); latest weights synced from HuggingFace",
        "runtime": "llama.cpp GGUF (Q4_K_M, ~86MB)",
        "ownership": "NeuralAI - First Owned Base Model",
        "status": "ready",
        "chat_format": "neuralai-intel",
        "inference_note": "Latest SFT v3 weights are live. The merged safetensors is also available at models/k1/current/model.safetensors.",
        "interaction_mode": "chat",
    },
    "mamba-k2": {
        "id": "mamba-k2",
        "label": "Mamba K2 (793M, base Q4_K_M)",
        "path": "/home/workspace/Projects/NeuralAI/models/k2/gguf/mamba-790m-hf.Q4_K_M.gguf",
        "hf_repo": "Subject-Emu-5259/NeuralAI-Mamba-K2",
        "params": "793M",
        "type": "mamba",
        "architecture": "Mamba SSM (state-spaces/mamba-790m-hf)",
        "training": "Pretrained Mamba base; SFT LoRA v1 queued on cleaned UltraChat (intel format)",
        "runtime": "llama.cpp GGUF (Q4_K_M, ~460MB)",
        "ownership": "NeuralAI - In Training",
        "status": "ready",
        "chat_format": "neuralai-intel",
        "inference_note": "Base-only 793M Mamba. Chat uses OpenRouter fallback until K2 SFT v1 completes and a chat GGUF is published.",
        "interaction_mode": "chat",
    },
    "mamba-k3": {
        "id": "mamba-k3",
        "label": "Mamba K3 (2.8B, base)",
        "path": "/home/workspace/Projects/NeuralAI/models/k3/base/model.safetensors",
        "hf_repo": "Subject-Emu-5259/NeuralAI-Mamba-K3",
        "params": "2.8B",
        "type": "mamba",
        "architecture": "Mamba SSM (state-spaces/mamba-2.8b-slimpj)",
        "training": "Base pretrained weights downloaded from state-spaces/mamba-2.8b-slimpj; SFT queued",
        "runtime": "transformers PyTorch safetensors (~1.4GB); no GGUF yet",
        "ownership": "NeuralAI - Next-Gen (in training)",
        "status": "base_needs_sft",
        "chat_format": "neuralai-intel",
        "inference_note": "Base-only 2.8B Mamba; no inference weights yet. HF repo currently contains metadata only.",
        "interaction_mode": "chat",
    },
}
DEFAULT_MODEL = "mamba-k1"


def _ensure_config():
    os.makedirs(CONFIG_DIR, exist_ok=True)
    if not os.path.exists(CONFIG_PATH):
        set_active(DEFAULT_MODEL)


def active_model_id():
    _ensure_config()
    try:
        with open(CONFIG_PATH) as f:
            data = json.load(f)
        mid = data.get("id", DEFAULT_MODEL)
    except Exception:
        mid = DEFAULT_MODEL
    return mid if mid in MODELS else DEFAULT_MODEL


def active_model():
    return MODELS[active_model_id()]


def set_active(mid):
    if mid not in MODELS:
        print(f"Unknown model id: {mid}. Available: {', '.join(MODELS)}", file=sys.stderr)
        sys.exit(1)
    os.makedirs(CONFIG_DIR, exist_ok=True)
    with open(CONFIG_PATH, "w") as f:
        json.dump({"id": mid}, f, indent=2)


def sc_restart():
    return subprocess.run(
        ["supervisorctl", "-c", SUPERVISORD_CONF, "restart", "neuralai-lmstudio"],
        capture_output=True,
        text=True,
    )


def main():
    parser = argparse.ArgumentParser(description="NeuralAI inference model manager")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("get", help="print current active model JSON")
    sub.add_parser("get-id", help="print current active model id")
    sub.add_parser("get-path", help="print current active model GGUF path")
    sub.add_parser("get-format", help="print current active model chat format")
    sub.add_parser("list", help="print all available models JSON")
    p_set = sub.add_parser("set", help="set active model and restart inference")
    p_set.add_argument("id", help="model id to activate")
    args = parser.parse_args()

    if args.cmd == "get":
        print(json.dumps(active_model()))
    elif args.cmd == "get-id":
        print(active_model_id())
    elif args.cmd == "get-path":
        print(active_model()["path"])
    elif args.cmd == "get-format":
        print(active_model().get("chat_format", "intel"))
    elif args.cmd == "list":
        print(json.dumps(list(MODELS.values())))
    elif args.cmd == "set":
        old = active_model_id()
        set_active(args.id)
        print(json.dumps({"success": True, "previous": old, "current": active_model_id()}))
        res = sc_restart()
        print(res.stdout.strip())
        if res.returncode != 0:
            print(res.stderr.strip(), file=sys.stderr)
            sys.exit(res.returncode)


if __name__ == "__main__":
    main()
