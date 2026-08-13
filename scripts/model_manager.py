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
        "codename": "MambaK1Project",
        "display_version": "future",
        "label": "Mamba K1 (130M, training)",
        "path": "/home/workspace/Projects/NeuralAI/models/k1/current/gguf/NeuralAI-Mamba-K1-v3.Q4_K_M.gguf",
        "hf_repo": "Subject-Emu-5259/NeuralAI-Mamba-K1",
        "params": "130M",
        "type": "mamba",
        "architecture": "Mamba SSM (state-spaces/mamba-130m-hf)",
        "training": "NeuralAI-owned base model; currently being retrained with a fixed instruction format",
        "runtime": "llama.cpp GGUF (Q4_K_M, ~86MB)",
        "ownership": "NeuralAI - First Owned Base Model",
        "status": "training",
        "chat_format": "neuralai-intel",
        "inference_note": "Mamba K1 is not available for chat while training completes. SmolLM2 360M is the active default.",
        "interaction_mode": "chat",
        "selectable": False,
        "blocked_reason": "this model is training",
    },
    "smollm2-360m": {
        "id": "smollm2-360m",
        "display_version": "base",
        "label": "SmolLM2 360M Instruct (third-party)",
        "path": "/root/models/smollm2-360m-instruct-q4_k_m.gguf",
        "hf_repo": "HuggingFaceTB/SmolLM2-360M-Instruct",
        "params": "360M",
        "type": "third_party",
        "architecture": "Transformer decoder (Hugging Face TB)",
        "training": "Pre-trained and instruction-tuned by Hugging Face",
        "runtime": "llama.cpp GGUF (Q4_K_M, ~259MB)",
        "ownership": "Third-party — Hugging Face TB",
        "status": "ready",
        "chat_format": "chatml",
        "inference_note": "Default local chat model while Mamba K1 is in training.",
        "interaction_mode": "chat",
        "selectable": True,
    },
    "smol-awareness-merged": {
        "id": "smol-awareness-merged",
        "label": "NeuralAI SmolLM2 360M Awareness v1.0",
        "display_version": "v1.0",
        "codename": "NeuralAIPoweredBySmoIL360M",
        "path": "/home/workspace/Projects/NeuralAI/models/NeuralAI-Smol-Awareness-Q8_0.gguf",
        "hf_repo": "HuggingFaceTB/SmolLM2-360M-Instruct",
        "params": "360M",
        "type": "awareness_tuned",
        "architecture": "Transformer decoder (Hugging Face TB) + NeuralAI LoRA",
        "training": "Locally LoRA-SFT'd on NeuralAI brand/site/tool/model-family/chat/companion/identity data",
        "runtime": "llama.cpp GGUF (Q8_0, ~385MB)",
        "ownership": "Third-party base + NeuralAI LoRA",
        "status": "ready",
        "chat_format": "chatml",
        "inference_note": "Baseline awareness-tuned SmolLM2; still selectable.",
        "interaction_mode": "chat",
        "selectable": True,
    },
    "smol-awareness-v2-merged": {
        "id": "smol-awareness-v2-merged",
        "label": "NeuralAI SmolLM2 360M Awareness v2.0",
        "display_version": "v2.0",
        "codename": "NeuralAIPoweredBySmoIL360M",
        "path": "/home/workspace/Projects/NeuralAI/models/NeuralAI-Smol-Awareness-v2-Q8_0.gguf",
        "hf_repo": "HuggingFaceTB/SmolLM2-360M-Instruct",
        "params": "360M",
        "type": "awareness_tuned",
        "architecture": "Transformer decoder (Hugging Face TB) + NeuralAI LoRA",
        "training": "Locally LoRA-SFT'd v2 on expanded NeuralAI awareness data with paraphrases and refusal examples",
        "runtime": "llama.cpp GGUF (Q8_0, ~385MB)",
        "ownership": "Third-party base + NeuralAI LoRA",
        "status": "ready",
        "chat_format": "chatml",
        "inference_note": "Active awareness-tuned SmolLM2 used as the live chat backend.",
        "interaction_mode": "chat",
        "selectable": True,
    },
}
DEFAULT_MODEL = "smollm2-360m"


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
