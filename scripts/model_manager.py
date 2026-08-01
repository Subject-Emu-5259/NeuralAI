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

MODELS = {
    "mamba-k1": {
        "id": "mamba-k1",
        "label": "Mamba K1 (130M, SFT)",
        "path": "/home/workspace/Projects/NeuralAI/models/mamba-k1",
        "params": "129M",
        "type": "mamba",
        "architecture": "Mamba SSM (state-spaces/mamba-130m-hf)",
        "training": "SFT on UltraChat (50 steps, 1K samples)",
        "runtime": "transformers (PyTorch)",
        "ownership": "NeuralAI - First Owned Base Model",
    },
    "mamba-k2": {
        "id": "mamba-k2",
        "label": "Mamba K2 (790M, SFT 10K+)",
        "path": "/home/workspace/Projects/NeuralAI/models/mamba-k2",
        "gguf": "models/NeuralAI-Mamba-K2.Q4_K_M.gguf",
        "params": "793M",
        "type": "mamba",
        "architecture": "Mamba SSM (state-spaces/mamba-790m-hf)",
        "training": "SFT on UltraChat (500-1000 steps, 10K+ samples)",
        "runtime": "llama.cpp GGUF / transformers (PyTorch)",
        "ownership": "NeuralAI - Second Owned Base Model",
    },

    "neuralai-v17-dpo": {
        "id": "neuralai-v17-dpo",
        "label": "NeuralAI v17 DPO",
        "path": "/home/workspace/Projects/NeuralAI/models/NeuralAI-v17-dpo.Q4_K_M.gguf",
        "params": "360M",
        "type": "dpo",
    },
}
DEFAULT_MODEL = "neuralai-v17-dpo"


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
