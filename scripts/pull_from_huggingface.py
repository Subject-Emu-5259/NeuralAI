#!/usr/bin/env python3
"""
NeuralAI ← Hugging Face Pull Script
Downloads model checkpoints, training data, and configs from the Hub.
"""

import os
import sys
from huggingface_hub import HfApi, snapshot_download

REPO_ID = "Subject-Emu-5259/NeuralAI"
LOCAL_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def main():
    api = HfApi()
    
    user = api.whoami(token=True)
    print(f"🔗 Connected as: {user['name']}")
    print(f"📦 Repo: {REPO_ID}")
    
    action = sys.argv[1] if len(sys.argv) > 1 else "all"
    
    if action == "model" or action == "all":
        print("\n📥 Pulling model adapter...")
        model_files = [
            "adapter_config.json",
            "adapter_model.safetensors",
            "chat_template.jinja",
            "tokenizer.json",
            "tokenizer_config.json",
            "training_log.json",
        ]
        for f in model_files:
            try:
                api.hf_hub_download(
                    repo_id=REPO_ID,
                    filename=f,
                    local_dir=os.path.join(LOCAL_DIR, "checkpoints", "v2_model"),
                    local_dir_use_symlinks=False,
                    token=True
                )
                print(f"  ✅ {f}")
            except Exception as e:
                print(f"  ⏭️  {f} (not found on Hub)")
    
    if action == "data" or action == "all":
        print("\n📥 Pulling training data...")
        try:
            files = api.list_repo_files(REPO_ID, repo_type="model", token=True)
            data_files = [f for f in files if f.startswith("data/")]
            for f in data_files:
                local_path = os.path.join(LOCAL_DIR, f)
                os.makedirs(os.path.dirname(local_path), exist_ok=True)
                api.hf_hub_download(
                    repo_id=REPO_ID,
                    filename=f,
                    local_dir=LOCAL_DIR,
                    local_dir_use_symlinks=False,
                    token=True
                )
                print(f"  ✅ {f}")
        except Exception as e:
            print(f"  ❌ Error: {e}")
    
    if action == "scripts" or action == "all":
        print("\n📥 Pulling training scripts...")
        try:
            files = api.list_repo_files(REPO_ID, repo_type="model", token=True)
            for prefix in ["training/", "services/", "tools/"]:
                for f in [x for x in files if x.startswith(prefix)]:
                    local_path = os.path.join(LOCAL_DIR, f)
                    os.makedirs(os.path.dirname(local_path), exist_ok=True)
                    api.hf_hub_download(
                        repo_id=REPO_ID,
                        filename=f,
                        local_dir=LOCAL_DIR,
                        local_dir_use_symlinks=False,
                        token=True
                    )
                    print(f"  ✅ {f}")
        except Exception as e:
            print(f"  ❌ Error: {e}")
    
    print(f"\n{'='*50}")
    print(f"✅ Pull complete! Files synced to {LOCAL_DIR}")
    print(f"{'='*50}")

if __name__ == "__main__":
    main()
