#!/usr/bin/env python3
"""
NeuralAI → Hugging Face Sync Script
Uploads all local training data, scripts, services, and configs to the Hub.
"""

import os
import sys
from huggingface_hub import HfApi

REPO_ID = "Subject-Emu-5259/NeuralAI"

def upload_folder(api, folder, target, description, extension=None):
    print(f"\n📤 Uploading {description}...")
    for f in sorted(os.listdir(folder)):
        local_path = os.path.join(folder, f)
        if os.path.isfile(local_path) and (extension is None or f.endswith(extension)):
            print(f"  {f}...", end=" ", flush=True)
            try:
                api.upload_file(
                    path_or_fileobj=local_path,
                    path_in_repo=f"{target}/{f}",
                    repo_id=REPO_ID,
                    repo_type="model",
                    create_pr=False,
                    token=True
                )
                print("✅")
            except Exception as e:
                print(f"❌ {e}")

def upload_file(api, local_path, repo_path, description):
    print(f"  Uploading {description}...", end=" ", flush=True)
    try:
        api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=repo_path,
            repo_id=REPO_ID,
            repo_type="model",
            create_pr=False,
            token=True
        )
        print("✅")
    except Exception as e:
        print(f"❌ {e}")

def main():
    api = HfApi()
    
    # Verify connection
    user = api.whoami(token=True)
    print(f"🔗 Connected as: {user['name']}")
    print(f"📦 Repo: {REPO_ID}")
    
    # Training data
    upload_folder(api, "data", "data", "training data", extension=(".jsonl", ".json"))
    
    # Training scripts
    upload_folder(api, "training", "training", "training scripts", extension=".py")
    
    # Services
    upload_folder(api, "services", "services", "services", extension=(".py", ".sh"))
    
    # Tools
    upload_folder(api, "tools", "tools", "tools", extension=".py")
    
    # Config & docs
    root_files = ["AGENTS.md", "MODEL_ALIGNMENT.md", "ORCHESTRATOR.md", "GEMINI.md", "README.md"]
    for f in root_files:
        local = os.path.join(".", f)
        if os.path.exists(local):
            upload_file(api, local, f"configs/{f}", f)

    # Training config
    if os.path.exists("checkpoints/training_config.json"):
        upload_file(api, "checkpoints/training_config.json", "checkpoints/training_config.json", "training config")
    
    # Neural-Brain knowledge base
    if os.path.exists("neural-brain"):
        print("\n📤 Uploading neural-brain knowledge base...")
        for root, dirs, files in os.walk("neural-brain"):
            for f in files:
                local = os.path.join(root, f)
                repo_path = os.path.join("neural-brain", os.path.relpath(local, "neural-brain"))
                try:
                    api.upload_file(
                        path_or_fileobj=local,
                        path_in_repo=repo_path,
                        repo_id=REPO_ID,
                        repo_type="model",
                        create_pr=False,
                        token=True
                    )
                except Exception as e:
                    print(f"  ❌ {repo_path}: {e}")
        print("  ✅ neural-brain uploaded")
    
    print(f"\n{'='*50}")
    print(f"✅ Sync complete! View at:")
    print(f"   https://huggingface.co/{REPO_ID}")
    print(f"{'='*50}")

if __name__ == "__main__":
    main()
