#!/usr/bin/env python3
"""Upload the NeuralAI v2 Space files using upload_folder"""
import os
from huggingface_hub import HfApi, login

token = os.getenv("HF_Write")
if token:
    login(token=token)

api = HfApi()
api.upload_folder(
    folder_path="/home/workspace/Projects/NeuralAI/hf-space",
    repo_id="Subject-Emu-5259/neuralai-demo",
    repo_type="space",
    commit_message="Deploy NeuralAI v2 static demo with valid emoji",
)
print("Upload complete!")