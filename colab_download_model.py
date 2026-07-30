# Download model.safetensors directly from ZO (bypass Drive truncation issue)
ZO_IP = "185.209.179.179"
ZO_PORT = 8765

import os
from pathlib import Path
import urllib.request

url = f"http://{ZO_IP}:{ZO_PORT}/NeuralAI-Air-135M-HF/model.safetensors"
out_path = "/content/model/model.safetensors"

Path("/content/model").mkdir(parents=True, exist_ok=True)

print(f"Downloading from {url}")
print("This will take 2-5 minutes for 511MB...")

urllib.request.urlretrieve(url, out_path)

size_mb = os.path.getsize(out_path) / 1e6
print(f"Downloaded: {size_mb:.1f} MB")

if size_mb < 500:
    raise ValueError(f"File truncated! Expected ~511 MB, got {size_mb:.1f} MB")
else:
    print("File complete. Proceeding to model load.")
