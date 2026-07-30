# Download model.safetensors directly from ZO via Cloudflare tunnel
# PASTE YOUR URL from the ZO terminal below:
ZO_URL = "https://YOUR_URL_HERE.trycloudflare.com"  # <-- PASTE HERE

import os
from pathlib import Path

# Ensure model dir exists
Path("/content/model").mkdir(parents=True, exist_ok=True)

# Download
url = f"{ZO_URL}/NeuralAI-Air-135M-HF/model.safetensors"
out_path = "/content/model/model.safetensors"

!curl -L -o {out_path} {url}

# Verify
size_mb = os.path.getsize(out_path) / 1e6
print(f"Downloaded: {size_mb:.1f} MB")

if size_mb < 500:
    raise ValueError(f"File truncated! Expected ~511 MB, got {size_mb:.1f} MB")
else:
    print("File complete. Ready to load model.")
