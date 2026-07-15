#!/usr/bin/env python3
import base64
import json
import os
import sys
from pathlib import Path
from huggingface_hub import get_session

REPO = "Subject-Emu-5259/neuralai-demo"
BRANCH = "main"
SRC = Path("/home/workspace/Projects/NeuralAI/hf-space")

FILES = ["README.md", "index.html", "styles.css", "chat.html", "chat.js"]


def main():
    sess = get_session()
    token = os.environ.get("HF_Write")
    if not token:
        print("HF_Write not in env", file=sys.stderr)
        sys.exit(1)
    headers = {"authorization": f"Bearer {token}"}

    pre = []
    for name in FILES:
        p = SRC / name
        b = p.read_bytes()
        sample = base64.b64encode(b[: min(1024, len(b))]).decode()
        pre.append({"path": name, "sample": sample, "size": len(b)})
    r = sess.post(
        f"https://huggingface.co/api/spaces/{REPO}/preupload/{BRANCH}",
        json={"files": pre},
        headers=headers,
    )
    print("preupload", r.status_code, r.text[:200])
    if r.status_code not in (200, 204):
        sys.exit(1)

    lines = [
        json.dumps({
            "key": "header",
            "value": {"summary": "Update NeuralAI v2 Space", "description": ""},
        })
    ]
    for name in FILES:
        p = SRC / name
        b64 = base64.b64encode(p.read_bytes()).decode()
        lines.append(json.dumps({"key": "file", "value": {"content": b64, "path": name}}))

    body = "\n".join(lines).encode()
    r = sess.post(
        f"https://huggingface.co/api/spaces/{REPO}/commit/{BRANCH}",
        data=body,
        headers={**headers, "content-type": "application/x-ndjson"},
    )
    print("commit", r.status_code, r.text[:200])
    if r.status_code >= 300:
        sys.exit(1)
    print("OK", r.json().get("commitUrl", "(no url)"))


if __name__ == "__main__":
    main()
