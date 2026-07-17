"""AI image generation for NeuralAI (keyed Pollinations AI endpoint).

Primary: keyed Pollinations AI images endpoint (OpenAI-compatible) at
https://image.pollinations.ai/v1/generations (model "flux"). Verified 200 with
key. Returns RAW JPEG bytes (not b64 JSON) which we write to disk.

Fallback: OpenRouter Gemini image model only if OPENROUTER_API_KEY present.

Generated images live in the user's NeuralDrive (NEURAL_DRIVE/generated).
"""
import logging
from typing import Dict, Any

import os
import uuid
import json
import urllib.request
from datetime import datetime
from pathlib import Path
from urllib.parse import quote

logger = logging.getLogger("neuralai.image")

NEURAL_DRIVE = Path(os.environ.get("NEURAL_DRIVE", "/home/workspace/NeuralDrive"))
GENERATED_DIR = NEURAL_DRIVE / "generated"
GENERATED_DIR.mkdir(parents=True, exist_ok=True)


class ImageGenerator:
    """Generate images via Pollinations keyed endpoint (primary), OpenRouter (fallback)."""

    POLLINATIONS_URL = "https://image.pollinations.ai/v1/generations"
    MODEL = "flux"
    OPENROUTER_URL = "https://openrouter.ai/api/v1/images/generations"

    @property
    def _api_key(self) -> str:
        return (
            os.environ.get("Pollinations_Api_key")
            or os.environ.get("POLLINATIONS_API_KEY")
            or os.environ.get("POLLINATIONS_KEY")
            or ""
        )

    @property
    def _openrouter_key(self) -> str:
        # Live env var is Open_Router_API (per Zo Advanced secrets + AGENTS.md).
        return (
            os.environ.get("Open_Router_API")
            or os.environ.get("OPENROUTER_API_KEY")
            or ""
        )

    def _dims(self, aspect_ratio: str):
        ratios = {
            "1:1": (1024, 1024),
            "16:9": (1280, 720),
            "9:16": (720, 1280),
            "4:3": (1152, 864),
            "3:4": (864, 1152),
            "3:2": (1200, 800),
            "2:3": (800, 1200),
        }
        return ratios.get(aspect_ratio, (1024, 1024))

    def generate(
        self,
        prompt: str,
        aspect_ratio: str = "1:1",
        style: str = "",
        provider: str = "",
    ) -> Dict[str, Any]:
        full_prompt = prompt
        if style:
            full_prompt = f"{prompt}, {style} style"
        if style == "realistic" and "photorealistic" not in full_prompt.lower():
            full_prompt += ", photorealistic, high detail"

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_stem = f"neuralai_{timestamp}_{uuid.uuid4().hex[:6]}"
        output_path = GENERATED_DIR / f"{file_stem}_1.png"

        def _write(raw: bytes) -> bool:
            if raw[:1] in (b"\x89", b"\xff", b"G"):  # PNG/JPEG/GIF magic
                output_path.write_bytes(raw)
                return True
            return False

        ok = False
        # Public Pollinations endpoint (no key needed) — primary path.
        try:
            width, height = self._dims(aspect_ratio)
            params = f"https://image.pollinations.ai/prompt/{quote(full_prompt)}?width={width}&height={height}&model=flux&nologo=true&seed={uuid.uuid4().int % 10**9}"
            req = urllib.request.Request(params, headers={"User-Agent": "NeuralAI/1.0"})
            with urllib.request.urlopen(req, timeout=120) as resp:
                raw = resp.read()
            ok = _write(raw)
        except Exception as pe:  # noqa: BLE001
            logger.warning("Pollinations public image failed: %s", pe)

        # Optional keyed override only if the public path failed AND a key exists.
        if not ok:
            api_key = self._api_key
            if api_key:
                try:
                    width, height = self._dims(aspect_ratio)
                    payload = json.dumps({
                        "model": self.MODEL,
                        "prompt": full_prompt,
                        "n": 1,
                        "size": f"{width}x{height}",
                    }).encode("utf-8")
                    req = urllib.request.Request(
                        self.POLLINATIONS_URL,
                        data=payload,
                        headers={
                            "Authorization": f"Bearer {api_key}",
                            "Content-Type": "application/json",
                        },
                        method="POST",
                    )
                    with urllib.request.urlopen(req, timeout=120) as resp:
                        raw = resp.read()
                    ok = _write(raw)
                except Exception as pe2:  # noqa: BLE001
                    logger.warning("Pollinations keyed image failed: %s", pe2)

        # OpenRouter Gemini image fallback (only if a key is present).
        if not ok:
            or_key = self._openrouter_key
            if or_key:
                try:
                    width, height = self._dims(aspect_ratio)
                    payload = json.dumps({
                        "model": "google/gemini-2.5-flash-image",
                        "prompt": full_prompt,
                        "n": 1,
                        "size": f"{width}x{height}",
                    }).encode("utf-8")
                    req = urllib.request.Request(
                        self.OPENROUTER_URL,
                        data=payload,
                        headers={
                            "Authorization": f"Bearer {or_key}",
                            "Content-Type": "application/json",
                        },
                        method="POST",
                    )
                    with urllib.request.urlopen(req, timeout=120) as resp:
                        j = json.loads(resp.read().decode("utf-8", "ignore"))
                    img_b64 = None
                    url = None
                    msg = (j.get("choices", [{}])[0].get("message", {}) or {})
                    content = msg.get("content")
                    if isinstance(content, list):
                        for part in content:
                            if isinstance(part, dict):
                                if part.get("type") == "image_url":
                                    url = part.get("image_url", {}).get("url", "")
                                elif part.get("image") and isinstance(part["image"], dict):
                                    img_b64 = part["image"].get("data")
                    elif isinstance(content, str) and content.startswith("data:"):
                        img_b64 = content
                    if url and url.startswith("http"):
                        with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "NeuralAI/1.0"}), timeout=120) as r:
                            ok = _write(r.read())
                    elif img_b64 and img_b64.startswith("data:"):
                        import base64
                        _, b = img_b64.split(",", 1)
                        output_path.write_bytes(base64.b64decode(b))
                        ok = True
                except Exception as oe:  # noqa: BLE001
                    logger.warning("OpenRouter Gemini image failed: %s", oe)

        if not ok:
            return {
                "success": False,
                "image_path": "",
                "image_url": "",
                "prompt": prompt,
                "error": "Image generation failed: Pollinations public + keyed + OpenRouter Gemini all returned no valid image.",
            }

        return {
            "success": True,
            "prompt": prompt,
            "full_prompt": full_prompt,
            "file_stem": file_stem,
            "output_dir": str(GENERATED_DIR),
            "aspect_ratio": aspect_ratio,
            "provider": "pollinations-public",
            "image_path": str(output_path),
            "image_url": f"/neuraldrive/generated/{output_path.name}",
            "error": "",
        }

    def _dims_unused(self):
        return (1024, 1024)

    def list_images(self) -> list:
        if not GENERATED_DIR.exists():
            return []
        images = []
        for f in GENERATED_DIR.glob("*.png"):
            images.append({
                "name": f.name,
                "path": str(f),
                "url": f"/neuraldrive/generated/{f.name}",
                "created": datetime.fromtimestamp(f.stat().st_mtime).isoformat(),
            })
        return sorted(images, key=lambda x: x["created"], reverse=True)

    def delete_image(self, filename: str) -> bool:
        filepath = GENERATED_DIR / filename
        if filepath.exists() and filepath.is_relative_to(GENERATED_DIR):
            filepath.unlink()
            return True
        return False


image_generator = ImageGenerator()
