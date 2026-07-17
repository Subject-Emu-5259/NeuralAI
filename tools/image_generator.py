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

        # Primary: keyed Pollinations AI OpenAI-compatible images endpoint.
        # Returns RAW image bytes (not b64 JSON), so write the body directly.
        pollinations_err = None
        poll_err = ""
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
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=120) as resp:
                raw = resp.read()
            if raw[:1] in (b"\x89", b"\xff", b"G"):  # PNG/JPEG/GIF magic
                output_path.write_bytes(raw)
                return {
                    "success": True,
                    "prompt": prompt,
                    "full_prompt": full_prompt,
                    "file_stem": file_stem,
                    "output_dir": str(GENERATED_DIR),
                    "aspect_ratio": aspect_ratio,
                    "provider": "pollinations",
                    "image_path": str(output_path),
                    "image_url": f"/neuraldrive/generated/{output_path.name}",
                    "error": "",
                }
            try:
                j = json.loads(raw.decode("utf-8", "replace"))
                poll_err = j.get("error") or j.get("message") or raw.decode("utf-8", "replace")[:200]
            except Exception:
                poll_err = raw.decode("utf-8", "replace")[:200]
            raise RuntimeError(f"Pollinations: {poll_err}")
        except Exception as pe:  # noqa: BLE001
            pollinations_err = pe
            logger.warning("Pollinations image failed: %s; trying OpenRouter", pe)

        # Fallback: OpenRouter Gemini image model (needs valid key)
        try:
            api_key = os.environ.get("Open_Router_API") or os.environ.get("OPENROUTER_API_KEY")
            if not api_key:
                return {
                    "success": False,
                    "image_path": "",
                    "image_url": "",
                    "prompt": prompt,
                    "error": f"Image generation failed (Pollinations: {pollinations_err}). OpenRouter key also missing.",
                }
            import base64
            payload = json.dumps({
                "model": self.MODEL,
                "prompt": full_prompt,
                "n": 1,
                "response_format": "b64_json",
            }).encode("utf-8")
            req = urllib.request.Request(
                self.OPENROUTER_URL,
                data=payload,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            item = (data.get("data") or [{}])[0]
            b64 = item.get("b64_json") or ""
            if not b64 and item.get("url"):
                b64 = base64.b64encode(urllib.request.urlopen(item["url"], timeout=60).read()).decode()
            if not b64:
                return {
                    "success": False,
                    "image_path": "",
                    "image_url": "",
                    "prompt": prompt,
                    "error": "Image generation failed: Pollinations keyed endpoint unavailable; OpenRouter returned no image.",
                }
            output_path.write_bytes(base64.b64decode(b64))
            return {
                "success": True,
                "prompt": prompt,
                "full_prompt": full_prompt,
                "file_stem": file_stem,
                "output_dir": str(GENERATED_DIR),
                "aspect_ratio": aspect_ratio,
                "provider": "openrouter-gemini",
                "image_path": str(output_path),
                "image_url": f"/neuraldrive/generated/{output_path.name}",
                "error": "",
            }
        except Exception as e2:
            logger.exception("image_gen failed")
            return {
                "success": False,
                "image_path": "",
                "image_url": "",
                "prompt": prompt,
                "error": f"Image generation failed: {pollinations_err}; OpenRouter: {e2}",
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
