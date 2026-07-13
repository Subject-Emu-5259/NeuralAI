import os
import torch
from PIL import Image
import time

try:
    from diffusers import AutoPipelineForText2Image, AutoPipelineForImage2Image
    _HAS_DIFFUSERS = True
except Exception:
    _HAS_DIFFUSERS = False


class NeuralAIDiffusion:
    """Local image generation sidecar for NeuralAI.

    Default backend: ``segmind/sdxl-turbo`` -- a 1-4 step distilled SDXL model
    that produces real images on CPU in a few seconds. Optionally loads a
    NeuralAI brand LoRA (``NEURALAI_LORA_PATH``) for the signature dark/neon
    "vibe stack" aesthetic.

    Falls back to SD 1.5 / tiny-sd if the turbo checkpoint is unavailable.
    """

    def __init__(self, model_id=None, device=None, lora_path=None):
        self.model_id = model_id or os.environ.get(
            "NEURALAI_DIFFUSION_MODEL", "segmind/sdxl-turbo"
        )
        self.lora_path = lora_path or os.environ.get("NEURALAI_LORA_PATH", "")
        if device:
            self.device = device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.pipe = None
        self.is_loaded = False
        self.using_turbo = False
        print(f"[NeuralAI Diffusion] Initialized on {self.device} (model={self.model_id})")

    def load_model(self):
        if self.is_loaded:
            return
        if not _HAS_DIFFUSERS:
            raise RuntimeError("diffusers not installed; cannot run local diffusion")

        print(f"[NeuralAI Diffusion] Loading {self.model_id}...")
        try:
            dtype = torch.float16 if self.device == "cuda" else torch.float32
            self.pipe = AutoPipelineForText2Image.from_pretrained(
                self.model_id, torch_dtype=dtype, safety_checker=None, use_safetensors=True
            )
            self.pipe.to(self.device)
            if self.device == "cpu":
                self.pipe.enable_attention_slicing()
                try:
                    self.pipe.enable_model_cpu_offload()
                except Exception:
                    pass

            # SDXL-Turbo / LCM are single-step distilled models.
            self.using_turbo = (
                "turbo" in self.model_id.lower() or "lcm" in self.model_id.lower()
            )

            # Load NeuralAI brand LoRA if provided (SDXL base only).
            if self.lora_path and os.path.exists(self.lora_path):
                try:
                    self.pipe.load_lora_weights(self.lora_path)
                    print(f"[NeuralAI Diffusion] Loaded brand LoRA from {self.lora_path}")
                except Exception as e:
                    print(f"[NeuralAI Diffusion] LoRA load failed (ignored): {e}")

            self.is_loaded = True
            print("[NeuralAI Diffusion] Model loaded successfully.")
        except Exception as e:
            print(f"[NeuralAI Diffusion] Error loading {self.model_id}: {e}")
            if self.model_id != "segmind/tiny-sd":
                print("[NeuralAI Diffusion] Falling back to tiny-sd...")
                self.model_id = "segmind/tiny-sd"
                self.using_turbo = False
                self.load_model()

    def generate(self, prompt, output_path, negative_prompt=None, num_steps=20, guidance_scale=7.5):
        self.load_model()

        # SDXL-Turbo / LCM: 1-4 steps, guidance ~0.0. Standard SD: use requested steps.
        if self.using_turbo:
            num_steps = max(1, min(num_steps, 4))
            guidance_scale = 0.0
        else:
            num_steps = max(10, min(num_steps, 50))

        # NeuralAI brand styling appended to every prompt for a consistent look.
        brand = (
            "cinematic, dark mode, neon accent lighting, high contrast, "
            "hyper-detailed, 8k, vibe stack aesthetic"
        )
        full_prompt = f"{prompt}, {brand}"

        print(f"[NeuralAI Diffusion] Generating: {full_prompt}")
        start_time = time.time()
        try:
            gen_kwargs = dict(
                prompt=full_prompt,
                num_inference_steps=num_steps,
                guidance_scale=guidance_scale,
            )
            if negative_prompt and not self.using_turbo:
                gen_kwargs["negative_prompt"] = negative_prompt

            image = self.pipe(**gen_kwargs).images[0]
            image.save(output_path)
            print(
                f"[NeuralAI Diffusion] Image saved to {output_path} "
                f"(took {time.time() - start_time:.2f}s)"
            )
            return True
        except Exception as e:
            print(f"[NeuralAI Diffusion] Generation failed: {e}")
            return False

    def transform(self, prompt, image_path, output_path, strength=0.75, num_steps=20):
        """Img2img using the same checkpoint (turbo supports it too)."""
        if not _HAS_DIFFUSERS:
            return False
        if not self.is_loaded:
            self.load_model()
        try:
            from diffusers import AutoPipelineForImage2Image
            dtype = torch.float16 if self.device == "cuda" else torch.float32
            i2i = AutoPipelineForImage2Image.from_pretrained(
                self.model_id, torch_dtype=dtype, safety_checker=None, use_safetensors=True
            ).to(self.device)
            init = Image.open(image_path).convert("RGB")
            steps = 1 if self.using_turbo else max(10, min(num_steps, 50))
            out = i2i(
                prompt=f"{prompt}, cinematic, neon, high detail",
                image=init,
                strength=strength,
                num_inference_steps=steps,
                guidance_scale=0.0 if self.using_turbo else 7.5,
            ).images[0]
            out.save(output_path)
            return True
        except Exception as e:
            print(f"[NeuralAI Diffusion] Transform failed: {e}")
            return False


if __name__ == "__main__":
    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else "gen"
    prompt = sys.argv[2] if len(sys.argv) > 2 else "A high-tech AI logo"
    output = sys.argv[3] if len(sys.argv) > 3 else "output.png"

    engine = NeuralAIDiffusion()
    if mode == "edit" and len(sys.argv) > 4:
        engine.transform(prompt, sys.argv[4], output)
    else:
        engine.generate(prompt, output)
