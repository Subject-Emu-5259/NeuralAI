import os
import torch
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline
from PIL import Image
import time
import requests
from io import BytesIO

class NeuralAIDiffusion:
    def __init__(self, model_id="runwayml/stable-diffusion-v1-5", device=None):
        self.model_id = model_id
        if device:
            self.device = device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.pipe = None
        self.img2img_pipe = None
        self.is_loaded = False
        print(f"[NeuralAI Diffusion] Initialized on {self.device}")

    def load_model(self, mode="text2img"):
        if self.is_loaded and (self.pipe if mode == "text2img" else self.img2img_pipe):
            return
        
        print(f"[NeuralAI Diffusion] Loading {mode} model {self.model_id}...")
        try:
            # Using float32 for CPU to avoid errors, float16 for CUDA
            dtype = torch.float16 if self.device == "cuda" else torch.float32
            
            if mode == "text2img":
                self.pipe = StableDiffusionPipeline.from_pretrained(
                    self.model_id, 
                    torch_dtype=dtype,
                    safety_checker=None # Disable safety checker for faster loading if needed, or keep for safety
                )
                self.pipe.to(self.device)
                # Optimization for CPU
                if self.device == "cpu":
                    self.pipe.enable_attention_slicing()
            else:
                self.img2img_pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
                    self.model_id,
                    torch_dtype=dtype,
                    safety_checker=None
                )
                self.img2img_pipe.to(self.device)
                if self.device == "cpu":
                    self.img2img_pipe.enable_attention_slicing()
                
            self.is_loaded = True
            print(f"[NeuralAI Diffusion] {mode} model loaded successfully.")
        except Exception as e:
            print(f"[NeuralAI Diffusion] Error loading model: {e}")
            if self.model_id != "segmind/tiny-sd":
                print("[NeuralAI Diffusion] Attempting fallback to tiny-sd...")
                self.model_id = "segmind/tiny-sd"
                self.load_model(mode)

    def generate(self, prompt, output_path, negative_prompt=None, num_steps=20, guidance_scale=7.5):
        self.load_model("text2img")
        
        # Enhanced Prompting for "Better Images"
        quality_boost = "masterpiece, high quality, 8k, highly detailed, professional photography"
        if "moon" in prompt.lower():
            quality_boost += ", sharp craters, lunar surface detail, space background, realistic"
            
        full_prompt = f"{prompt}, {quality_boost}"
        
        if negative_prompt is None:
            negative_prompt = "blurry, low quality, distorted, watermark, text, grainy, low resolution"
            
        print(f"[NeuralAI Diffusion] Generating: {full_prompt}")
        start_time = time.time()
        try:
            image = self.pipe(
                prompt=full_prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_steps,
                guidance_scale=guidance_scale
            ).images[0]
            image.save(output_path)
            print(f"[NeuralAI Diffusion] Image saved to {output_path} (took {time.time() - start_time:.2f}s)")
            return True
        except Exception as e:
            print(f"[NeuralAI Diffusion] Generation failed: {e}")
            return False

    def transform(self, prompt, image_path, output_path, strength=0.75, num_steps=20):
        self.load_model("img2img")
        
        quality_boost = "masterpiece, high quality, 8k, highly detailed"
        full_prompt = f"{prompt}, {quality_boost}"
        
        print(f"[NeuralAI Diffusion] Transforming image with prompt: {full_prompt}")
        start_time = time.time()
        try:
            if image_path.startswith("http"):
                response = requests.get(image_path)
                init_image = Image.open(BytesIO(response.content)).convert("RGB")
            else:
                init_image = Image.open(image_path).convert("RGB")
            
            init_image = init_image.resize((512, 512))
            
            image = self.img2img_pipe(
                prompt=full_prompt,
                image=init_image,
                strength=strength,
                num_inference_steps=num_steps
            ).images[0]
            
            image.save(output_path)
            print(f"[NeuralAI Diffusion] Transformed image saved to {output_path} (took {time.time() - start_time:.2f}s)")
            return True
        except Exception as e:
            print(f"[NeuralAI Diffusion] Transformation failed: {e}")
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
