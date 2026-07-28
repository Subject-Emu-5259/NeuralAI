import torch
from diffusers import StableDiffusionPipeline
import os
import sys

def generate_image(prompt, output_path):
    model_id = "runwayml/stable-diffusion-v1-5"
    
    print(f"Loading model {model_id}...")
    # Using float16 and CPU offload if possible, or just CPU if no CUDA
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id, 
        torch_dtype=torch.float32 if device == "cpu" else torch.float16
    )
    pipe = pipe.to(device)
    
    print(f"Generating image for prompt: {prompt}")
    image = pipe(prompt, num_inference_steps=20).images[0]
    
    image.save(output_path)
    print(f"Image saved to {output_path}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python neuralai_diffusion.py <prompt> <output_path>")
        sys.exit(1)
    
    prompt = sys.argv[1]
    output_path = sys.argv[2]
    generate_image(prompt, output_path)
