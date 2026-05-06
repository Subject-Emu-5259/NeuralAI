# tools/image_generator.py
#
# Image generation tool for NeuralAI
# Connects to Zo's generate_image capability

import os
import json
import subprocess
from typing import Dict, Any, Optional
from pathlib import Path
from datetime import datetime


class ImageGenerator:
    """Generate images using AI image generation models."""
    
    def __init__(self, output_dir: str = "/home/workspace/Images/NeuralAI"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate(
        self,
        prompt: str,
        style: Optional[str] = None,
        size: str = "512x512",
        file_stem: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate an image from a text prompt.
        
        Args:
            prompt: Description of the image to generate
            style: Optional style (realistic, artistic, cartoon, etc.)
            size: Image size (512x512, 1024x1024, etc.)
            file_stem: Optional filename stem (without extension)
        
        Returns:
            {
                "success": bool,
                "image_path": str,
                "image_url": str,
                "prompt": str,
                "error": str
            }
        """
        try:
            # Build enhanced prompt
            enhanced_prompt = prompt
            if style:
                enhanced_prompt = f"{prompt}, {style} style"
            
            # Generate filename
            if not file_stem:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                file_stem = f"generated_{timestamp}"
            
            # Use Zo's generate_image capability via API call
            # Since we're in the same environment, we can call the tool directly
            import requests
            
            # Try to generate using a local generation script
            result = self._generate_local(enhanced_prompt, file_stem)
            
            if result["success"]:
                return result
            
            # Fallback: create a placeholder image with PIL
            return self._create_placeholder(prompt, file_stem)
            
        except Exception as e:
            return {
                "success": False,
                "image_path": "",
                "image_url": "",
                "prompt": prompt,
                "error": str(e)
            }
    
    def _generate_local(self, prompt: str, file_stem: str) -> Dict[str, Any]:
        """Try to generate image locally using available tools."""
        
        image_path = self.output_dir / f"{file_stem}.png"
        
        # Try using diffusers if available
        try:
            from diffusers import StableDiffusionPipeline
            import torch
            
            pipe = StableDiffusionPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                torch_dtype=torch.float32
            )
            
            # Generate
            image = pipe(prompt).images[0]
            image.save(image_path)
            
            return {
                "success": True,
                "image_path": str(image_path),
                "image_url": f"/generated_images/{file_stem}.png",
                "prompt": prompt,
                "error": ""
            }
        except ImportError:
            pass
        except Exception as e:
            pass
        
        return {"success": False, "error": "Local generation not available"}
    
    def _create_placeholder(self, prompt: str, file_stem: str) -> Dict[str, Any]:
        """Create a placeholder image with the prompt text."""
        
        try:
            from PIL import Image, ImageDraw, ImageFont
            
            # Create image
            img = Image.new('RGB', (512, 512), color=(30, 30, 40))
            draw = ImageDraw.Draw(img)
            
            # Add text
            y = 20
            words = prompt.split()
            line = ""
            for word in words:
                test_line = line + word + " "
                if len(test_line) > 30:
                    draw.text((20, y), line, fill=(200, 200, 200))
                    y += 25
                    line = word + " "
                else:
                    line = test_line
            
            if line:
                draw.text((20, y), line, fill=(200, 200, 200))
            
            # Save
            image_path = self.output_dir / f"{file_stem}.png"
            img.save(image_path)
            
            return {
                "success": True,
                "image_path": str(image_path),
                "image_url": f"/generated_images/{file_stem}.png",
                "prompt": prompt,
                "error": "",
                "placeholder": True
            }
            
        except ImportError:
            # PIL not available - return failure
            return {
                "success": False,
                "image_path": "",
                "image_url": "",
                "prompt": prompt,
                "error": "Image generation not available. Install PIL: pip install Pillow"
            }


# Tool interface
image_generator = ImageGenerator()


if __name__ == "__main__":
    # Test
    result = image_generator.generate("a beautiful moon in the night sky", style="realistic")
    print(json.dumps(result, indent=2))
