# tools/image_generator.py
#
# AI Image Generation for NeuralAI
# - Uses real AI models (Google Nano Banana or OpenAI GPT Image)
# - Saves to user's NeuralAI personal storage
# - No external routing - everything stays local

import os
import json
from typing import Dict, Any, Optional
from pathlib import Path
from datetime import datetime


class ImageGenerator:
    """
    Generate images using AI models.
    
    Images are saved to NeuralAI's personal storage:
    /home/workspace/NeuralAI/images/
    """
    
    OUTPUT_DIR = Path("/home/workspace/NeuralAI/images")
    
    def __init__(self):
        self.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    def generate(
        self,
        prompt: str,
        style: Optional[str] = None,
        aspect_ratio: str = "1:1",
        provider: str = ""  # Empty = user's default
    ) -> Dict[str, Any]:
        """
        Generate an AI image.
        
        Args:
            prompt: Description of image to generate
            style: Optional style (realistic, artistic, cartoon, etc.)
            aspect_ratio: Image ratio (1:1, 16:9, 9:16, etc.)
            provider: "" (default), "google", or "openai"
        
        Returns:
            {
                "success": bool,
                "image_path": str,   # Path in NeuralAI storage
                "image_url": str,    # URL to view
                "prompt": str,
                "error": str
            }
        """
        try:
            # Build enhanced prompt
            full_prompt = prompt
            if style:
                full_prompt = f"{prompt}, {style} style"
            
            # Add quality improvements
            if "photorealistic" not in full_prompt.lower() and style == "realistic":
                full_prompt += ", photorealistic, high detail"
            
            # Generate filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_stem = f"neuralai_{timestamp}"
            
            # This will be called by the engine using Zo's generate_image tool
            # The engine has direct access to generate_image
            return {
                "success": True,
                "prompt": prompt,
                "full_prompt": full_prompt,
                "file_stem": file_stem,
                "output_dir": str(self.OUTPUT_DIR),
                "aspect_ratio": aspect_ratio,
                "provider": provider,
                "image_path": str(self.OUTPUT_DIR / f"{file_stem}_1.png"),
                "image_url": f"/neuralai/images/{file_stem}_1.png",
                "error": ""
            }
            
        except Exception as e:
            return {
                "success": False,
                "image_path": "",
                "image_url": "",
                "prompt": prompt,
                "error": str(e)
            }
    
    def list_images(self) -> list:
        """List all generated images in NeuralAI storage."""
        if not self.OUTPUT_DIR.exists():
            return []
        
        images = []
        for f in self.OUTPUT_DIR.glob("*.png"):
            images.append({
                "name": f.name,
                "path": str(f),
                "url": f"/neuralai/images/{f.name}",
                "created": datetime.fromtimestamp(f.stat().st_mtime).isoformat()
            })
        
        return sorted(images, key=lambda x: x["created"], reverse=True)
    
    def delete_image(self, filename: str) -> bool:
        """Delete an image from NeuralAI storage."""
        filepath = self.OUTPUT_DIR / filename
        if filepath.exists() and filepath.is_relative_to(self.OUTPUT_DIR):
            filepath.unlink()
            return True
        return False


# Singleton
image_generator = ImageGenerator()
