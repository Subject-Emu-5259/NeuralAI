# tools/voice_transcriber.py
#
# Voice note transcription for NeuralAI
# Transcribes audio files using Zo's transcription capability

import os
import json
from typing import Dict, Any, Optional
from pathlib import Path

class VoiceTranscriber:
    """Transcribe voice notes and audio files."""
    
    SUPPORTED_FORMATS = ['.mp3', '.wav', '.m4a', '.ogg', '.flac', '.aac', '.opus']
    
    def __init__(self, upload_dir: str = "/home/workspace/Documents"):
        self.upload_dir = Path(upload_dir)
        self.upload_dir.mkdir(parents=True, exist_ok=True)
    
    def transcribe(self, audio_path: str, language: str = "en") -> Dict[str, Any]:
        """
        Transcribe an audio file.
        
        Args:
            audio_path: Path to the audio file
            language: Language code (en, es, fr, etc.)
        
        Returns:
            {
                "success": bool,
                "text": str,
                "language": str,
                "duration": float,
                "error": str
            }
        """
        try:
            path = Path(audio_path)
            
            # Check if file exists
            if not path.exists():
                return {
                    "success": False,
                    "text": "",
                    "language": language,
                    "duration": 0,
                    "error": f"File not found: {audio_path}"
                }
            
            # Check format
            if path.suffix.lower() not in self.SUPPORTED_FORMATS:
                return {
                    "success": False,
                    "text": "",
                    "language": language,
                    "duration": 0,
                    "error": f"Unsupported format: {path.suffix}. Supported: {', '.join(self.SUPPORTED_FORMATS)}"
                }
            
            # Use Zo's transcribe_audio capability
            # This is a placeholder - actual implementation would call Zo's API
            # For now, return a helpful message
            return {
                "success": True,
                "text": f"[Transcription ready] Audio file '{path.name}' can be transcribed. Use the transcribe tool to process it.",
                "language": language,
                "duration": 0,
                "audio_path": str(path),
                "error": ""
            }
            
        except Exception as e:
            return {
                "success": False,
                "text": "",
                "language": language,
                "duration": 0,
                "error": str(e)
            }
    
    def list_audio_files(self, directory: str = None) -> Dict[str, Any]:
        """List all audio files in a directory."""
        search_dir = Path(directory) if directory else self.upload_dir
        
        if not search_dir.exists():
            return {"success": False, "files": [], "error": "Directory not found"}
        
        files = []
        for fmt in self.SUPPORTED_FORMATS:
            files.extend(search_dir.glob(f"*{fmt}"))
        
        return {
            "success": True,
            "files": [str(f) for f in files],
            "count": len(files),
            "error": ""
        }


voice_transcriber = VoiceTranscriber()

if __name__ == "__main__":
    result = voice_transcriber.list_audio_files()
    print(json.dumps(result, indent=2))
