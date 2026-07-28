"""YouTube transcript fetching for NeuralAI.

Uses yt_dlp to pull the auto/subtitled transcript. Returns
{"transcript": str, "title": str, "duration": float}.
"""
import os
import re


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def fetch_transcript(url: str) -> dict:
    url = (url or "").strip()
    if "youtube.com" not in url and "youtu.be" not in url:
        raise ValueError("not a YouTube URL")
    try:
        import yt_dlp
    except Exception as e:
        raise RuntimeError(f"yt_dlp not installed: {e}")

    try:
        ydl_opts = {
            "quiet": True,
            "no_warnings": True,
            "skip_download": True,
            "writesubtitles": True,
            "writeautomaticsub": True,
            "subtitleslangs": ["en"],
            "simulate": True,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            title = info.get("title", "")
            duration = float(info.get("duration") or 0)
            subs = info.get("subtitles") or info.get("automatic_captions") or {}
            lines = []
            if "en" in subs:
                for seg in subs["en"]:
                    txt = _clean(seg.get("text", ""))
                    if txt:
                        lines.append(txt)
            transcript = " ".join(lines)
            return {"transcript": transcript, "title": title, "duration": duration}
    except Exception as e:
        return {"transcript": "", "title": "", "duration": 0, "error": str(e)}
