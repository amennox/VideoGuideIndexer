"""
transcription.py – Speech-to-text utilities for Video Indexer.
Transcribes audio files using Whisper (or faster-whisper).
All configuration is read from config.py.
"""

import whisper
import torch
from pathlib import Path
from typing import Dict, Any, Optional
from core.config import WHISPER_MODEL
import logging

log = logging.getLogger("indexing.transcription")


def transcribe_audio(
    audio_path: Path,
    model_name: str = WHISPER_MODEL,
    language: Optional[str] = None,
    return_segments: bool = False
) -> Dict[str, Any]:
    """
    Transcribes an audio file using Whisper.
    - audio_path: Path to WAV/MP3/etc.
    - model_name: Whisper model (e.g. "base", "medium", "large").
    - language: Optional ISO code (e.g. "it", "en"). If None, auto-detect.
    - return_segments: If True, returns segment list as well.
    Returns dict with at least {'text': ...}, plus 'segments' if requested.
    """
    # Auto-select CUDA if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Loading Whisper model: {model_name} on {device} ...")
    model = whisper.load_model(model_name, device=device)
    log.info(f"Transcribing audio: {audio_path.name}")

    # Prepare options
    options = {}
    if language:
        options["language"] = language

    result = model.transcribe(str(audio_path), **options)
    text = result["text"].strip()
    if return_segments:
        return {"text": text, "segments": result.get("segments", [])}
    return {"text": text}

# Example usage
if __name__ == "__main__":
    import sys
    from core.config import TEMP_DIR

    # Demo: transcribe the first .wav file in TEMP_DIR
    wav_files = list(TEMP_DIR.glob("*.wav"))
    if not wav_files:
        log.error(f"o WAV file found in TEMP_DIR. Please generate audio first.")
        print("No WAV file found in TEMP_DIR. Please generate audio first.")
        sys.exit(1)

    wav_path = wav_files[0]
    out = transcribe_audio(wav_path)
    print("\n--- TRANSCRIPTION ---\n")
    print(out["text"])
