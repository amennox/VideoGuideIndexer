"""
ocr.py – OCR utilities for Video Indexer.
Extracts text from images using Tesseract OCR.
Configuration via config.py (languages, tesseract_cmd).
"""

from pathlib import Path
from typing import Optional
from PIL import Image
from utils.ollama import describe_screen_with_ollama
import pytesseract
import os

from core.config import TESSERACT_LANGUAGES

# (Optional) Set explicit tesseract cmd if needed
TESSERACT_CMD = os.getenv("TESSERACT_CMD")
if TESSERACT_CMD:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

def extract_text_from_image(
    image_path: Path,
    lang: Optional[str] = None
) -> str:
    """
    Extracts text from an image using Tesseract OCR.
    - image_path: Path to image file
    - lang: OCR language(s), defaults to TESSERACT_LANGUAGES from config
    Returns the extracted text as a string.
    """
    lang = lang or TESSERACT_LANGUAGES
    img = Image.open(image_path)
    text = pytesseract.image_to_string(img, lang=lang)
    return text.strip()

# Example usage
if __name__ == "__main__":
    from core.config import FRAMES_DIR
    import sys

    # Demo: OCR the first frame in FRAMES_DIR
    frames = list(FRAMES_DIR.rglob("frame_*.jpg"))
    if not frames:
        print("No frames found in FRAMES_DIR.")
        sys.exit(1)
    frame_path = frames[0]
    print(f"OCR on {frame_path.name} ...")
    txt = extract_text_from_image(frame_path)
    print("\n--- OCR RESULT ---\n")
    print(txt)
    descr = describe_screen_with_ollama(frame_path, txt)
    print("\n--- OLLAMA UI DESCRIPTION ---\n", descr)