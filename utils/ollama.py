"""
utils/ollama.py – Helper functions for calling Ollama models.
"""

import requests
import base64
from core.config import OLLAMA_URL, OLLAMA_LLM_MODEL, OLLAMA_SCREENSHOT_PROMPT,OLLAMA_TITLE_ABSTRACT_PROMPT,OLLAMA_TEXT_PROMPT
import logging

log = logging.getLogger("utils.ollama")

def describe_screen_with_ollama(image_path, ocr_text) -> str:
    """
    Sends image + OCR text to Ollama (Gemma3:4b) for a structured screenshot description.
    """
    with open(image_path, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode()
    # Build prompt from config, replacing {ocr_text}
    prompt = OLLAMA_SCREENSHOT_PROMPT.replace("{ocr_text}", ocr_text)
    payload = {
        "model": OLLAMA_LLM_MODEL,
        "prompt": prompt,
        "images": [image_b64],
        "stream": False
    }
    
    log.info(f"Sending image {image_path} to Ollama")
    r = requests.post(OLLAMA_URL, json=payload, timeout=300)
    if r.status_code == 200:
        description = r.json().get("response", "").strip()
        description = description.replace("**", "")
        return description
    return f"[Ollama Error] Status {r.status_code}: {r.text}"

def split_text_for_ollama(text, max_chars=300_000):
    """Splits text on sentence boundaries, max max_chars per block."""
    import re
    sentences = re.split(r'(?<=[.!?]) +', text)
    blocks = []
    curr = ""
    for sent in sentences:
        if len(curr) + len(sent) > max_chars:
            if curr:
                blocks.append(curr.strip())
            curr = sent
        else:
            curr += " " + sent
    if curr.strip():
        blocks.append(curr.strip())
    return blocks

def correct_text_with_ollama(text, context=None, max_chars=300_000):
    import requests

    blocks = split_text_for_ollama(text, max_chars)
    corrected = []
    for i, block in enumerate(blocks):
        prompt = OLLAMA_TEXT_PROMPT
        if context:
            prompt += f"\nContext: {context}\n"
        prompt += f"\nText to correct:\n{block}\n"
        payload = {
            "model": OLLAMA_LLM_MODEL,
            "prompt": prompt,
            "stream": False
        }
        log.info(f"Sending chunk {i+1}/{len(blocks)} to Ollama ({len(block)} chars)")
        r = requests.post(OLLAMA_URL, json=payload, timeout=600)
        if r.status_code == 200:
            resp = r.json().get("response", "").strip()
            corrected.append(resp)
        else:
            corrected.append(block)
    return "\n".join(corrected)

def generate_title_and_abstract(segments):
    prompt = OLLAMA_TITLE_ABSTRACT_PROMPT.replace(
        "{fulltext}",
        "\n\n".join(s["fulltext"] for s in segments)
    )
    payload = {
        "model": OLLAMA_LLM_MODEL,
        "prompt": prompt,
        "stream": False
    }
    try:
        r = requests.post(OLLAMA_URL, json=payload, timeout=120)
        if r.status_code == 200:
            lines = r.json().get("response", "").strip().split("\n", 1)
            title = lines[0].strip()
            abstract = lines[1].strip() if len(lines) > 1 else ""
            return title, abstract
    except Exception as e:
        import logging
        logging.getLogger("ollama").error(f"Ollama error in title/abstract generation: {e}")
    return "Video tutorial", ""