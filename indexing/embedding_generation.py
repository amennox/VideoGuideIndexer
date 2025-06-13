import json
import requests
from pathlib import Path
from elasticsearch import Elasticsearch
import base64
from core.config import (
    OLLAMA_EMBED_URL,
    EMBEDDING_MODEL,
    DOCCLIP_API_URL,
    ELASTICSEARCH_URL,
    ELASTICSEARCH_INDEX,
    DEFAULT_BUSINESSID,
    DEFAULT_ELEMENTID,
    DEFAULT_SCOPE,
    TEXT_EMBED_DIM,
    IMAGE_EMBED_DIM
)
import os
import torch
from PIL import Image
from io import BytesIO
import open_clip
import numpy as np

import logging

log = logging.getLogger("indexing.embedding_generation")


def get_text_embedding(text):
    text = text.encode("utf-8", errors="ignore").decode("utf-8")
    payload = {
        "model": EMBEDDING_MODEL,
        "input": text
    }
    
    resp = requests.post(OLLAMA_EMBED_URL, json=payload, timeout=120)
    resp.raise_for_status()
    result = resp.json()
    embedding = result.get("embeddings", [[]])[0]
    log.info(f"Text sent to Ollama for Embedding")
    if not embedding or len(embedding) != TEXT_EMBED_DIM:
        log.warning(f"Ollama returned invalid text embedding (dim={len(embedding)}")
        return None
    return embedding

def get_image_embedding(image_path):
    try:
        with open(image_path, "rb") as f:
            img_bytes = f.read()
        img_b64 = base64.b64encode(img_bytes).decode()
        payload = {"input": img_b64}
        
        # DEBUG: verifica payload
        print(f"[CLIENT DEBUG] Payload length: {len(img_b64)} chars")
        print(f"[CLIENT DEBUG] Payload snippet: {img_b64[:50]}...")

        resp = requests.post(DOCCLIP_API_URL, json=payload, timeout=120)
        resp.raise_for_status()
        result = resp.json()
        embedding = result.get("embeddings", [[]])[0]
        log.info(f"Image sent to DOCCLIP for Embedding")

        if not embedding or len(embedding) != IMAGE_EMBED_DIM:
            log.warning(f"DocCLIP returned invalid embedding (dim={len(embedding)}) for {image_path}")
            return None
        return embedding
    except Exception as e:
        log.error(f"DocCLIP embedding for image failed: {e}")
        return None

def index_embeddings_from_pipeline(output_json_path):
    es = Elasticsearch(ELASTICSEARCH_URL)
    output_json_path = Path(output_json_path)
    with open(output_json_path, "r", encoding="utf-8") as f:
        indicizzazione = json.load(f)
    title = Path(indicizzazione["video"]).stem

    for segment in indicizzazione.get("segments", []):
        for screenshot in segment.get("screenshots", []):
            image_name = screenshot.get("image")
            description = screenshot.get("description", "")
            imageurl = f"{title}/{image_name}"
            fulltext = description

         
            # --- Text embedding
            fulltextvect = get_text_embedding(description)
            # --- Image embedding (base64)
            image_path = Path("frames") / title / image_name
            log.info(f"Image path: {image_path}")
            imagevect = get_image_embedding(str(image_path))

            # --- Validazione: salta se embedding assente o sbagliato
            if not fulltextvect:
                log.warning(f"Text embedding missing or invalid for {imageurl}")
                continue
            if not imagevect:
                log.warning(f"Image embedding missing or invalid for {imageurl}")
                continue

            doc = {
                "title": title,
                "scope": DEFAULT_SCOPE,
                "businessid": DEFAULT_BUSINESSID,
                "fulltext": fulltext,
                "fulltextvect": fulltextvect,
                "imageurl": imageurl,
                "imagevect": imagevect,
                "elementid": DEFAULT_ELEMENTID
            }
            try:
                es.index(index=ELASTICSEARCH_INDEX, document=doc)
                
                log.info(f"Elastic Indexed: {imageurl}")
            except Exception as e:
                log.error(f"Failed to index {imageurl}: {e}")
                
EMBED_MODEL_CACHE = {}

def get_ftimage_embedding(scope: str, image_input):
    """
    Computes embedding for an image using the fine-tuned model for the given scope.
    - image_input: path to file (str/Path) or a file-like object (with .read())
    Returns: embedding numpy array.
    """
    log.info(f"Getting image embedding for scope '{scope}' from {image_input}")
    model_path = f"./ft_images_{scope}.pth"
    if not os.path.isfile(model_path):
        log.error(f"Model file not found for scope '{scope}': {model_path}")
        raise FileNotFoundError(f"Model for scope '{scope}' not found: {model_path}")
    if scope not in EMBED_MODEL_CACHE:
        model_name = "ViT-B-32"
        model, preprocess, _ = open_clip.create_model_and_transforms(model_name)
        state_dict = torch.load(model_path, map_location="cpu")
        model.load_state_dict(state_dict)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        model.eval()
        EMBED_MODEL_CACHE[scope] = (model, preprocess, device)
    model, preprocess, device = EMBED_MODEL_CACHE[scope]
    # Determina se image_input è path o file-like
    if hasattr(image_input, "read"):
        # file-like
        image_data = image_input.read()
        image = Image.open(BytesIO(image_data)).convert("RGB")
    else:
        # path
        image = Image.open(image_input).convert("RGB")
    with torch.no_grad():
        image_tensor = preprocess(image).unsqueeze(0).to(device)
        img_feat = model.encode_image(image_tensor)
        emb = img_feat.cpu().numpy()[0]
    log.info(f"Image embedding computed for {image_input} (dim={len(emb)})")
    return emb

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python embedding_generation.py <output_json_from_pipeline>")
        exit(1)
    index_embeddings_from_pipeline(sys.argv[1])
