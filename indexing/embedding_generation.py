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

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python embedding_generation.py <output_json_from_pipeline>")
        exit(1)
    index_embeddings_from_pipeline(sys.argv[1])
