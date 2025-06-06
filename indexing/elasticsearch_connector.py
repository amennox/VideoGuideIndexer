import requests
import base64
import numpy as np
from elasticsearch import Elasticsearch
from pathlib import Path
import logging
from core.config import (
    OLLAMA_EMBED_URL,
    EMBEDDING_MODEL,
    DOCCLIP_API_URL,
    ELASTICSEARCH_URL,
    ELASTICSEARCH_INDEX,
    IMAGE_EMBED_DIM,
    TEXT_EMBED_DIM
)

log = logging.getLogger("indexing.elasticsearch_connector")

# Funzioni di embedding
def get_text_embedding(text):
    payload = {"model": EMBEDDING_MODEL, "input": text}
    resp = requests.post(OLLAMA_EMBED_URL, json=payload, timeout=120)
    resp.raise_for_status()
    embedding = resp.json().get("embeddings", [[]])[0]
    return embedding


def get_image_embedding(image_path):
    with open(image_path, "rb") as f:
        img_bytes = f.read()
    img_b64 = base64.b64encode(img_bytes).decode()
    payload = {"input": img_b64}
    resp = requests.post(DOCCLIP_API_URL, json=payload, timeout=120)
    resp.raise_for_status()
    embedding = resp.json().get("embeddings", [[]])[0]
    # LOG: dimensione effettiva
    log.info(f"Image embedding length: {len(embedding)}")
    log.info(f"First 5 embedding vect values: {embedding[:5]}")
    return embedding


def knn_search(image_path: Path, text: str = None, k: int = 10):
    es = Elasticsearch(ELASTICSEARCH_URL)

    # Image embedding
    image_embedding = get_image_embedding(image_path)
    if not image_embedding or len(image_embedding) != IMAGE_EMBED_DIM:
        log.error("Invalid image embedding.")
        return []

    query = {
        "size": k,
        "query": {
            "script_score": {
                "query": {"match_all": {}},
                "script": {
                    "source": "cosineSimilarity(params.query_vector, 'imageVect') + 1.0",
                    "params": {"query_vector": image_embedding}
                }
            }
        }
    }

    # Se vuoi aggiungere anche l'embedding del testo come boost
    if text:
        text_embedding = get_text_embedding(text)
        if text_embedding and len(text_embedding) == TEXT_EMBED_DIM:
            # Qui si potrebbe fare una somma pesata dei punteggi, o una query mista (per semplicità mostro solo imagevect)
            pass
        else:
            log.warning("Invalid text embedding, skipping text similarity.")

    res = es.search(index=ELASTICSEARCH_INDEX, body=query)
    hits = res.get("hits", {}).get("hits", [])
    if not hits:
        log.warning("No hits found in Elasticsearch.")
        return []

    max_score = hits[0]["_score"]
    results = []
    for hit in hits:
        score = hit["_score"]
        ranking = (score / max_score) * 100 if max_score else 0
        source = hit["_source"]
        results.append({
            "title": source.get("title"),
            "imageurl": source.get("imageurl"),
            "fulltext": source.get("fulltext"),
            "ranking": round(ranking, 2)
        })
    return results



if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_image = Path("test.jpg")
    if not test_image.exists():
        log.error("File 'test.jpg' not found.")

    results = knn_search(test_image, k=10)

    log.info("Top 5 Elasticsearch results:")
    for idx, res in enumerate(results, start=1):
        log.info(f"{idx}. Title: {res['title']} - Image URL: {res['imageurl']} - Ranking: {res['ranking']}% Fulltext: {res['fulltext'][:100]}...")