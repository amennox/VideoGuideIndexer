import json
import requests
from pathlib import Path
import sys
import logging
import argparse
from indexing.embedding_generation import get_ftimage_embedding

# Setup logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("indexing.import_server")

API_BASE_URL = "http://localhost:5209"
ELEMENTS_ENDPOINT = f"{API_BASE_URL}/elements"
IMAGES_ENDPOINT = f"{API_BASE_URL}/images"

def import_elements_from_json(json_path, id_chunk, scope, business_id, videotitile, log=None):
    import json
    import requests
    from pathlib import Path

    API_BASE_URL = "http://localhost:5209"
    ELEMENTS_ENDPOINT = f"{API_BASE_URL}/elements"
    IMAGES_ENDPOINT = f"{API_BASE_URL}/images"

    # Carica file JSON generato dalla pipeline
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    video_name = data.get("video")
    title = videotitile #Path(video_name).stem
    snappath=Path(video_name).stem
    segments = data.get("segments", [])

    for seg_idx, segment in enumerate(segments):
        interval = segment.get("interval", {})
        start = interval.get("start", 0)
        element_id = f"{id_chunk}-{start}"
        elements_payload = {
            "id": element_id,
            "scope": scope,
            "title": title,
            "commands": [
                {
                    "commandName": "Apri il video",
                    "commandUrl": f"http://localhost:5209/video?id={id_chunk}&start={start}"
                }
            ],
            "fulltext": segment.get("fulltext", ""),
            "businessId": business_id,
            "liveDataUrl": "",
            "liveDataTemplate": "",
            "liveDataValidation": ""
        }
        try:
            resp = requests.post(ELEMENTS_ENDPOINT, json=elements_payload)
            resp.raise_for_status()
            if log: log.info(f"[Segment {seg_idx}] POST /elements OK [{resp.status_code}] id={element_id}")
        except Exception as e:
            if log: log.error(f"[Segment {seg_idx}] Error POST /elements: {e} payload={elements_payload}")

        for shot_idx, screenshot in enumerate(segment.get("screenshots", [])):
            image_name = screenshot.get("image")
            image_path = Path("frames") / snappath / image_name
            if not image_path.exists():
                if log: log.warning(f"  -> [Screenshot {shot_idx}] Immagine non trovata: {image_path}")
                continue
            data_img = {
                "Scope": scope,
                "BusinessId": business_id,
                "Title": image_name,
                "Fulltext": screenshot.get("description", ""),
                "ElementId": element_id
            }
            
            if not image_path.exists():
                log.warning(f"  -> [Screenshot {shot_idx}] Immagine non trovata: {image_path}")
                continue
            try:
                image_vect = get_ftimage_embedding(scope, image_path)
                log.info(f"  -> [Screenshot {shot_idx}] Embedding image {image_name} OK")
                data_img["ImageVect"] = json.dumps(image_vect.tolist())
            except Exception as e:
                log.warning(f"Embedding failed for {image_path}: {e}")
                data_img["ImageVect"] = "[]"
                
            log.info(f"  -> [Screenshot {shot_idx}] POST /images {image_name}")
            
            files = {"Image": open(image_path, "rb")}
            try:
                resp = requests.post(IMAGES_ENDPOINT, data=data_img, files=files)
                resp.raise_for_status()
                if log: log.info(f"    [Screenshot {shot_idx}] POST /images OK [{resp.status_code}] {image_name}")
            except Exception as e:
                if log: log.error(f"    [Screenshot {shot_idx}] Error POST /images: {e} for {image_name} data_img={data_img}")
            finally:
                files["Image"].close()
