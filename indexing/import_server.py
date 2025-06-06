import json
import requests
from pathlib import Path
import sys
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("indexing.import_server")

# === CONFIG ===
API_BASE_URL = "http://localhost:5209"
ELEMENTS_ENDPOINT = f"{API_BASE_URL}/elements"
IMAGES_ENDPOINT = f"{API_BASE_URL}/images"
ID_CHUNK = "sr-01-cmd"
SCOPE = "video_guide"
BUSINESS_ID = "znext_0001"

def main(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    video_name = data.get("video")
    # Title richiesto: nome file video senza estensione, _ e - inclusi, estensione esclusa
    title = Path(video_name).stem

    segments = data.get("segments", [])
    for seg_idx, segment in enumerate(segments):
        interval = segment.get("interval", {})
        start = interval.get("start", 0)
        element_id = f"{ID_CHUNK}-{start}"
        # Costruisci payload elements - mettere nella commandUrl il link al video con start
        elements_payload = {
            "id": element_id,
            "scope": SCOPE,
            "title": title,
            "commands": [
                {
                    "commandName": "Apri il video",
                    "commandUrl": f"http://localhost:5209/video?id=SR-01&start={start}"                   
                }
            ],
            "fulltext": segment.get("fulltext", ""),
            "businessId": BUSINESS_ID,
            "liveDataUrl": "",
            "liveDataTemplate": "",
            "liveDataValidation": ""
        }
        # POST /elements
        #log.info(f"[Segment {seg_idx}] POST /elements: {elements_payload}")
        try:
            resp = requests.post(ELEMENTS_ENDPOINT, json=elements_payload)
            resp.raise_for_status()
            log.info(f"  -> Success [{resp.status_code}]")
        except Exception as e:
            log.error(f"  -> Error POST /elements: {e}")

        # Per ogni screenshot del segmento
        for shot_idx, screenshot in enumerate(segment.get("screenshots", [])):
            image_name = screenshot.get("image")
            image_path = Path("frames") / title / image_name
            if not image_path.exists():
                log.warning(f"  -> [Screenshot {shot_idx}] Immagine non trovata: {image_path}")
                continue

   
            data_img = {
                "Scope": SCOPE,
                "BusinessId": BUSINESS_ID,
                "Title": image_name,
                "Fulltext": screenshot.get("description", ""),
                "ElementId": element_id
            }
            files = {"Image": open(image_path, "rb")}
            # POST /images
             #log.info(f"    [Screenshot {shot_idx}] POST /images: {data_img}, image={image_path}")
            try:
                resp = requests.post(IMAGES_ENDPOINT, data=data_img, files=files)
                resp.raise_for_status()
                log.info(f"      -> Success [{resp.status_code}]")
            except Exception as e:
                log.error(f"      -> Error POST /images: {e}")
            finally:
                files["Image"].close()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python import_server.py <path_json>")
        sys.exit(1)
    main(sys.argv[1])
