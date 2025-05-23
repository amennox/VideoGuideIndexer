from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
from pathlib import Path
import uuid
import logging
from core.config import TEMP_DIR
from indexing.elasticsearch_connector import knn_search
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse

from utils.ollama import describe_screen_with_ollama
from indexing.ocr import extract_text_from_image  # o percorso corretto della funzione OCR


log = logging.getLogger("api.query")
app = FastAPI()

class QueryUrlRequest(BaseModel):
    image_url: str
    text: str = None
    k: int = 5

class QueryFileRequest(BaseModel):
    file_id: str
    text: str = None
    k: int = 5
    
# monta la cartella "static" all'URL /static
app.mount("/static", StaticFiles(directory="static"), name="static")
@app.get("/")

def home():
    return RedirectResponse("/static/testclient.html")

@app.post("/query_url")
def query_from_url(req: QueryUrlRequest):
    # Scarica l'immagine da URL e salva su TEMP_DIR
    try:
        response = requests.get(req.image_url)
        if response.status_code != 200:
            raise HTTPException(status_code=400, detail=f"Impossibile scaricare l'immagine ({req.image_url})")
        ext = req.image_url.split('.')[-1].split('?')[0][:4]  # jpg/png/webp
        tmpfile = TEMP_DIR / f"query_{uuid.uuid4().hex[:8]}.{ext}"
        with open(tmpfile, "wb") as f:
            f.write(response.content)
        log.info(f"Scaricata immagine da {req.image_url} in {tmpfile}")
    except Exception as e:
        log.error(f"Errore download immagine: {e}")
        raise HTTPException(status_code=400, detail=f"Errore download immagine: {e}")

    try:
        results = knn_search(tmpfile, text=req.text, k=req.k)
    finally:
        try:
            tmpfile.unlink()  # Rimuove file temporaneo
        except Exception:
            pass

    return {"results": results}

@app.post("/query")
def query_from_file(req: QueryFileRequest):
    img_path = TEMP_DIR / req.file_id
    if not img_path.exists():
        raise HTTPException(status_code=404, detail="Immagine non trovata")

    # Se text è vuoto, usa OCR + descrizione ollama
    text = req.text
    #if not text or not text.strip():
    #    try:
    #        ocr_text = extract_text_from_image(img_path)
   #         text = describe_screen_with_ollama(img_path, ocr_text)
    #        log.info(f"Text from image: {text}")
    #    except Exception as e:
     #       raise HTTPException(status_code=500, detail=f"Errore estrazione descrizione automatica: {e}")

    results = knn_search(img_path, text=text, k=req.k)
    try:
        img_path.unlink(missing_ok=True)  # Pulizia automatica
    except Exception:
        pass
    return {"results": results}

from fastapi import UploadFile, File

@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    ext = file.filename.split('.')[-1]
    fname = f"{uuid.uuid4().hex[:8]}.{ext}"
    dest = TEMP_DIR / fname
    with open(dest, "wb") as f:
        f.write(await file.read())
    log.info(f"Ricevuto file: {fname}")
    return {"file_id": fname}

# Esempio test (da chiamare solo manualmente)
# uvicorn api.query:app --host 0.0.0.0 --port 5050 --reload
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("query:app", host="0.0.0.0", port=5050, reload=True)
