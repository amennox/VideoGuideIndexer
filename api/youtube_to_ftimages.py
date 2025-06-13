from fastapi import FastAPI, Request, BackgroundTasks, HTTPException, status, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from pathlib import Path
import shutil
import tempfile
import logging
import requests
import time
import json
from typing import List, Union, Optional
from io import BytesIO
import base64
from PIL import Image
import torch
import open_clip
import os
import numpy as np
import asyncio
import subprocess

# Model cache for loaded models (keyed by scope name)
EMBED_MODEL_CACHE = {}

from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse

from indexing.video_processing import download_youtube_video, extract_keyframes
from indexing.training_img_embedding import fine_tune_openclip_from_ftimages
from utils.ollama import describe_screen_with_ollama
from indexing.ocr import extract_text_from_image
from indexing.import_server import import_elements_from_json

from indexing.embedding_generation import get_ftimage_embedding

from indexing.pipeline import index_video
from core.config import TEMP_DIR, DOWNLOADS_DIR

#uvicorn api.youtube_to_ftimages:app --host 0.0.0.0 --port 8000 --reload

# Config personalizzata
DEFAULT_PROMPT = """
Descrivi la schermata software cms WORDPRESS elencando solo gli elementi dell’interfaccia utente (UI) e le loro funzioni, senza riferimenti visivi o commenti generali.
Elenca le funzioni presenti nel menu principale (barra in alto e a sinistra), tipicamente contiene le sezioni di navigazione principali del cms, ognuna rappresentata da un’etichetta testuale o icona 
Descrivi dettagliatamente il contenuto del corpo centrale, tipicamente contiene il pannello funzionale della maschera. Cerca di identificare gli elementi più importanti come: elenco utenti, impostazione documento.
Per il corpo centrale elenca tutti i principali elementi di UI trovati e il contenuto del testo dei bottoni presenti e delle caselle di testo cercando di spiegarne l'uso nel contesto della maschera.

Non includere riferimenti al nome del software. Ignora elementi circa la versione del software, barre degli indirizzi web.
Non usare il formato markdown, non usare i tag HTML, non usare le virgolette o asterischi *, non aggiungere valutazioni o tue spiegazioni.
La descrizione deve essere oggettiva, strutturata e funzionale all’individuazione degli oggetti UI e delle loro funzioni, senza spiegazioni aggiuntive.
"""

FTIMAGES_ENDPOINT = "http://localhost:5209/ftimages"

log = logging.getLogger("api.youtube_to_ftimages")
log.setLevel(logging.INFO)

class YouTubeToFTImagesRequest(BaseModel):
    scope: str
    businessId: str
    youtube_url: str
    prompt: str = None

def iter_progress(steps, label):
    for i in range(steps):
        yield f"data: {json.dumps({'progress': int((i+1)/steps*100), 'stage': label})}\n\n"

def stream_process(req: YouTubeToFTImagesRequest):
    try:
        # 1. Scarica il video da YouTube
        yield f"data: {json.dumps({'progress': 0, 'stage': 'downloading_video'})}\n\n"
        local_video = download_youtube_video(req.youtube_url)
        yield f"data: {json.dumps({'progress': 5, 'stage': 'video_downloaded'})}\n\n"

        # 2. Estrai keyframes (ogni 2s)
        frames = extract_keyframes(local_video, interval_s=2)
        total_frames = len(frames)
        if total_frames == 0:
            yield f"data: {json.dumps({'error': 'No frames extracted'})}\n\n"
            return

        for idx, frame_path in enumerate(frames):
            # 3. OCR sul frame
            ocr_text = extract_text_from_image(frame_path)
            # 4. Descrivi via Ollama (usando prompt custom o default)
            prompt = req.prompt if req.prompt else DEFAULT_PROMPT
            description = None
            for attempt in range(2):  # Retry se Ollama fail temporaneo
                try:
                    description = describe_screen_with_ollama(frame_path,ocr_text,prompt)
                    break
                except Exception as e:
                    time.sleep(1)
            description = description or "[Ollama error]"

            # 5. Invio su API ftimages
            files = {"File": open(frame_path, "rb")}
            data = {
                "BusinessId": req.businessId,
                "Scope": req.scope,
                "Description": description,
            }
            try:
                resp = requests.post(FTIMAGES_ENDPOINT, data=data, files=files)
            finally:
                files["File"].close()
                

            if resp.status_code not in (200, 201):
                yield f"data: {json.dumps({'error': f'Failed uploading image {frame_path.name}'})}\n\n"
            progress = int(5 + (idx+1) / total_frames * 95)
            yield f"data: {json.dumps({'progress': progress, 'stage': 'processing_frame', 'frame': frame_path.name})}\n\n"

        yield f"data: {json.dumps({'progress': 100, 'stage': 'done'})}\n\n"
        
        
        
    except Exception as e:
        yield f"data: {json.dumps({'error': str(e)})}\n\n"

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # O "*" per test, ma meglio il dominio specifico
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# monta la cartella "static" all'URL /static
app.mount("/static", StaticFiles(directory="static"), name="static")
@app.get("/")

def home():
    return RedirectResponse("/static/test_youtube.html")

@app.post("/youtube_to_ftimages/")
def youtube_to_elements(req: YouTubeToFTImagesRequest):
    """
    Scarica video YouTube, estrae keyframe, descrive ogni immagine con Ollama, invia a /ftimages.
    Streamma la progressione (Server-Sent Events, SSE).
    """
    return StreamingResponse(
        stream_process(req),
        media_type="text/event-stream"
    )

@app.post("/train_embeddings/")
async def train_embeddings(request: Request):
    data = await request.json()
    scope = data.get("scope")
    if not scope:
        return {"error": "Missing scope parameter"}

    async def event_stream():
        queue = asyncio.Queue()
        def sse_callback(progress, stage, info):
            msg = {"progress": progress, "stage": stage, "info": info}
            queue.put_nowait(f"data: {json.dumps(msg)}\n\n")
        # Lancia il training in un thread separato per non bloccare l'async generator
        import threading
        def training():
            fine_tune_openclip_from_ftimages(scope, sse_callback)
            queue.put_nowait(f"data: {json.dumps({'progress': 100, 'stage': 'done', 'info': 'Training completato'})}\n\n")
        threading.Thread(target=training, daemon=True).start()
        while True:
            chunk = await queue.get()
            yield chunk
            if '"progress": 100' in chunk:
                break

    return StreamingResponse(event_stream(), media_type="text/event-stream")

class EmbedRequest(BaseModel):
    input: Union[str, List[str]]
    model: Optional[str] = None  # Here, model == scope

@app.post("/api/embed")
async def embed_image(request: EmbedRequest):
    """
    Embedding API compatible with Ollama /api/embed.
    'model' must be the scope; loads the corresponding fine-tuned model 'ft_images_{scope}.pth'.
    Input: list of base64-encoded images or a single image (string).
    Returns: dict with 'embeddings' list and model name.
    """
    scope = request.model
    if not scope:
        return {"error": "Missing 'model' (scope) parameter."}
    model_path = f"./ft_images_{scope}.pth"
    if not os.path.isfile(model_path):
        return {"error": f"Model file for scope '{scope}' not found: {model_path}"}

    # Accept both a single string or list of strings
    imgs = request.input if isinstance(request.input, list) else [request.input]
    embeddings = []
    for idx, item in enumerate(imgs):
        try:
            # Remove data URI prefix if present
            if item.startswith("data:image"):
                item = item.split(",", 1)[1]
            image_data = base64.b64decode(item)
        except Exception as e:
            return {"error": f"Invalid base64 image at position {idx}: {e}"}
        try:
            emb = get_ftimage_embedding(scope, BytesIO(image_data))
            # sempre lista di float
            embeddings.append([float(x) for x in emb.tolist()] if hasattr(emb, "tolist") else [float(emb)])
        except Exception as e:
            return {"error": f"Error encoding image at position {idx}: {e}"}
        #embeddings.append(list(emb) if hasattr(emb, "__iter__") else emb)
    return {"embeddings": embeddings, "model": scope}

def get_ftimage_embedding_from_file(scope: str, file: UploadFile):
    """
    Calcola embedding immagine usando modello fine-tuning relativo allo scope.
    """
    emb = get_ftimage_embedding(scope, file.file)

    return emb

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    # Cosine sim standard: np.dot(a, b) / (norm(a) * norm(b))
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))



@app.post("/test_embeddings")
async def test_embeddings(
    scope: str = Form(...),
    immagine1: UploadFile = File(...),
    immagine2: UploadFile = File(...)
):
    """
    Calcola embedding per immagine1/immagine2 con modello fine-tuning {scope} e restituisce la similarità coseno.
    """
    try:
        emb1 = get_ftimage_embedding_from_file(scope, immagine1)
        emb2 = get_ftimage_embedding_from_file(scope, immagine2)
        sim = cosine_similarity(np.array(emb1), np.array(emb2))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {e}")
    return {"cosine_similarity": sim}

class YTIndexRequest(BaseModel):
    youtube_url: str = None
    scope: str
    businessId: str = "ssense_0001"

# --- Utility SSE ---
def sse(progress, stage, info=None, extra=None):
    payload = {"progress": progress, "stage": stage}
    if info: payload["info"] = info
    if extra: payload.update(extra)
    return f"data: {json.dumps(payload)}\n\n"

# --- ENDPOINT YTINDEX CON PROGRESS SSE ---
@app.post("/ytindex")
async def ytindex(
    scope: str = Form(...),
    videotitle : str = Form(...), 
    businessId: str = Form("ssense_0001"),
    youtube_url: str = Form(None),
    file: UploadFile = File(None), # Il parametro UploadFile
):
    # Dobbiamo leggere il contenuto del file *immediatamente* se presente,
    # prima che qualsiasi altra cosa possa chiudere il flusso.
    uploaded_file_content = None
    if file is not None:
        try:
            # Leggi tutto il contenuto del file in memoria come bytes
            # Questo è il punto critico dove si verifica l'errore.
            # Se fallisce ancora qui, significa che il file è chiuso prima ancora
            # che l'esecuzione entri nel blocco try della funzione generator().
            uploaded_file_content = await file.read()
            log.info(f"Successfully read {len(uploaded_file_content)} bytes from uploaded file.")
        except Exception as e:
            log.error(f"FATAL: Could not read uploaded file content: {e}", exc_info=True)
            # Dato che questo è un errore critico prima di iniziare il generator,
            # lo restituiamo direttamente come JSONResponse.
            return JSONResponse(
                {"error": f"Impossibile leggere il file caricato all'inizio: {str(e)}"}, 
                status_code=500
            )

    async def generator():
        log.info("Generator start for /ytindex")
        
        temp_files_to_clean = [] 
        local_video_path = None 

        try:
            if youtube_url:
                # Caso: URL YouTube
                yield sse(5, "downloading_video", "Scaricamento video da YouTube")
                downloaded_path = download_youtube_video(youtube_url)
                local_video_path = Path(downloaded_path)
                temp_files_to_clean.append(local_video_path)
                log.info(f"Video downloaded from YouTube: {local_video_path}")

            elif uploaded_file_content is not None:
                # Caso: File caricato (il contenuto è già in uploaded_file_content)
                yield sse(5, "saving_file", "Salvataggio file caricato")
                
                unique_filename = f"{os.urandom(8).hex()}_{file.filename}" # Usa file.filename dall'oggetto UploadFile
                local_video_path = TEMP_DIR / unique_filename
                
                # Scrivi il contenuto (già letto in memoria) sul disco
                try:
                    with open(local_video_path, "wb") as buffer:
                        buffer.write(uploaded_file_content)
                    log.info(f"Uploaded file content saved to: {local_video_path}")
                    temp_files_to_clean.append(local_video_path)
                except Exception as e:
                    log.error(f"Error writing uploaded file content to disk: {e}", exc_info=True)
                    yield sse(-1, "error", f"Errore nello scrivere il file caricato su disco: {str(e)}")
                    return 

            else:
                # Nessun URL YouTube o file fornito
                log.error("No video source provided (youtube_url or file).")
                yield sse(-1, "error", "Fornire un URL YouTube o caricare un file video.")
                return

            if not local_video_path or not local_video_path.exists():
                log.error(f"Local video file does not exist after initial processing: {local_video_path}")
                yield sse(-1, "error", "Impossibile accedere al file video salvato per l'elaborazione.")
                return


            yield sse(15, "uploading_media", "Upload video su /Media/upload")
            with open(local_video_path, "rb") as f:
                files = {"File": f}
                data = {"FolderType": "videos"}
                resp = requests.post("http://localhost:5209/Media/upload", files=files, data=data)

            if resp.status_code != 200:
                log.error(f"Error uploading video to /Media/upload (Status: {resp.status_code}, Response: {resp.text})")
                yield sse(-1, "error", f"Errore upload su Media: {resp.status_code} - {resp.text}")
                return

            resp_json = resp.json()
            video_url = resp_json.get("url")
            if not video_url:
                log.error("No URL found in Media upload response.")
                yield sse(-1, "error", "Errore: URL video non trovato nella risposta di Media upload.")
                return
            
            id_chunk = Path(video_url).stem

            yield sse(35, "indexing", "Esecuzione pipeline indicizzazione")
            json_output_path = TEMP_DIR / f"{id_chunk}_index.json"
            log.info(f"Indexing video ID: {id_chunk} from {local_video_path} to JSON: {json_output_path}")
            
            index_video(local_video_path, json_output_path)
            temp_files_to_clean.append(json_output_path)

            yield sse(75, "importing", "Import elementi via import_server.py")
            import_elements_from_json(str(json_output_path), id_chunk, scope, businessId,videotitle, log=log)

            yield sse(100, "done", "Processo completato!", extra={"video_id": id_chunk})
            yield f"data: {json.dumps({'success': True, 'video_id': id_chunk})}\n\n"

        except Exception as e:
            log.error(f"General process error in /ytindex: {str(e)}", exc_info=True)
            yield sse(-1, "error", f"Si è verificato un errore inaspettato durante l'elaborazione: {str(e)}")
        finally:
            #for f_path in temp_files_to_clean:
            #    try:
            #        if f_path and f_path.exists():
            #            f_path.unlink()
            #            log.info(f"Cleaned up temporary file: {f_path}")
            #    except Exception as clean_e:
            #        log.warning(f"Failed to clean up temporary file {f_path}: {clean_e}")
            log.info("Generator finished and cleaned up.")

    return StreamingResponse(generator(), media_type="text/event-stream")


# --- (facoltativo) endpoint ytindex_sync per old client ---
@app.post("/ytindex_sync")
async def ytindex_sync(
    scope: str = Form(...),
    businessId: str = Form("ssense_0001"),
    youtube_url: str = Form(None),
    file: UploadFile = File(None),
):
    """
    Versione tradizionale per chiamate senza SSE (risponde solo a fine processo).
    """
    try:
        # ...stessa logica della generator(), ma senza yield/sse...
        if youtube_url:
            local_video = download_youtube_video(youtube_url)
        elif file is not None:
            video_path = TEMP_DIR / file.filename
            with open(video_path, "wb") as f:
                shutil.copyfileobj(file.file, f)
            local_video = video_path
        else:
            return JSONResponse({"error": "Provide either youtube_url or file"}, status_code=400)
        with open(local_video, "rb") as f:
            files = {"File": f}
            data = {"FolderType": "videos"}
            resp = requests.post("http://localhost:5209/Media/upload", files=files, data=data)
        if resp.status_code != 200:
            return JSONResponse({"error": "Error uploading video to Media"}, status_code=500)
        resp_json = resp.json()
        video_url = resp_json.get("url")
        id_chunk = Path(video_url).stem
        json_output = TEMP_DIR / f"{id_chunk}_index.json"
        index_video(local_video, json_output)
        import_cmd = [
            "python", "import_server.py", 
            str(json_output), 
            "--id_chunk", id_chunk,
            "--scope", scope,
            "--business_id", businessId
        ]
        result = subprocess.run(import_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            return JSONResponse({"error": result.stderr}, status_code=500)
        return {"success": True, "video_id": id_chunk}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)