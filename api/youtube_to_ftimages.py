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

# Model cache for loaded models (keyed by scope name)
EMBED_MODEL_CACHE = {}

from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse

from indexing.video_processing import download_youtube_video, extract_keyframes
from indexing.training_img_embedding import fine_tune_openclip_from_ftimages
from utils.ollama import describe_screen_with_ollama
from indexing.ocr import extract_text_from_image



# Config personalizzata
DEFAULT_PROMPT = """
Descrivi la schermata software elencando solo gli elementi dell’interfaccia utente (UI) e le loro funzioni, senza riferimenti visivi o commenti generali.
Elenca le funzioni presenti nel menu principale (barra blu in alto), tipicamente contiene le sezioni di navigazione principali del gestionale, ognuna rappresentata da un’etichetta testuale o icona (es. Front-Office, Controllo Accessi, Report, Planning, Istruttore, Pianificazione, Impostazioni).
Elenca le funzioni presenti nel menu secondario (barra grigia sotto il menu principale), tipicamente mostra le opzioni contestuali della sezione selezionata, con etichette testuali e icone (es. Rubrica, Proshop, Servizi, Dashboard CRM, Agenda Utenti, Giftcard/Coupon).
Descrivi dettagliatamente il contenuto del corpo centrale, tipicamente contiene il pannello funzionale della maschera. Cerca di identificare gli elementi più importanti come: elenco utenti, impostazione documento.
Per il corpo centrale elenca tutti i principali elementi di UI trovati e il contenuto del testo dei bottoni presenti e delle caselle di testo cercando di spiegarne l'uso nel contesto della maschera.
Non includere riferimenti al nome del software. Ignora elementi in basso o sulla barra nera.
Non usare il formato markdown, non usare i tag HTML, non usare le virgolette o asterischi *, non aggiungere valutazioni o tue spiegazioni.
La descrizione deve essere oggettiva, strutturata e funzionale all’individuazione degli oggetti UI e delle loro funzioni, senza spiegazioni aggiuntive.
"""

FTIMAGES_ENDPOINT = "http://localhost:5209/ftimages"

logger = logging.getLogger("api.youtube_to_ftimages")
logger.setLevel(logging.INFO)

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
    # Load model from cache if present, else load it and cache
    if scope not in EMBED_MODEL_CACHE:
        try:
            model_name = "ViT-B-32"
            model, preprocess, _ = open_clip.create_model_and_transforms(model_name)
            state_dict = torch.load(model_path, map_location="cpu")
            model.load_state_dict(state_dict)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = model.to(device)
            model.eval()
            EMBED_MODEL_CACHE[scope] = (model, preprocess, device)
        except Exception as e:
            return {"error": f"Could not load fine-tuned model: {e}"}
    model, preprocess, device = EMBED_MODEL_CACHE[scope]
    # Accept both a single string or list of strings
    imgs = request.input if isinstance(request.input, list) else [request.input]
    embeddings = []
    for idx, item in enumerate(imgs):
        try:
            # Remove data URI prefix if present
            if item.startswith("data:image"):
                item = item.split(",", 1)[1]
            image_data = base64.b64decode(item)
            image = Image.open(BytesIO(image_data)).convert("RGB")
        except Exception as e:
            return {"error": f"Invalid base64 image at position {idx}: {e}"}
        try:
            with torch.no_grad():
                image_tensor = preprocess(image).unsqueeze(0).to(device)
                img_feat = model.encode_image(image_tensor)
                emb = img_feat.cpu().numpy()[0]
        except Exception as e:
            return {"error": f"Error encoding image at position {idx}: {e}"}
        embeddings.append(emb.tolist())
    return {"embeddings": embeddings, "model": scope}

def get_ftimage_embedding_from_file(scope: str, file: UploadFile):
    """
    Calcola embedding immagine usando modello fine-tuning relativo allo scope.
    """
    model_path = f"./ft_images_{scope}.pth"
    if not os.path.isfile(model_path):
        raise HTTPException(status_code=404, detail=f"Model for scope '{scope}' not found: {model_path}")
    # Cache modello per efficienza
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
    # Leggi l'immagine da UploadFile
    image_data = file.file.read()
    image = Image.open(BytesIO(image_data)).convert("RGB")
    with torch.no_grad():
        image_tensor = preprocess(image).unsqueeze(0).to(device)
        img_feat = model.encode_image(image_tensor)
        emb = img_feat.cpu().numpy()[0]
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
