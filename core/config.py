"""
config.py – Centralized configuration for Video Indexer project
Handles environment variables, default values, and project constants.
"""

import os
from pathlib import Path
import logging

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

def setup_logging():
    logging.basicConfig(
        level=LOG_LEVEL,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            # Puoi aggiungere qui un FileHandler se vuoi anche un log file
        ]
    )

setup_logging()

# === BASE DIRS ===
BASE_DIR = Path(__file__).resolve().parent.parent  # src/
TEMP_DIR = BASE_DIR / "temp"
TEMP_DIR.mkdir(exist_ok=True)

# === ENVIRONMENT VARIABLES ===
# (Docker/Cloud: set via os.environ, else fallback to default)
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
ELASTICSEARCH_URL = os.getenv("ELASTICSEARCH_URL", "http://localhost:9200")
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:5180")

# === MODELS & RESOURCES ===
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")  # Or "medium", "large", etc.
OLLAMA_LLM_MODEL = os.getenv("OLLAMA_LLM_MODEL", "gemma3:4b")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "snowflake-arctic-embed2:latest")
OLLAMA_SCREENSHOT_PROMPT = os.getenv("OLLAMA_SCREENSHOT_PROMPT", """
Stai analizzando uno screenshot.
Qui sotto è riportato il testo estratto tramite OCR:
-----
{ocr_text}
-----

Il tuo compito è:

- Descrivere il layout visivo e i principali elementi presenti nello screenshot, come menu, pulsanti, titoli e qualsiasi altro componente rilevante.
- Indicare chiaramente eventuali aree chiave, barre di navigazione, finestre di dialogo o zone evidenziate se visibili.
- Identificare, se possibile, il contesto o la sezione principale della schermata (ad esempio: dashboard, impostazioni, profilo utente, report, ecc.).
- Scrivere la risposta in italiano.
- Non ripetere il testo OCR grezzo.
- Non fornire spiegazioni, introduzioni o affermazioni che si tratti di un’interfaccia software.
- Fornire solo la descrizione, in modo diretto.
- Non usare formati markdown
""")

OLLAMA_TEXT_PROMPT=os.getenv("OLLAMA_TEXT_PROMPT", """
Questa è la trascrizione di un segmento video.

- Correggi tutti gli errori grammaticali, ortografici, di punteggiatura e di formattazione.
- Se è fornito un contesto, utilizzalo per chiarire eventuali ambiguità.
- Non fornire spiegazioni, non introdurre la risposta e non ripetere la richiesta.
- Restituisci esclusivamente il testo corretto.
- Non usare formati markdown
""")

OLLAMA_TITLE_ABSTRACT_PROMPT = os.getenv("OLLAMA_TITLE_ABSTRACT_PROMPT", """
Dati i seguenti segmenti trascritti di un video tutorial, genera un titolo adatto e un breve abstract che riassuma i principali argomenti e la struttura del contenuto.
Rispondi in italiano. La prima riga deve essere il titolo, seguito dall'abstract nelle righe successive. Non usare formati markdown
Segmenti trascritti:
{fulltext}
""")



# === VIDEO / AUDIO ===
MAX_SEGMENT_DURATION = int(os.getenv("MAX_SEGMENT_DURATION", 420))  # seconds
FRAME_INTERVAL_SEC = int(os.getenv("FRAME_INTERVAL_SEC", 2))        # keyframes

# === OCR ===
TESSERACT_LANGUAGES = os.getenv("TESSERACT_LANGUAGES", "ita+eng")

# === OUTPUT ===
MARKDOWN_DIR = BASE_DIR / "outputs" / "markdown"
TEMP_DIR =  BASE_DIR / "temp"
FRAMES_DIR =  BASE_DIR / "frames"
DOWNLOADS_DIR =  BASE_DIR / "downloads"
PDF_DIR = BASE_DIR / "outputs" / "pdf"
for d in [MARKDOWN_DIR, PDF_DIR,TEMP_DIR,FRAMES_DIR,DOWNLOADS_DIR,PDF_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# === LOGGING ===
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# === Utility functions ===
def get_env_bool(var, default=False):
    val = os.getenv(var)
    if val is None:
        return default
    return val.lower() in ("1", "true", "yes", "on")