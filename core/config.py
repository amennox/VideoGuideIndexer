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
OLLAMA_SCREENSHOT_PROMPT_OLD = os.getenv("OLLAMA_SCREENSHOT_PROMPT", """
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

OLLAMA_SCREENSHOT_PROMPT = os.getenv("OLLAMA_SCREENSHOT_PROMPT", """
Descrivi la schermata software elencando solo gli elementi dell’interfaccia utente (UI) e le loro funzioni, senza riferimenti visivi o commenti generali.
Elenca le funzioni presenti nel menu principale (barra blu in alto), tipicamente contiene le sezioni di navigazione principali del gestionale, ognuna rappresentata da un’etichetta testuale o icona (es. Front-Office, Controllo Accessi, Report, Planning, Istruttore, Pianificazione, Impostazioni).
Elenca le funzioni presenti nel menu secondario (barra grigia sotto il menu principale), tipicamente mostra le opzioni contestuali della sezione selezionata, con etichette testuali e icone (es. Rubrica, Proshop, Servizi, Dashboard CRM, Agenda Utenti, Giftcard/Coupon).

Descrivi dettagliatamente il contenuto del corpo centrale, tipicamente contiene il pannello funzionale della maschera. Cerca di identificare gli elementi più importanti come: elenco utenti, impostazione documento.
Per il corpo centrale elenca tutti i principali elementi di UI trovati e il contenuto del testo dei bottoni presenti e delle caselle di testo cercando di spiegarne l'uso nel contesto della maschera.

Non includere riferimenti al nome del software. Ignora elementi in basso o sulla barra nera.
Non usare il formato markdown, non usare i tag HTML, non usare le virgolette o asterischi *, non aggiungere valutazioni o tue spiegazioni.
La descrizione deve essere oggettiva, strutturata e funzionale all’individuazione degli oggetti UI e delle loro funzioni, senza spiegazioni aggiuntive.
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

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "snowflake-arctic-embed2")
OLLAMA_EMBED_URL = os.getenv("OLLAMA_EMBED_URL", "http://localhost:11434/api/embed")
DOCCLIP_API_URL = os.getenv("DOCCLIP_API_URL", "http://localhost:11436/api/embed")
ELASTICSEARCH_URL = os.getenv("ELASTICSEARCH_URL", "http://localhost:9200")
ELASTICSEARCH_INDEX = os.getenv("ELASTICSEARCH_INDEX", "images")

TEXT_EMBED_DIM=1024
IMAGE_EMBED_DIM=512

DEFAULT_BUSINESSID = os.getenv("DEFAULT_BUSINESSID", "znext_0001")
DEFAULT_ELEMENTID = os.getenv("DEFAULT_ELEMENTID", "1")
DEFAULT_SCOPE = os.getenv("DEFAULT_SCOPE", "video_guide")

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