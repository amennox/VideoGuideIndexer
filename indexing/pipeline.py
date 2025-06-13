import json
import subprocess
import re
from pathlib import Path
import logging

log = logging.getLogger("indexing.pipeline")

# Configurazioni chiave (da spostare su config.py)
MIN_SEGMENT_LEN = 180    # sec
MAX_SEGMENT_LEN = 420    # sec
SILENCE_LEN = 5.0        # sec
SILENCE_THRESH = -35.0   # dB
SCENE_SSIM = 0.7         # threshold per cambio immagine
FRAME_INTERVAL = 5       # secondi tra frame
FRAME_SIM_THRESHOLD = 0.85  # SSIM soglia per frame diversi
MAX_CHUNK_CHARS = 3000   # limite per testo per chunk (adatta secondo embedding)

def detect_silence(audio_path, silence_len=SILENCE_LEN, silence_thresh=SILENCE_THRESH):
    cmd = [
        "ffmpeg", "-i", str(audio_path),
        "-af", f"silencedetect=noise={silence_thresh}dB:d={silence_len}",
        "-f", "null", "-"
    ]
    p = subprocess.run(cmd, stderr=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    output = p.stderr
    starts, ends = [], []
    for line in output.splitlines():
        if "silence_start:" in line:
            starts.append(float(line.split("silence_start:")[1].strip()))
        elif "silence_end:" in line:
            m = re.search(r"silence_end: ([\d\.]+)", line)
            if m:
                ends.append(float(m.group(1)))
    silences = list(zip(starts, ends))
    # Restituisce solo i tempi di fine silenzio (candidati come possibili cut)
    return [int(end) for _, end in silences if end is not None]

def segment_video(
    video_path: Path,
    min_len: int = MIN_SEGMENT_LEN,
    max_len: int = MAX_SEGMENT_LEN,
    silence_len: float = SILENCE_LEN,
    silence_thresh: float = SILENCE_THRESH,
    scene_ssim: float = SCENE_SSIM
):
    from indexing.video_processing import extract_audio,detect_scenes, get_video_duration
    
    # 1. Estrai l'audio per la detection dei silenzi
    audio_path = extract_audio(video_path, out_wav=None, start=0, duration=None)
    silence_cuts = detect_silence(audio_path, silence_len, silence_thresh)
    print("Silenzi trovati (fine):", silence_cuts)

    # 2. Detect scene change forti
    scene_cuts = detect_scenes(video_path, min_gap=min_len, max_len=max_len, ssim_threshold=scene_ssim)
    print("Scene cuts trovati:", scene_cuts)

    # 3. Unisci e ordina tutti i tagli
    cuts = sorted(set([0] + silence_cuts + scene_cuts))
    print("Cut points preliminari:", cuts)

    # 4. Forza min_len e max_len (no tagli troppo ravvicinati o troppo lontani)
    final_cuts = [0]
    for t in cuts[1:]:
        if t - final_cuts[-1] < min_len:
            continue
        if t - final_cuts[-1] > max_len:
            # split in step di max_len
            while final_cuts[-1] + max_len < t:
                final_cuts.append(final_cuts[-1] + max_len)
        final_cuts.append(t)
    # Finale
    duration = get_video_duration(video_path)
    if final_cuts[-1] != duration:
        final_cuts.append(duration)
    segments = list(zip(final_cuts[:-1], final_cuts[1:]))
    print("Segmenti finali:", segments)
    return segments

def split_text_by_length(text, max_chars):
    """Splitta il testo su punto fermo per stare sotto max_chars."""
    sentences = re.split(r'(?<=[.!?]) +', text)
    chunks = []
    curr = ""
    for sent in sentences:
        if len(curr) + len(sent) > max_chars:
            if curr:
                chunks.append(curr.strip())
            curr = sent
        else:
            curr += " " + sent
    if curr.strip():
        chunks.append(curr.strip())
    return chunks

def index_video(
    video_path: Path,
    output_json: Path,
    min_len: int = MIN_SEGMENT_LEN,
    max_len: int = MAX_SEGMENT_LEN,
    frame_interval: int = FRAME_INTERVAL,
    frame_sim_threshold: float = FRAME_SIM_THRESHOLD,
    max_chunk_chars: int = MAX_CHUNK_CHARS
):
    from indexing.video_processing import extract_audio, extract_diverse_frames
    from indexing.transcription import transcribe_audio
    from indexing.ocr import extract_text_from_image, describe_screen_with_ollama
    from utils.ollama import correct_text_with_ollama

    # 1. Segmentazione avanzata
    segment_intervals = segment_video(
        video_path, min_len, max_len
    )

    all_segments = []
    
   
    
    for start, end in segment_intervals:
        log.info(f"Processing segment {start}-{end}s...")
        
        log.info(f"Extracting audio from {video_path} (start={start}, duration={end})")
        
        audio_path = extract_audio(video_path, start=start, duration=end-start)
        
        log.info(f"Transcribe audio from {video_path} (start={start}, duration={end})")
        
        stt_out = transcribe_audio(audio_path)
        transcript = stt_out["text"]

        log.info(f"Send audio to ollama {video_path} (start={start}, duration={end})")
        transcript_corrected = correct_text_with_ollama(transcript)

        # --- Split chunk per lunghezza massima ---
        chunks = split_text_by_length(transcript_corrected, max_chunk_chars)

        log.info(f"Frame extraction {video_path} (start={start}, duration={end})")
        # --- Frame extraction  ---
        segment_frames = extract_diverse_frames(
            video_path,
            interval_s=frame_interval,
            similarity_threshold=frame_sim_threshold
        )
        segment_frames = [
            f for f in segment_frames
            if start <= extract_frame_time_from_filename(f.name) < end
        ]

        screenshots = []
        for frame_path in segment_frames:
            #log.info(f"Frame OCR {video_path} (start={start}, duration={end})")
            #ocr_text = extract_text_from_image(frame_path)
            #log.info(f"Frame describe by Ollama {video_path} (start={start}, duration={end})")
            #description = describe_screen_with_ollama(frame_path, ocr_text)
            description="snapshot"
            screenshots.append({
                "image": frame_path.name,
                "description": description
            })

        for chunk in chunks:
            all_segments.append({
                "interval": { "start": int(start), "end": int(end) },
                "fulltext": chunk,
                "screenshots": screenshots
            })

    # --- Salva JSON ---
    output_dict = {
        "video": video_path.name,  
        "segments": all_segments
    }
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(output_dict, f, indent=2, ensure_ascii=False)
    log.info(f"Indexing complete, output: {output_json}")

def extract_frame_time_from_filename(fname: str) -> int:
    """
    Expects frame files named like 'frame_00012_60.jpg'
    Returns 60 (secondi) se presente, altrimenti 0.
    """
    m = re.search(r"_(\d+)\.jpg$", fname)
    return int(m.group(1)) if m else 0

if __name__ == "__main__":
    video_path = Path(r"C:\Progetti\VideoGuide\downloads\CREAZIONE_MODELLO_DOCUMENTO.mp4")
    output_json = Path("indicizzazione_output_CREAZIONE_MODELLO_DOCUMENTO.json")
    index_video(video_path, output_json)
