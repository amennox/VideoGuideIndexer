"""
video_processing.py – Video processing utilities for Video Indexer.
Handles local MP4 files and automatic YouTube download, audio extraction, and keyframe extraction.
All paths are configured via config.py.
"""

import subprocess
import uuid
from pathlib import Path
from typing import List, Tuple, Optional

from core.config import TEMP_DIR, FRAMES_DIR, DOWNLOADS_DIR
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import logging

log = logging.getLogger("indexing.video_processing")


def download_youtube_video(youtube_url: str, output_dir: Path = DOWNLOADS_DIR) -> Path:
    """
    Downloads a YouTube video as MP4 (max 480p) using yt-dlp.
    Returns the path to the downloaded file.
    """
    import yt_dlp
    output_dir.mkdir(parents=True, exist_ok=True)
    video_id = uuid.uuid4().hex[:10]
    outtmpl = str(output_dir / f"{video_id}.%(ext)s")
    ydl_opts = {
        "outtmpl": outtmpl,
        "quiet": True,        
        "format": "bestvideo+bestaudio/best",
        "merge_output_format": "mp4",
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(youtube_url, download=True)
        if not info or "ext" not in info:
            raise RuntimeError("YouTube download failed")
        video_path = output_dir / f"{video_id}.{info['ext']}"
    return video_path

def extract_audio(
    video_path: Path,
    out_wav: Optional[Path] = None,
    start: int = 0,
    duration: Optional[int] = None
) -> Path:
    """
    Extracts audio from the video as a mono 16kHz WAV file using FFmpeg.
    If out_wav is not provided, a file is created in TEMP_DIR.
    Returns the path to the WAV file.
    """
    if out_wav is None:
        out_wav = TEMP_DIR / f"{uuid.uuid4().hex[:10]}.wav"
    cmd = [
        "ffmpeg", "-y", "-ss", str(start), "-i", str(video_path),
        "-ar", "16000", "-ac", "1", "-vn", str(out_wav)
    ]
    if duration:
        cmd[6:6] = ["-t", str(duration)]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    return out_wav

def extract_keyframes(
    video_path: Path,
    output_dir: Optional[Path] = None,
    interval_s: int = 2,
    resize: Optional[Tuple[int, int]] = None
) -> List[Path]:
    """
    Extracts keyframes every 'interval_s' seconds from the video using FFmpeg.
    Optionally resizes frames to (width, height).
    Returns a sorted list of image paths.
    """
    if output_dir is None:
        output_dir = FRAMES_DIR / video_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)
    output_pattern = output_dir / "frame_%05d.jpg"
    vf = f"fps=1/{interval_s}"
    if resize:
        width, height = resize
        vf += f",scale={width}:{height}"
    cmd = [
        "ffmpeg", "-i", str(video_path),
        "-vf", vf,
        str(output_pattern)
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    return sorted(output_dir.glob("frame_*.jpg"))

def extract_diverse_frames(
    video_path: Path,
    output_dir: Optional[Path] = None,
    interval_s: int = 5,
    similarity_threshold: float = 0.85,
    resize: Optional[Tuple[int, int]] = (320, 180)
) -> List[Path]:
    """
    Extracts frames every 'interval_s' seconds from video,
    keeps only frames that are sufficiently different (SSIM below threshold).
    Returns a sorted list of kept image paths.
    """
    import cv2
    from skimage.metrics import structural_similarity as ssim
    import numpy as np

    if output_dir is None:
        output_dir = FRAMES_DIR / video_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps if fps > 0 else 0

    last_frame = None
    kept_frames = []

    sec = 0
    frame_idx = 0

    while sec < duration:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(sec * fps))
        ret, frame = cap.read()
        if not ret:
            break

        # Resize and grayscale for comparison
        frame_small = cv2.resize(frame, resize)
        frame_gray = cv2.cvtColor(frame_small, cv2.COLOR_BGR2GRAY)

        # First frame always saved
        save = False
        if last_frame is None:
            save = True
        else:
            # Compare with previous saved
            score, _ = ssim(last_frame, frame_gray, full=True)
            if score < similarity_threshold:
                save = True

        if save:
            out_path = output_dir / f"frame_{frame_idx:05d}_{int(sec)}.jpg"
            cv2.imwrite(str(out_path), frame)
            kept_frames.append(out_path)
            last_frame = frame_gray
            frame_idx += 1

        sec += interval_s

    cap.release()
    return kept_frames

def get_video_duration(video_path: Path) -> int:
    """
    Returns the duration of the video in seconds (integer).
    """
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(video_path)],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
    )
    duration_str = result.stdout.decode().strip()
    try:
        return int(float(duration_str))
    except Exception:
        return 0

def ensure_mp4(input_path: Path, output_path: Path) -> Path:
    """
    Ensures the video is in MP4 format with compatible codecs for processing.
    Converts if necessary using FFmpeg.
    Returns the output path.
    """
    if input_path.suffix.lower() == ".mp4":
        return input_path
    cmd = [
        "ffmpeg", "-i", str(input_path), "-c:v", "libx264", "-c:a", "aac", "-strict", "experimental", str(output_path)
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    return output_path

def detect_scenes(
    video_path: Path,
    min_gap: int = 5,
    max_len: int = 420,
    ssim_threshold: float = 0.7
) -> List[int]:
    """
    Returns a list of time points (seconds) where there are strong scene changes in the video.
    Always includes 0 and the final duration.
    """
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps) if fps > 0 else 0
    prev, cuts = None, [0]

    for sec in range(min_gap, total, min_gap):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(sec * fps))
        ok, fr = cap.read()
        if not ok:
            break
        if prev is not None:
            g0 = cv2.resize(cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY), (320, 180))
            g1 = cv2.resize(cv2.cvtColor(fr,  cv2.COLOR_BGR2GRAY), (320, 180))
            score, _ = ssim(g0, g1, full=True)
            if score < ssim_threshold or sec - cuts[-1] >= max_len:
                cuts.append(sec)
        prev = fr
    cap.release()

    if total - cuts[-1] > min_gap:
        cuts.append(total)
    return cuts


# Example usage (for testing)
if __name__ == "__main__":
    # Test on a local video or YouTube URL
    video_urls = [
        #"https://www.youtube.com/watch?v=8By43mNELXU",
        #"https://www.youtube.com/watch?v=VBtZoNgJyok",
        #"https://www.youtube.com/watch?v=NgNeossBiwo",
        #"https://www.youtube.com/watch?v=8txXLXPK0oA",
        #"https://www.youtube.com/watch?v=BwycYTuAnGw",
        #"https://www.youtube.com/watch?v=OBZT0nDZJU8",
        #"https://www.youtube.com/watch?v=zGdNsmgYjTQ",
        #"https://www.youtube.com/watch?v=oEKVxgXtiYU",
        #"https://www.youtube.com/watch?v=gibH2Xho3BQ",
        #"https://www.youtube.com/watch?v=xTL1dxa76ro",
        #"https://www.youtube.com/watch?v=a0SJnEf7Gj8",
        "https://www.youtube.com/watch?v=Dlo6AD-0bB0"
        #"https://www.youtube.com/watch?v=klV0kRUmDVM",
        #"https://www.youtube.com/watch?v=2Nqe2oT6HwU",
        #"https://www.youtube.com/watch?v=sk2luddzKOI",
        #"https://www.youtube.com/watch?v=NOxJLGwV9zc",
        #"https://www.youtube.com/watch?v=0sWj_EOpLUA"
        
        # aggiungi altre URL qui
    ]

    # 2. Ciclo su ogni video
    for video_url in video_urls:
        print(f"\nProcessing video: {video_url}")

        # Scarica il video da YouTube
        local_video = download_youtube_video(video_url)
        print("Downloaded video to:", local_video)

        # Estrai audio
        wav_path = extract_audio(local_video)
        print("Extracted audio to:", wav_path)

        # Estrai frame chiave diversi ogni 5 secondi (o come preferisci)
        frames = extract_diverse_frames(local_video, interval_s=2, similarity_threshold=0.95)
        print(f"Extracted {len(frames)} diverse frames.")

