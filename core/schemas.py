"""
schemas.py – Pydantic models for API request/response and core domain objects.
"""

from typing import List, Optional, Any
from pydantic import BaseModel, Field

# === Base Models ===

class VideoMeta(BaseModel):
    video_id: str
    title: str
    description: Optional[str] = ""
    duration: Optional[int] = None  # seconds

class FrameInfo(BaseModel):
    timestamp: int  # seconds
    image_path: str
    ocr_text: Optional[str] = ""

class Segment(BaseModel):
    start: int  # seconds
    end: int
    text: str
    img_descr: Optional[str] = ""

class Chunk(BaseModel):
    id_chunk: str
    id_video: str
    processed_text: str
    embedding: List[float]
    start: int
    end: int
    frame_paths: List[str]

class OCRResult(BaseModel):
    timestamp: int
    text: str
    image_path: Optional[str] = None

class EmbeddingResult(BaseModel):
    embedding: List[float]

# === API Schemas ===

class RecognizeScreenRequest(BaseModel):
    image_b64: str  # Base64-encoded screenshot
    # Optional: metadata (timestamp, etc.)

class RecognizeScreenResponse(BaseModel):
    context_id: str  # e.g., video_id+timestamp or chunk id
    ocr_text: str

class ChatRAGRequest(BaseModel):
    user_question: str
    context_id: Optional[str] = None
    ocr_text: Optional[str] = None

class ChatRAGResponse(BaseModel):
    generated_answer: str
    sources: Optional[List[Any]] = None  # Can include IDs, texts, etc.

# === Video Upload API ===

class UploadVideoRequest(BaseModel):
    file_name: str

class UploadVideoResponse(BaseModel):
    video_id: str
    title: str
    description: Optional[str] = ""

# === Error Model (Optional) ===
class APIError(BaseModel):
    detail: str

# === Example scope for dropdowns ===
class Scope(BaseModel):
    id: str
    name: str

# List responses
class ScopesResponse(BaseModel):
    scopes: List[Scope]

# More can be added as needed!
