"""DeepGuard API — Asynchronous Dual-Pipeline (Visual & Audio) Deepfake Detection Service."""

import asyncio
import os
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

import torch
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from services.detector import DeepfakeDetector, DetectorError
from services.audio_detector import AudioDeepfakeDetector, AudioDetectorError
from services.video_processor import (
    SUPPORTED_EXTENSIONS,
    SUPPORTED_IMAGE_EXTENSIONS,
    VideoProcessingError,
    extract_frames,
    process_image,
)

SUPPORTED_AUDIO_EXTENSIONS = {'.wav', '.mp3', '.m4a', '.flac', '.ogg'}
UPLOAD_DIR = Path("uploads")
UPLOAD_CHUNK_SIZE = 1024 * 1024

video_detector: DeepfakeDetector | None = None
audio_detector: AudioDeepfakeDetector | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global video_detector, audio_detector
    print("Loading DeepGuard AI Visual Model (ViT-Base)...")
    video_detector = DeepfakeDetector()
    print("Loading DeepGuard AI Audio Model (Vocoder Forensics)...")
    audio_detector = AudioDeepfakeDetector()
    UPLOAD_DIR.mkdir(exist_ok=True)
    yield
    video_detector = None
    audio_detector = None


app = FastAPI(title="DeepGuard Multimodal API", version="0.3.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class FrameDetail(BaseModel):
    frame_number: int
    real_prob: float
    fake_prob: float
    classification: str
    confidence: float
    has_face: bool = True
    timestamp: float = 0.0


class VotingSummary(BaseModel):
    fake_frames: int
    real_frames: int
    total_frames: int
    verdict: str | None = None


class DetectionResponse(BaseModel):
    filename: str
    verdict: str
    confidence: float
    voting_summary: VotingSummary
    frame_details: list[FrameDetail]
    video_metadata: dict
    forensics_summary: dict | None = None
    model_info: dict | None = None


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "DeepGuard Multimodal",
        "video_model_loaded": video_detector is not None,
        "audio_model_loaded": audio_detector is not None,
        "cuda_available": torch.cuda.is_available()
    }


@app.post("/api/v1/detect", response_model=DetectionResponse)
async def detect_video(file: UploadFile = File(...)):
    if video_detector is None:
        raise HTTPException(status_code=503, detail="Visual model is still initializing.")

    ext = Path(file.filename or "").suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Unsupported format '{ext}'.")

    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(dir=UPLOAD_DIR, suffix=ext, delete=False) as buf:
            temp_path = buf.name
            while chunk := await file.read(UPLOAD_CHUNK_SIZE):
                buf.write(chunk)

        if ext in SUPPORTED_IMAGE_EXTENSIONS:
            frames, metadata = await asyncio.to_thread(process_image, temp_path)
        else:
            frames, metadata = await asyncio.to_thread(extract_frames, temp_path)

        result = await asyncio.to_thread(video_detector.analyze_frames, frames)

        return DetectionResponse(
            filename=file.filename,
            verdict=result["verdict"],
            confidence=result["confidence"],
            voting_summary=VotingSummary(**result["voting_summary"]),
            frame_details=[FrameDetail(**f) for f in result["frame_details"]],
            video_metadata=metadata,
            forensics_summary=result.get("forensics_summary"),
            model_info=result.get("model_info"),
        )
    except (VideoProcessingError, DetectorError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass


@app.post("/api/v1/detect-audio")
async def detect_audio(file: UploadFile = File(...)):
    if audio_detector is None:
        raise HTTPException(status_code=503, detail="Audio model is still initializing.")

    ext = Path(file.filename or "").suffix.lower()
    if ext not in SUPPORTED_AUDIO_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Unsupported audio format '{ext}'. Supported: {sorted(SUPPORTED_AUDIO_EXTENSIONS)}")

    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(dir=UPLOAD_DIR, suffix=ext, delete=False) as buf:
            temp_path = buf.name
            while chunk := await file.read(UPLOAD_CHUNK_SIZE):
                buf.write(chunk)

        result = await asyncio.to_thread(audio_detector.analyze_audio, temp_path)
        result["filename"] = file.filename
        return result
    except AudioDetectorError as e:
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass


@app.get("/demo")
def demo():
    return FileResponse(Path(__file__).parent / "static" / "demo.html")


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)