"""
VideoGen API Server
FastAPI + Celery async job queue — Production-ready inference server

Start with:
    uvicorn api.server:app --host 0.0.0.0 --port 8000 --workers 1
"""

import os
import shutil
import tempfile
import uuid
import time
import json
import asyncio
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime

from fastapi import FastAPI, HTTPException, BackgroundTasks, Header, Depends, File, Form, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator

# Import our model pipeline
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from model.video_gen_pipeline import VideoGenModel, VideoGenConfig

# Audio processing (optional — install openai-whisper and librosa to enable)
try:
    from audio.audio_processor import (
        transcribe_speech, analyze_audio_mood, build_enriched_prompt, LANGUAGE_CODES,
    )
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False


# ─────────────────────────────────────────────
# App & Config
# ─────────────────────────────────────────────

app = FastAPI(
    title="VideoGen API",
    description="Text-to-video generation API (Veo-style)",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory job store (replace with Redis in production)
jobs: Dict[str, Dict[str, Any]] = {}

# Single model instance (load once)
MODEL: Optional[VideoGenModel] = None
API_KEY = os.getenv("VIDEOGEN_API_KEY", "dev-secret-key")


# ─────────────────────────────────────────────
# Auth
# ─────────────────────────────────────────────

def verify_api_key(x_api_key: str = Header(...)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return x_api_key


# ─────────────────────────────────────────────
# Request / Response schemas
# ─────────────────────────────────────────────

class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=3, max_length=1000)
    negative_prompt: Optional[str] = "blurry, low quality, distorted, watermark"
    duration_seconds: Optional[float] = Field(default=4.0, ge=1.0, le=16.0)
    fps: Optional[int] = Field(default=12, ge=8, le=24)
    height: Optional[int] = Field(default=480)
    width: Optional[int] = Field(default=720)
    num_inference_steps: Optional[int] = Field(default=50, ge=10, le=100)
    guidance_scale: Optional[float] = Field(default=6.0, ge=1.0, le=20.0)
    seed: Optional[int] = None
    enhance_prompt: bool = True

    @validator("duration_seconds")
    def cap_duration(cls, v):
        return min(v, 16.0)


class JobStatus(BaseModel):
    job_id: str
    status: str           # queued | processing | completed | failed
    created_at: str
    updated_at: str
    progress: int = 0     # 0–100
    video_url: Optional[str] = None
    error: Optional[str] = None
    metadata: Optional[dict] = None
    transcribed_text: Optional[str] = None   # set for audio-sourced jobs


# ─────────────────────────────────────────────
# Lifecycle
# ─────────────────────────────────────────────

@app.on_event("startup")
async def startup():
    global MODEL
    print("[Server] Loading video generation model...")
    cfg = VideoGenConfig(enable_cpu_offload=True)
    MODEL = VideoGenModel(cfg)
    MODEL.load()
    print("[Server] Model ready. API accepting requests.")


@app.on_event("shutdown")
async def shutdown():
    global MODEL
    if MODEL:
        MODEL.unload()
    print("[Server] Shutdown complete.")


# ─────────────────────────────────────────────
# Background generation task
# ─────────────────────────────────────────────

def _run_generation(job_id: str, req: GenerateRequest):
    """Runs in a background thread (not async — PyTorch blocks)."""
    global MODEL, jobs

    try:
        jobs[job_id]["status"] = "processing"
        jobs[job_id]["updated_at"] = datetime.utcnow().isoformat()

        # Enhance prompt if requested
        prompt = req.prompt
        if req.enhance_prompt and MODEL:
            prompt = MODEL.enhance_prompt(prompt)

        num_frames = int(req.duration_seconds * req.fps)

        result = MODEL.generate(
            prompt=prompt,
            negative_prompt=req.negative_prompt,
            num_frames=num_frames,
            num_inference_steps=req.num_inference_steps,
            guidance_scale=req.guidance_scale,
            seed=req.seed,
        )

        jobs[job_id].update({
            "status": "completed",
            "progress": 100,
            "video_path": result["video_path"],
            "video_url": f"/videos/{job_id}",
            "updated_at": datetime.utcnow().isoformat(),
            "metadata": result["metadata"],
        })

    except Exception as e:
        jobs[job_id].update({
            "status": "failed",
            "error": str(e),
            "updated_at": datetime.utcnow().isoformat(),
        })
        print(f"[Server] Job {job_id} failed: {e}")


# ─────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model_loaded": MODEL is not None,
        "active_jobs": sum(1 for j in jobs.values() if j["status"] == "processing"),
        "total_jobs": len(jobs),
    }


@app.post("/generate", response_model=JobStatus)
async def generate_video(
    req: GenerateRequest,
    background_tasks: BackgroundTasks,
    _: str = Depends(verify_api_key),
):
    """
    Submit a text-to-video generation job.
    Returns a job_id to poll for status.
    """
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Block concurrent jobs on single-GPU setup
    processing = [j for j in jobs.values() if j["status"] == "processing"]
    if len(processing) >= 1:
        raise HTTPException(
            status_code=429,
            detail="A job is already processing. Try again shortly."
        )

    job_id = str(uuid.uuid4())
    now = datetime.utcnow().isoformat()

    jobs[job_id] = {
        "job_id":     job_id,
        "status":     "queued",
        "created_at": now,
        "updated_at": now,
        "progress":   0,
        "request":    req.dict(),
    }

    background_tasks.add_task(_run_generation, job_id, req)

    return JobStatus(**jobs[job_id])


@app.get("/jobs/{job_id}", response_model=JobStatus)
async def get_job(
    job_id: str,
    _: str = Depends(verify_api_key),
):
    """Poll this endpoint until status == 'completed'."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return JobStatus(**jobs[job_id])


@app.get("/videos/{job_id}")
async def download_video(
    job_id: str,
    # No API key required — ExoPlayer/browser can't send custom headers for video streams.
    # UUIDs are unguessable so this is safe.
):
    """Download the generated MP4 file."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = jobs[job_id]
    if job["status"] != "completed":
        raise HTTPException(status_code=202, detail=f"Job status: {job['status']}")

    video_path = job.get("video_path")
    if not video_path or not Path(video_path).exists():
        raise HTTPException(status_code=404, detail="Video file not found")

    return FileResponse(
        path=video_path,
        media_type="video/mp4",
        filename=f"videogen_{job_id}.mp4",
    )


@app.get("/jobs")
async def list_jobs(
    limit: int = 20,
    _: str = Depends(verify_api_key),
):
    """List recent jobs, newest first."""
    sorted_jobs = sorted(
        jobs.values(),
        key=lambda j: j["created_at"],
        reverse=True
    )
    return {"jobs": sorted_jobs[:limit], "total": len(jobs)}


@app.post("/generate-from-audio", response_model=JobStatus)
async def generate_from_audio(
    background_tasks: BackgroundTasks,
    speech_file: UploadFile = File(..., description="Voice recording (m4a/mp3/wav) — Odia, Telugu, or any language"),
    music_file:  Optional[UploadFile] = File(default=None, description="Background music for mood analysis (optional)"),
    duration_seconds: float  = Form(default=4.0),
    language:         str    = Form(default="auto", description="'auto', 'odia', 'telugu', 'hindi', 'english'"),
    seed:             Optional[int] = Form(default=None),
    _: str = Depends(verify_api_key),
):
    """
    Submit a video generation job from audio input.

    1. Transcribes speech with Whisper (supports Odia 'or', Telugu 'te', auto-detect).
    2. Optionally analyzes music file for mood/style tags.
    3. Combines transcript + mood into a prompt and queues video generation.
    """
    if not AUDIO_AVAILABLE:
        raise HTTPException(
            status_code=501,
            detail="Audio processing unavailable. Install openai-whisper and librosa.",
        )
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    processing = [j for j in jobs.values() if j["status"] == "processing"]
    if len(processing) >= 1:
        raise HTTPException(status_code=429, detail="A job is already processing. Try again shortly.")

    whisper_lang = LANGUAGE_CODES.get(language.lower())  # None → auto-detect

    tmp_dir = tempfile.mkdtemp()
    try:
        # Save speech file
        s_suffix = Path(speech_file.filename or "audio.m4a").suffix or ".m4a"
        speech_path = os.path.join(tmp_dir, f"speech{s_suffix}")
        with open(speech_path, "wb") as f:
            f.write(await speech_file.read())

        # Save music file (optional)
        music_path = None
        if music_file and music_file.filename:
            m_suffix = Path(music_file.filename).suffix or ".mp3"
            music_path = os.path.join(tmp_dir, f"music{m_suffix}")
            with open(music_path, "wb") as f:
                content = await music_file.read()
                if content:
                    f.write(content)
                else:
                    music_path = None

        # Transcribe speech in thread so we don't block the event loop
        transcript = await asyncio.to_thread(transcribe_speech, speech_path, whisper_lang)
        if not transcript:
            raise HTTPException(status_code=422, detail="Could not transcribe audio — please speak clearly.")

        # Analyze music mood (best-effort, non-fatal)
        music_style = None
        if music_path and os.path.getsize(music_path) > 0:
            try:
                _, music_style = await asyncio.to_thread(analyze_audio_mood, music_path)
            except Exception as e:
                print(f"[Server] Music analysis skipped: {e}")

        prompt = build_enriched_prompt(transcript, music_style)

    except HTTPException:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise
    except Exception as e:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=f"Audio processing error: {e}")

    job_id = str(uuid.uuid4())
    now    = datetime.utcnow().isoformat()
    req    = GenerateRequest(
        prompt           = prompt,
        duration_seconds = min(max(duration_seconds, 1.0), 16.0),
        seed             = seed,
    )

    jobs[job_id] = {
        "job_id":           job_id,
        "status":           "queued",
        "created_at":       now,
        "updated_at":       now,
        "progress":         0,
        "request":          req.dict(),
        "transcribed_text": transcript,
    }

    def _generate_and_cleanup(jid: str, request: GenerateRequest, tmp: str):
        _run_generation(jid, request)
        shutil.rmtree(tmp, ignore_errors=True)

    background_tasks.add_task(_generate_and_cleanup, job_id, req, tmp_dir)
    return JobStatus(**jobs[job_id])


@app.delete("/jobs/{job_id}")
async def delete_job(
    job_id: str,
    _: str = Depends(verify_api_key),
):
    """Delete job record and associated video file."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = jobs.pop(job_id)
    video_path = job.get("video_path")
    if video_path and Path(video_path).exists():
        Path(video_path).unlink()

    return {"deleted": job_id}
