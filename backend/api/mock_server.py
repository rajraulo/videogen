"""
VideoGen Mock Server — Full API with fake video generation (no GPU / no ML needed).
Generates animated colour-gradient videos using numpy so the Android app can be
tested end-to-end without any GPU or large model download.

Start with:
    uvicorn api.mock_server:app --host 0.0.0.0 --port 8000 --reload
"""

import os
import uuid
import asyncio
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime

import numpy as np
import imageio_ffmpeg          # explicit ffmpeg writer — guarantees yuv420p H.264
from fastapi import (
    FastAPI, HTTPException, BackgroundTasks,
    Header, Depends, File, Form, UploadFile,
)
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ─────────────────────────────────────────────
# App setup
# ─────────────────────────────────────────────

app = FastAPI(
    title="VideoGen Mock API",
    description="Test server — no GPU required. Generates animated gradient videos.",
    version="1.0.0-mock",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

jobs: Dict[str, Dict[str, Any]] = {}
API_KEY   = os.getenv("VIDEOGEN_API_KEY", "dev-secret-key")
OUTPUT_DIR = Path("outputs/mock_videos")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────
# Auth
# ─────────────────────────────────────────────

def verify_api_key(x_api_key: str = Header(...)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return x_api_key


# ─────────────────────────────────────────────
# Schemas
# ─────────────────────────────────────────────

class GenerateRequest(BaseModel):
    prompt:               str           = Field(..., min_length=3, max_length=1000)
    negative_prompt:      Optional[str] = None
    duration_seconds:     Optional[float] = Field(default=4.0, ge=1.0, le=16.0)
    fps:                  Optional[int]   = Field(default=12, ge=8, le=24)
    height:               Optional[int]   = Field(default=480)
    width:                Optional[int]   = Field(default=720)
    num_inference_steps:  Optional[int]   = Field(default=50)
    guidance_scale:       Optional[float] = Field(default=6.0)
    seed:                 Optional[int]   = None
    enhance_prompt:       bool = True


class JobStatus(BaseModel):
    job_id:           str
    status:           str
    created_at:       str
    updated_at:       str
    progress:         int            = 0
    video_url:        Optional[str]  = None
    error:            Optional[str]  = None
    metadata:         Optional[dict] = None
    transcribed_text: Optional[str]  = None


# ─────────────────────────────────────────────
# Video generator (numpy only — no GPU)
# ─────────────────────────────────────────────

def _generate_gradient_video(
    job_id: str,
    prompt: str,
    duration: float,
    fps: int,
    height: int,
    width: int,
    seed: Optional[int] = None,
) -> str:
    """
    Create an animated colour-gradient MP4 encoded as H.264 yuv420p.
    yuv420p is required for Android hardware decoder (ExoPlayer) compatibility.
    Different prompts produce different colour palettes via prompt hash.
    """
    # H.264 requires even dimensions
    height = height - (height % 2)
    width  = width  - (width  % 2)

    rng   = np.random.default_rng(seed if seed is not None else abs(hash(prompt)) % (2 ** 32))
    hue_r = rng.uniform(0, 2 * np.pi)
    hue_g = rng.uniform(0, 2 * np.pi)
    hue_b = rng.uniform(0, 2 * np.pi)

    n    = int(duration * fps)
    x_v  = np.linspace(0, 1, width,  dtype=np.float32)
    y_v  = np.linspace(0, 1, height, dtype=np.float32)
    X, Y = np.meshgrid(x_v, y_v)          # (H, W) each

    out_path = str(OUTPUT_DIR / f"{job_id}.mp4")

    # imageio_ffmpeg writes raw RGB24 frames; we tell ffmpeg to encode as yuv420p H.264
    writer = imageio_ffmpeg.write_frames(
        out_path,
        size         = (width, height),
        fps          = fps,
        codec        = "libx264",
        pix_fmt_in   = "rgb24",
        pix_fmt_out  = "yuv420p",          # ← required for Android ExoPlayer
        ffmpeg_log_level = "quiet",
        output_params = ["-preset", "fast", "-crf", "23", "-movflags", "+faststart"],
    )
    writer.send(None)   # initialise the generator

    for i in range(n):
        t = i / max(n - 1, 1)
        r = (np.sin(2 * np.pi * (t * 1.2 + X * 0.6 + hue_r)) * 127 + 128).astype(np.uint8)
        g = (np.sin(2 * np.pi * (t * 0.9 + Y * 0.6 + hue_g)) * 127 + 128).astype(np.uint8)
        b = (np.sin(2 * np.pi * (t * 0.7 + (X + Y) * 0.4 + hue_b)) * 127 + 128).astype(np.uint8)
        frame = np.stack([r, g, b], axis=-1)   # (H, W, 3) uint8 RGB
        writer.send(frame.tobytes())

    writer.close()
    print(f"[Mock] {out_path}  ({n} frames, {width}×{height}, yuv420p H.264)")
    return out_path


# ─────────────────────────────────────────────
# Background task
# ─────────────────────────────────────────────

def _run_generation(job_id: str, req: GenerateRequest):
    import time
    jobs[job_id]["status"]     = "processing"
    jobs[job_id]["updated_at"] = datetime.utcnow().isoformat()

    try:
        # Simulate processing time proportional to duration
        steps = max(2, int(req.duration_seconds))
        for i in range(steps):
            time.sleep(1)
            jobs[job_id]["progress"] = int((i + 1) / steps * 90)

        video_path = _generate_gradient_video(
            job_id   = job_id,
            prompt   = req.prompt,
            duration = req.duration_seconds,
            fps      = req.fps,
            height   = req.height,
            width    = req.width,
            seed     = req.seed,
        )

        jobs[job_id].update({
            "status":     "completed",
            "progress":   100,
            "video_path": video_path,
            "video_url":  f"/videos/{job_id}",
            "updated_at": datetime.utcnow().isoformat(),
            "metadata":   {"prompt": req.prompt, "mode": "mock"},
        })

    except Exception as e:
        jobs[job_id].update({
            "status":     "failed",
            "error":      str(e),
            "updated_at": datetime.utcnow().isoformat(),
        })
        print(f"[Mock] Job {job_id} failed: {e}")


# ─────────────────────────────────────────────
# Endpoints  (identical interface to server.py)
# ─────────────────────────────────────────────

@app.get("/health")
async def health():
    return {
        "status":      "ok",
        "mode":        "mock — no GPU required",
        "model":       "gradient-generator",
        "active_jobs": sum(1 for j in jobs.values() if j["status"] == "processing"),
        "total_jobs":  len(jobs),
    }


@app.post("/generate", response_model=JobStatus)
async def generate_video(
    req: GenerateRequest,
    background_tasks: BackgroundTasks,
    _: str = Depends(verify_api_key),
):
    processing = [j for j in jobs.values() if j["status"] == "processing"]
    if len(processing) >= 1:
        raise HTTPException(status_code=429, detail="A job is already processing.")

    job_id = str(uuid.uuid4())
    now    = datetime.utcnow().isoformat()

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


@app.post("/generate-from-audio", response_model=JobStatus)
async def generate_from_audio(
    background_tasks: BackgroundTasks,
    speech_file: UploadFile       = File(...),
    music_file:  Optional[UploadFile] = File(default=None),
    duration_seconds: float = Form(default=4.0),
    language:         str   = Form(default="auto"),
    seed:             Optional[int] = Form(default=None),
    _: str = Depends(verify_api_key),
):
    # Mock transcription — echo filename as "transcript"
    transcript = f"[Mock transcript of {speech_file.filename}]"

    # Consume uploaded bytes so the connection doesn't stall
    await speech_file.read()
    if music_file:
        await music_file.read()

    req = GenerateRequest(
        prompt           = transcript,
        duration_seconds = min(max(duration_seconds, 1.0), 16.0),
        seed             = seed,
    )

    job_id = str(uuid.uuid4())
    now    = datetime.utcnow().isoformat()

    jobs[job_id] = {
        "job_id":           job_id,
        "status":           "queued",
        "created_at":       now,
        "updated_at":       now,
        "progress":         0,
        "request":          req.dict(),
        "transcribed_text": transcript,
    }

    background_tasks.add_task(_run_generation, job_id, req)
    return JobStatus(**jobs[job_id])


@app.get("/jobs/{job_id}", response_model=JobStatus)
async def get_job(job_id: str, _: str = Depends(verify_api_key)):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return JobStatus(**jobs[job_id])


@app.get("/videos/{job_id}")
async def download_video(job_id: str):   # no auth — ExoPlayer can't send headers
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    job = jobs[job_id]
    if job["status"] != "completed":
        raise HTTPException(status_code=202, detail=f"Job status: {job['status']}")
    video_path = job.get("video_path")
    if not video_path or not Path(video_path).exists():
        raise HTTPException(status_code=404, detail="Video file not found")
    return FileResponse(video_path, media_type="video/mp4",
                        filename=f"videogen_{job_id}.mp4")


@app.get("/jobs")
async def list_jobs(limit: int = 20, _: str = Depends(verify_api_key)):
    sorted_jobs = sorted(jobs.values(), key=lambda j: j["created_at"], reverse=True)
    return {"jobs": sorted_jobs[:limit], "total": len(jobs)}


@app.delete("/jobs/{job_id}")
async def delete_job(job_id: str, _: str = Depends(verify_api_key)):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    job = jobs.pop(job_id)
    video_path = job.get("video_path")
    if video_path and Path(video_path).exists():
        Path(video_path).unlink()
    return {"deleted": job_id}
