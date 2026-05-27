"""
VideoGen Image Server — Stable Diffusion image generation on Apple Silicon MPS.
Generates a real AI image then animates it with a Ken Burns zoom effect.
No CUDA needed. Works on M4 Mac with ~30 sec per video.

Start with:
    uvicorn api.image_server:app --host 0.0.0.0 --port 8000 --reload
"""

import os, uuid, shutil, tempfile, threading, time, gc
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime

import numpy as np
import imageio_ffmpeg
from PIL import Image
from fastapi import FastAPI, HTTPException, BackgroundTasks, Header, Depends, File, Form, UploadFile
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ─────────────────────────────────────────────
# App setup
# ─────────────────────────────────────────────

app = FastAPI(title="VideoGen Image Server (SD + Ken Burns)")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

jobs: Dict[str, Dict[str, Any]] = {}
API_KEY    = os.getenv("VIDEOGEN_API_KEY", "dev-secret-key")
OUTPUT_DIR = Path("outputs/image_videos")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

pipe = None   # loaded on startup


def verify_api_key(x_api_key: str = Header(...)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return x_api_key


# ─────────────────────────────────────────────
# Schemas
# ─────────────────────────────────────────────

class GenerateRequest(BaseModel):
    prompt:              str             = Field(..., min_length=3, max_length=1000)
    negative_prompt:     Optional[str]   = "blurry, distorted, watermark, low quality, nsfw"
    duration_seconds:    Optional[float] = Field(default=3.0, ge=1.0, le=10.0)
    fps:                 Optional[int]   = Field(default=8, ge=8, le=24)
    height:              Optional[int]   = Field(default=512)
    width:               Optional[int]   = Field(default=512)
    num_inference_steps: Optional[int]   = Field(default=25, ge=10, le=50)
    guidance_scale:      Optional[float] = Field(default=7.5)
    seed:                Optional[int]   = None
    enhance_prompt:      bool            = True


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
# Model load
# ─────────────────────────────────────────────

@app.on_event("startup")
async def startup():
    global pipe
    import torch
    from diffusers import StableDiffusionPipeline

    model_id = os.getenv("VIDEOGEN_MODEL", "Lykon/dreamshaper-8")
    hf_token = os.getenv("HF_TOKEN")
    device   = "mps" if torch.backends.mps.is_available() else "cpu"
    dtype    = torch.float16 if device == "mps" else torch.float32

    # Login via huggingface_hub — works across all diffusers versions
    if hf_token:
        from huggingface_hub import login
        login(token=hf_token, add_to_git_credential=False)

    print(f"[ImageServer] Loading {model_id} on {device} ({dtype}) ...")
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=dtype,
    ).to(device)

    pipe.safety_checker = None   # disable for speed
    pipe.enable_attention_slicing()
    gc.collect()
    print("[ImageServer] Model ready.")


@app.on_event("shutdown")
async def shutdown():
    global pipe
    del pipe
    gc.collect()


# ─────────────────────────────────────────────
# Ken Burns animation
# ─────────────────────────────────────────────

def _image_to_video(image: Image.Image, out_path: str, fps: int,
                    duration: float, width: int, height: int):
    """Animate a PIL image with a slow Ken Burns zoom-in effect."""
    n_frames = max(8, int(duration * fps))
    w, h = width - (width % 2), height - (height % 2)
    img  = image.resize((int(w * 1.15), int(h * 1.15)), Image.LANCZOS)  # slightly larger for zoom

    writer = imageio_ffmpeg.write_frames(
        out_path, size=(w, h), fps=fps,
        codec="libx264", pix_fmt_in="rgb24", pix_fmt_out="yuv420p",
        ffmpeg_log_level="quiet",
        output_params=["-preset", "fast", "-crf", "20", "-movflags", "+faststart"],
    )
    writer.send(None)

    iw, ih = img.size
    for i in range(n_frames):
        t       = i / max(n_frames - 1, 1)           # 0.0 → 1.0
        scale   = 1.0 - t * 0.12                      # zoom from 100% to 88%
        cw, ch  = int(iw * scale), int(ih * scale)
        left    = (iw - cw) // 2
        top     = (ih - ch) // 2
        frame   = img.crop((left, top, left + cw, top + ch)).resize((w, h), Image.LANCZOS)
        arr     = np.array(frame.convert("RGB"), dtype=np.uint8)
        writer.send(arr.tobytes())

    writer.close()


# ─────────────────────────────────────────────
# Background generation
# ─────────────────────────────────────────────

def _run_generation(job_id: str, req: GenerateRequest):
    import torch
    jobs[job_id]["status"]     = "processing"
    jobs[job_id]["updated_at"] = datetime.utcnow().isoformat()
    try:
        if pipe is None:
            raise RuntimeError("Model not loaded.")

        generator = None
        if req.seed is not None:
            device = "mps" if torch.backends.mps.is_available() else "cpu"
            generator = torch.Generator(device=device).manual_seed(req.seed)

        print(f"[ImageServer] Generating: '{req.prompt[:60]}'")
        result = pipe(
            prompt=req.prompt,
            negative_prompt=req.negative_prompt,
            height=req.height,
            width=req.width,
            num_inference_steps=req.num_inference_steps,
            guidance_scale=req.guidance_scale,
            generator=generator,
        )
        image = result.images[0]

        out_path = str(OUTPUT_DIR / f"{job_id}.mp4")
        _image_to_video(image, out_path, req.fps, req.duration_seconds, req.width, req.height)

        gc.collect()

        jobs[job_id].update({
            "status": "completed", "progress": 100,
            "video_path": out_path, "video_url": f"/videos/{job_id}",
            "updated_at": datetime.utcnow().isoformat(),
            "metadata": {"prompt": req.prompt, "model": "stable-diffusion-2-1", "mode": "image+ken-burns"},
        })
        print(f"[ImageServer] Done → {out_path}")

    except Exception as e:
        gc.collect()
        jobs[job_id].update({"status": "failed", "error": str(e),
                              "updated_at": datetime.utcnow().isoformat()})
        print(f"[ImageServer] FAILED: {e}")


# ─────────────────────────────────────────────
# Endpoints  (same interface as server.py)
# ─────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "model": "stable-diffusion-2-1", "mode": "image+ken-burns",
            "model_loaded": pipe is not None,
            "active_jobs": sum(1 for j in jobs.values() if j["status"] == "processing"),
            "total_jobs": len(jobs)}


@app.post("/generate", response_model=JobStatus)
async def generate_video(req: GenerateRequest, background_tasks: BackgroundTasks,
                          _: str = Depends(verify_api_key)):
    if pipe is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    if any(j["status"] == "processing" for j in jobs.values()):
        raise HTTPException(status_code=429, detail="A job is already processing.")
    job_id = str(uuid.uuid4())
    now    = datetime.utcnow().isoformat()
    jobs[job_id] = {"job_id": job_id, "status": "queued",
                    "created_at": now, "updated_at": now, "progress": 0, "request": req.dict()}
    background_tasks.add_task(_run_generation, job_id, req)
    return JobStatus(**jobs[job_id])


@app.get("/jobs/{job_id}", response_model=JobStatus)
async def get_job(job_id: str, _: str = Depends(verify_api_key)):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return JobStatus(**jobs[job_id])


@app.get("/videos/{job_id}")
async def download_video(job_id: str):
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
    vp = job.get("video_path")
    if vp and Path(vp).exists():
        Path(vp).unlink()
    return {"deleted": job_id}


@app.post("/generate-from-audio", response_model=JobStatus)
async def generate_from_audio(
    background_tasks: BackgroundTasks,
    speech_file: UploadFile = File(...),
    music_file:  Optional[UploadFile] = File(default=None),
    duration_seconds: float = Form(default=3.0),
    language: str           = Form(default="auto"),
    seed: Optional[int]     = Form(default=None),
    _: str = Depends(verify_api_key),
):
    import whisper
    LANG_CODES = {"auto": None, "odia": "or", "telugu": "te", "hindi": "hi", "english": "en"}
    whisper_lang = LANG_CODES.get(language.lower())
    tmp_dir = tempfile.mkdtemp()
    try:
        s_suffix = Path(speech_file.filename or "audio.m4a").suffix or ".m4a"
        speech_path = os.path.join(tmp_dir, f"speech{s_suffix}")
        with open(speech_path, "wb") as f:
            f.write(await speech_file.read())
        if music_file:
            await music_file.read()
        wmodel  = whisper.load_model("small")
        result  = wmodel.transcribe(speech_path, language=whisper_lang)
        transcript = result.get("text", "").strip()
        del wmodel; gc.collect()
        if not transcript:
            raise HTTPException(status_code=422, detail="Could not transcribe audio.")
    except HTTPException:
        shutil.rmtree(tmp_dir, ignore_errors=True); raise
    except Exception as e:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=f"Audio error: {e}")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    req = GenerateRequest(prompt=transcript,
                          duration_seconds=min(max(duration_seconds, 1.0), 10.0), seed=seed)
    job_id = str(uuid.uuid4())
    now    = datetime.utcnow().isoformat()
    jobs[job_id] = {"job_id": job_id, "status": "queued",
                    "created_at": now, "updated_at": now,
                    "progress": 0, "request": req.dict(), "transcribed_text": transcript}
    background_tasks.add_task(_run_generation, job_id, req)
    return JobStatus(**jobs[job_id])
