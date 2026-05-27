# VideoGen — Text-to-Video Generation System
## A Google Veo-inspired pipeline: Model + API Server + Android App

---

## Project Structure

```
videogen/
├── backend/
│   ├── model/
│   │   └── video_gen_pipeline.py   ← Core generation model (CogVideoX)
│   ├── api/
│   │   └── server.py               ← FastAPI inference server
│   ├── train_lora.py               ← LoRA fine-tuning script
│   ├── requirements.txt
│   └── Dockerfile
└── android/
    └── app/src/main/java/com/videogen/
        ├── ui/
        │   ├── MainActivity.kt          ← Entry point + navigation
        │   ├── screens/
        │   │   ├── GenerateScreen.kt    ← Prompt UI + job status
        │   │   └── VideoPlayerScreen.kt ← ExoPlayer video viewer
        │   └── viewmodels/
        │       └── GenerateViewModel.kt ← State management + API calls
        └── api/
            └── VideoGenApiClient.kt     ← Retrofit HTTP client
```

---

## Quick Start — Backend

### 1. Install dependencies
```bash
# Python 3.10+ required
pip install -r backend/requirements.txt

# Install ffmpeg (system)
sudo apt install ffmpeg         # Ubuntu/Debian
brew install ffmpeg             # macOS
```

### 2. Set your API key
```bash
export VIDEOGEN_API_KEY="your-secret-key"
```

### 3. Start the server
```bash
cd backend
uvicorn api.server:app --host 0.0.0.0 --port 8000 --workers 1
```

The server will download the CogVideoX-5B model (~18GB) on first run.

### 4. Test it
```bash
curl -X POST http://localhost:8000/generate \
  -H "x-api-key: your-secret-key" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "A cat walking through a sunny garden", "duration_seconds": 4}'

# → Returns: {"job_id": "abc123", "status": "queued", ...}

# Poll for result:
curl http://localhost:8000/jobs/abc123 \
  -H "x-api-key: your-secret-key"

# Download when status == "completed":
curl http://localhost:8000/videos/abc123 \
  -H "x-api-key: your-secret-key" \
  -o result.mp4
```

---

## Docker Deployment

```bash
# Build
docker build -t videogen-api ./backend

# Run (mount a volume for model cache so it persists)
docker run -d \
  --gpus all \
  -p 8000:8000 \
  -v /data/models:/models/cache \
  -v /data/outputs:/outputs \
  -e VIDEOGEN_API_KEY=your-secret-key \
  videogen-api
```

---

## GPU Cloud Deployment (Runpod — cheapest option)

1. Create a Runpod account at runpod.io
2. Deploy a pod with: A100 40GB or RTX 4090 (minimum 24GB VRAM)
3. Use the Docker image above
4. Expose port 8000 and note the public URL
5. Update `BASE_URL` in `android/VideoGenApiClient.kt`

**Estimated cost:** ~$1–2/hour on Runpod for an A100

---

## Fine-tuning on your own videos

### Prepare your dataset
```
data/
├── my_videos/
│   ├── clip001.mp4
│   ├── clip002.mp4
│   └── ...
└── captions.json     ← {"clip001.mp4": "A dog running on the beach", ...}
```

### Run fine-tuning
```bash
python backend/train_lora.py \
  --model_id THUDM/CogVideoX-5b \
  --video_dir data/my_videos \
  --caption_file data/captions.json \
  --output_dir checkpoints/my_style \
  --num_epochs 100 \
  --lr 1e-4 \
  --batch_size 1 \
  --grad_accum 4
```

Requires: A100 40GB, ~12–24 hours for 100 epochs on 100 videos.

---

## Android App Setup

### 1. Configure the API endpoint
In `VideoGenApiClient.kt`, update:
```kotlin
private const val BASE_URL = "https://YOUR_SERVER_IP:8000/"
private const val API_KEY  = "your-secret-key"
```

### 2. Open in Android Studio
- File → Open → select the `android/` directory
- Let Gradle sync complete

### 3. Run on device
- Connect an Android device (API 26+) or start an emulator
- Click Run ▶

### 4. Build release APK
```bash
cd android
./gradlew assembleRelease
# Output: app/build/outputs/apk/release/app-release.apk
```

---

## Architecture Overview

```
[Android App]
     │  HTTP POST /generate (prompt)
     ▼
[FastAPI Server]  ──►  [Job Queue (in-memory / Redis)]
     │
     ▼
[CogVideoX Model]
  • T5 text encoder     ← encodes your prompt
  • 3D VAE              ← encodes/decodes video latents
  • DiT Transformer     ← denoises with temporal attention
  • DDIM Scheduler      ← controls denoising steps
     │
     ▼
[MP4 Output]  ──►  [HTTP GET /videos/{job_id}]
     │
     ▼
[Android ExoPlayer]  ←  streams/downloads video
```

---

## Hardware Requirements

| Setup          | GPU           | VRAM   | Speed          |
|----------------|---------------|--------|----------------|
| Minimum (dev)  | RTX 3090      | 24 GB  | ~5–8 min/video |
| Recommended    | A100 40GB     | 40 GB  | ~1–2 min/video |
| Production     | 2× A100 80GB  | 160 GB | ~30–60s/video  |

---

## Model Options — Best Free (Apache 2.0) Models

| Shortname      | HuggingFace ID                           | VRAM   | Quality   | Speed          |
|----------------|------------------------------------------|--------|-----------|----------------|
| **wan-1.3b** ← default | `Wan-AI/Wan2.1-T2V-1.3B-Diffusers` | **8 GB** | Good | ~3 min/video |
| ltx            | `Lightricks/LTX-Video-0.9.1`             | 24 GB  | Very Good | **~30s/video** |
| wan-14b        | `Wan-AI/Wan2.1-T2V-14B-Diffusers`        | 48 GB  | Excellent | ~2 min/video   |
| hunyuan        | `tencent/HunyuanVideo`                   | 60 GB  | **Best**  | ~5 min/video   |
| cogvideox      | `THUDM/CogVideoX-5b`                     | 24 GB  | Good      | ~5 min/video   |

**All models are 100% free — Apache 2.0 license.**

### Switch model via environment variable (no code changes needed)
```bash
# Use Wan2.1-1.3B (default — 8 GB VRAM)
export VIDEOGEN_MODEL=Wan-AI/Wan2.1-T2V-1.3B-Diffusers

# Use LTX-Video (fastest — 24 GB VRAM)
export VIDEOGEN_MODEL=Lightricks/LTX-Video-0.9.1

# Use HunyuanVideo (best quality — 60 GB VRAM, Runpod A100 80GB)
export VIDEOGEN_MODEL=tencent/HunyuanVideo

# Or pass any HuggingFace model ID directly
export VIDEOGEN_MODEL=Wan-AI/Wan2.1-T2V-14B-Diffusers
```

### Or set in code
```python
cfg = VideoGenConfig(model_id="Lightricks/LTX-Video-0.9.1")
```

---

## API Reference

| Method | Endpoint           | Description                   |
|--------|--------------------|-------------------------------|
| GET    | /health            | Server health check           |
| POST   | /generate          | Submit generation job         |
| GET    | /jobs/{id}         | Poll job status               |
| GET    | /videos/{id}       | Download completed video (MP4)|
| GET    | /jobs              | List all jobs                 |
| DELETE | /jobs/{id}         | Delete job + file             |

All endpoints require header: `x-api-key: <your-key>`

---

## Roadmap to Improve Quality

1. **Better text encoder** — Use LLaMA-3 or GPT-4 to rewrite prompts
2. **Motion conditioning** — Add optical flow supervision during training
3. **Higher resolution** — Scale to 1080p with tiling
4. **Longer videos** — Sliding window attention for >60 second clips
5. **Style LoRAs** — Train domain-specific adapters (anime, realistic, abstract)
6. **RLHF reward** — Human preference scoring to guide generation

---

## License
MIT — use freely for personal and commercial projects.
Model weights (CogVideoX) are subject to their own license: Apache 2.0.
