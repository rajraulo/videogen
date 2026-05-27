"""
VideoGen Pipeline — Multi-model text-to-video generation
Supports the best free/open-source models (all Apache 2.0):

  Model                        VRAM    Quality   HuggingFace ID
  ──────────────────────────────────────────────────────────────────────────────
  Wan2.1-T2V-1.3B  ← DEFAULT   8 GB    Good      Wan-AI/Wan2.1-T2V-1.3B-Diffusers
  LTX-Video-0.9.1               24 GB   Very Good Lightricks/LTX-Video-0.9.1
  Wan2.1-T2V-14B                48 GB   Excellent Wan-AI/Wan2.1-T2V-14B-Diffusers
  HunyuanVideo                  60 GB   Best      tencent/HunyuanVideo
  CogVideoX-5B     (legacy)     24 GB   Good      THUDM/CogVideoX-5b

To change model: set model_id in VideoGenConfig or the VIDEOGEN_MODEL env var.
"""

import os
import gc
import time
import uuid
import torch
import imageio
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

from diffusers.utils import export_to_video


# ─────────────────────────────────────────────
# Best free models registry
# ─────────────────────────────────────────────

BEST_FREE_MODELS: dict[str, str] = {
    "wan-1.3b":  "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",   # default — 8 GB VRAM
    "wan-14b":   "Wan-AI/Wan2.1-T2V-14B-Diffusers",     # best quality, 48 GB
    "ltx":       "Lightricks/LTX-Video-0.9.1",           # fastest, 24 GB
    "hunyuan":   "tencent/HunyuanVideo",                 # top quality, 60 GB
    "cogvideox": "THUDM/CogVideoX-5b",                  # legacy fallback, 24 GB
}

# Per-model sensible defaults
_MODEL_DEFAULTS: dict[str, dict] = {
    "wan":       {"num_frames": 81,  "fps": 16, "height": 480,  "width": 832,  "guidance_scale": 5.0,  "steps": 50},
    "ltx":       {"num_frames": 121, "fps": 24, "height": 512,  "width": 768,  "guidance_scale": 3.0,  "steps": 50},
    "hunyuan":   {"num_frames": 61,  "fps": 24, "height": 720,  "width": 1280, "guidance_scale": 6.0,  "steps": 50},
    "cogvideox": {"num_frames": 49,  "fps": 12, "height": 480,  "width": 720,  "guidance_scale": 6.0,  "steps": 50},
}


def _detect_family(model_id: str) -> str:
    m = model_id.lower()
    if "wan" in m:
        return "wan"
    if "ltx" in m:
        return "ltx"
    if "hunyuan" in m:
        return "hunyuan"
    return "cogvideox"


def _best_device() -> str:
    """Auto-detect the best available device: CUDA → MPS → CPU."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _best_dtype(device: str) -> torch.dtype:
    """bfloat16 on CUDA, float16 on MPS (bfloat16 has limited MPS support), float32 on CPU."""
    if device == "cuda":
        return torch.bfloat16
    if device == "mps":
        return torch.float16
    return torch.float32


def _load_pipeline(model_id: str, dtype: torch.dtype, device: str, cpu_offload: bool):
    """Load the correct pipeline class for the given model ID."""
    family = _detect_family(model_id)
    on_mps = device == "mps"

    if family == "wan":
        from diffusers import WanPipeline
        pipe = WanPipeline.from_pretrained(model_id, torch_dtype=dtype)

    elif family == "ltx":
        from diffusers import LTXPipeline
        pipe = LTXPipeline.from_pretrained(model_id, torch_dtype=dtype)

    elif family == "hunyuan":
        from diffusers import HunyuanVideoPipeline
        from diffusers.models import HunyuanVideoTransformer3DModel
        transformer = HunyuanVideoTransformer3DModel.from_pretrained(
            model_id, subfolder="transformer", torch_dtype=torch.bfloat16,
        )
        pipe = HunyuanVideoPipeline.from_pretrained(
            model_id, transformer=transformer, torch_dtype=torch.float16,
        )

    else:  # cogvideox
        from diffusers import CogVideoXPipeline, CogVideoXDDIMScheduler
        pipe = CogVideoXPipeline.from_pretrained(model_id, torch_dtype=dtype)
        pipe.scheduler = CogVideoXDDIMScheduler.from_config(
            pipe.scheduler.config, timestep_spacing="trailing",
        )

    # Memory optimisations
    # cpu_offload uses CUDA hooks — not supported on MPS; move directly instead
    if cpu_offload and not on_mps:
        pipe.enable_sequential_cpu_offload()
    else:
        pipe.to(device)

    if hasattr(pipe, "vae"):
        if hasattr(pipe.vae, "enable_slicing"):
            pipe.vae.enable_slicing()
        if hasattr(pipe.vae, "enable_tiling"):
            pipe.vae.enable_tiling()

    return pipe, family


# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────

@dataclass
class VideoGenConfig:
    # Default: Wan2.1-1.3B — best free model that runs on 8 GB VRAM
    # Override via env var VIDEOGEN_MODEL=wan-14b|ltx|hunyuan|cogvideox|<hf-id>
    model_id: str = os.getenv(
        "VIDEOGEN_MODEL",
        BEST_FREE_MODELS["wan-1.3b"],
    )

    # Generation parameters (auto-filled from model defaults in __post_init__)
    num_frames: int = 0          # 0 → use model default
    fps: int        = 0
    height: int     = 0
    width: int      = 0
    num_inference_steps: int  = 0
    guidance_scale: float     = 0.0

    # Hardware — auto-detected; override with VIDEOGEN_DEVICE env var
    device: str = os.getenv("VIDEOGEN_DEVICE", _best_device())
    dtype: torch.dtype = field(default=None)           # filled in __post_init__
    enable_cpu_offload: bool = True    # ignored on MPS; set False on 40 GB+ CUDA

    # Output
    output_dir: str = "outputs/videos"
    max_sequence_length: int = 226

    def __post_init__(self):
        # Resolve short names like "wan-1.3b" → full HuggingFace repo ID
        if self.model_id in BEST_FREE_MODELS:
            self.model_id = BEST_FREE_MODELS[self.model_id]
        if self.dtype is None:
            self.dtype = _best_dtype(self.device)
        family = _detect_family(self.model_id)
        defaults = _MODEL_DEFAULTS.get(family, _MODEL_DEFAULTS["cogvideox"])
        if self.num_frames        == 0:   self.num_frames        = defaults["num_frames"]
        if self.fps               == 0:   self.fps               = defaults["fps"]
        if self.height            == 0:   self.height            = defaults["height"]
        if self.width             == 0:   self.width             = defaults["width"]
        if self.num_inference_steps == 0: self.num_inference_steps = defaults["steps"]
        if self.guidance_scale    == 0.0: self.guidance_scale    = defaults["guidance_scale"]


# ─────────────────────────────────────────────
# Pipeline
# ─────────────────────────────────────────────

class VideoGenModel:
    """
    Unified wrapper for Wan2.1, LTX-Video, HunyuanVideo, and CogVideoX.
    Switch models by changing model_id in VideoGenConfig or the VIDEOGEN_MODEL env var.
    """

    def __init__(self, config: VideoGenConfig):
        self.config = config
        self.pipe   = None
        self.family = None
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)

    def load(self):
        cfg = self.config
        print(f"[VideoGen] Loading {cfg.model_id} ...")
        t0 = time.time()

        print(f"[VideoGen] Device: {cfg.device}  dtype: {cfg.dtype}")
        self.pipe, self.family = _load_pipeline(
            cfg.model_id, cfg.dtype, cfg.device, cfg.enable_cpu_offload,
        )

        print(f"[VideoGen] Model ready ({self.family}) in {time.time()-t0:.1f}s")
        print(f"[VideoGen] Defaults — {cfg.num_frames} frames | {cfg.fps} fps | "
              f"{cfg.width}×{cfg.height}")

    def unload(self):
        del self.pipe
        self.pipe   = None
        self.family = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    @torch.inference_mode()
    def generate(
        self,
        prompt:               str,
        negative_prompt:      str   = "blurry, low quality, distorted, artifacts, watermark",
        seed:                 Optional[int]   = None,
        num_frames:           Optional[int]   = None,
        num_inference_steps:  Optional[int]   = None,
        guidance_scale:       Optional[float] = None,
    ) -> dict:
        if self.pipe is None:
            raise RuntimeError("Model not loaded. Call .load() first.")

        cfg    = self.config
        job_id = str(uuid.uuid4())[:8]
        t0     = time.time()

        frames  = num_frames          or cfg.num_frames
        steps   = num_inference_steps or cfg.num_inference_steps
        scale   = guidance_scale      or cfg.guidance_scale

        generator = None
        if seed is not None:
            generator = torch.Generator(device="cpu").manual_seed(seed)

        print(f"[VideoGen] [{job_id}] '{prompt[:60]}…'")
        print(f"[VideoGen] [{job_id}] {frames}f | {steps} steps | CFG {scale} | {self.family}")

        # ── Call pipeline ──────────────────────────────────────
        if self.family == "hunyuan":
            # HunyuanVideo does not use negative_prompt
            result = self.pipe(
                prompt               = prompt,
                num_frames           = frames,
                num_inference_steps  = steps,
                guidance_scale       = scale,
                generator            = generator,
                height               = cfg.height,
                width                = cfg.width,
                output_type          = "pil",
            )
        else:
            result = self.pipe(
                prompt               = prompt,
                negative_prompt      = negative_prompt,
                num_frames           = frames,
                num_inference_steps  = steps,
                guidance_scale       = scale,
                generator            = generator,
                height               = cfg.height,
                width                = cfg.width,
                output_type          = "pil",
            )

        # result.frames[0] is a list of PIL images
        video_frames = result.frames[0]

        # ── Export ─────────────────────────────────────────────
        out_path = os.path.join(cfg.output_dir, f"{job_id}.mp4")
        export_to_video(video_frames, out_path, fps=cfg.fps)

        elapsed  = time.time() - t0
        duration = frames / cfg.fps

        print(f"[VideoGen] [{job_id}] Done in {elapsed:.1f}s → {out_path}")

        return {
            "job_id":          job_id,
            "video_path":      out_path,
            "duration_seconds": duration,
            "generation_time": elapsed,
            "metadata": {
                "model":          cfg.model_id,
                "model_family":   self.family,
                "prompt":         prompt,
                "frames":         frames,
                "fps":            cfg.fps,
                "height":         cfg.height,
                "width":          cfg.width,
                "steps":          steps,
                "guidance_scale": scale,
                "seed":           seed,
            },
        }

    def enhance_prompt(self, prompt: str) -> str:
        quality_tokens = (
            "cinematic, high quality, 4K, smooth motion, "
            "professional lighting, sharp focus"
        )
        style_map = {
            "sunset": "golden hour, warm tones, dramatic sky",
            "ocean":  "deep blue water, waves, coastal atmosphere",
            "city":   "urban environment, city lights, architectural detail",
            "nature": "lush vegetation, natural lighting, tranquil",
        }
        enhanced = prompt
        for kw, addition in style_map.items():
            if kw in prompt.lower():
                enhanced += f", {addition}"
                break
        return f"{enhanced}, {quality_tokens}"


# ─────────────────────────────────────────────
# Fine-tuning helper (LoRA)
# ─────────────────────────────────────────────

class VideoGenFineTuner:
    """
    LoRA fine-tuning wrapper.
    See train_lora.py for the full training loop.
    """

    def __init__(self, base_model_id: str, lora_rank: int = 64):
        self.base_model_id = base_model_id
        self.lora_rank     = lora_rank

    def prepare_dataset(self, video_dir: str, caption_file: str):
        from torch.utils.data import Dataset
        import json

        class VideoDataset(Dataset):
            def __init__(self, video_dir, caption_file, height=480, width=832, num_frames=81):
                self.video_dir   = Path(video_dir)
                self.captions    = json.load(open(caption_file))
                self.files       = list(self.video_dir.glob("*.mp4"))
                self.height      = height
                self.width       = width
                self.num_frames  = num_frames

            def __len__(self):
                return len(self.files)

            def __getitem__(self, idx):
                video_path = self.files[idx]
                caption    = self.captions.get(video_path.name, "")
                reader     = imageio.get_reader(str(video_path))
                frames     = []
                for i, frame in enumerate(reader):
                    if i >= self.num_frames:
                        break
                    frames.append(frame)
                reader.close()
                while len(frames) < self.num_frames:
                    frames.append(frames[-1])
                tensor = torch.from_numpy(np.stack(frames)).float() / 255.0
                tensor = tensor.permute(0, 3, 1, 2)
                return {"pixel_values": tensor, "caption": caption}

        return VideoDataset(video_dir, caption_file)

    def train(self, video_dir: str, output_dir: str,
              caption_file: str = "captions.json",
              num_epochs: int = 10, lr: float = 1e-4, batch_size: int = 1):
        print(f"[FineTune] Preparing dataset from {video_dir}")
        dataset = self.prepare_dataset(video_dir, caption_file)
        print(f"[FineTune] {len(dataset)} videos | {num_epochs} epochs → {output_dir}")
        print("[FineTune] See train_lora.py for the full training loop.")


# ─────────────────────────────────────────────
# Quick smoke test
# ─────────────────────────────────────────────

if __name__ == "__main__":
    # Defaults to Wan2.1-1.3B — override with VIDEOGEN_MODEL env var
    cfg = VideoGenConfig(num_inference_steps=20, num_frames=17, enable_cpu_offload=True)
    model = VideoGenModel(cfg)
    model.load()

    result = model.generate(
        prompt="A golden retriever running through a sunlit meadow, cinematic slow motion",
        seed=42,
    )
    print(f"Video: {result['video_path']}  ({result['generation_time']:.1f}s)")
