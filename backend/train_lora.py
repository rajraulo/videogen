"""
train_lora.py — LoRA fine-tuning for CogVideoX / Open-Sora
Run on single A100 (40GB) or better.

Usage:
    python train_lora.py \
        --model_id THUDM/CogVideoX-5b \
        --video_dir data/my_videos \
        --caption_file data/captions.json \
        --output_dir checkpoints/my_style \
        --num_epochs 100 \
        --lr 1e-4
"""

import argparse
import json
import os
from pathlib import Path

import imageio
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from peft import LoraConfig, get_peft_model
from diffusers import CogVideoXPipeline, CogVideoXDDIMScheduler
from diffusers.training_utils import compute_snr
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed

logger = get_logger(__name__)


# ─────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────

class VideoTextDataset(Dataset):
    def __init__(self, video_dir, caption_file, num_frames=49, height=480, width=720):
        self.video_dir   = Path(video_dir)
        self.captions    = json.load(open(caption_file))
        self.video_files = sorted(self.video_dir.glob("*.mp4"))
        self.num_frames  = num_frames
        self.height      = height
        self.width       = width

        print(f"[Dataset] {len(self.video_files)} videos loaded")

    def __len__(self):
        return len(self.video_files)

    def _load_video(self, path):
        reader = imageio.get_reader(str(path), format="ffmpeg")
        frames = []
        try:
            for i, frame in enumerate(reader):
                if i >= self.num_frames:
                    break
                # Resize frame
                from PIL import Image
                img = Image.fromarray(frame).resize((self.width, self.height), Image.LANCZOS)
                frames.append(np.array(img))
        finally:
            reader.close()

        # Pad to num_frames
        while len(frames) < self.num_frames:
            frames.append(frames[-1] if frames else np.zeros((self.height, self.width, 3), dtype=np.uint8))

        tensor = torch.from_numpy(np.stack(frames)).float() / 255.0  # [T, H, W, 3]
        tensor = tensor.permute(0, 3, 1, 2)                          # [T, 3, H, W]
        tensor = (tensor - 0.5) / 0.5                                 # normalize to [-1, 1]
        return tensor

    def __getitem__(self, idx):
        vf      = self.video_files[idx]
        caption = self.captions.get(vf.name, self.captions.get(vf.stem, ""))
        video   = self._load_video(vf)
        return {"pixel_values": video, "caption": caption}


# ─────────────────────────────────────────────
# LoRA configuration
# ─────────────────────────────────────────────

LORA_CONFIG = LoraConfig(
    r=64,
    lora_alpha=64,
    init_lora_weights="gaussian",
    target_modules=[
        "to_k", "to_q", "to_v", "to_out.0",
        "proj_in", "proj_out",
        "ff.net.0.proj", "ff.net.2",
    ],
)


# ─────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────

def train(args):
    accelerator = Accelerator(
        gradient_accumulation_steps=args.grad_accum,
        mixed_precision="bf16",
        log_with="tensorboard",
        project_dir=args.output_dir,
    )

    if accelerator.is_main_process:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)

    # ── Load pipeline components ──────────────
    logger.info(f"Loading model: {args.model_id}")
    pipe = CogVideoXPipeline.from_pretrained(args.model_id, torch_dtype=torch.bfloat16)
    pipe.scheduler = CogVideoXDDIMScheduler.from_config(
        pipe.scheduler.config, timestep_spacing="trailing"
    )

    transformer = pipe.transformer
    vae         = pipe.vae
    text_encoder = pipe.text_encoder
    tokenizer    = pipe.tokenizer
    noise_scheduler = pipe.scheduler

    # Freeze everything except transformer LoRA
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    transformer.requires_grad_(False)

    # Inject LoRA into transformer
    transformer = get_peft_model(transformer, LORA_CONFIG)
    transformer.print_trainable_parameters()

    # ── Dataset & loader ──────────────────────
    dataset = VideoTextDataset(
        args.video_dir,
        args.caption_file,
        num_frames=args.num_frames,
        height=args.height,
        width=args.width,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    # ── Optimiser ────────────────────────────
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, transformer.parameters()),
        lr=args.lr,
        weight_decay=1e-4,
        betas=(0.9, 0.999),
    )

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.num_epochs * len(loader),
        eta_min=args.lr * 0.01,
    )

    transformer, optimizer, loader, lr_scheduler = accelerator.prepare(
        transformer, optimizer, loader, lr_scheduler
    )

    vae         = vae.to(accelerator.device)
    text_encoder = text_encoder.to(accelerator.device)

    # ── Training ──────────────────────────────
    global_step = 0

    for epoch in range(args.num_epochs):
        transformer.train()
        epoch_loss = 0.0

        for step, batch in enumerate(loader):
            with accelerator.accumulate(transformer):
                pixel_values = batch["pixel_values"].to(accelerator.device, dtype=torch.bfloat16)
                captions     = batch["caption"]

                # Encode video to latents
                with torch.no_grad():
                    latents = vae.encode(pixel_values).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor

                # Encode text
                with torch.no_grad():
                    tokens = tokenizer(
                        captions,
                        padding="max_length",
                        max_length=226,
                        truncation=True,
                        return_tensors="pt",
                    ).input_ids.to(accelerator.device)
                    encoder_hidden_states = text_encoder(tokens)[0]

                # Sample noise and timestep
                noise     = torch.randn_like(latents)
                bsz       = latents.shape[0]
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps,
                    (bsz,), device=accelerator.device,
                ).long()

                # Add noise to latents
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Predict noise
                model_pred = transformer(
                    hidden_states=noisy_latents,
                    timestep=timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    return_dict=False,
                )[0]

                # SNR-weighted loss
                if args.snr_gamma > 0:
                    snr = compute_snr(noise_scheduler, timesteps)
                    weight = torch.stack([snr, args.snr_gamma * torch.ones_like(snr)], dim=1).min(dim=1)[0] / snr
                    loss = (F.mse_loss(model_pred.float(), noise.float(), reduction="none").mean(dim=list(range(1, noise.ndim))) * weight).mean()
                else:
                    loss = F.mse_loss(model_pred.float(), noise.float())

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(transformer.parameters(), 1.0)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            epoch_loss += loss.item()
            global_step += 1

            if global_step % args.log_every == 0 and accelerator.is_main_process:
                avg = epoch_loss / (step + 1)
                logger.info(f"Epoch {epoch+1}/{args.num_epochs} | Step {global_step} | Loss {avg:.4f} | LR {lr_scheduler.get_last_lr()[0]:.2e}")

        # ── Save checkpoint ───────────────────
        if (epoch + 1) % args.save_every == 0 and accelerator.is_main_process:
            ckpt_dir = Path(args.output_dir) / f"epoch_{epoch+1:04d}"
            ckpt_dir.mkdir(exist_ok=True)
            unwrapped = accelerator.unwrap_model(transformer)
            unwrapped.save_pretrained(ckpt_dir)
            logger.info(f"Saved checkpoint: {ckpt_dir}")

    # ── Final save ────────────────────────────
    if accelerator.is_main_process:
        final_dir = Path(args.output_dir) / "final_lora"
        final_dir.mkdir(exist_ok=True)
        unwrapped = accelerator.unwrap_model(transformer)
        unwrapped.save_pretrained(final_dir)
        logger.info(f"Training complete! Final LoRA saved to: {final_dir}")


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id",     default="THUDM/CogVideoX-5b")
    parser.add_argument("--video_dir",    required=True)
    parser.add_argument("--caption_file", required=True)
    parser.add_argument("--output_dir",   required=True)
    parser.add_argument("--num_epochs",   type=int,   default=100)
    parser.add_argument("--batch_size",   type=int,   default=1)
    parser.add_argument("--lr",           type=float, default=1e-4)
    parser.add_argument("--grad_accum",   type=int,   default=4)
    parser.add_argument("--num_frames",   type=int,   default=49)
    parser.add_argument("--height",       type=int,   default=480)
    parser.add_argument("--width",        type=int,   default=720)
    parser.add_argument("--snr_gamma",    type=float, default=5.0)
    parser.add_argument("--seed",         type=int,   default=42)
    parser.add_argument("--log_every",    type=int,   default=50)
    parser.add_argument("--save_every",   type=int,   default=10)
    args = parser.parse_args()

    train(args)
