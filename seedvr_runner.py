"""Thin wrapper around the SeedVR2 7B inference pipeline for single images.

The official SeedVR repo (https://github.com/ByteDance-Seed/SeedVR) ships an
inference script designed for video. This module wraps the underlying pieces
(DiT + VAE + text encoder) so that we can call it with a single PIL image and
get a single upscaled PIL image back.

NOTE: SeedVR's internals shift between releases. The integration points below
are written against the layout in the public repo as of early 2026. If the
upstream code path changes, adjust the imports inside ``_lazy_import``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from PIL import Image


MODEL_DIR = Path(os.environ.get("SEEDVR_MODEL_DIR", "/runpod-volume/seedvr/models"))


class SeedVR2Runner:
    """Loads SeedVR2-7B once and runs single-image upscales."""

    def __init__(self, model_dir: Path = MODEL_DIR, device: str = "cuda", dtype=torch.bfloat16):
        self.model_dir = Path(model_dir)
        self.device = device
        self.dtype = dtype
        self._loaded = False
        self._dit = None
        self._vae = None
        self._text_encoder = None
        self._scheduler = None

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------
    def load(self) -> None:
        if self._loaded:
            return
        dit, vae, text_encoder, scheduler = self._lazy_import()
        self._dit = dit
        self._vae = vae
        self._text_encoder = text_encoder
        self._scheduler = scheduler
        self._loaded = True

    def _lazy_import(self):
        """Import and instantiate the SeedVR2 components.

        Kept inside a method so the heavy imports happen only on the worker,
        not at module import time.
        """
        import sys
        sys.path.insert(0, "/app/SeedVR")

        # The exact module paths come from the upstream repo. Adjust if upstream
        # restructures.
        from projects.video_diffusion_sr.infer import VideoDiffusionInfer  # type: ignore
        from omegaconf import OmegaConf

        cfg_path = "/app/SeedVR/configs_7b/main.yaml"
        cfg = OmegaConf.load(cfg_path)

        cfg.dit.checkpoint = str(self.model_dir / "seedvr2_ema_7b.pth")
        cfg.vae.checkpoint = str(self.model_dir / "ema_vae.pth")
        cfg.text_encoder.checkpoint = str(self.model_dir / "pos_emb.pt")
        cfg.text_encoder.neg_checkpoint = str(self.model_dir / "neg_emb.pt")

        runner = VideoDiffusionInfer(cfg)
        runner.configure_dit_model(device=self.device, checkpoint=cfg.dit.checkpoint)
        runner.configure_vae_model()
        return runner, runner.vae, runner.text_encoder, runner.scheduler

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    @torch.inference_mode()
    def upscale(
        self,
        image: Image.Image,
        target_size: Tuple[int, int],
        seed: int = 42,
        steps: int = 1,
        cfg_scale: float = 1.0,
    ) -> Image.Image:
        """Upscale a single PIL image to ``target_size`` (width, height)."""
        if not self._loaded:
            self.load()

        target_w, target_h = target_size
        # SeedVR2 requires sizes divisible by the patch stride (typically 16).
        target_w = (target_w // 16) * 16
        target_h = (target_h // 16) * 16

        arr = np.asarray(image.convert("RGB"))
        # SeedVR expects video tensors: (T, C, H, W) in [-1, 1]
        tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(self.device)
        tensor = tensor.to(self.dtype) / 127.5 - 1.0

        torch.manual_seed(seed)
        out = self._dit.inference(
            video=tensor,
            target_size=(target_h, target_w),
            num_inference_steps=steps,
            guidance_scale=cfg_scale,
        )

        # out: (T, C, H, W) in [-1, 1]
        out = out.squeeze(0).clamp(-1, 1)
        out = ((out + 1.0) * 127.5).to(torch.uint8).permute(1, 2, 0).cpu().numpy()
        return Image.fromarray(out)


_runner: Optional[SeedVR2Runner] = None


def get_runner() -> SeedVR2Runner:
    global _runner
    if _runner is None:
        _runner = SeedVR2Runner()
        _runner.load()
    return _runner
