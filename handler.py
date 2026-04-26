"""RunPod serverless handler for SeedVR2 image upscaling.

Uses numz/ComfyUI-SeedVR2_VideoUpscaler as a Python library (not subprocess),
so the DiT + VAE stay loaded in VRAM across requests on a warm worker. First
request pays the model-load cost (~10-15 s), subsequent requests are pure
inference (~5-10 s).
"""

from __future__ import annotations

import argparse
import base64
import io
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict

import requests
import runpod
from PIL import Image

# The wrapper expects to be importable from /app/seedvr-cli with cwd there.
CLI_DIR = Path("/app/seedvr-cli")
sys.path.insert(0, str(CLI_DIR))
os.chdir(str(CLI_DIR))

# Defer heavy imports until first call to avoid blowing up handler start when
# something's misconfigured.
_loaded: Dict[str, Any] = {}


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("seedvr")

MODEL_DIR = os.environ.get("SEEDVR_MODEL_DIR", "/runpod-volume/seedvr/SEEDVR2")
DEFAULT_MODEL = "seedvr2_ema_7b_fp16.safetensors"
ALLOWED_MODELS = {
    "seedvr2_ema_3b_fp8_e4m3fn.safetensors",
    "seedvr2_ema_3b_fp16.safetensors",
    "seedvr2_ema_7b_fp8_e4m3fn_mixed_block35_fp16.safetensors",
    "seedvr2_ema_7b_fp16.safetensors",
    "seedvr2_ema_7b_sharp_fp8_e4m3fn_mixed_block35_fp16.safetensors",
    "seedvr2_ema_7b_sharp_fp16.safetensors",
}


def _ensure_loaded():
    """Lazy-import wrapper internals + create per-model runner_cache.

    Import order matters: bring in src.utils.* (which transitively imports
    diffusers + torch via attn_video_vae) BEFORE inference_cli. The latter
    eagerly probes CUDA memory at import time, and doing that before the
    diffusers init has registered its CUDA allocator config triggers a
    "CUDAAllocatorConfig backend mismatch" assert and segfault.
    """
    if "process_frames_core" in _loaded:
        return
    # 1) Warm up diffusers + torch via the lighter-weight utility modules.
    from src.utils.downloads import download_weight
    from src.utils.model_registry import DEFAULT_VAE
    from src.utils.debug import Debug
    # 2) Now safe to import inference_cli (does CUDA memory probes at load).
    from inference_cli import _process_frames_core
    _loaded["process_frames_core"] = _process_frames_core
    _loaded["debug"] = Debug(enabled=False)
    _loaded["caches"] = {}  # one runner_cache per dit_model id
    _loaded["download_weight"] = download_weight
    _loaded["DEFAULT_VAE"] = DEFAULT_VAE
    _loaded["downloaded"] = set()
    log.info("wrapper loaded; ready for inference")


def _ensure_weights(model: str):
    """Download VAE + DiT model to MODEL_DIR if not already present."""
    if model in _loaded["downloaded"]:
        return
    download_weight = _loaded["download_weight"]
    DEFAULT_VAE = _loaded["DEFAULT_VAE"]
    log.info("downloading weights: %s + %s", DEFAULT_VAE, model)
    download_weight(DEFAULT_VAE, MODEL_DIR)
    download_weight(model, MODEL_DIR)
    _loaded["downloaded"].add(model)


def _build_args(model: str, resolution: int, seed: int, batch_size: int) -> argparse.Namespace:
    """Replicate the argparse Namespace that the wrapper expects."""
    a = argparse.Namespace()
    a.input = "<unused>"
    a.output = "/tmp/out.png"
    a.output_format = "png"
    a.video_backend = "opencv"
    setattr(a, "10bit", False)
    a.model_dir = MODEL_DIR
    a.dit_model = model
    a.resolution = resolution
    a.max_resolution = 0
    a.batch_size = batch_size
    a.uniform_batch_size = False
    a.seed = seed
    a.skip_first_frames = 0
    a.load_cap = 0
    a.chunk_size = 0
    a.prepend_frames = 0
    a.temporal_overlap = 0
    a.color_correction = "lab"
    a.input_noise_scale = 0.0
    a.latent_noise_scale = 0.0
    a.cuda_device = "0"
    # CLI defaults: dit/vae are "none", tensor is "cpu". Caching requires
    # offload set to a real device, so use "cpu" for dit/vae too.
    a.dit_offload_device = "cpu"
    a.vae_offload_device = "cpu"
    a.tensor_offload_device = "cpu"
    a.blocks_to_swap = 0
    a.swap_io_components = False
    a.vae_encode_tiled = False
    a.vae_encode_tile_size = 512
    a.vae_encode_tile_overlap = 64
    a.vae_decode_tiled = False
    a.vae_decode_tile_size = 512
    a.vae_decode_tile_overlap = 64
    a.tile_debug = "false"
    a.attention_mode = "sdpa"
    a.compile_dit = False
    a.compile_vae = False
    a.compile_backend = "inductor"
    a.compile_mode = "default"
    a.compile_fullgraph = False
    a.compile_dynamic = False
    a.compile_dynamo_cache_size_limit = 64
    a.compile_dynamo_recompile_limit = 16
    a.cache_dit = True
    a.cache_vae = True
    a.debug = False
    return a


def _decode_image(payload: str) -> Image.Image:
    if payload.startswith(("http://", "https://")):
        r = requests.get(payload, timeout=60)
        r.raise_for_status()
        return Image.open(io.BytesIO(r.content)).convert("RGB")
    if "," in payload and payload[:32].lower().startswith("data:"):
        payload = payload.split(",", 1)[1]
    return Image.open(io.BytesIO(base64.b64decode(payload))).convert("RGB")


def _encode_image(img: Image.Image, fmt: str = "PNG", quality: int = 92) -> str:
    fmt = fmt.upper()
    if fmt == "JPEG" and img.mode == "RGBA":
        img = img.convert("RGB")
    buf = io.BytesIO()
    if fmt == "JPEG":
        img.save(buf, format="JPEG", quality=quality, optimize=True)
    else:
        img.save(buf, format=fmt, optimize=False)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _maybe_downscale(img: Image.Image, spec: Any) -> Image.Image:
    if not spec or not isinstance(spec, dict):
        return img
    w, h = img.size
    if "width" in spec and "height" in spec:
        tw, th = int(spec["width"]), int(spec["height"])
    elif "max_side" in spec:
        s = float(spec["max_side"]) / max(w, h)
        tw, th = int(w * s), int(h * s)
    elif "scale" in spec:
        s = float(spec["scale"])
        tw, th = int(w * s), int(h * s)
    else:
        return img
    if (tw, th) == (w, h):
        return img
    log.info("downscaling input %dx%d -> %dx%d", w, h, tw, th)
    return img.resize((tw, th), Image.LANCZOS)


def _resolve_short_side(img: Image.Image, inp: Dict[str, Any]) -> int:
    in_w, in_h = img.size
    short_in = min(in_w, in_h)
    if "output_width" in inp and "output_height" in inp:
        ow, oh = int(inp["output_width"]), int(inp["output_height"])
        ratio = ow / in_w if in_w <= in_h else oh / in_h
        return max(64, int(short_in * ratio))
    if "output_scale" in inp:
        return max(64, int(short_in * float(inp["output_scale"])))
    if "output_max_side" in inp:
        long_in = max(in_w, in_h)
        s = float(inp["output_max_side"]) / long_in
        return max(64, int(short_in * s))
    return max(64, short_in * 2)


def handler(event: Dict[str, Any]) -> Dict[str, Any]:
    t0 = time.time()
    inp = event.get("input") or {}
    if "image" not in inp:
        return {"error": "missing 'image' (base64 or URL)"}

    try:
        img = _decode_image(inp["image"])
    except Exception as e:
        return {"error": f"failed to decode image: {e}"}

    img = _maybe_downscale(img, inp.get("downscale"))
    short_target = _resolve_short_side(img, inp)
    seed = int(inp.get("seed", 42))
    batch_size = int(inp.get("batch_size", 1))
    model = inp.get("model", DEFAULT_MODEL)
    if model not in ALLOWED_MODELS:
        return {"error": f"unknown model {model!r}"}
    out_fmt = (inp.get("format") or "PNG").upper()
    quality = int(inp.get("quality", 92))

    log.info("input=%dx%d short_target=%d model=%s seed=%d",
             img.width, img.height, short_target, model, seed)

    # Heavy imports + cache setup happen on first request.
    t_load = time.time()
    try:
        _ensure_loaded()
    except Exception as e:
        log.exception("wrapper import failed")
        return {"error": f"wrapper import failed: {e}"}

    try:
        _ensure_weights(model)
    except Exception as e:
        log.exception("weight download failed")
        return {"error": f"weight download failed: {e}"}

    process_frames_core = _loaded["process_frames_core"]
    debug = _loaded["debug"]
    cache = _loaded["caches"].setdefault(model, {})
    was_warm = bool(cache)

    # PIL -> torch tensor in [0,1], shape (T=1, H, W, C)
    import torch
    import numpy as np
    arr = np.asarray(img).astype(np.float32) / 255.0
    frames = torch.from_numpy(arr).unsqueeze(0)

    args = _build_args(model=model, resolution=short_target, seed=seed,
                       batch_size=batch_size)

    t_inf = time.time()
    try:
        out_tensor = process_frames_core(
            frames_tensor=frames, args=args, device_id="0",
            debug=debug, runner_cache=cache,
        )
    except Exception as e:
        log.exception("inference failed")
        return {"error": f"inference failed: {type(e).__name__}: {e}"}

    # out_tensor: (T, H, W, C) float32 in [0,1]
    out_arr = (out_tensor[0].clamp(0, 1).cpu().numpy() * 255).astype("uint8")
    out_img = Image.fromarray(out_arr)

    t_done = time.time()
    encoded = _encode_image(out_img, fmt=out_fmt, quality=quality)
    return {
        "image_base64": encoded,
        "format": out_fmt,
        "width": out_img.width,
        "height": out_img.height,
        "model": model,
        "timings": {
            "total_s": round(t_done - t0, 3),
            "load_s": round(t_inf - t_load, 3),
            "inference_s": round(t_done - t_inf, 3),
        },
        "warm": "warm" if was_warm else "cold",
    }


if __name__ == "__main__":
    # Eager-import wrapper now so PyTorch/CUDA init happens once at worker
    # startup rather than mid-request. Avoids a "CUDAAllocatorConfig" ABI
    # assert that fires when the wrapper's import path runs after a fresh
    # torch import in the same process.
    try:
        _ensure_loaded()
    except Exception:
        log.exception("startup _ensure_loaded failed; will retry per-request")
    runpod.serverless.start({"handler": handler})
