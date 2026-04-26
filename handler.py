"""RunPod serverless handler for SeedVR2 image upscaling.

Wraps numz/ComfyUI-SeedVR2_VideoUpscaler's standalone CLI as a subprocess.
The CLI auto-downloads the chosen model on first call and caches it on
the network volume at ``$SEEDVR_MODEL_DIR``.
"""

from __future__ import annotations

import base64
import io
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict

import requests
import runpod
from PIL import Image


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("seedvr")

CLI_DIR = Path("/app/seedvr-cli")
CLI_SCRIPT = CLI_DIR / "inference_cli.py"
MODEL_DIR = Path(os.environ.get("SEEDVR_MODEL_DIR", "/runpod-volume/seedvr/SEEDVR2"))

DEFAULT_MODEL = "seedvr2_ema_7b_fp16.safetensors"
ALLOWED_MODELS = {
    "seedvr2_ema_3b_fp8_e4m3fn.safetensors",
    "seedvr2_ema_3b_fp16.safetensors",
    "seedvr2_ema_7b_fp8_e4m3fn_mixed_block35_fp16.safetensors",
    "seedvr2_ema_7b_fp16.safetensors",
    "seedvr2_ema_7b_sharp_fp8_e4m3fn_mixed_block35_fp16.safetensors",
    "seedvr2_ema_7b_sharp_fp16.safetensors",
}


# ----------------------------------------------------------------------
# I/O helpers
# ----------------------------------------------------------------------
def _decode_image(payload: str) -> Image.Image:
    if payload.startswith(("http://", "https://")):
        r = requests.get(payload, timeout=60)
        r.raise_for_status()
        return Image.open(io.BytesIO(r.content)).convert("RGB")
    if "," in payload and payload[:32].lower().startswith("data:"):
        payload = payload.split(",", 1)[1]
    return Image.open(io.BytesIO(base64.b64decode(payload))).convert("RGB")


def _encode_image(path: Path, fmt: str = "PNG") -> str:
    img = Image.open(path)
    if fmt.upper() == "JPEG" and img.mode == "RGBA":
        img = img.convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format=fmt, optimize=False)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _maybe_downscale(img: Image.Image, spec: Any) -> Image.Image:
    """``spec``: {"width":W,"height":H} | {"max_side":N} | {"scale":f} | None"""
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
    log.info("Downscaling input %dx%d -> %dx%d", w, h, tw, th)
    return img.resize((tw, th), Image.LANCZOS)


def _resolve_short_side(img: Image.Image, inp: Dict[str, Any]) -> int:
    """The CLI takes ``--resolution`` = target SHORT-side in pixels.

    Map our existing API to that single integer.
    """
    in_w, in_h = img.size
    short_in = min(in_w, in_h)
    if "output_width" in inp and "output_height" in inp:
        # Pick whichever maps to the short side proportionally.
        ow, oh = int(inp["output_width"]), int(inp["output_height"])
        ratio = ow / in_w if in_w <= in_h else oh / in_h
        return max(64, int(short_in * ratio))
    if "output_scale" in inp:
        return max(64, int(short_in * float(inp["output_scale"])))
    if "output_max_side" in inp:
        long_in = max(in_w, in_h)
        s = float(inp["output_max_side"]) / long_in
        return max(64, int(short_in * s))
    # Default: 2x
    return max(64, short_in * 2)


# ----------------------------------------------------------------------
# Handler
# ----------------------------------------------------------------------
def _run_cli(in_path: Path, out_path: Path, *, resolution: int, model: str,
             seed: int, batch_size: int) -> None:
    cmd = [
        sys.executable, str(CLI_SCRIPT), str(in_path),
        "--output", str(out_path),
        "--output_format", "png",
        "--resolution", str(resolution),
        "--batch_size", str(batch_size),
        "--seed", str(seed),
        "--model_dir", str(MODEL_DIR),
        "--dit_model", model,
    ]
    log.info("running: %s", " ".join(cmd))
    proc = subprocess.run(cmd, cwd=str(CLI_DIR), capture_output=True, text=True)
    if proc.returncode != 0:
        log.error("CLI failed (rc=%d):\nSTDOUT:\n%s\nSTDERR:\n%s",
                  proc.returncode, proc.stdout[-4000:], proc.stderr[-4000:])
        raise RuntimeError(f"inference_cli failed: {proc.stderr.splitlines()[-1] if proc.stderr else 'no output'}")


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
        return {"error": f"unknown model {model!r}; allowed: {sorted(ALLOWED_MODELS)}"}
    out_fmt = (inp.get("format") or "PNG").upper()

    log.info("input=%dx%d short_target=%d model=%s seed=%d",
             img.width, img.height, short_target, model, seed)

    with tempfile.TemporaryDirectory() as tmp:
        tmpd = Path(tmp)
        in_path = tmpd / "in.png"
        out_path = tmpd / "out.png"
        img.save(in_path, format="PNG")

        t_run = time.time()
        try:
            _run_cli(in_path, out_path, resolution=short_target,
                     model=model, seed=seed, batch_size=batch_size)
        except Exception as e:
            return {"error": str(e)}

        if not out_path.exists():
            return {"error": "CLI exited 0 but no output file was written"}

        t_done = time.time()
        encoded = _encode_image(out_path, fmt=out_fmt)
        with Image.open(out_path) as o:
            ow, oh = o.size

    return {
        "image_base64": encoded,
        "format": out_fmt,
        "width": ow,
        "height": oh,
        "model": model,
        "timings": {
            "total_s": round(t_done - t0, 3),
            "inference_s": round(t_done - t_run, 3),
        },
    }


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
