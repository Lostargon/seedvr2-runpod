"""RunPod serverless handler for SeedVR2-7B image upscaling."""

from __future__ import annotations

import base64
import io
import logging
import time
from typing import Any, Dict

import requests
import runpod
from PIL import Image

from seedvr_runner import get_runner

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("seedvr")


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


def _encode_image(img: Image.Image, fmt: str = "PNG") -> str:
    buf = io.BytesIO()
    img.save(buf, format=fmt, optimize=False)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _maybe_downscale(img: Image.Image, spec: Any) -> Image.Image:
    """``spec`` can be:
      - {"width": W, "height": H}
      - {"max_side": N}            -> longest side becomes N
      - {"scale": 0.5}             -> multiplicative factor
      - None / falsy               -> no-op
    """
    if not spec:
        return img
    w, h = img.size
    if isinstance(spec, dict):
        if "width" in spec and "height" in spec:
            tw, th = int(spec["width"]), int(spec["height"])
        elif "max_side" in spec:
            longest = max(w, h)
            scale = float(spec["max_side"]) / longest
            tw, th = int(w * scale), int(h * scale)
        elif "scale" in spec:
            scale = float(spec["scale"])
            tw, th = int(w * scale), int(h * scale)
        else:
            return img
    else:
        return img

    if (tw, th) == (w, h):
        return img
    log.info("Downscaling input %dx%d -> %dx%d", w, h, tw, th)
    return img.resize((tw, th), Image.LANCZOS)


def _resolve_output_size(img: Image.Image, inp: Dict[str, Any]) -> tuple[int, int]:
    in_w, in_h = img.size
    if "output_width" in inp and "output_height" in inp:
        return int(inp["output_width"]), int(inp["output_height"])
    if "output_scale" in inp:
        s = float(inp["output_scale"])
        return int(in_w * s), int(in_h * s)
    if "output_max_side" in inp:
        longest = max(in_w, in_h)
        s = float(inp["output_max_side"]) / longest
        return int(in_w * s), int(in_h * s)
    # Default: 4x upscale
    return in_w * 4, in_h * 4


# ----------------------------------------------------------------------
# Handler
# ----------------------------------------------------------------------
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

    target_w, target_h = _resolve_output_size(img, inp)
    seed = int(inp.get("seed", 42))
    steps = int(inp.get("steps", 1))
    cfg = float(inp.get("cfg_scale", 1.0))
    fmt = (inp.get("format") or "PNG").upper()

    log.info(
        "input=%dx%d target=%dx%d seed=%d steps=%d",
        img.width, img.height, target_w, target_h, seed, steps,
    )

    runner = get_runner()
    t_load = time.time()

    try:
        out = runner.upscale(
            img,
            target_size=(target_w, target_h),
            seed=seed,
            steps=steps,
            cfg_scale=cfg,
        )
    except Exception as e:
        log.exception("upscale failed")
        return {"error": f"upscale failed: {e}"}

    t_done = time.time()
    encoded = _encode_image(out, fmt=fmt)
    return {
        "image_base64": encoded,
        "format": fmt,
        "width": out.width,
        "height": out.height,
        "timings": {
            "total_s": round(t_done - t0, 3),
            "model_s": round(t_done - t_load, 3),
        },
    }


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
