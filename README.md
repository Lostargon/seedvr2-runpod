# SeedVR2-7B RunPod Worker

Docker image for [SeedVR2-7B](https://github.com/ByteDance-Seed/SeedVR)
image upscaling, designed to run as a RunPod serverless worker on A100.

This repo contains **only** the build artefacts — no API keys, no
RunPod-account tooling. Build it, push it, point a serverless endpoint at it.

## Files

| File | Purpose |
|---|---|
| `Dockerfile` | Based on `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04` |
| `requirements.txt` | Python deps |
| `handler.py` | Serverless entrypoint — input downscale, output sizing, b64/URL I/O |
| `seedvr_runner.py` | Wraps SeedVR2 inference for single images |
| `download_to_volume.py` | One-shot: pre-fetch model weights into a network volume |
| `.github/workflows/build.yml` | Builds & pushes to Docker Hub on every `main` push |

## Build via GitHub Actions

One-time secrets on the repo:

```bash
# Docker Hub access token: https://hub.docker.com/settings/security
gh secret set DOCKERHUB_USERNAME --body "lostargon"
gh secret set DOCKERHUB_TOKEN    --body "<access-token>"
```

Then push to `main` (or trigger manually):

```bash
gh workflow run build.yml
```

Build runs ~6–10 min on a free `ubuntu-latest` runner. Images:

- `lostargon/seedvr2-runpod:latest`
- `lostargon/seedvr2-runpod:<git-sha>`

## Build locally (alternative)

```bash
docker buildx build --platform linux/amd64 \
    -t lostargon/seedvr2-runpod:latest \
    --push .
```

## Seed the network volume

Once the image exists, attach a RunPod network volume to any cheap GPU pod
and run:

```bash
HF_TOKEN=hf_xxx python download_to_volume.py
```

This drops ~14–25 GB of weights into `/runpod-volume/seedvr/models`. The
serverless workers then load weights from the volume, no per-cold-start
download.

## Endpoint API

`POST https://api.runpod.ai/v2/<endpoint-id>/run` (async) or `/runsync`.

```json
{
  "input": {
    "image": "<base64 or http(s) URL>",
    "downscale": {"max_side": 1024},
    "output_width": 4096,
    "output_height": 4096,
    "output_scale": 4,
    "output_max_side": 4096,
    "seed": 42,
    "steps": 1,
    "cfg_scale": 1.0,
    "format": "PNG"
  }
}
```

Sizing precedence:
- **Input downscale**: `width+height` > `max_side` > `scale` (omit for no-op)
- **Output size**: `output_width+output_height` > `output_scale` > `output_max_side` > default 4×

Response:

```json
{
  "image_base64": "...",
  "format": "PNG",
  "width": 4096,
  "height": 4096,
  "timings": {"total_s": 22.4, "model_s": 21.7}
}
```
