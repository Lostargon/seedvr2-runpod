FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/runpod-volume/hf-cache \
    TRANSFORMERS_CACHE=/runpod-volume/hf-cache \
    SEEDVR_MODEL_DIR=/runpod-volume/seedvr/models

RUN apt-get update && apt-get install -y --no-install-recommends \
        git git-lfs ffmpeg libgl1 libglib2.0-0 ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && git lfs install

WORKDIR /app

# SeedVR2 source (used as a library inside the handler)
RUN git clone --depth=1 https://github.com/ByteDance-Seed/SeedVR.git /app/SeedVR

COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install -r /app/requirements.txt

# flash-attn needs the CUDA dev toolchain (present in the -devel base image).
# Pin a wheel-friendly version; build from sdist as a fallback.
RUN pip install --no-build-isolation flash-attn==2.6.3 || true

COPY handler.py /app/handler.py
COPY seedvr_runner.py /app/seedvr_runner.py
COPY download_to_volume.py /app/download_to_volume.py

ENV PYTHONPATH=/app:/app/SeedVR

CMD ["python", "-u", "handler.py"]
