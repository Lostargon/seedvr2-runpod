FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    SEEDVR_MODEL_DIR=/runpod-volume/seedvr/SEEDVR2

RUN apt-get update && apt-get install -y --no-install-recommends \
        git ffmpeg libgl1 libglib2.0-0 ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Use numz/ComfyUI-SeedVR2_VideoUpscaler which bundles a working standalone
# inference CLI (no apex, runs on Python 3.11, supports single images).
RUN git clone --depth=1 https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler.git /app/seedvr-cli

COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install -r /app/requirements.txt

COPY handler.py /app/handler.py

CMD ["python", "-u", "handler.py"]
