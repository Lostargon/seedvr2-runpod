"""Download SeedVR2-7B weights into the network volume.

Run this ONCE inside a RunPod GPU pod (or any machine that has the volume
mounted at ``/runpod-volume``). After it finishes, the serverless workers
load weights from the volume — no per-cold-start downloads.

Usage on the pod:
    HF_TOKEN=... python download_to_volume.py
"""

from __future__ import annotations

import os
from pathlib import Path

from huggingface_hub import snapshot_download


REPO_ID = "ByteDance-Seed/SeedVR2-7B"
TARGET = Path(os.environ.get("SEEDVR_MODEL_DIR", "/runpod-volume/seedvr/models"))


def main() -> None:
    TARGET.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {REPO_ID} -> {TARGET}")

    snapshot_download(
        repo_id=REPO_ID,
        local_dir=str(TARGET),
        local_dir_use_symlinks=False,
        token=os.environ.get("HF_TOKEN") or os.environ.get("HF"),
        # Repo is ~67 GB total because it contains a 33 GB *_sharp.pth alt
        # checkpoint we do NOT need for standard inference. Skip it.
        allow_patterns=[
            "seedvr2_ema_7b.pth",
            "ema_vae.pth",
            "*.json",
            "*.yaml",
            "*.txt",
            "README.md",
        ],
    )

    # SeedVR2 also needs a T5 / pos+neg prompt embeddings; the upstream repo
    # bundles those in the same release. If not present, fetch the auxiliary
    # repo (adjust here if upstream splits things differently).
    print("Done. Files:")
    for p in sorted(TARGET.iterdir()):
        size = p.stat().st_size / (1024 * 1024)
        print(f"  {p.name:40s} {size:8.1f} MB")


if __name__ == "__main__":
    main()
