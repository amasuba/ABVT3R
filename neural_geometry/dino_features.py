#!/usr/bin/env python3
"""
neural_geometry/dino_features.py
===================================
Stage A: extract frozen DINOv2 embeddings for every specimen's RGB views.

Encodes each view with the pre-trained (frozen, not fine-tuned) DINOv2
ViT-B/14 backbone via DINOv2Encoder, mean-pools the per-view CLS tokens
into one 768-dim embedding per specimen (DINOv2Encoder.encode_multiview),
and caches both the aggregated embedding and the per-view CLS tokens to
disk. Nothing here is trained -- this is inference only, so it runs fine
on a small GPU or CPU.

Usage
-----
    python neural_geometry/dino_features.py            # all specimens
    python neural_geometry/dino_features.py --specimen M001
    python neural_geometry/dino_features.py --force     # re-encode even if cached
"""

import sys
import time
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image

from neural_geometry.backbone.dinov2_encoder import DINOv2Encoder
from shared.config import ACQUISITION_DIR, NEURAL_GEOMETRY_DIR

SPECIMENS_DIR = ACQUISITION_DIR / "dataset" / "specimens"
FEATURES_DIR  = NEURAL_GEOMETRY_DIR / "dino_features"


def load_views(specimen_id: str) -> list[np.ndarray]:
    rgb_dir = SPECIMENS_DIR / specimen_id / "rgb"
    paths = sorted(rgb_dir.glob("*.jpg"))
    return [np.array(Image.open(p).convert("RGB")) for p in paths], [p.name for p in paths]


def extract_all(specimen_ids: list[str] = None, force: bool = False):
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)

    if specimen_ids is None:
        specimen_ids = sorted(
            d.name for d in SPECIMENS_DIR.iterdir()
            if d.is_dir() and (d / "rgb").exists()
        )

    print(f"[DINOv2] Loading dinov2_vitb14 (frozen)...")
    enc = DINOv2Encoder(model_name="dinov2_vitb14", freeze=True)
    if not enc.load_pretrained():
        print("[DINOv2] Failed to load backbone -- aborting")
        return

    for specimen_id in specimen_ids:
        out_path = FEATURES_DIR / f"{specimen_id}.npz"
        if out_path.exists() and not force:
            print(f"[DINOv2] {specimen_id}: cached, skipping")
            continue

        views, names = load_views(specimen_id)
        if not views:
            print(f"[DINOv2] {specimen_id}: no RGB views found, skipping")
            continue

        t0 = time.time()
        result = enc.encode_multiview(views)
        per_view_cls = np.stack(
            [pv["cls_token"].cpu().numpy().squeeze(0) for pv in result["per_view"]],
            axis=0,
        )  # (V, 768)
        aggregated = result["aggregated"].squeeze(0)  # (768,)

        np.savez(out_path, aggregated=aggregated, per_view=per_view_cls, view_names=names)
        elapsed = time.time() - t0
        print(f"[DINOv2] {specimen_id}: {len(views)} views -> {out_path.name} ({elapsed:.1f}s)")


def main():
    p = argparse.ArgumentParser(description="Extract frozen DINOv2 embeddings per specimen")
    p.add_argument("--specimen", action="append", help="Specific specimen ID(s); default: all")
    p.add_argument("--force", action="store_true", help="Re-encode even if a cached .npz exists")
    args = p.parse_args()
    extract_all(args.specimen, args.force)


if __name__ == "__main__":
    main()
