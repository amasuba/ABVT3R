#!/usr/bin/env python3
"""
neural_geometry/nerf/build_transforms.py
==========================================
Build a Nerfstudio-compatible transforms.json for a dual-camera 6-step
specimen, using camera poses *derived from the rig's own assumed geometry*
rather than a calibration or COLMAP solve.

Why this works without calibration
-----------------------------------
procedure_alpha's coarse registration (Registration.arrange_views_in_circle)
already assumes: each view was captured with the plant rotated by a known
angle in front of a fixed camera. It aligns views by centering each view's
point cloud on its own centroid, then rotating by that view's true angle:

    world_point = R_y(angle) @ (cam_point - centroid)

That is exactly a camera-to-world transform in disguise:

    R_c2w = R_y(angle)
    t_c2w = -R_y(angle) @ centroid

This module derives the same R_c2w/t_c2w per view directly from each
view's own preprocessed point cloud (no assumed radius, no checkerboards),
then converts from this project's point-cloud axis convention (X right,
Y up, Z *into* the scene) to Nerfstudio's (X right, Y up, Z *out of* the
scene, camera looks down -Z) with a single Z-axis flip:

    R_nerf = R_y(-angle)
    t_nerf = -R_nerf @ (centroid with Z negated)

Usage
-----
    python neural_geometry/nerf/build_transforms.py --specimen M001
"""

import sys
import argparse
import shutil
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from shared.config import (
    REPO_ROOT, SPECIMENS_DIR, HALF_SWEEP_ANGLES_DEG,
    KINECT_FX, KINECT_FY, KINECT_CX, KINECT_CY,
    DEPTH_WIDTH, DEPTH_HEIGHT,
)
from procedure_alpha.pipeline import ProcedureAlpha

EXTERNAL_PLANTS_DIR = REPO_ROOT / "dataset" / "plants"
NERF_DATA_DIR       = REPO_ROOT / "neural_geometry" / "nerf_data"


def _view_entries(half_angles_deg):
    """(camera_label, loop_angle, true_angle) for all 12 views, matching
    the exact order ProcedureAlpha._load_specimen_depths_dual uses."""
    entries = [("A", a, a) for a in half_angles_deg]
    entries += [("B", a, (a + 180) % 360) for a in half_angles_deg]
    return entries


def build_assumed_transforms(specimen_id: str, half_angles_deg=None) -> Path:
    """Write transforms.json + copy source images for one specimen.

    Uses BOTH stages of procedure_alpha's registration, not just the coarse
    one: coarse (per-view centroid + known rotation angle) *composed with*
    the fine ICP correction that follows it. An earlier version of this
    script only used the coarse stage — that's fine for point-cloud fusion
    (ICP cleans up the residual error afterwards), but nerfacto has no
    equivalent self-correction for pose errors of that magnitude, and a
    50k-iteration training run confirmed it: point-cloud extent got *worse*
    with more training (more iterations trying, and failing, to reconcile
    multi-view photometric constraints from poses off by the coarse-stage
    residual), not better. Composing in the ICP correction fixes this.
    """
    half_angles_deg = half_angles_deg or HALF_SWEEP_ANGLES_DEG

    pa = ProcedureAlpha()
    depths, true_angles = pa._load_specimen_depths_dual(specimen_id, half_angles_deg)
    clouds = pa._preprocess_all(depths)
    _, icp_transforms, _ = pa._register(clouds, true_angles)

    entries = _view_entries(half_angles_deg)
    assert [e[2] for e in entries] == true_angles, "view ordering mismatch"

    out_dir = NERF_DATA_DIR / f"{specimen_id}_assumed"
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    src_plant_dir = EXTERNAL_PLANTS_DIR / specimen_id / "images"

    M = np.diag([1.0, 1.0, -1.0])  # our convention -> Nerfstudio convention

    frames = []
    for (cam, loop_angle, true_angle), cloud, icp in zip(entries, clouds, icp_transforms):
        centroid = cloud.mean(axis=0)

        theta = np.deg2rad(true_angle)
        c, s = np.cos(theta), np.sin(theta)
        R_coarse = np.array([[c, 0, s],
                              [0, 1, 0],
                              [-s, 0, c]])

        # Compose coarse + fine ICP (fine_point = R_icp @ coarse_point + t_icp,
        # coarse_point = R_coarse @ (raw_point - centroid)):
        R_full = icp['R'] @ R_coarse
        t_full = -R_full @ centroid + icp['t']

        # Convert (X right, Y up, Z into scene) -> Nerfstudio (X right, Y up,
        # Z out of scene) via a single Z-axis flip conjugation.
        R_nerf = M @ R_full @ M
        t_nerf = M @ t_full

        c2w = np.eye(4)
        c2w[:3, :3] = R_nerf
        c2w[:3, 3]  = t_nerf

        src_img = src_plant_dir / f"cam{cam}_{loop_angle:03d}.png"
        dst_name = f"cam{cam}_{loop_angle:03d}.png"
        if not src_img.exists():
            raise FileNotFoundError(f"Missing source image: {src_img}")
        shutil.copy(src_img, img_dir / dst_name)

        frames.append({
            "file_path": f"images/{dst_name}",
            "transform_matrix": c2w.tolist(),
        })

    transforms = {
        "camera_model": "OPENCV",
        "fl_x": KINECT_FX, "fl_y": KINECT_FY,
        "cx": KINECT_CX,   "cy": KINECT_CY,
        "w": DEPTH_WIDTH,  "h": DEPTH_HEIGHT,
        "k1": 0.0, "k2": 0.0, "p1": 0.0, "p2": 0.0,
        "frames": frames,
    }

    transforms_path = out_dir / "transforms.json"
    transforms_path.write_text(json.dumps(transforms, indent=2))
    print(f"[NeRF] Wrote {len(frames)} frames -> {transforms_path}")
    return transforms_path


def main():
    p = argparse.ArgumentParser(description="Build assumed-geometry transforms.json for Nerfstudio")
    p.add_argument("--specimen", required=True)
    args = p.parse_args()
    build_assumed_transforms(args.specimen)


if __name__ == "__main__":
    main()
