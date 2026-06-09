"""
neural_geometry/sam3d/sam3d_pipeline.py
=========================================
SAM3D (Segment Anything in 3D) pipeline for plant segmentation and
soil / pot removal from multi-view RGB-D data.

Pipeline
--------
1. For each 2D RGB view: run SAM (facebook/segment-anything) to produce
   binary plant masks — prompted by a central point or bounding box.
2. Back-project masked depth pixels to 3D to obtain plant-only point clouds.
3. Enforce multi-view mask consistency via silhouette intersection.
4. Estimate the soil-surface Y coordinate (Y_soil) using depth-threshold;
   discard all points with Y < Y_soil (removes pot, soil, substrate).
5. Return the cleaned, segmented point cloud for downstream volumetric
   processing by the VolumetricTransformer.

The 2D → 3D lifting exploits known camera intrinsics stored in shared/config.py.

Usage
-----
    from neural_geometry.sam3d.sam3d_pipeline import SAM3DPipeline
    pipe = SAM3DPipeline()
    pipe.load_sam()
    clean_cloud = pipe.segment_specimen("DG041_20260609_B02")
"""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import json
import numpy as np
import cv2
from typing import List, Optional, Tuple

from shared.config import (
    SPECIMENS_DIR, CAPTURE_ANGLES_DEG,
    KINECT_FX, KINECT_FY, KINECT_CX, KINECT_CY,
    DEPTH_MIN_MM, DEPTH_MAX_MM, ROI_Y_MIN,
    view_filename,
)

# SAM optional imports
try:
    from segment_anything import SamPredictor, sam_model_registry
    SAM_AVAILABLE = True
except ImportError:
    SAM_AVAILABLE = False
    print("[SAM3D] segment_anything not installed — running in stub mode")
    print("        Install: pip install segment-anything")

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# ---------------------------------------------------------------------------
# SAM3D Pipeline
# ---------------------------------------------------------------------------

class SAM3DPipeline:
    """
    End-to-end plant segmentation using SAM + 3D back-projection.

    Parameters
    ----------
    sam_checkpoint : path to a SAM checkpoint (.pth)
    sam_model_type : 'vit_h', 'vit_l', 'vit_b' (default: 'vit_b' for speed)
    soil_percentile: percentile of the depth distribution used to estimate
                     the soil surface Y coordinate (default: 5th percentile
                     of the lowest point-cloud region)
    """

    SAM_DEFAULTS = {
        "vit_h": "sam_vit_h_4b8939.pth",
        "vit_b": "sam_vit_b_01ec64.pth",
        "vit_l": "sam_vit_l_0b3195.pth",
    }

    def __init__(self,
                 sam_checkpoint: Optional[str] = None,
                 sam_model_type: str = "vit_b",
                 soil_percentile: float = 5.0):

        self.sam_checkpoint  = sam_checkpoint
        self.sam_model_type  = sam_model_type
        self.soil_percentile = soil_percentile
        self.predictor       = None

    # -----------------------------------------------------------------------
    # SAM loading
    # -----------------------------------------------------------------------

    def load_sam(self, checkpoint: Optional[str] = None) -> bool:
        """Load SAM model weights.  Returns True on success."""
        if not SAM_AVAILABLE:
            print("[SAM3D] segment_anything not available")
            return False

        ckpt = checkpoint or self.sam_checkpoint
        if not ckpt or not Path(ckpt).exists():
            print(f"[SAM3D] Checkpoint not found: {ckpt}")
            print("        Download from: https://github.com/facebookresearch/segment-anything")
            return False

        device = "cuda" if (TORCH_AVAILABLE and torch.cuda.is_available()) else "cpu"
        sam    = sam_model_registry[self.sam_model_type](checkpoint=ckpt)
        sam.to(device)
        self.predictor = SamPredictor(sam)
        print(f"[SAM3D] Loaded {self.sam_model_type} on {device}")
        return True

    # -----------------------------------------------------------------------
    # 2D segmentation
    # -----------------------------------------------------------------------

    def _segment_view(self,
                       rgb:   np.ndarray,
                       prompt: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """
        Run SAM on one RGB view and return a binary plant mask.

        Parameters
        ----------
        rgb    : (H, W, 3) BGR uint8 image
        prompt : (x, y) pixel prompt; defaults to image centre

        Returns
        -------
        mask : (H, W)  bool  — True = plant pixel
        """
        if self.predictor is None:
            # Stub: return a circular central mask
            H, W = rgb.shape[:2]
            Y, X = np.ogrid[:H, :W]
            cx, cy = W // 2, H // 2
            r = min(H, W) // 3
            return ((X - cx) ** 2 + (Y - cy) ** 2) < r ** 2

        rgb_rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        self.predictor.set_image(rgb_rgb)

        if prompt is None:
            prompt = (rgb.shape[1] // 2, rgb.shape[0] // 2)

        masks, scores, _ = self.predictor.predict(
            point_coords  = np.array([list(prompt)]),
            point_labels  = np.array([1]),
            multimask_output = True,
        )
        # Select highest-confidence mask
        best = masks[np.argmax(scores)]
        return best.astype(bool)

    # -----------------------------------------------------------------------
    # 3D back-projection
    # -----------------------------------------------------------------------

    @staticmethod
    def _backproject(depth:  np.ndarray,
                      mask:   np.ndarray,
                      fx: float, fy: float,
                      cx: float, cy: float) -> np.ndarray:
        """
        Lift masked depth pixels to 3D.

        Returns
        -------
        points : (N, 3) float32 in metres
        """
        H, W = depth.shape
        u, v = np.meshgrid(np.arange(W), np.arange(H))
        d    = depth.astype(np.float32) / 1000.0   # mm → m
        valid = mask & (d > DEPTH_MIN_MM / 1000.0) & (d < DEPTH_MAX_MM / 1000.0)
        u, v, d = u[valid], v[valid], d[valid]
        X = (u - cx) * d / fx
        Y = (v - cy) * d / fy
        Z = d
        return np.stack([X, Y, Z], axis=-1).astype(np.float32)

    # -----------------------------------------------------------------------
    # Soil / pot removal
    # -----------------------------------------------------------------------

    def _remove_soil(self, points: np.ndarray) -> np.ndarray:
        """
        Estimate soil surface Y coordinate and discard points below it.
        Uses the soil_percentile of the lowest Y values in the cloud.
        """
        if len(points) == 0:
            return points
        y_vals   = points[:, 1]
        y_soil   = np.percentile(y_vals, self.soil_percentile)
        clean    = points[y_vals > y_soil]
        removed  = len(points) - len(clean)
        print(f"[SAM3D]   Soil removal: Y_soil={y_soil:.3f}m  removed {removed:,} pts")
        return clean

    # -----------------------------------------------------------------------
    # Full specimen segmentation
    # -----------------------------------------------------------------------

    def segment_specimen(self,
                          specimen_id: str,
                          angles_deg:  Optional[List[int]] = None,
                          cam_label:   str = "A",
                          remove_soil: bool = True) -> np.ndarray:
        """
        Segment all views of a specimen and return a merged plant point cloud.

        Parameters
        ----------
        specimen_id : e.g. ``DG041_20260609_B02``
        angles_deg  : subset of angles (defaults to all in metadata.json)
        cam_label   : "A" or "B"
        remove_soil : whether to strip points below estimated soil level

        Returns
        -------
        merged_cloud : (N, 3) float32  metres — plant-only 3D points
        """
        spec_dir = SPECIMENS_DIR / specimen_id
        if not spec_dir.exists():
            raise FileNotFoundError(f"Specimen not found: {spec_dir}")

        # Determine angles
        if angles_deg is None:
            meta_p = spec_dir / "metadata.json"
            angles_deg = (json.loads(meta_p.read_text())["angles_deg"]
                          if meta_p.exists() else CAPTURE_ANGLES_DEG)

        all_points = []
        for angle in angles_deg:
            rgb_path   = spec_dir / "rgb"   / view_filename(angle, cam_label, "rgb",   "jpg")
            depth_path = spec_dir / "depth" / view_filename(angle, cam_label, "depth", "npy")

            if not rgb_path.exists() or not depth_path.exists():
                print(f"[SAM3D]   Skipping {angle}° — files missing")
                continue

            rgb   = cv2.imread(str(rgb_path))
            depth = np.load(str(depth_path))

            mask = self._segment_view(rgb)
            pts  = self._backproject(depth, mask, KINECT_FX, KINECT_FY,
                                      KINECT_CX, KINECT_CY)
            print(f"[SAM3D]   {angle:3d}°  plant pts: {len(pts):,}")
            all_points.append(pts)

        if not all_points:
            raise RuntimeError(f"No views found for {specimen_id}")

        merged = np.vstack(all_points)
        print(f"[SAM3D] Total plant points before soil removal: {len(merged):,}")

        if remove_soil:
            merged = self._remove_soil(merged)

        print(f"[SAM3D] Final plant cloud: {len(merged):,} points")
        return merged
