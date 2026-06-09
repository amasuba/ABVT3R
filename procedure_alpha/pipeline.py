"""
procedure_alpha/pipeline.py
=============================
Classical depth-based 3D reconstruction pipeline — Procedure Alpha.

Implements the Level-1 baseline described in Chapter 3 of the thesis:
  Depth → Point Cloud → Preprocessing → Registration → Reconstruction → Features

Fully generalised for N-view captures at any angular step.  Natively reads
the new 30-degree dataset layout while remaining backward-compatible with
the legacy 4-view (90°) data_collection/ directory.

Usage
-----
    # New 30-degree dataset
    from procedure_alpha.pipeline import ProcedureAlpha
    pa = ProcedureAlpha()
    pa.run_specimen("DG041_20260609_B02")

    # Legacy plant_1 data
    pa.run_legacy(plant_id=1)
"""

import sys
import os
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1]))

import time
import json
from pathlib import Path
from typing import Callable

import numpy as np

try:
    import matplotlib.pyplot as plt
    import open3d as o3d
except ImportError:
    plt = None
    o3d = None

from procedure_alpha.preprocessing  import PreProcessing
from procedure_alpha.registration   import Registration
from procedure_alpha.reconstruction import ThreeDReconstruction
from shared.config import (
    KINECT_FX, KINECT_FY, KINECT_CX, KINECT_CY,
    ROI_X_MIN, ROI_X_MAX, ROI_Y_MIN, ROI_Y_MAX, ROI_Z_MIN, ROI_Z_MAX,
    ICP_MAX_ITER, ICP_TOLERANCE, ICP_MAX_CORR_DIST_M,
    VOXEL_SIZE_M,
    CAPTURE_ANGLES_DEG, LEGACY_ANGLES_DEG,
    SPECIMENS_DIR, RECON_OUTPUTS_DIR,
    legacy_depth_path, view_filename,
)


class ProcedureAlpha:
    """
    Classical RGB-D reconstruction and biomass feature extraction pipeline.

    Corresponds to Odwa Nombambela's 3D reconstruction methodology and
    forms the Level-1 baseline for the ABVT3R thesis.
    """

    def __init__(self, progress_callback: Callable = None):
        self.icp_params = dict(
            max_iterations = ICP_MAX_ITER,
            tolerance      = ICP_TOLERANCE,
            max_corr_dist  = ICP_MAX_CORR_DIST_M,
        )
        self.fx, self.fy = KINECT_FX, KINECT_FY
        self.cx, self.cy = KINECT_CX, KINECT_CY

        self.preprocessor   = PreProcessing()
        self.registrar      = Registration()
        self.reconstructor  = ThreeDReconstruction(verbose=True)
        self.progress_cb    = progress_callback

        RECON_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # Progress reporting
    # -------------------------------------------------------------------------

    def _progress(self, msg: str, pct: int = None):
        print(f"[Alpha] {msg}" + (f"  ({pct}%)" if pct else ""))
        if self.progress_cb:
            self.progress_cb(msg, pct)

    # -------------------------------------------------------------------------
    # Data loaders
    # -------------------------------------------------------------------------

    def _load_specimen_depths(self,
                               specimen_id: str,
                               angles_deg: list[int],
                               cam_label: str = "A") -> list[np.ndarray]:
        """
        Load depth maps from acquisition/dataset/specimens/{specimen_id}/depth/.
        Returns a list of uint16 arrays in capture order.
        """
        spec_depth = SPECIMENS_DIR / specimen_id / "depth"
        depths = []
        for a in angles_deg:
            fname = view_filename(a, cam_label, "depth", "npy")
            p = spec_depth / fname
            if not p.exists():
                raise FileNotFoundError(f"Depth file missing: {p}")
            depths.append(np.load(str(p)))
        return depths

    def _load_legacy_depths(self, plant_id: int) -> tuple[list[np.ndarray], list[int]]:
        """Load depths from the old data_collection/ flat directory."""
        depths = []
        for a in LEGACY_ANGLES_DEG:
            p = legacy_depth_path(a, plant_id)
            if not p.exists():
                raise FileNotFoundError(f"Legacy depth missing: {p}")
            depths.append(np.load(str(p)))
        return depths, LEGACY_ANGLES_DEG

    # -------------------------------------------------------------------------
    # Core pipeline
    # -------------------------------------------------------------------------

    def _preprocess_all(self, depth_maps: list[np.ndarray]) -> list[np.ndarray]:
        """Run preprocessing pipeline on every depth map."""
        clouds = []
        for i, dm in enumerate(depth_maps):
            pts, _, _ = self.preprocessor.complete_preprocessing_pipeline(
                dm,
                self.fx, self.fy, self.cx, self.cy,
                ROI_X_MIN, ROI_X_MAX,
                ROI_Y_MIN, ROI_Y_MAX,
                ROI_Z_MIN, ROI_Z_MAX,
            )
            print(f"[Alpha]   View {i:2d}: {len(pts):,} pts after preprocessing")
            clouds.append(pts)
        return clouds

    def _register(self,
                  clouds:     list[np.ndarray],
                  angles_deg: list[int]) -> tuple[list[np.ndarray], list, list]:
        """Coarse + sequential ICP registration."""
        angles_rad = [np.deg2rad(a) for a in angles_deg]

        # Coarse
        coarse, _ = self.registrar.arrange_views_in_circle(
            clouds, angles_rad, radius=0.13)

        # Fine
        fine, transforms, stats = self.registrar.sequential_icp_registration(
            coarse, self.icp_params, angles_deg=angles_deg)

        return fine, transforms, stats

    def _reconstruct(self, fine_pcs: list[np.ndarray]) -> dict:
        params = dict(
            voxel_size       = VOXEL_SIZE_M,
            hole_threshold   = 0,
            smooth_iterations= 0,
            preserve_features= True,
            fill_holes       = False,
        )
        return self.reconstructor.complete_reconstruction_pipeline(
            fine_pcs, method='grid_based', **params)

    # -------------------------------------------------------------------------
    # Output / save
    # -------------------------------------------------------------------------

    def _save_outputs(self, label: str, recon: dict, reg_stats: list) -> Path:
        """
        Save all outputs for a processed specimen to procedure_alpha/outputs/.

        Returns the path to the stats text file.
        """
        out = RECON_OUTPUTS_DIR
        vertices  = recon['final_vertices']
        triangles = recon['final_triangles']
        merged    = recon['merged_cloud']

        np.save(str(out / f"final_vertices_{label}.npy"),  vertices)
        np.save(str(out / f"final_triangles_{label}.npy"), triangles)
        np.save(str(out / f"merged_points_{label}.npy"),   merged)

        # Surface normals if available
        if 'surface_normals' in recon:
            np.save(str(out / f"surface_normals_{label}.npy"), recon['surface_normals'])

        # Write stats text
        stats = recon['reconstruction_stats']
        mq    = recon['mesh_quality']
        sq    = recon['surface_quality']

        x, y, z = vertices[:, 0], vertices[:, 1], vertices[:, 2]
        width  = (x.max() - x.min()) * 100   # cm
        height = (y.max() - y.min()) * 100
        depth  = (z.max() - z.min()) * 100

        stats_path = out / f"reconstruction_stats_{label}.txt"
        with stats_path.open("w") as f:
            f.write(f"=== Procedure Alpha — Reconstruction Stats ===\n")
            f.write(f"Label                : {label}\n")
            f.write(f"Merged points        : {stats.get('merged_points', len(merged)):,}\n")
            f.write(f"Final vertices       : {len(vertices):,}\n")
            f.write(f"Final triangles      : {len(triangles):,}\n")
            f.write(f"Surface area         : {mq.get('surface_area', 0):.4f} m²\n")
            f.write(f"Volume               : {mq.get('volume', 0):.6f} m³\n")
            f.write(f"Height               : {height:.1f} cm\n")
            f.write(f"Width                : {width:.1f} cm\n")
            f.write(f"Depth                : {depth:.1f} cm\n")
            f.write(f"Mesh manifold        : {mq.get('is_manifold', '?')}\n")
            f.write(f"Mesh watertight      : {mq.get('is_watertight', '?')}\n")
            f.write(f"Overall quality Q    : {sq.get('overall_quality', 0):.4f}\n")
            f.write(f"  Geometric fidelity : {sq.get('geometric_fidelity', 0):.4f}\n")
            f.write(f"  Coverage balance   : {sq.get('coverage_balance', 0):.4f}\n")
            f.write(f"  Surface smoothness : {sq.get('surface_smoothness', 0):.4f}\n")
            f.write(f"  Manifold integrity : {sq.get('manifold_integrity', 0):.4f}\n")
            f.write("\n--- ICP Registration Stats ---\n")
            for rs in reg_stats:
                f.write(f"  {rs.get('view','?'):>8}  RMSE={rs.get('final_rmse_mm',0):.3f}mm  "
                        f"iters={rs.get('iterations',0)}  "
                        f"converged={rs.get('converged', False)}\n")

        # Optional Open3D mesh export
        if o3d is not None:
            try:
                mesh = o3d.geometry.TriangleMesh()
                mesh.vertices  = o3d.utility.Vector3dVector(vertices)
                mesh.triangles = o3d.utility.Vector3iVector(triangles)
                mesh.compute_vertex_normals()
                o3d.io.write_triangle_mesh(str(out / f"mesh_{label}.ply"), mesh)
                o3d.io.write_triangle_mesh(str(out / f"mesh_{label}.obj"), mesh)
            except Exception as e:
                print(f"[Alpha] Open3D mesh export failed: {e}")

        print(f"[Alpha] Outputs written to {out}/")
        return stats_path

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def run_specimen(self,
                     specimen_id: str,
                     angles_deg:  list[int] = None,
                     cam_label:   str = "A") -> dict:
        """
        Run the full pipeline for a new 30-degree dataset specimen.

        Parameters
        ----------
        specimen_id : e.g. ``DG041_20260609_B02``
        angles_deg  : subset of angles to use; defaults to all captured angles
        cam_label   : which camera to use for depth ("A" or "B")

        Returns
        -------
        results dict with keys: label, stats_path, reconstruction, reg_stats
        """
        t0 = time.time()
        self._progress(f"Loading specimen {specimen_id}", 5)

        # Discover angles from metadata if not provided
        if angles_deg is None:
            meta_path = SPECIMENS_DIR / specimen_id / "metadata.json"
            if meta_path.exists():
                angles_deg = json.loads(meta_path.read_text())["angles_deg"]
            else:
                angles_deg = CAPTURE_ANGLES_DEG

        depths = self._load_specimen_depths(specimen_id, angles_deg, cam_label)

        self._progress("Preprocessing depth maps", 15)
        clouds = self._preprocess_all(depths)

        self._progress("Registering point clouds (coarse + ICP)", 35)
        fine, transforms, reg_stats = self._register(clouds, angles_deg)

        self._progress("Reconstructing 3D mesh", 60)
        recon = self._reconstruct(fine)

        label = f"specimen_{specimen_id}"
        self._progress("Saving outputs", 85)
        stats_path = self._save_outputs(label, recon, reg_stats)

        elapsed = time.time() - t0
        self._progress(f"Done in {elapsed:.1f}s", 100)

        return dict(label=label, stats_path=stats_path,
                    reconstruction=recon, reg_stats=reg_stats,
                    elapsed_s=elapsed)

    def run_legacy(self, plant_id: int) -> dict:
        """
        Run the pipeline on a legacy 4-view (90°) plant from data_collection/.

        Parameters
        ----------
        plant_id : integer 1…40

        Returns
        -------
        results dict (same schema as run_specimen)
        """
        t0 = time.time()
        self._progress(f"Loading legacy plant_{plant_id}", 5)

        depths, angles_deg = self._load_legacy_depths(plant_id)

        self._progress("Preprocessing", 15)
        clouds = self._preprocess_all(depths)

        self._progress("Registration", 35)
        fine, transforms, reg_stats = self._register(clouds, angles_deg)

        self._progress("Reconstruction", 60)
        recon = self._reconstruct(fine)

        label = f"plant_{plant_id}"
        self._progress("Saving", 85)
        stats_path = self._save_outputs(label, recon, reg_stats)

        elapsed = time.time() - t0
        self._progress(f"Done in {elapsed:.1f}s", 100)

        return dict(label=label, stats_path=stats_path,
                    reconstruction=recon, reg_stats=reg_stats,
                    elapsed_s=elapsed)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Procedure Alpha — 3D Reconstruction Pipeline")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--specimen", metavar="ID",
                   help="New-protocol specimen ID, e.g. DG041_20260609_B02")
    g.add_argument("--legacy",   metavar="N", type=int,
                   help="Legacy plant number (1…40)")
    p.add_argument("--cam", default="A", choices=["A", "B"],
                   help="Camera label for depth (new protocol only)")
    args = p.parse_args()

    pa = ProcedureAlpha()
    if args.specimen:
        pa.run_specimen(args.specimen, cam_label=args.cam)
    else:
        pa.run_legacy(args.legacy)
