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
    CAPTURE_ANGLES_DEG, LEGACY_ANGLES_DEG, HALF_SWEEP_ANGLES_DEG,
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

    def _load_specimen_depths_dual(self,
                                    specimen_id: str,
                                    half_angles_deg: list[int]) -> tuple[list[np.ndarray], list[int]]:
        """
        Load depths from BOTH cameras for the manual 6-step dual-camera
        protocol (see HALF_SWEEP_ANGLES_DEG in shared/config.py).

        Camera A's files are labelled with their true capture angle
        directly. Camera B is mounted rigidly 180 degrees behind Camera A,
        so its files -- saved under the SAME loop angle as the simultaneous
        Camera A shot -- represent the true angle (loop_angle + 180) % 360.
        """
        spec_depth = SPECIMENS_DIR / specimen_id / "depth"
        depths: list[np.ndarray] = []
        true_angles: list[int] = []

        for a in half_angles_deg:
            p = spec_depth / view_filename(a, "A", "depth", "npy")
            if not p.exists():
                raise FileNotFoundError(f"Depth file missing: {p}")
            depths.append(np.load(str(p)))
            true_angles.append(a)

        for a in half_angles_deg:
            p = spec_depth / view_filename(a, "B", "depth", "npy")
            if not p.exists():
                raise FileNotFoundError(f"Depth file missing: {p}")
            depths.append(np.load(str(p)))
            true_angles.append((a + 180) % 360)

        return depths, true_angles

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

    def _save_outputs(self, label: str, recon: dict, reg_stats: list,
                      view_angles: list = None, view_cams: list = None) -> Path:
        """
        Save all outputs for a processed specimen to procedure_alpha/outputs/.

        Parameters
        ----------
        view_angles, view_cams : optional per-view (true angle, camera label)
            lists, same order as the point clouds fed into registration —
            used to build a legend for the by-view coloured point cloud.

        Returns the path to the stats text file.
        """
        out = RECON_OUTPUTS_DIR
        vertices  = recon['final_vertices']
        triangles = recon['final_triangles']
        merged    = recon['merged_cloud']
        view_labels = recon.get('view_labels')

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
        mgq   = recon['merge_quality']

        x, y, z = vertices[:, 0], vertices[:, 1], vertices[:, 2]
        width  = (x.max() - x.min()) * 100   # cm
        height = (y.max() - y.min()) * 100
        depth  = (z.max() - z.min()) * 100

        # Pot / shoot segmentation on the merged point cloud (height-density
        # valley heuristic — see ThreeDReconstruction.segment_pot_shoot).
        # Shoot-only volume is what should actually be compared against the
        # ground-truth net (above-ground) weight, since the whole-cloud
        # volume above includes the pot and soil surface.
        split_y     = self.reconstructor.segment_pot_shoot(merged)
        pot_mask    = merged[:, 1] <= split_y
        shoot_mask  = ~pot_mask
        pot_points  = merged[pot_mask]
        shoot_points = merged[shoot_mask]
        shoot_height_cm = (shoot_points[:, 1].max() - split_y) * 100 if len(shoot_points) else 0.0
        pot_height_cm   = (split_y - merged[:, 1].min()) * 100
        shoot_volume = (self.reconstructor.calculate_volume_voxel_approximation(shoot_points, voxel_size=VOXEL_SIZE_M)
                        if len(shoot_points) else 0.0)
        pot_volume   = (self.reconstructor.calculate_volume_voxel_approximation(pot_points, voxel_size=VOXEL_SIZE_M)
                        if len(pot_points) else 0.0)

        stats_path = out / f"reconstruction_stats_{label}.txt"
        with stats_path.open("w") as f:
            f.write(f"=== Procedure Alpha — Reconstruction Stats ===\n")
            f.write(f"Label                : {label}\n")
            f.write(f"Merged points        : {stats.get('merged_points', len(merged)):,}\n")
            f.write(f"Final vertices       : {len(vertices):,}\n")
            f.write(f"Final triangles      : {len(triangles):,}\n")
            f.write(f"Surface area         : {stats.get('surface_area', 0):.4f} m²\n")
            f.write(f"Volume               : {stats.get('volume', 0):.6f} m³\n")
            f.write(f"Height               : {height:.1f} cm\n")
            f.write(f"Width                : {width:.1f} cm\n")
            f.write(f"Depth                : {depth:.1f} cm\n")
            f.write(f"Mesh manifold        : {stats.get('is_manifold', '?')}\n")
            f.write(f"Mesh watertight      : {stats.get('is_closed', '?')}\n")
            f.write(f"Overall quality Q    : {stats.get('overall_quality', 0):.4f}\n")
            f.write(f"  Geometric fidelity : {sq.get('geometric_fidelity', 0):.4f}\n")
            f.write(f"  Coverage balance   : {mgq.get('coverage_balance', 0):.4f}\n")
            f.write(f"  Surface smoothness : {sq.get('smoothness', 0):.4f}\n")
            f.write(f"  Manifold integrity : {1.0 if stats.get('is_manifold') else 0.0:.4f}\n")
            f.write("\n--- Pot / Shoot Segmentation (height-density heuristic) ---\n")
            f.write(f"Split height (Y)     : {split_y:.4f} m\n")
            f.write(f"Pot height           : {pot_height_cm:.1f} cm\n")
            f.write(f"Shoot height         : {shoot_height_cm:.1f} cm\n")
            f.write(f"Pot points           : {len(pot_points):,}\n")
            f.write(f"Shoot points         : {len(shoot_points):,}\n")
            f.write(f"Pot volume (approx)  : {pot_volume:.6f} m³\n")
            f.write(f"Shoot volume (approx): {shoot_volume:.6f} m³\n")
            f.write("\n--- ICP Registration Stats ---\n")
            for rs in reg_stats:
                f.write(f"  {rs.get('view','?'):>8}  RMSE={rs.get('final_rmse_mm',0):.3f}mm  "
                        f"iters={rs.get('iterations',0)}  "
                        f"converged={rs.get('converged', False)}\n")
            if view_angles is not None:
                f.write("\n--- View Legend (for by-view coloured point cloud) ---\n")
                for i, angle in enumerate(view_angles):
                    cam = view_cams[i] if view_cams else "?"
                    f.write(f"  View {i:2d}  = cam{cam}  {angle:3d}deg\n")

        # Optional Open3D point cloud + mesh export
        if o3d is not None:
            try:
                # Merged point cloud, coloured by height (Y) — matches the
                # thesis appendix's viridis-style renders. Kinect RGB texture
                # mapping isn't used here: the reference implementation found
                # it unreliable due to IR interference from the depth sensors.
                y = merged[:, 1]
                y_norm = (y - y.min()) / (y.max() - y.min() + 1e-9)
                if plt is not None:
                    colors = plt.cm.viridis(y_norm)[:, :3]
                else:
                    colors = np.tile([0.6, 0.6, 0.6], (len(merged), 1))
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(merged)
                pcd.colors = o3d.utility.Vector3dVector(colors)
                o3d.io.write_point_cloud(str(out / f"merged_cloud_{label}.ply"), pcd)

                # By-view coloured point cloud — shows how the individual
                # camera/angle captures were fused into the merged model.
                if view_labels is not None and plt is not None:
                    n_views = int(view_labels.max()) + 1
                    view_cmap = plt.cm.get_cmap('tab20', max(n_views, 1))
                    view_colors = view_cmap(view_labels % 20)[:, :3]
                    pcd_views = o3d.geometry.PointCloud()
                    pcd_views.points = o3d.utility.Vector3dVector(merged)
                    pcd_views.colors = o3d.utility.Vector3dVector(view_colors)
                    o3d.io.write_point_cloud(str(out / f"merged_cloud_byview_{label}.ply"), pcd_views)

                # Pot/shoot segmented point cloud — brown pot, green shoot.
                seg_colors = np.tile([0.36, 0.25, 0.20], (len(merged), 1))
                seg_colors[shoot_mask] = [0.25, 0.70, 0.30]
                pcd_seg = o3d.geometry.PointCloud()
                pcd_seg.points = o3d.utility.Vector3dVector(merged)
                pcd_seg.colors = o3d.utility.Vector3dVector(seg_colors)
                o3d.io.write_point_cloud(str(out / f"merged_cloud_segmented_{label}.ply"), pcd_seg)

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

    @staticmethod
    def _append_timing(stats_path: Path, elapsed_s: float, n_views: int):
        """Append wall-clock processing time to a stats file — an efficiency
        metric (throughput/latency), not part of reconstruction quality."""
        with stats_path.open("a") as f:
            f.write(f"\n--- Efficiency ---\n")
            f.write(f"Processing time      : {elapsed_s:.1f} s\n")
            f.write(f"Views processed      : {n_views}\n")
            f.write(f"Time per view        : {elapsed_s / n_views:.2f} s\n")

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
        stats_path = self._save_outputs(label, recon, reg_stats,
                                        view_angles=angles_deg,
                                        view_cams=[cam_label] * len(angles_deg))

        elapsed = time.time() - t0
        self._append_timing(stats_path, elapsed, len(angles_deg))
        self._progress(f"Done in {elapsed:.1f}s", 100)

        return dict(label=label, stats_path=stats_path,
                    reconstruction=recon, reg_stats=reg_stats,
                    elapsed_s=elapsed)

    def run_specimen_dual(self,
                          specimen_id: str,
                          half_angles_deg: list[int] = None) -> dict:
        """
        Run the full pipeline for the manual 6-step dual-camera protocol:
        Camera A and Camera B fire simultaneously at each of 6 physical
        positions, with Camera B rigidly offset 180 degrees behind Camera A,
        together covering all 12 real viewing angles from 6 repositioning
        steps instead of 12.

        Parameters
        ----------
        specimen_id     : e.g. ``DG001_20260723_B01``
        half_angles_deg : the 6 loop angles used at capture time; defaults
                          to metadata.json's angles_deg, else
                          HALF_SWEEP_ANGLES_DEG (0,30,...,150)

        Returns
        -------
        results dict (same schema as run_specimen)
        """
        t0 = time.time()
        self._progress(f"Loading specimen {specimen_id} (dual-camera, 6-step)", 5)

        if half_angles_deg is None:
            meta_path = SPECIMENS_DIR / specimen_id / "metadata.json"
            if meta_path.exists():
                half_angles_deg = json.loads(meta_path.read_text())["angles_deg"]
            else:
                half_angles_deg = HALF_SWEEP_ANGLES_DEG

        depths, angles_deg = self._load_specimen_depths_dual(specimen_id, half_angles_deg)

        self._progress("Preprocessing depth maps", 15)
        clouds = self._preprocess_all(depths)

        self._progress("Registering point clouds (coarse + ICP)", 35)
        fine, transforms, reg_stats = self._register(clouds, angles_deg)

        self._progress("Reconstructing 3D mesh", 60)
        recon = self._reconstruct(fine)

        label = f"specimen_{specimen_id}"
        self._progress("Saving outputs", 85)
        n_half = len(half_angles_deg)
        stats_path = self._save_outputs(label, recon, reg_stats,
                                        view_angles=angles_deg,
                                        view_cams=["A"] * n_half + ["B"] * n_half)

        elapsed = time.time() - t0
        self._append_timing(stats_path, elapsed, len(angles_deg))
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
        self._append_timing(stats_path, elapsed, len(angles_deg))
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
                   help="Camera label for depth (new protocol, single-camera mode only)")
    p.add_argument("--dual", action="store_true",
                   help="Merge Camera A + Camera B (180-degree offset) for the manual "
                        "6-step protocol, instead of using a single camera's 12 views")
    args = p.parse_args()

    pa = ProcedureAlpha()
    if args.specimen:
        if args.dual:
            pa.run_specimen_dual(args.specimen)
        else:
            pa.run_specimen(args.specimen, cam_label=args.cam)
    else:
        pa.run_legacy(args.legacy)
