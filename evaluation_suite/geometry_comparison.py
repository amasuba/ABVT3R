#!/usr/bin/env python3
"""
evaluation_suite/geometry_comparison.py
==========================================
Cross-method surface geometry comparison between the classical
procedure_alpha mesh and the NeRF (Nerfstudio nerfacto) exported point
cloud, for the same specimen reconstructed from the same 12 views. The
NeRF camera poses are derived directly from procedure_alpha's own
registration math (neural_geometry/nerf/build_transforms.py), not an
independent calibration, so both methods are being asked to reconstruct
the same physical scene in the same nominal coordinate frame.

IMPORTANT — what this measures and what it does NOT measure
-------------------------------------------------------------
There is no independently-scanned ground-truth geometry for these plants
(no laser scanner, no structured-light scan, no reference CAD model). So
Chamfer Distance / F-score / HD95 / Normal Consistency reported here are
a measure of AGREEMENT BETWEEN THE TWO METHODS, not accuracy against
ground truth. If both methods share a common bias (e.g. both smooth over
the same fine leaf structure, or both inherit the same pose error), this
comparison cannot detect it. Report and interpret accordingly — do not
relabel this "reconstruction accuracy" in the dissertation.

Alignment
---------
No scale normalisation is applied or permitted: both point sets are
supposed to already be metrically consistent (same poses, same physical
scene), and rescaling would hide exactly the metric-grounding error this
comparison exists to surface. A small RIGID (rotation + translation only)
ICP refinement corrects for residual NeRF optimisation drift; the
resulting correction magnitude is reported as a sanity check on the pose
derivation — a large correction would mean the two methods disagree about
more than fine surface detail.

Floater handling
-----------------
NeRF point clouds exported via `ns-export pointcloud` commonly contain
floater noise (points far from the object, from under-constrained density
in weakly-observed regions — worse with few views / limited training
budget). DBSCAN isolates the largest coherent cluster before comparison;
the discarded fraction is reported as a NeRF quality diagnostic in its own
right, not hidden.

Usage
-----
    python evaluation_suite/geometry_comparison.py --specimen M001 \\
        --nerf-experiment M001_assumed_fixed_full
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import open3d as o3d

from evaluation_suite.metrics import (
    chamfer_distance, fscore_3d, hausdorff_95, normal_consistency,
)
from shared.config import RECON_OUTPUTS_DIR, NEURAL_GEOMETRY_DIR, EVAL_REPORTS_DIR

# Nerfstudio's world-frame convention (X-right, Y-up, Z-back) vs. this
# project's own convention (X-right, Y-up, Z-forward) — see
# neural_geometry/nerf/build_transforms.py for the derivation. Applies to
# both points and normals (a reflection is its own inverse-transpose).
Z_FLIP = np.diag([1.0, 1.0, -1.0])


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_classical_surface(specimen_id: str, n_points: int = 100_000):
    mesh_path = RECON_OUTPUTS_DIR / f"mesh_specimen_{specimen_id}.ply"
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()
    pcd = mesh.sample_points_uniformly(number_of_points=n_points, use_triangle_normal=False)
    return pcd


def load_nerf_pointcloud(experiment_name: str):
    ply_path = NEURAL_GEOMETRY_DIR / "nerf_outputs" / experiment_name / "pointcloud" / "point_cloud.ply"
    pcd = o3d.io.read_point_cloud(str(ply_path))
    pts = np.asarray(pcd.points) @ Z_FLIP.T
    pcd.points = o3d.utility.Vector3dVector(pts)
    if pcd.has_normals():
        n = np.asarray(pcd.normals) @ Z_FLIP.T
        pcd.normals = o3d.utility.Vector3dVector(n)
    return pcd


# ---------------------------------------------------------------------------
# Floater removal
# ---------------------------------------------------------------------------

def dbscan_largest_cluster(pcd, eps: float = 0.05, min_points: int = 20):
    labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=False))
    n_total = len(labels)
    if labels.max() < 0:
        raise RuntimeError("DBSCAN found no clusters — eps/min_points too strict for this cloud")
    counts = np.bincount(labels[labels >= 0])
    largest = int(np.argmax(counts))
    keep = labels == largest
    stats = dict(
        n_total=n_total,
        n_clusters=int(labels.max()) + 1,
        n_kept=int(keep.sum()),
        n_discarded=int(n_total - keep.sum()),
        discarded_frac=float(1 - keep.sum() / n_total),
    )
    return pcd.select_by_index(np.where(keep)[0]), stats


# ---------------------------------------------------------------------------
# Rigid alignment (rotation + translation only — no scaling)
# ---------------------------------------------------------------------------

def rigid_align(source, target, max_corr_dist: float = 0.15):
    result = o3d.pipelines.registration.registration_icp(
        source, target, max_corr_dist, np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=200),
    )
    aligned = source.transform(result.transformation)
    T = result.transformation
    translation_mag = float(np.linalg.norm(T[:3, 3]))
    rotation_angle_deg = float(np.degrees(np.arccos(np.clip((np.trace(T[:3, :3]) - 1) / 2, -1, 1))))
    return aligned, dict(
        fitness=result.fitness, rmse=result.inlier_rmse,
        translation_m=translation_mag, rotation_deg=rotation_angle_deg,
    )


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def run(specimen_id: str, nerf_experiment: str, n_classical_points: int = 100_000,
        dbscan_eps: float = 0.05, dbscan_min_points: int = 20,
        icp_max_corr_dist: float = 0.15,
        f_score_thresholds_frac=(0.01, 0.02, 0.05)):
    lines = []
    def out(s=""):
        print(s)
        lines.append(s)

    out("=" * 72)
    out(f"CROSS-METHOD GEOMETRY COMPARISON — {specimen_id}")
    out(f"classical mesh (procedure_alpha)  vs.  NeRF point cloud ({nerf_experiment})")
    out("=" * 72)
    out("\nNOTE: this is METHOD AGREEMENT, not ground-truth accuracy — there is no")
    out("independently scanned reference geometry for this specimen. See module")
    out("docstring for the full caveat.")

    classical = load_classical_surface(specimen_id, n_classical_points)
    nerf_raw = load_nerf_pointcloud(nerf_experiment)
    out(f"\nClassical mesh surface samples : {len(classical.points):,}")
    out(f"NeRF raw exported points       : {len(nerf_raw.points):,}")

    nerf_filtered, dbscan_stats = dbscan_largest_cluster(nerf_raw, dbscan_eps, dbscan_min_points)
    out(f"\n--- Floater filtering (DBSCAN, eps={dbscan_eps}m, min_points={dbscan_min_points}) ---")
    out(f"  Clusters found       : {dbscan_stats['n_clusters']}")
    out(f"  Largest-cluster kept : {dbscan_stats['n_kept']:,} / {dbscan_stats['n_total']:,} "
        f"({100*(1-dbscan_stats['discarded_frac']):.1f}%)")
    out(f"  Discarded as noise   : {dbscan_stats['n_discarded']:,} ({100*dbscan_stats['discarded_frac']:.1f}%)")
    if dbscan_stats["discarded_frac"] > 0.3:
        out("  WARNING: >30% of exported points discarded as floater noise — treat the")
        out("  geometry numbers below as low-confidence for this training run.")

    nerf_aligned, icp_stats = rigid_align(nerf_filtered, classical, icp_max_corr_dist)
    out(f"\n--- Rigid alignment (ICP, rotation+translation only, no scaling) ---")
    out(f"  Fitness              : {icp_stats['fitness']:.3f}")
    out(f"  Inlier RMSE          : {icp_stats['rmse']*1000:.1f} mm")
    out(f"  Correction applied   : {icp_stats['translation_m']*1000:.1f} mm translation, "
        f"{icp_stats['rotation_deg']:.2f} deg rotation")
    if icp_stats["translation_m"] > 0.10 or icp_stats["rotation_deg"] > 15:
        out("  WARNING: large rigid correction — poses derived for the NeRF run may not")
        out("  agree with procedure_alpha's own registration as closely as assumed.")

    classical_pts = np.asarray(classical.points)
    classical_n = np.asarray(classical.normals)
    nerf_pts = np.asarray(nerf_aligned.points)
    nerf_n = np.asarray(nerf_aligned.normals) if nerf_aligned.has_normals() else None

    bbox_diag = float(np.linalg.norm(classical_pts.max(0) - classical_pts.min(0)))
    out(f"\nClassical mesh bounding-box diagonal : {bbox_diag:.3f} m  (used to scale F-score thresholds)")

    out(f"\n--- Surface geometry agreement ---")
    cd_sq = chamfer_distance(nerf_pts, classical_pts)
    out(f"  Chamfer Distance     : {cd_sq*1e6:.1f} mm^2  (RMS surface distance = {np.sqrt(cd_sq)*1000:.1f} mm)")

    hd95 = hausdorff_95(nerf_pts, classical_pts)
    out(f"  HD95 (symmetric)     : {hd95*1000:.1f} mm")

    out(f"\n  F-score @ threshold (rigid alignment, NO scale normalisation):")
    for frac in f_score_thresholds_frac:
        tau = frac * bbox_diag
        p, r, f = fscore_3d(nerf_pts, classical_pts, threshold=tau)
        out(f"    tau={tau*1000:6.1f} mm ({frac*100:.0f}% of bbox diag)  "
            f"P={p:.3f}  R={r:.3f}  F={f:.3f}")

    if nerf_n is not None:
        nc = normal_consistency(nerf_pts, nerf_n, classical_pts, classical_n)
        out(f"\n  Normal Consistency   : {nc:.3f}  (1.0 = perfect agreement, |cos angle|)")
    else:
        out("\n  Normal Consistency   : skipped (NeRF point cloud has no normals)")

    out("\n" + "=" * 72)

    EVAL_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = EVAL_REPORTS_DIR / f"geometry_comparison_{specimen_id}.txt"
    report_path.write_text("\n".join(lines) + "\n")
    print(f"\n[GeometryComparison] Report saved -> {report_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--specimen", required=True, help="e.g. M001")
    ap.add_argument("--nerf-experiment", required=True,
                     help="Nerfstudio experiment name, e.g. M001_assumed_fixed_full "
                          "(expects neural_geometry/nerf_outputs/<name>/pointcloud/point_cloud.ply)")
    ap.add_argument("--n-classical-points", type=int, default=100_000)
    ap.add_argument("--dbscan-eps", type=float, default=0.05)
    ap.add_argument("--dbscan-min-points", type=int, default=20)
    ap.add_argument("--icp-max-corr-dist", type=float, default=0.15)
    args = ap.parse_args()

    run(args.specimen, args.nerf_experiment, args.n_classical_points,
        args.dbscan_eps, args.dbscan_min_points, args.icp_max_corr_dist)


if __name__ == "__main__":
    main()
