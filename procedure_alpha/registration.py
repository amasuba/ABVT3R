"""
procedure_alpha/registration.py
=================================
Point-cloud registration — coarse (rotation-based) + fine (ICP).

Fully generalised for N views at any angular step:
  • 4-view  legacy protocol  (90° steps)
  • 12-view new protocol     (30° steps)
  • Any custom angle list

Key changes over the original registration_class.py
----------------------------------------------------
- ``arrange_views_in_circle`` no longer hard-codes 4 views; it accepts the
  actual capture angles so any N-view rig works.
- ``sequential_icp_registration`` and ``pairwise_icp_registration`` derive
  their view labels from the angle list passed at call time, not from a
  hardcoded ['0°','90°','180°','270°'] literal.
- No other algorithmic changes — ICP core (SVD, correspondence finding) is
  identical to the implementation verified by Odwa Nombambela.
"""

import numpy as np
from sklearn.neighbors import NearestNeighbors


class Registration:
    """
    Multi-view point-cloud registration.

    Stage 1 — Coarse  : rotate each view into a common frame based on its
                        known capture angle (exploits the known rig geometry).
    Stage 2 — Fine    : ICP (point-to-point, SVD-based) to remove residual
                        mis-alignment introduced by sensor noise or small
                        turntable positioning errors.
    """

    def __init__(self):
        pass

    # =========================================================================
    # Helpers
    # =========================================================================

    def calculate_centroid(self, pc: np.ndarray) -> np.ndarray:
        return np.mean(pc, axis=0)

    def center_pc(self, pc: np.ndarray, centroid: np.ndarray) -> np.ndarray:
        return pc - centroid

    def get_rotation_matrix_y(self, angle_rad: float) -> np.ndarray:
        c, s = np.cos(angle_rad), np.sin(angle_rad)
        return np.array([[c, 0, s],
                         [0, 1, 0],
                         [-s, 0, c]])

    # =========================================================================
    # Stage 1 — Coarse registration
    # =========================================================================

    def arrange_views_in_circle(self,
                                point_clouds: list[np.ndarray],
                                angles_rad:   list[float],
                                radius:       float = 0.0) -> tuple:
        """
        Apply the known capture rotation to each view.

        Each view was captured with the plant rotated by ``angle`` around
        the vertical axis.  Applying the *inverse* rotation aligns all
        views into the 0° reference frame.

        Parameters
        ----------
        point_clouds : list of (N_i, 3) arrays — one per captured view
        angles_rad   : capture angle for each view in radians
                       (e.g. for 30° protocol: 0, π/6, π/3, …, 11π/6)
        radius       : unused legacy parameter, kept for API compatibility

        Returns
        -------
        arranged_pcs       : list of (N_i, 3) arrays in the common frame
        construction_center: centroid of the merged raw cloud
        """
        all_points        = np.vstack(point_clouds)
        construction_center = self.calculate_centroid(all_points)
        print(f"[Reg] Global centroid (coarse): {construction_center}")

        arranged = []
        for pc, angle in zip(point_clouds, angles_rad):
            deg = np.degrees(angle)
            centroid = self.calculate_centroid(pc)
            centered = self.center_pc(pc, centroid)
            R        = self.get_rotation_matrix_y(angle)
            rotated  = centered @ R.T
            arranged.append(rotated)
            print(f"[Reg]   Coarse-aligned {deg:.0f}°  ({len(pc):,} pts)")

        return arranged, construction_center

    def check_alignment_quality(self, transformed_pcs: list[np.ndarray]) -> list[float]:
        """Return centroid distances of views 1…N relative to view 0."""
        centroids = [self.calculate_centroid(pc) for pc in transformed_pcs]
        ref = centroids[0]
        return [np.linalg.norm(c - ref) for c in centroids[1:]]

    # =========================================================================
    # Stage 2 helpers — ICP core
    # =========================================================================

    def find_correspondences(self,
                             source_pc: np.ndarray,
                             target_pc: np.ndarray,
                             max_distance: float = 0.01):
        kdt = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(target_pc)
        distances, indices = kdt.kneighbors(source_pc)
        valid = distances.flatten() < max_distance
        return (list(zip(np.where(valid)[0],
                         indices[valid].flatten(),
                         distances[valid].flatten())),
                source_pc[valid],
                target_pc[indices[valid].flatten()])

    def estimate_transformation(self,
                                source_pts: np.ndarray,
                                target_pts: np.ndarray):
        """SVD-based optimal rigid-body registration."""
        sc = np.mean(source_pts, axis=0)
        tc = np.mean(target_pts, axis=0)
        H  = (source_pts - sc).T @ (target_pts - tc)
        U, _, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[-1] *= -1
            R = Vt.T @ U.T
        t    = tc - R @ sc
        rmse = np.sqrt(np.mean(np.sum(((R @ source_pts.T).T + t - target_pts) ** 2, axis=1)))
        return R, t, rmse

    def transform_pc(self, pc: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
        return (R @ pc.T).T + t

    def icp_registration(self,
                         source_pc: np.ndarray,
                         target_pc: np.ndarray,
                         max_iterations: int   = 200,
                         tolerance: float      = 1e-6,
                         max_corr_dist: float  = 0.01):
        """
        Standard point-to-point ICP.

        Returns
        -------
        result_dict  : {R, t, rmse, iterations, converged}
        registered   : transformed source cloud
        history      : list of per-iteration RMSE values
        """
        current   = source_pc.copy()
        cum_R     = np.eye(3)
        cum_t     = np.zeros(3)
        history   = []
        prev_rmse = float('inf')
        converged = False
        iters     = 0

        for i in range(max_iterations):
            iters = i + 1
            corr, vsrc, vtgt = self.find_correspondences(current, target_pc, max_corr_dist)
            if len(corr) < 10:
                print(f"[ICP] Stopping: only {len(corr)} correspondences at iter {iters}")
                break
            R, t, rmse = self.estimate_transformation(vsrc, vtgt)
            current    = self.transform_pc(current, R, t)
            cum_R      = R @ cum_R
            cum_t      = R @ cum_t + t
            history.append(rmse)
            if abs(prev_rmse - rmse) < tolerance:
                print(f"[ICP] Converged at iter {iters}  RMSE={rmse*1000:.3f}mm")
                converged = True
                break
            prev_rmse = rmse

        result = dict(R=cum_R, t=cum_t, rmse=rmse if iters else 0.0,
                      iterations=iters, converged=converged)
        print(f"[ICP] RMSE={result['rmse']*1000:.3f}mm  iters={iters}  converged={converged}")
        return result, current, history

    # =========================================================================
    # Stage 2 — Multi-view ICP strategies
    # =========================================================================

    def sequential_icp_registration(self,
                                     transformed_pcs: list[np.ndarray],
                                     icp_params:      dict = None,
                                     angles_deg:      list = None):
        """
        Sequentially register view i against the accumulated registered cloud
        from views 0…i-1.

        Works for any number of views.

        Parameters
        ----------
        transformed_pcs : coarse-registered point clouds (output of arrange_views_in_circle)
        icp_params      : ICP hyper-parameters dict
        angles_deg      : capture angles in degrees, used only for labelling output

        Returns
        -------
        fine_pcs            : list of fine-registered clouds
        transformations     : list of {R, t} dicts
        registration_stats  : list of per-view stat dicts
        """
        if icp_params is None:
            icp_params = dict(max_iterations=200, tolerance=1e-6, max_corr_dist=0.02)

        n = len(transformed_pcs)
        labels = ([f"{a}°" for a in angles_deg]
                  if angles_deg and len(angles_deg) == n
                  else [f"view_{i}" for i in range(n)])

        fine_pcs = [transformed_pcs[0].copy()]
        transforms = [dict(R=np.eye(3), t=np.zeros(3))]
        stats = []

        for i in range(1, n):
            accum  = np.vstack(fine_pcs)
            src    = transformed_pcs[i]
            init_e = np.linalg.norm(self.calculate_centroid(src) - self.calculate_centroid(accum))
            print(f"\n[Reg] Sequential ICP: {labels[i]} → accumulated  "
                  f"(init Δcent={init_e*1000:.1f}mm)")
            result, reg_src, hist = self.icp_registration(
                src, accum,
                icp_params['max_iterations'],
                icp_params['tolerance'],
                icp_params['max_corr_dist'])
            fine_pcs.append(reg_src)
            transforms.append(result)
            stats.append(dict(view=labels[i],
                              initial_error_mm=init_e * 1000,
                              final_rmse_mm=result['rmse'] * 1000,
                              iterations=result['iterations'],
                              converged=result['converged']))

        avg_rmse = np.mean([s['final_rmse_mm'] for s in stats]) if stats else 0.0
        print(f"\n[Reg] Sequential ICP done  avg_RMSE={avg_rmse:.3f}mm  "
              f"all_converged={all(s['converged'] for s in stats)}")
        return fine_pcs, transforms, stats

    def pairwise_icp_registration(self,
                                   transformed_pcs: list[np.ndarray],
                                   icp_params:      dict = None,
                                   angles_deg:      list = None):
        """
        Register every view directly to the 0° reference view.

        Avoids error accumulation compared to sequential ICP at the cost of
        not using overlapping information between adjacent views.

        Parameters
        ----------
        Same as sequential_icp_registration.
        """
        if icp_params is None:
            icp_params = dict(max_iterations=200, tolerance=1e-6, max_corr_dist=0.05)

        n = len(transformed_pcs)
        labels = ([f"{a}°" for a in angles_deg]
                  if angles_deg and len(angles_deg) == n
                  else [f"view_{i}" for i in range(n)])

        ref      = transformed_pcs[0]
        fine_pcs = [ref.copy()]
        transforms = [dict(R=np.eye(3), t=np.zeros(3))]
        stats = []

        for i in range(1, n):
            src    = transformed_pcs[i]
            init_e = np.linalg.norm(self.calculate_centroid(src) - self.calculate_centroid(ref))
            print(f"\n[Reg] Pairwise ICP: {labels[i]} → {labels[0]}  "
                  f"(init Δcent={init_e*1000:.1f}mm)")
            result, reg_src, hist = self.icp_registration(
                src, ref,
                icp_params['max_iterations'],
                icp_params['tolerance'],
                icp_params['max_corr_dist'])
            fine_pcs.append(reg_src)
            transforms.append(result)
            stats.append(dict(view=labels[i],
                              initial_error_mm=init_e * 1000,
                              final_rmse_mm=result['rmse'] * 1000,
                              iterations=result['iterations'],
                              converged=result['converged'],
                              improvement_mm=init_e * 1000 - result['rmse'] * 1000))

        avg_rmse = np.mean([s['final_rmse_mm'] for s in stats]) if stats else 0.0
        print(f"\n[Reg] Pairwise ICP done  avg_RMSE={avg_rmse:.3f}mm  "
              f"all_converged={all(s['converged'] for s in stats)}")
        return fine_pcs, transforms, stats
