"""
evaluation_suite/metrics.py
=============================
Standard evaluation metrics for all ABVT3R subsystems.

Covers
------
Biomass regression  : MAE, RMSE, MARE, R², Pearson r
3D reconstruction   : Chamfer Distance, IoU (volumetric), F-score, PSNR/SSIM
Classification      : Accuracy, F1, Confusion Matrix
"""

from __future__ import annotations

import numpy as np
from typing import Tuple, Optional


# ---------------------------------------------------------------------------
# Regression metrics
# ---------------------------------------------------------------------------

def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error (kg)."""
    return float(np.mean(np.abs(y_pred - y_true)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Square Error (kg)."""
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))


def mare(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Relative Error (%)."""
    return float(np.mean(np.abs(y_pred - y_true) / (np.abs(y_true) + 1e-9)) * 100)


def r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient of determination R²."""
    ss_res = np.sum((y_pred - y_true) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return float(1.0 - ss_res / (ss_tot + 1e-12))


def pearson_r(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Pearson correlation coefficient."""
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def regression_report(y_true: np.ndarray,
                       y_pred: np.ndarray,
                       model_name: str = "Model") -> dict:
    """Return a complete regression metrics dict and print a summary."""
    m = dict(
        model = model_name,
        n     = len(y_true),
        mae   = mae(y_true, y_pred),
        rmse  = rmse(y_true, y_pred),
        mare  = mare(y_true, y_pred),
        r2    = r_squared(y_true, y_pred),
        r     = pearson_r(y_true, y_pred),
    )
    print(f"\n{'='*48}")
    print(f"  {model_name}  (n={m['n']})")
    print(f"{'='*48}")
    print(f"  MAE   : {m['mae']:.4f} kg")
    print(f"  RMSE  : {m['rmse']:.4f} kg")
    print(f"  MARE  : {m['mare']:.2f} %")
    print(f"  R²    : {m['r2']:.4f}")
    print(f"  r     : {m['r']:.4f}")
    print(f"{'='*48}\n")
    return m


# ---------------------------------------------------------------------------
# 3D reconstruction metrics
# ---------------------------------------------------------------------------

def chamfer_distance(pred_pts: np.ndarray,
                      gt_pts:   np.ndarray,
                      bidirectional: bool = True) -> float:
    """
    Chamfer Distance between two point clouds (metres).

    CD = (1/|P|)Σ_{p∈P} min_{q∈Q} ||p-q||² + (1/|Q|)Σ_{q∈Q} min_{p∈P} ||p-q||²
    """
    from sklearn.neighbors import NearestNeighbors

    def _one_way(src: np.ndarray, tgt: np.ndarray) -> float:
        kdt = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(tgt)
        d, _ = kdt.kneighbors(src)
        return float(np.mean(d ** 2))

    cd = _one_way(pred_pts, gt_pts)
    if bidirectional:
        cd = (cd + _one_way(gt_pts, pred_pts)) / 2.0
    return cd


def volumetric_iou(pred_voxels: np.ndarray,
                    gt_voxels:   np.ndarray) -> float:
    """
    Intersection over Union on binary voxel grids.

    Parameters
    ----------
    pred_voxels, gt_voxels : (X, Y, Z) bool arrays of identical shape
    """
    inter = np.logical_and(pred_voxels, gt_voxels).sum()
    union = np.logical_or( pred_voxels, gt_voxels).sum()
    return float(inter / (union + 1e-9))


def fscore_3d(pred_pts: np.ndarray,
               gt_pts:   np.ndarray,
               threshold: float = 0.01) -> Tuple[float, float, float]:
    """
    Precision, Recall, F-score at a distance threshold (metres).

    F-score used in 3D reconstruction benchmarks (e.g., Tanks and Temples).
    """
    from sklearn.neighbors import NearestNeighbors

    def _recall(src, tgt):
        kdt = NearestNeighbors(n_neighbors=1).fit(tgt)
        d, _ = kdt.kneighbors(src)
        return float((d.flatten() < threshold).mean())

    precision = _recall(pred_pts, gt_pts)
    recall    = _recall(gt_pts, pred_pts)
    if precision + recall == 0:
        return 0.0, 0.0, 0.0
    f  = 2 * precision * recall / (precision + recall)
    return precision, recall, f


# ---------------------------------------------------------------------------
# Classification metrics
# ---------------------------------------------------------------------------

def classification_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(y_true == y_pred))


def confusion_matrix(y_true: np.ndarray,
                      y_pred: np.ndarray,
                      n_classes: Optional[int] = None) -> np.ndarray:
    if n_classes is None:
        n_classes = int(max(y_true.max(), y_pred.max())) + 1
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true.astype(int), y_pred.astype(int)):
        cm[t, p] += 1
    return cm
