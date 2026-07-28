#!/usr/bin/env python3
"""
biomass_engine/evaluation_metrics.py
=======================================
Extended biomass regression metrics beyond plain MAE/RMSE, following the
evaluation-metrics reference used for this project's dissertation (Section
6: Biomass regression metrics). Implements:

  - Bias (mean error)               — §6.2, catches systematic under/over-
                                        prediction that MAE/RMSE hide
  - nRMSE                            — §6.1, RMSE normalised by mean(y) so
                                        results are comparable across
                                        species/growth stages
  - Lin's Concordance Correlation
    Coefficient (CCC)                — §6.3, agreement with the 1:1 line;
                                        catches systematic scale error that
                                        R² alone can miss
  - Bland-Altman limits of agreement — §6.3, reveals whether error grows
                                        with specimen size (the expected
                                        pattern if occlusion scales with
                                        canopy density)

All metrics operate on whatever unit y/y_hat are already in (grams, here).
"""

import numpy as np
import matplotlib.pyplot as plt


def extended_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    err = y_pred - y_true

    mae  = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    bias = float(np.mean(err))                      # §6.2

    mean_y = float(np.mean(y_true))
    nrmse  = rmse / mean_y if mean_y != 0 else float("nan")   # §6.1

    ss_res = np.sum(err ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

    # Lin's CCC — §6.3
    mu_y, mu_yhat = y_true.mean(), y_pred.mean()
    var_y, var_yhat = y_true.var(), y_pred.var()
    if len(y_true) > 1:
        rho = np.corrcoef(y_true, y_pred)[0, 1]
    else:
        rho = float("nan")
    sigma_y, sigma_yhat = np.sqrt(var_y), np.sqrt(var_yhat)
    denom = var_y + var_yhat + (mu_y - mu_yhat) ** 2
    ccc = float(2 * rho * sigma_y * sigma_yhat / denom) if denom > 0 else float("nan")

    return dict(mae=mae, rmse=rmse, bias=bias, nrmse=nrmse, r2=r2, ccc=ccc,
                n=len(y_true))


def print_metrics_table(results: dict):
    """results: {model_name: metrics_dict}"""
    print(f"\n{'Model':<8}{'n':>4}{'MAE (g)':>10}{'RMSE (g)':>10}"
          f"{'Bias (g)':>10}{'nRMSE':>9}{'R²':>8}{'CCC':>8}")
    print("-" * 67)
    for name, m in results.items():
        print(f"{name:<8}{m['n']:>4}{m['mae']:>10.1f}{m['rmse']:>10.1f}"
              f"{m['bias']:>+10.1f}{m['nrmse']:>9.3f}{m['r2']:>8.3f}{m['ccc']:>8.3f}")


def bland_altman_panel(ax, y_true: np.ndarray, y_pred: np.ndarray,
                        label: str, color: str):
    """Draw one Bland-Altman panel: (y_hat - y) vs (y_hat + y)/2, with mean
    difference and +/-1.96 SD limits of agreement."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    diff = y_pred - y_true
    mean = (y_pred + y_true) / 2

    md = diff.mean()
    sd = diff.std(ddof=1) if len(diff) > 1 else 0.0
    loa_hi, loa_lo = md + 1.96 * sd, md - 1.96 * sd

    ax.scatter(mean, diff, c=color, alpha=0.8, s=45, zorder=3)
    ax.axhline(md,     color="black", lw=1.5, ls="-",  label=f"Mean diff = {md:+.1f}g")
    ax.axhline(loa_hi, color="red",   lw=1.2, ls="--", label=f"+1.96 SD = {loa_hi:+.1f}g")
    ax.axhline(loa_lo, color="red",   lw=1.2, ls="--", label=f"-1.96 SD = {loa_lo:+.1f}g")
    ax.axhline(0,       color="grey",  lw=0.8, ls=":")
    ax.set_xlabel("Mean of predicted & measured (g)")
    ax.set_ylabel("Predicted - Measured (g)")
    ax.set_title(f"Bland-Altman — {label}")
    ax.legend(fontsize=7, loc="best")


def save_bland_altman_figure(rf_true, rf_pred, ann_true, ann_pred, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    bland_altman_panel(axes[0], rf_true,  rf_pred,  "RF",  "#4C72B0")
    bland_altman_panel(axes[1], ann_true, ann_pred, "ANN", "#DD8452")
    fig.suptitle("Bland-Altman Limits of Agreement — Mango Biomass (LOOCV)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(str(out_path), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[Eval] Bland-Altman figure -> {out_path}")
