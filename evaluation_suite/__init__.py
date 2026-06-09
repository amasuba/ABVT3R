"""
evaluation_suite — Subsystem 5: Results & Evaluation
======================================================
Provides cross-method analysis, statistical significance testing,
and publication-quality figures comparing:
  - Level 1: Procedure Alpha (RF + ANN on classical 3D features)
  - Level 2: Neural Geometry (DINOv2 + Volumetric Transformer)

Entry points
------------
metrics.py      Metric primitives (MAE, RMSE, MARE, R², Chamfer, IoU)
comparison.py   Cross-method evaluation with Wilcoxon test + LaTeX table
"""
