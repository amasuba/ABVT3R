"""
procedure_alpha — Subsystem 2: Classical 3D Reconstruction Pipeline
====================================================================
Implements the depth-based reconstruction methodology described by
Odwa Nombambela and extended for the 30-degree angular protocol.

Pipeline stages
---------------
1. Depth → Point Cloud (pinhole back-projection)
2. Preprocessing  (passthrough filter → SOR → MLS smoothing)
3. Coarse Registration  (rotation-based circular arrangement)
4. Fine Registration    (sequential ICP, SVD-based)
5. 3D Surface Reconstruction (voxel grid + Laplacian smoothing)
6. Geometric Feature Extraction (volume, surface area, height, compactness)

Outputs are written to procedure_alpha/outputs/
"""
