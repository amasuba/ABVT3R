"""
neural_geometry — Subsystem 4: Self-Supervised & Meta-Learning
================================================================
Implements the VGGT-inspired architecture described in the thesis:

  Multi-view RGB-D acquisition
    → DINOv2 Vision Transformer backbone  (feature extraction)
    → Volumetric Transformer              (3D reasoning in voxel space)
    → Visual Geometry Grounding           (geometry-aware refinement)
    → SAM3D segmentation                  (plant / pot separation)
    → Biomass inference                   (volume regression)

Submodules
----------
backbone/dinov2_encoder.py      DINOv2 ViT feature extractor + multi-view fusion
volumetric/volumetric_transformer.py  Windowed 3-D attention over voxel grid
sam3d/sam3d_pipeline.py        Segment Anything 3D — plant mask & soil removal
pretraining/dino_pretraining.py Teacher-student DINO self-supervised pre-training

This subsystem feeds into evaluation_suite/ where its results are compared
against the Procedure Alpha (classical) baseline.
"""
