# Automated Plant Biomass Estimation System

A non-destructive plant phenotyping system for biomass estimation through 3D reconstruction using dual Kinect V2 RGB-D cameras. The system employs a four-view capture strategy with voxel-based reconstruction and Random Forest regression to achieve automated plant biomass predictions within 2 minutes per specimen.

## Project Overview

This project implements an automated plant phenotyping system designed for ornamental plants, specifically Duranta Gold mini trees, in greenhouse environments. The system achieves non-destructive biomass estimation through volumetric 3D reconstruction from multiple viewpoints, enabling continuous monitoring of plant growth without harvesting.

### Key Specifications

- **Biomass Estimation Accuracy**: 93.4% (exceeds 90% target)
- **Dimensional Accuracy**: 84.2% (height: 92.2%, width: 86.9%, depth: 73.4%)
- **Spatial Resolution**: 7mm voxel size
- **Processing Time**: ~95 seconds per plant (within 2-minute target)
- **Capture Configuration**: Four views at 90° intervals (0°, 90°, 180°, 270°)

## System Architecture

The system employs a client-server architecture with four major subsystems:

1. **Hardware Capture Subsystem**: Custom 6×3×2m gantry with dual Kinect V2 cameras mounted on opposing sides, rotating around stationary plants via Arduino-controlled NEMA 23 stepper motors.

2. **Processing Subsystem**: ODROID N2+ embedded platform executing the complete computational pipeline autonomously.

3. **Communication Subsystem**: TCP socket-based client-server protocols separating user interaction (GUI client) from computational processing (host server).

4. **Control Subsystem**: Arduino Uno R3 with TB6600 drivers controlling stepper motors for linear gantry motion and rotational mechanisms.

## Processing Pipeline

The software implements a five-stage modular pipeline:

### 1. Pre-Processing
Transforms raw depth maps into filtered point clouds suitable for registration:
- Point cloud generation from depth maps using pinhole camera model
- PassThrough filtering for region-of-interest extraction
- Statistical Outlier Removal (SOR) for noise reduction (k=50, α=1.0)
- Moving Least Squares (MLS) surface smoothing (search radius: 20mm)

### 2. Registration
Aligns multiple point cloud views into a unified coordinate system:
- **Coarse Registration**: Rotation-based circular arrangement exploiting known 90° capture geometry
- **Fine Registration**: Iterative Closest Point (ICP) with Sequential alignment strategy
  - Convergence threshold: 1×10⁻⁶
  - Maximum iterations: 200
  - Maximum correspondence distance: 20mm
  - Average registration RMSE: 14.1mm

### 3. 3D Reconstruction
Generates watertight triangular meshes from aligned point clouds:
- Grid-based voxelization with 7mm resolution
- Marching Cubes-based surface extraction
- Automatic non-manifold edge repair
- Mesh quality assessment metrics

### 4. Dimensional Analysis
Extracts geometric features from reconstructed meshes:
- Height, width, depth from bounding box dimensions
- Volume calculation via voxel-based method
- Surface area from triangle mesh summation
- Compactness metric: C = A^(3/2) / V
- Mesh quality scores

### 5. Biomass Prediction
Random Forest regression model predicting biomass from geometric features:
- 50 decision trees with maximum depth of 3
- Leave-One-Out Cross-Validation (LOOCV)
- Feature importance: Height (40%), Surface Area (28%), Volume (15%), Compactness (12%), Quality (5%)
- Training dataset: 40 plant specimens
- Performance: MAE = 0.088kg, RMSE = 0.098kg, R² = 0.463

## Implementation Details

All core algorithms were implemented from first principles using NumPy, without reliance on high-level libraries like Open3D or PCL for processing. This approach provides complete control over algorithmic behaviour and enables optimization for Kinect V2 characteristics and plant morphology.

### Dependencies

```
numpy >= 1.20      # Core array operations and linear algebra
scikit-learn >= 1.0 # KD-tree for nearest neighbor search only
opencv-python       # Image operations
pylibfreenect2      # Kinect V2 camera interfacing
pyserial            # Arduino communication
tkinter             # GUI development
```

### Key Classes

- `PreProcessing`: Depth map conversion, filtering, smoothing
- `Registration`: Coarse rotation arrangement, ICP alignment
- `ThreeDReconstruction`: Voxel-based mesh generation
- `RandomForestRegressor`: Decision tree ensemble for biomass prediction (implemented from scratch)

## Hardware Requirements

- **Cameras**: Dual Microsoft Kinect V2 (1920×1080 RGB, 512×424 depth)
- **Processing**: ODROID N2+ (4GB RAM, Quad-core Cortex-A73)
- **Motors**: NEMA 23 stepper motors with TB6600 drivers
- **Microcontroller**: Arduino Uno R3
- **Structure**: Custom 6×3×2m aluminum gantry system

## Validation Results

Testing on ten plants of varying morphologies under greenhouse conditions:

| Metric | Result |
|--------|--------|
| Mean Biomass Accuracy | 93.4% |
| Mean Absolute Error (Biomass) | 0.088 kg |
| Root Mean Square Error (Biomass) | 0.098 kg |
| Dimensional Accuracy | 84.2% |
| Height MAE | 74.5 mm |
| Width MAE | 55.0 mm |
| Depth MAE | 122.4 mm |
| Average Processing Time | 94.89 seconds |
| ICP Convergence Rate | 100% (within 100 iterations) |

## Known Limitations

1. **Texture Mapping**: Not implemented due to infrared interference between Kinect depth sensors preventing simultaneous RGB-D capture with texture preservation.

2. **Dimensional Accuracy**: Falls short of 95% target, primarily due to:
   - 7mm voxel quantization at bounding box extremities
   - Wind-induced motion artifacts during sequential capture (depth axis most affected)
   - Plants exceeding 100cm height constraint

3. **Size Range**: Limited by Kinect V2 operational range (0.5–4.5m) and current height constraint of 100cm.

4. **Environmental Sensitivity**: Greenhouse ventilation perpendicular to depth axis causes elongated point clouds, contributing to depth measurement errors (MAE = 122.4mm).

5. **Biomass Prediction Limitations**: R² of 0.463 indicates geometric features explain only 46% of biomass variance. Remaining variance stems from density variations, foliage distribution, and moisture content not captured by geometry alone.

## Future Work

- Implement Poisson surface reconstruction for improved dimensional accuracy
- Deploy stereo vision cameras instead of infrared depth sensors for texture mapping capability
- Expand training dataset beyond 40 samples to improve model generalization
- Incorporate texture features from RGB data into biomass prediction model
- Migrate to more powerful embedded platform (e.g., Jetson) for real-time processing
- Implement environmental control protocols to minimize wind-induced motion artifacts

## Project Context

This system was developed as a Final Year Project (EPR402) at the University of Pretoria's EECE department, representing a shift from traditional agricultural crop phenotyping to ornamental plant applications. The work demonstrates cost-effective phenotyping using consumer-grade RGB-D sensors while maintaining professional accuracy standards suitable for greenhouse monitoring and plant breeding applications.

## Academic Contribution

The project contributes to plant phenotyping literature by:
- Demonstrating sequential ICP registration significantly outperforms pairwise approaches (RMSE: 14.1mm vs. 20.8mm)
- Validating voxel-based volume calculations prove more reliable than convex hull methods for sparse plant structures
- Confirming Random Forest superiority over neural networks for small phenotyping datasets
- Providing open-source reference implementation for educational institutions

## License

This project is academic work completed at the University of Pretoria. Please refer to the university's policies regarding usage and distribution of student work.

## Author

**Aaron Masuba**  
Student ID: 25737806  
Email: [aaron.masuba@tuks.co.za](mailto:aaron.masuba@tuks.co.za)  
University of Pretoria, Department of Electrical, Electronic and Computer Engineering  
Program: MEng in Computer Engineering  
Project Supervisor: Prof. H. Myburgh

## Citation

If you use this work in your research, please cite:

```
Masuba, A. (2025). Automated Plant Biomass Estimation System using 3D Reconstruction 
from Dual RGB-D Cameras. MEng Project Report, University of Pretoria.
```

## Quick Start

### Smoke test (no Kinect required)

Run these commands one at a time in your terminal:

**1. Navigate to the project folder**
```powershell
cd C:\Users\u25737806\ABVT3R
```

**2. Install dependencies**
```powershell
python -m pip install numpy scipy scikit-learn opencv-python matplotlib pyserial
```
> Note: `open3d` has no wheel for Python 3.13. The pipeline will automatically skip PLY/OBJ export if `open3d` is not available — all NumPy outputs are still saved.

**3. Run the smoke test**
```powershell
python smoke_run.py
```

Expected final output:
```
[100%] Pipeline complete!
Smoke run completed successfully.
```

**4. View the results**
```powershell
Get-Content reconstruction_output\reconstruction_stats_plant_1.txt
```

Outputs are written to `reconstruction_output/`:
| File | Description |
|---|---|
| `reconstruction_stats_plant_1.txt` | Dimensions, volume, surface area, biomass prediction, processing time |
| `final_vertices_plant_1.npy` | Mesh vertex coordinates |
| `final_triangles_plant_1.npy` | Mesh triangle indices |
| `merged_points_plant_1.npy` | Merged registered point cloud |
| `surface_normals_plant_1.npy` | Surface normals |

### Full client/server system (requires Kinect hardware + Linux)

1. Start the host on the processing machine (ODROID/Jetson):
```bash
python host.py
```
2. Start the GUI client on the operator machine:
```bash
python GUI.py
```
> Requires: `pylibfreenect2` or `freenect2`, `open3d`, Arduino serial connection, and a Linux-based platform with Kinect V2 drivers.

