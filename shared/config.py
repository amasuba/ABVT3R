"""
shared/config.py
================
Central configuration for all ABVT3R subsystems.
All camera intrinsics, acquisition geometry, paths, and hyper-parameters
are defined here so that a single edit propagates everywhere.
"""

from pathlib import Path
import numpy as np

# ---------------------------------------------------------------------------
# Repository root (all other paths are relative to this)
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Subsystem directories
# ---------------------------------------------------------------------------
ACQUISITION_DIR      = REPO_ROOT / "acquisition"
PROCEDURE_ALPHA_DIR  = REPO_ROOT / "procedure_alpha"
BIOMASS_ENGINE_DIR   = REPO_ROOT / "biomass_engine"
NEURAL_GEOMETRY_DIR  = REPO_ROOT / "neural_geometry"
EVALUATION_SUITE_DIR = REPO_ROOT / "evaluation_suite"

# Dataset locations
DATASET_DIR          = ACQUISITION_DIR  / "dataset"
SPECIMENS_DIR        = DATASET_DIR      / "specimens"
GROUND_TRUTH_DIR     = DATASET_DIR      / "ground_truth"
GROUND_TRUTH_CSV     = GROUND_TRUTH_DIR / "registry.csv"

# Pipeline output locations
RECON_OUTPUTS_DIR    = PROCEDURE_ALPHA_DIR  / "outputs"
TRAINED_MODELS_DIR   = BIOMASS_ENGINE_DIR   / "trained"
EVAL_REPORTS_DIR     = EVALUATION_SUITE_DIR / "reports"
EVAL_FIGURES_DIR     = EVALUATION_SUITE_DIR / "figures"

# ---------------------------------------------------------------------------
# Kinect v2 camera intrinsics — IR / depth stream at 512×424
# Kinect v2 factory defaults; override at runtime from device.getIrCameraParams()
# ---------------------------------------------------------------------------
KINECT_FX    = 365.456
KINECT_FY    = 365.456
KINECT_CX    = 254.878
KINECT_CY    = 205.395
DEPTH_WIDTH  = 512
DEPTH_HEIGHT = 424
COLOR_WIDTH  = 1920
COLOR_HEIGHT = 1080
DEPTH_MIN_MM = 300      # < 0.3 m  → sensor noise
DEPTH_MAX_MM = 6000     # > 6.0 m  → out of range

# ---------------------------------------------------------------------------
# Acquisition geometry — 30-degree angular protocol
# 12 evenly-spaced views around the plant (0°, 30°, 60°, …, 330°)
# Two cameras (A = green, B = red) fire simultaneously at each position.
# ---------------------------------------------------------------------------
ANGULAR_STEP_DEG     = 30
N_VIEWS              = 360 // ANGULAR_STEP_DEG          # 12
CAPTURE_ANGLES_DEG   = list(range(0, 360, ANGULAR_STEP_DEG))   # [0,30,…,330]
CAPTURE_ANGLES_RAD   = [np.deg2rad(a) for a in CAPTURE_ANGLES_DEG]

# Legacy 4-view protocol (kept for backward compatibility with existing dataset)
LEGACY_ANGLES_DEG    = [0, 90, 180, 270]

# ---------------------------------------------------------------------------
# Point-cloud preprocessing / ROI filter (metres)
# ---------------------------------------------------------------------------
ROI_X_MIN, ROI_X_MAX = -0.5,  0.5
ROI_Y_MIN, ROI_Y_MAX = -0.6,  0.65
ROI_Z_MIN, ROI_Z_MAX =  0.2,  1.5
SOR_K                = 50       # neighbours for statistical outlier removal
SOR_ALPHA            = 1.0      # std-dev multiplier

# ---------------------------------------------------------------------------
# ICP / registration
# ---------------------------------------------------------------------------
ICP_MAX_ITER         = 300
ICP_TOLERANCE        = 1e-6
ICP_MAX_CORR_DIST_M  = 0.10     # 100 mm

# ---------------------------------------------------------------------------
# Voxel / mesh reconstruction
# ---------------------------------------------------------------------------
VOXEL_SIZE_M         = 0.007    # 7 mm — speed/quality trade-off

# ---------------------------------------------------------------------------
# Specimen identifier format
# {SPECIES_CODE}{SEQ:03d}_{DATE_ISO}_{BATCH_CODE}
# e.g.  DG001_20260609_B01  = Duranta Gold #1, 9 Jun 2026, Batch 1
# ---------------------------------------------------------------------------
SPECIES_CODES = {
    "Duranta Gold mini":  "DG",
    "Duranta repens":     "DR",
    "Ficus benjamina":    "FB",
    "Unknown":            "XX",
}

# ---------------------------------------------------------------------------
# File-name templates for per-specimen view files
# view_{ANGLE:03d}deg_cam{LABEL}_{MODALITY}.{EXT}
# e.g. view_030deg_camA_rgb.jpg  |  view_030deg_camA_depth.npy
# ---------------------------------------------------------------------------
def view_filename(angle_deg: int, cam_label: str, modality: str, ext: str) -> str:
    """Return a canonical view filename."""
    return f"view_{angle_deg:03d}deg_cam{cam_label}_{modality}.{ext}"


# ---------------------------------------------------------------------------
# Legacy data_collection  → new specimen path helper
# ---------------------------------------------------------------------------
LEGACY_DATA_COLLECTION = REPO_ROOT / "data_collection"
LEGACY_RECON_OUTPUT    = REPO_ROOT / "reconstruction_output"

def legacy_depth_path(angle_deg: int, plant_id: int) -> Path:
    return LEGACY_DATA_COLLECTION / f"{angle_deg}_degrees_depth_plant_{plant_id}.npy"

def legacy_rgb_path(angle_deg: int, plant_id: int) -> Path:
    return LEGACY_DATA_COLLECTION / f"{angle_deg}_degrees_rgb_plant_{plant_id}.npy"
