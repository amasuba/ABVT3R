#!/usr/bin/env bash
# =============================================================================
# ubuntu_setup.sh
# =============================================================================
# ABVT3R — Level-1 Classical Pipeline: Ubuntu Setup & Runner
# Aaron Masuba | MEng Dissertation | University of Pretoria
#
# Run this script once after transferring the repo to Ubuntu:
#   chmod +x ubuntu_setup.sh
#   ./ubuntu_setup.sh
#
# What it does:
#   1. Checks Python 3.10+ is available
#   2. Installs all pip dependencies
#   3. Verifies key imports
#   4. Smoke-tests the pipeline on plant_1 (data_collection/)
#   5. Prompts you to run the full batch once you have all 40 plants' data
# =============================================================================

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

echo ""
echo "============================================================"
echo "  ABVT3R — Ubuntu Environment Setup"
echo "  Repo: $REPO_DIR"
echo "============================================================"
echo ""

# -----------------------------------------------------------------------------
# 1. Python version check
# -----------------------------------------------------------------------------
info "Checking Python version..."
PY=$(python3 --version 2>&1 | awk '{print $2}')
MAJOR=$(echo "$PY" | cut -d. -f1)
MINOR=$(echo "$PY" | cut -d. -f2)

if [[ "$MAJOR" -lt 3 ]] || [[ "$MAJOR" -eq 3 && "$MINOR" -lt 10 ]]; then
    error "Python 3.10+ required. Found: $PY"
fi
info "Python $PY ✓"

# -----------------------------------------------------------------------------
# 2. Upgrade pip
# -----------------------------------------------------------------------------
info "Upgrading pip..."
python3 -m pip install --upgrade pip --break-system-packages -q

# -----------------------------------------------------------------------------
# 3. Install dependencies
# -----------------------------------------------------------------------------
info "Installing core dependencies..."
python3 -m pip install \
    "numpy>=1.20" \
    "scipy>=1.6" \
    "scikit-learn>=1.0" \
    "opencv-python>=4.5" \
    "matplotlib>=3.3" \
    --break-system-packages -q
info "Core deps installed ✓"

info "Installing Open3D (optional — mesh export)..."
python3 -m pip install open3d --break-system-packages -q && \
    info "open3d installed ✓" || \
    warn "open3d not installed — mesh .ply/.obj export will be skipped (pipeline still runs)"

# -----------------------------------------------------------------------------
# 4. Verify imports
# -----------------------------------------------------------------------------
info "Verifying imports..."
python3 - <<'PYCHECK'
import sys
failures = []
for mod in ["numpy", "scipy", "sklearn", "cv2", "matplotlib"]:
    try:
        __import__(mod)
    except ImportError:
        failures.append(mod)
if failures:
    print(f"MISSING: {failures}")
    sys.exit(1)
else:
    print("  All required imports OK")
PYCHECK
info "Import check passed ✓"

# -----------------------------------------------------------------------------
# 5. Smoke test — plant_1
# -----------------------------------------------------------------------------
echo ""
info "Running smoke test on plant_1 (data_collection/)..."

PLANT1_OK=true
for angle in 0 90 180 270; do
    f="data_collection/${angle}_degrees_depth_plant_1.npy"
    if [[ ! -f "$f" ]]; then
        warn "Missing: $f"
        PLANT1_OK=false
    fi
done

if [[ "$PLANT1_OK" == true ]]; then
    info "All plant_1 depth files found. Running pipeline..."
    python3 -c "
from procedure_alpha.pipeline import ProcedureAlpha
pa = ProcedureAlpha()
result = pa.run_legacy(1)
mq = result['reconstruction'].get('mesh_quality', {})
sq = result['reconstruction'].get('surface_quality', {})
print()
print('=== Smoke Test Result ===')
print(f\"  Volume       : {mq.get('volume', 0)*1e6:.1f} cm³\")
print(f\"  Surface area : {mq.get('surface_area', 0)*1e4:.1f} cm²\")
print(f\"  Quality Q    : {sq.get('overall_quality', 0):.4f}\")
print(f\"  Elapsed      : {result['elapsed_s']:.1f}s\")
print('  Smoke test PASSED ✓')
"
else
    warn "plant_1 depth files incomplete — skipping smoke test."
    warn "Check that data_collection/ was copied correctly."
fi

# -----------------------------------------------------------------------------
# 6. GPU check (for later DINOv2 / Level-3 experiments)
# -----------------------------------------------------------------------------
echo ""
info "Checking GPU availability..."
python3 - <<'GPUCHECK'
try:
    import torch
    if torch.cuda.is_available():
        n = torch.cuda.device_count()
        for i in range(n):
            name = torch.cuda.get_device_name(i)
            mem  = torch.cuda.get_device_properties(i).total_memory / 1e9
            print(f"  GPU {i}: {name}  ({mem:.1f} GB VRAM)")
    else:
        print("  No CUDA GPU detected (CPU-only mode)")
except ImportError:
    print("  PyTorch not installed yet — GPU check skipped")
    print("  Install later with: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")
GPUCHECK

# -----------------------------------------------------------------------------
# 7. Next steps
# -----------------------------------------------------------------------------
echo ""
echo "============================================================"
echo "  SETUP COMPLETE — Next Steps"
echo "============================================================"
echo ""
echo "  AFTER collecting all 40 plants' depth files:"
echo ""
echo "  A) Run the full 40-plant reconstruction pipeline:"
echo "       python3 batch_run.py"
echo ""
echo "  B) Once all 40 plants are reconstructed, run LOOCV:"
echo "       python3 train_loocv.py"
echo ""
echo "  C) Results saved to:"
echo "       procedure_alpha/outputs/        ← reconstruction per plant"
echo "       evaluation_suite/reports/       ← loocv_results.csv"
echo "       evaluation_suite/figures/       ← loocv_scatter.png"
echo ""
echo "  IMPORTANT — Ground Truth Labels:"
echo "   weights.txt currently contains TOTAL MASS (plant + pot + soil)."
echo "   For correct AGB estimation you must:"
echo "     1. Weigh the whole setup (total_mass_kg)"
echo "     2. Remove plant at soil surface, weigh pot+soil (pot_mass_kg)"
echo "     3. Oven-dry the above-ground plant 72h at 70°C"
echo "     4. Weigh dry plant → this is agb_kg (your regression target)"
echo "     5. Fill the agb_kg column in acquisition/dataset/ground_truth/registry.csv"
echo "     6. Run: python3 train_loocv.py --use-registry"
echo ""
echo "============================================================"
echo ""
