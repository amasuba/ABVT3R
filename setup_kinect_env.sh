#!/usr/bin/env bash
# Setup script for the ABVT3R Kinect v2 environment.
#
# Creates abvt310/ (Python 3.10 venv) and installs all dependencies
# including pylibfreenect2, which requires:
#   - Python <= 3.10 (dev headers must be installed)
#   - Cython < 3
#   - libfreenect2 installed system-wide (headers + shared library)
#
# Prerequisites (install once with apt):
#   sudo apt-get install python3.10-dev libfreenect2-dev
#
# Usage:
#   bash setup_kinect_env.sh
#   source abvt310/bin/activate
#   python host.py

set -e

VENV_DIR="abvt310"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== ABVT3R Kinect v2 environment setup ==="

# Check Python 3.10
if ! command -v python3.10 &>/dev/null; then
    echo "ERROR: python3.10 not found. Install with: sudo apt-get install python3.10"
    exit 1
fi

# Check libfreenect2
if ! ldconfig -p 2>/dev/null | grep -q libfreenect2; then
    echo "WARNING: libfreenect2 not found in ldconfig. Build and install it first:"
    echo "  See libfreenect2/README.md"
fi

# Create venv
cd "$SCRIPT_DIR"
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating Python 3.10 virtual environment at $VENV_DIR/ ..."
    python3.10 -m venv "$VENV_DIR"
else
    echo "Virtual environment $VENV_DIR/ already exists, updating packages..."
fi

PIP="$SCRIPT_DIR/$VENV_DIR/bin/pip"

echo "Installing base dependencies..."
"$PIP" install --upgrade pip
"$PIP" install "cython<3" numpy scipy scikit-learn opencv-python matplotlib pyserial

echo "Installing pylibfreenect2 (Kinect v2 Python bindings)..."
"$PIP" install pylibfreenect2 --no-build-isolation

echo ""
echo "=== Setup complete ==="
echo "Activate the environment with:"
echo "  source $VENV_DIR/bin/activate"
echo "Then run the host:"
echo "  python host.py"
