#!/bin/bash
# Self-contained ROCm venv installer for the workshop_crash_surrogate folder.
#
# FALLBACK path — prefer environment/Apptainer.def if Apptainer is
# available. This script installs the identical pinned ROCm wheel set
# (same recipe as the source project's install_physicsnemo_env.sh) but has
# no dependency on that project's directory layout — it only assumes ROCm
# is already installed on this machine (module load or /opt/rocm) and that
# `uv` is on PATH.
#
# Usage:
#   ./install_venv.sh                # creates ./venv here
#   VENV=/path/to/venv ./install_venv.sh
set -euo pipefail

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="${VENV:-$BASE_DIR/venv}"
LOG="$BASE_DIR/INSTALL_LOG.txt"

echo "Base directory: $BASE_DIR"
echo "Venv target: $VENV"
echo "Install log: $LOG"

echo "=== Detecting ROCm (using whatever is currently active in this shell) ===" | tee "$LOG"

ROCM_PATH="${ROCM_PATH:-}"
if [ -z "$ROCM_PATH" ] && [ -d /opt/rocm ]; then
  ROCM_PATH=$(readlink -f /opt/rocm)
fi

ROCM_VERSION=""
if [ -n "$ROCM_PATH" ] && [ -f "$ROCM_PATH/.info/version" ]; then
  ROCM_VERSION=$(cat "$ROCM_PATH/.info/version" | grep -oE '^[0-9]+\.[0-9]+\.[0-9]+')
elif command -v hipconfig >/dev/null 2>&1; then
  ROCM_VERSION=$(hipconfig --version 2>/dev/null | grep -oE '^[0-9]+\.[0-9]+\.[0-9]+')
fi

if [ -z "$ROCM_VERSION" ]; then
  echo "ERROR: could not auto-detect ROCm version. Set ROCM_PATH and re-run, or" | tee -a "$LOG"
  echo "       check 'module avail rocm' / '/opt/rocm' manually." | tee -a "$LOG"
  exit 1
fi

echo "Detected ROCm version: $ROCM_VERSION" | tee -a "$LOG"
echo "ROCM_PATH: $ROCM_PATH" | tee -a "$LOG"

export ROCM_PATH
export LD_LIBRARY_PATH="$ROCM_PATH/lib:$ROCM_PATH/lib64:${LD_LIBRARY_PATH:-}"
export PATH="$ROCM_PATH/bin:$PATH"

ROCM_REPO="https://repo.radeon.com/rocm/manylinux/rocm-rel-${ROCM_VERSION}"

# Known-good wheel builds per ROCm release, pinned to match the checkpoint
# and toy-training code in this folder exactly (torch>=2.9 is also required
# by the `muon` optimizer used in checkpoint/config.yaml and
# checkpoint/toy_config.yaml). Add a case entry here if a new ROCm version
# needs a different pinned wheel build tag.
case "$ROCM_VERSION" in
  7.2.1)
    TORCH_WHEEL="${ROCM_REPO}/torch-2.9.1%2Brocm7.2.1.lw.gitff65f5bc-cp312-cp312-linux_x86_64.whl"
    TORCHVISION_WHEEL="${ROCM_REPO}/torchvision-0.24.0%2Brocm7.2.1.gitb919bd0c-cp312-cp312-linux_x86_64.whl"
    ;;
  *)
    echo "WARNING: no pinned wheel known for ROCm ${ROCM_VERSION}." | tee -a "$LOG"
    echo "Check ${ROCM_REPO}/ for available torch/torchvision wheel filenames" | tee -a "$LOG"
    echo "and add a case entry in this script, then re-run." | tee -a "$LOG"
    exit 1
    ;;
esac

echo | tee -a "$LOG"
echo "=== Checking uv ===" | tee -a "$LOG"
command -v uv >/dev/null 2>&1 || { echo "ERROR: uv not found. Install uv first (https://astral.sh/uv)." | tee -a "$LOG"; exit 1; }

echo | tee -a "$LOG"
echo "=== Creating venv at $VENV ===" | tee -a "$LOG"
uv venv --python 3.12 "$VENV"
PY="$VENV/bin/python3"

echo | tee -a "$LOG"
echo "=== [1/4] Installing ROCm torch + torchvision + physicsnemo (--no-deps) ===" | tee -a "$LOG"
uv pip install --python "$PY" --no-deps "$TORCH_WHEEL"
uv pip install --python "$PY" --no-deps "$TORCHVISION_WHEEL"
uv pip install --python "$PY" --no-deps "nvidia-physicsnemo==2.1.1"

echo | tee -a "$LOG"
echo "=== [2/4] Installing dependency-resolved packages ===" | tee -a "$LOG"
# NOTE: tensordict (not warp-lang) depends on torch and will pull in a CUDA
# build if installed normally. It is intentionally kept in the --no-deps group.
uv pip install --python "$PY" \
  typing_extensions filelock networkx sympy jinja2 fsspec \
  numpy pyvista tabulate tensorboard torchinfo tqdm rich \
  warp-lang einops timm omegaconf hydra-core requests h5py \
  onnx treelib termcolor gitpython s3fs cftime pandas urllib3 \
  importlib-metadata jaxtyping nvtx packaging \
  jupyterlab ipykernel matplotlib

echo | tee -a "$LOG"
echo "=== [3/4] Installing tensordict (--no-deps) ===" | tee -a "$LOG"
uv pip install --python "$PY" --no-deps "tensordict"

echo | tee -a "$LOG"
echo "=== [4/4] Reinstalling ROCm torch/torchvision (guards against any resolver clobber) ===" | tee -a "$LOG"
uv pip install --python "$PY" --no-deps "$TORCH_WHEEL"
uv pip install --python "$PY" --no-deps "$TORCHVISION_WHEEL"

echo | tee -a "$LOG"
echo "=== Verifying ===" | tee -a "$LOG"
"$PY" -c "
import torch
print('torch:', torch.__version__)
print('cuda available (HIP device visible):', torch.cuda.is_available())
print('device count:', torch.cuda.device_count())
if torch.cuda.is_available():
    print('device name:', torch.cuda.get_device_name(0))
assert '+rocm' in torch.__version__, 'torch was clobbered with a non-ROCm build!'
" 2>&1 | tee -a "$LOG"
"$PY" -c "import physicsnemo; print('physicsnemo:', physicsnemo.__version__)" 2>&1 | tee -a "$LOG"
"$PY" -c "import pyvista; print('pyvista:', pyvista.__version__)" 2>&1 | tee -a "$LOG"
"$PY" -c "import hydra, omegaconf; print('hydra-core:', hydra.__version__); print('omegaconf:', omegaconf.__version__)" 2>&1 | tee -a "$LOG"

echo | tee -a "$LOG"
echo "=== Full pip freeze ===" | tee -a "$LOG"
uv pip freeze --python "$PY" >> "$LOG"

echo
echo "Done. Venv at: $VENV"
echo "Install log written to: $LOG"
echo
echo "Activate with: source $VENV/bin/activate"
echo "Then launch the notebook with: jupyter lab workshop_crash_surrogate.ipynb"
