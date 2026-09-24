#!/usr/bin/env bash
# ROCm 7.2.4 + torch 2.9.1 -- baseline/control test.
#
# Same pinned-wheel repo.radeon.com scheme as setup_env.sh (7.2.1), bumped
# to the latest stable ROCm 7.x point release, but torch held at 2.9.1 --
# the exact version already confirmed working on ROCm 7.2.1. Purpose: rule
# out "7.2.4 itself is broken" as a confound before testing 2.10.0 on the
# same ROCm release (see setup_env_rocm7.2.4_torch2.10.sh). Expected result:
# this should work exactly like the original 7.2.1/2.9.1 run.
#
# Usage (from physics_nemo/environment/):
#   apptainer pull workshop_base_rocm724.sif docker://rocm/dev-ubuntu-22.04:7.2.4-complete
#   apptainer exec --bind $(pwd)/..:/workshop workshop_base_rocm724.sif \
#     bash /workshop/environment/setup_env_rocm7.2.4_torch2.9.sh
#
# Result: a venv at physics_nemo/environment/venv_rocm724_torch29 on the
# HOST filesystem.

set -eux

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export UV_INSTALL_DIR="$HERE/.uv-bin"
export UV_PYTHON_INSTALL_DIR="$HERE/.uv-python"
export UV_CACHE_DIR="$HERE/.uv-cache"
mkdir -p "$UV_INSTALL_DIR" "$UV_PYTHON_INSTALL_DIR" "$UV_CACHE_DIR"

if [ ! -x "$UV_INSTALL_DIR/uv" ]; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$UV_INSTALL_DIR:$PATH"

export ROCM_PATH=/opt/rocm
export LD_LIBRARY_PATH="$ROCM_PATH/lib:$ROCM_PATH/lib64"
export PATH="$ROCM_PATH/bin:$PATH"

ROCM_REPO="https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.4"
TORCH_WHEEL="${ROCM_REPO}/torch-2.9.1%2Brocm7.2.4.lw.git39497456-cp312-cp312-linux_x86_64.whl"
TORCHVISION_WHEEL="${ROCM_REPO}/torchvision-0.24.0%2Brocm7.2.4.gitb919bd0c-cp312-cp312-linux_x86_64.whl"

echo "=== [0/4] Ensuring Python 3.12 (via uv, no apt/root needed) ==="
uv python install 3.12

VENV="$HERE/venv_rocm724_torch29"
uv venv --python 3.12 "$VENV"
PY="$VENV/bin/python3"

echo "=== [1/4] Installing ROCm torch + torchvision + physicsnemo (--no-deps) ==="
uv pip install --python "$PY" --no-deps "$TORCH_WHEEL"
uv pip install --python "$PY" --no-deps "$TORCHVISION_WHEEL"
uv pip install --python "$PY" --no-deps "nvidia-physicsnemo==2.1.1"

echo "=== [2/4] Installing dependency-resolved packages ==="
uv pip install --python "$PY" \
  typing_extensions filelock networkx sympy jinja2 fsspec \
  numpy pyvista tabulate tensorboard torchinfo tqdm rich \
  einops timm omegaconf hydra-core requests h5py \
  onnx treelib termcolor gitpython s3fs cftime pandas urllib3 \
  importlib-metadata jaxtyping nvtx packaging \
  jupyterlab ipykernel matplotlib
uv pip install --python "$PY" --no-deps warp-lang

echo "=== [3/4] Installing tensordict (--no-deps) + its actual small deps ==="
uv pip install --python "$PY" --no-deps "tensordict"
uv pip install --python "$PY" pyvers cloudpickle orjson

echo "=== [4/4] Reinstalling ROCm torch/torchvision (cheap safety net -- served from uv's local cache, no network) ==="
uv pip install --python "$PY" --no-deps "$TORCH_WHEEL"
uv pip install --python "$PY" --no-deps "$TORCHVISION_WHEEL"

echo "=== Verifying ==="
"$PY" -c "
import torch
print('torch:', torch.__version__)
assert '+rocm' in torch.__version__, 'torch was clobbered with a non-ROCm build!'
" | tee "$HERE/INSTALL_LOG_rocm724_torch29.txt"
"$PY" -c "import physicsnemo; print('physicsnemo:', physicsnemo.__version__)" | tee -a "$HERE/INSTALL_LOG_rocm724_torch29.txt"
"$PY" -c "import pyvista; print('pyvista:', pyvista.__version__)" | tee -a "$HERE/INSTALL_LOG_rocm724_torch29.txt"
"$PY" -c "import hydra, omegaconf; print('hydra-core:', hydra.__version__); print('omegaconf:', omegaconf.__version__)" | tee -a "$HERE/INSTALL_LOG_rocm724_torch29.txt"
echo "=== Full pip freeze ===" | tee -a "$HERE/INSTALL_LOG_rocm724_torch29.txt"
uv pip freeze --python "$PY" >> "$HERE/INSTALL_LOG_rocm724_torch29.txt"

echo "Done. venv at: $VENV"
