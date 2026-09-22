#!/usr/bin/env bash
# Rootless alternative to Apptainer.def's %post section.
#
# Apptainer.def needs `apptainer build`, which requires fakeroot (and
# therefore a /etc/subuid + /etc/subgid entry) to run %post as root inside
# the container -- that's only needed there for `apt-get install python3.12`.
# This script does the exact same pinned-wheel install, but run via plain
# `apptainer exec` (no --fakeroot, runs as your own uid) against the base
# image pulled with `apptainer pull` (also no fakeroot needed, since pull
# does no %post). Python 3.12 comes from `uv python install` instead of
# apt, which needs no root. Net effect: identical resulting environment,
# zero privileged operations, works for any user on any node.
#
# Usage (from physics_nemo/environment/):
#   apptainer pull workshop_base.sif docker://rocm/dev-ubuntu-22.04:7.2.1-complete
#   apptainer exec --rocm --bind $(pwd)/..:/workshop workshop_base.sif \
#     bash /workshop/environment/setup_env.sh
#
# Result: a venv at physics_nemo/environment/venv on the HOST filesystem
# (persists after the container exits, reusable by later `apptainer exec`
# calls that just launch jupyter/python against it directly).

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

ROCM_REPO="https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.1"
TORCH_WHEEL="${ROCM_REPO}/torch-2.9.1%2Brocm7.2.1.lw.gitff65f5bc-cp312-cp312-linux_x86_64.whl"
TORCHVISION_WHEEL="${ROCM_REPO}/torchvision-0.24.0%2Brocm7.2.1.gitb919bd0c-cp312-cp312-linux_x86_64.whl"

echo "=== [0/4] Ensuring Python 3.12 (via uv, no apt/root needed) ==="
uv python install 3.12

VENV="$HERE/venv"
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
  warp-lang einops timm omegaconf hydra-core requests h5py \
  onnx treelib termcolor gitpython s3fs cftime pandas urllib3 \
  importlib-metadata jaxtyping nvtx packaging \
  jupyterlab ipykernel matplotlib

echo "=== [3/4] Installing tensordict (--no-deps) + its actual small deps ==="
# tensordict lists torch/torchvision as deps, so a normal (non --no-deps)
# install lets the resolver fetch a stock CUDA torch/torchvision from PyPI
# (multiple GB) before we'd overwrite it below -- wasted bandwidth for no
# benefit. Its only *real* extra runtime deps (found by following the
# ModuleNotFoundError traceback with --no-deps) are pyvers and cloudpickle,
# neither of which touches torch, so install those explicitly instead.
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
" | tee "$HERE/INSTALL_LOG.txt"
"$PY" -c "import physicsnemo; print('physicsnemo:', physicsnemo.__version__)" | tee -a "$HERE/INSTALL_LOG.txt"
"$PY" -c "import pyvista; print('pyvista:', pyvista.__version__)" | tee -a "$HERE/INSTALL_LOG.txt"
"$PY" -c "import hydra, omegaconf; print('hydra-core:', hydra.__version__); print('omegaconf:', omegaconf.__version__)" | tee -a "$HERE/INSTALL_LOG.txt"
echo "=== Full pip freeze ===" | tee -a "$HERE/INSTALL_LOG.txt"
uv pip freeze --python "$PY" >> "$HERE/INSTALL_LOG.txt"

echo "Done. venv at: $VENV"
