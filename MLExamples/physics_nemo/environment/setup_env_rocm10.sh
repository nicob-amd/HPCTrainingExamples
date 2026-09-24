#!/usr/bin/env bash
# ROCm 10 variant of setup_env.sh.
#
# Differs from setup_env.sh in exactly one way that matters: where torch/
# torchvision come from. ROCm 10 wheels are NOT published under the old
# repo.radeon.com/rocm/manylinux/rocm-rel-X.Y.Z/ layout (that repo tops out
# at rocm-rel-7.2.4 as of this writing) -- AMD moved to a new, modular pip
# index for ROCm 10+ at https://stable.repo.amd.com/rocm/whl-next/, where
# `torch`/`torchvision` are thin meta-packages that pull in a GPU-arch-
# specific backend wheel (e.g. `amd-torch-device-gfx942` for MI300A/MI300X)
# via an extras tag: `torch[device-gfx942]==<ver>+rocm10.0.0`.
#
# Everything else (uv-managed Python 3.12, plain venv, --no-deps batching
# for torch-dependent packages, the tensordict fix) is identical to
# setup_env.sh -- see that file's comments for the reasoning.
#
# Usage (from physics_nemo/environment/):
#   apptainer pull workshop_base_rocm10.sif docker://rocm/dev-ubuntu-22.04:10.0.0-full
#   apptainer exec --bind $(pwd)/..:/workshop workshop_base_rocm10.sif \
#     bash /workshop/environment/setup_env_rocm10.sh
#
# Result: a venv at physics_nemo/environment/venv_rocm10 on the HOST
# filesystem (kept separate from setup_env.sh's `venv`, so you can compare
# the two ROCm versions side by side without clobbering either).

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

# AMD's new modular ROCm pip index (replaces repo.radeon.com/rocm/manylinux
# for ROCm 10+). Contains torch/torchvision/torchaudio as thin meta-packages
# plus per-GPU-arch backend wheels (amd-torch-device-gfx942, etc.) and all
# of torch's own core deps (filelock, fsspec, jinja2, numpy, sympy, triton,
# ...) -- so it's self-contained enough to resolve without ever touching
# PyPI, as long as we pass it as --index-url (replaces the default index)
# rather than --extra-index-url (which would merely add to it).
ROCM_WHL_INDEX="https://stable.repo.amd.com/rocm/whl-next/"
GFX_ARCH="gfx942"  # MI300A / MI300X
ROCM_VERSION="10.0.0"
# 2.13.0+rocm10.0.0 hits a real HIP kernel bug (hipErrorInvalidConfiguration)
# in torch.cdist during physicsnemo's RadiusSearch torch-fallback path on
# MI300A/gfx942. Testing 2.11.0 (the oldest torch build published for ROCm
# 10) to see if this is a torch-2.13-specific regression or affects all of
# ROCm 10. torchvision pinned to its matching paired release (2.11.0 <->
# 0.26.0, same as upstream PyTorch's normal torch/torchvision pairing).
TORCH_VERSION="2.11.0"
TORCHVISION_VERSION="0.26.0"

echo "=== [0/4] Ensuring Python 3.12 (via uv, no apt/root needed) ==="
uv python install 3.12

VENV="$HERE/venv_rocm10"
uv venv --python 3.12 "$VENV"
PY="$VENV/bin/python3"

echo "=== [1/4] Installing ROCm 10 torch + torchvision (device-${GFX_ARCH}) + physicsnemo (--no-deps) ==="
# NOT --no-deps here: the device-arch backend wheel (amd-torch-device-gfx942)
# is only pulled in via the `[device-gfx942]` extra, which --no-deps would
# skip entirely, leaving a useless torch meta-package with no GPU backend.
# Scoping to --index-url (not --extra-index-url) keeps resolution entirely
# inside AMD's index, so this can't accidentally reach out to PyPI for a
# stock CUDA build.
uv pip install --python "$PY" --index-url "$ROCM_WHL_INDEX" \
  "torch[device-${GFX_ARCH}]==${TORCH_VERSION}+rocm${ROCM_VERSION}" \
  "torchvision[device-${GFX_ARCH}]==${TORCHVISION_VERSION}+rocm${ROCM_VERSION}"
uv pip install --python "$PY" --no-deps "nvidia-physicsnemo==2.1.1"

echo "=== [2/4] Installing dependency-resolved packages ==="
# warp-lang installed separately below with --no-deps: on Linux its wheel
# pulls a large NVIDIA CUDA JIT toolchain (nvidia-cublas, nvidia-cudnn-cu13,
# nvidia-nccl-cu13, triton, cuda-bindings, cuda-toolkit, ...) that's dead
# weight here -- there's no NVIDIA CUDA driver on a ROCm box, so warp has no
# GPU backend regardless (physicsnemo's RadiusSearch falls back to a plain
# torch implementation at runtime either way). `import warp` itself only
# needs numpy (already installed below), confirmed by a real run where warp
# imported and printed its own "no CUDA driver" warning without needing any
# of the CUDA packages to be present.
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

echo "=== [4/4] Reinstalling ROCm 10 torch/torchvision (cheap safety net -- served from uv's local cache, no network) ==="
uv pip install --python "$PY" --index-url "$ROCM_WHL_INDEX" --no-deps \
  "torch[device-${GFX_ARCH}]==${TORCH_VERSION}+rocm${ROCM_VERSION}" \
  "torchvision[device-${GFX_ARCH}]==${TORCHVISION_VERSION}+rocm${ROCM_VERSION}"

echo "=== Verifying ==="
"$PY" -c "
import torch
print('torch:', torch.__version__)
assert 'rocm${ROCM_VERSION}' in torch.__version__, 'torch was clobbered or is not the expected ROCm 10 build!'
" | tee "$HERE/INSTALL_LOG_rocm10.txt"
"$PY" -c "import physicsnemo; print('physicsnemo:', physicsnemo.__version__)" | tee -a "$HERE/INSTALL_LOG_rocm10.txt"
"$PY" -c "import pyvista; print('pyvista:', pyvista.__version__)" | tee -a "$HERE/INSTALL_LOG_rocm10.txt"
"$PY" -c "import hydra, omegaconf; print('hydra-core:', hydra.__version__); print('omegaconf:', omegaconf.__version__)" | tee -a "$HERE/INSTALL_LOG_rocm10.txt"
echo "=== Full pip freeze ===" | tee -a "$HERE/INSTALL_LOG_rocm10.txt"
uv pip freeze --python "$PY" >> "$HERE/INSTALL_LOG_rocm10.txt"

echo "Done. venv at: $VENV"
