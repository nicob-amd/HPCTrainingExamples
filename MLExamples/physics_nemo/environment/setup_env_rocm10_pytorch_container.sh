#!/usr/bin/env bash
# ROCm 10 variant using AMD's official prebuilt PyTorch+ROCm container,
# instead of building torch from wheels on top of a bare ROCm base image
# (that's what setup_env_rocm10.sh does).
#
# Container pulled from:
#   docker://rocm/pytorch:rocm10.0_ubuntu24.04_py3.12_pytorch_release_2.11.0
#   (https://hub.docker.com/r/rocm/pytorch/tags)
#
# That image ships torch 2.11.0+rocm10.0.0 (+ torchvision, + apex) already
# installed system-wide at /opt/venv, built with AMDGPU_FAMILY=device-all --
# i.e. every GPU arch AMD publishes a backend for, not just gfx942/MI300A --
# so unlike setup_env_rocm10.sh there is no torch/torchvision install step
# here at all. Chosen py3.12 + torch 2.11.0 tags to match setup_env_rocm10.sh
# exactly, for an apples-to-apples comparison of the two install paths.
#
# Differences from setup_env_rocm10.sh that matter:
#
#   1. No uv, no fresh Python install, no venv -- everything runs through
#      the container's own /opt/venv Python 3.12.
#   2. Packages can't be installed *into* the container itself: a `.sif`
#      built via `apptainer pull` is a read-only filesystem, so `pip
#      install` into /opt/venv would not persist between `apptainer exec`
#      invocations. Instead, physicsnemo and the rest of the workshop's
#      deps are installed with `pip install --target=<host dir>`, and that
#      host directory is bind-mounted + added to PYTHONPATH at run time
#      (see the "Running it" section below) -- this is a plain extra
#      site-packages directory, not a venv.
#   3. `timm` needs `--no-deps` here in a way it didn't in
#      setup_env_rocm10.sh. There, torch is installed and pinned first
#      (via --index-url scoped to AMD's index), so by the time `timm` is
#      installed pip already sees a satisfying torch and leaves it alone.
#      Here, doing a plain `pip install timm` (no matching --index-url
#      pin) let pip's resolver disregard the already-installed ROCm torch
#      and start pulling a stock PyPI `torch==2.14.0` (CUDA build) plus the
#      *entire* NVIDIA CUDA toolchain (cuDNN, NCCL, cuBLAS, cuSolver,
#      cuFFT, cuSparse, Triton, ~3GB+ of downloads) as a `timm` dependency
#      -- caught and killed before anything actually landed in the target
#      dir, then fixed by installing `timm` (+ its two small non-torch
#      deps, `huggingface_hub` and `safetensors`) with `--no-deps`
#      separately from the rest of the batch.
#
# Usage (from physics_nemo/environment/):
#   apptainer pull workshop_rocm10_pytorch.sif \
#     docker://rocm/pytorch:rocm10.0_ubuntu24.04_py3.12_pytorch_release_2.11.0
#   apptainer exec --rocm \
#     --bind $(pwd)/site-packages_rocm10_pytorch_container:/extra \
#     workshop_rocm10_pytorch.sif \
#     bash /workshop/environment/setup_env_rocm10_pytorch_container.sh
#   (adjust binds as needed -- this script itself only needs to reach the
#   /extra target dir it installs into; it does not read/write anywhere
#   else under $HERE.)
#
# Running the workshop afterward needs BOTH the site-packages bind AND
# PYTHONPATH set (unlike setup_env_rocm10.sh's venv, which is fully
# self-contained and needs neither):
#   apptainer exec --rocm \
#     --bind $(pwd)/site-packages_rocm10_pytorch_container:/extra \
#     --bind $(pwd)/..:/physics_nemo --pwd /physics_nemo \
#     --env PYTHONPATH=/extra \
#     --env FORCE_ADAM_MI300A=1 --env RANK=0 --env WORLD_SIZE=1 \
#     --env LOCAL_RANK=0 --env MASTER_ADDR=127.0.0.1 --env MASTER_PORT=29500 \
#     workshop_rocm10_pytorch.sif \
#     /opt/venv/bin/python3 -u /physics_nemo/workshop_crash_surrogate.py
#
# Result: verified end-to-end on real MI300A hardware -- same
# bq_torch_patch.py fix applies unchanged, toy training loop converges
# normally, and the existing pretrained checkpoint (no retraining)
# evaluates with the same accuracy as the setup_env_rocm10.sh path (see
# TESTING.md), confirming the fix holds across two independent
# torch+ROCm10 build paths, not just one.

set -eux

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET="${1:-/extra}"

echo "=== [0/3] Verifying the container's preinstalled torch (never reinstalled) ==="
/opt/venv/bin/python3 -c "
import torch
print('torch:', torch.__version__)
assert 'rocm10.0.0' in torch.__version__, 'expected the container-provided ROCm 10 torch build!'
print('cuda/HIP available:', torch.cuda.is_available())
"

mkdir -p "$TARGET"

echo "=== [1/3] Installing physicsnemo (--no-deps, same reasoning as setup_env_rocm10.sh) ==="
/opt/venv/bin/pip install --target="$TARGET" --no-deps "nvidia-physicsnemo==2.1.1"

echo "=== [2/3] Installing dependency-resolved packages (timm excluded -- see header) ==="
/opt/venv/bin/pip install --target="$TARGET" \
  pyvista tabulate torchinfo tqdm rich einops omegaconf hydra-core requests h5py \
  onnx treelib termcolor gitpython s3fs cftime pandas jaxtyping nvtx \
  matplotlib jupyterlab ipykernel

echo "=== [3/3] Installing torch-dependency-unaware packages (--no-deps) + their real small deps ==="
# timm: see header comment -- must be --no-deps here, unlike setup_env_rocm10.sh.
/opt/venv/bin/pip install --target="$TARGET" --no-deps timm huggingface_hub safetensors
# warp-lang / tensordict: same reasoning as setup_env_rocm10.sh (avoids pulling
# an unused NVIDIA CUDA JIT toolchain / duplicate torch via their deps).
/opt/venv/bin/pip install --target="$TARGET" --no-deps warp-lang tensordict
/opt/venv/bin/pip install --target="$TARGET" pyvers cloudpickle orjson importlib-metadata

echo "=== Verifying (with PYTHONPATH pointed at the install target) ==="
PYTHONPATH="$TARGET" /opt/venv/bin/python3 -c "
import torch; print('torch:', torch.__version__)
import physicsnemo; print('physicsnemo:', physicsnemo.__version__)
import pyvista; print('pyvista:', pyvista.__version__)
import hydra, omegaconf
print('hydra-core:', hydra.__version__, '| omegaconf:', omegaconf.__version__)
import warp, tensordict, timm  # noqa: F401 -- import-only smoke check
print('warp / tensordict / timm import OK')
" | tee "$HERE/INSTALL_LOG_rocm10_pytorch_container.txt"
echo "=== Full pip list (target dir) ===" | tee -a "$HERE/INSTALL_LOG_rocm10_pytorch_container.txt"
/opt/venv/bin/pip list --path "$TARGET" >> "$HERE/INSTALL_LOG_rocm10_pytorch_container.txt"

echo "Done. Extra site-packages at: $TARGET (set PYTHONPATH to this dir when running the workshop)"
