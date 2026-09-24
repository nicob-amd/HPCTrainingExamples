#!/usr/bin/env bash
# ROCm 7.14 variant of setup_env.sh -- structurally different from the other
# two scripts because neither repo.radeon.com/rocm/manylinux nor
# stable.repo.amd.com/rocm/whl-next publish a standalone torch/torchvision
# wheel for ROCm 7.14. The only matching build is AMD's pre-baked
# `rocm/pytorch` image, which already has torch/torchvision installed at
# /opt/venv/lib/python3.12/site-packages inside the image.
#
# NOT using `--system-site-packages` (tried first, doesn't work here):
# /opt/venv is itself a venv (confirmed via its own pyvenv.cfg: home=/usr/bin,
# include-system-site-packages=false). Pointing `uv venv --system-site-
# packages` at /opt/venv/bin/python3 makes uv walk up to *that* venv's OWN
# base interpreter (bare Ubuntu /usr/bin/python3.12) and only exposes ITS
# site-packages -- never /opt/venv's own site-packages, which is where torch
# actually lives. Confirmed the hard way: resulting venv's pyvenv.cfg showed
# home=/usr/bin, and `import torch` failed outright.
#
# Instead: build a completely normal, self-contained venv (own uv-managed
# Python 3.12, matching the other two scripts), then drop a .pth file
# pointing at /opt/venv/lib/python3.12/site-packages -- this adds that
# directory straight to sys.path, sidestepping the whole system-site-
# packages mechanism. (Standard CPython 3.12 ABI compatibility applies here,
# same as the other two scripts running repo.radeon.com/whl-next cp312
# wheels against a *different* uv-managed 3.12 build than they were
# compiled with -- already proven to work.)
#
# Pinned to torch 2.10.0 -- a build ROCm 10 doesn't offer at all (its oldest
# is 2.11.0), chosen as a data point closer to the known-good 2.9.1 rather
# than reusing the already-confirmed-buggy 2.11.0/2.13.0.
#
# Usage (from physics_nemo/environment/):
#   apptainer pull workshop_base_rocm714.sif \
#     docker://rocm/pytorch:rocm7.14.1_ubuntu24.04_py3.12_pytorch_release_2.10.0
#   apptainer exec --bind $(pwd)/..:/workshop workshop_base_rocm714.sif \
#     bash /workshop/environment/setup_env_rocm7.14.sh
#
# Result: a venv at physics_nemo/environment/venv_rocm714 on the HOST
# filesystem. torch/torchvision are NOT physically inside it -- resolved via
# the .pth file from the image's own /opt/venv at runtime, so this venv only
# works when run via apptainer against the SAME rocm/pytorch image it was
# built against (not the bare OS images used by the other two scripts).

set -eux

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export UV_INSTALL_DIR="$HERE/.uv-bin"
export UV_CACHE_DIR="$HERE/.uv-cache"
mkdir -p "$UV_INSTALL_DIR" "$UV_CACHE_DIR"

if [ ! -x "$UV_INSTALL_DIR/uv" ]; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$UV_INSTALL_DIR:$PATH"

# No manual ROCM_PATH/LD_LIBRARY_PATH exports here: this image is built to
# work out of the box for torch users, unlike the bare OS images used by
# the other two scripts.
#
# Hardcoded path, not `command -v python3`: Apptainer merges the CALLING
# HOST SHELL's environment into the container by default (no --cleanenv),
# so a polluted host $PATH (leftover exports from other venvs/modules used
# earlier in the same terminal session) can shadow the image's intended
# /opt/venv/bin-first ordering and silently resolve to the bare system
# python3 instead.
IMAGE_SITE_PACKAGES="/opt/venv/lib/python3.12/site-packages"
if ! /opt/venv/bin/python3 -c "import torch" >/dev/null 2>&1; then
    echo "ERROR: /opt/venv/bin/python3 can't import torch -- wrong image, or" \
         "its layout differs from" \
         "rocm7.14.1_ubuntu24.04_py3.12_pytorch_release_2.10.0." \
         "Check 'apptainer exec <image> find / -maxdepth 6 -iname torch -type d'" \
         "and update IMAGE_SITE_PACKAGES." >&2
    exit 1
fi

echo "=== [1/4] Ensuring Python 3.12 (via uv, no apt/root needed) + linking image torch ==="
uv python install 3.12

VENV="$HERE/venv_rocm714"
uv venv --python 3.12 "$VENV"
PY="$VENV/bin/python3"

# .pth file: adds the image's torch/torchvision straight to sys.path. Note
# this is NOT a normal "installed package" as far as uv/pip's own metadata
# is concerned -- `uv pip list`/dependency resolution still has zero record
# of torch existing, which is exactly why every torch-dependent package
# below still needs --no-deps despite `import torch` working fine.
VENV_SITE_PACKAGES="$VENV/lib/python3.12/site-packages"
echo "$IMAGE_SITE_PACKAGES" > "$VENV_SITE_PACKAGES/_rocm714_image_torch.pth"
"$PY" -c "import torch; print('torch (via .pth):', torch.__version__)"

echo "=== [2/4] Installing physicsnemo (--no-deps, torch/torchvision come from the image) ==="
uv pip install --python "$PY" --no-deps "nvidia-physicsnemo==2.1.1"

echo "=== [3/4] Installing dependency-resolved packages ==="
# uv has NO metadata record of torch (it's only on sys.path via the .pth
# file above, not pip-installed), so anything declaring torch/torchvision
# as a dependency must be --no-deps here or uv will fetch a stock CUDA
# build from PyPI to satisfy it. Confirmed the hard way on a first attempt
# without this split: pulled stock torch==2.14.0+torchvision==0.29.0+the
# full nvidia-cu13 runtime stack, which then shadowed the image's real
# torch in this venv's own site-packages. timm unconditionally depends on
# torch+torchvision (verified via PyPI metadata), so it needs --no-deps
# (same root cause as the warp-lang fix, different trigger).
# torchinfo/tensorboard were checked too and do NOT declare torch as a
# dependency, so they're safe in the normal batch. See
# PhysicsNemo/INSTALL.md section 2 for the original writeup of this exact
# pitfall (there, for the Lmod --system-site-packages module case).
uv pip install --python "$PY" \
  typing_extensions filelock networkx sympy jinja2 fsspec \
  numpy pyvista tabulate tensorboard torchinfo tqdm rich \
  einops omegaconf hydra-core requests h5py \
  onnx treelib termcolor gitpython s3fs cftime pandas urllib3 \
  importlib-metadata jaxtyping nvtx packaging \
  jupyterlab ipykernel matplotlib
uv pip install --python "$PY" --no-deps warp-lang timm
# timm's actual (non-torch) runtime deps, skipped by --no-deps above --
# confirmed via PyPI metadata: torch, torchvision, pyyaml, huggingface_hub,
# safetensors. Only the last three need adding explicitly.
uv pip install --python "$PY" pyyaml huggingface_hub safetensors

echo "=== [4/4] Installing tensordict (--no-deps) + its actual small deps ==="
uv pip install --python "$PY" --no-deps "tensordict"
uv pip install --python "$PY" pyvers cloudpickle orjson

echo "=== Verifying ==="
"$PY" -c "
import torch
print('torch:', torch.__version__)
assert '+rocm' in torch.__version__, 'torch is not a ROCm build -- .pth link or image is wrong!'
print('cuda/HIP device available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('device name:', torch.cuda.get_device_name(0))
" | tee "$HERE/INSTALL_LOG_rocm714.txt"
"$PY" -c "import physicsnemo; print('physicsnemo:', physicsnemo.__version__)" | tee -a "$HERE/INSTALL_LOG_rocm714.txt"
"$PY" -c "import pyvista; print('pyvista:', pyvista.__version__)" | tee -a "$HERE/INSTALL_LOG_rocm714.txt"
"$PY" -c "import hydra, omegaconf; print('hydra-core:', hydra.__version__); print('omegaconf:', omegaconf.__version__)" | tee -a "$HERE/INSTALL_LOG_rocm714.txt"
echo "=== Full pip freeze ===" | tee -a "$HERE/INSTALL_LOG_rocm714.txt"
uv pip freeze --python "$PY" >> "$HERE/INSTALL_LOG_rocm714.txt"

echo "Done. venv at: $VENV"
