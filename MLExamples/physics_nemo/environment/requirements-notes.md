# Environment setup

This project runs on AMD ROCm/HIP (torch built as `2.9.1+rocm7.2.1`, not a
CUDA build), and requires torch >= 2.9 for the `muon` optimizer used by
both the real checkpoint's config and the toy training config.

Two install paths, both installing the *identical* pinned ROCm wheel set —
pick whichever fits your setup:

- **`Apptainer.def`** (recommended) — builds a self-contained image with
  everything baked in, including an `INSTALL_LOG.txt` recording exact
  versions. Build with `apptainer build workshop.sif Apptainer.def`, run
  with `apptainer run --rocm --bind $(pwd):/workshop workshop.sif`.

- **`install_venv.sh`** (fallback) — same pinned wheels, installed into a
  local `venv/` via `uv`, for machines without Apptainer available. Assumes
  ROCm is already installed on the host (`/opt/rocm` or `module load
  rocm/...`). Writes the same kind of `INSTALL_LOG.txt` for parity with the
  container path.

Both write `INSTALL_LOG.txt` with exact installed versions and a full
`pip freeze`, so any later "it works on my machine" mismatch has a
ground-truth record to diff against.
