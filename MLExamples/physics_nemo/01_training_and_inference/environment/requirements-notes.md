# Environment setup

This project runs on AMD ROCm/HIP (torch built as `2.9.1+rocm7.2.1`, not a
CUDA build), and requires torch >= 2.9 for the `muon` optimizer used by
both the real checkpoint's config and the toy training config.

Three install paths, all installing the *identical* pinned ROCm wheel set —
pick whichever fits your setup:

- **`Apptainer.def`** — builds a self-contained image with everything
  baked in, including an `INSTALL_LOG.txt` recording exact versions.
  Build with `apptainer build workshop.sif Apptainer.def`, run with
  `apptainer run --rocm --bind $(pwd):/workshop workshop.sif`.
  **Caveat:** `apptainer build` needs your user to have an entry in
  `/etc/subuid`/`/etc/subgid` on the build host (it runs `%post` as
  fake-root). Many shared/HPC login nodes don't grant this by default —
  if the build fails with a `setgroups`/"user namespace mappings"
  error, use `setup_env.sh` instead.

- **`setup_env.sh`** (recommended on shared/HPC nodes, or if the above
  fails) — same pinned wheels, same container base image, but built via
  `apptainer pull` (no `%post`, no fakeroot) + a plain `apptainer exec`
  install script. Needs no special account privileges at all. Usage:
  ```bash
  apptainer pull workshop_base.sif docker://rocm/dev-ubuntu-22.04:7.2.1-complete
  apptainer exec --bind $(pwd)/..:/workshop workshop_base.sif \
    bash /workshop/environment/setup_env.sh
  ```
  Installs into `environment/venv/` (inside the bind-mounted folder, so
  it persists on the host). Activate with an interactive
  `apptainer shell --rocm --bind $(pwd)/..:/workshop workshop_base.sif`
  then `source /workshop/environment/venv/bin/activate` — the venv's
  symlinks are bind-path-relative, so they only resolve inside a
  container session with the same bind mount, not directly on the bare
  host.

- **`install_venv.sh`** (fallback, no container at all) — same pinned
  wheels, installed into a local `venv/` via `uv` directly on the host.
  Assumes ROCm is already installed on the host (`/opt/rocm` or
  `module load rocm/...`). Writes the same kind of `INSTALL_LOG.txt` for
  parity with the container paths.

All three write `INSTALL_LOG.txt` with exact installed versions and a
full `pip freeze`, so any later "it works on my machine" mismatch has a
ground-truth record to diff against.
