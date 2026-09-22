# Testing this folder on a new system

This folder is self-contained (no hardcoded paths back to the machine it was
built on — verified via `grep -rn "/home/nicob" .`, the only hit is a code
comment referencing project history, not a live path). Steps to verify it
actually works on a different machine, not just that it copies over cleanly.

## 1. Copy the folder over

```bash
# to a target host over SSH:
rsync -avz workshop_crash_surrogate/ user@target:/path/to/workshop_crash_surrogate/

# or tar it up for offline transfer (USB stick, etc.):
tar czf workshop_crash_surrogate.tar.gz workshop_crash_surrogate/
```

## 2. Pick an install path based on what the target has

**Apptainer available — two sub-paths depending on account permissions:**

```bash
cd workshop_crash_surrogate/environment
apptainer build workshop.sif Apptainer.def      # pulls rocm/dev-ubuntu-22.04 base + wheels, several minutes
cd ..
apptainer run --rocm --bind $(pwd):/workshop environment/workshop.sif
```

Watch the build log's `%post` verification block near the end — it should
print `torch: 2.9.1+rocm7.2.1...` and not trip the `assert '+rocm' in
torch.__version__` guard. If the target's ROCm driver is a different major
version than 7.2.1, the container's userspace libs may still work (ROCm has
some forward compatibility) but this isn't guaranteed.

**This has now been tried on a real shared HPC login node and the
`apptainer build` step failed there** with:

```
ERROR  : Could not write info to setgroups: Permission denied
ERROR  : Error while waiting event for user namespace mappings: no event received
FATAL:   ... while running %post section: exit status 1
```

Root cause: `apptainer build` runs `%post` as fake-root, which requires
an `/etc/subuid`/`/etc/subgid` entry for your account on that node.
Confirmed independently with a plain `unshare -Ur` (same "Operation not
permitted"). This is a per-account, per-node restriction, unrelated to
which directory you build into — check `grep $(whoami) /etc/subuid
/etc/subgid` before assuming this path will work.

If you hit this, use `setup_env.sh` instead (same repo, same folder) —
**this path HAS been verified end-to-end** (installs cleanly, correct
`torch: 2.9.1+rocm7.2.1...`, `physicsnemo: 2.1.1`, `pyvista`,
`hydra-core`/`omegaconf` all import cleanly):

```bash
cd workshop_crash_surrogate/environment
apptainer pull workshop_base.sif docker://rocm/dev-ubuntu-22.04:7.2.1-complete
apptainer exec --bind $(pwd)/..:/workshop workshop_base.sif \
  bash /workshop/environment/setup_env.sh
cd ..
apptainer exec --rocm --bind $(pwd):/workshop environment/workshop_base.sif \
  environment/venv/bin/jupyter lab --notebook-dir=/workshop --ip=0.0.0.0 --no-browser
```

This works because `apptainer pull` does no `%post` (no fakeroot needed),
and `apptainer exec` without `--fakeroot` just runs as your own uid —
neither needs the subuid/subgid entry that `build` does. `setup_env.sh`
gets Python 3.12 via `uv python install` instead of `apt-get`, which is
the only step in `Apptainer.def` that actually needed root.

**No Apptainer — venv fallback (this path HAS been verified end-to-end,
see section 3 below):**

```bash
cd workshop_crash_surrogate/environment
./install_venv.sh          # needs ROCm already on the host + `uv` on PATH
cat INSTALL_LOG.txt        # confirm a +rocm torch build and clean imports
```

## 3. Confirm the notebook actually runs, not just that it installs

Installing cleanly isn't the same as running correctly. Execute the
notebook headlessly and check for errors:

```bash
source environment/venv/bin/activate   # or the apptainer equivalent
jupyter nbconvert --to notebook --execute \
  --ExecutePreprocessor.timeout=600 \
  --output /tmp/executed.ipynb workshop_crash_surrogate.ipynb
```

Then check for silent failures (nbconvert can exit 0 even if it's showing a
stale cached output on a re-run — check outputs directly):

```bash
python3 -c "
import nbformat
nb = nbformat.read('/tmp/executed.ipynb', as_version=4)
for i, c in enumerate(nb.cells):
    for out in c.get('outputs', []):
        if out.get('output_type') == 'error':
            print(f'cell {i} FAILED:', out['ename'], out['evalue'])
print('done checking')
"
```

This is exactly how the folder was verified during development (CPU-only,
no GPU allocated in that session) — full run, every cell inspected for
`output_type == 'error'`, real numbers confirmed sane (e.g. peak-intrusion
relative error ~0.5% on a held-out run, consistent with the production
project's ~5-31% baseline band).

## 3b. Running end-to-end on a real GPU node (verified command)

On a GPU-allocated node (e.g. via `salloc`/`srun --gpus=1`), with the
`setup_env.sh` install from section 2:

```bash
cd workshop_crash_surrogate
apptainer exec --rocm --bind $(pwd):/workshop \
  --env FORCE_ADAM_MI300A=1 \
  --env RANK=0 --env WORLD_SIZE=1 --env LOCAL_RANK=0 \
  --env MASTER_ADDR=127.0.0.1 --env MASTER_PORT=29500 \
  environment/workshop_base.sif \
  environment/venv/bin/python3 -u workshop_crash_surrogate.py
```

Notes on the env vars:

- `RANK`/`WORLD_SIZE`/`LOCAL_RANK`/`MASTER_ADDR`/`MASTER_PORT` — needed
  when running under Slurm (`srun`): `physicsnemo`'s checkpoint loader
  auto-detects a distributed job from `SLURM_PROCID` and crashes without
  these explicitly set, since a plain single-task `srun` session doesn't
  set the rest of the Slurm env vars it expects.
- `FORCE_ADAM_MI300A=1` — on some hardware the default Muon optimizer
  used in the toy training loop (cell 12) is dramatically slower than
  Adam; this toggle (in the notebook and in `workshop_crash_surrogate.py`,
  its plain-script export) swaps it out. Confirmed on one MI300A node:
  full run 30 epochs -> ~2 minutes with this set; without it, a single
  optimizer step took several minutes.
- Prefer running `workshop_crash_surrogate.py` (exported via `jupyter
  nbconvert --to script workshop_crash_surrogate.ipynb`) over `nbconvert
  --execute` for interactive runs — `nbconvert --execute` buffers all
  cell output into the resulting notebook file and only prints to the
  terminal live if a cell errors, so you won't see per-epoch loss or
  progress until the whole thing finishes (or fails).

## 4. Things a same-machine test can't catch — check these on the real target

- **GPU visibility.** Confirm `torch.cuda.is_available()` in notebook
  section 1 reports `True` on the target (it was only ever exercised on CPU
  fallback during development, since no GPU was allocated in that session).
  If it's `False` on a machine that should have a GPU, check `--rocm` was
  passed to `apptainer run`, or that `ROCM_PATH`/`LD_LIBRARY_PATH` are set
  correctly for the venv path.
- **ROCm version drift.** Both `install_venv.sh` and `Apptainer.def` are
  hard-pinned to ROCm 7.2.1 wheels. A target on a different ROCm release
  will hit the script's explicit "no pinned wheel known for ROCm X.Y.Z"
  error rather than silently installing something broken — that's
  intentional, but means a new wheel URL (from
  `repo.radeon.com/rocm/manylinux/rocm-rel-X.Y.Z`) needs adding to the
  `case` statement in both files for that version.
- **Toy-training timing.** ~130s/epoch on 8 samples was measured on CPU
  only. Time it on the real target GPU — this determines whether the
  30-epoch default in `checkpoint/toy_config.yaml` fits a workshop's live
  pacing or needs trimming (`training.epochs` in that file).
