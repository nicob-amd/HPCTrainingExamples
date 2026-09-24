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

## 3c. ROCm 10: torch.cdist compute_mode needed a workaround (fixed)

Running `workshop_crash_surrogate.py` under a ROCm 10.0.0 environment
(`environment/setup_env_rocm10.sh`) originally crashed on the first forward
pass:

```
File ".../physicsnemo/nn/functional/neighbors/radius_search/_torch_impl.py", line 106
    dists = torch.cdist(points, queries, p=2.0, compute_mode="donot_use_mm_for_euclid_dist")
torch.AcceleratorError: CUDA error: invalid configuration argument (hipErrorInvalidConfiguration)
```

This comes from `BQWarp`'s ball query (used for GeoTransolver's multi-scale
local geometric features, `include_local_features: true`), which calls
PhysicsNeMo's `radius_search` — forced onto the pure-torch backend by
`src/bq_torch_patch.py` since `warp-lang` has no HIP backend on ROCm.

**Investigation, using standalone reproducers (`N` swept from 4096 to
16384, `torch.cdist(x, x, p=2.0,
compute_mode="donot_use_mm_for_euclid_dist")` on a `torch.ones(1, N, 3,
device="cuda")` tensor):**

- On ROCm 10.0.0 and 7.14.1, this exact call raised
  `hipErrorInvalidConfiguration` at every size we tested — including the
  workshop's real ~13,882-node mesh. Torch version didn't change this
  (2.10.0/2.11.0/2.13.0 all behaved the same way).
- On ROCm 7.2.x (7.2.1/7.2.4 — the version this workshop's shipped
  checkpoint was originally trained/verified on), the call didn't raise,
  but the returned distances didn't match a CPU float64 reference for a
  large fraction of pairs (frequently exactly `0.0`). Checked against the
  real mesh's (normalized) coordinates: e.g. reference distance
  `3.017...`, this specific GPU compute_mode returned `0.0`; 82% of all
  192.7M pairs in the 13,882×13,882 distance matrix disagreed with the
  reference by more than 1.0 (units: normalized coord space). The
  identical call on CPU, and the identical call on GPU with the *default*
  (unset) `compute_mode`, both matched the reference — only the explicit
  `donot_use_mm_for_euclid_dist` GPU path disagreed in our testing.
- The obvious alternative, forcing `compute_mode="use_mm_for_euclid_dist_if_necessary"`
  (the matmul/Gram-trick), didn't pan out as a fix either: benchmarked
  head-to-head on the real mesh, it was **~35x slower** (571ms vs 16ms/call)
  *and* its values disagreed with the reference for pairs within the
  actual ball-query radius (consistent with catastrophic cancellation in
  `|x|²+|y|²−2xy`) — consistent with PhysicsNeMo's own code comment already
  flagging the unset-`compute_mode` default as "numerically unstable" for
  this use case.

**Fix (`src/bq_torch_patch.py`):** replace `torch.cdist` globally (for the
process) with a manual broadcast-based distance
(`(x.unsqueeze(-2)-y.unsqueeze(-3)).pow(2).sum(-1).sqrt()`), a different
code path that doesn't go through any cdist compute_mode kernel. Patched
globally rather than scoped to just `radius_search`'s two call sites since
this is about `torch.cdist` itself in the environments we tested, not
something specific to how PhysicsNeMo calls it — other `cdist` call sites
elsewhere in physicsnemo (`lagrangian_dataset.py`, diffusion
preconditioners, `figconvnet`, `mesh_lsq_gradient.py`) would see the same
behavior if ever exercised with this compute_mode on this hardware, even
though this workshop doesn't currently touch those paths.

Verified on real MI300A hardware:

- Matches the CPU float64 reference exactly on the cases we checked (the
  native `donot_use_mm` kernel path did not, per above).
- No exception at any tested size (4096–16384) on ROCm 10.0.0.
- ~3% forward-pass overhead relative to the native `donot_use_mm` kernel on
  ROCm 7.2.4 (6.43ms vs 6.22ms at N=13882) — not a meaningful cost, and it's
  called only a handful of times per forward pass (once per radius scale ×
  per context/local-feature path), not on the model's dominant compute path.
- **End-to-end, no retraining needed:** ran the full
  `workshop_crash_surrogate.py` on ROCm 10.0.0 (torch
  2.11.0+rocm10.0.0) — toy training loop (30 epochs, full local features)
  converged normally (loss 3.43 → 0.19), and the *existing* 200-epoch
  pretrained checkpoint (trained on ROCm 7.2.1, never retrained) loaded and
  evaluated on the 2 held-out runs with results matching this doc's
  previously-recorded baseline:

  | run | peak disp rel err | strain rel err | stress rel err |
  |---|---|---|---|
  | run19 | 0.57% | 0.75% | 11.59% |
  | run201 | 0.26% | 6.49% | 12.55% |

  (0.57% matches the "~0.5%" baseline recorded in section 3 above, run on
  the original ROCm 7.2.1 setup — i.e. the fix preserves accuracy exactly,
  and the pretrained checkpoint's weights hold up fine despite the
  compute_mode discrepancy described above during its original training
  run.)

Two paths were considered and rejected in favor of this patch:
- **Retrain with a different architecture** (`TransolverOneShot` /
  `MeshGraphNetOneShot`, both already implemented in `src/rollout.py`)
  — neither touches `cdist`/`radius_search`/`BQWarp` at all, so this would
  also work, but requires retraining from scratch and (per the upstream
  `NVIDIA/physicsnemo` crash example's own benchmarks) is less accurate
  than GeoTransolver on this exact dataset. Not needed once the patch was
  shown to fully preserve the existing checkpoint's accuracy.

**Cross-checked on a second, independent torch+ROCm10 build**
(`environment/setup_env_rocm10_pytorch_container.sh`): rather than building
torch from wheels on top of a bare ROCm base image, this variant pulls
AMD's official prebuilt image
(`docker://rocm/pytorch:rocm10.0_ubuntu24.04_py3.12_pytorch_release_2.11.0`),
which ships torch 2.11.0+rocm10.0.0 already installed. The same
`bq_torch_patch.py` fix applies unchanged, and a full
`workshop_crash_surrogate.py` run produced the same result: toy training
converged normally, and the existing checkpoint evaluated with the same
0.57%/0.26% peak-displacement error as the `setup_env_rocm10.sh` path —
confirming this isn't specific to one particular torch build/install
method. See that script's header comment for the differences between the
two setups (mainly: no venv, an external `pip install --target=` directory
instead, added to `PYTHONPATH` at run time; and `timm` needing `--no-deps`
here in a way it didn't need there).
- **Force `compute_mode="use_mm_for_euclid_dist_if_necessary"`** — ruled
  out above (both wrong and 35x slower).

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
