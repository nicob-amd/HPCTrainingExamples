import sys, os, time
T0 = time.time()
def _log(msg): print(f"[T+{time.time()-T0:6.1f}s] {msg}", flush=True)

import sys, os
sys.path.insert(0, os.path.abspath("src"))

import torch
import physicsnemo
import pyvista as pv
import hydra, omegaconf

print("torch:", torch.__version__)
print("cuda/HIP device available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device name:", torch.cuda.get_device_name(0))
print("physicsnemo:", physicsnemo.__version__)
print("pyvista:", pv.__version__)
print("hydra-core:", hydra.__version__, "| omegaconf:", omegaconf.__version__)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

# `bq_torch_patch` forces PhysicsNeMo's radius_search onto the pure-torch backend.
# On ROCm, warp-lang imports successfully but has no HIP backend and crashes at
# runtime — this patch must be imported before any model is built. Harmless if
# warp happens to work fine on your hardware (e.g. real CUDA).
_log('cell 3: importing bq_torch_patch...')
import bq_torch_patch
_log('cell 3: done')

_log('cell 5: reading mesh...')
mesh = pv.read("data/vtp_train/run1.vtp")
print("n_points (mesh nodes):", mesh.n_points)
print("n_cells:", mesh.n_cells)

disp_fields = sorted([k for k in mesh.point_data.keys() if k.startswith("displacement_t")])
print(f"\n{len(disp_fields)} displacement timesteps, e.g.:", disp_fields[:3], "...", disp_fields[-1])

strain_fields = sorted([k for k in mesh.cell_data.keys() if "plastic_strain" in k])
stress_fields = sorted([k for k in mesh.cell_data.keys() if "stress_vm" in k])
print(f"{len(strain_fields)} plastic-strain timesteps, {len(stress_fields)} stress timesteps (per-cell)")
_log('cell 5: done')

_log('cell 6: plotting mesh (matplotlib)...')
# Plot the mesh at t0 and at the final timestep, colored by displacement magnitude.
# Uses matplotlib (not pyvista's VTK renderer) so this works headless without
# a GPU/EGL/OSMesa OpenGL context -- pyvista's off_screen renderer needs one of
# those and isn't guaranteed to be present in a minimal container/notebook env.
import numpy as np
import matplotlib.pyplot as plt

coords0 = mesh.points
final_disp = np.asarray(mesh.point_data[disp_fields[-1]])
coords_final = coords0 + final_disp
disp_mag = np.linalg.norm(final_disp, axis=-1)

# Subsample points for a readable scatter (full mesh is ~14k nodes)
rng = np.random.default_rng(0)
idx = rng.choice(coords0.shape[0], size=min(4000, coords0.shape[0]), replace=False)

fig = plt.figure(figsize=(10, 5))

ax1 = fig.add_subplot(1, 2, 1, projection="3d")
ax1.scatter(coords0[idx, 0], coords0[idx, 1], coords0[idx, 2], s=1, c="gray")
ax1.set_title("t=0 (undeformed)")

ax2 = fig.add_subplot(1, 2, 2, projection="3d")
sc = ax2.scatter(coords_final[idx, 0], coords_final[idx, 1], coords_final[idx, 2],
                  s=1, c=disp_mag[idx], cmap="inferno")
ax2.set_title("t=0.5s (crushed), colored by |displacement|")
fig.colorbar(sc, ax=ax2, shrink=0.6, label="displacement magnitude (mm)")

plt.tight_layout()
plt.show()
_log('cell 6: done')

from omegaconf import OmegaConf
from hydra.utils import instantiate

toy_cfg = OmegaConf.load("checkpoint/toy_config.yaml")
print(OmegaConf.to_yaml(toy_cfg, resolve=True))

_log('cell 9: instantiating reader + toy dataset...')
reader = instantiate(toy_cfg.reader)

train_dataset = instantiate(
    toy_cfg.datapipe,
    name="toy_train",
    reader=reader,
    split="train",  # "train" computes and saves fresh normalization stats
    logger=None,
)
print(f"Train dataset: {train_dataset.num_samples} samples")

sample0 = train_dataset[0]
print(sample0)
print("\nglobal_features (design variables + load case) for sample 0:")
for k, v in sample0.global_features.items():
    print(f"  {k}: {v.item():.3f}")
_log('cell 9: done')

_log('cell 12: instantiating model + optimizer...')
from utils import combined_crash_loss, build_muon_optimizer

model = instantiate(toy_cfg.model).to(DEVICE)
model.train()
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

data_stats = dict(
    node={k: v.to(DEVICE) for k, v in train_dataset.node_stats.items()},
    edge={},
    feature={k: v.to(DEVICE) for k, v in getattr(train_dataset, "feature_stats", {}).items()},
    target={k: v.to(DEVICE) for k, v in getattr(train_dataset, "target_stats", {}).items()},
)
target_scale = data_stats["target"].get("target_scale", torch.zeros(0, device=DEVICE))
channel_std = torch.cat([torch.ones(3, device=DEVICE), target_scale]) if target_scale.numel() > 0 else None

import os
if os.environ.get("FORCE_ADAM_MI300A"):
    print("FORCE_ADAM_MI300A set -- using Adam (Muon's Newton-Schulz bf16 GEMMs hit a known hipBLASLt/MI300A-228CU tuning gap, see ROCm/rocm-systems#4084)")
    optimizer = torch.optim.Adam(model.parameters(), lr=toy_cfg.training.start_lr)
else:
    try:
        optimizer = build_muon_optimizer(model, toy_cfg)
        print("Using Muon optimizer (matches the real training run)")
    except ImportError:
        print("Muon unavailable (needs torch>=2.9) -- falling back to Adam for this toy loop")
        optimizer = torch.optim.Adam(model.parameters(), lr=toy_cfg.training.start_lr)
_log('cell 12: done')

_log('cell 13: starting toy training loop...')
import time

EPOCHS = toy_cfg.training.epochs
losses = []
t0 = time.time()

for epoch in range(EPOCHS):
    epoch_loss = 0.0
    for i in range(train_dataset.num_samples):
        sample = train_dataset[i].to(DEVICE)
        optimizer.zero_grad()
        pred = model(sample=sample, data_stats=data_stats)
        loss = combined_crash_loss(
            pred,
            sample.node_target,
            sample.node_features["coords"],
            data_stats["node"]["pos_std"],
            alpha=toy_cfg.training.displacement_loss_alpha,
            peak_loss_weight=toy_cfg.training.peak_loss_weight,
            peak_loss_top_k=toy_cfg.training.peak_loss_top_k,
            channel_std=channel_std,
            peak_plastic_strain_loss_weight=toy_cfg.training.peak_plastic_strain_loss_weight,
        )
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    avg_loss = epoch_loss / train_dataset.num_samples
    losses.append(avg_loss)
    print(f"epoch {epoch+1:3d}/{EPOCHS}  avg_loss={avg_loss:.4f}  elapsed={time.time()-t0:.1f}s")

print("\nDone. This is a mechanics demo, not a converged surrogate -- see part 2 for the real result.")
_log('cell 13: done')

_log('cell 14: plotting loss curve...')
import matplotlib.pyplot as plt

plt.figure(figsize=(6, 4))
plt.plot(range(1, len(losses) + 1), losses, marker="o")
plt.xlabel("epoch")
plt.ylabel("avg training loss")
plt.title(f"Toy training loop ({train_dataset.num_samples} samples)")
plt.grid(alpha=0.3)
plt.show()
_log('cell 14: done')

_log('cell 16: instantiating real_model + loading checkpoint...')
from physicsnemo.utils import load_checkpoint

real_cfg = OmegaConf.load("checkpoint/config.yaml")

real_model = instantiate(real_cfg.model).to(DEVICE).eval()
epoch_loaded = load_checkpoint("checkpoint/checkpoints", models=real_model, device=DEVICE)
print(f"Loaded real checkpoint at epoch {epoch_loaded}")
print(f"Model parameters: {sum(p.numel() for p in real_model.parameters()):,}")
_log('cell 16: done')

_log('cell 17: instantiating real test dataset...')
# split="test" LOADS the existing stats from checkpoint/stats/ (computed on the
# real 225-run dataset) instead of recomputing them from this small VTP set --
# this is the same pattern used by surrogate_opt/predictor_2d_thickness.py.
test_reader = instantiate(real_cfg.reader)
test_dataset = instantiate(
    real_cfg.datapipe,
    name="real_eval",
    reader=test_reader,
    split="test",
    logger=None,
)
print(f"Evaluation dataset: {test_dataset.num_samples} held-out samples (run19, run201)")

real_data_stats = dict(
    node={k: v.to(DEVICE) for k, v in test_dataset.node_stats.items()},
    edge={},
    feature={k: v.to(DEVICE) for k, v in getattr(test_dataset, "feature_stats", {}).items()},
)
_log('cell 17: done')

_log('cell 18: running inference on held-out runs...')
# Compare surrogate prediction vs. ground truth on each held-out run,
# for all three channels: peak intrusion (displacement), peak plastic
# strain, peak von Mises stress.
run_names = ["run19", "run201"]

print(f"{'run':<8}{'peak_disp_pred':>16}{'peak_disp_true':>16}{'disp_rel%':>11}"
      f"{'strain_pred':>13}{'strain_true':>13}{'strain_rel%':>13}"
      f"{'stress_pred':>13}{'stress_true':>13}{'stress_rel%':>13}")

with torch.no_grad():
    for idx, run_name in enumerate(run_names):
        sample = test_dataset[idx].to(DEVICE)
        pred = real_model(sample=sample, data_stats=real_data_stats)  # [N, T, 5]
        target = sample.node_target
        coords0 = sample.node_features["coords"]
        pos_std = test_dataset.node_stats["pos_std"].to(DEVICE)

        pred_disp_mm = (pred[:, :, :3] - coords0.unsqueeze(1)) * pos_std.view(1, 1, -1)
        true_disp_mm = (target[:, :, :3] - coords0.unsqueeze(1)) * pos_std.view(1, 1, -1)
        pred_peak_disp = torch.linalg.norm(pred_disp_mm, dim=-1).max().item()
        true_peak_disp = torch.linalg.norm(true_disp_mm, dim=-1).max().item()
        disp_rel = 100.0 * abs(pred_peak_disp - true_peak_disp) / true_peak_disp

        pred_peak_strain = pred[:, :, 3].max().item()
        true_peak_strain = target[:, :, 3].max().item()
        strain_rel = 100.0 * abs(pred_peak_strain - true_peak_strain) / true_peak_strain

        pred_peak_stress = pred[:, :, 4].max().item()
        true_peak_stress = target[:, :, 4].max().item()
        stress_rel = 100.0 * abs(pred_peak_stress - true_peak_stress) / true_peak_stress

        print(f"{run_name:<8}{pred_peak_disp:>16.1f}{true_peak_disp:>16.1f}{disp_rel:>11.2f}"
              f"{pred_peak_strain:>13.4f}{true_peak_strain:>13.4f}{strain_rel:>13.2f}"
              f"{pred_peak_stress:>13.4f}{true_peak_stress:>13.4f}{stress_rel:>13.2f}")
_log('cell 18: done')
