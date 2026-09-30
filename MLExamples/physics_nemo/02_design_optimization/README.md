# Bumper design optimization with automatic differentiation

How should we distribute material between the crash boxes and bumper beam to reduce
mass while limiting crash deformation? This example searches two independent thickness
scales using the pretrained model supplied in [application 01](../01_training_and_inference/).
It loads that checkpoint once, evaluates a coarse design grid, and refines a candidate
using a constrained optimizer supplied with derivatives from PyTorch autograd.

No training or additional checkpoint download is needed. The script uses the sibling
application's model code, mesh, configuration, and original normalization statistics.
The checkpoint is the 200-epoch GeoTransolverOneShot model from the 225-run project,
with displacement, plastic strain, and stress outputs; this objective uses displacement.

## Run

Prepare the environment and data using [application 01's setup instructions](../01_training_and_inference/README.md).
The environment also needs SciPy (`python3 -m pip install scipy` in that environment
if it is missing). Run with a prepared Python environment on an allocated
GPU node (CPU is supported but may be slow):

```bash
cd MLExamples/physics_nemo/02_design_optimization
RANK=0 WORLD_SIZE=1 LOCAL_RANK=0 MASTER_ADDR=127.0.0.1 MASTER_PORT=29500 \
  python3 -u optimize.py --limit-mm 350
```

For the ROCm 10 container prepared by application 01, start in `physics_nemo/` and bind
both applications so the existing checkpoint remains accessible:

```bash
apptainer exec --rocm --bind "$(pwd):/examples" --pwd /examples/02_design_optimization \
  --env RANK=0 --env WORLD_SIZE=1 --env LOCAL_RANK=0 \
  --env MASTER_ADDR=127.0.0.1 --env MASTER_PORT=29500 \
  01_training_and_inference/environment/workshop_base_rocm10.sif \
  /examples/01_training_and_inference/environment/venv_rocm10/bin/python3 -u \
  optimize.py --limit-mm 350
```

Launch one process with one GPU. The rank variables establish a single-process
checkpoint-loading context under Slurm. Use a distinct master port for concurrent runs.
`python3 optimize.py --help` works without importing the ML dependencies.

Defaults: impact velocity `-7`, wall position `0`, 5 × 5 grid, 40 maximum SLSQP
iterations, and thickness bounds `[0.7, 1.3]`. Change `--velocity` within `[-7,-3]`
and `--wall-position` within `[0,240]` to explore other loads. Each invocation fixes
one load case; it does not enforce constraints across all possible impacts.
The 350 mm default is an illustrative surrogate constraint, not an engineering standard.

## Mathematical problem

Let the dimensionless design vector be

$$
x=(x_c,x_b),\qquad 0.7\leq x_c,x_b\leq1.3,
$$

where `1` is nominal thickness, `c` denotes crash boxes, and `b` denotes the beam.
With fixed mesh and impact conditions, the trained network predicts normalized
positions $\hat p_{it}(x)$ for each node $i$ and future frame $t$.
Using the checkpoint's coordinate standard deviations $\sigma_p$, recover displacement:

$$
u_{it}(x)=(\hat p_{it}(x)-p_{i0})\odot\sigma_p,
\qquad D(x)=\max_{i,t}\|u_{it}(x)\|_2.
$$

The coordinate mean cancels in the subtraction. The result is in the mesh's length
units, millimetres. Here “peak intrusion” means maximum nodal displacement magnitude
over the predicted trajectory; it is not displacement at a particular occupant location
or along a specified intrusion axis.

We solve

$$
\min_x m(x)=\tfrac12 x_c+\tfrac12 x_b
\quad\text{subject to}\quad sD(x)\leq D_{\rm limit}.
$$

The mass proxy assumes equal nominal mass contributions from the two zones. It is
normalized to one at the nominal design. A value of `0.90` means a 10% reduction in
this proxy; actual mass requires material densities, element areas, and zone thicknesses.
Replace the equal weights with measured nominal zone mass fractions when available.

The optional multiplier `--safety-factor` is $s\geq1$, defaulting to one. It can
represent an externally calibrated margin, but this script does not estimate it or
claim a probability of safety. Historical calibration from another checkpoint is not
transferred here. Being within the parameter bounds does not guarantee model accuracy.

## Automatic differentiation

Weights are frozen with `model.requires_grad_(False)`, while the two design inputs
are created with `requires_grad=True`. The forward pass remains differentiable:
input thicknesses → global model features → predicted positions → physical
displacements → peak magnitude. `torch.autograd.grad(peak, design)` applies the chain
rule through this computation and returns

$$
\nabla_x D=\left[\frac{\partial D}{\partial x_c},
                        \frac{\partial D}{\partial x_b}\right].
$$

`model.eval()` selects inference behavior; it does not disable gradients. Grid
evaluations disable gradients to save memory, while optimizer evaluations enable
them. These are derivatives of the learned model, not derivatives of OpenRadioss.
The mesh remains fixed and thickness enters through continuous global inputs;
this example does not differentiate through remeshing or contact detection.

A derivative of `-100 mm/scale` means a local increase of `0.01` in that thickness
scale predicts approximately `1 mm` less peak displacement. Compare the two derivatives
relative to their mass costs to understand where additional material is most effective.
The model is not constrained to be monotone: positive derivatives should be investigated.

The maximum is piecewise differentiable. When the controlling node or frame changes,
the gradient can jump; at ties PyTorch uses its max-operation subgradient convention.
SLSQP assumes smooth local behavior, so convergence is not guaranteed. The script
compares autograd at the nominal design with central differences using steps `0.01`
and `0.001`. Disagreement can expose an implementation error, a switch in the controlling
node/frame, or floating-point cancellation. This diagnostic is printed and saved.

## Search and output

1. Reload the checkpoint and original statistics using the test data path.
2. Print nominal displacement and the automatic/finite-difference derivatives.
3. Evaluate a grid and write `grid.csv` with both thicknesses, mass proxy, raw peak,
   and margin-adjusted peak.
4. Seed SLSQP from the lightest predicted-feasible grid point. If none exists, start
   from the point with the smallest predicted peak; an empty feasible grid alone
   does not prove the continuous problem infeasible.
5. Minimize the linear mass proxy with analytic gradient `[0.5, 0.5]`. Supply the
   normalized inequality $g(x)=1-sD(x)/D_{\rm limit}\geq0$ and its autograd-derived
   Jacobian $-s\nabla D/D_{\rm limit}$. Cache the last value/gradient pair to avoid
   duplicate network evaluations when SciPy requests both.
6. Reevaluate the returned candidate and report constraint violation separately
   from SciPy's success flag. Predicted feasibility permits at most `0.01 mm`
   numerical violation and requires the thickness bounds to hold.

Each run creates a new timestamped `results/` directory; `--output-dir` selects a new
directory relative to the caller. Alongside `grid.csv`, `checkpoint_config.yaml` records
the resolved configuration, and `result.json` records the checkpoint SHA256, software
and device, arguments, gradient checks, best feasible grid point, candidate, optimizer
termination, and elapsed time. Console output includes:

```text
Loaded epoch ... on ...; checkpoint SHA256 ...
Nominal peak: ... mm; autograd [crash-box, beam]: ... mm/scale
Central differences h=0.01: ...
Central differences h=0.001: ...
Grid: .../... predicted feasible; refining from ...
```

The final JSON includes `solver_success`, `solver_message`, `violation_mm`,
`predicted_feasible`, and `fea_validated: false`. A completed run can report optimizer
failure or an infeasible candidate; `status: complete` means the experiment finished.
Compare the candidate's mass with the saved feasible grid point before accepting any
claimed improvement. Grid plus local refinement does not establish a global optimum.

## Closing the engineering loop

Run independent FEA at the proposed thickness pair under the same load and geometry,
measure the same displacement statistic, and accept or reject the candidate using that
result. If the surrogate is inaccurate, add the new simulation to a future training or
calibration set and repeat. This package proposes designs; it does not launch FEA or
certify crash safety. See [VALIDATION.md](VALIDATION.md) for checks of this packaged version.
