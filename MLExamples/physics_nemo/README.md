# AI Surrogates by Example

Guided, hands-on examples of using AI alongside scientific simulations, focused on design
optimization and inverse problems. The available bumper workflow starts with training and
inference, then reuses the supplied checkpoint to optimize two independent thickness parameters.
Inverse problems remain a planned application.

<p align="center">
  <img src="https://raw.githubusercontent.com/NVIDIA/physicsnemo/main/docs/img/crash/bumper_beam.gif" alt="PhysicsNeMo bumper-beam crash animation" width="80%" />
</p>

Animation from [NVIDIA PhysicsNeMo](https://github.com/NVIDIA/physicsnemo/tree/main/examples/structural_mechanics/crash).
This illustrates the bumper example; it is not an output from the workshop measurements.

| Example | Scope | Current stage |
|---|---|---|
| [01_training_and_inference](01_training_and_inference) | Toy training, real model size, inference timing, and error against two simulations | Available |
| [02_design_optimization](02_design_optimization) | Optimize independent crash-box and beam thicknesses using the supplied checkpoint and automatic derivatives | Available |
| [03_inverse_problems](03_inverse_problems) | Infer simulation inputs from observations | Outline |

## Workshop progression

For the prebuilt ROCm 10 PyTorch container with `/opt/venv/bin/python3`, run
the shared installer from this directory inside the container:

```bash
bash setup_env_rocm10_pytorch_venv.sh
```

It creates `$HOME/venvs/physicsnemo-rocm10`, reuses the container's PyTorch, and
installs the workshop dependencies including SciPy. Pass a different new directory
as the first argument to change the destination. Use the environment's Python
directly, without setting `PYTHONPATH`:

```bash
cd 02_design_optimization
"$HOME/venvs/physicsnemo-rocm10/bin/python3" -u optimize.py
```

The environment persists in your home directory and must be used inside the same
container image. See the [optimization setup](02_design_optimization/README.md#run)
for details and the [training setup](01_training_and_inference/README.md) for other images.

Start with **training and inference**: use PhysicsNeMo to predict bumper deformation, plastic
strain, and stress from a mesh and design and impact parameters. Train a small toy model, then
load the separate supplied checkpoint and measure its size, inference speed, and error against
two simulations. Data preparation, model accuracy, and inference speed support the applications.

Continue with **design optimization**: reload the same checkpoint to search independent crash-box
and beam thicknesses. Minimize a mass proxy subject to a predicted peak-displacement limit using
a grid scan followed by SLSQP with automatic derivatives. The example explains the mathematics,
checks gradients against finite differences, and reports predicted feasibility. Proposed designs
still require independent FEA validation.

**Inverse problems** will address recovering unknown simulation inputs from observations.
The folder currently contains an outline; the physical example and runnable implementation
are still to be developed.

Each application follows the same progression: describe the engineering problem, prepare the
environment and data, run the example, inspect the artifacts, and explain where the reported
numbers come from. Configurations and measurements can be revised on the target system without
changing this organization.

## Where the numbers come from

The training and inference example includes a supplied checkpoint and
[verification notes](01_training_and_inference/TESTING.md). The optimization example includes
[GPU validation notes](02_design_optimization/VALIDATION.md) and
[saved results](02_design_optimization/sample_results/result.json) from that checkpoint,
including the configuration, gradient checks, and optimizer outcome. These results describe
surrogate predictions and a mass proxy, not FEA-confirmed safety or measured physical mass savings.

New results belong under each example's `results/` directory, alongside the configuration and
environment that produced them. No results have been measured for the inverse-problem outline.

The original `physics_nemo/workshop_crash_surrogate.py` and notebook now live in
[`01_training_and_inference/`](01_training_and_inference). Change into that folder for the setup and
notebook commands.
