# AI Surrogates by Example

Guided, hands-on examples of using AI alongside scientific simulations. Each application lives in
its own folder, with its setup, inputs, runnable stages, and results described together. Start with
the training and inference example: it demonstrates learning on a small dataset, then measures a
pretrained bumper model’s size, inference speed, and error against two simulations.

<p align="center">
  <img src="https://raw.githubusercontent.com/NVIDIA/physicsnemo/main/docs/img/crash/bumper_beam.gif" alt="PhysicsNeMo bumper-beam crash animation" width="80%" />
</p>

Animation from [NVIDIA PhysicsNeMo](https://github.com/NVIDIA/physicsnemo/tree/main/examples/structural_mechanics/crash).
This illustrates the bumper example; it is not an output from the workshop measurements.

| Application | Scope | Current stage |
|---|---|---|
| [01_training_and_inference](01_training_and_inference) | Toy training, real model size, inference timing, and error against two simulations | Available |
| [02_dataset_generation](02_dataset_generation) | Represent simulation datasets with generative models and sample from them | Outline |
| [03_parameter_tuning](03_parameter_tuning) | Adapt simulation parameters as conditions change | Outline |
| [04_inverse_problems](04_inverse_problems) | Infer simulation inputs from observations | Outline |
| [05_low_latency_prediction](05_low_latency_prediction) | Use fast predictions in interactive or time-critical workflows | Outline |

## The applications

The first application uses PhysicsNeMo to predict bumper deformation, plastic strain, and stress
from a mesh and a set of design and impact parameters. The remaining folders reserve space for
separate applications; their READMEs describe the intended experiment and artifacts. They do not
yet contain runnable implementations.

Each application follows the same progression: describe the engineering problem, prepare the
environment and data, run the example, inspect the artifacts, and explain where the reported
numbers come from. Configurations and measurements can be revised on the target system without
changing this organization.

## Where the numbers come from

The training and inference example includes a supplied checkpoint and historical verification notes. New results
belong under that application's `results/` directory, alongside the configuration and environment
that produced them. No performance figures for the other applications have been measured here.

The original `physics_nemo/workshop_crash_surrogate.py` and notebook now live in
[`01_training_and_inference/`](01_training_and_inference). Change into that folder for the setup and
notebook commands.
