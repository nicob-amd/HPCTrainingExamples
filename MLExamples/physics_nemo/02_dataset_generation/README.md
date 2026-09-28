# Surrogate by Example: Dataset Compression and Generation

Planned application. This folder reserves the structure; no runnable implementation or measurements
are included yet. See the [application index](../README.md) for the available bumper walkthrough.

## The application

Represent a simulation dataset with a CVAE or normalizing flow, then sample from the learned distribution.

## Configuration

Dataset split, model family, latent dimension, training budget, sampling seed, and physical validity checks.

## Results and artifacts

Resolved configuration, training curves, generated samples, distribution comparisons, storage footprint, and sampling timings.
When implemented, keep generated outputs in a `results/<run-id>/` directory and document the exact
launch command, software/hardware, and inputs beside each report.

## Where the numbers come from

No results have been measured for this application. Add figures only with their configuration,
measurement method, and supporting artifacts.
