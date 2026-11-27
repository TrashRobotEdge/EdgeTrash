# 0002: Hailo calibration and optimization level

- **Status:** proposed (team decides; recompile if someone has Ubuntu + an NVIDIA GPU)
- **Date:** 27/11/2026 · **Issue:** #43

## Context
- Our `yolov8n-coco-640-int8-hailo.hef` was compiled on the VM (no GPU) at **optimization level 0**: the DFC warns "less data than the recommended amount (1024), and there's no available GPU" and drops to 0 (no accuracy recovery, no compression).
- It calibrated on **64 images**: DFC 3.33's default `calibset_size` (`hailo_model_optimization/.../mo_config_model.py`). `compile_hailo.py` links 200; the zoo recipe doesn't raise the default. The manifest now records what the DFC actually used.
- Hailo's zoo `.hef` was built with Hailo's own settings. Comparing ours with it = two compile settings (p.35).
- The only GPU on hand (RTX 5070 Ti) is on a Windows host. The VM can't see it, and Hailo supports the DFC on Ubuntu only.

## Options
1. Keep level 0 + 64 images, and say so on every Hailo result.
2. Recompile with ≥ 1024 calibration images and `model_optimization_config(calibration, calibset_size=1024)` on an Ubuntu machine with an NVIDIA GPU (a teammate's, the lab's): the DFC's default level 2.
3. Same as 2 on the CPU (force the level in the alls script): unverified that the DFC allows it, and level 2 without a GPU would take hours.

## Decision
Option 1 for now. Option 2 if the on-Pi accuracy of our `.hef` fails the gate (D4) against its FP32 reference at 0.2 (protocol §3), or as soon as a GPU machine turns up.

## Because
- Level 0 needs no extra hardware and is honest when labelled: the manifest carries `optimization_level: '0'` and `calib_images: 64`.
- Whether level 0 is good enough is a measurement (accuracy on the Pi), not a guess.
- Option 3 has no evidence behind it.

## Consequences
- Every Hailo result and figure says "opt level 0, 64 calibration images". The zoo `.hef` is a reference, never a result about our models (protocol §2).
- Before trying option 2, check that DFC 3.33's bundled TensorFlow/CUDA supports the GPU (a 5070 Ti is Blackwell, which needs a recent CUDA): unverified.
- Option 2 needs a ≥ 1024-image calibration split, disjoint from eval (`data/coco_splits.py` only makes 200).
