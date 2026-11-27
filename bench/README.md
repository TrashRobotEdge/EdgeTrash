# Benchmark harness

- **L1 (here):** ROS-free, the same replayed frames on every device. Timer = frame in RAM → boxes after NMS, per stage. System monitor (and Pi 5 PMIC power) logged in parallel on the same clock.
- **L2 (TrashLocalisation):** inside ROS on the robot. Timer = image `header.stamp` → `cmd_vel` publish.

Run from the repo root with the venv active and `PYTHONPATH` unset ([README](../README.md#quick-start-clone-to-figures-vm-ubuntu-2204)).

Rules and every number: [docs/protocol.md](../docs/protocol.md). Device settings: `configs/<device>.yaml` (frames, warm-up, duration, reps, variants).

| Command | What | Output |
|---|---|---|
| `python bench/smoke.py --config bench/configs/pi5-cpu.yaml` | 20 frames per variant, prints p50 | stdout |
| `python bench/run.py --config … --idle` | ≥ 60 s idle baseline | `results/raw/<device>/idle/<utc>/` |
| `python bench/run.py --config … --variant <v> --rep 0` | one variant, one repetition | `results/raw/<device>/<v>/<utc>-rep0/{frames,system}.csv, meta.json` |
| `python bench/sweep.py --config …` | idle + all variants × reps, interleaved | same |
| `python bench/accuracy.py --device vm --variant <v> --split data/splits/coco-eval.txt --ann data/coco/annotations/coco-eval-subset.json` | mAP, P/R/F1 on the device's output | `results/raw/<device>/accuracy/<v>.json` |
| `python bench/analyze.py [--meter meter.csv]` | tables + figures from raw logs only | `results/summary.csv`, `results/accuracy.csv`, `results/figures/` |

Power: `power.py`. Pi 5 PMIC is read automatically. The inline meter's export format is still open (#6): until then, give `analyze.py` a CSV `t_unix,watts`.
