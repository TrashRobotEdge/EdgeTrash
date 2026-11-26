# Benchmark protocol

> Written **before** measuring (Benchmarking p.21). Built from the team's draft (26/11) + feedback grounded in the slides (end of this file).
> Rule: every number is **derived** (formula shown), **measured** (command + results file), or marked **to decide** / **to measure**.
> Slide refs: "p.N" = `edge-ai-labs/Benchmarking.pdf`, "EA1 p.N" = `EA1-edge_ai_intro.pdf`, "AS1 p.N" = `AS1-autonomous_systems_introduction.pdf`.

## Open decisions (the team decides, then we fill the numbers below)
| # | Decision | Proposal | Because |
|---|---|---|---|
| D1 | **d** = max distance the robot moves between two decisions | 2 cm | Sets the decision rate (R1). Pick it from the approach precision we need |
| D2 | **Stop margin** = distance at which the robot must have decided | 5 cm | Sets the latency budget (R2) |
| D3 | **Runtime** on one battery | 1 h (draft) | Sets the perception power budget (R4) |
| D4 | Quality gate | mAP@0.5:0.95 drop ≤ **1 point** vs FP32 | p.31 "e.g. > 1 % drop". Points, not relative %: say it explicitly |
| D5 | Precision / recall targets | keep **95 %** as a *target*, not a pass/fail | Nano models on a few hundred images may not reach it. The abacus shows which configs do |
| D6 | Architectures (issue #11) | YOLOv8n, YOLO11n, YOLOv5nu + SSDLite-MobileNetV3 | All fine-tuned on our cans. Same Ultralytics training for 3, a different family for the 4th (p.24) |
| D7 | Dataset size | 300–500 labelled images + negatives | 4 models chasing 95 % need more than 150 images |
| D8 | Hailo PCIe link | leave the default, record it | Gen 2 vs Gen 3 changes results (p.19, p.28): if we change it, it becomes a fixed factor written here |
| D9 | Colour vs grayscale | second priority, only if time | Costs one fine-tune per model |
| D10 | One OS on both Pis | Raspberry Pi OS Trixie | [ADR 0001](decisions/0001-one-os-raspberry-pi-os.md) (proposed, 3 checks pending) |

## 1. Goal: requirements → abacus axes (p.22–23)
We produce abacuses that link a config (model × optimisation × device) to latency, energy, accuracy and cost. We read them backwards: start from the robot's requirements and list the configs that fit (p.22).

| Req | Requirement (from the robot) | Value | Metric | Dimension (p.23) |
|---|---|---|---|---|
| R1 | Decide often enough: at most **d** moved between decisions | f ≥ v / d. TB3 Burger max speed v = 0.22 m/s ([ROBOTIS spec](https://emanual.robotis.com/docs/en/platform/turtlebot3/features/)). With d = 2 cm (D1): **f ≥ 11 Hz** | sustained FPS | real-time |
| R2 | Decide before the stop margin | latency ≤ margin / v. With 5 cm (D2): **≤ 227 ms**. The draft's 500 ms = 11 cm travelled at full speed | pipeline latency **p95** and p99 | real-time |
| R3 | Leave the Pi 4 for ROS + drivers | **to decide** after measuring ROS alone | CPU % on the robot config | integration |
| R4 | Run **1 h** (D3) on one battery | P_perception ≤ E_battery / 1 h − P_robot. Retailer listing (not ROBOTIS): 11.1 V × 1.8 Ah ≈ 20 Wh, **check our battery label**; P_robot **to measure** | idle W, active W, **mJ / inference** | energy |
| R5 | Few missed cans, few false cans | targets 95 % P / 95 % R (D5), gate D4 | P, R, F1 at the operating threshold, mAP@0.5, mAP@0.5:0.95, **measured on the device output** (p.8, p.31) | quality |
| R6 | See the can before the camera can't | min distance > 10 cm (draft). OAK-D Pro depth min range: **to check** in its spec sheet (#5) | recall vs distance | functional |
| R7 | See the can early | max distance **to measure** | recall vs distance, per input size | functional |
| R8 | Don't crash, don't drift | 30–60 min continuous run | crashes, p95 drift, throttling flags | stability (p.8) |
| R9 | Stay cool enough | 0–50 °C is a **datasheet limit**, we can't test it | temperature + throttling under sustained load at room temperature (p.7, p.28) | thermal |
| R10 | Cost per config | Pi4 + OAK · Pi5 + Hailo · Pi5 + OAK: **to fill** from the spec sheets (#2, #3, #5) | € per config, not total | cost |

## 2. Workload (p.12–13, p.24)
- **Models:** the D6 architectures fine-tuned on our dataset. COCO-pretrained weights are only the starting point. Vendor pre-compiled files (Hailo zoo `.hef`) are a **reference**, never a result about our models.
- **Weights don't change speed, only accuracy.** The pipeline is built and debugged now with COCO weights (the "dry run", `results/raw/vm/`). Our weights drop in later with no code change.
- **Accuracy set:** our test split (file list + checksums in `data/splits/`), with **negatives** (no can, red objects that aren't cans). Slices for robustness: low light, partly hidden (R5). Dry run: 500 fixed COCO val2017 images (`data/splits/coco-eval.txt`, seed 0).
- **Calibration set:** disjoint from test (p.13). Dry run: 200 other COCO images (`coco-calib.txt`).
- **Replayed frames (L1):** the same fixed list on every device (p.13). Dry run: 100 COCO images (`coco-bench.txt`). Then: frames captured with our OAK.
- **Batch 1, single stream** (the robot handles one frame at a time).
- **Pre-processing is part of the workload:** letterbox to a square, grey 114 padding, bilinear, RGB, [0, 1] (`edgetrash/preprocess.py`).

## 3. Variables (p.24)
| Fixed | Varied | Measured |
|---|---|---|
| OS image (D10), runtime versions (`pyproject.toml`), batch 1, frames, calibration set, power supply, cooling, governor = performance, room, cable/port | **architecture** (D6) · **precision** FP32 / FP16 / INT8 · **pruning** 0 % vs ~50 % MACs · **input size** 640 (main), 320 · **toolchain** · **device** | latency (per stage), FPS, power, energy / inference, peak RSS, load time, CPU %, temperature, throttling, accuracy, model size, % ops on the accelerator |

**Order (p.24):** one factor at a time on a reference config, then cross the factors that matter.

**Valid cells** (✓ = built in `models/`, ✗ = not possible, ? = to try):
| Toolchain → device | FP32 | FP16 | INT8 | Pruned | 320 |
|---|---|---|---|---|---|
| ONNX Runtime → Pi4 / Pi5 CPU | ✓ | ? (which ops have ARM fp16 kernels: to check) | ✓ static QDQ | ✓ | ✓ |
| LiteRT (TFLite) → Pi4 / Pi5 CPU | ✓ | SSDLite ✓ · YOLO ? (no FP16 in Ultralytics' exporter) | ✓ static int8 | ? | ? |
| TVM (untuned) → Pi4 / Pi5 CPU | ✓ | ✗ | ✗ | ? | ? |
| Hailo DFC → Pi5 + Hailo-8L | ✗ | ✗ | ✓ only | ✗ | ? |
| OpenVINO blob → OAK-D Pro VPU | ✗ | ✓ only (SSDLite: after swapping HardSigmoid, problems.md) | ✗ | ✗ | ? |

Hailo and OAK accept one precision each, so "precision" and "device" are partly confounded: say so in the report (p.35 "comparing different precisions without saying so").

## 4. Metrics (p.7–9, p.25)
- **L1 pipeline latency** (`bench/run.py`): timer start = frame decoded in RAM, timer stop = detections in frame pixels after NMS. Three stages timed with `time.perf_counter`: pre-processing, inference (includes host↔accelerator transfer for Hailo/OAK, p.6), post-processing. No camera, so this is not end-to-end (p.9).
- **L2 end-to-end latency** (TrashLocalisation): image `header.stamp` → `cmd_vel` published, on the robot config, no sim time.
- **Percentiles:** p50, p95, p99 per run (p.34). Never the mean alone (p.35).
- **Throughput:** single stream, no overlap, so FPS = frames / steady time ≈ 1 / latency (p.9).
- **Energy / inference** = (P_active − P_idle) × T / N (p.25), with P from the meter (reference) and/or the Pi 5 PMIC.
- **Accuracy:** mAP@0.5:0.95 and mAP@0.5 with COCO tools (conf 0.001, max 300 detections). Precision / recall / F1 at the operating threshold (conf 0.25, IoU 0.5). **Drop** = reference FP32 ONNX − variant, in points (`bench/accuracy.py`, `bench/analyze.py`).
- **Others:** peak RSS, load time (file → ready), CPU % (mean, and busiest core), SoC temperature, clock, `vcgencmd get_throttled`.

## 5. Setup (p.26–28)
- **Power, point B (reference):** inline USB-C meter between supply and board (#6: model, sampling rate and log export **to check**). **Point D:** `vcgencmd pmic_read_adc` on the Pi 5 (board rails only: no USB devices, not the 5 V input). Check D against B (p.27). Pi 4 and OAK have no power sensor: meter only.
- **What is powered** is written in each run's notes: host only / host + Hailo / host + OAK.
- **Same for every run:** supply, cable, USB 3 port (blue) for the OAK, cooling (case/fan **to note** in `docs/hardware/`), room temperature written down.
- **Clocks:** CPU governor = performance (`setup/setup.sh` sets it, `bench/run.py` warns if it's not set).
- **Links:** USB speed (`lsusb -t`: OAK must be 5000M) and Hailo PCIe link (`lspci -vv`) recorded by `setup/setup.sh` (D8).
- **Isolation:** same OS image on both Pis (D10), no desktop session during runs, nothing else running.
- **No benchmark right after power-up** (p.28): idle baseline first.

## 6. Measure (p.29–30)
Per device: `python bench/sweep.py --config bench/configs/<device>.yaml` (or the UI).
1. **Idle** ≥ 60 s, nothing running → P_idle.
2. Each variant in a **fresh process**: load (timed), **50 warm-up frames** (kept in the CSV, flagged, excluded from stats), then **≥ 60 s and ≥ 300 frames** steady. Every frame kept.
3. **5 repetitions**, variants **interleaved** in a new random order each repetition (seed = rep), 10 s cool-down between runs (p.28, p.31).
4. Accuracy of each variant on its device: `python bench/accuracy.py --device <device> --variant <v> ...` (p.31 "verify outputs on target").
5. Stability (R8): one 30–60 min run of the chosen config.

## 7. Analyze (p.31, p.34)
`python bench/analyze.py` regenerates everything from `results/raw/`:
- Per config: **median over the 5 repetitions** of each run's p50 / p95 / p99, FPS and energy, with the **IQR** (p.31).
- **Quality gate** D4 applied in `results/accuracy.csv`; configs below it are flagged, not hidden.
- Figures: latency **CDF**, **box plots**, stage breakdown, **Pareto** front accuracy × latency (and × energy once measured).
- Abacus read backwards (p.22): configs meeting R1, R2, R4, R5.

## 8. Report (p.35, p.37)
Repository = the reproducibility package: code, configs, raw logs, one command per step (README). Every `meta.json` records commit, command, OS, kernel, board, governor, runtime versions and the model's manifest entry (checksum + the command that built it). Threats to validity go in the report: replayed frames vs camera, VM dry run ≠ device results, confounded precision/device on Hailo/OAK, COCO vs our data.

---

## Feedback on the team draft (26/11) and what changed
Good instincts: it already covers the p.23 dimensions (real-time, energy, thermal, cost, quality, robustness). Three structural fixes:
1. The draft mixed **requirements** (step 1), **variables** (step 3, e.g. colour vs grayscale) and **metrics** (step 4). They are now split into the 8 steps (p.21).
2. "P95" was used everywhere. P95 is a **latency percentile** (p.34). Precision and recall are not percentiles: they're measured once per config on the test set.
3. Numbers are **derived from the robot** instead of picked (p.23: "requirements define the axes").

| Draft row | Feedback | Now |
|---|---|---|
| Sampling 20–40 fps | Not derived, may fail the Pi 4 for no reason | R1: f ≥ v / d (D1) |
| Latency < 500 ms, P95 | Good: total latency with p95. But 500 ms = 11 cm at 0.22 m/s | R2 from the stop margin (D2). Timestamps defined (step 4). p50/p95/p99 |
| Consumption ≥ 1 h | Good idea, needs converting to watts | R4: battery Wh ÷ 1 h − robot draw → perception budget in W |
| Precision ≥ 95 %, Recall ≥ 95 % | "P95" doesn't apply. 95 % may be unrealistic | R5 + D4/D5: P/R/F1 at a fixed threshold, mAP, on the device output, gate vs FP32 |
| "Few true labels → precision high" | Right instinct: class imbalance | Negatives in the test set, F1 |
| Cost < 1000 € total | Total cost doesn't compare configs | R10: cost per config |
| Ambient 0–50 °C | We can't test that range | R9: datasheet limit + measured temperature/throttling |
| Min detection distance > 10 cm | Good, linked to FOV + depth min range | R6, OAK spec sheet check |
| Max detection distance | Good | R7: recall vs distance, depends on input size |
| Time to crash | Good: p.8 "stability" | R8 |
| Colour vs grayscale | A variable, not a requirement | D9, optional varied factor |
| Robustness (light, occlusion, football, rain) | Good idea, too wide | Test-set slices (low light, partly hidden). Rain: out of scope (indoor) |
| Model factors table | Good: matches p.24 | Step 3 |

**Missing in the draft vs the slides, now in:** throughput vs latency (p.9), dropped frames (p.8; single stream drops none, revisit for L2), peak memory and load time (p.7), model size (p.5), % ops on the accelerator (p.7, fallback pitfall p.19), idle power (p.30), fixed/varied/measured split (p.24).
