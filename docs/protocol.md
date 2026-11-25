# Benchmark protocol

> Written **before** measuring (Benchmarking slides p.21). Steps 1–5: draft in S0 (issue "Benchmark protocol draft"), all 8 frozen by the end of S1.
> Full reasoning: plan §14.

## 1. Goal: requirements from the robot
| Req | What | Metric | Threshold (from measurements) |
|---|---|---|---|
| R1 Real-time | decision fast enough at TB3 speed | pipeline latency p95/p99, sustained FPS | TODO |
| R2 Headroom | leaves the Pi4 CPU for ROS + drivers | CPU % on robot config | TODO |
| R3 Range | detects a can far enough | recall vs distance, vs input size | TODO |
| R4 Energy | perception budget vs TB3 battery | idle W, active W, mJ/decision | TODO |
| R5 Accuracy | few missed/false cans | mAP@0.5, mAP@0.5:0.95, drop vs FP32 | TODO |
| R6 Integration | porting effort, operator support | hours, % ops on accelerator, crashes | TODO |

## 2. Workload
TODO: models, test set (file list + checksums), replayed frame sequence, batch 1.

## 3. Variables (valid cells only)
TODO: from `docs/hardware/` spec sheets.

## 4. Metrics
TODO: exact timestamp pairs for L1 and L2; energy per inference = ∫(P − P_idle)dt / N.

## 5. Setup
TODO: meter point B + internal sensors, what's powered, governor, cooling, link speed, ambient, clock sync, no sim time.

## 6. Measure · 7. Analyze · 8. Report
TODO (S1).
