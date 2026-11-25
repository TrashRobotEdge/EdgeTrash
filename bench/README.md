# Benchmark harness

- **L1 (here)**: ROS-free, the same replayed frame sequence on every device. Timer = frame read → postprocessed boxes, per stage. Power logged in parallel and aligned on timestamps.
- **L2 (TrashLocalisation)**: inside ROS on the robot config. Timer = image `header.stamp` → `cmd_vel` publish.

Rules come from [docs/protocol.md](../docs/protocol.md): idle ≥60 s, warm-up discarded, ≥60 s steady, every sample kept, ≥5 reps, runs interleaved across devices, **no sim time**.
