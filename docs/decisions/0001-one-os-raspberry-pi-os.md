# 0001: Raspberry Pi OS Trixie on both Pis

- **Status:** proposed (3 checks below, in class 01–02/12)
- **Date:** 26/11/2026 · **Issue:** #7

## Context
Benchmarking p.28: "same OS image" on every device, or the OS becomes an uncontrolled variable. Our Pis: Pi 4 (on the TurtleBot3, runs ROS) and Pi 5 + Hailo-8L. The OAK-D Pro plugs into either.

## Options
1. **Raspberry Pi OS Trixie (64-bit) on both.**
2. Ubuntu 22.04 on both: ROS Humble native.
3. Mixed: Pi 4 on the ROBOTIS Ubuntu image, Pi 5 on Raspberry Pi OS.

## Decision
Option 1. Fallback: option 3, with the OS written down as a known difference in every results file (it is already in each `meta.json`).

## Because
- Hailo is officially supported on Raspberry Pi OS: `hailo-all` is in Raspberry Pi's Trixie apt repo, with HailoRT 4.23.0 (checked in the repo index, 26/11).
- Ubuntu 22.04 doesn't support the Pi 5, so option 2 is out.
- The TB3's Pi 4 already runs ROS in Docker (AS1 p.38), so ROS Humble can stay in a container on Raspberry Pi OS.
- DepthAI 3.12.2 ships aarch64 wheels for Python 3.13 (Trixie's Python).

## Consequences
- Pis run Python 3.13, the VM runs 3.10: every pin in `pyproject.toml` has wheels for both.
- Checks before accepting: (a) the TB3 Docker setup works on Raspberry Pi OS · (b) `hailortcli fw-control identify` works on Trixie · (c) DepthAI sees the OAK on Trixie. `setup/setup.sh` runs (b) and (c).
