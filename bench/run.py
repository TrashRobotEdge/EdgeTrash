"""L1 benchmark: one variant, one repetition (Benchmarking p.29-30).

    python bench/run.py --config bench/configs/vm-dev.yaml --variant yolov8n-coco-640-fp32-onnx --rep 0
    python bench/run.py --config bench/configs/pi5-cpu.yaml --idle     # idle power/system baseline

Timer (L1 "pipeline latency" without the camera, p.9): frame already decoded in RAM ->
detections in frame pixels. Frames are replayed from a fixed list, the same on every device.
Single stream, batch 1, no overlap: throughput = 1 / latency.

Writes results/raw/<device>/<variant>/<utc>-rep<k>/ : frames.csv (every frame, warm-up
flagged), system.csv (monitor), meta.json (versions, config, manifest entry, commit).
"""

import argparse
import json
import time
from pathlib import Path

import cv2
import pandas as pd
import yaml

from edgetrash import load_variant
from edgetrash.detector import read_manifest
from meta import RAW, REPO, run_meta
from monitor import Monitor


def load_config(path: Path) -> dict:
    return yaml.safe_load(Path(path).read_text())


def load_frames(cfg: dict) -> list:
    paths = [REPO / p.strip() for p in (REPO / cfg["frames"]).read_text().splitlines() if p.strip()]
    return [cv2.imread(str(p)) for p in paths[: cfg.get("n_frames", len(paths))]]


def out_dir(device: str, name: str, rep: int | None) -> Path:
    stamp = time.strftime("%Y%m%dT%H%M%S", time.gmtime())
    d = RAW / device / name / (f"{stamp}-rep{rep}" if rep is not None else stamp)
    d.mkdir(parents=True, exist_ok=True)
    return d


def run_idle(cfg: dict) -> Path:
    """>= 60 s with nothing running: P_idle for the energy formula (p.30)."""
    mon = Monitor(cfg.get("sample_period_s", 0.5))
    t0 = time.time()
    mon.start()
    time.sleep(cfg.get("idle_s", 60))
    rows = mon.stop()
    d = out_dir(cfg["device"], "idle", None)
    pd.DataFrame(rows).to_csv(d / "system.csv", index=False)
    meta = {"kind": "idle", "t_start": t0, "t_end": time.time(), "config": cfg, "meta": run_meta()}
    (d / "meta.json").write_text(json.dumps(meta, indent=2))
    return d


def run_variant(cfg: dict, variant: str, rep: int | None) -> Path:
    frames = load_frames(cfg)
    t_load = time.perf_counter()
    det = load_variant(variant, conf=cfg.get("conf", 0.25), iou=cfg.get("iou", 0.45))
    load_s = time.perf_counter() - t_load  # cold start: model file -> ready (p.7)

    mon = Monitor(cfg.get("sample_period_s", 0.5))
    mon.start()
    rows, i = [], 0
    warmup, duration, min_frames = cfg.get("warmup_frames", 50), cfg.get("duration_s", 60), cfg.get("min_frames", 300)
    t_steady = time.time() if warmup == 0 else None
    while True:
        frame_idx = i % len(frames)
        t_unix = time.time()
        dets, t = det(frames[frame_idx])
        rows.append(
            {
                "i": i,
                "frame": frame_idx,
                "t_unix": t_unix,
                "warmup": i < warmup,
                **t,
                "total_ms": t["pre_ms"] + t["infer_ms"] + t["post_ms"],
                "n_dets": len(dets),
            }
        )
        i += 1
        if i == warmup:
            t_steady = time.time()
        if t_steady and time.time() - t_steady >= duration and i - warmup >= min_frames:
            break
    t_end = time.time()
    sys_rows = mon.stop()

    d = out_dir(cfg["device"], variant, rep)
    pd.DataFrame(rows).to_csv(d / "frames.csv", index=False)
    pd.DataFrame(sys_rows).to_csv(d / "system.csv", index=False)
    meta = {
        "kind": "run",
        "variant": variant,
        "rep": rep,
        "load_s": load_s,
        "t_steady": t_steady,
        "t_end": t_end,
        "steady_frames": i - warmup,
        "backend": det.backend.info(),
        "manifest": read_manifest()[variant],
        "config": cfg,
        "meta": run_meta(),
    }
    (d / "meta.json").write_text(json.dumps(meta, indent=2, default=str))
    steady = pd.DataFrame(rows[warmup:])["total_ms"]
    print(
        f"{variant} rep {rep}: p50 {steady.quantile(0.5):.1f} ms  p95 {steady.quantile(0.95):.1f} ms  "
        f"({len(steady)} frames) -> {d.relative_to(REPO)}"
    )
    return d


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--variant")
    ap.add_argument("--rep", type=int)
    ap.add_argument("--idle", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    governor = run_meta()["governor"]
    if cfg.get("governor") and governor not in (None, cfg["governor"]):
        print(f"WARNING: CPU governor is {governor}, protocol says {cfg['governor']}: run setup/setup.sh")
    if args.idle:
        print(f"idle -> {run_idle(cfg).relative_to(REPO)}")
    elif args.variant:
        run_variant(cfg, args.variant, args.rep)
    else:
        ap.error("give --variant or --idle")


if __name__ == "__main__":
    main()
