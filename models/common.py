"""Helpers shared by the model scripts: manifest entries and calibration images."""

import importlib.metadata
import os
import platform
import shlex
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import yaml

from edgetrash.detector import MANIFEST, MODEL_DIR, sha256
from edgetrash.preprocess import letterbox, to_tensor

REPO = MANIFEST.parent.parent
# Ultralytics installs missing packages on the fly, unpinned. Off: a missing package must fail loudly.
os.environ.setdefault("YOLO_AUTOINSTALL", "false")


def versions(*packages: str) -> dict:
    out = {"python": platform.python_version()}
    for p in packages:
        try:
            out[p] = importlib.metadata.version(p)
        except importlib.metadata.PackageNotFoundError:
            pass
    return out


def register(name: str, file: Path | None, packages: tuple[str, ...] = (), **entry) -> None:
    """Add or replace one variant in models/manifest.yaml. Records the command that made it."""
    data = yaml.safe_load(MANIFEST.read_text()) if MANIFEST.exists() else {}
    data = data or {}
    variants = data.setdefault("variants", {})
    if file is not None:
        entry = {"file": file.name, "sha256": sha256(file), "size_mb": round(file.stat().st_size / 1e6, 2), **entry}
    rel = [str(Path(a).relative_to(REPO)) if Path(a).is_absolute() and Path(a).is_relative_to(REPO) else a for a in sys.argv]
    entry["made_by"] = "python " + shlex.join(rel)  # repo paths relative: same string on every machine
    entry["versions"] = versions(*packages)
    old = {k: v for k, v in variants.get(name, {}).items() if k != "made_at"}
    if old == entry:  # same file, same command, same tools: keep the entry (and its date) as it is
        print(f"[manifest] {name} unchanged")
        return
    made_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    variants[name] = {**{k: v for k, v in entry.items() if k != "versions"}, "made_at": made_at, "versions": entry["versions"]}
    MANIFEST.write_text(yaml.safe_dump(data, sort_keys=False))
    print(f"[manifest] {name}")


def model_path(name: str, ext: str) -> Path:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    return MODEL_DIR / f"{name}.{ext}"


def pretrained(arch: str) -> Path:
    """COCO weights from Ultralytics, cached in models/files/ (never git)."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    return MODEL_DIR / f"{arch}.pt"


def read_list(list_file: Path) -> list[Path]:
    """A split file: one image path per line, relative to the repo root."""
    return [REPO / line.strip() for line in Path(list_file).read_text().splitlines() if line.strip()]


def calib_tensors(list_file: Path, size: int, layout: str = "nchw", dtype: str = "float32", limit: int | None = None):
    """Calibration inputs with exactly our preprocessing (Benchmarking p.13: calibration set disjoint from test)."""
    for path in read_list(list_file)[:limit]:
        rgb, _ = letterbox(cv2.imread(str(path)), size)
        yield to_tensor(rgb, layout, dtype)


def calib_array(list_file: Path, size: int, limit: int) -> np.ndarray:
    """N x H x W x 3 RGB uint8, the format the Hailo and TFLite calibrators want."""
    return np.concatenate(list(calib_tensors(list_file, size, "nhwc", "uint8", limit)))
