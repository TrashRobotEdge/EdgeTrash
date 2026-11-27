"""Detector = shared preprocess + one backend + shared postprocess, each stage timed."""

import hashlib
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml

from edgetrash import postprocess as post
from edgetrash.backends import get_backend
from edgetrash.preprocess import letterbox, to_tensor

REPO = Path(__file__).resolve().parent.parent
MANIFEST = REPO / "models" / "manifest.yaml"
MODEL_DIR = REPO / "models" / "files"


@dataclass
class Detection:
    cls: int
    score: float
    box: tuple[float, float, float, float]  # x0 y0 x1 y1, frame pixels


class Detector:
    def __init__(self, backend, head: str, input_size: int, conf=0.25, iou=0.45, max_det=300, **head_opts):
        self.backend, self.head, self.size = backend, head, input_size
        self.conf, self.iou, self.max_det = conf, iou, max_det
        self.head_opts = head_opts

    def preprocess(self, frame):
        if self.head == "hsv":
            return cv2.cvtColor(frame, cv2.COLOR_BGR2HSV), None
        rgb, lb = letterbox(frame, self.size)
        return to_tensor(rgb, self.backend.layout, self.backend.dtype), lb

    def infer(self, x):
        return self.backend.infer(x)

    def postprocess(self, raw, lb, frame_shape) -> list[Detection]:
        if self.head == "hsv":
            boxes, scores, classes = raw
        elif self.head == "yolo":
            boxes, scores, classes = post.decode_yolo(raw[0], self.size, self.conf, self.head_opts.get("normalized", False))
        elif self.head == "ssd":
            boxes, scores, classes = post.decode_ssd(raw[0], raw[1], self.head_opts["anchors"], self.conf)
        elif self.head == "hailo_nms":
            boxes, scores, classes = post.decode_hailo_nms(raw[0], self.size, self.conf)
        else:
            raise ValueError(f"unknown head {self.head!r}")
        keep = post.nms(boxes, scores, classes, self.iou, self.max_det)
        boxes = boxes[keep] if lb is None else post.unletterbox(boxes[keep], lb, frame_shape)
        return [Detection(int(c), float(s), tuple(map(float, b))) for b, s, c in zip(boxes, scores[keep], classes[keep], strict=True)]

    def __call__(self, frame):
        """Returns (detections, {pre_ms, infer_ms, post_ms})."""
        t0 = time.perf_counter()
        x, lb = self.preprocess(frame)
        t1 = time.perf_counter()
        raw = self.infer(x)
        t2 = time.perf_counter()
        dets = self.postprocess(raw, lb, frame.shape)
        t3 = time.perf_counter()
        return dets, {"pre_ms": (t1 - t0) * 1e3, "infer_ms": (t2 - t1) * 1e3, "post_ms": (t3 - t2) * 1e3}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_manifest(path: Path = MANIFEST) -> dict:
    return (yaml.safe_load(path.read_text()) or {}).get("variants") or {}


def load_variant(name: str, conf=0.25, iou=0.45, max_det=300, manifest=MANIFEST, model_dir=MODEL_DIR) -> Detector:
    """Build the Detector for one variant of models/manifest.yaml, after checking the file's checksum."""
    v = read_manifest(manifest)[name]
    path = None
    if v.get("file"):
        path = Path(model_dir) / v["file"]
        if not path.exists():
            raise FileNotFoundError(f"{path} missing: copy it into {model_dir} (see models/README.md)")
        if v.get("sha256") and sha256(path) != v["sha256"]:
            raise ValueError(f"{path} checksum differs from manifest: wrong or corrupted file")
    head_opts = {"normalized": v.get("normalized_boxes", False)}
    if v.get("anchors"):
        head_opts["anchors"] = np.load(Path(model_dir) / v["anchors"])
    backend = get_backend(v["backend"])(path, **v.get("backend_opts", {}))
    return Detector(backend, v["head"], v.get("input_size", 0), conf, iou, max_det, **head_opts)
