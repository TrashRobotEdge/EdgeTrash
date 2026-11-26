"""Shared pre-processing. Part of the dataset spec (Benchmarking p.13): letterbox
with grey padding, bilinear resize, BGR -> RGB, scale to [0, 1]."""

from dataclasses import dataclass

import cv2
import numpy as np

PAD_VALUE = 114  # Ultralytics' padding colour, so our pipeline matches training


@dataclass
class Letterbox:
    scale: float
    pad_x: float
    pad_y: float


def letterbox(frame: np.ndarray, size: int) -> tuple[np.ndarray, Letterbox]:
    """Resize keeping the aspect ratio, pad to size x size. Returns RGB uint8 HWC."""
    h, w = frame.shape[:2]
    scale = min(size / h, size / w)
    nh, nw = round(h * scale), round(w * scale)
    resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
    pad_y, pad_x = (size - nh) / 2, (size - nw) / 2
    top, left = round(pad_y - 0.1), round(pad_x - 0.1)
    out = cv2.copyMakeBorder(
        resized,
        top,
        size - nh - top,
        left,
        size - nw - left,
        cv2.BORDER_CONSTANT,
        value=(PAD_VALUE,) * 3,
    )
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB), Letterbox(scale, left, top)


def to_tensor(rgb: np.ndarray, layout: str, dtype: str) -> np.ndarray:
    """RGB uint8 HWC -> batch of 1 in the layout/dtype the backend wants."""
    x = rgb[None]  # NHWC
    if layout == "nchw":
        x = x.transpose(0, 3, 1, 2)
    if dtype == "uint8":
        return np.ascontiguousarray(x)
    return np.ascontiguousarray(x, dtype=np.float32) / 255.0
