"""Shared post-processing: decode the raw head, NMS, map boxes back to the frame.

One decoder per output format ("head" in models/manifest.yaml):
- yolo: Ultralytics head (YOLOv5u/v8/11), one tensor (1, 4 + classes, anchors), boxes cx cy w h
- ssd: our SSDLite export, (1, A, 4) box deltas + (1, A, 1 + classes) logits, anchors in a .npy
- hailo_nms: Hailo's NMS already ran (in HailoRT, on the host CPU), per class (k, 5) y0 x0 y1 x1 score in [0, 1]
"""

import cv2
import numpy as np

from edgetrash.preprocess import Letterbox

SSD_BOX_WEIGHTS = (10.0, 10.0, 5.0, 5.0)  # torchvision SSD BoxCoder defaults


def nms(boxes: np.ndarray, scores: np.ndarray, classes: np.ndarray, iou: float, max_det: int):
    """Per-class NMS. boxes are x0 y0 x1 y1. Returns kept indices, best first."""
    if len(scores) == 0:
        return np.empty(0, dtype=int)
    xywh = np.column_stack([boxes[:, :2], boxes[:, 2:] - boxes[:, :2]])
    keep = []
    for c in np.unique(classes):  # one small NMS per class: NMSBoxesBatched is one big O(n^2) NMS (problems.md)
        idx = np.flatnonzero(classes == c)
        kept = cv2.dnn.NMSBoxes(xywh[idx].tolist(), scores[idx].tolist(), 0.0, iou)
        keep.extend(idx[np.asarray(kept, dtype=int).reshape(-1)])
    keep = np.asarray(keep, dtype=int)
    return keep[np.argsort(-scores[keep])][:max_det]


def decode_yolo(out: np.ndarray, size: int, conf: float, normalized: bool):
    out = out[0]
    if out.shape[0] > out.shape[1]:  # some exports give (anchors, 4 + classes)
        out = out.T
    scores_all = out[4:]
    classes = scores_all.argmax(0)
    scores = scores_all[classes, np.arange(scores_all.shape[1])]
    m = scores > conf
    cx, cy, w, h = out[:4, m]
    boxes = np.column_stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])
    if normalized:
        boxes *= size
    return boxes, scores[m], classes[m]


def decode_ssd(deltas: np.ndarray, logits: np.ndarray, anchors: np.ndarray, conf: float):
    wx, wy, ww, wh = SSD_BOX_WEIGHTS
    aw, ah = anchors[:, 2] - anchors[:, 0], anchors[:, 3] - anchors[:, 1]
    acx, acy = anchors[:, 0] + aw / 2, anchors[:, 1] + ah / 2
    d = deltas[0]
    cx, cy = d[:, 0] / wx * aw + acx, d[:, 1] / wy * ah + acy
    w, h = np.exp(np.minimum(d[:, 2] / ww, 4.135)) * aw, np.exp(np.minimum(d[:, 3] / wh, 4.135)) * ah
    boxes = np.column_stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])
    e = np.exp(logits[0] - logits[0].max(1, keepdims=True))
    probs = (e / e.sum(1, keepdims=True))[:, 1:]  # drop background
    a, c = np.nonzero(probs > conf)
    return boxes[a], probs[a, c], c


def decode_hailo_nms(per_class: list, size: int, conf: float):
    boxes, scores, classes = [], [], []
    for c, dets in enumerate(per_class):
        for y0, x0, y1, x1, s in np.asarray(dets).reshape(-1, 5):
            if s > conf:
                boxes.append([x0 * size, y0 * size, x1 * size, y1 * size])
                scores.append(s)
                classes.append(c)
    return np.asarray(boxes, dtype=np.float32).reshape(-1, 4), np.asarray(scores), np.asarray(classes, dtype=int)


def unletterbox(boxes: np.ndarray, lb: Letterbox, frame_shape) -> np.ndarray:
    h, w = frame_shape[:2]
    boxes = (boxes - [lb.pad_x, lb.pad_y, lb.pad_x, lb.pad_y]) / lb.scale
    return np.clip(boxes, 0, [w, h, w, h])
