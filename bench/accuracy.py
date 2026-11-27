"""Accuracy of one variant, measured on the device's own output (Benchmarking p.8, p.31).

    python bench/accuracy.py --device vm --variant yolov8n-coco-640-fp32-onnx \
        --split data/splits/coco-eval.txt --ann data/coco/annotations/coco-eval-subset.json

Writes results/raw/<device>/accuracy/<variant>.json: mAP@0.5:0.95, mAP@0.5 (COCO tools, conf 0.001),
and precision / recall / F1 at the operating threshold (--conf, IoU 0.5).
"""

import argparse
import contextlib
import io
import json
from pathlib import Path

import cv2
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from edgetrash import load_variant
from edgetrash.detector import read_manifest
from meta import RAW, REPO, run_meta


def category_ids(classes, coco: COCO) -> list[int]:
    """Model class index -> COCO category id."""
    if classes == "coco80":
        return sorted(coco.getCatIds())  # Ultralytics' 80 classes are COCO's ids in order
    if classes == "coco91":
        return list(range(1, 92))  # torchvision: label l is COCO id l; we dropped background (0)
    by_name = {c["name"]: c["id"] for c in coco.loadCats(coco.getCatIds())}
    return [by_name[n] for n in classes]


def iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    x0, y0 = np.maximum(box[0], boxes[:, 0]), np.maximum(box[1], boxes[:, 1])
    x1, y1 = np.minimum(box[2], boxes[:, 2]), np.minimum(box[3], boxes[:, 3])
    inter = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
    area = lambda b: (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])  # noqa: E731
    return inter / (area(box) + area(boxes) - inter + 1e-9)


def xyxy(b):
    return np.array([b[0], b[1], b[0] + b[2], b[1] + b[3]])


def precision_recall(dets: list[dict], coco: COCO, image_ids: list[int], conf: float):
    """Greedy one-to-one matching at IoU >= 0.5, all classes pooled (micro average)."""
    tp = fp = fn = 0
    by_image = {}
    for d in dets:
        if d["score"] >= conf:
            by_image.setdefault(d["image_id"], []).append(d)
    for img in image_ids:
        gts = coco.loadAnns(coco.getAnnIds(imgIds=img, iscrowd=False))
        mine = by_image.get(img, [])
        for cat in {a["category_id"] for a in gts} | {d["category_id"] for d in mine}:
            g = np.array([xyxy(a["bbox"]) for a in gts if a["category_id"] == cat]).reshape(-1, 4)
            used = np.zeros(len(g), bool)
            for x in sorted((d for d in mine if d["category_id"] == cat), key=lambda d: -d["score"]):
                ious = iou(xyxy(x["bbox"]), g) if len(g) else np.zeros(0)
                j = int(np.argmax(ious)) if len(ious) else -1
                if j >= 0 and ious[j] >= 0.5 and not used[j]:
                    used[j], tp = True, tp + 1
                else:
                    fp += 1
            fn += int((~used).sum())
    p, r = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return p, r, 2 * p * r / max(p + r, 1e-9)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", required=True, help="device label, same as in bench/configs/<device>.yaml")
    ap.add_argument("--variant", required=True)
    ap.add_argument("--split", type=Path, required=True, help="image list, one path per line")
    ap.add_argument("--ann", type=Path, required=True, help="COCO-format annotations json")
    ap.add_argument("--conf", type=float, default=0.25, help="operating threshold for P/R/F1")
    args = ap.parse_args()

    with contextlib.redirect_stdout(io.StringIO()):
        coco = COCO(str(args.ann))
    by_file = {im["file_name"]: im["id"] for im in coco.dataset["images"]}
    v = read_manifest()[args.variant]
    cats = category_ids(v["classes"], coco)
    det = load_variant(args.variant, conf=0.001, max_det=300)  # COCO protocol: low threshold for mAP

    paths = [REPO / line.strip() for line in args.split.read_text().splitlines() if line.strip()]
    results, image_ids = [], []
    for i, path in enumerate(paths):
        image_ids.append(by_file[path.name])
        for d in det(cv2.imread(str(path)))[0]:
            x0, y0, x1, y1 = d.box
            results.append({"image_id": image_ids[-1], "category_id": cats[d.cls], "bbox": [x0, y0, x1 - x0, y1 - y0], "score": d.score})
        if (i + 1) % 100 == 0:
            print(f"{i + 1}/{len(paths)}")

    with contextlib.redirect_stdout(io.StringIO()):
        ev = COCOeval(coco, coco.loadRes(results) if results else COCO(), "bbox")
        ev.params.imgIds = image_ids
        ev.evaluate(), ev.accumulate(), ev.summarize()
    p, r, f1 = precision_recall(results, coco, image_ids, args.conf)
    out = {
        "variant": args.variant,
        "split": str(args.split),
        "n_images": len(paths),
        "map50_95": round(float(ev.stats[0]), 4),
        "map50": round(float(ev.stats[1]), 4),
        "conf": args.conf,
        "precision": round(p, 4),
        "recall": round(r, 4),
        "f1": round(f1, 4),
        "backend": det.backend.info(),
        "meta": run_meta(),
    }
    dest = RAW / args.device / "accuracy" / f"{args.variant}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2))
    print(json.dumps({k: out[k] for k in ("variant", "map50_95", "map50", "precision", "recall", "f1")}))


if __name__ == "__main__":
    main()
