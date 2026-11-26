"""Fine-tune one architecture on our dataset (COCO-pretrained start). VM / laptop only.

    python models/train.py --arch yolov8n --data data/cans.yaml --epochs 100
    python models/train.py --arch ssdlite --data data/cans.yaml --epochs 100

Writes models/files/<arch>-<tag>.pt. Then: models/export.py --weights that file.
"""

import argparse
import math
import shutil
from pathlib import Path

import torch
import yaml

from common import MODEL_DIR, REPO, pretrained

ULTRALYTICS = {"yolov5nu", "yolov8n", "yolo11n"}


def train_ultralytics(args) -> Path:
    from ultralytics import YOLO

    model = YOLO(str(pretrained(args.arch)))
    model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        seed=0,
        deterministic=True,
        project=str(REPO / "runs"),
        name=f"{args.arch}-{args.tag}",
        exist_ok=True,
        device=0 if torch.cuda.is_available() else "cpu",
    )
    return Path(model.trainer.best)


def train_ssdlite(args) -> Path:
    import ssdlite

    spec = yaml.safe_load(Path(args.data).read_text())
    root = Path(args.data).parent / spec.get("path", ".")
    names = list(spec["names"].values()) if isinstance(spec["names"], dict) else spec["names"]
    data = ssdlite.YoloFolder(root / spec["train"], train=True)
    loader = torch.utils.data.DataLoader(data, batch_size=args.batch, shuffle=True, collate_fn=lambda b: tuple(zip(*b, strict=True)))
    torch.manual_seed(0)
    model = ssdlite.build(len(names)).train()
    opt = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=4e-5)
    steps = args.epochs * len(loader)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.5 * (1 + math.cos(math.pi * s / steps)))
    for epoch in range(args.epochs):
        total = 0.0
        for images, targets in loader:
            loss = sum(model(list(images), list(targets)).values())
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            total += loss.item()
        print(f"epoch {epoch + 1}/{args.epochs} loss {total / len(loader):.3f}")
    out = REPO / "runs" / f"ssdlite-{args.tag}.pt"
    out.parent.mkdir(exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "names": names}, out)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arch", required=True, choices=sorted(ULTRALYTICS | {"ssdlite"}))
    p.add_argument("--data", required=True, type=Path, help="Ultralytics-format dataset yaml")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--imgsz", type=int, default=640, help="ignored for ssdlite (always 320)")
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--tag", default="cans", help="weights name in the variant, e.g. cans (no '-')")
    args = p.parse_args()
    best = train_ultralytics(args) if args.arch in ULTRALYTICS else train_ssdlite(args)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    out = MODEL_DIR / f"{args.arch}-{args.tag}.pt"
    shutil.copy(best, out)
    print(f"[train] {out}")


if __name__ == "__main__":
    main()
