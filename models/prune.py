"""Structured pruning (torch-pruning): remove whole channels until MACs <= --target of the
original, then optionally fine-tune. Ultralytics models only. VM / laptop only.

    python models/prune.py --arch yolov8n --weights coco --target 0.5                      # dry run, no fine-tune
    python models/prune.py --arch yolov8n --weights models/files/yolov8n-cans.pt --target 0.5 --data data/cans.yaml --epochs 30

Writes models/files/<arch>-<tag>_p<percent removed>[ft].pt (ft = fine-tuned after pruning). Then export.py --weights that file.
Why structured: zeroed weights (unstructured) don't make ONNX Runtime / LiteRT any faster,
removed channels do. The Detect head is left untouched so the decoder stays the same.
"""

import argparse
from pathlib import Path

import torch
import torch_pruning as tp

from common import MODEL_DIR, REPO
from pruning import split_c2f


def prune(model: torch.nn.Module, size: int, target: float, steps: int = 20):
    example = torch.randn(1, 3, size, size)
    base_macs, base_params = tp.utils.count_ops_and_params(model, example)
    pruner = tp.pruner.MetaPruner(
        model,
        example,
        importance=tp.importance.GroupMagnitudeImportance(p=2),
        pruning_ratio=0.9,
        iterative_steps=steps,
        ignored_layers=[model.model[-1]],
    )
    macs = base_macs
    for _ in range(steps):
        if macs <= target * base_macs:
            break
        pruner.step()
        macs, params = tp.utils.count_ops_and_params(model, example)
    print(
        f"[prune] MACs {base_macs / 1e9:.2f} G -> {macs / 1e9:.2f} G ({macs / base_macs:.0%}), "
        f"params {base_params / 1e6:.2f} M -> {params / 1e6:.2f} M"
    )
    return macs / base_macs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arch", required=True, choices=["yolov5nu", "yolov8n", "yolo11n"])
    ap.add_argument("--weights", default="coco")
    ap.add_argument("--size", type=int, default=640)
    ap.add_argument("--target", type=float, default=0.5, help="keep at most this fraction of the MACs")
    ap.add_argument("--data", type=Path, help="dataset yaml: fine-tune after pruning")
    ap.add_argument("--epochs", type=int, default=30)
    args = ap.parse_args()

    from ultralytics import YOLO
    from ultralytics.models.yolo.detect import DetectionTrainer

    yolo = YOLO(f"{args.arch}.pt" if args.weights == "coco" else args.weights)
    model = yolo.model.float().eval()
    split_c2f(model)
    for p in model.parameters():
        p.requires_grad_(True)
    kept = prune(model, args.size, args.target)

    if args.data:

        class PrunedTrainer(DetectionTrainer):  # Ultralytics rebuilds the model from its yaml otherwise
            def get_model(self, cfg=None, weights=None, verbose=True):
                return model

        yolo.train(
            data=str(args.data),
            epochs=args.epochs,
            imgsz=args.size,
            seed=0,
            deterministic=True,
            trainer=PrunedTrainer,
            project=str(REPO / "runs"),
            name=f"{args.arch}-pruned",
            exist_ok=True,
        )
        model = yolo.trainer.model

    tag = "coco" if args.weights == "coco" else Path(args.weights).stem.split("-", 1)[-1]
    out = MODEL_DIR / f"{args.arch}-{tag}_p{round(100 * (1 - kept))}{'ft' if args.data else ''}.pt"  # ft = fine-tuned
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.half(), "train_args": {"imgsz": args.size}, "pruned_macs_kept": kept, "fine_tuned": bool(args.data)}, out)
    print(f"[prune] {out}")


if __name__ == "__main__":
    main()
