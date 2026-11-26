"""SSDLite-MobileNetV3 (torchvision): the non-YOLO family. Used by train.py and export.py.

We export the raw head (box deltas + class logits) and decode on the CPU in
edgetrash.postprocess.decode_ssd, so the exported graph has no NMS that the Hailo/OAK
compilers would choke on. Input: letterboxed RGB in [0, 1], like every other model.
"""

from functools import partial
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torchvision.models.detection import SSDLite320_MobileNet_V3_Large_Weights, ssdlite320_mobilenet_v3_large
from torchvision.models.detection import _utils as det_utils
from torchvision.models.detection.image_list import ImageList
from torchvision.models.detection.ssdlite import SSDLiteClassificationHead

from edgetrash.preprocess import letterbox

SIZE = 320  # SSDLite320's native input


def build(num_classes: int | None = None) -> nn.Module:
    """COCO-pretrained SSDLite. With num_classes, the class head is replaced for fine-tuning."""
    model = ssdlite320_mobilenet_v3_large(weights=SSDLite320_MobileNet_V3_Large_Weights.COCO_V1)
    if num_classes is not None:
        channels = det_utils.retrieve_out_channels(model.backbone, (SIZE, SIZE))
        anchors = model.anchor_generator.num_anchors_per_location()
        norm = partial(nn.BatchNorm2d, eps=0.001, momentum=0.03)
        model.head.classification_head = SSDLiteClassificationHead(channels, anchors, num_classes + 1, norm)
    return model


def load(weights: str) -> tuple[nn.Module, list[str] | None]:
    """'coco' -> torchvision weights (91 COCO ids, background = 0), else our checkpoint."""
    if weights == "coco":
        return build().eval(), None
    ckpt = torch.load(weights, map_location="cpu")
    model = build(len(ckpt["names"]))
    model.load_state_dict(ckpt["state_dict"])
    return model.eval(), ckpt["names"]


class HardSigmoid(nn.Module):
    """relu6(x + 3) / 6: exactly nn.Hardsigmoid, but exported as Add/Clip/Div. OpenVINO 2022.1 on the
    Myriad X (OAK) has no HardSigmoid layer (docs/problems.md). Used for every runtime, so all run one graph."""

    def forward(self, x):
        return torch.nn.functional.relu6(x + 3) / 6


class HardSwish(nn.Module):
    def forward(self, x):
        return x * torch.nn.functional.relu6(x + 3) / 6


def plain_activations(module: nn.Module) -> nn.Module:
    for name, child in module.named_children():
        if isinstance(child, nn.Hardsigmoid):
            setattr(module, name, HardSigmoid())
        elif isinstance(child, nn.Hardswish):
            setattr(module, name, HardSwish())
        else:
            plain_activations(child)
    return module


class RawHead(nn.Module):
    """[0, 1] RGB in -> (box deltas, class logits) out. Normalisation matches torchvision's transform."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = plain_activations(model)

    def forward(self, x):
        feats = list(self.model.backbone((x - 0.5) / 0.5).values())
        out = self.model.head(feats)
        return out["bbox_regression"], out["cls_logits"]


@torch.no_grad()
def anchors(model: nn.Module) -> np.ndarray:
    x = torch.zeros(1, 3, SIZE, SIZE)
    feats = list(model.eval().backbone(x).values())
    return model.anchor_generator(ImageList(x, [(SIZE, SIZE)]), feats)[0].detach().numpy()


class YoloFolder(torch.utils.data.Dataset):
    """Ultralytics-format split: images/<split>/x.jpg + labels/<split>/x.txt (cls cx cy w h, normalised)."""

    def __init__(self, image_dir: Path, train: bool):
        self.images = sorted(p for p in Path(image_dir).iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
        self.train = train

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        path = self.images[i]
        frame = cv2.imread(str(path))
        h, w = frame.shape[:2]
        rgb, lb = letterbox(frame, SIZE)
        label_file = Path(str(path.parent).replace("images", "labels")) / f"{path.stem}.txt"
        rows = np.loadtxt(label_file, ndmin=2) if label_file.exists() else np.zeros((0, 5))
        cls, cx, cy, bw, bh = rows.T if len(rows) else (np.zeros(0),) * 5
        boxes = np.column_stack([(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h])
        boxes = boxes * lb.scale + [lb.pad_x, lb.pad_y, lb.pad_x, lb.pad_y]
        img = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255
        if self.train and torch.rand(1) < 0.5:
            img = img.flip(-1)
            boxes[:, [0, 2]] = SIZE - boxes[:, [2, 0]]
        target = {"boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4), "labels": torch.tensor(cls + 1, dtype=torch.int64)}
        return img, target
