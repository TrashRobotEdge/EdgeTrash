"""Pruning helper shared by prune.py and (when unpickling a pruned model) export.py."""

import copy

import torch
from torch import nn


class SplitC2f(nn.Module):
    """Ultralytics C2f with its first conv split in two halves: same maths, no tensor.chunk().
    torch-pruning can't follow chunk() (wrong channel indices), its YOLOv8 example does the same.
    Lives in its own module, not prune.py's __main__, so a pruned model saved with torch.save loads in export.py."""

    def __init__(self, c2f: nn.Module):
        super().__init__()
        conv, half = c2f.cv1, c2f.cv1.conv.out_channels // 2
        self.cv0, self.cv1 = copy.deepcopy(conv), copy.deepcopy(conv)
        for part, keep in ((self.cv0, slice(0, half)), (self.cv1, slice(half, None))):
            c, bn = conv.conv, conv.bn
            part.conv = nn.Conv2d(c.in_channels, half, c.kernel_size, c.stride, c.padding, groups=c.groups, bias=False)
            part.conv.weight.data = c.weight.data[keep].clone()
            part.bn = nn.BatchNorm2d(half, eps=bn.eps, momentum=bn.momentum)
            part.bn.weight.data, part.bn.bias.data = bn.weight.data[keep].clone(), bn.bias.data[keep].clone()
            part.bn.running_mean, part.bn.running_var = bn.running_mean[keep].clone(), bn.running_var[keep].clone()
        self.cv2, self.m = c2f.cv2, c2f.m
        for attr in ("f", "i", "type", "np"):  # Ultralytics' graph wiring
            if hasattr(c2f, attr):
                setattr(self, attr, getattr(c2f, attr))
        self.train(c2f.training)  # new BatchNorms start in training mode otherwise

    def forward(self, x):
        y = [self.cv0(x), self.cv1(x)]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


def split_c2f(module: nn.Module) -> None:
    from ultralytics.nn.modules.block import C2f

    for name, child in module.named_children():
        if isinstance(child, C2f):
            setattr(module, name, SplitC2f(child))
        else:
            split_c2f(child)
