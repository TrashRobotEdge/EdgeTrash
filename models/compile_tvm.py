"""ONNX FP32 variant -> TVM (Relax) shared library for one CPU, untuned (no auto-tuning yet:
that is the 06/01 lecture, issue #31). Needs a C compiler on the machine that runs it.

    python models/compile_tvm.py --variant yolov8n-coco-640-fp32-onnx --target pi5   # run on the Pi itself
    python models/compile_tvm.py --variant yolov8n-coco-640-fp32-onnx --target vm

Registers <arch>-<tag>-<size>-fp32-tvm_<target>: the .so only runs on that CPU.
"""

import argparse

import onnx
import tvm
from tvm.relax.frontend.onnx import from_onnx

from common import MODEL_DIR, model_path, register
from edgetrash.detector import read_manifest

TARGETS = {
    "vm": "llvm",
    "pi4": "llvm -mtriple=aarch64-linux-gnu -mcpu=cortex-a72",
    "pi5": "llvm -mtriple=aarch64-linux-gnu -mcpu=cortex-a76",
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", required=True, help="an fp32-onnx variant from models/manifest.yaml")
    ap.add_argument("--target", required=True, choices=sorted(TARGETS))
    args = ap.parse_args()
    v = read_manifest()[args.variant]
    size = v["input_size"]
    mod = from_onnx(onnx.load(MODEL_DIR / v["file"]), shape_dict={"images": [1, 3, size, size]})
    with tvm.transform.PassContext(opt_level=3):
        exe = tvm.compile(mod, target=tvm.target.Target(TARGETS[args.target]))
    name = args.variant.replace("-fp32-onnx", f"-fp32-tvm_{args.target}")
    out = model_path(name, "so")
    exe.export_library(str(out))
    keep = {k: v[k] for k in ("arch", "weights", "input_size", "head", "classes", "anchors") if k in v}
    register(
        name,
        out,
        ("apache-tvm", "onnx"),
        backend="tvm",
        precision="fp32",
        source=args.variant,
        tvm_target=TARGETS[args.target],
        tuned=False,
        **keep,
    )


if __name__ == "__main__":
    main()
