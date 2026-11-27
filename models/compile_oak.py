"""ONNX FP32 variant -> OAK-D Pro .blob (Myriad X VPU, FP16) with Luxonis' blobconverter.
Uploads the ONNX to Luxonis' online converter (OK'd by the team, 26/11: nothing sensitive).

    python models/compile_oak.py --variant yolov8n-coco-640-fp32-onnx

The blob takes uint8 RGB planar; /255 is compiled in (--scale_values), so the OAK backend
sends the letterboxed frame as is. Registers <arch>-<tag>-<size>-fp16-oak.
"""

import argparse
import shutil
import tempfile
from pathlib import Path

import blobconverter

from common import MODEL_DIR, model_path, register
from edgetrash.detector import read_manifest

OPENVINO = "2022.1"  # last version blobconverter offers for Myriad X


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", required=True, help="an fp32-onnx variant from models/manifest.yaml")
    ap.add_argument("--shaves", type=int, default=6, help="SHAVE cores the blob may use (Myriad X has 16)")
    args = ap.parse_args()
    v = read_manifest()[args.variant]
    name = args.variant.replace("-fp32-onnx", "-fp16-oak")
    out = model_path(name, "blob")
    with tempfile.TemporaryDirectory() as t:
        blob = blobconverter.from_onnx(
            model=str(MODEL_DIR / v["file"]),
            data_type="FP16",
            shaves=args.shaves,
            version=OPENVINO,
            optimizer_params=["--scale_values=[255,255,255]"],
            output_dir=t,
            use_cache=False,
        )
        shutil.copy(Path(blob), out)
    keep = {k: v[k] for k in ("arch", "weights", "input_size", "head", "classes", "anchors") if k in v}
    register(
        name,
        out,
        ("blobconverter",),
        backend="oak",
        precision="fp16",
        source=args.variant,
        openvino=OPENVINO,
        backend_opts={"shaves": args.shaves},
        **keep,
    )


if __name__ == "__main__":
    main()
