"""Export one model to the CPU formats, one variant per --to value. VM / laptop only.

    python models/export.py --arch yolov8n --to onnx-fp32 onnx-int8 tflite-fp32 tflite-fp16 tflite-int8 \
        --calib data/splits/coco-calib.txt
    python models/export.py --arch ssdlite --weights models/files/ssdlite-cans.pt --to onnx-fp32 --calib data/splits/cans-calib.txt

Variant name = <arch>-<weights tag>-<size>-<precision>-<runtime>, e.g. yolov8n-coco-640-int8-tflite.
Files go to models/files/ (never git), entries to models/manifest.yaml (in git).
INT8 = static full-integer quantization (int8 weights and activations), calibrated on --calib.
OAK / Hailo / TVM have their own compile_*.py, starting from the onnx-fp32 variant.

Every tool writes into a fresh temporary folder; we copy out the one file we need.
"""

import argparse
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np

from common import calib_array, calib_tensors, model_path, pretrained, read_list, register
from edgetrash.detector import MODEL_DIR, read_manifest

ULTRALYTICS = {"yolov5nu", "yolov8n", "yolo11n"}
TARGETS = ("onnx-fp32", "onnx-int8", "tflite-fp32", "tflite-fp16", "tflite-int8")
CALIB_LIMIT = 200  # images; keeps export to a few minutes
# Ultralytics 8.4 exports TFLite with litert-torch: one .tflite per call ("int8" = static int8 weights +
# activations). onnx2tf (SSDLite) writes them all at once; its dynamic-range "_int8" is not used.
ONNX2TF_TFLITE = {"fp32": "_float32.tflite", "fp16": "_float16.tflite", "int8": "_full_integer_quant.tflite"}
PKGS = ("ultralytics", "torch", "torchvision", "onnx", "onnxruntime", "litert-torch", "ai-edge-litert", "tensorflow", "onnx2tf")


def tag_of(weights: str) -> str:
    return "coco" if weights == "coco" else Path(weights).stem.split("-", 1)[-1]


def one(folder: Path, suffix: str) -> Path:
    found = sorted(folder.rglob(f"*{suffix}"))
    if len(found) != 1:
        raise RuntimeError(f"expected one *{suffix} in the export folder, found {[p.name for p in folder.rglob('*')]}")
    return found[0]


def ultralytics_export(args, tmp: Path, fmt: str, **opts):
    """Run Ultralytics' exporter on a copy of the weights inside tmp, so its outputs land in tmp."""
    from ultralytics import YOLO

    src = pretrained(args.arch) if args.weights == "coco" else Path(args.weights)
    if not src.exists():
        YOLO(str(src))  # downloads the COCO weights to models/files/
    work = tmp / f"{args.arch}.pt"
    shutil.copy(src, work)
    model = YOLO(str(work))
    model.export(format=fmt, imgsz=args.size, batch=1, **opts)
    return model


def calib_yaml(calib: Path, tmp: Path) -> Path:
    """Ultralytics wants a dataset yaml for INT8 calibration: point it at our calibration list."""
    images = tmp / "calib.txt"
    images.write_text("\n".join(str(p) for p in read_list(calib)[:CALIB_LIMIT]) + "\n")
    data = tmp / "calib.yaml"
    data.write_text(f"path: {tmp}\ntrain: {images}\nval: {images}\nnames: {{0: object}}\n")
    return data


def export_onnx(args, name: str) -> tuple[Path, dict]:
    out = model_path(name, "onnx")
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        if args.arch in ULTRALYTICS:
            model = ultralytics_export(args, tmp, "onnx", opset=17, simplify=True, dynamic=False)
            shutil.copy(one(tmp, ".onnx"), out)
            names = list(model.names.values())
            extra = {"head": "yolo", "classes": "coco80" if args.weights == "coco" else names}
        else:
            import torch

            import ssdlite

            model, names = ssdlite.load(args.weights)
            x = torch.zeros(1, 3, ssdlite.SIZE, ssdlite.SIZE)
            torch.onnx.export(
                ssdlite.RawHead(model).eval(),
                x,
                out,
                opset_version=17,
                input_names=["images"],
                output_names=["deltas", "logits"],
                dynamo=False,
            )
            anchors = model_path(name.removesuffix("-fp32-onnx"), "anchors.npy")
            np.save(anchors, ssdlite.anchors(model))
            extra = {"head": "ssd", "anchors": anchors.name, "classes": names or "coco91"}
    return out, extra


def yolo_decode_nodes(onnx_path: Path) -> list[str]:
    """The Detect head's decode tail (DFL, slices, sigmoid, concat): every non-Conv node of the last module.
    Its output concatenates boxes (0-640) and class scores (0-1): one INT8 scale for both rounds every
    score to 0 (docs/problems.md). Left in float; the head's convolutions and the backbone stay INT8."""
    import onnx

    nodes = onnx.load(str(onnx_path)).graph.node
    head = nodes[-1].name.split("/")[1]  # e.g. "model.22"
    return [n.name for n in nodes if n.name.startswith(f"/{head}/") and n.op_type != "Conv"]


def quantize_onnx(fp32: Path, out: Path, size: int, calib: Path, exclude: list[str]):
    """ONNX Runtime static INT8 (QDQ, per-channel weights), calibrated on our calibration split."""
    from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static
    from onnxruntime.quantization.shape_inference import quant_pre_process

    class Reader(CalibrationDataReader):
        def __init__(self):
            self.it = calib_tensors(calib, size, "nchw", "float32", CALIB_LIMIT)

        def get_next(self):
            x = next(self.it, None)
            return None if x is None else {"images": x}

    with tempfile.TemporaryDirectory() as t:
        pre = Path(t) / "pre.onnx"
        quant_pre_process(str(fp32), str(pre))
        quantize_static(
            str(pre),
            str(out),
            Reader(),
            quant_format=QuantFormat.QDQ,
            per_channel=True,
            activation_type=QuantType.QUInt8,
            weight_type=QuantType.QInt8,
            nodes_to_exclude=exclude,
        )


def export_tflite(args, fp32_onnx: Path, base: str, precisions: list[str]) -> dict[str, Path]:
    outs = {}
    for p in precisions:
        outs[p] = model_path(f"{base}-{p}-tflite", "tflite")
        with tempfile.TemporaryDirectory() as t:
            tmp = Path(t)
            if args.arch in ULTRALYTICS:  # Ultralytics -> litert-torch; its TFLite head normalises boxes to [0, 1]
                opts = {"quantize": {"fp32": None, "fp16": 16, "int8": 8}[p]}
                if p == "int8":
                    opts["data"] = str(calib_yaml(args.calib, tmp))
                ultralytics_export(args, tmp, "tflite", **opts)
                shutil.copy(one(tmp, ".tflite"), outs[p])
            else:  # our SSDLite ONNX -> onnx2tf, calibrated with our preprocessing
                import onnx2tf

                calib = None
                if p == "int8":
                    np.save(tmp / "calib.npy", calib_array(args.calib, args.size, CALIB_LIMIT))
                    calib = [["images", str(tmp / "calib.npy"), [[[[0, 0, 0]]]], [[[[255, 255, 255]]]]]]
                # onnx2tf validates with a sample file from cwd, else downloads one numpy can't load (pickled):
                # give it 20 of our calibration images instead
                sample = calib_array(args.calib, 128, 20).astype(np.float32) / 255
                np.save(tmp / "calibration_image_sample_data_20x128x128x3_float32.npy", sample)
                work = tmp / fp32_onnx.name  # onnx2tf overwrites its input file (docs/problems.md): give it a copy
                shutil.copy(fp32_onnx, work)
                cwd = os.getcwd()
                os.chdir(tmp)
                try:
                    onnx2tf.convert(
                        input_onnx_file_path=str(work),
                        output_folder_path=str(tmp / "tf"),
                        not_use_onnxsim=True,
                        verbosity="error",
                        output_integer_quantized_tflite=calib is not None,
                        custom_input_op_name_np_data_path=calib,
                        output_signaturedefs=True,
                    )
                finally:
                    os.chdir(cwd)
                shutil.copy(one(tmp, ONNX2TF_TFLITE[p]), outs[p])
    return outs


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arch", required=True, choices=sorted(ULTRALYTICS | {"ssdlite"}))
    p.add_argument("--weights", default="coco", help="'coco' (pretrained) or a .pt from train.py / prune.py")
    p.add_argument("--size", type=int, default=640)
    p.add_argument("--to", nargs="+", required=True, choices=TARGETS)
    p.add_argument("--calib", type=Path, help="calibration split (needed for int8)")
    args = p.parse_args()
    if args.arch == "ssdlite":
        args.size = 320
    if any(t.endswith("int8") for t in args.to) and not args.calib:
        p.error("--calib is needed for int8 targets")
    if args.arch in ULTRALYTICS and "tflite-fp16" in args.to:
        p.error("tflite-fp16: Ultralytics' LiteRT exporter has no FP16 (docs/problems.md)")
    base = f"{args.arch}-{tag_of(args.weights)}-{args.size}"
    common = {"arch": args.arch, "weights": args.weights, "input_size": args.size}
    calib = {"calib": str(args.calib), "calib_images": CALIB_LIMIT} if args.calib else {}

    known = read_manifest().get(f"{base}-fp32-onnx")
    if "onnx-fp32" in args.to or not known or not (MODEL_DIR / known["file"]).exists():
        fp32, extra = export_onnx(args, f"{base}-fp32-onnx")
        register(f"{base}-fp32-onnx", fp32, PKGS, backend="onnx", precision="fp32", **common, **extra)
    else:  # reuse it: the INT8 / TFLite variants then all come from the same FP32 file
        fp32 = MODEL_DIR / known["file"]
        extra = {k: known[k] for k in ("head", "classes", "anchors") if k in known}
    if "onnx-int8" in args.to:
        out = model_path(f"{base}-int8-onnx", "onnx")
        exclude = yolo_decode_nodes(fp32) if args.arch in ULTRALYTICS else []
        quantize_onnx(fp32, out, args.size, args.calib, exclude)
        register(f"{base}-int8-onnx", out, PKGS, backend="onnx", precision="int8", float_nodes=len(exclude), **common, **extra, **calib)
    tflite = [t.split("-")[1] for t in args.to if t.startswith("tflite")]
    if tflite:
        for precision, out in export_tflite(args, fp32, base, tflite).items():
            register(
                f"{base}-{precision}-tflite",
                out,
                PKGS,
                backend="tflite",
                precision=precision,
                normalized_boxes=args.arch in ULTRALYTICS,
                **common,
                **extra,
                **(calib if precision == "int8" else {}),
            )


if __name__ == "__main__":
    main()
