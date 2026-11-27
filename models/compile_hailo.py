"""ONNX FP32 variant -> Hailo-8L .hef (INT8) with the Hailo Model Zoo's `hailomz compile`.
Runs in its own venv (.venv-hailo, Python 3.10) with DFC 3.33.0 + Model Zoo v2.17: the pair
that targets HailoRT 4.23.0, the version Raspberry Pi OS Trixie ships. See setup/README.md.

    .venv-hailo/bin/python models/compile_hailo.py --variant yolov8n-coco-640-fp32-onnx --calib data/splits/coco-calib.txt
    python models/compile_hailo.py --zoo yolov8n     # reference only: Hailo's own pre-compiled COCO .hef, no DFC needed

Model Zoo recipes give the end nodes, on-chip normalisation and the NMS config, so only
architectures the zoo knows are accepted. Registers <arch>-<tag>-<size>-int8-hailo.
"""

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

from common import MODEL_DIR, model_path, read_list, register
from edgetrash.detector import read_manifest

ZOO_NAME = {"yolov8n": "yolov8n", "yolo11n": "yolov11n"}  # yolov5nu / ssdlite: no zoo recipe (docs/problems.md)
CALIB_LIMIT = 200
OPT_LEVEL = re.compile(r"Reducing optimization level to (\d)")  # the DFC's own warning
CALIB_USED = re.compile(r"Using dataset with (\d+) entries for calibration")  # the zoo recipe may use fewer than we give
HEAD_CONV = re.compile(r"/model\.\d+/(cv[23])\.(\d)/cv[23]\.\d\.2/Conv")
ZOO_URL = "https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.17.0/{hw}/{name}.hef"  # Model Zoo v2.17 docs


def head_convs(onnx_file: Path) -> list[str]:
    """The Detect head's 6 last convs (box + class per scale), where Hailo's NMS takes over.
    The yolov8n recipe names none and the parser's guess fails on current Ultralytics exports
    (docs/problems.md); the yolov11 recipes list these same 6."""
    import onnx

    names = [n.name for n in onnx.load(str(onnx_file)).graph.node if n.op_type == "Conv"]
    convs = sorted((m.group(2), m.group(1), n) for n in names if (m := HEAD_CONV.fullmatch(n)))
    if len(convs) != 6:
        raise SystemExit(f"{onnx_file.name}: expected 6 head convs, found {[c[2] for c in convs]}")
    return [n for _, _, n in convs]  # per scale: box (cv2) then class (cv3)


def run_logged(cmd: list[str], cwd: Path) -> str:
    """Run hailomz, show its output, return what it actually used: optimization level, calibration images."""
    level, n_calib = "default", None
    with subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) as proc:
        for line in proc.stdout:
            print(line, end="", flush=True)
            if m := OPT_LEVEL.search(line):
                level = m.group(1)
            if m := CALIB_USED.search(line):
                n_calib = int(m.group(1))
    if proc.returncode:
        raise subprocess.CalledProcessError(proc.returncode, cmd)
    if n_calib is None:
        raise SystemExit("no 'Using dataset with N entries' line in the DFC output: update CALIB_USED for this DFC version")
    return level, n_calib


def fetch_zoo(arch: str, hw: str) -> None:
    """Vendor reference (plan: zoo files are a reference, never a result about our models)."""
    url = ZOO_URL.format(hw=hw, name=ZOO_NAME[arch])
    out = model_path(f"{arch}-zoo-640-int8-hailo", "hef")
    with urllib.request.urlopen(url) as r, open(out, "wb") as f:
        shutil.copyfileobj(r, f)
    register(
        out.stem,
        out,
        (),
        backend="hailo",
        precision="int8",
        source=url,
        hw_arch=hw,
        arch=arch,
        weights="zoo",
        input_size=640,
        head="hailo_nms",
        classes="coco80",
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", help="an fp32-onnx variant from models/manifest.yaml")
    ap.add_argument("--calib", type=Path)
    ap.add_argument("--hw-arch", default="hailo8l")
    ap.add_argument("--zoo", choices=sorted(ZOO_NAME), help="download Hailo's pre-compiled COCO .hef instead")
    args = ap.parse_args()
    if args.zoo:
        return fetch_zoo(args.zoo, args.hw_arch)
    if not (args.variant and args.calib):
        ap.error("--variant and --calib are needed to compile")
    v = read_manifest()[args.variant]
    if v["arch"] not in ZOO_NAME:
        raise SystemExit(f"{v['arch']}: no Hailo Model Zoo recipe, see docs/problems.md")
    n_classes = 80 if v["classes"] == "coco80" else len(v["classes"])
    name = args.variant.replace("-fp32-onnx", "-int8-hailo")
    out = model_path(name, "hef")
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        calib_dir = tmp / "calib"
        calib_dir.mkdir()
        for p in read_list(args.calib)[:CALIB_LIMIT]:
            (calib_dir / p.name).symlink_to(p)
        hailomz = Path(sys.executable).parent / "hailomz"  # the venv's own CLI, activated or not
        if not hailomz.exists():
            raise SystemExit(f"{hailomz} not found: run this with .venv-hailo/bin/python (setup.sh --config vm-hailo)")
        cmd = [
            str(hailomz),
            "compile",
            ZOO_NAME[v["arch"]],
            "--ckpt",
            str(MODEL_DIR / v["file"]),
            "--hw-arch",
            args.hw_arch,
            "--calib-path",
            str(calib_dir),
            "--classes",
            str(n_classes),
            "--end-node-names",
            *head_convs(MODEL_DIR / v["file"]),
        ]
        print(" ".join(cmd))
        level, n_calib = run_logged(cmd, tmp)
        shutil.copy(tmp / f"{ZOO_NAME[v['arch']]}.hef", out)
    register(
        name,
        out,
        ("hailo-dataflow-compiler", "hailo-model-zoo"),
        backend="hailo",
        precision="int8",
        source=args.variant,
        hw_arch=args.hw_arch,
        arch=v["arch"],
        weights=v["weights"],
        input_size=v["input_size"],
        head="hailo_nms",
        classes=v["classes"],
        calib=str(args.calib),
        calib_images=n_calib,  # what the DFC used, not the CALIB_LIMIT we linked
        optimization_level=level,  # 0 = no accuracy recovery: the DFC drops to it without a GPU (docs/problems.md)
    )


if __name__ == "__main__":
    main()
