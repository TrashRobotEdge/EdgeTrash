#!/usr/bin/env bash
# The whole VM dry run in one command: every model file, accuracy, latency, tables and figures.
# COCO weights, so it checks the pipeline, not a device (protocol step 2). Needs: setup.sh --config vm-dev.
#   bash bench/dryrun.sh models     # every model file + models/manifest.yaml: commit the manifest, then
#   bash bench/dryrun.sh measure    # accuracy, latency, tables, figures (~1 h on the VM)
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PYTHONPATH="" PYTHONIOENCODING=utf-8
PY=.venv/bin/python
CALIB=data/splits/coco-calib.txt

if [[ "${1:-}" == models ]]; then  # models/README.md
$PY models/export.py --arch yolov8n --to onnx-fp32 onnx-int8 tflite-fp32 tflite-int8 --calib $CALIB
$PY models/export.py --arch yolov8n --size 320 --to onnx-fp32
$PY models/prune.py --arch yolov8n --target 0.5
$PY models/export.py --arch yolov8n --weights models/files/yolov8n-coco_p51.pt --to onnx-fp32
$PY models/export.py --arch ssdlite --to onnx-fp32 onnx-int8 tflite-fp32 tflite-fp16 tflite-int8 --calib $CALIB
$PY models/compile_oak.py --variant yolov8n-coco-640-fp32-onnx
$PY models/compile_oak.py --variant ssdlite-coco-320-fp32-onnx
$PY models/compile_hailo.py --zoo yolov8n
elif [[ "${1:-}" == measure ]]; then  # accuracy on 500 COCO images, then latency (idle + variants x reps, interleaved)
for v in $(grep -E '^\s+- ' bench/configs/vm-dev.yaml | awk '{print $2}' | grep -v '^hsv-'); do
  $PY bench/accuracy.py --device vm --variant "$v" --split data/splits/coco-eval.txt --ann data/coco/annotations/coco-eval-subset.json
done
# FP32 reference for the Hailo .hef files, scored like them: boxes under 0.2 dropped (protocol §3)
$PY bench/accuracy.py --device vm --variant yolov8n-coco-640-fp32-onnx --min-score 0.2 --split data/splits/coco-eval.txt --ann data/coco/annotations/coco-eval-subset.json
$PY bench/sweep.py --config bench/configs/vm-dev.yaml
$PY bench/analyze.py
else
  echo "usage: bash bench/dryrun.sh models|measure"; exit 2
fi
