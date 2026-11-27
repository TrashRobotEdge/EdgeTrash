# Models

**No model files in git.** They live in `models/files/` (ignored). Each one is listed in `manifest.yaml` (in git) with its SHA-256, size, settings, tool versions and **the exact command that built it**. `edgetrash` refuses to load a file whose checksum doesn't match. To move files to a Pi: `setup/bundle.sh` ([setup/README.md](../setup/README.md)).

**Download** (no need to rebuild): `gh release download models-v1 -R TrashRobotEdge/EdgeTrash -D models/files && (cd models/files && sha256sum -c SHA256SUMS)`. Release `models-v1` = `yolov8n-coco-640-fp32-onnx`, used by TrashLocalisation.

Variant name = `<arch>-<weights>-<input size>-<precision>-<runtime>`, e.g. `yolov8n-coco-640-int8-tflite`. `weights` = `coco` (pretrained), `zoo` (vendor file), our tag (e.g. `cans`), `_p<N>` = N % of MACs pruned.

| Step | Command (VM) | Gives |
|---|---|---|
| Fine-tune | `python models/train.py --arch yolov8n\|yolo11n\|yolov5nu\|ssdlite --data data/cans.yaml` | `files/<arch>-cans.pt` |
| Prune (structured) | `python models/prune.py --arch yolov8n --weights models/files/yolov8n-cans.pt --target 0.5 --data data/cans.yaml` | `files/yolov8n-cans_p50ft.pt` (`ft` = fine-tuned after pruning) |
| CPU formats | `python models/export.py --arch yolov8n --weights … --to onnx-fp32 onnx-int8 tflite-fp32 tflite-int8 --calib data/splits/<calib>.txt [--size 320]` | ONNX FP32, ORT INT8 (static QDQ), LiteRT FP32 / INT8 (static) |
| OAK | `python models/compile_oak.py --variant <…-fp32-onnx>` | `.blob` FP16 (Luxonis online converter) |
| Hailo | `.venv-hailo/bin/python models/compile_hailo.py --variant <…-fp32-onnx> --calib …` · `--zoo yolov8n` for the vendor reference | `.hef` INT8 |
| TVM | `python models/compile_tvm.py --variant <…-fp32-onnx> --target pi4\|pi5` (on that Pi: TVM conflicts with the VM's litert-torch) | `.so`, untuned |

Known gaps (see [docs/problems.md](../docs/problems.md)): no FP16 TFLite (litert-torch, our one TFLite converter, has no FP16 mode) · Hailo: no zoo recipe for YOLOv5nu / SSDLite · a pruned model without `--data` (no fine-tune) detects nothing (mAP 0.0 at 51 %, scores near 0 already at 7 %).
