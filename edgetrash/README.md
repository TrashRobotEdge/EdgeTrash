# edgetrash

ROS-free detector used by both the benchmark (`bench/`) and the robot's `trash_detector` node, so what we measure is what runs on the robot.

```python
from edgetrash import load_variant

det = load_variant("yolov8n-coco-640-int8-onnx")  # name from models/manifest.yaml, checksum checked
detections, timings = det(bgr_frame)  # [Detection(cls, score, box)], {pre_ms, infer_ms, post_ms}
```

- `preprocess.py`: letterbox (shared by every backend and by calibration).
- `backends/<runtime>_backend.py`: one small class per runtime (`onnx`, `tflite`, `tvm`, `hailo`, `oak`, `hsv`): `layout`, `dtype`, `infer(x)`, `info()`. Imported lazily, so a Pi without DepthAI still runs ONNX.
- `postprocess.py`: decoders per head (`yolo`, `ssd`, `hailo_nms`) + NMS + boxes back to frame pixels.

Adding a runtime = one file in `backends/` + its name in `BACKENDS`. Test: `pytest` (one smoke test, HSV, no model file needed).
