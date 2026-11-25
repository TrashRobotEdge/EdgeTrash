# Models

**No model files in git.** Trained/compiled models are attached to GitHub Releases with a SHA-256 checksum. `models/download.sh` (TODO) fetches them.

Per model we log: source weights, training config, export chain (ONNX → TFLite / `.hef` via Hailo DFC / `.blob` via OpenVINO / TVM), tool versions, precision, calibration set. Model choice: see the compile-test ADR in `docs/decisions/`.
