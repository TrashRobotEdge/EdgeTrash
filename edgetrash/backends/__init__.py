"""One file per runtime. Imported lazily, so a Pi without depthai can still run onnx.

Each backend has the same small interface:
    layout: "nchw" | "nhwc" | "bgr"     dtype: "float32" | "uint8"
    infer(x) -> list of raw outputs     info() -> dict of versions/settings for meta.json
"""

import importlib

BACKENDS = ("onnx", "tflite", "hailo", "oak", "tvm", "hsv")


def get_backend(name: str):
    if name not in BACKENDS:
        raise ValueError(f"unknown backend {name!r}, pick one of {BACKENDS}")
    return importlib.import_module(f"edgetrash.backends.{name}_backend").Backend
