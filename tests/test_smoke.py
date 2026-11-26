"""The one smoke test (CLAUDE.md): edgetrash imports and runs on one sample input.
Uses the HSV backend, so it needs no model file and runs on every machine."""

import numpy as np

from edgetrash.backends import get_backend
from edgetrash.detector import Detector


def test_hsv_finds_one_red_can():
    frame = np.full((480, 640, 3), 40, np.uint8)
    frame[200:300, 300:350] = (30, 30, 200)  # BGR red block: a "can"
    det = Detector(get_backend("hsv")(), head="hsv", input_size=0)
    dets, timings = det(frame)
    assert len(dets) == 1
    x0, y0, x1, y1 = dets[0].box
    assert (x0, y0, x1, y1) == (300, 200, 350, 300)
    assert set(timings) == {"pre_ms", "infer_ms", "post_ms"}
