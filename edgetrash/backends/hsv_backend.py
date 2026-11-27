"""Classical baseline: red colour mask -> contours -> boxes. No model file.
Thresholds come from backend_opts in models/manifest.yaml."""

import cv2
import numpy as np


class Backend:
    layout, dtype = "bgr", "uint8"

    def __init__(self, path=None, ranges=((0, 120, 70, 10, 255, 255), (170, 120, 70, 180, 255, 255)), min_area: int = 400):
        self.ranges = [(np.array(r[:3]), np.array(r[3:])) for r in ranges]
        self.min_area = min_area

    def infer(self, hsv):
        mask = sum(cv2.inRange(hsv, lo, hi) for lo, hi in self.ranges).clip(0, 255).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = [cv2.boundingRect(c) for c in contours if cv2.contourArea(c) >= self.min_area]
        boxes = np.array([[x, y, x + w, y + h] for x, y, w, h in boxes], dtype=np.float32).reshape(-1, 4)
        return boxes, np.ones(len(boxes), dtype=np.float32), np.zeros(len(boxes), dtype=int)

    def info(self):
        return {"runtime": "opencv", "version": cv2.__version__, "min_area": self.min_area}
