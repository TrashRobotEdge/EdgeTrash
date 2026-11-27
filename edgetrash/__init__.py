"""edgetrash: ROS-free trash detection pipeline.

Every backend runs the same three stages, each timed with time.perf_counter:

    preprocess(frame) -> input tensor      (shared letterbox, edgetrash.preprocess)
    infer(tensor)     -> raw outputs       (one file per backend, edgetrash.backends)
    postprocess(raw)  -> detections        (shared decoders + NMS, edgetrash.postprocess)

The same code runs in the L1 benchmark (bench/) and in TrashLocalisation's
trash_detector ROS node, so what we measure is what runs on the robot.
"""

from edgetrash.detector import Detection, Detector, load_variant

__version__ = "0.1.0"
__all__ = ["Detection", "Detector", "load_variant"]
