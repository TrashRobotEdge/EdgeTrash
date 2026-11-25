"""edgetrash: ROS-free trash detection pipeline.

Contract (to implement in S1): every backend (hsv, onnx, tflite, hailo, oak, tvm)
exposes the same three stages, each timed with time.perf_counter:

    preprocess(frame)  -> input tensor
    infer(tensor)      -> raw outputs
    postprocess(raw)   -> list of detections (class, score, bbox in pixels)

The same code runs in the L1 benchmark (bench/) and in TrashLocalisation's
trash_detector ROS node, so what we measure is what runs on the robot.
"""

__version__ = "0.0.1"
