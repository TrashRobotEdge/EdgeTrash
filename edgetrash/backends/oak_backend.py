"""OAK-D Pro (Myriad X VPU) through DepthAI v3. For the L1 benchmark we replay frames,
so every frame is sent host -> VPU over USB (that transfer is in infer_ms, Benchmarking
p.6 "transfer time"). The .blob takes uint8 planar RGB; scaling to [0, 1] is compiled in."""

import depthai as dai
import numpy as np


class Backend:
    layout, dtype = "nchw", "uint8"

    def __init__(self, path, shaves: int = 6):
        self.pipeline = dai.Pipeline()
        nn = self.pipeline.create(dai.node.NeuralNetwork)
        nn.setBlobPath(str(path))
        self.in_q = nn.input.createInputQueue()
        self.out_q = nn.out.createOutputQueue()
        self.pipeline.start()
        self.shaves = shaves

    def infer(self, x):
        frame = dai.ImgFrame()
        frame.setType(dai.ImgFrame.Type.RGB888p)
        frame.setWidth(x.shape[3])
        frame.setHeight(x.shape[2])
        frame.setData(x.reshape(-1))
        self.in_q.send(frame)
        out = self.out_q.get()
        return [np.asarray(out.getTensor(name, dequantize=True), dtype=np.float32) for name in out.getAllLayerNames()]

    def info(self):
        return {"runtime": "depthai", "version": dai.__version__, "shaves": self.shaves}

    def __del__(self):
        if hasattr(self, "pipeline"):
            self.pipeline.stop()
