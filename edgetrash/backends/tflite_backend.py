import ai_edge_litert
import numpy as np
from ai_edge_litert.interpreter import Interpreter


class Backend:
    """LiteRT (ex TFLite) on CPU, XNNPACK delegate by default. Full-integer models get
    their input quantized and outputs dequantized here, so that cost is in infer_ms."""

    dtype = "float32"

    def __init__(self, path, threads: int = 4):
        self.threads = threads
        self.it = Interpreter(model_path=str(path), num_threads=threads)
        self.it.allocate_tensors()
        self.inp = self.it.get_input_details()[0]
        self.outs = sorted(self.it.get_output_details(), key=lambda o: o["name"])
        # litert-torch keeps PyTorch's NCHW input; other converters give NHWC: read it from the model
        self.layout = "nchw" if self.inp["shape"][1] == 3 else "nhwc"

    def infer(self, x):
        if self.inp["dtype"] != np.float32:
            scale, zero = self.inp["quantization"]
            lim = np.iinfo(self.inp["dtype"])
            x = np.clip(np.round(x / scale + zero), lim.min, lim.max).astype(self.inp["dtype"])
        self.it.set_tensor(self.inp["index"], x)
        self.it.invoke()
        outs = []
        for o in self.outs:
            y = self.it.get_tensor(o["index"])
            if o["dtype"] != np.float32:
                scale, zero = o["quantization"]
                y = (y.astype(np.float32) - zero) * scale
            outs.append(y)
        return outs

    def info(self):
        return {
            "runtime": "ai-edge-litert",
            "version": ai_edge_litert.__version__,
            "threads": self.threads,
            "input_dtype": str(self.inp["dtype"]),
            "layout": self.layout,
        }
