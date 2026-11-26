"""TVM Relax VM on CPU. Threads: TVM_NUM_THREADS (default = all cores)."""

import os

import tvm
from tvm import relax


class Backend:
    layout, dtype = "nchw", "float32"

    def __init__(self, path):
        self.dev = tvm.cpu()
        self.vm = relax.VirtualMachine(tvm.runtime.load_module(str(path)), self.dev)

    def infer(self, x):
        out = self.vm["main"](tvm.runtime.tensor(x, self.dev))
        return [o.numpy() for o in out] if isinstance(out, (list, tuple, tvm.ir.Array)) else [out.numpy()]

    def info(self):
        return {"runtime": "tvm", "version": tvm.__version__, "threads": os.environ.get("TVM_NUM_THREADS", "all")}
