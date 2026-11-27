import onnxruntime as ort


class Backend:
    layout, dtype = "nchw", "float32"

    def __init__(self, path, threads: int = 4):
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        self.threads = threads
        self.sess = ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])
        self.input = self.sess.get_inputs()[0].name

    def infer(self, x):
        return self.sess.run(None, {self.input: x})

    def info(self):
        return {"runtime": "onnxruntime", "version": ort.__version__, "threads": self.threads}
