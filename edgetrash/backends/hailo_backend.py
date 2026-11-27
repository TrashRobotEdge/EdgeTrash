"""Hailo-8L through HailoRT 4.23 (apt package python3-hailort on Raspberry Pi OS).
The .hef files carry Hailo's NMS (models/compile_hailo.py), so the output is per-class boxes:
head "hailo_nms". That NMS runs in HailoRT on the Pi's CPU (zoo recipe engine=cpu), inside
infer(): infer_ms = chip + host NMS. Scores under the .hef's nms_scores_th (0.2) are dropped."""

import hailo_platform as hp


class Backend:
    layout, dtype = "nhwc", "uint8"

    def __init__(self, path):
        self.hef = hp.HEF(str(path))
        self.device = hp.VDevice()
        params = hp.ConfigureParams.create_from_hef(self.hef, interface=hp.HailoStreamInterface.PCIe)
        self.group = self.device.configure(self.hef, params)[0]
        in_params = hp.InputVStreamParams.make(self.group, format_type=hp.FormatType.UINT8)
        out_params = hp.OutputVStreamParams.make(self.group, format_type=hp.FormatType.FLOAT32)
        self.in_name = self.hef.get_input_vstream_infos()[0].name
        self.out_names = [i.name for i in self.hef.get_output_vstream_infos()]
        # Kept open for the whole run: activation + vstreams cost far more than one inference.
        self._active = self.group.activate(self.group.create_params())
        self._active.__enter__()
        self._pipe = hp.InferVStreams(self.group, in_params, out_params)
        self._pipe.__enter__()

    def infer(self, x):
        res = self._pipe.infer({self.in_name: x})
        return [res[n][0] for n in self.out_names]  # batch of 1

    def info(self):
        return {"runtime": "hailort", "version": hp.__version__, "outputs": self.out_names}

    def __del__(self):
        for ctx in ("_pipe", "_active"):
            if hasattr(self, ctx):
                getattr(self, ctx).__exit__(None, None, None)
