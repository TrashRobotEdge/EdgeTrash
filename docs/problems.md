# Problems log

One entry per real problem. The professor grades the **diagnosis and alternatives**, not the absence of problems.

## Template
### YYYY-MM-DD: short title (issue #N)
- **Symptom:** what we saw
- **Diagnosis:** why it happens (evidence)
- **Alternatives:** options considered
- **Chosen:** what we did, because …

### 2026-11-26: VM disk grown, still 1.4 GB free (benchmark kit)
- **Symptom:** after resizing the `.vdi` to 60 GB, `df -h /` still showed 24 GB.
- **Diagnosis:** `lsblk`: the disk was 60 GB but partition `sda3` was still 24.5 GB. Resizing the disk doesn't resize the partition or the filesystem.
- **Alternatives:** GParted live ISO · `growpart` + `resize2fs` from inside the VM.
- **Chosen:** `sudo growpart /dev/sda 3 && sudo resize2fs /dev/sda3`, because it runs on the live system in one line.

### 2026-11-26: export script deleted the EdgeTrash working copy (benchmark kit)
- **Symptom:** during the first TFLite export, the whole `EdgeTrash/` folder (with `.git`) disappeared.
- **Diagnosis:** `models/export.py` cleaned up with `shutil.rmtree(Path(made).parent)`, where `made` was the path Ultralytics returned. For TFLite it returned a *folder*, so `.parent` was the repo root.
- **Alternatives:** check the returned path before deleting · never delete a path we didn't create.
- **Chosen:** the second: every tool runs inside a fresh `tempfile.TemporaryDirectory()`, we copy out exactly one expected file (error if 0 or 2+), and the context manager removes only that temp folder. Recovery: everything committed was on GitHub (re-cloned); uncommitted work was rewritten from the session. Lesson: push often, and never `rmtree` a path computed from someone else's return value.

### 2026-11-26: Ultralytics silently changed our pinned versions (benchmark kit)
- **Symptom:** after the first TFLite export, `ai-edge-litert` was 2.3.0 instead of the pinned 2.4.1, and torchao, jax and litert-torch were installed.
- **Diagnosis:** Ultralytics 8.4 exports TFLite through `litert-torch` and auto-installs missing requirements (`AutoUpdate`), unpinned. litert-torch 0.10.0 needs ai-edge-litert 2.3.0.
- **Alternatives:** let it auto-install · pin what it needs and turn auto-install off.
- **Chosen:** pin `litert-torch==0.10.0` and its dependencies, pin `ai-edge-litert==2.3.0` on **every** device (the Pis run the same LiteRT that built the file), and set `YOLO_AUTOINSTALL=false` in `models/common.py`: a missing package now fails loudly.

### 2026-11-26: ROS leaks into our Python venvs (benchmark kit)
- **Symptom:** `pip` inside `.venv` complained about `launch-ros` needing pyyaml.
- **Diagnosis:** `~/.bashrc` sources ROS, which sets `PYTHONPATH`; a venv doesn't clear it.
- **Chosen:** `setup/setup.sh` runs with `PYTHONPATH=""`. On the robot, ROS stays in its own container (ADR 0001).

### 2026-11-26: Hailo compiler not available yet (benchmark kit, issue #43)
- **Symptom:** we can't build our own `.hef`.
- **Diagnosis:** the Dataflow Compiler is behind a Hailo Developer Zone login (registration in progress). It must be **DFC 3.33.0**: Hailo Model Zoo v2.17's changelog pairs it with HailoRT 4.23.0, the version Raspberry Pi OS Trixie ships. Hailo also recommends ≥ 16 GB RAM for the DFC; the VM has 12 GB (to see if it's enough for a nano model).
- **Alternatives:** wait · Hailo's pre-compiled zoo `.hef` · compile on a teammate's bigger machine.
- **Chosen (for now):** `models/compile_hailo.py --zoo yolov8n` fetches Hailo's Model Zoo v2.17 `yolov8n.hef` for Hailo-8L, so the class-day hello-world works. It is a **vendor reference**, never a result about our models. `compile_hailo.py` (our ONNX → `.hef`) is ready for when the wheel arrives. Not covered by the zoo recipes: YOLOv5nu, SSDLite.

### 2026-11-26: no FP16 TFLite for the YOLO models (benchmark kit)
- **Symptom:** `export(format="tflite", quantize=16)` fails: "quantize=16 (FP16) is not supported; format='litert' supports 8, w8a16, w8a32, 32".
- **Diagnosis:** Ultralytics 8.4 moved TFLite export to litert-torch, which has no FP16 mode. Also: its TFLite input stays **NCHW** (PyTorch layout), so the TFLite backend reads the layout from the model instead of assuming NHWC.
- **Alternatives:** convert YOLO with onnx2tf just for FP16 · skip FP16 on LiteRT for YOLO.
- **Chosen:** skip for now (cell marked "?" in the protocol), because using a second converter for one cell would make precision and converter change together. Revisit if the team wants FP16 on the Pi 5 (Cortex-A76 has FP16 arithmetic, the Pi 4's A72 doesn't).

### 2026-11-26: pruning crashed inside torch-pruning (benchmark kit, issue #28)
- **Symptom:** `IndexError: index 384 is out of bounds for dimension 0 with size 256` in `pruner.step()`.
- **Diagnosis:** Ultralytics' C2f block splits a tensor with `chunk()`; torch-pruning's dependency graph loses track of which channels go where.
- **Alternatives:** unstructured pruning (no speed-up on dense runtimes) · rewrite C2f without `chunk()` (what torch-pruning's YOLOv8 example does).
- **Chosen:** `SplitC2f` in `models/common.py`: the first conv split into two convs. Checked to give exactly the same output as the original model (max abs difference 0.0 on a random 640×640 input) before pruning.

### 2026-11-26: TVM and the TFLite exporter can't share the VM's environment (benchmark kit)
- **Symptom:** `import tvm` failed: `cannot import name 'get_annotations' from 'typing_extensions'` (4.12.2 installed).
- **Diagnosis:** `pip check`: apache-tvm-ffi needs typing_extensions ≥ 4.13, while litert-torch → `ai-edge-litert[model-utils]` → `xdsl==0.28.0` needs < 4.13.
- **First fix, wrong:** pin 4.16.0 because a TFLite export still ran. A clean `setup.sh --config vm-dev` then failed with `ResolutionImpossible`: forcing a version pip refuses isn't reproducible.
- **Alternatives:** a separate `.venv-tvm` on the VM · compile TVM where it runs.
- **Chosen:** TVM only on the Pis (`compile_tvm.py --target pi4|pi5`, run on the Pi), because the `.so` is CPU-specific anyway and the Pis never install litert-torch. Before removing it from the VM, checked the TVM backend once against ONNX Runtime on one image: max abs difference 0.0009 on outputs up to 640 px, same 9 detections.
- **Same cause, second clash:** `onnx 1.22` needs typing_extensions ≥ 4.15 (pip's report: the only other conflict). Pinned `onnx==1.20.1` (needs ≥ 4.7.1), and every model file was re-exported from a venv built by `setup.sh` alone (`bench/dryrun.sh`).

### 2026-11-26: SSDLite wouldn't compile for the OAK (benchmark kit)
- **Symptom:** blobconverter: `Failed to compile layer ".../scale_activation/HardSigmoid": unsupported layer type "HardSigmoid"`.
- **Diagnosis:** MobileNetV3 uses hard-sigmoid / hard-swish. OpenVINO 2022.1's Myriad X plugin (the last version blobconverter offers for the OAK) has no HardSigmoid layer. This is the "operator support" pitfall (Benchmarking p.19), caught at compile time instead of as a silent CPU fallback.
- **Alternatives:** drop SSDLite on the OAK · rewrite the activations with ops the VPU has.
- **Chosen:** at export, `nn.Hardsigmoid` → `relu6(x + 3) / 6` and `nn.Hardswish` → `x · relu6(x + 3) / 6` (`models/ssdlite.py`). Same maths: max abs difference 0.0 on a random input. Used for every runtime, so all SSDLite variants run the same graph.
- Also: onnx2tf validates its conversion with a sample `.npy` from the working directory and otherwise downloads one that numpy refuses to load (pickled). `export.py` writes that file from 20 of our calibration images instead.

### 2026-11-26: YOLOv8n ORT INT8 gave mAP 0.0 (benchmark kit, issue #25)
- **Symptom:** `yolov8n-coco-640-int8-onnx`: mAP@0.5:0.95 = 0.000. FP32: 0.396.
- **Diagnosis:** on one image, INT8 boxes matched FP32 but every class score was exactly 0 (one distinct value). The model's single output concatenates box coordinates (0–640 px) and class scores (0–1); static INT8 gives that concat one scale, so a step is ~2.5 and every score rounds to 0.
- **Alternatives:** per-tensor tricks on the output only · normalise boxes to [0, 1] in the graph (what Ultralytics does for TFLite) · leave the head's decode tail in float.
- **Chosen:** `export.py` excludes the Detect head's non-Conv nodes (DFL, slices, sigmoid, concat: 47 nodes for YOLOv8n) from quantization. Backbone, neck and head convolutions stay INT8. After the fix, max class score 0.939 vs 0.929 FP32 on the same image. Lesson for the report: check INT8 on outputs, not just "it runs" (p.31 "verify outputs on target").

### 2026-11-26: onnx2tf modified our registered ONNX file (benchmark kit)
- **Symptom:** `bench/accuracy.py` refused `ssdlite-coco-320-fp32-onnx`: "checksum differs from manifest".
- **Diagnosis:** the file was rewritten 2 s after the TFLite conversion. onnx2tf calls `onnx.save(..., input_onnx_file_path)`: it saves its shape-estimated graph over its own input. The SSDLite OAK blob had then been compiled from the modified file.
- **Chosen:** onnx2tf now converts a copy inside its temp folder. All SSDLite variants were re-exported from one clean FP32 file. The manifest checksum check did its job: without it, two variants would have silently come from different graphs.

### 2026-11-26: SSDLite accuracy run took > 18 min (benchmark kit)
- **Symptom:** `bench/accuracy.py` on SSDLite ran for 18 min without a progress line (stdout is buffered into the log file, so "no line" didn't mean "stuck").
- **Diagnosis:** profiled one image at the mAP threshold (conf 0.001): 11.7k candidate boxes, 622 ms of 634 ms in OpenCV's `NMSBoxesBatched`, which runs one NMS over all classes at once (cost grows with n²). First guess (too many candidates, add a cap) was wrong: measured 11.7k, below any sensible cap.
- **Chosen:** one `cv2.dnn.NMSBoxes` per class (same semantics). SSDLite post-processing 736 → 104 ms at conf 0.001; YOLOv8n at the operating threshold (0.25) 1.9 ms.

### 2026-11-26: pruned YOLOv8n gives mAP 0.0 without fine-tuning (benchmark kit, issue #28)
- **Symptom:** `yolov8n-coco_p51-640-fp32-onnx` (51 % of the MACs removed): mAP 0.0 on the 500 eval images. Max class score 0.002 vs 0.93 unpruned on the same image.
- **Diagnosis:** not a pruning bug. Group by group, the pruned model matches the unpruned one with those channels zeroed (max score diff 4e-7). But zeroing the channels the pruner picks already kills the model, even at the first step (7 % of the MACs: max score 0.01). Likely cause (not verified): a channel with small weights still passes its BatchNorm offset on, and removing it shifts every layer after it, over 47 groups.
- **Alternatives:** fine-tune after pruning (`prune.py --data`) · prune step by step with a short fine-tune after each · an importance that accounts for BatchNorm.
- **Chosen:** fine-tune after pruning, on our cans dataset (#28). Not feasible on COCO on the VM's CPU. In the dry run the pruned variant only measures latency, and it fails the quality gate as it should.

### 2026-11-26: VM latency is noisy, INT8 is sometimes slower than FP32 (benchmark kit)
- **Symptom:** same variant, rep to rep: SSDLite FP32 ONNX p50 13.7 → 23.5 → 13.7 ms. LiteRT INT8 slower than LiteRT FP32 (YOLOv8n 83 vs 67 ms, SSDLite 71 vs 26 ms); ORT INT8 faster for YOLOv8n (66 vs 104 ms), slower for SSDLite (20 vs 14 ms).
- **Diagnosis:** the VM shares 4 cores with the host and has no fixed governor or cooling, which is why the protocol uses repetitions and IQR. INT8 speed depends on the CPU's integer kernels: not diagnosed on x86, the Pis' ARM cores are what matter.
- **Chosen:** VM numbers check the code only (said in every figure title). Check INT8 vs FP32 on the Pis before drawing any conclusion.

### 2026-11-26: Hailo Model Zoo wouldn't install next to the compiler (benchmark kit, issue #43)
- **Symptom:** `setup.sh --config vm-hailo` failed: "The Dataflow Compiler package 'hailo-dataflow-compiler' was not found", although the wheel was in the same `pip install`. After a fix, `pip` reported numpy 2.2.6 while the DFC pins numpy 1.26.4.
- **Diagnosis:** the zoo's `setup.py` checks the DFC is installed, but pip builds it in an isolated environment that can't see it. Then our own `opencv-python-headless==5.1.0.95` pulled numpy 2.x.
- **Chosen:** install the DFC first, then the zoo with `--no-build-isolation`. No extra OpenCV in `.venv-hailo` (the zoo brings 4.11). `setup.sh` now imports `cv2`, `numpy`, `yaml`, `hailo_sdk_client` at the end and stops if one fails. Checked by deleting `.venv-hailo` and rebuilding it from scratch.
