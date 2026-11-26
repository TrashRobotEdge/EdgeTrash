# Setup: one script, every device

`bash setup/setup.sh` asks which config the machine is, installs exactly that (pinned versions), sets the CPU governor, prints versions, runs a smoke test and starts the benchmark UI. Re-run it any time: it only adds what's missing.

| Config | Machine | Installs | Checks |
|---|---|---|---|
| `pi4-cpu` | Pi 4 | ONNX Runtime, LiteRT, TVM | board is a Pi 4, Trixie |
| `pi5-cpu` | Pi 5 | same | board is a Pi 5, Trixie |
| `pi5-hailo` | Pi 5 + Hailo-8L | `hailo-all` (apt, HailoRT 4.23.0), venv sees its Python binding | `hailortcli fw-control identify`, PCIe link |
| `oak` | any Pi with the OAK-D Pro | DepthAI 3.12.2 + udev rule | OAK found, USB 3 (5000M) |
| `vm-dev` | VM / laptop | training + export + compile tools (not TVM: compiled on the Pis), pre-commit | asks to download COCO val2017 (1 GB) for the dry run |
| `vm-hailo` | VM / laptop | Hailo Dataflow Compiler 3.33.0 + Model Zoo v2.17 in `.venv-hailo` | needs the DFC wheel (below) |

## Class day runbook (Pi)
1. **Flash** with Raspberry Pi Imager: *Raspberry Pi OS (64-bit)*, Trixie ([ADR 0001](../docs/decisions/0001-one-os-raspberry-pi-os.md)). In *OS customisation*: hostname (`pi4-trash`, `pi5-trash`), user + password, **Wi-Fi = the phone hotspot** we use in class, **enable SSH**.
2. **On the laptop (VM):** `bash setup/bundle.sh` → `edgetrash-bundle.tar` (repo + model files + bench frames, checksummed). Copy it to a USB stick.
   - A USB cable between Pi and laptop is *not* a file link out of the box (on the Pi 5 that port is power). Use a stick, or `scp edgetrash-bundle.tar <user>@<pi>.local:~` over the hotspot.
3. **On the Pi** (keyboard or `ssh <user>@<pi>.local`):
   ```bash
   tar xf /media/$USER/<stick>/edgetrash-bundle.tar -C ~     # or ~/edgetrash-bundle.tar if you used scp
   bash ~/edgetrash-bundle/install.sh                         # clones ~/EdgeTrash, copies models + frames, runs setup.sh
   ```
   Pick the config. `pi5-hailo` the first time: it installs the driver, then asks for `sudo reboot`. Reboot and run `bash ~/EdgeTrash/setup/setup.sh` again.
4. **Open the UI** from the laptop: the script prints `http://<pi-ip>:8501`. Pick config → *Smoke test* → *Idle baseline* → *Run benchmark* → *Analyze*.
5. **Bring results back:** `scp -r <user>@<pi>.local:~/EdgeTrash/results/raw/<device> results/raw/` on the laptop, then commit them.

**Success = the smoke test prints a p50 for every variant that has a model file.**

Same commands without the UI: `python bench/smoke.py|run.py|sweep.py|analyze.py --config bench/configs/<config>.yaml` (see [bench/README.md](../bench/README.md)).

## Hailo Dataflow Compiler (to build our own `.hef`)
Only needed on the machine that compiles (VM/laptop), never on the Pi. It must be **DFC 3.33.0**: Hailo Model Zoo v2.17 pairs it with HailoRT 4.23.0, which is what Raspberry Pi OS Trixie ships. A `.hef` from a newer DFC may not run on an older HailoRT.
1. **Account:** https://hailo.ai/developer-zone/request-access/ (the "Sign up" link on https://hailo.ai/developer-zone/). Use your university email, Université Côte d'Azur, education/student project. Confirm the email. Download access may not be instant.
2. **Log in:** https://hailo.ai/authorization/
3. **Download:** https://hailo.ai/developer-zone/software-downloads/ → filters *Device: Hailo-8/8L · Software Package: Dataflow Compiler · Architecture: x86 · OS: Linux · Python: 3.10*. The list shows only the latest (3.34): top right **"Filter by:" → "Archive"**, then pick **3.33.0**. Keep the exact file name.
4. **Install:** `bash setup/setup.sh --config vm-hailo --dfc ~/Downloads/<the wheel>.whl`
5. **Compile:** `.venv-hailo/bin/python models/compile_hailo.py --variant yolov8n-coco-640-fp32-onnx --calib data/splits/coco-calib.txt`

If 3.33.0 isn't listed even in *Archive*, the account may lack access: ask on community.hailo.ai. **Don't take 3.34.** Until then, `python models/compile_hailo.py --zoo yolov8n` gets Hailo's pre-compiled reference `.hef` (a vendor reference, not our model).

## When it fails
Write it in [docs/problems.md](../docs/problems.md) (symptom → diagnosis → alternatives). Known ones:
- `sudo` asks a password from the UI's *Install* button → run `bash setup/setup.sh` in a terminal instead.
- Hailo not answering after the first install → reboot.
- OAK not found → replug after the udev rule, use a blue USB 3 port.
- Governor back to `ondemand` after a reboot → re-run the script (`bench/run.py` warns).
