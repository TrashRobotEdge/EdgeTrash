#!/usr/bin/env bash
# One setup script for every device. Asks which config, installs exactly that, checks it,
# then starts the benchmark UI. Safe to re-run: it only adds what is missing.
#
#   bash setup/setup.sh                          # menu
#   bash setup/setup.sh --config pi5-hailo       # no menu
#   bash setup/setup.sh --config vm-hailo --dfc ~/Downloads/hailo_dataflow_compiler-3.33.0-py3-none-linux_x86_64.whl
#   options: --no-ui (don't start the UI) · --yes (no questions, sudo must not ask a password)
#
# Versions are pinned in pyproject.toml. What each config does: setup/README.md.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIGS=(pi4-cpu pi5-cpu pi5-hailo oak vm-dev vm-hailo)
TORCH_INDEX="https://download.pytorch.org/whl/cpu"
HAILO_ZOO="git+https://github.com/hailo-ai/hailo_model_zoo.git@v2.17"  # pairs with DFC 3.33.0 / HailoRT 4.23.0
CONFIG="" UI=1 YES=0 DFC=""
export PYTHONPATH=""            # a sourced ROS setup leaks its packages into our venv otherwise
export PYTHONIOENCODING=utf-8

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) CONFIG="$2"; shift 2 ;;
    --dfc) DFC="$2"; shift 2 ;;
    --no-ui) UI=0; shift ;;
    --yes) YES=1; shift ;;
    -h|--help) sed -n '2,10p' "$0"; exit 0 ;;
    *) echo "unknown option $1"; exit 2 ;;
  esac
done

say()  { printf '\n\033[1m== %s\033[0m\n' "$*"; }
ok()   { printf '  [ok] %s\n' "$*"; }
warn() { printf '  \033[33m[!!] %s\033[0m\n' "$*"; }
die()  { printf '  \033[31m[XX] %s\033[0m\n' "$*"; exit 1; }
SUDO="sudo"; [[ $YES == 1 ]] && SUDO="sudo -n"

pick_config() {
  [[ -n "$CONFIG" ]] && return
  echo "Which config is this machine? (details: setup/README.md)"
  select CONFIG in "${CONFIGS[@]}"; do [[ -n "$CONFIG" ]] && break; done
}

board() { tr -d '\0' < /proc/device-tree/model 2>/dev/null || echo "not a Raspberry Pi"; }

check_board() {  # $1 = "4" or "5": refuse to set up a Pi 5 config on a Pi 4 by mistake
  board | grep -q "Raspberry Pi $1" || die "this config is for a Raspberry Pi $1, this machine is: $(board)"
  ok "board: $(board)"
  grep -q 'VERSION_CODENAME=trixie' /etc/os-release || warn "not Raspberry Pi OS Trixie (protocol: same OS on every Pi, docs/decisions/0001)"
}

apt_install() {
  local missing=()
  for p in "$@"; do dpkg -s "$p" &>/dev/null || missing+=("$p"); done
  [[ ${#missing[@]} -eq 0 ]] && { ok "apt: $* already there"; return; }
  $SUDO apt-get update -qq && $SUDO apt-get install -y -qq "${missing[@]}"
  ok "apt: installed ${missing[*]}"
}

need_venv() {  # the venv module, whatever the distro calls its package (python3-venv, python3.10-venv)
  "${1:-python3}" -c "import ensurepip" 2>/dev/null && { ok "python venv module: there"; return; }
  apt_install "$(basename "${1:-python3}")-venv"
}

make_venv() {  # $1 = venv dir, $2 = extra venv flags, $3 = python
  [[ -x "$1/bin/python" ]] || "${3:-python3}" -m venv ${2:-} "$1"
  "$1/bin/pip" install -q --upgrade "pip==25.3"
  ok "venv: $1 ($("$1/bin/python" --version))"
}

pip_extras() {  # $1 = extras, rest = extra pip args
  local extras="$1"; shift
  (cd "$REPO" && .venv/bin/pip install -q -e ".[$extras]" "$@")
  ok "pip: edgetrash[$extras]"
}

performance_governor() {  # lasts until reboot; run.py warns if it is not set (Benchmarking p.28)
  local f
  for f in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do echo performance | $SUDO tee "$f" >/dev/null; done
  ok "CPU governor: $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor) (until reboot: re-run this script after a reboot)"
}

versions() {
  say "Versions (also written into every results meta.json)"
  echo "  $(grep PRETTY_NAME /etc/os-release | cut -d= -f2) · kernel $(uname -r) · $(board)"
  (cd "$REPO" && .venv/bin/pip list 2>/dev/null | grep -i -E '^(numpy|opencv-python-headless|onnxruntime|ai-edge-litert|depthai|apache-tvm|streamlit|torch|ultralytics|tensorflow) ' | sed 's/^/  /') || true
}

smoke() {
  say "Smoke test: a few frames of every variant this config can run"
  (cd "$REPO" && .venv/bin/python bench/smoke.py --config "bench/configs/$CONFIG.yaml")
}

start_ui() {
  [[ $UI == 1 ]] || return 0
  say "Benchmark UI: open http://$(hostname -I | awk '{print $1}'):8501 from your laptop (Ctrl-C to stop)"
  cd "$REPO" && exec .venv/bin/streamlit run bench/ui.py --server.address 0.0.0.0 --server.port 8501 --server.headless true
}

pi_cpu() {
  check_board "$1"
  need_venv
  apt_install python3-dev
  # Pi 5: pi5-cpu and pi5-hailo share this venv, and Hailo's Python binding only comes from apt
  make_venv "$REPO/.venv" "$([[ $1 == 5 ]] && echo --system-site-packages)"
  pip_extras "onnx,tflite,tvm,bench"
  performance_governor
}

pi5_hailo() {
  check_board 5
  need_venv
  apt_install python3-dev dkms hailo-all
  hailortcli fw-control identify &>/dev/null || die "Hailo not answering. First install needs a reboot: sudo reboot, then re-run this script"
  ok "hailo: $(hailortcli fw-control identify | grep -i -E 'firmware version|device architecture' | tr -s ' ' | paste -sd' ')"
  echo "  PCIe link (recorded, not changed: docs/protocol.md step 5): $($SUDO lspci -vv 2>/dev/null | grep -A40 -i hailo | grep -m1 LnkSta: | xargs || echo unknown)"
  make_venv "$REPO/.venv" --system-site-packages  # sees the apt python3-hailort binding
  grep -q "include-system-site-packages = true" "$REPO/.venv/pyvenv.cfg" ||
    die ".venv was made without system packages: rm -rf $REPO/.venv and re-run"
  pip_extras "onnx,tflite,bench"
  "$REPO/.venv/bin/python" -c "import hailo_platform" || die "hailo_platform not importable from the venv"
  performance_governor
}

oak() {
  need_venv
  apt_install python3-dev usbutils
  local rule=/etc/udev/rules.d/80-movidius.rules
  if [[ ! -f $rule ]]; then  # DepthAI needs USB access without root
    echo 'SUBSYSTEM=="usb", ATTRS{idVendor}=="03e7", MODE="0666"' | $SUDO tee $rule >/dev/null
    $SUDO udevadm control --reload-rules && $SUDO udevadm trigger
    ok "udev rule for the OAK written: unplug and replug the camera"
  fi
  make_venv "$REPO/.venv"
  pip_extras "oak,onnx,bench"
  "$REPO/.venv/bin/python" -c "import depthai as d; n=len(d.Device.getAllAvailableDevices()); print(f'  [ok] OAK devices found: {n}') if n else print('  [!!] no OAK found: plug it into a blue USB 3 port')"
  echo "  USB tree (OAK must show 5000M = USB 3, Benchmarking p.6):"; lsusb -t | sed 's/^/    /'
  if [[ -d /sys/devices/system/cpu/cpu0/cpufreq ]]; then performance_governor; fi
}

vm_dev() {
  need_venv
  apt_install python3-dev
  make_venv "$REPO/.venv"
  pip_extras "models,onnx,tflite,tvm,oak,bench,dev" --extra-index-url "$TORCH_INDEX"
  (cd "$REPO" && .venv/bin/pre-commit install >/dev/null && ok "pre-commit hook installed")
  if [[ ! -d "$REPO/data/coco/val2017" ]]; then
    local yn=y; [[ $YES == 1 ]] || read -r -p "  Download COCO val2017 (1 GB) for the dry run? [y/N] " yn
    if [[ $yn == [yY]* ]]; then
      mkdir -p "$REPO/data/coco" && cd "$REPO/data/coco"
      curl -fL --retry 5 -O http://images.cocodataset.org/zips/val2017.zip
      curl -fL --retry 5 -O http://images.cocodataset.org/annotations/annotations_trainval2017.zip
      unzip -q val2017.zip && unzip -q annotations_trainval2017.zip annotations/instances_val2017.json
      rm val2017.zip annotations_trainval2017.zip && cd "$REPO"
    fi
  fi
  if [[ -d "$REPO/data/coco/val2017" ]]; then
    (cd "$REPO" && .venv/bin/python data/coco_splits.py >/dev/null && ok "COCO splits written")
  fi
}

vm_hailo() {  # compiler only: no UI, no smoke test (nothing runs without the chip)
  [[ -f "$DFC" ]] || die "give the Dataflow Compiler wheel: --dfc <path to hailo_dataflow_compiler-3.33.0-...whl> (how to get it: setup/README.md)"
  need_venv python3.10
  apt_install python3.10-dev python3-tk graphviz libgraphviz-dev
  make_venv "$REPO/.venv-hailo" "" python3.10
  "$REPO/.venv-hailo/bin/pip" install -q "$DFC" "$HAILO_ZOO"
  (cd "$REPO" && .venv-hailo/bin/pip install -q --no-deps -e . && .venv-hailo/bin/pip install -q "pyyaml==6.0.4" "opencv-python-headless==5.1.0.95")
  ok "$("$REPO/.venv-hailo/bin/pip" list 2>/dev/null | grep -i -E '^hailo' | tr -s ' ' | paste -sd',')"
  UI=0
}

pick_config
say "Config: $CONFIG"
case "$CONFIG" in
  pi4-cpu)   pi_cpu 4 ;;
  pi5-cpu)   pi_cpu 5 ;;
  pi5-hailo) pi5_hailo ;;
  oak)       oak ;;
  vm-dev)    vm_dev ;;
  vm-hailo)  vm_hailo; exit 0 ;;
  *) die "unknown config $CONFIG (one of: ${CONFIGS[*]})" ;;
esac
versions
smoke || warn "smoke test failed: see above, then docs/problems.md"
start_ui
