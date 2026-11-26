"""What every results file records, so each number traces back to a commit and a command."""

import platform
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RAW = REPO / "results" / "raw"


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True).stdout.strip()


def read(path: str) -> str | None:
    try:
        return Path(path).read_text().strip("\x00 \n")
    except OSError:
        return None


def os_name() -> str:
    text = read("/etc/os-release") or ""
    for line in text.splitlines():
        if line.startswith("PRETTY_NAME="):
            return line.split("=", 1)[1].strip('"')
    return platform.platform()


def run_meta() -> dict:
    return {
        "commit": git("rev-parse", "HEAD"),
        "dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
        "command": "python " + shlex.join(sys.argv),
        "utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "host": platform.node(),
        "machine": platform.machine(),
        "os": os_name(),
        "kernel": platform.release(),
        "python": platform.python_version(),
        "model": read("/proc/device-tree/model"),  # "Raspberry Pi 5 Model B ..." on a Pi
        "governor": read("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"),
    }
