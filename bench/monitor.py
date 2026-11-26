"""Background system sampler: CPU load, memory, temperature, clock, throttling, PMIC power
(Benchmarking p.16, p.28). One row every `period` seconds, timestamped with time.time()."""

import os
import shutil
import subprocess
import threading
import time
from pathlib import Path

import psutil

from power import has_pmic, pmic_watts

TEMP = Path("/sys/class/thermal/thermal_zone0/temp")
FREQ = Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq")


def read_num(path: Path, scale: float) -> float | None:
    try:
        return int(path.read_text()) / scale
    except (OSError, ValueError):
        return None


def throttled() -> str | None:
    """Pi only. Non-zero = under-voltage or throttling happened (bit meanings: vcgencmd docs)."""
    if not shutil.which("vcgencmd"):
        return None
    return subprocess.run(["vcgencmd", "get_throttled"], capture_output=True, text=True).stdout.strip().split("=")[-1]


class Monitor(threading.Thread):
    def __init__(self, period: float = 0.5):
        super().__init__(daemon=True)
        self.period, self.rows, self.stop_flag = period, [], threading.Event()
        self.proc = psutil.Process(os.getpid())
        self.pmic = has_pmic()
        psutil.cpu_percent(percpu=True)  # first call primes the counters

    def run(self):
        while not self.stop_flag.is_set():
            cores = psutil.cpu_percent(percpu=True)
            row = {
                "t_unix": time.time(),
                "cpu_pct": sum(cores) / len(cores),
                "cpu_max_core_pct": max(cores),
                "rss_mb": self.proc.memory_info().rss / 1e6,
                "mem_used_mb": psutil.virtual_memory().used / 1e6,
                "temp_c": read_num(TEMP, 1000),
                "freq_mhz": read_num(FREQ, 1000),
                "throttled": throttled(),
            }
            if self.pmic:
                row["pmic_w"] = pmic_watts()
            self.rows.append(row)
            self.stop_flag.wait(self.period)

    def stop(self):
        self.stop_flag.set()
        self.join()
        return self.rows
