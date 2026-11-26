"""Power sources, aligned with the latency log on time.time() (Benchmarking p.26-27, p.30).

- Pi 5 PMIC (point D): `vcgencmd pmic_read_adc`, sum of V x I over the rails it reports.
  Covers the board's own rails only: not USB devices (OAK) and not the full 5 V input.
- Inline USB-C meter (point B, the reference): its model and export format are still open
  (issue #6). Until then: export its log to CSV with columns `t_unix,watts` and pass it to
  analyze.py with --meter. If it has no log, write down readings by hand in the same format.
"""

import re
import shutil
import subprocess
from pathlib import Path

import pandas as pd

RAIL = re.compile(r"^\s*(\S+)_([AV]) \w+\(\d+\)=([\d.]+)[AV]\s*$", re.M)


def has_pmic() -> bool:
    if not shutil.which("vcgencmd"):
        return False
    out = subprocess.run(["vcgencmd", "pmic_read_adc"], capture_output=True, text=True).stdout
    return bool(RAIL.search(out))


def pmic_watts() -> float:
    amps, volts = {}, {}
    out = subprocess.run(["vcgencmd", "pmic_read_adc"], capture_output=True, text=True).stdout
    for line in out.splitlines():
        m = RAIL.match(line)
        if m:
            (amps if m[2] == "A" else volts)[m[1]] = float(m[3])
    return sum(amps[r] * volts[r] for r in amps if r in volts)


def load_meter_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = {"t_unix", "watts"} - set(df.columns)
    if missing:
        raise ValueError(f"{path}: meter CSV needs columns t_unix,watts (missing {missing})")
    return df[["t_unix", "watts"]]
