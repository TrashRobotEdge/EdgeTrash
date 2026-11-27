"""All variants of one device config x N repetitions (Benchmarking p.28, p.31).

    python bench/sweep.py --config bench/configs/pi5-cpu.yaml

Idle baseline first, then each repetition runs every variant in a new random order
(interleaved, seed = rep), each in a fresh process, with a cool-down in between.
Variants whose model file is missing are skipped and listed at the end.
"""

import argparse
import random
import subprocess
import sys
import time
from pathlib import Path

from edgetrash.detector import MODEL_DIR, read_manifest
from run import load_config

HERE = Path(__file__).resolve().parent


def available(variants: list[str]) -> tuple[list[str], list[str]]:
    manifest = read_manifest()
    ok = [v for v in variants if v in manifest and (not manifest[v].get("file") or (MODEL_DIR / manifest[v]["file"]).exists())]
    return ok, [v for v in variants if v not in ok]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--reps", type=int, help="override the config's reps")
    ap.add_argument("--no-idle", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    variants, missing = available(cfg["variants"])
    run = [sys.executable, str(HERE / "run.py"), "--config", str(args.config)]
    if not args.no_idle:
        subprocess.run([*run, "--idle"], check=True)
    for rep in range(args.reps or cfg.get("reps", 5)):
        order = variants[:]
        random.Random(rep).shuffle(order)
        for v in order:
            time.sleep(cfg.get("cooldown_s", 10))
            subprocess.run([*run, "--variant", v, "--rep", str(rep)], check=True)
    if missing:
        print("skipped (no model file / not in manifest):", ", ".join(missing))


if __name__ == "__main__":
    main()
