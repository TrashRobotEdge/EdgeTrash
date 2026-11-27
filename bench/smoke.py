"""Setup check: run a few frames of every variant this device config can run, print p50.

    python bench/smoke.py --config bench/configs/pi5-hailo.yaml

Exit code 1 if any variant with a model file fails. Used at the end of setup/setup.sh.
"""

import argparse
import statistics
import sys
import traceback
from pathlib import Path

from edgetrash import load_variant
from run import load_config, load_frames
from sweep import available


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--frames", type=int, default=20)
    args = ap.parse_args()
    cfg = load_config(args.config)
    frames = load_frames({**cfg, "n_frames": args.frames})
    ok, missing = available(cfg["variants"])
    failed = []
    for v in ok:
        try:
            det = load_variant(v)
            times = [det(f)[1] for f in frames]
            total = [t["pre_ms"] + t["infer_ms"] + t["post_ms"] for t in times[5:]]
            print(f"[ok]   {v:42} p50 {statistics.median(total):7.1f} ms")
        except Exception:
            failed.append(v)
            print(f"[FAIL] {v}\n{traceback.format_exc(limit=3)}")
    for v in missing:
        print(f"[--]   {v:42} no model file: VM -> bash bench/dryrun.sh models, Pi -> USB bundle (setup/README.md)")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
