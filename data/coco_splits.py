"""Fixed COCO val2017 subsets for the pipeline dry run (Benchmarking p.13: fixed subset,
seed, checksums; calibration disjoint from test).

    python data/coco_splits.py   # needs data/coco/val2017 + annotations (see data/README.md)

Writes data/splits/coco-{eval,calib,bench}.txt. Lists are in git, images are not.
"""

import random
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SEED, N_EVAL, N_CALIB, N_BENCH = 0, 500, 200, 100


def main():
    images = sorted((REPO / "data/coco/val2017").glob("*.jpg"))
    random.Random(SEED).shuffle(images)
    rel = [str(p.relative_to(REPO)) for p in images]
    splits = {"eval": rel[:N_EVAL], "calib": rel[N_EVAL : N_EVAL + N_CALIB], "bench": rel[:N_BENCH]}
    out = REPO / "data/splits"
    out.mkdir(exist_ok=True)
    for name, files in splits.items():
        (out / f"coco-{name}.txt").write_text("\n".join(files) + "\n")
        print(f"coco-{name}.txt: {len(files)} images")


if __name__ == "__main__":
    main()
