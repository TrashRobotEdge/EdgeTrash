"""COCO eval images + annotations for a device, without shipping COCO images (Flickr terms: we don't redistribute).

    python3 data/coco_eval.py make                 # VM: needs data/coco/ (setup.sh --config vm-dev)
    python3 data/coco_eval.py fetch                # Pi or laptop: downloads the images, checks their sha256
    python3 data/coco_eval.py fetch --root <dir>   # offline Pi: fill <dir> on a USB stick, then cp -r <dir>/. ~/EdgeTrash/

make writes data/coco/annotations/coco-eval-subset.json: the images of data/splits/coco-eval.txt
(bench frames are a subset of it) with their annotations, URL and sha256. setup/bundle.sh ships
that file; fetch reads it. Standard library only: runs on a Pi before any venv exists.
"""

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SPLIT = REPO / "data/splits/coco-eval.txt"
FULL = REPO / "data/coco/annotations/instances_val2017.json"
SUBSET = REPO / "data/coco/annotations/coco-eval-subset.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make():
    files = [line.strip() for line in SPLIT.read_text().splitlines() if line.strip()]
    by_name = {Path(f).name: f for f in files}
    full = json.loads(FULL.read_text(encoding="utf-8"))
    images = [im for im in full["images"] if im["file_name"] in by_name]
    for im in images:
        im["path"] = by_name[im["file_name"]]
        im["sha256"] = sha256(REPO / im["path"])
    ids = {im["id"] for im in images}
    # segmentation dropped: bbox mAP only needs bbox, area and iscrowd
    keep = ("id", "image_id", "category_id", "bbox", "area", "iscrowd")
    anns = [{k: a[k] for k in keep} for a in full["annotations"] if a["image_id"] in ids]
    subset = {"info": full["info"], "licenses": full["licenses"], "images": images, "annotations": anns, "categories": full["categories"]}
    SUBSET.write_text(json.dumps(subset), encoding="utf-8")
    print(f"{SUBSET.relative_to(REPO)}: {len(images)} images, {len(anns)} annotations")


def fetch(root: Path):
    images = json.loads(SUBSET.read_text(encoding="utf-8"))["images"]
    bad = []
    for i, im in enumerate(images):
        dest = root / im["path"]
        if not (dest.exists() and sha256(dest) == im["sha256"]):
            dest.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(im["coco_url"], dest)
            if sha256(dest) != im["sha256"]:
                bad.append(im["path"])
        if (i + 1) % 100 == 0:
            print(f"{i + 1}/{len(images)}")
    if bad:
        raise SystemExit(f"{len(bad)} images differ from the subset's sha256, e.g. {bad[0]}: delete them and fetch again")
    print(f"[ok] {len(images)} COCO eval images in {root / 'data/coco/val2017'}, checksums match")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["make", "fetch"])
    ap.add_argument("--root", type=Path, default=REPO, help="fetch: folder the image paths are relative to")
    args = ap.parse_args()
    if args.step == "make":
        make()
    elif not SUBSET.exists():
        raise SystemExit(f"{SUBSET} missing: on the VM run python3 data/coco_eval.py make; on a Pi it comes with the bundle")
    else:
        fetch(args.root)


if __name__ == "__main__":
    main()
