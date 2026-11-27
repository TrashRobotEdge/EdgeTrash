# Dataset spec

> Capture protocol written in the "Data capture protocol" issue (S0). Labelling starts in S1.

- **Sensor:** OAK-D Pro (RGB + depth kept together), phone as extra
- **Classes (rung 1):** `coke_can`. Rung-2 test set: other cans. Negatives: red objects that aren't cans
- **Size target:** 300–500 labelled images + negatives (protocol D7)
- **Splits:** train / val / test / **calibration** (calibration disjoint from test), fixed seed, file lists in `data/splits/` (in git)
- **Formats:** Ultralytics YOLO txt for training (`data/cans.yaml`), COCO json for `bench/accuracy.py` (most labelling tools export both)
- **Pre-processing (part of the spec):** letterbox to a square, grey 114 padding, bilinear resize, RGB, [0, 1]: `edgetrash/preprocess.py`
- **Labelling tool + conventions:** TODO (box tight on the can, occlusion rule)
- **Storage:** outside git (location TODO), versioned
- **Licence / consent:** our own photos, no faces. Licence of any public data checked before use

## COCO dry run (until our data exists)
`bash setup/setup.sh --config vm-dev` downloads COCO val2017 into `data/coco/` (ignored) and runs `python data/coco_splits.py`: 500 eval, 200 calibration (disjoint), 100 bench frames (first 100 of eval), seed 0. `data/coco_eval.py make` writes the eval annotation subset (with each image's URL + sha256); on a Pi, `data/coco_eval.py fetch` downloads the images. COCO terms (cocodataset.org): annotations CC BY 4.0, images under the Flickr terms of use. Used only to test the pipeline, images never redistributed (not in git, releases or the USB bundle; the bundle carries only the annotation subset).
