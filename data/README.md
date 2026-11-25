# Dataset spec

> Capture protocol written in the "Data capture protocol" issue (S0). Labelling starts in S1.

- **Sensor:** OAK-D Pro (RGB + depth kept together), phone as extra
- **Classes (rung 1):** `coke_can`. Rung-2 test set: other cans. Negatives: red objects that aren't cans
- **Splits:** train / val / test / **calibration** (calibration disjoint from test), fixed seed, file lists + checksums committed
- **Pre-processing (part of the spec):** resize method, normalisation, colour order: TODO
- **Labelling tool + conventions:** TODO (box tight on the can, occlusion rule)
- **Storage:** outside git (location TODO), versioned
- **Licence / consent:** our own photos, no faces. Licence of any public data checked before use
