# Solar Filament Segmentation 2026 — Classical Baseline

A fast, interpretable baseline for the
[Solar Filament Segmentation Challenge 2026](https://www.kaggle.com/competitions/filament-segmentation-2026).

## Pipeline

1. Robust radial normalization removes center-to-limb brightness variation.
2. A disk-relative dark quantile selects candidate filament material.
3. Four-connected components preserve separate filament instances.
4. Minimum-area filtering suppresses noise and fragmentation.
5. `pycocotools` writes the required compressed COCO RLE counts.

The initial real-data defaults use the darkest 25% of the radially normalized
solar disk, cap normalized intensity at 0.95, and retain up to 64 largest
components. They are baseline parameters, not leaderboard-tuned values.

This is deliberately a classical baseline: it is CPU-friendly, exposes failure
modes, and can generate pseudo-labels or a post-processing prior for a U-Net.

An experimental, GPU-oriented torchvision Mask R-CNN pipeline is provided in
[`kaggle/train_maskrcnn.py`](kaggle/train_maskrcnn.py), with run instructions in
[`kaggle/README.md`](kaggle/README.md). It collapses chirality categories into one filament
class, selects one deterministic annotation record per duplicated physical
image, and uses a stable year/observatory-stratified validation split. This
avoids introducing the AGPL-licensed Ultralytics runtime into this MIT project.

## Test

```bash
python -m unittest discover -s tests -v
```

Tests cover morphology recovery on a synthetic limb-darkened disk and exact
COCO RLE round-trip.

The pinned [`requirements.txt`](requirements.txt) is the complete, minimal
environment for the selected CPU classical pipeline. The non-selected GPU
experiments use the separately pinned
[`requirements-experiments.txt`](requirements-experiments.txt), so reproducing
the final 0.48 route does not install an unused multi-gigabyte deep-learning
stack. GPU notebooks additionally record OS, CUDA, GPU, PyTorch and torchvision
versions in their output.

Run a real MAGFiLO validation subset:

```bash
python -m solarfil.evaluate data/official/MAGFiLO_1.0_Kaggle_2026 --limit 20
```

The report includes a prediction/truth count ratio, unmatched instances, and
one-to-many / many-to-one overlap diagnostics. These expose over-segmentation
that a matched-Dice-only summary can hide; they are not claimed to reproduce
the competition's hidden fragmentation analysis.

Generate the Kaggle submission:

```bash
python -m solarfil.infer \
  data/official/MAGFiLO_1.0_Kaggle_2026/test/test_images \
  outputs/submission.csv

python scripts/audit_submission.py \
  outputs/submission.csv \
  data/official/MAGFiLO_1.0_Kaggle_2026/test/test_images
```

The final route has been executed over all 180 available official test images.
Its deterministic output and full structural audit receipt are recorded in
[`FINAL_CLASSICAL_EXECUTION.md`](FINAL_CLASSICAL_EXECUTION.md).
The final notebook accepts `SOLAR_IMAGE_DIR` and `SOLAR_OUTPUT` for local
verification while retaining its default Kaggle paths; `SOLAR_SKIP_INSTALL=1`
may be used only when the exact pinned dependencies are already installed.

An alternative instance-count calibration is documented in
[`PQ_CALIBRATION.md`](PQ_CALIBRATION.md). On a disjoint 10-image official-label holdout it more than
doubled local Panoptic Quality while sharply reducing false positives, and its separate 180-image
CSV passed the full structural audit. It remains a **candidate**, not the selected final route,
until a Kaggle submission verifies that the official platform score exceeds `0.48`.

For the final competition handoff, use the self-contained
[`kaggle/final_classical_pipeline.ipynb`](kaggle/final_classical_pipeline.ipynb).
The report source in the organizer's `acmart/sigconf` format and the
requirement-by-requirement release gates are in
[`report/main.tex`](report/main.tex) and
[`FINAL_SUBMISSION_CHECKLIST.md`](FINAL_SUBMISSION_CHECKLIST.md).
The compiled artifact is [`report/solar-filament-report.pdf`](report/solar-filament-report.pdf).
With Tectonic and Poppler installed, reproduce its release checks with:

```bash
bash scripts/verify_report.sh
```

The check requires exactly four pages, embedded fonts, and no overfull boxes or
undefined references; CI runs the same gate from a pinned Tectonic archive.

## Next experiments

- Tune threshold and minimum area by observatory/site.
- Add orientation-aware closing to reconnect thin barbs.
- Train and threshold-sweep the torchvision Mask R-CNN on a Kaggle GPU.
- Compare Mask R-CNN with a semantic U-Net plus watershed instance recovery.
- Compare the audited PQ-calibrated classical candidate on the official leaderboard without
  replacing the preserved `0.48` route unless the platform verifies an improvement.

No synthetic score is represented as an official leaderboard score.

## License

MIT.
