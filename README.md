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

## Test

```bash
python -m unittest discover -s tests -v
```

Tests cover morphology recovery on a synthetic limb-darkened disk and exact
COCO RLE round-trip.

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
```

## Next experiments

- Tune threshold and minimum area by observatory/site.
- Add orientation-aware closing to reconnect thin barbs.
- Train a Flat U-Net on radial-normalized crops.
- Match predicted/true instances with Hungarian Dice for local validation.

No synthetic score is represented as an official leaderboard score.

## License

MIT.
