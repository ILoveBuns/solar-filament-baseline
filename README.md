# Solar Filament Segmentation 2026 — Classical Baseline

A fast, interpretable baseline for the
[Solar Filament Segmentation Challenge 2026](https://www.kaggle.com/competitions/filament-segmentation-2026).

## Pipeline

1. Robust radial normalization removes center-to-limb brightness variation.
2. A disk-relative dark quantile selects candidate filament material.
3. Four-connected components preserve separate filament instances.
4. Minimum-area filtering suppresses noise and fragmentation.
5. `pycocotools` writes the required compressed COCO RLE counts.

This is deliberately a classical baseline: it is CPU-friendly, exposes failure
modes, and can generate pseudo-labels or a post-processing prior for a U-Net.

## Test

```bash
python -m unittest discover -s tests -v
```

Tests cover morphology recovery on a synthetic limb-darkened disk and exact
COCO RLE round-trip.

## Next experiments

- Tune threshold and minimum area by observatory/site.
- Add orientation-aware closing to reconnect thin barbs.
- Train a Flat U-Net on radial-normalized crops.
- Match predicted/true instances with Hungarian Dice for local validation.

No synthetic score is represented as an official leaderboard score.

## License

MIT.

