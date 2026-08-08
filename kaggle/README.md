# Kaggle GPU training

This directory contains the experimental torchvision Mask R-CNN route. It is
kept separate from the CPU classical baseline because training requires a GPU
and PyTorch/torchvision.

## Why this route

- Native instance masks, so the model does not need connected components to
  invent filament identities.
- COCO-pretrained Mask R-CNN is available from torchvision's documented weight
  registry.
- PyTorch and torchvision use BSD-style licensing; this avoids adding the
  AGPL-licensed Ultralytics runtime to this MIT repository.
- Predictions are capped at 32 per image and validation reports the
  prediction/truth ratio alongside matched Dice.
- After training, validation predictions are cached once and a joint grid over
  confidence, mask and minimum-area thresholds is reported. The best operating
  point is stored in the checkpoint and automatically reused for prediction
  instead of assuming the defaults are optimal.
- Calibration ranks operating points with Panoptic Quality, using IoU > 0.5
  one-to-one matches and the official TP + 0.5 FP + 0.5 FN denominator. This
  follows the competition's August 7 metric update and directly penalizes
  fragmented duplicate predictions. Matched Dice and the prediction/truth ratio
  remain in the report as diagnostics.

The pretrained weights were trained on COCO. Their public provenance and use
must be disclosed in the final technical report.

## Kaggle notebook commands

Enable a GPU and Internet in a private Kaggle notebook, then run:

```bash
git clone https://github.com/ILoveBuns/solar-filament-baseline.git
cd solar-filament-baseline
python -m pip install -q -e .

ROOT=/kaggle/input/competitions/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026
python kaggle/train_maskrcnn.py train "$ROOT" \
  --epochs 12 \
  --checkpoint /kaggle/working/maskrcnn-best.pt

python kaggle/train_maskrcnn.py predict "$ROOT" \
  --checkpoint /kaggle/working/maskrcnn-best.pt \
  --output /kaggle/working/submission-maskrcnn.csv
```

To reuse an existing trained checkpoint after calibration logic changes, run
only the deterministic validation sweep and prediction steps:

```bash
python kaggle/train_maskrcnn.py calibrate "$ROOT" \
  --checkpoint /kaggle/working/maskrcnn-best.pt
python kaggle/train_maskrcnn.py predict "$ROOT" \
  --checkpoint /kaggle/working/maskrcnn-best.pt \
  --output /kaggle/working/submission-maskrcnn.csv
```

Do not submit the CSV until the epoch metrics, row count, image coverage and
RLE round-trip checks have been reviewed. Record the Kaggle image, torch,
torchvision, CUDA and GPU versions in the run notes for reproducibility.
