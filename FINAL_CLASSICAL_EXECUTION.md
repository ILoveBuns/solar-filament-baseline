# Final classical execution evidence

This receipt records a complete local execution against the 180 official test
images on 2026-10-02. Competition data and the generated CSV remain untracked.

## Environment and command

- Python 3
- `numpy==2.4.6`
- `scipy==1.17.1`
- `Pillow==12.3.0`
- `pycocotools==2.0.11`

```bash
python -m solarfil.infer \
  data/official/MAGFiLO_1.0_Kaggle_2026/test/test_images \
  outputs/submission.csv

python scripts/audit_submission.py \
  outputs/submission.csv \
  data/official/MAGFiLO_1.0_Kaggle_2026/test/test_images
```

The inference completed in 13 minutes on CPU. The audit validates the exact
header, image coverage, unique and consecutive IDs, 64-instance limit, COCO
RLE validity, source-image dimensions, minimum area and pairwise disjointness.

## Audit receipt

| Field | Result |
|---|---:|
| Official test images | 180 |
| Submission rows | 11,520 |
| CSV bytes | 19,627,844 |
| SHA-256 | `b7c9faa9815739c92abd86cafa6fe08ad6c92cb8d9fd4919b3d2906df10df223` |
| Instances per image | 64–64 |
| Instance area | 78–524,605 pixels |
| Total foreground area | 77,426,928 pixels |
| Overlapping pixels | 0 |

The newly generated CSV was also byte-for-byte identical to the preserved
July output (`outputs/submission.csv`) with the same SHA-256. This proves that
the selected final route is deterministic for the available official test
set. It does not replace the entrant's final Kaggle notebook execution or
submission confirmation.
