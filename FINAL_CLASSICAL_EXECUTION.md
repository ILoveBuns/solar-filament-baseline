# Final classical execution evidence

This receipt records complete local CLI and final-notebook executions against
the 180 official test images on 2026-10-02. Competition data and generated
CSVs remain untracked.

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

The final notebook also supports a local verification without changing its
default Kaggle paths:

```bash
SOLAR_IMAGE_DIR=data/official/MAGFiLO_1.0_Kaggle_2026/test/test_images \
SOLAR_OUTPUT=/tmp/submission-classical.csv \
SOLAR_SKIP_INSTALL=1 \
python scripts/execute_notebook_cells.py kaggle/final_classical_pipeline.ipynb
```

`SOLAR_SKIP_INSTALL=1` is only for an environment where the exact pinned
requirements are already installed. On Kaggle the variable is absent, so the
notebook installs `requirements.txt` and uses its normal `/kaggle` paths.

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

All four code cells from `kaggle/final_classical_pipeline.ipynb` were then
executed sequentially in one namespace using the checked-in executor, exact
pinned dependencies, and local path overrides. The notebook itself reported 180
input images, 11,520 prediction rows, 180 audited images, zero overlaps, and
the same receipt above before exiting successfully.

The CLI output and notebook output were both byte-for-byte identical to the
preserved July output (`outputs/submission.csv`) with the same SHA-256. This
proves that the selected final route is deterministic for the available
official test set. It does not replace the entrant's final Kaggle submission
confirmation.
