# An Interpretable Classical Baseline for Solar Filament Instance Segmentation

**Competition:** Solar Filament Segmentation Challenge 2026  
**Entrant:** ILoveBuns  
**Repository version:** record the final Git commit before submission

## Abstract

We present a deterministic, CPU-friendly instance-segmentation pipeline for solar filaments. The method removes center-to-limb brightness variation with robust radial normalization, identifies dark filament candidates relative to the visible solar disk, separates them into connected components, filters small components, and emits compressed COCO run-length encodings. Our strongest official Kaggle submission is platform reference `55113085`, with a public score of **0.48**. This is higher than our later Mask R-CNN (`0.31`, ref `55354276`) and frozen public YOLO plus U-Net (`0.34`, ref `55354612`) experiments. The classical pipeline is therefore the selected final method. Its main advantages are reproducibility, low compute cost, and transparent failure modes; its main limitation is that intensity connectivity alone cannot reliably resolve touching or low-contrast filaments.

## 1. Task and data handling

The task is instance segmentation: each detected filament must be represented as a separate binary mask. Images exhibit substantial radial brightness variation, so a fixed global threshold confounds limb darkening with actual dark solar structures. We process official JPEG inputs in grayscale and do not use external training data or pretrained model weights for the selected method.

The repository's final notebook discovers the attached official test directory, runs the complete inference path, writes the required `filament_id,segmentation_rle` schema, checks identifier uniqueness and non-empty encodings, and decodes a sample mask as an integrity check. Dependencies are pinned in `requirements.txt`.

## 2. Method

### 2.1 Robust radial normalization

For an image of height \(H\) and width \(W\), we take the geometric center and define a solar disk radius of \(0.48\min(H,W)\). Disk pixels are assigned to 96 radial bins. For each bin we estimate the local background with its upper quartile; this is less affected by dark filament pixels than a mean or median. Dividing each pixel by the corresponding radial profile produces a normalized image in which center-to-limb variation is substantially reduced.

### 2.2 Candidate extraction and instances

Within the disk, the candidate threshold is the smaller of the 25th percentile of normalized intensity and 0.95. Pixels no brighter than this threshold form the candidate foreground. Four-connected component labeling converts the foreground into candidate instances. Components smaller than 20 pixels are discarded during the submitted inference path, and at most the 64 largest components per image are retained. Every retained mask is encoded using compressed COCO RLE.

### 2.3 Determinism and complexity

There is no training stage, random initialization, or model checkpoint. Radial statistics, thresholding, labeling, sorting, and encoding are deterministic for fixed inputs and library versions. Runtime and memory scale approximately linearly with image pixels, apart from component bookkeeping. The pipeline runs on CPU and avoids model-download or accelerator requirements.

## 3. Results and model selection

The following values are official platform results recorded for our submissions; they are not synthetic local estimates.

| Method | Platform reference | Public score | Decision |
|---|---:|---:|---|
| Radial normalization + connected components | `55113085` | **0.48** | Selected final method |
| Torchvision Mask R-CNN | `55354276` | 0.31 | Frozen; below baseline |
| Public YOLO detector + U-Net refiner | `55354612` | 0.34 | Frozen; below baseline |

The result supports a conservative engineering conclusion: for this dataset and our tested configurations, additional model complexity did not translate into a higher public score. We therefore preserve the empirically strongest submission rather than reporting the source author's score or extrapolating from local metrics. The `0.69` associated with the upstream public YOLO notebook belongs to its original author and is explicitly not claimed as our result.

## 4. Limitations, reproducibility, and responsible use

Connected components may fragment one filament when its contrast is discontinuous, or merge neighboring filaments that touch after thresholding. Bright active regions and image artifacts can also distort radial background estimates. Public leaderboard performance may not equal final/private evaluation, and the three experiments do not establish statistical superiority across all possible architectures.

Reproduction requires the public repository, official competition data, pinned dependencies, and `kaggle/final_classical_pipeline.ipynb`. The notebook executes the entire selected method and produces the submission CSV without additional weights or private files. Unit tests cover synthetic filament recovery, mask disjointness, and COCO RLE round trips. The repository is MIT licensed.

AI-assisted coding was used in preparing and reviewing code and documentation. No AI-generated score is presented as an evaluation result. The entrant remains responsible for inspecting the implementation, running the notebook, confirming the platform references, and making the final submission.

## References

1. Solar Filament Segmentation Challenge 2026, official competition rules, data, and evaluation materials.
2. Lin et al., *Microsoft COCO: Common Objects in Context*, ECCV 2014.
3. SciPy documentation, connected-component labeling (`scipy.ndimage.label`).

> Typesetting note: transfer this content into the organizer's official Overleaf template and verify the four-page limit before final submission. Do not submit this Markdown file directly.
