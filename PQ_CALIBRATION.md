# Classical Panoptic Quality calibration

This record preserves a local, official-label calibration experiment without replacing the
platform-verified `0.48` classical submission. Competition data and generated CSVs remain ignored.

## Evaluation correction

The previous local evaluator reported greedy matched Dice and fragmentation diagnostics, but did
not aggregate Panoptic Quality. It also allowed zero-overlap prediction/truth pairs to consume a
diagnostic match, making the `unmatched_*` counts too optimistic. The evaluator now:

- matches IoU strictly above `0.5`, consistent with the repository's existing PQ implementation;
- aggregates matched IoU, true positives, false positives and false negatives into PQ;
- treats zero-overlap pairs as unmatched in the fragmentation diagnostics; and
- accepts `--offset` so parameter selection and holdout verification use disjoint images.

Unit tests cover duplicate-prediction penalties, zero-overlap diagnostics, and offset validation.

## Development slice

The first 10 lexically sorted official training images contained 74 truth instances. The existing
route (`darkness_quantile=0.25`, `max_instances=64`) predicted 640 instances and reached local PQ
`0.03352`. A bounded scan varied only the darkness quantile and maximum instance count; it did not
search against the holdout slice. The coarse scan selected this initial candidate:

```text
darkness_quantile = 0.20
min_area = 20
max_normalized_intensity = 0.95
max_instances = 12
```

It reached local PQ `0.06632` on the development slice, with a prediction/truth ratio of `1.62`.
After validating that initial candidate as described below, one final bounded refinement over
`0.18`–`0.22` reused only this same development slice. It identified `0.18` at PQ `0.07279`.
To preserve a clean decision, `0.18` was confirmed first on a new offset-50 slice before being
evaluated retrospectively on either earlier holdout.

## Disjoint holdout

The next 10 sorted images (offset 10) contained 66 truth instances.

| Route | Predictions | Prediction/truth | Matched Dice | PQ | TP / FP / FN |
| --- | ---: | ---: | ---: | ---: | ---: |
| Existing 0.48 route | 640 | 9.70 | 0.3608 | 0.03456 | 20 / 620 / 46 |
| Initial 0.20 candidate | 120 | 1.82 | 0.1814 | **0.07121** | 11 / 109 / 55 |

The initial candidate more than doubled local PQ on the untouched slice while reducing the false-positive
count by 82%. Its lower matched Dice is expected because the legacy statistic does not penalize
unmatched predictions and therefore rewards retaining many components. This 20-image experiment is
still small and lexically sampled; it is evidence for one platform comparison, not proof that the
candidate beats `0.48` on the public or private leaderboard.

### Extended untouched holdout

After freezing the candidate, the following 30 images (offset 20) were evaluated without any
further parameter search. They contained 298 truth instances.

| Route | Predictions | Prediction/truth | Matched Dice | PQ | TP / FP / FN |
| --- | ---: | ---: | ---: | ---: | ---: |
| Existing 0.48 route | 1,920 | 6.44 | 0.4029 | 0.05613 | 97 / 1,823 / 201 |
| Initial 0.20 candidate | 360 | 1.21 | 0.2188 | **0.11430** | 57 / 303 / 241 |

Across both untouched slices (40 images, 364 truth instances), aggregation from the underlying
matched-IoU and TP/FP/FN totals gives PQ `0.05092` for the existing route and `0.10480` for the
0.20 candidate, a **2.06×** improvement. It reduced false positives from 2,443 to 412. This
second, larger confirmation was performed only after the candidate parameters were frozen, so it
strengthens the case for one platform comparison without converting local evidence into a claimed
leaderboard result.

### Final refinement confirmation

The `0.18` refinement was then compared with `0.20` and the existing route on a new, untouched
30-image slice beginning at offset 50. It contained 512 truth instances and had not been inspected
during either parameter scan.

| Route | Predictions | Prediction/truth | Matched Dice | PQ | TP / FP / FN |
| --- | ---: | ---: | ---: | ---: | ---: |
| Existing 0.48 route | 1,920 | 3.75 | 0.3109 | 0.07092 | 135 / 1,785 / 377 |
| Initial 0.20 candidate | 360 | 0.70 | 0.1412 | 0.09799 | 65 / 295 / 447 |
| Final 0.18 candidate | 360 | 0.70 | 0.1475 | **0.10384** | 69 / 291 / 443 |

Only after `0.18` won this fresh confirmation was it evaluated retrospectively over the earlier
40 holdout images, where it also improved on `0.20` (`0.11292` versus `0.10480`). Across all 70
images never used for parameter selection (876 truth instances), aggregate PQ was `0.06000` for
the existing route, `0.10134` for `0.20`, and **`0.10831` for `0.18`**. The final candidate improved
PQ by **1.81×** over the existing route and reduced false positives from 4,228 to 698.

### Rejected observatory-specific cap

After all holdout decisions were complete, the full training annotations were inspected only to
decide whether a new observatory-specific experiment was justified. Mean truth counts by suffix
ranged narrowly from `10.38` (Mh) to `12.73` (Bh); medians were 9 or 10 at every site, and the
interquartile ranges overlapped heavily (approximately 5–16). The global mean was `11.60`.

Those distributions do not support six separate instance caps, especially after the frozen global
cap of 12 generalized consistently. No site-specific parameters were introduced and no additional
candidate was selected from this post-hoc inspection.

## Full test-set candidate receipt

The candidate was executed over all 180 available official test images as a separate ignored file:

```bash
python -m solarfil.infer \
  data/official/MAGFiLO_1.0_Kaggle_2026/test/test_images \
  outputs/submission-pq-calibrated-q018.csv \
  --darkness-quantile 0.18 \
  --max-instances 12
```

The full structural audit decoded every RLE and verified image coverage, consecutive IDs, source
dimensions, nonempty masks, and pairwise disjointness.

| Field | Result |
| --- | ---: |
| Official test images | 180 |
| Submission rows | 2,160 |
| CSV bytes | 13,808,475 |
| SHA-256 | `fa4de92ba216877841e70c8a104689e76d7789c64c7bec529c87e4f7dca1bc9c` |
| Instances per image | 12–12 |
| Instance area | 59–465,462 pixels |
| Total foreground area | 60,684,891 pixels |
| Overlapping pixels | 0 |

## Decision gate

Do not replace the selected `0.48` route from local evidence. If the competition still permits a
comparison submission, upload this exact candidate, preserve its platform reference and score, and
switch the final notebook/report only if the organizer's metric verifies an improvement. Otherwise
retain the existing deterministic submission and its already verified report.

The earlier `0.20` CSV and its receipt remain preserved as intermediate evidence; it is not the
file named by this decision gate.
