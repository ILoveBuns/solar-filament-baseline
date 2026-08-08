import tempfile
import unittest
from itertools import product
from pathlib import Path

import numpy as np
from types import SimpleNamespace

from solarfil.metrics import dice
from solarfil.calibration import (
    calibration_selection_score,
    panoptic_quality,
    resolve_prediction_thresholds,
    score_instances,
    sweep_thresholds,
)
from solarfil.coco_data import (
    greedy_scores_from_matrix,
    select_best_image_records,
    stable_stratified_split,
)
from solarfil.evaluate import label_overlap_metrics, matched_dice, matched_label_dice
from solarfil.segment import segment_instances
from solarfil.submission import decode_mask, write_submission
class PipelineTest(unittest.TestCase):
    def test_panoptic_quality_penalizes_duplicate_predictions(self):
        truth = np.ones((1, 3, 3), dtype=bool)
        probabilities = np.repeat(truth.astype(float), 2, axis=0)
        result = score_instances(
            np.array([0.9, 0.8]), probabilities, truth, 0.5, 0.5, 1
        )
        self.assertEqual(1, result["true_positives"])
        self.assertEqual(1, result["false_positives"])
        self.assertEqual(0, result["false_negatives"])
        self.assertAlmostEqual(2 / 3, panoptic_quality(
            result["matched_iou_sum"],
            result["true_positives"],
            result["false_positives"],
            result["false_negatives"],
        ))

    def test_calibration_penalizes_fragmented_instance_count(self):
        balanced = calibration_selection_score(0.60, 1.0)
        fragmented = calibration_selection_score(0.62, 3.0)
        self.assertGreater(balanced, fragmented)

    def test_checkpoint_calibration_overrides_prediction_defaults(self):
        args = SimpleNamespace(score_threshold=0.35, mask_threshold=0.5, min_area=24)
        checkpoint = {"calibration": {
            "score_threshold": 0.25,
            "mask_threshold": 0.6,
            "min_area": 48,
        }}
        self.assertEqual((0.25, 0.6, 48), resolve_prediction_thresholds(checkpoint, args))

    def test_threshold_sweep_reuses_probabilities_and_selects_best_point(self):
        truth = np.zeros((1, 6, 6), dtype=bool)
        truth[0, 1:5, 1:5] = True
        probabilities = np.zeros((2, 6, 6), dtype=float)
        probabilities[0, 1:5, 1:5] = 0.9
        probabilities[1, 0, 0] = 0.8
        cached = [(np.array([0.8, 0.2]), probabilities, truth)]
        results = sweep_thresholds(cached, [0.1, 0.5], [0.5], [1])
        self.assertEqual(0.5, results[0]["score_threshold"])
        self.assertEqual(1.0, results[0]["mean_matched_dice"])
        self.assertEqual(1.0, results[0]["prediction_truth_ratio"])
        self.assertEqual(1.0, results[0]["panoptic_quality"])

    def test_optimized_threshold_sweep_matches_individual_scoring(self):
        rng = np.random.default_rng(7)
        probabilities = rng.random((4, 9, 8))
        truths = rng.random((3, 9, 8)) > 0.72
        confidence = np.array([0.2, 0.45, 0.7, 0.95])
        score_thresholds = [0.1, 0.5, 0.9]
        mask_thresholds = [0.35, 0.6]
        min_areas = [1, 12]
        actual = sweep_thresholds(
            [(confidence, probabilities, truths)],
            score_thresholds,
            mask_thresholds,
            min_areas,
        )
        by_configuration = {
            (row["score_threshold"], row["mask_threshold"], row["min_area"]): row
            for row in actual
        }
        for configuration in product(score_thresholds, mask_thresholds, min_areas):
            score_threshold, mask_threshold, min_area = configuration
            expected = score_instances(
                confidence,
                probabilities,
                truths,
                score_threshold,
                mask_threshold,
                min_area,
            )
            row = by_configuration[configuration]
            self.assertAlmostEqual(
                expected["dice_sum"] / expected["truth_count"],
                row["mean_matched_dice"],
            )
            self.assertAlmostEqual(
                expected["prediction_count"] / expected["truth_count"],
                row["prediction_truth_ratio"],
            )
            for key in ("true_positives", "false_positives", "false_negatives"):
                self.assertEqual(expected[key], row[key])
            self.assertAlmostEqual(
                panoptic_quality(
                    expected["matched_iou_sum"],
                    expected["true_positives"],
                    expected["false_positives"],
                    expected["false_negatives"],
                ),
                row["panoptic_quality"],
            )

    def test_instance_scoring_filters_tiny_masks(self):
        truth = np.ones((1, 3, 3), dtype=bool)
        probabilities = np.stack([truth[0], np.eye(3, dtype=bool)]).astype(float)
        result = score_instances(np.array([0.9, 0.9]), probabilities, truth, 0.5, 0.5, 4)
        self.assertEqual(1, result["prediction_count"])
        self.assertEqual(1.0, result["dice_sum"])

    def test_selects_most_complete_duplicate_annotation_record(self):
        coco = {
            "images": [
                {"id": 1, "file_name": "20140101000000Mh.jpeg"},
                {"id": 2, "file_name": "20140101000000Mh.jpeg"},
            ],
            "annotations": [
                {"image_id": 1, "area": 50},
                {"image_id": 2, "area": 40},
                {"image_id": 2, "area": 20},
            ],
        }
        self.assertEqual(2, select_best_image_records(coco)["20140101000000Mh.jpeg"]["id"])

    def test_stable_split_is_disjoint_and_order_independent(self):
        files = [f"2014{month:02d}01000000{site}.jpeg" for month in range(1, 7) for site in ("Mh", "Ch")]
        train_a, valid_a = stable_stratified_split(files, seed=7)
        train_b, valid_b = stable_stratified_split(list(reversed(files)), seed=7)
        self.assertEqual((train_a, valid_a), (train_b, valid_b))
        self.assertFalse(set(train_a) & set(valid_a))
        self.assertEqual(set(files), set(train_a) | set(valid_a))

    def test_greedy_scores_from_precomputed_matrix(self):
        matrix = np.array([[0.9, 0.8, 0.0], [0.7, 0.1, 0.6]])
        scores = greedy_scores_from_matrix(matrix, truth_count=3)
        self.assertEqual(3, len(scores))
        self.assertAlmostEqual(1.5, sum(scores))

    def test_detects_dark_connected_filament(self):
        size = 128
        yy, xx = np.indices((size, size))
        radius = np.hypot(xx - 63.5, yy - 63.5)
        image = (1.0 - 0.25 * (radius / 62)).clip(0.5, 1.0)
        truth = np.zeros_like(image, dtype=np.uint8)
        truth[58:63, 35:91] = 1
        truth[54:67, 58:64] = 1
        image[truth.astype(bool)] *= 0.3
        masks = segment_instances(image, darkness_quantile=0.05, min_area=20)
        self.assertTrue(masks)
        self.assertGreater(max(dice(mask, truth) for mask in masks), 0.85)

    def test_coco_rle_roundtrip(self):
        mask = np.zeros((32, 32), dtype=np.uint8)
        mask[4:9, 7:20] = 1
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "submission.csv"
            self.assertEqual(1, write_submission(path, {"demo": [mask]}))
            counts = path.read_text().splitlines()[1].split(",", 1)[1]
            np.testing.assert_array_equal(mask, decode_mask(counts, mask.shape))

    def test_fast_label_matching_matches_mask_reference(self):
        labels = np.zeros((12, 12), dtype=np.int32)
        labels[1:4, 1:5] = 1
        labels[7:11, 6:10] = 2
        truths = [
            np.pad(np.ones((3, 3), dtype=np.uint8), ((1, 8), (2, 7))),
            np.pad(np.ones((3, 4), dtype=np.uint8), ((7, 2), (6, 2))),
        ]
        expected = matched_dice([(labels == 1), (labels == 2)], truths)
        actual = matched_label_dice(labels, [1, 2], truths)
        self.assertEqual(len(actual), len(expected))
        self.assertTrue(np.allclose(sorted(actual), sorted(expected)))

    def test_overlap_diagnostics_detect_fragmentation_and_merging(self):
        labels = np.zeros((8, 12), dtype=np.int32)
        labels[1:3, 1:5] = 1
        labels[3:5, 1:5] = 2
        labels[1:5, 7:11] = 3
        truths = [
            (np.pad(np.ones((4, 4), dtype=np.uint8), ((1, 3), (1, 7)))),
            (np.pad(np.ones((4, 2), dtype=np.uint8), ((1, 3), (7, 3)))),
            (np.pad(np.ones((4, 2), dtype=np.uint8), ((1, 3), (9, 1)))),
        ]
        _, diagnostics = label_overlap_metrics(labels, [1, 2, 3], truths, 0.1)
        self.assertEqual(1, diagnostics["one_to_many_truths"])
        self.assertEqual(1, diagnostics["many_to_one_predictions"])


if __name__ == "__main__":
    unittest.main()
