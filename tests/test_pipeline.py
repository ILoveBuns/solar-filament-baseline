import tempfile
import unittest
from pathlib import Path

import numpy as np

from solarfil.metrics import dice
from solarfil.evaluate import label_overlap_metrics, matched_dice, matched_label_dice
from solarfil.segment import segment_instances
from solarfil.submission import decode_mask, write_submission


class PipelineTest(unittest.TestCase):
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
