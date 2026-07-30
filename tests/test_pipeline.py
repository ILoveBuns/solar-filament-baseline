import tempfile
import unittest
from pathlib import Path

import numpy as np

from solarfil.metrics import dice
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


if __name__ == "__main__":
    unittest.main()

