import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from solarfil.submission import encode_mask


SCRIPT = Path(__file__).parents[1] / "scripts" / "audit_submission.py"
SPEC = importlib.util.spec_from_file_location("audit_submission", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class SubmissionAuditTest(unittest.TestCase):
    def test_fully_decodes_valid_disjoint_submission(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images = root / "images"
            images.mkdir()
            Image.new("L", (8, 8)).save(images / "sample.jpeg")
            first = np.zeros((8, 8), dtype=bool)
            first[:4, :5] = True
            second = np.zeros((8, 8), dtype=bool)
            second[4:, 3:] = True
            output = root / "submission.csv"
            with output.open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["filament_id", "segmentation_rle"])
                writer.writerow(["sample_1", encode_mask(first)])
                writer.writerow(["sample_2", encode_mask(second)])

            receipt = MODULE.audit(output, images)

            self.assertEqual(2, receipt["rows"])
            self.assertEqual(0, receipt["overlap_pixels"])
            self.assertEqual(20, receipt["instance_area_min"])

    def test_rejects_overlapping_instances(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images = root / "images"
            images.mkdir()
            Image.new("L", (8, 8)).save(images / "sample.jpeg")
            mask = np.ones((8, 8), dtype=bool)
            output = root / "submission.csv"
            with output.open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["filament_id", "segmentation_rle"])
                writer.writerow(["sample_1", encode_mask(mask)])
                writer.writerow(["sample_2", encode_mask(mask)])

            with self.assertRaisesRegex(ValueError, "overlap"):
                MODULE.audit(output, images)


if __name__ == "__main__":
    unittest.main()
