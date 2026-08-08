import importlib.util
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "build_public_inference_notebook.py"
SPEC = importlib.util.spec_from_file_location("build_public_inference_notebook", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class PublicInferenceNotebookTest(unittest.TestCase):
    def setUp(self):
        self.source = Path("/workspace/solar-filament-public-yolo-unet/yolo-unet-solar-filament-segmentation.ipynb")

    def test_generated_notebook_is_inference_only_and_attributed(self):
        notebook = MODULE.build(self.source)
        joined = "\n".join(cell["source"] for cell in notebook["cells"])
        self.assertIn(MODULE.PUBLIC_HANDLE, joined)
        self.assertIn("Apache-2.0", joined)
        self.assertIn("submission.to_csv", joined)
        self.assertIn("frozen hash mismatch", joined)
        self.assertNotIn("model.train(", joined)
        self.assertNotIn("train_refiner(", joined)

    def test_expected_hashes_match_frozen_assets(self):
        for name, expected in MODULE.EXPECTED.items():
            self.assertEqual(MODULE.digest(self.source.parent / name), expected)


if __name__ == "__main__":
    unittest.main()
