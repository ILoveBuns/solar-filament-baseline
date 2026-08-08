import importlib.util
import json
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "build_public_inference_notebook.py"
SPEC = importlib.util.spec_from_file_location("build_public_inference_notebook", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class PublicInferenceNotebookTest(unittest.TestCase):
    def setUp(self):
        generated = Path(__file__).parents[1] / "kaggle" / "infer_public_yolo_unet.ipynb"
        self.notebook = json.loads(generated.read_text())

    def test_generated_notebook_is_inference_only_and_attributed(self):
        joined = "\n".join(cell["source"] for cell in self.notebook["cells"])
        self.assertIn(MODULE.PUBLIC_HANDLE, joined)
        self.assertIn("Apache-2.0", joined)
        self.assertIn("submission.to_csv", joined)
        self.assertIn("frozen hash mismatch", joined)
        self.assertNotIn("model.train(", joined)
        self.assertNotIn("train_refiner(", joined)

    def test_expected_hashes_are_frozen_in_generated_notebook(self):
        joined = "\n".join(cell["source"] for cell in self.notebook["cells"])
        for name, expected in MODULE.EXPECTED.items():
            self.assertEqual(len(expected), 64)
            int(expected, 16)
            self.assertIn(name, joined)
            self.assertIn(expected, joined)


if __name__ == "__main__":
    unittest.main()
