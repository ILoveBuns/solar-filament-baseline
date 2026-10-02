import json
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class FinalMaterialsTest(unittest.TestCase):
    def test_final_notebook_is_valid_and_complete(self):
        notebook = json.loads((ROOT / "kaggle/final_classical_pipeline.ipynb").read_text())
        self.assertEqual(4, notebook["nbformat"])
        source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
        for required in ("infer_directory", "submission-classical.csv", "55113085", "decode_mask"):
            self.assertIn(required, source)

    def test_report_has_required_sections_and_honest_results(self):
        report = (ROOT / "report/technical_report.md").read_text()
        for required in ("## Abstract", "## 1. Task", "## 2. Method", "## 3. Results", "## 4. Limitations"):
            self.assertIn(required, report)
        self.assertIn("55113085", report)
        self.assertIn("0.48", report)
        self.assertIn("not claimed as our result", report)

    def test_latex_report_uses_competition_template_and_score_provenance(self):
        report = (ROOT / "report/main.tex").read_text()
        self.assertIn(r"\documentclass[sigconf]{acmart}", report)
        self.assertIn(r"\subtitle{A Solution to the Solar Filament Segmentation Challenge 2026}", report)
        self.assertIn("55113085", report)
        self.assertIn("55354276", report)
        self.assertIn("55354612", report)
        self.assertIn("do not attribute that score to our entry", report)
        self.assertIn(r"\section{Acknowledgment}", report)

    def test_requirements_are_exactly_pinned(self):
        lines = (ROOT / "requirements.txt").read_text().splitlines()
        self.assertEqual(
            {line.split("==", 1)[0].lower() for line in lines},
            {"numpy", "pillow", "pycocotools", "scipy"},
        )
        self.assertTrue(all(re.fullmatch(r"[A-Za-z0-9_.-]+==[^=\s]+", line) for line in lines))

        experiments = (ROOT / "requirements-experiments.txt").read_text().splitlines()
        self.assertEqual("-r requirements.txt", experiments[0])
        self.assertTrue(all(
            re.fullmatch(r"[A-Za-z0-9_.-]+==[^=\s]+", line)
            for line in experiments[1:]
        ))

    def test_final_notebook_does_not_depend_on_experimental_stack(self):
        notebook = json.loads((ROOT / "kaggle/final_classical_pipeline.ipynb").read_text())
        source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
        for excluded in ("pandas", "torch", "torchvision", "ultralytics", "cv2", "sklearn"):
            self.assertNotIn(excluded, source)


if __name__ == "__main__":
    unittest.main()
