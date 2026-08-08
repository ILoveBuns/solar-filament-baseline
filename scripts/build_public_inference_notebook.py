#!/usr/bin/env python3
"""Build an inference-only Kaggle notebook from a frozen public baseline.

The source notebook is Apache-2.0 licensed and published at:
https://www.kaggle.com/code/ektarr/yolo-unet-solar-filament-segmentation
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


PUBLIC_HANDLE = "ektarr/yolo-unet-solar-filament-segmentation"
EXPECTED = {
    "best_model.pt": "0721382bc86742d06fdfd9c13f37501f5590cecbda99d5c87d1b77913cdbe365",
    "best_refiner.pt": "b79cf01b614b54117e465ad2b8e76941258435b8ceddbf654aff6bc179624ec2",
}


def code(source: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source}


def markdown(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(source_path: Path) -> dict:
    source = json.loads(source_path.read_text())
    cells = source["cells"]
    refiner_defs = cells[7]["source"].split("def build_crop_arrays", 1)[0]

    fetch = f'''import hashlib
import shutil
from pathlib import Path
import kagglehub

PUBLIC_HANDLE = {PUBLIC_HANDLE!r}
EXPECTED = {EXPECTED!r}
asset_dir = Path(kagglehub.notebook_output_download(PUBLIC_HANDLE))
for name, expected in EXPECTED.items():
    matches = list(asset_dir.rglob(name))
    if len(matches) != 1:
        raise RuntimeError(f"expected exactly one {{name}}, found {{len(matches)}}")
    actual = hashlib.sha256(matches[0].read_bytes()).hexdigest()
    if actual != expected:
        raise RuntimeError(f"frozen hash mismatch for {{name}}: {{actual}}")
    shutil.copy2(matches[0], name)
print("verified frozen public weights", sorted(EXPECTED))
'''

    load_models = '''best_model = YOLO("best_model.pt")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
'''

    validate = '''assert list(submission.columns) == ["filament_id", "segmentation_rle"]
assert not submission.empty
assert submission["filament_id"].is_unique
assert submission["segmentation_rle"].map(lambda value: isinstance(value, str) and bool(value)).all()
known_images = {Path(name).stem for name in test_files}
submitted_images = {value.rsplit("_", 1)[0] for value in submission["filament_id"]}
assert submitted_images <= known_images
report = {
    "source": PUBLIC_HANDLE,
    "source_license": "Apache-2.0",
    "asset_sha256": EXPECTED,
    "test_images": len(known_images),
    "images_with_predictions": len(submitted_images),
    "prediction_rows": len(submission),
    "missing_prediction_images": sorted(known_images - submitted_images),
}
Path("public-yolo-unet-inference-report.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
'''

    return {
        "cells": [
            markdown(
                "# Frozen public YOLO + U-Net inference candidate\n\n"
                "Derived from Maxim/ektarr's Apache-2.0 Kaggle notebook. The published 0.69 "
                "score belongs to the source author; this notebook claims no score until our own "
                "submission is accepted and scored. It downloads and verifies the frozen public "
                "weights, then performs inference only."
            ),
            code("%pip install -q ultralytics kagglehub"),
            code(cells[1]["source"]),
            code(cells[2]["source"]),
            code(fetch),
            code(refiner_defs),
            code(load_models),
            code(cells[10]["source"]),
            code(cells[13]["source"]),
            code(validate),
        ],
        "metadata": source.get("metadata", {}),
        "nbformat": 4,
        "nbformat_minor": 4,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    for name, expected in EXPECTED.items():
        asset = args.source.parent / name
        if digest(asset) != expected:
            raise SystemExit(f"local source asset hash mismatch: {asset}")

    notebook = build(args.source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(notebook, indent=1))


if __name__ == "__main__":
    main()
