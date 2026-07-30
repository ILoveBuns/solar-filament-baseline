from __future__ import annotations

import argparse
import csv
import gc
from pathlib import Path

import numpy as np
from PIL import Image

from .segment import segment_labels
from .submission import encode_mask


def infer_directory(image_dir: Path, output: Path, limit: int | None = None) -> int:
    files = sorted(image_dir.glob("*.jpeg"))
    if limit is not None:
        files = files[:limit]
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    with output.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["filament_id", "segmentation_rle"])
        for file_index, path in enumerate(files, 1):
            image = np.asarray(Image.open(path).convert("L"))
            labels, label_ids = segment_labels(image, min_area=20)
            for index, label_id in enumerate(label_ids, 1):
                writer.writerow([f"{path.stem}_{index}", encode_mask(labels == label_id)])
                rows += 1
            del image, labels, label_ids
            gc.collect()
            if file_index == 1 or file_index % 10 == 0 or file_index == len(files):
                print(f"processed {file_index}/{len(files)}: {path.name}", flush=True)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("image_dir", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    rows = infer_directory(args.image_dir, args.output, args.limit)
    print(f"submission_rows={rows}")


if __name__ == "__main__":
    main()
