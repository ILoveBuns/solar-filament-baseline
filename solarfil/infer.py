from __future__ import annotations

import argparse
import csv
import gc
from pathlib import Path

import numpy as np
from PIL import Image

from .segment import segment_labels
from .submission import encode_mask


def infer_directory(
    image_dir: Path,
    output: Path,
    limit: int | None = None,
    *,
    darkness_quantile: float = 0.25,
    min_area: int = 20,
    max_normalized_intensity: float = 0.95,
    max_instances: int = 64,
) -> int:
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
            labels, label_ids = segment_labels(
                image,
                darkness_quantile=darkness_quantile,
                min_area=min_area,
                max_normalized_intensity=max_normalized_intensity,
                max_instances=max_instances,
            )
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
    parser.add_argument("--darkness-quantile", type=float, default=0.25)
    parser.add_argument("--min-area", type=int, default=20)
    parser.add_argument("--max-normalized-intensity", type=float, default=0.95)
    parser.add_argument("--max-instances", type=int, default=64)
    args = parser.parse_args()
    rows = infer_directory(
        args.image_dir,
        args.output,
        args.limit,
        darkness_quantile=args.darkness_quantile,
        min_area=args.min_area,
        max_normalized_intensity=args.max_normalized_intensity,
        max_instances=args.max_instances,
    )
    print(f"submission_rows={rows}")


if __name__ == "__main__":
    main()
