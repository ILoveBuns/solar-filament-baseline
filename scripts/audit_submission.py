from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from PIL import Image
from pycocotools import mask as mask_utils


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit(csv_path: Path, image_dir: Path) -> dict[str, object]:
    images = {path.stem: path for path in sorted(image_dir.glob("*.jpeg"))}
    if not images:
        raise ValueError(f"no JPEG images found under {image_dir}")

    grouped: dict[str, list[tuple[int, str]]] = defaultdict(list)
    identifiers: set[str] = set()
    with csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        expected = ["filament_id", "segmentation_rle"]
        if reader.fieldnames != expected:
            raise ValueError(f"expected header {expected}, got {reader.fieldnames}")
        for row_number, row in enumerate(reader, 2):
            identifier = row["filament_id"]
            encoding = row["segmentation_rle"]
            if not identifier or not encoding:
                raise ValueError(f"empty field on CSV row {row_number}")
            if identifier in identifiers:
                raise ValueError(f"duplicate identifier: {identifier}")
            identifiers.add(identifier)
            try:
                image_id, rank_text = identifier.rsplit("_", 1)
                rank = int(rank_text)
            except (ValueError, TypeError) as error:
                raise ValueError(f"malformed identifier: {identifier}") from error
            if image_id not in images:
                raise ValueError(f"unknown image identifier: {image_id}")
            grouped[image_id].append((rank, encoding))

    if set(grouped) != set(images):
        missing = sorted(set(images) - set(grouped))
        raise ValueError(f"images without predictions: {missing[:5]}")

    total_area = 0
    min_area: int | None = None
    max_area = 0
    overlap_pixels = 0
    for image_number, (image_id, path) in enumerate(images.items(), 1):
        instances = grouped[image_id]
        ranks = [rank for rank, _ in instances]
        if ranks != list(range(1, len(instances) + 1)):
            raise ValueError(f"non-consecutive or unordered ranks for {image_id}")
        if len(instances) > 64:
            raise ValueError(f"too many instances for {image_id}: {len(instances)}")

        with Image.open(path) as image:
            width, height = image.size
        occupied: dict[str, object] | None = None
        for rank, encoding in instances:
            rle = {"counts": encoding.encode("ascii"), "size": [height, width]}
            area = int(mask_utils.area(rle))
            if area < 20:
                raise ValueError(f"instance below minimum area: {image_id}_{rank}={area}")
            if occupied is None:
                occupied = rle
            else:
                intersection = mask_utils.merge([occupied, rle], intersect=True)
                overlap_pixels += int(mask_utils.area(intersection))
                occupied = mask_utils.merge([occupied, rle], intersect=False)
            total_area += area
            min_area = area if min_area is None else min(min_area, area)
            max_area = max(max_area, area)
        if image_number == 1 or image_number % 10 == 0 or image_number == len(images):
            print(f"audited {image_number}/{len(images)}: {path.name}", flush=True)

    if overlap_pixels:
        raise ValueError(f"predicted instances overlap by {overlap_pixels} pixels")

    counts = [len(grouped[image_id]) for image_id in images]
    return {
        "csv": str(csv_path),
        "bytes": csv_path.stat().st_size,
        "sha256": sha256(csv_path),
        "images": len(images),
        "rows": len(identifiers),
        "instances_per_image_min": min(counts),
        "instances_per_image_max": max(counts),
        "instance_area_min": min_area,
        "instance_area_max": max_area,
        "total_foreground_pixels": total_area,
        "overlap_pixels": overlap_pixels,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Fully audit a competition submission")
    parser.add_argument("csv", type=Path)
    parser.add_argument("image_dir", type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.csv, args.image_dir), indent=2))


if __name__ == "__main__":
    main()
