#!/usr/bin/env python3
"""Remove within-image instance overlap from an existing submission."""

from __future__ import annotations

import argparse
import csv
from collections import OrderedDict
from pathlib import Path

from solarfil.submission import decode_mask, encode_mask, make_masks_disjoint


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--start-image", type=int, default=0)
    parser.add_argument("--max-images", type=int)
    parser.add_argument("--size", type=int, default=2048)
    args = parser.parse_args()

    groups: OrderedDict[str, list[str]] = OrderedDict()
    with args.input.open(newline="") as handle:
        for row in csv.DictReader(handle):
            image_id = row["filament_id"].rsplit("_", 1)[0]
            groups.setdefault(image_id, []).append(row["segmentation_rle"])

    items = list(groups.items())[args.start_image :]
    if args.max_images is not None:
        items = items[: args.max_images]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    input_masks = output_masks = 0
    with args.output.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["filament_id", "segmentation_rle"])
        for image_id, counts in items:
            masks = [decode_mask(value, (args.size, args.size)) for value in counts]
            clean_masks = make_masks_disjoint(masks)
            input_masks += len(masks)
            output_masks += len(clean_masks)
            for index, mask in enumerate(clean_masks, 1):
                writer.writerow([f"{image_id}_{index}", encode_mask(mask)])

    print({
        "images": len(items),
        "input_masks": input_masks,
        "output_masks": output_masks,
        "dropped_empty_masks": input_masks - output_masks,
    })


if __name__ == "__main__":
    main()
