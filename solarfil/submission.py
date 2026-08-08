from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from pycocotools import mask as mask_utils


def encode_mask(mask: np.ndarray) -> str:
    encoded = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    return encoded["counts"].decode("ascii")


def decode_mask(counts: str, size: tuple[int, int]) -> np.ndarray:
    return mask_utils.decode({"counts": counts.encode("ascii"), "size": list(size)})


def make_masks_disjoint(masks: list[np.ndarray]) -> list[np.ndarray]:
    """Assign overlapping pixels to the first (highest-priority) mask only."""
    if not masks:
        return []
    occupied = np.zeros(np.asarray(masks[0]).shape, dtype=bool)
    result: list[np.ndarray] = []
    for mask in masks:
        clean = np.asarray(mask, dtype=bool).copy()
        if clean.shape != occupied.shape:
            raise ValueError("all masks must have the same shape")
        clean[occupied] = False
        if clean.any():
            occupied |= clean
            result.append(clean)
    return result


def write_submission(path: str | Path, predictions: dict[str, list[np.ndarray]]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["filament_id", "segmentation_rle"])
        for image_id, masks in sorted(predictions.items()):
            for index, mask in enumerate(make_masks_disjoint(masks), 1):
                writer.writerow([f"{image_id}_{index}", encode_mask(mask)])
                count += 1
    return count
