from __future__ import annotations

import argparse
import gc
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils

from .metrics import dice
from .segment import segment_labels


def annotation_mask(annotation: dict, height: int, width: int) -> np.ndarray:
    rles = mask_utils.frPyObjects(annotation["segmentation"], height, width)
    return mask_utils.decode(mask_utils.merge(rles)).astype(np.uint8)


def matched_dice(predictions: list[np.ndarray], truths: list[np.ndarray]) -> list[float]:
    """Greedy one-to-one Dice matching; unmatched ground truths score zero."""
    candidates = sorted(
        (
            (dice(prediction, truth), pi, ti)
            for pi, prediction in enumerate(predictions)
            for ti, truth in enumerate(truths)
        ),
        reverse=True,
    )
    used_p: set[int] = set()
    used_t: set[int] = set()
    scores: list[float] = []
    for score, pi, ti in candidates:
        if pi not in used_p and ti not in used_t:
            used_p.add(pi)
            used_t.add(ti)
            scores.append(score)
    scores.extend([0.0] * (len(truths) - len(used_t)))
    return scores


def matched_label_dice(labels: np.ndarray, label_ids: list[int], truths: list[np.ndarray]) -> list[float]:
    sizes = np.bincount(labels.ravel())
    candidates: list[tuple[float, int, int]] = []
    for label_id in label_ids:
        area = int(sizes[label_id])
        for truth_id, truth in enumerate(truths):
            intersection = int(np.count_nonzero((labels == label_id) & truth.astype(bool)))
            score = 2 * intersection / (area + int(truth.sum()))
            candidates.append((score, label_id, truth_id))
    candidates.sort(reverse=True)
    used_labels: set[int] = set()
    used_truths: set[int] = set()
    scores: list[float] = []
    for score, label_id, truth_id in candidates:
        if label_id not in used_labels and truth_id not in used_truths:
            used_labels.add(label_id)
            used_truths.add(truth_id)
            scores.append(score)
    scores.extend([0.0] * (len(truths) - len(used_truths)))
    return scores


def evaluate_dataset(root: Path, limit: int | None = None) -> dict[str, float]:
    annotation_path = root / "train/MAGFiLO_1.0_Annotations_kaggle2026_train.json"
    data = json.loads(annotation_path.read_text())
    images_by_id = {item["id"]: item for item in data["images"]}
    annotations_by_file: dict[str, list[dict]] = defaultdict(list)
    for annotation in data["annotations"]:
        filename = images_by_id[annotation["image_id"]]["file_name"]
        annotations_by_file[filename].append(annotation)

    scores: list[float] = []
    predicted_count = truth_count = 0
    files = sorted(annotations_by_file)[:limit]
    for filename in files:
        image = np.asarray(Image.open(root / "train/train_images" / filename).convert("L"))
        truths = [annotation_mask(x, *image.shape) for x in annotations_by_file[filename]]
        labels, label_ids = segment_labels(image, min_area=20)
        scores.extend(matched_label_dice(labels, label_ids, truths))
        predicted_count += len(label_ids)
        truth_count += len(truths)
        del image, truths, labels, label_ids
        gc.collect()
    return {
        "images": len(files),
        "instances_truth": truth_count,
        "instances_predicted": predicted_count,
        "mean_matched_dice": float(np.mean(scores)) if scores else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    print(json.dumps(evaluate_dataset(args.root, args.limit), indent=2))


if __name__ == "__main__":
    main()
