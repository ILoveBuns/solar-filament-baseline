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


def label_overlap_metrics(
    labels: np.ndarray,
    label_ids: list[int],
    truths: list[np.ndarray],
    overlap_threshold: float = 0.1,
) -> tuple[list[float], dict[str, int]]:
    """Return greedy Dice scores and instance-fragmentation diagnostics.

    The overlap graph is diagnostic rather than a reimplementation of the
    hidden competition metric. An edge means the pair has Dice at least
    ``overlap_threshold``.
    """
    sizes = np.bincount(labels.ravel())
    candidates: list[tuple[float, int, int]] = []
    truth_degrees = np.zeros(len(truths), dtype=np.int32)
    prediction_degrees = {label_id: 0 for label_id in label_ids}
    # One label histogram per truth yields every pairwise intersection without
    # rescanning the full-resolution image for every predicted component.
    for truth_id, truth in enumerate(truths):
        truth_bool = truth.astype(bool, copy=False)
        intersections = np.bincount(labels[truth_bool], minlength=len(sizes))
        truth_area = int(np.count_nonzero(truth_bool))
        for label_id in label_ids:
            area = int(sizes[label_id])
            intersection = int(intersections[label_id])
            score = 2 * intersection / (area + truth_area)
            candidates.append((score, label_id, truth_id))
            if score >= overlap_threshold:
                truth_degrees[truth_id] += 1
                prediction_degrees[label_id] += 1
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
    diagnostics = {
        "unmatched_truths": len(truths) - len(used_truths),
        "unmatched_predictions": len(label_ids) - len(used_labels),
        "one_to_many_truths": int(np.count_nonzero(truth_degrees > 1)),
        "many_to_one_predictions": sum(degree > 1 for degree in prediction_degrees.values()),
    }
    return scores, diagnostics


def matched_label_dice(labels: np.ndarray, label_ids: list[int], truths: list[np.ndarray]) -> list[float]:
    return label_overlap_metrics(labels, label_ids, truths)[0]


def evaluate_dataset(
    root: Path,
    limit: int | None = None,
    darkness_quantile: float = 0.25,
    min_area: int = 24,
    max_normalized_intensity: float = 0.95,
    max_instances: int = 64,
    overlap_threshold: float = 0.1,
) -> dict[str, float | int]:
    annotation_path = root / "train/MAGFiLO_1.0_Annotations_kaggle2026_train.json"
    data = json.loads(annotation_path.read_text())
    images_by_id = {item["id"]: item for item in data["images"]}
    annotations_by_file: dict[str, list[dict]] = defaultdict(list)
    for annotation in data["annotations"]:
        filename = images_by_id[annotation["image_id"]]["file_name"]
        annotations_by_file[filename].append(annotation)

    scores: list[float] = []
    predicted_count = truth_count = 0
    diagnostic_totals: dict[str, int] = defaultdict(int)
    files = sorted(annotations_by_file)[:limit]
    for filename in files:
        image = np.asarray(Image.open(root / "train/train_images" / filename).convert("L"))
        truths = [annotation_mask(x, *image.shape) for x in annotations_by_file[filename]]
        labels, label_ids = segment_labels(
            image,
            darkness_quantile=darkness_quantile,
            min_area=min_area,
            max_normalized_intensity=max_normalized_intensity,
            max_instances=max_instances,
        )
        image_scores, diagnostics = label_overlap_metrics(
            labels, label_ids, truths, overlap_threshold=overlap_threshold
        )
        scores.extend(image_scores)
        for name, value in diagnostics.items():
            diagnostic_totals[name] += value
        predicted_count += len(label_ids)
        truth_count += len(truths)
        del image, truths, labels, label_ids
        gc.collect()
    result: dict[str, float | int] = {
        "images": len(files),
        "instances_truth": truth_count,
        "instances_predicted": predicted_count,
        "mean_matched_dice": float(np.mean(scores)) if scores else 0.0,
        "prediction_truth_ratio": predicted_count / truth_count if truth_count else 0.0,
    }
    result.update(diagnostic_totals)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--darkness-quantile", type=float, default=0.25)
    parser.add_argument("--min-area", type=int, default=24)
    parser.add_argument("--max-normalized-intensity", type=float, default=0.95)
    parser.add_argument("--max-instances", type=int, default=64)
    parser.add_argument("--overlap-threshold", type=float, default=0.1)
    args = parser.parse_args()
    print(json.dumps(evaluate_dataset(
        args.root,
        args.limit,
        darkness_quantile=args.darkness_quantile,
        min_area=args.min_area,
        max_normalized_intensity=args.max_normalized_intensity,
        max_instances=args.max_instances,
        overlap_threshold=args.overlap_threshold,
    ), indent=2))


if __name__ == "__main__":
    main()
