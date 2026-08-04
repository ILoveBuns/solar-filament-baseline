from __future__ import annotations

import hashlib
from collections import defaultdict
from pathlib import Path

import numpy as np


def select_best_image_records(coco: dict) -> dict[str, dict]:
    """Choose one labeler record per physical image deterministically.

    MAGFiLO contains repeated ``file_name`` records from multiple annotators.
    We retain the record with the most annotations, then the largest total
    annotated area, matching the public strong baseline while making the
    tie-break explicit and stable.
    """
    annotations_by_id: dict[int, list[dict]] = defaultdict(list)
    for annotation in coco["annotations"]:
        annotations_by_id[annotation["image_id"]].append(annotation)

    records_by_file: dict[str, list[dict]] = defaultdict(list)
    for image in coco["images"]:
        records_by_file[image["file_name"]].append(image)

    def quality(image: dict) -> tuple[int, float]:
        annotations = annotations_by_id[image["id"]]
        return (
            len(annotations),
            sum(float(item.get("area", 0.0)) for item in annotations),
        )

    return {
        filename: max(sorted(records, key=lambda image: str(image["id"])), key=quality)
        for filename, records in records_by_file.items()
    }


def observation_group(filename: str) -> tuple[str, str]:
    """Return (year, observatory) strata encoded in MAGFiLO filenames."""
    stem = Path(filename).stem
    if len(stem) < 6:
        raise ValueError(f"unexpected MAGFiLO filename: {filename}")
    return stem[:4], stem[-2:]


def stable_stratified_split(
    filenames: list[str], validation_fraction: float = 0.2, seed: int = 42
) -> tuple[list[str], list[str]]:
    """Split by stable hashes within year/site strata.

    This keeps rare years and observatories represented without depending on
    scikit-learn or input order. The same physical filename cannot cross the
    split even when multiple labeler records exist.
    """
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between zero and one")
    if len(set(filenames)) != len(filenames):
        raise ValueError("filenames must be unique before splitting")

    strata: dict[tuple[str, str], list[str]] = defaultdict(list)
    for filename in filenames:
        strata[observation_group(filename)].append(filename)

    train: list[str] = []
    validation: list[str] = []
    for group, members in sorted(strata.items()):
        ranked = sorted(
            members,
            key=lambda name: hashlib.sha256(f"{seed}:{group}:{name}".encode()).digest(),
        )
        validation_count = int(round(len(ranked) * validation_fraction))
        if len(ranked) > 1:
            validation_count = min(max(validation_count, 1), len(ranked) - 1)
        validation.extend(ranked[:validation_count])
        train.extend(ranked[validation_count:])
    return sorted(train), sorted(validation)


def greedy_scores_from_matrix(score_matrix: np.ndarray, truth_count: int) -> list[float]:
    """Greedy one-to-one scores from a precomputed prediction/truth matrix."""
    candidates = sorted(
        (
            (float(score_matrix[prediction_id, truth_id]), prediction_id, truth_id)
            for prediction_id in range(score_matrix.shape[0])
            for truth_id in range(score_matrix.shape[1])
        ),
        reverse=True,
    )
    used_predictions: set[int] = set()
    used_truths: set[int] = set()
    scores: list[float] = []
    for score, prediction_id, truth_id in candidates:
        if prediction_id not in used_predictions and truth_id not in used_truths:
            used_predictions.add(prediction_id)
            used_truths.add(truth_id)
            scores.append(score)
    scores.extend([0.0] * (truth_count - len(used_truths)))
    return scores
