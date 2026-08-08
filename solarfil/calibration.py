"""Threshold calibration helpers for instance-segmentation models."""

from __future__ import annotations

from itertools import product
import math

import numpy as np

from .coco_data import greedy_scores_from_matrix


def calibration_selection_score(
    mean_matched_dice: float,
    prediction_truth_ratio: float,
    count_penalty_weight: float = 0.05,
) -> float:
    """Balance matched Dice against multiplicative instance-count error.

    Matched Dice alone does not penalize unmatched predictions, so a fragmented
    operating point can look artificially strong. The log-ratio penalty treats
    two-times too many and two-times too few instances symmetrically.
    """
    if prediction_truth_ratio <= 0:
        return float("-inf")
    return mean_matched_dice - count_penalty_weight * abs(math.log(prediction_truth_ratio))


def panoptic_quality(
    matched_iou_sum: float,
    true_positives: int,
    false_positives: int,
    false_negatives: int,
) -> float:
    """Compute the official PQ aggregation from matched IoU and instance counts."""
    denominator = true_positives + 0.5 * false_positives + 0.5 * false_negatives
    return matched_iou_sum / denominator if denominator else 0.0


def match_panoptic_iou(
    iou_matrix: np.ndarray,
    match_threshold: float = 0.5,
) -> tuple[float, int]:
    """Greedily match a small prediction/truth IoU matrix for PQ aggregation."""
    candidates = sorted(
        (
            (float(iou_matrix[prediction_id, truth_id]), prediction_id, truth_id)
            for prediction_id in range(iou_matrix.shape[0])
            for truth_id in range(iou_matrix.shape[1])
            if iou_matrix[prediction_id, truth_id] > match_threshold
        ),
        reverse=True,
    )
    used_predictions: set[int] = set()
    used_truths: set[int] = set()
    matched_iou_sum = 0.0
    for iou, prediction_id, truth_id in candidates:
        if prediction_id in used_predictions or truth_id in used_truths:
            continue
        used_predictions.add(prediction_id)
        used_truths.add(truth_id)
        matched_iou_sum += iou
    return matched_iou_sum, len(used_predictions)


def resolve_prediction_thresholds(checkpoint: dict, defaults) -> tuple[float, float, int]:
    """Prefer a validation-selected operating point stored in a checkpoint."""
    calibration = checkpoint.get("calibration") or {}
    return (
        float(calibration.get("score_threshold", defaults.score_threshold)),
        float(calibration.get("mask_threshold", defaults.mask_threshold)),
        int(calibration.get("min_area", defaults.min_area)),
    )


def score_instances(
    confidence: np.ndarray,
    probabilities: np.ndarray,
    truths: np.ndarray,
    score_threshold: float,
    mask_threshold: float,
    min_area: int = 1,
) -> dict[str, float | int]:
    """Score cached model outputs at one operating point.

    Unmatched truth instances receive zero, matching the repository's local
    validation convention.  Count ratio is retained because matched Dice alone
    can hide fragmentation and duplicate predictions.
    """
    keep = confidence >= score_threshold
    predictions = probabilities[keep] >= mask_threshold
    if len(predictions):
        predictions = predictions[predictions.reshape(len(predictions), -1).sum(1) >= min_area]

    matched_iou_sum = 0.0
    true_positives = 0
    if len(predictions) and len(truths):
        flat_predictions = predictions.reshape(len(predictions), -1)
        flat_truths = truths.astype(bool).reshape(len(truths), -1)
        # Avoid materializing a predictions × truths × pixels broadcast, which
        # can exceed a gigabyte for one 2048² image. Each temporary is bounded
        # to predictions × pixels instead.
        intersections = np.stack([
            np.logical_and(flat_predictions, truth).sum(1)
            for truth in flat_truths
        ], axis=1)
        denominators = flat_predictions.sum(1)[:, None] + flat_truths.sum(1)[None]
        matrix = 2 * intersections / np.maximum(denominators, 1)
        scores = greedy_scores_from_matrix(matrix, len(truths))

        unions = flat_predictions.sum(1)[:, None] + flat_truths.sum(1)[None] - intersections
        iou_matrix = intersections / np.maximum(unions, 1)
        matched_iou_sum, true_positives = match_panoptic_iou(iou_matrix)
    else:
        scores = [0.0] * len(truths)

    return {
        "dice_sum": float(sum(scores)),
        "truth_count": int(len(truths)),
        "prediction_count": int(len(predictions)),
        "matched_iou_sum": matched_iou_sum,
        "true_positives": true_positives,
        "false_positives": int(len(predictions) - true_positives),
        "false_negatives": int(len(truths) - true_positives),
    }


def sweep_thresholds(
    cached_outputs: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    score_thresholds: list[float],
    mask_thresholds: list[float],
    min_areas: list[int],
) -> list[dict[str, float | int]]:
    """Evaluate a threshold grid without repeating neural-network inference."""
    results = []
    for score_threshold, mask_threshold, min_area in product(
        score_thresholds, mask_thresholds, min_areas
    ):
        totals = {
            "dice_sum": 0.0,
            "truth_count": 0,
            "prediction_count": 0,
            "matched_iou_sum": 0.0,
            "true_positives": 0,
            "false_positives": 0,
            "false_negatives": 0,
        }
        for confidence, probabilities, truths in cached_outputs:
            metrics = score_instances(
                confidence, probabilities, truths, score_threshold, mask_threshold, min_area
            )
            for key in totals:
                totals[key] += metrics[key]
        truth_count = int(totals["truth_count"])
        mean_matched_dice = totals["dice_sum"] / truth_count if truth_count else 0.0
        prediction_truth_ratio = totals["prediction_count"] / truth_count if truth_count else 0.0
        pq = panoptic_quality(
            totals["matched_iou_sum"],
            int(totals["true_positives"]),
            int(totals["false_positives"]),
            int(totals["false_negatives"]),
        )
        results.append({
            "score_threshold": score_threshold,
            "mask_threshold": mask_threshold,
            "min_area": min_area,
            "mean_matched_dice": mean_matched_dice,
            "prediction_truth_ratio": prediction_truth_ratio,
            "panoptic_quality": pq,
            "true_positives": int(totals["true_positives"]),
            "false_positives": int(totals["false_positives"]),
            "false_negatives": int(totals["false_negatives"]),
            "selection_score": pq,
        })
    return sorted(
        results,
        key=lambda row: (
            -row["selection_score"],
            -row["mean_matched_dice"],
            abs(row["prediction_truth_ratio"] - 1.0),
        ),
    )
