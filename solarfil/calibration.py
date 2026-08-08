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

    if len(predictions) and len(truths):
        flat_predictions = predictions.reshape(len(predictions), -1)
        flat_truths = truths.astype(bool).reshape(len(truths), -1)
        intersections = np.logical_and(flat_predictions[:, None], flat_truths[None]).sum(2)
        denominators = flat_predictions.sum(1)[:, None] + flat_truths.sum(1)[None]
        matrix = 2 * intersections / np.maximum(denominators, 1)
        scores = greedy_scores_from_matrix(matrix, len(truths))
    else:
        scores = [0.0] * len(truths)

    return {
        "dice_sum": float(sum(scores)),
        "truth_count": int(len(truths)),
        "prediction_count": int(len(predictions)),
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
        totals = {"dice_sum": 0.0, "truth_count": 0, "prediction_count": 0}
        for confidence, probabilities, truths in cached_outputs:
            metrics = score_instances(
                confidence, probabilities, truths, score_threshold, mask_threshold, min_area
            )
            for key in totals:
                totals[key] += metrics[key]
        truth_count = int(totals["truth_count"])
        mean_matched_dice = totals["dice_sum"] / truth_count if truth_count else 0.0
        prediction_truth_ratio = totals["prediction_count"] / truth_count if truth_count else 0.0
        results.append({
            "score_threshold": score_threshold,
            "mask_threshold": mask_threshold,
            "min_area": min_area,
            "mean_matched_dice": mean_matched_dice,
            "prediction_truth_ratio": prediction_truth_ratio,
            "selection_score": calibration_selection_score(
                mean_matched_dice, prediction_truth_ratio
            ),
        })
    return sorted(
        results,
        key=lambda row: (
            -row["selection_score"],
            -row["mean_matched_dice"],
            abs(row["prediction_truth_ratio"] - 1.0),
        ),
    )
