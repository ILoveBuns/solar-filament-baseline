import numpy as np


def dice(prediction: np.ndarray, truth: np.ndarray) -> float:
    prediction, truth = prediction.astype(bool), truth.astype(bool)
    total = prediction.sum() + truth.sum()
    return float(2 * np.logical_and(prediction, truth).sum() / total) if total else 1.0

