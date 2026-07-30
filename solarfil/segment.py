from __future__ import annotations

import numpy as np
from scipy import ndimage


def radial_normalize(image: np.ndarray, bins: int = 96) -> tuple[np.ndarray, np.ndarray]:
    """Remove center-to-limb brightness variation using robust radial medians."""
    height, width = image.shape
    yy, xx = np.indices(image.shape)
    cy, cx = (height - 1) / 2, (width - 1) / 2
    radius = np.hypot(xx - cx, yy - cy)
    disk_radius = min(height, width) * 0.48
    disk = radius <= disk_radius
    indices = np.minimum((radius / disk_radius * bins).astype(int), bins - 1)
    profile = np.ones(bins, dtype=np.float32)
    for index in range(bins):
        values = image[(indices == index) & disk]
        if values.size:
            # Upper quartile resists dark filament contamination of the limb profile.
            profile[index] = max(float(np.quantile(values, 0.75)), 1e-6)
    normalized = image.astype(np.float32) / profile[indices]
    return normalized, disk


def _components(binary: np.ndarray, min_area: int, max_instances: int) -> tuple[np.ndarray, list[int]]:
    labels, count = ndimage.label(binary)
    sizes = np.bincount(labels.ravel())
    keep = sorted(
        (index for index in range(1, count + 1) if sizes[index] >= min_area),
        key=lambda index: int(sizes[index]),
        reverse=True,
    )[:max_instances]
    return labels, keep


def segment_instances(
    image: np.ndarray,
    darkness_quantile: float = 0.25,
    min_area: int = 24,
    max_normalized_intensity: float = 0.95,
    max_instances: int = 64,
) -> list[np.ndarray]:
    labels, keep = segment_labels(
        image,
        darkness_quantile=darkness_quantile,
        min_area=min_area,
        max_normalized_intensity=max_normalized_intensity,
        max_instances=max_instances,
    )
    return [(labels == index).astype(np.uint8) for index in keep]


def segment_labels(
    image: np.ndarray,
    darkness_quantile: float = 0.25,
    min_area: int = 24,
    max_normalized_intensity: float = 0.95,
    max_instances: int = 64,
) -> tuple[np.ndarray, list[int]]:
    """Memory-efficient variant returning one label map and selected IDs."""
    normalized, disk = radial_normalize(image)
    threshold = min(
        float(np.quantile(normalized[disk], darkness_quantile)),
        max_normalized_intensity,
    )
    candidates = (normalized <= threshold) & disk
    return _components(candidates, min_area, max_instances)
