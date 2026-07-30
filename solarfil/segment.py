from __future__ import annotations

from collections import deque

import numpy as np


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


def _components(binary: np.ndarray, min_area: int) -> list[np.ndarray]:
    seen = np.zeros(binary.shape, dtype=bool)
    components: list[np.ndarray] = []
    height, width = binary.shape
    for y, x in zip(*np.nonzero(binary)):
        if seen[y, x]:
            continue
        queue, pixels = deque([(y, x)]), []
        seen[y, x] = True
        while queue:
            py, px = queue.popleft()
            pixels.append((py, px))
            for ny, nx in ((py - 1, px), (py + 1, px), (py, px - 1), (py, px + 1)):
                if 0 <= ny < height and 0 <= nx < width and binary[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    queue.append((ny, nx))
        if len(pixels) >= min_area:
            mask = np.zeros(binary.shape, dtype=np.uint8)
            ys, xs = zip(*pixels)
            mask[ys, xs] = 1
            components.append(mask)
    return components


def segment_instances(
    image: np.ndarray,
    darkness_quantile: float = 0.08,
    min_area: int = 24,
    max_normalized_intensity: float = 0.78,
) -> list[np.ndarray]:
    normalized, disk = radial_normalize(image)
    threshold = min(
        float(np.quantile(normalized[disk], darkness_quantile)),
        max_normalized_intensity,
    )
    candidates = (normalized <= threshold) & disk
    return _components(candidates, min_area)
