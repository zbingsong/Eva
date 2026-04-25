from __future__ import annotations

import numpy as np

WHITE_PIXEL_FRACTION = 0.999


def is_near_white_tile(tile: np.ndarray, threshold: float) -> bool:
    """Return whether a normalized tile is mostly white at ``threshold``.

    Parameters
    ----------
    tile
        Tile data with shape ``(height, width, channels)``. Values must already
        be normalized to the closed interval ``[0, 1]``.
    threshold
        Minimum value each channel of a pixel must satisfy for that pixel to
        count as white.

    Returns
    -------
    bool
        ``True`` when at least 99.9% of pixels are white across all channels.

    Raises
    ------
    ValueError
        If any tile value falls outside ``[0, 1]``.
    """

    if np.any((tile < 0) | (tile > 1)):
        raise ValueError("tile values must already be normalized to [0, 1]")

    # A pixel counts as white only if every RGB channel clears the threshold.
    white_pixels = np.all(tile >= threshold, axis=-1)
    return bool(np.mean(white_pixels) >= WHITE_PIXEL_FRACTION)
