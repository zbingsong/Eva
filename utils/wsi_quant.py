from __future__ import annotations

import numpy as np


def quantize_uint16(arr: np.ndarray, quant_min: float, quant_max: float) -> np.ndarray:
    """Linearly map a float array into the full ``uint16`` range."""

    if not np.isfinite(quant_min) or not np.isfinite(quant_max):
        raise ValueError("quant_min and quant_max must be finite")

    if quant_max <= quant_min:
        raise ValueError("quant_max must be greater than quant_min")

    array = np.asarray(arr)
    if not np.all(np.isfinite(array)):
        raise ValueError("arr must contain only finite values")

    clipped = np.clip(array.astype(np.float64, copy=False), quant_min, quant_max)
    normalized = (clipped - quant_min) / (quant_max - quant_min)
    scaled = np.rint(normalized * np.iinfo(np.uint16).max)
    return scaled.astype(np.uint16)
