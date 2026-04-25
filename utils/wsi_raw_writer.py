from __future__ import annotations

from os import PathLike

import numpy as np
from numpy.lib.format import open_memmap
from numpy.typing import DTypeLike


class RawPredictionWriter:
    """Write slide-sized prediction arrays into a memmap-backed ``.npy`` file."""

    def __init__(
        self,
        path: str | PathLike[str],
        shape: tuple[int, ...],
        dtype: DTypeLike = np.float32,
    ) -> None:
        if len(shape) < 2:
            raise ValueError("shape must include at least height and width")

        if any(not isinstance(dim, int) or isinstance(dim, bool) or dim <= 0 for dim in shape):
            raise ValueError("shape dimensions must be positive integers")

        self._shape = shape
        self._dtype = np.dtype(dtype)
        self._array: np.memmap | None = open_memmap(
            path,
            mode="w+",
            dtype=self._dtype,
            shape=shape,
        )
        self._array[...] = 0

    def write_tile(self, x: int, y: int, tile: np.ndarray) -> None:
        if not isinstance(x, int) or isinstance(x, bool) or x < 0:
            raise ValueError("x must be a non-negative integer")

        if not isinstance(y, int) or isinstance(y, bool) or y < 0:
            raise ValueError("y must be a non-negative integer")

        array = self._require_open()
        tile_array = np.asarray(tile, dtype=self._dtype)
        if tile_array.ndim != len(self._shape):
            raise ValueError("tile rank must match writer shape")

        if tile_array.shape[2:] != self._shape[2:]:
            raise ValueError("tile channel dimensions must match writer shape")

        tile_height, tile_width = tile_array.shape[:2]
        end_y = y + tile_height
        end_x = x + tile_width
        if end_y > self._shape[0] or end_x > self._shape[1]:
            raise ValueError("tile write exceeds writer bounds")

        # Writer shape is ``(height, width, channels...)`` and x/y are level-0 origins.
        array[y:end_y, x:end_x, ...] = tile_array

    def close(self) -> None:
        if self._array is None:
            return

        self._array.flush()
        self._array = None

    def _require_open(self) -> np.memmap:
        if self._array is None:
            raise ValueError("writer is closed")

        return self._array
