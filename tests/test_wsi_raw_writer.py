from __future__ import annotations

import numpy as np
import pytest

from utils.wsi_raw_writer import RawPredictionWriter


def test_raw_writer_persists_tile_into_memmap(tmp_path) -> None:
    path = tmp_path / "pred.npy"
    writer = RawPredictionWriter(path=path, shape=(448, 448, 2), dtype=np.float32)
    tile = np.ones((224, 224, 2), dtype=np.float32)

    writer.write_tile(224, 0, tile)
    writer.close()

    arr = np.load(path, mmap_mode="r")
    assert arr.shape == (448, 448, 2)
    assert arr.dtype == np.float32
    assert np.all(arr[0:224, 224:448] == 1.0)
    assert np.all(arr[0:224, 0:224] == 0.0)
    assert np.all(arr[224:, :] == 0.0)


def test_raw_writer_rejects_tile_shape_and_bounds_mismatches(tmp_path) -> None:
    path = tmp_path / "pred.npy"
    writer = RawPredictionWriter(path=path, shape=(448, 448, 2), dtype=np.float32)

    with pytest.raises(ValueError, match="tile channel dimensions must match writer shape"):
        writer.write_tile(0, 0, np.zeros((224, 224, 3), dtype=np.float32))

    with pytest.raises(ValueError, match="tile write exceeds writer bounds"):
        writer.write_tile(300, 0, np.zeros((224, 224, 2), dtype=np.float32))

    writer.close()


def test_raw_writer_rejects_write_after_close_before_tile_validation(tmp_path) -> None:
    path = tmp_path / "pred.npy"
    writer = RawPredictionWriter(path=path, shape=(448, 448, 2), dtype=np.float32)
    writer.close()

    with pytest.raises(ValueError, match="writer is closed"):
        writer.write_tile(0, 0, np.zeros((224, 224, 3), dtype=np.float32))
