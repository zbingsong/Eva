import pytest
import numpy as np

from utils.wsi_background import is_near_white_tile


def test_is_near_white_tile_detects_blank_rgb() -> None:
    tile = np.ones((224, 224, 3), dtype=np.float32)

    assert is_near_white_tile(tile, threshold=0.95) is True


def test_is_near_white_tile_allows_up_to_fifty_non_white_pixels() -> None:
    tile = np.ones((224, 224, 3), dtype=np.float32)
    tile[:50, 0, :] = np.float32(199.0 / 255.0)

    assert is_near_white_tile(tile, threshold=np.float32(200.0 / 255.0)) is True


def test_is_near_white_tile_rejects_tile_once_non_white_pixel_budget_is_exceeded() -> None:
    tile = np.ones((224, 224, 3), dtype=np.float32)
    tile[:51, 0, :] = np.float32(199.0 / 255.0)

    assert is_near_white_tile(tile, threshold=np.float32(200.0 / 255.0)) is False


def test_is_near_white_tile_requires_all_channels_to_clear_threshold() -> None:
    tile = np.ones((224, 224, 3), dtype=np.float32)
    tile[:51, 0, :] = 1.0
    tile[:51, 0, 2] = np.float32(199.0 / 255.0)

    assert is_near_white_tile(tile, threshold=np.float32(200.0 / 255.0)) is False


@pytest.mark.parametrize(
    ("tile", "message"),
    [
        (
            np.full((4, 4, 3), 255, dtype=np.uint8),
            "tile values must already be normalized to \\[0, 1\\]",
        ),
        (
            np.full((4, 4, 3), -0.1, dtype=np.float32),
            "tile values must already be normalized to \\[0, 1\\]",
        ),
    ],
)
def test_is_near_white_tile_rejects_tiles_outside_normalized_unit_range(
    tile: np.ndarray,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        is_near_white_tile(tile, threshold=0.95)
