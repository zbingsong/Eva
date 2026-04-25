import pytest

from utils.wsi_tiling import iter_level_tiles


def test_iter_level_tiles_yields_only_full_non_overlapping_windows() -> None:
    windows = list(iter_level_tiles((500, 500), tile_size=224, stride=224))

    assert windows == [
        (0, 0, 224, 224),
        (224, 0, 224, 224),
        (0, 224, 224, 224),
        (224, 224, 224, 224),
    ]


def test_iter_level_tiles_drops_partial_edge_windows() -> None:
    windows = list(iter_level_tiles((300, 224), tile_size=224, stride=128))

    assert windows == [(0, 0, 224, 224)]


@pytest.mark.parametrize(
    ("tile_size", "stride", "message"),
    [
        (0, 224, "tile_size must be a positive integer"),
        (-1, 224, "tile_size must be a positive integer"),
        (224, 0, "stride must be a positive integer"),
        (224, -1, "stride must be a positive integer"),
    ],
)
def test_iter_level_tiles_rejects_invalid_window_parameters(
    tile_size: int,
    stride: int,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        list(iter_level_tiles((500, 500), tile_size=tile_size, stride=stride))
