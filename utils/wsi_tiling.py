from __future__ import annotations

from collections.abc import Iterator


def iter_level_tiles(
    level_size: tuple[int, int],
    tile_size: int,
    stride: int,
) -> Iterator[tuple[int, int, int, int]]:
    """Yield full tile windows for a level in row-major order.

    Parameters
    ----------
    level_size
        Level dimensions as ``(width, height)``.
    tile_size
        Square tile edge length in pixels.
    stride
        Step size between tile origins in pixels.

    Yields
    ------
    tuple[int, int, int, int]
        Tile windows as ``(x, y, width, height)`` for full windows only.

    Raises
    ------
    ValueError
        If ``tile_size`` or ``stride`` is not a positive integer.
    """

    if not isinstance(tile_size, int) or isinstance(tile_size, bool) or tile_size <= 0:
        raise ValueError("tile_size must be a positive integer")

    if not isinstance(stride, int) or isinstance(stride, bool) or stride <= 0:
        raise ValueError("stride must be a positive integer")

    level_width, level_height = level_size
    max_x = level_width - tile_size
    max_y = level_height - tile_size

    if max_x < 0 or max_y < 0:
        return

    for y in range(0, max_y + 1, stride):
        for x in range(0, max_x + 1, stride):
            yield (x, y, tile_size, tile_size)
