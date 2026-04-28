from __future__ import annotations

from collections.abc import Sequence
from os import PathLike
from pathlib import Path
import tempfile

import numpy as np
import tifffile

from utils.wsi_tiling import iter_level_tiles
from utils.wsi_quant import quantize_uint16


def build_ome_metadata(
    channel_names: Sequence[str],
    level: int,
) -> dict[str, object]:
    """Build the minimal OME metadata that ``tifffile`` preserves for ``CYX`` output."""

    if isinstance(channel_names, (str, bytes)) or not isinstance(channel_names, Sequence):
        raise ValueError("channel_names must be a non-string sequence of strings")

    names = list(channel_names)
    if not names:
        raise ValueError("channel_names must not be empty")

    if any(not isinstance(name, str) for name in names):
        raise ValueError("channel_names must contain only strings")

    if not isinstance(level, int) or isinstance(level, bool) or level < 0:
        raise ValueError("level must be a non-negative integer")

    return {
        "axes": "CYX",
        "Name": f"level{level}",
        "Channel": {"Name": names},
    }


def write_level_ome_tiff(
    raw_npy_path: str | PathLike[str],
    ome_path: str | PathLike[str],
    channel_names: Sequence[str],
    level: int,
    channel_chunk_size: int = 1,
    quant_min: float | None = None,
    quant_max: float | None = None,
    quant_mode: str = "global",
    tile_size: int = 224,
    ome_dtype: str | None = None,
) -> Path:
    """Write a single-level ``CYX`` OME-TIFF from raw ``HWC`` float predictions."""

    if not isinstance(channel_chunk_size, int) or isinstance(channel_chunk_size, bool) or channel_chunk_size <= 0:
        raise ValueError("channel_chunk_size must be a positive integer")
    if (quant_min is None) != (quant_max is None):
        raise ValueError("quant_min and quant_max must be provided together")
    if quant_min is not None:
        if not np.isfinite(quant_min) or not np.isfinite(quant_max):
            raise ValueError("quant_min and quant_max must be finite when provided")
        if quant_max <= quant_min:
            raise ValueError("quant_max must be greater than quant_min")
    if quant_mode not in {"global", "tile", "none"}:
        raise ValueError("quant_mode must be one of 'global', 'tile', or 'none'")
    if not isinstance(tile_size, int) or isinstance(tile_size, bool) or tile_size <= 0:
        raise ValueError("tile_size must be a positive integer")
    if ome_dtype is not None and ome_dtype not in {"uint16", "float32"}:
        raise ValueError("ome_dtype must be either 'uint16', 'float32', or None")

    raw_path = Path(raw_npy_path)
    output_path = Path(ome_path)

    raw_predictions = np.load(raw_path, mmap_mode="r")
    channel_count, _ = _validate_raw_predictions(raw_predictions, channel_names)
    resolved_ome_dtype = _resolve_ome_dtype(quant_mode=quant_mode, ome_dtype=ome_dtype)
    metadata = build_ome_metadata(
        channel_names=channel_names,
        level=level,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    staging_path: Path | None = None
    staged_predictions: np.memmap | None = None

    try:
        with tempfile.NamedTemporaryFile(
            suffix=".npy",
            prefix="ome_quant_",
            dir=output_path.parent,
            delete=False,
        ) as handle:
            staging_path = Path(handle.name)

        stage_dtype = np.uint16 if resolved_ome_dtype == "uint16" else np.float32
        # Stage the output ``CYX`` array out-of-core so we never materialize the
        # full slide prediction in memory while converting or reordering channels.
        staged_predictions = np.lib.format.open_memmap(
            staging_path,
            mode="w+",
            dtype=stage_dtype,
            shape=(channel_count, raw_predictions.shape[0], raw_predictions.shape[1]),
        )
        staged_predictions[...] = 0

        for start_idx in range(0, channel_count, channel_chunk_size):
            stop_idx = min(start_idx + channel_chunk_size, channel_count)
            if quant_mode == "global":
                _quantize_channel_chunk(
                    raw_predictions=raw_predictions,
                    quantized=staged_predictions,
                    start_idx=start_idx,
                    stop_idx=stop_idx,
                    quant_min=quant_min,
                    quant_max=quant_max,
                )
            elif quant_mode == "tile":
                _quantize_channel_chunk_by_tile(
                    raw_predictions=raw_predictions,
                    quantized=staged_predictions,
                    start_idx=start_idx,
                    stop_idx=stop_idx,
                    tile_size=tile_size,
                )
            else:
                _copy_channel_chunk_float32(
                    raw_predictions=raw_predictions,
                    staged_predictions=staged_predictions,
                    start_idx=start_idx,
                    stop_idx=stop_idx,
                )
            staged_predictions.flush()

        tifffile.imwrite(
            output_path,
            staged_predictions,
            metadata=metadata,
            ome=True,
            photometric="minisblack",
        )
    finally:
        if staged_predictions is not None:
            staged_predictions.flush()
            del staged_predictions
        if staging_path is not None and staging_path.exists():
            staging_path.unlink()

    return output_path


def _resolve_ome_dtype(quant_mode: str, ome_dtype: str | None) -> str:
    if quant_mode == "none":
        expected_dtype = "float32"
    else:
        expected_dtype = "uint16"

    if ome_dtype is None:
        return expected_dtype
    if ome_dtype != expected_dtype:
        raise ValueError(
            f"ome_dtype {ome_dtype!r} is incompatible with quant_mode {quant_mode!r}; "
            f"expected {expected_dtype!r}"
        )
    return ome_dtype


def _validate_raw_predictions(
    raw_predictions: np.ndarray,
    channel_names: Sequence[str],
) -> tuple[int, tuple[int, int]]:
    if raw_predictions.ndim != 3:
        raise ValueError("raw predictions must have shape (height, width, channels)")

    height, width, channels = raw_predictions.shape
    if channels != len(channel_names):
        raise ValueError("raw prediction channels must match channel_names length")

    if height <= 0 or width <= 0:
        raise ValueError("raw prediction height and width must be positive")

    return channels, (height, width)


def _quantize_channel_chunk(
    raw_predictions: np.ndarray,
    quantized: np.memmap,
    start_idx: int,
    stop_idx: int,
    quant_min: float | None,
    quant_max: float | None,
) -> None:
    for channel_idx in range(start_idx, stop_idx):
        channel_view = raw_predictions[:, :, channel_idx]
        if np.any(np.isinf(channel_view)):
            raise ValueError("raw predictions must not contain infinite values")
        if quant_min is None:
            channel_range = _finite_min_max(channel_view)

            if channel_range is None:
                quantized[channel_idx, :, :] = np.uint16(0)
                continue

            channel_min, channel_max = channel_range

            if channel_max == channel_min:
                quantized[channel_idx, :, :] = np.uint16(0)
                continue
        else:
            channel_min = quant_min
            channel_max = quant_max

        quantized[channel_idx, :, :] = quantize_uint16(
            np.nan_to_num(channel_view, nan=channel_min),
            quant_min=channel_min,
            quant_max=channel_max,
        )


def _quantize_channel_chunk_by_tile(
    raw_predictions: np.ndarray,
    quantized: np.memmap,
    start_idx: int,
    stop_idx: int,
    tile_size: int,
) -> None:
    level_size = (raw_predictions.shape[1], raw_predictions.shape[0])

    for channel_idx in range(start_idx, stop_idx):
        channel_view = raw_predictions[:, :, channel_idx]

        for x, y, width, height in iter_level_tiles(level_size, tile_size=tile_size, stride=tile_size):
            tile_view = channel_view[y : y + height, x : x + width]
            tile_range = _finite_min_max(tile_view)

            if tile_range is None:
                quantized[channel_idx, y : y + height, x : x + width] = np.uint16(0)
                continue

            tile_min, tile_max = tile_range

            if tile_max == tile_min:
                quantized[channel_idx, y : y + height, x : x + width] = np.uint16(0)
                continue

            quantized[channel_idx, y : y + height, x : x + width] = quantize_uint16(
                np.nan_to_num(tile_view, nan=tile_min),
                quant_min=tile_min,
                quant_max=tile_max,
            )


def _copy_channel_chunk_float32(
    raw_predictions: np.ndarray,
    staged_predictions: np.memmap,
    start_idx: int,
    stop_idx: int,
) -> None:
    for channel_idx in range(start_idx, stop_idx):
        channel_view = raw_predictions[:, :, channel_idx]
        channel_range = _finite_min_max(channel_view)

        if channel_range is None:
            staged_predictions[channel_idx, :, :] = np.float32(0.0)
            continue

        channel_min, _ = channel_range
        staged_predictions[channel_idx, :, :] = np.nan_to_num(channel_view, nan=channel_min).astype(
            np.float32,
            copy=False,
        )


def _finite_min_max(values: np.ndarray) -> tuple[float, float] | None:
    if np.any(np.isinf(values)):
        raise ValueError("raw predictions must not contain infinite values")

    if np.all(np.isnan(values)):
        return None

    return float(np.nanmin(values)), float(np.nanmax(values))
