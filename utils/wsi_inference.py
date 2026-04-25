from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import TypedDict

import numpy as np
import torch

from utils.wsi_background import is_near_white_tile
from utils.wsi_eva import VirtualStainInputs, build_virtual_stain_inputs
from utils.wsi_ome import write_level_ome_tiff
from utils.wsi_raw_writer import RawPredictionWriter
from utils.wsi_tiling import iter_level_tiles


class EvaModelBatch(TypedDict):
    input: torch.Tensor
    marker_in: list[list[str]]
    marker_out: list[list[str]]
    infer_mask: torch.Tensor


@dataclass(frozen=True)
class LevelInferenceResult:
    level: int
    level_shape: tuple[int, int]
    level_output_dir: Path
    raw_npy_path: Path
    ome_tiff_path: Path
    total_tiles: int
    skipped_tiles: int
    predicted_tiles: int
    batches_run: int


@dataclass(frozen=True)
class _PendingTile:
    x: int
    y: int
    payload: VirtualStainInputs


ReadTileFn = Callable[[object, int, int, int, int, int], np.ndarray]
RunModelFn = Callable[[object, EvaModelBatch], np.ndarray | torch.Tensor]
BuildInputsFn = Callable[[np.ndarray, Sequence[str], int], VirtualStainInputs]
BackgroundFn = Callable[[np.ndarray, float], bool]
WriteOmeFn = Callable[
    [
        str | PathLike[str],
        str | PathLike[str],
        Sequence[str],
        int,
        int,
        float | None,
        float | None,
        str,
        int,
        str | None,
    ],
    Path,
]


def run_level_inference(
    slide: object,
    level: int,
    biomarkers: Sequence[str],
    model: object,
    output_dir: str | PathLike[str],
    tile_size: int,
    stride: int,
    white_threshold: float,
    batch_size: int = 1,
    patch_size: int = 8,
    device: str | torch.device | None = None,
    read_tile_fn: ReadTileFn | None = None,
    run_model_fn: RunModelFn | None = None,
    build_inputs_fn: BuildInputsFn = build_virtual_stain_inputs,
    background_fn: BackgroundFn = is_near_white_tile,
    raw_writer_factory: Callable[..., RawPredictionWriter] = RawPredictionWriter,
    write_ome_fn: WriteOmeFn = write_level_ome_tiff,
    ome_channel_chunk_size: int = 1,
    quant_min: float | None = None,
    quant_max: float | None = None,
    ome_quant_mode: str = "global",
    ome_dtype: str | None = None,
) -> LevelInferenceResult:
    """Run tile-wise inference for a single slide level.

    Parameters
    ----------
    slide
        Slide-like object with ``level_dimensions`` metadata. Reads are delegated
        through ``read_tile_fn`` so this function stays independent of OpenSlide.
    level
        Slide pyramid level to process.
    biomarkers
        Biomarker names to predict for each retained tile.
    model
        Model-like object consumed by ``run_model_fn``.
    output_dir
        Root directory where level-specific outputs are written.
    tile_size
        Square tile edge length in level pixels.
    stride
        Step size between tile origins in level pixels.
    white_threshold
        Background threshold passed to ``background_fn``.
    batch_size
        Number of retained tiles to run per model call.
    patch_size
        Eva patch size used when building inference payloads.
    device
        Optional torch device used by the default model adapter.
    read_tile_fn
        Optional override for slide reads with signature
        ``(slide, level, x, y, width, height) -> normalized_rgb_tile``.
    run_model_fn
        Optional override for model execution with signature
        ``(model, batch_payload) -> predictions``.
    build_inputs_fn
        Function that converts a normalized RGB tile into Eva inputs.
    background_fn
        Function that decides whether a normalized tile should be skipped.
    raw_writer_factory
        Factory returning a memmap-backed writer compatible with
        :class:`utils.wsi_raw_writer.RawPredictionWriter`.
    write_ome_fn
        Function that converts the raw memmap prediction into a single-level
        OME-TIFF with signature
        ``(raw_npy_path, ome_path, channel_names, level, channel_chunk_size)``.
    ome_channel_chunk_size
        Number of channels to quantize per chunk while writing the OME-TIFF.
    quant_min, quant_max
        Optional fixed quantization bounds shared across all output channels. If
        omitted, the OME writer auto-scales each channel independently.
    ome_quant_mode
        OME quantization mode. ``"global"`` scales each biomarker over the full
        level image, ``"tile"`` scales each full tile independently, and
        ``"none"`` writes raw float32 predictions directly to OME-TIFF.
    ome_dtype
        Optional OME-TIFF pixel type override. When omitted, quantized modes use
        ``uint16`` and ``"none"`` uses ``float32``.

    Returns
    -------
    LevelInferenceResult
        Output paths and simple counters for the processed level.
    """

    if not isinstance(level, int) or isinstance(level, bool) or level < 0:
        raise ValueError("level must be a non-negative integer")

    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    if not isinstance(tile_size, int) or isinstance(tile_size, bool) or tile_size <= 0:
        raise ValueError("tile_size must be a positive integer")
    if not isinstance(stride, int) or isinstance(stride, bool) or stride <= 0:
        raise ValueError("stride must be a positive integer")
    if stride < tile_size:
        raise ValueError("stride must be greater than or equal to tile_size")

    if not 0.0 <= white_threshold <= 1.0:
        raise ValueError("white_threshold must be within [0, 1]")

    if (
        not isinstance(ome_channel_chunk_size, int)
        or isinstance(ome_channel_chunk_size, bool)
        or ome_channel_chunk_size <= 0
    ):
        raise ValueError("ome_channel_chunk_size must be a positive integer")
    if (quant_min is None) != (quant_max is None):
        raise ValueError("quant_min and quant_max must be provided together")
    if quant_min is not None:
        if not np.isfinite(quant_min) or not np.isfinite(quant_max):
            raise ValueError("quant_min and quant_max must be finite when provided")
        if quant_max <= quant_min:
            raise ValueError("quant_max must be greater than quant_min")
    if ome_quant_mode not in {"global", "tile", "none"}:
        raise ValueError("ome_quant_mode must be one of 'global', 'tile', or 'none'")

    biomarker_names = _materialize_biomarkers(biomarkers)
    level_shape = _get_level_shape(slide, level)
    level_width, level_height = level_shape

    if level_width < tile_size or level_height < tile_size:
        raise ValueError("tile_size must fit within the selected slide level dimensions")

    level_output_dir = Path(output_dir) / f"level_{level}"
    level_output_dir.mkdir(parents=True, exist_ok=True)
    raw_npy_path = level_output_dir / "predictions.npy"
    ome_tiff_path = level_output_dir / "predictions.ome.tiff"

    tile_reader = read_tile_fn or _default_read_tile
    model_runner = run_model_fn or _default_run_model

    writer = raw_writer_factory(
        path=raw_npy_path,
        shape=(level_height, level_width, len(biomarker_names)),
        dtype=np.float32,
    )

    total_tiles = 0
    skipped_tiles = 0
    predicted_tiles = 0
    batches_run = 0
    pending_tiles: list[_PendingTile] = []

    try:
        for x, y, width, height in iter_level_tiles(level_shape, tile_size=tile_size, stride=stride):
            total_tiles += 1

            tile_rgb = tile_reader(slide, level, x, y, width, height)
            if background_fn(tile_rgb, white_threshold):
                skipped_tiles += 1
                continue

            payload = build_inputs_fn(tile_rgb, biomarker_names, patch_size)
            pending_tiles.append(_PendingTile(x=x, y=y, payload=payload))

            if len(pending_tiles) == batch_size:
                batches_run += 1
                predicted_tiles += _flush_pending_tiles(
                    pending_tiles=pending_tiles,
                    writer=writer,
                    model=model,
                    run_model_fn=model_runner,
                    expected_channels=len(biomarker_names),
                    device=device,
                )

        if pending_tiles:
            batches_run += 1
            predicted_tiles += _flush_pending_tiles(
                pending_tiles=pending_tiles,
                writer=writer,
                model=model,
                run_model_fn=model_runner,
                expected_channels=len(biomarker_names),
                device=device,
            )
    finally:
        writer.close()

    ome_tiff_path = write_ome_fn(
        raw_npy_path,
        ome_tiff_path,
        biomarker_names,
        level,
        ome_channel_chunk_size,
        quant_min,
        quant_max,
        ome_quant_mode,
        tile_size,
        ome_dtype,
    )

    return LevelInferenceResult(
        level=level,
        level_shape=level_shape,
        level_output_dir=level_output_dir,
        raw_npy_path=raw_npy_path,
        ome_tiff_path=ome_tiff_path,
        total_tiles=total_tiles,
        skipped_tiles=skipped_tiles,
        predicted_tiles=predicted_tiles,
        batches_run=batches_run,
    )


def _flush_pending_tiles(
    pending_tiles: list[_PendingTile],
    writer: RawPredictionWriter,
    model: object,
    run_model_fn: RunModelFn,
    expected_channels: int,
    device: str | torch.device | None,
) -> int:
    batch = _build_model_batch(pending_tiles, device=device)
    predictions = _coerce_prediction_batch(
        run_model_fn(model, batch),
        batch_size=len(pending_tiles),
        tile_shape=batch["input"].shape[1:3],
        channels=expected_channels,
    )

    for tile_state, tile_prediction in zip(pending_tiles, predictions, strict=True):
        writer.write_tile(tile_state.x, tile_state.y, tile_prediction)

    written_tiles = len(pending_tiles)
    pending_tiles.clear()
    return written_tiles


def _build_model_batch(
    pending_tiles: list[_PendingTile],
    device: str | torch.device | None,
) -> EvaModelBatch:
    input_np = np.stack([tile.payload["input"] for tile in pending_tiles], axis=0)
    input_tensor = torch.from_numpy(input_np).float()
    infer_mask = _coalesce_infer_mask(pending_tiles).float()

    if device is not None:
        device_obj = torch.device(device)
        input_tensor = input_tensor.to(device_obj)
        infer_mask = infer_mask.to(device_obj)

    return {
        # Eva expects NHWC tensors here; its public wrapper permutes internally.
        "input": input_tensor,
        "marker_in": [tile.payload["marker_in"] for tile in pending_tiles],
        "marker_out": [_extract_marker_out_group(tile.payload["marker_out"]) for tile in pending_tiles],
        # Eva currently expects a shared ``(channels, patches)`` inference mask.
        "infer_mask": infer_mask,
    }


def _coalesce_infer_mask(pending_tiles: list[_PendingTile]) -> torch.Tensor:
    """Validate that all tiles share one Eva-compatible inference mask."""

    reference_mask = pending_tiles[0].payload["infer_mask"]
    if reference_mask.ndim != 2:
        raise ValueError("infer_mask must have shape (channels, patches)")

    for tile in pending_tiles[1:]:
        if not torch.equal(tile.payload["infer_mask"], reference_mask):
            raise ValueError("all tiles in a batch must share the same infer_mask")

    return reference_mask


def _extract_marker_out_group(marker_out: Sequence[Sequence[str]]) -> list[str]:
    """Validate and unwrap the single Eva output group expected per tile."""

    if len(marker_out) != 1:
        raise ValueError("marker_out must contain exactly one output marker group per tile")

    output_group = marker_out[0]
    if any(not isinstance(name, str) for name in output_group):
        raise ValueError("marker_out groups must contain only strings")

    return list(output_group)


def _default_run_model(model: object, batch: EvaModelBatch) -> np.ndarray:
    if not callable(model):
        raise ValueError("model must be callable when run_model_fn is not provided")

    with torch.inference_mode():
        raw_output = model(
            batch["input"],
            marker_in=batch["marker_in"],
            marker_out=batch["marker_out"],
            infer_mask=batch["infer_mask"],
            channel_mask=None,
        )

    predictions = raw_output[0] if isinstance(raw_output, tuple) else raw_output
    if isinstance(predictions, torch.Tensor):
        return predictions.detach().cpu().numpy().astype(np.float32, copy=False)

    return np.asarray(predictions, dtype=np.float32)


def _coerce_prediction_batch(
    predictions: np.ndarray | torch.Tensor,
    batch_size: int,
    tile_shape: torch.Size,
    channels: int,
) -> np.ndarray:
    prediction_np = (
        predictions.detach().cpu().numpy() if isinstance(predictions, torch.Tensor) else np.asarray(predictions)
    )
    prediction_np = prediction_np.astype(np.float32, copy=False)

    tile_height = int(tile_shape[0])
    tile_width = int(tile_shape[1])
    expected_nhwc_shape = (batch_size, tile_height, tile_width, channels)
    expected_nchw_shape = (batch_size, channels, tile_height, tile_width)

    if expected_nhwc_shape == expected_nchw_shape and prediction_np.shape == expected_nhwc_shape:
        raise ValueError(
            "model predictions have ambiguous 4D layout because expected "
            f"NHWC and NCHW shapes are both {expected_nhwc_shape}"
        )

    if prediction_np.shape == expected_nhwc_shape:
        return prediction_np

    if prediction_np.shape == expected_nchw_shape:
        # The writer consumes tiles in NHWC order, so move channel axis to the end.
        return np.moveaxis(prediction_np, 1, -1)

    raise ValueError(
        "model predictions must have shape "
        f"{expected_nhwc_shape} (NHWC) or {expected_nchw_shape} (NCHW), "
        f"got {prediction_np.shape}"
    )


def _get_level_shape(slide: object, level: int) -> tuple[int, int]:
    level_dimensions = getattr(slide, "level_dimensions", None)
    if level_dimensions is None:
        raise ValueError("slide must expose level_dimensions")

    try:
        level_shape = level_dimensions[level]
    except (IndexError, TypeError) as exc:
        raise ValueError(f"slide does not expose level {level}") from exc

    if (
        not isinstance(level_shape, tuple)
        or len(level_shape) != 2
        or any(not isinstance(dim, int) or isinstance(dim, bool) or dim <= 0 for dim in level_shape)
    ):
        raise ValueError("slide level_dimensions entries must be (width, height) integer tuples")

    return level_shape


def _default_read_tile(
    slide: object,
    level: int,
    x: int,
    y: int,
    width: int,
    height: int,
) -> np.ndarray:
    if not hasattr(slide, "read_region"):
        raise ValueError("slide must be read through read_tile_fn or expose read_region")

    location = (x, y)
    if level > 0:
        level_downsamples = getattr(slide, "level_downsamples", None)
        if level_downsamples is None:
            raise ValueError("slide must expose valid level_downsamples to read levels above 0")

        try:
            downsample = float(level_downsamples[level])
        except (IndexError, TypeError, ValueError) as exc:
            raise ValueError(f"slide must expose valid level_downsamples for level {level}") from exc

        if not np.isfinite(downsample) or downsample <= 0.0:
            raise ValueError(f"slide must expose valid level_downsamples for level {level}")

        location = (int(round(x * downsample)), int(round(y * downsample)))

    region = slide.read_region(location, level, (width, height))
    tile = np.asarray(region)

    if tile.ndim != 3 or tile.shape[-1] not in (3, 4):
        raise ValueError("read tile must have shape (height, width, 3|4)")

    tile = tile[..., :3]
    if np.issubdtype(tile.dtype, np.integer):
        tile = tile.astype(np.float32) / np.float32(np.iinfo(tile.dtype).max)
    else:
        tile = tile.astype(np.float32, copy=False)

    return tile


def _materialize_biomarkers(biomarkers: Sequence[str]) -> list[str]:
    if isinstance(biomarkers, (str, bytes)) or not isinstance(biomarkers, Sequence):
        raise ValueError("biomarkers must be a non-string sequence of strings")

    biomarker_names = list(biomarkers)
    if not biomarker_names:
        raise ValueError("biomarkers must not be empty")

    if any(not isinstance(name, str) for name in biomarker_names):
        raise ValueError("biomarkers must contain only strings")

    return biomarker_names
