from __future__ import annotations

from collections.abc import Sequence
from typing import TypedDict

import numpy as np
import torch

from utils.constant import hande_marker

HE_MARKERS = tuple(hande_marker)


class VirtualStainInputs(TypedDict):
    input: np.ndarray
    marker_in: list[str]
    marker_out: list[list[str]]
    infer_mask: torch.Tensor


def _normalize_biomarkers(biomarkers: Sequence[str]) -> list[str]:
    """Validate and materialize biomarker names for virtual staining."""

    if isinstance(biomarkers, (str, bytes)) or not isinstance(biomarkers, Sequence):
        raise ValueError("biomarkers must be a non-string sequence of strings")

    biomarker_names = list(biomarkers)
    if any(not isinstance(name, str) for name in biomarker_names):
        raise ValueError("biomarkers must contain only strings")

    return biomarker_names


def build_virtual_stain_inputs(
    tile_rgb: np.ndarray,
    biomarkers: Sequence[str],
    patch_size: int = 8,
) -> VirtualStainInputs:
    """Build Eva virtual-staining inputs for an H&E tile.

    Parameters
    ----------
    tile_rgb
        Normalized H&E tile with shape ``(height, width, 3)`` and values in
        ``[0, 1]``.
    biomarkers
        Output biomarker names to predict from the H&E tile.
    patch_size
        Eva patch size used to determine the channel-by-patch inference mask.

    Returns
    -------
    VirtualStainInputs
        Payload matching the virtual-staining notebook contract.

    Raises
    ------
    ValueError
        If the tile shape, value range, or patch alignment does not satisfy the
        virtual-staining assumptions.
    """

    if not isinstance(patch_size, int) or isinstance(patch_size, bool) or patch_size <= 0:
        raise ValueError("patch_size must be a positive integer")

    tile = np.asarray(tile_rgb, dtype=np.float32)
    if tile.ndim != 3 or tile.shape[-1] != 3:
        raise ValueError("tile_rgb must have shape (height, width, 3)")

    biomarker_names = _normalize_biomarkers(biomarkers)

    if not np.all(np.isfinite(tile)):
        raise ValueError("tile_rgb values must be finite")

    if np.any((tile < 0.0) | (tile > 1.0)):
        raise ValueError("tile_rgb values must already be normalized to [0, 1]")

    height, width, _ = tile.shape
    if height % patch_size != 0 or width % patch_size != 0:
        raise ValueError("tile_rgb height and width must be divisible by patch_size")

    biomarker_placeholder = np.zeros((height, width, len(biomarker_names)), dtype=np.float32)
    input_np = np.concatenate([biomarker_placeholder, 1.0 - tile], axis=-1)

    marker_in = biomarker_names + list(HE_MARKERS)
    marker_out = [biomarker_names]

    num_patches = (height // patch_size) * (width // patch_size)
    # Eva expects a channel-by-patch mask where biomarker channels are inferred.
    infer_mask = torch.zeros((len(marker_in), num_patches), dtype=torch.float32)
    infer_mask[: len(biomarker_names), :] = 1.0

    return {
        "input": input_np,
        "marker_in": marker_in,
        "marker_out": marker_out,
        "infer_mask": infer_mask,
    }
