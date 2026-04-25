import numpy as np
import torch

from utils.wsi_eva import build_virtual_stain_inputs


def assert_raises_value_error(fn: object, message: str) -> None:
    try:
        fn()
    except ValueError as exc:
        assert message in str(exc)
    else:
        raise AssertionError(f"Expected ValueError containing: {message}")


def test_build_virtual_stain_inputs_shapes_and_marker_contract() -> None:
    tile = np.zeros((224, 224, 3), dtype=np.float32)
    biomarkers = ["A", "B"]

    payload = build_virtual_stain_inputs(tile, biomarkers)

    assert payload["input"].shape == (224, 224, 5)
    assert payload["marker_in"] == ["A", "B", "HECHA1", "HECHA2", "HECHA3"]
    assert payload["marker_out"] == [biomarkers]
    assert payload["infer_mask"].shape == (5, 784)
    assert torch.all(payload["infer_mask"][:2] == 1.0)
    assert torch.all(payload["infer_mask"][2:] == 0.0)


def test_build_virtual_stain_inputs_uses_zero_mif_placeholders_and_reversed_he() -> None:
    tile = np.array(
        [
            [[0.0, 0.25, 0.5], [0.75, 1.0, 0.125]],
            [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
        ],
        dtype=np.float32,
    )
    biomarkers = ["CD45"]

    payload = build_virtual_stain_inputs(tile, biomarkers, patch_size=1)

    assert np.all(payload["input"][..., :1] == 0.0)
    np.testing.assert_allclose(payload["input"][..., 1:], 1.0 - tile)
    assert payload["infer_mask"].shape == (4, 4)


def test_build_virtual_stain_inputs_rejects_invalid_tile_shape() -> None:
    tile = np.zeros((224, 224), dtype=np.float32)

    assert_raises_value_error(
        lambda: build_virtual_stain_inputs(tile, ["CD45"]),
        "tile_rgb must have shape (height, width, 3)",
    )


def test_build_virtual_stain_inputs_rejects_out_of_range_and_non_finite_values() -> None:
    out_of_range_tile = np.zeros((8, 8, 3), dtype=np.float32)
    out_of_range_tile[0, 0, 0] = 1.5

    non_finite_tile = np.zeros((8, 8, 3), dtype=np.float32)
    non_finite_tile[0, 0, 0] = np.nan

    assert_raises_value_error(
        lambda: build_virtual_stain_inputs(out_of_range_tile, ["CD45"]),
        "tile_rgb values must already be normalized to [0, 1]",
    )
    assert_raises_value_error(
        lambda: build_virtual_stain_inputs(non_finite_tile, ["CD45"]),
        "tile_rgb values must be finite",
    )


def test_build_virtual_stain_inputs_rejects_bad_patch_size_and_non_divisible_tiles() -> None:
    tile = np.zeros((10, 8, 3), dtype=np.float32)

    assert_raises_value_error(
        lambda: build_virtual_stain_inputs(tile, ["CD45"], patch_size=0),
        "patch_size must be a positive integer",
    )
    assert_raises_value_error(
        lambda: build_virtual_stain_inputs(tile, ["CD45"], patch_size=4),
        "tile_rgb height and width must be divisible by patch_size",
    )


def test_build_virtual_stain_inputs_rejects_bad_biomarkers_input() -> None:
    tile = np.zeros((8, 8, 3), dtype=np.float32)

    assert_raises_value_error(
        lambda: build_virtual_stain_inputs(tile, "CD45"),
        "biomarkers must be a non-string sequence of strings",
    )
    assert_raises_value_error(
        lambda: build_virtual_stain_inputs(tile, ["CD45", 7]),
        "biomarkers must contain only strings",
    )
