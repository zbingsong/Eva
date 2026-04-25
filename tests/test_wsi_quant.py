from __future__ import annotations

import numpy as np
import pytest

from utils.wsi_ome import build_ome_metadata
from utils.wsi_quant import quantize_uint16


def test_quantize_uint16_maps_range() -> None:
    arr = np.array([0.0, 0.5, 1.0], dtype=np.float32)

    out = quantize_uint16(arr, quant_min=0.0, quant_max=1.0)

    assert out.dtype == np.uint16
    assert out[0] == 0
    assert out[1] in {32767, 32768}
    assert out[-1] == 65535


def test_quantize_uint16_clips_values_outside_range() -> None:
    arr = np.array([-1.0, 0.5, 2.0], dtype=np.float32)

    out = quantize_uint16(arr, quant_min=0.0, quant_max=1.0)

    np.testing.assert_array_equal(out, np.array([0, 32768, 65535], dtype=np.uint16))


def test_quantize_uint16_accepts_large_finite_float64_inputs() -> None:
    arr = np.array([0.0, 1.0e308], dtype=np.float64)

    out = quantize_uint16(arr, quant_min=0.0, quant_max=1.0e308)

    np.testing.assert_array_equal(out, np.array([0, 65535], dtype=np.uint16))


@pytest.mark.parametrize(
    ("quant_min", "quant_max"),
    [
        (1.0, 1.0),
        (2.0, 1.0),
    ],
)
def test_quantize_uint16_rejects_invalid_quantization_range(
    quant_min: float,
    quant_max: float,
) -> None:
    with pytest.raises(ValueError, match="quant_max must be greater than quant_min"):
        quantize_uint16(np.array([0.0], dtype=np.float32), quant_min=quant_min, quant_max=quant_max)


@pytest.mark.parametrize("quant_min", [np.nan, np.inf, -np.inf])
def test_quantize_uint16_rejects_non_finite_quant_min(quant_min: float) -> None:
    with pytest.raises(ValueError, match="quant_min and quant_max must be finite"):
        quantize_uint16(np.array([0.0], dtype=np.float32), quant_min=quant_min, quant_max=1.0)


@pytest.mark.parametrize("quant_max", [np.nan, np.inf, -np.inf])
def test_quantize_uint16_rejects_non_finite_quant_max(quant_max: float) -> None:
    with pytest.raises(ValueError, match="quant_min and quant_max must be finite"):
        quantize_uint16(np.array([0.0], dtype=np.float32), quant_min=0.0, quant_max=quant_max)


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_quantize_uint16_rejects_non_finite_array_values(value: float) -> None:
    with pytest.raises(ValueError, match="arr must contain only finite values"):
        quantize_uint16(np.array([0.0, value], dtype=np.float32), quant_min=0.0, quant_max=1.0)


def test_build_ome_metadata_contains_channel_names() -> None:
    metadata = build_ome_metadata(["A", "B"], level=0)

    assert metadata["axes"] == "CYX"
    assert metadata["Channel"]["Name"] == ["A", "B"]
    assert metadata["Name"] == "level0"


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        (
            {"channel_names": "A", "level": 0},
            "channel_names must be a non-string sequence of strings",
        ),
        (
            {"channel_names": [], "level": 0},
            "channel_names must not be empty",
        ),
        (
            {"channel_names": ["A", 1], "level": 0},
            "channel_names must contain only strings",
        ),
        (
            {"channel_names": ["A"], "level": -1},
            "level must be a non-negative integer",
        ),
    ],
)
def test_build_ome_metadata_rejects_invalid_inputs(kwargs: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        build_ome_metadata(**kwargs)
