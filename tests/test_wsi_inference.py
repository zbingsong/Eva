from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
import torch

from utils.wsi_inference import _coerce_prediction_batch, _default_read_tile, run_level_inference


@dataclass
class FakeSlide:
    level_dimensions: list[tuple[int, int]]
    rgb_tiles: dict[tuple[int, int], np.ndarray]


@dataclass
class FakeModel:
    fill_value: float
    out_channels: int


@dataclass
class FakeOpenSlide:
    level_dimensions: list[tuple[int, int]]
    level_downsamples: list[float]
    regions: dict[tuple[tuple[int, int], int, tuple[int, int]], np.ndarray]
    read_calls: list[tuple[tuple[int, int], int, tuple[int, int]]]

    def read_region(
        self,
        location: tuple[int, int],
        level: int,
        size: tuple[int, int],
    ) -> np.ndarray:
        self.read_calls.append((location, level, size))
        return self.regions[(location, level, size)]


def test_run_level_inference_skips_blank_tiles_and_writes_outputs(tmp_path: Path) -> None:
    slide = FakeSlide(
        level_dimensions=[(448, 224)],
        rgb_tiles={
            (0, 0): np.ones((224, 224, 3), dtype=np.float32),
            (224, 0): np.zeros((224, 224, 3), dtype=np.float32),
        },
    )
    model = FakeModel(fill_value=3.0, out_channels=2)
    read_calls: list[tuple[int, int, int, int, int, int]] = []
    model_batches: list[dict[str, object]] = []

    def fake_read_tile(
        current_slide: FakeSlide,
        level: int,
        x: int,
        y: int,
        width: int,
        height: int,
    ) -> np.ndarray:
        read_calls.append((level, x, y, width, height, len(current_slide.rgb_tiles)))
        return current_slide.rgb_tiles[(x, y)]

    def fake_run_model(current_model: FakeModel, batch: dict[str, object]) -> np.ndarray:
        model_batches.append(batch)

        batch_input = batch["input"]
        infer_mask = batch["infer_mask"]
        marker_in = batch["marker_in"]
        marker_out = batch["marker_out"]

        assert isinstance(batch_input, torch.Tensor)
        assert batch_input.shape == (1, 224, 224, 5)
        assert torch.all(batch_input[..., :2] == 0.0)
        assert torch.all(batch_input[..., 2:] == 1.0)

        assert isinstance(infer_mask, torch.Tensor)
        assert infer_mask.shape == (5, 784)
        assert torch.all(infer_mask[:2, :] == 1.0)
        assert torch.all(infer_mask[2:, :] == 0.0)

        assert marker_in == [["A", "B", "HECHA1", "HECHA2", "HECHA3"]]
        assert marker_out == [["A", "B"]]

        batch_size = batch_input.shape[0]
        return np.full(
            (batch_size, 224, 224, current_model.out_channels),
            fill_value=current_model.fill_value,
            dtype=np.float32,
        )

    result = run_level_inference(
        slide=slide,
        level=0,
        biomarkers=["A", "B"],
        model=model,
        output_dir=tmp_path,
        tile_size=224,
        stride=224,
        white_threshold=0.95,
        read_tile_fn=fake_read_tile,
        run_model_fn=fake_run_model,
    )

    arr = np.load(result.raw_npy_path, mmap_mode="r")

    assert read_calls == [
        (0, 0, 0, 224, 224, 2),
        (0, 224, 0, 224, 224, 2),
    ]
    assert len(model_batches) == 1
    assert arr.shape == (224, 448, 2)
    assert np.all(np.isnan(arr[:, :224]))
    assert np.all(arr[:, 224:] == 3.0)

    assert result.level == 0
    assert result.level_shape == (448, 224)
    assert result.level_output_dir == tmp_path / "level_0"
    assert result.raw_npy_path == tmp_path / "level_0" / "predictions.npy"
    assert result.total_tiles == 2
    assert result.skipped_tiles == 1
    assert result.predicted_tiles == 1
    assert result.batches_run == 1


def test_default_read_tile_uses_level_zero_coordinates_for_downsampled_levels() -> None:
    slide = FakeOpenSlide(
        level_dimensions=[(896, 448), (224, 112)],
        level_downsamples=[1.0, 4.0],
        regions={
            ((448, 224), 1, (32, 16)): np.full((16, 32, 4), 255, dtype=np.uint8),
        },
        read_calls=[],
    )

    tile = _default_read_tile(slide, level=1, x=112, y=56, width=32, height=16)

    assert slide.read_calls == [((448, 224), 1, (32, 16))]
    assert tile.shape == (16, 32, 3)
    assert tile.dtype == np.float32
    assert np.all(tile == 1.0)


def test_default_read_tile_requires_level_downsamples_for_levels_above_zero() -> None:
    class MissingDownsampleSlide:
        level_dimensions = [(896, 448), (224, 112)]

        def read_region(
            self,
            location: tuple[int, int],
            level: int,
            size: tuple[int, int],
        ) -> np.ndarray:
            raise AssertionError("read_region should not be called without level_downsamples metadata")

    with pytest.raises(ValueError, match="level_downsamples"):
        _default_read_tile(MissingDownsampleSlide(), level=1, x=112, y=56, width=32, height=16)


@pytest.mark.parametrize(
    "level_downsamples",
    [
        [1.0],
        [1.0, 0.0],
        [1.0, np.nan],
        [1.0, "bad"],
    ],
)
def test_default_read_tile_rejects_invalid_level_downsamples(level_downsamples: list[object]) -> None:
    slide = FakeOpenSlide(
        level_dimensions=[(896, 448), (224, 112)],
        level_downsamples=level_downsamples,
        regions={},
        read_calls=[],
    )

    with pytest.raises(ValueError, match="level_downsamples"):
        _default_read_tile(slide, level=1, x=112, y=56, width=32, height=16)

    assert slide.read_calls == []


def test_coerce_prediction_batch_accepts_nhwc_without_reordering() -> None:
    predictions = np.arange(2 * 4 * 3 * 2, dtype=np.float32).reshape(2, 4, 3, 2)

    coerced = _coerce_prediction_batch(
        predictions,
        batch_size=2,
        tile_shape=torch.Size([4, 3]),
        channels=2,
    )

    assert coerced.shape == (2, 4, 3, 2)
    assert np.array_equal(coerced, predictions)


def test_coerce_prediction_batch_rejects_ambiguous_4d_layout() -> None:
    predictions = np.arange(8, dtype=np.float32).reshape(1, 2, 2, 2)

    with pytest.raises(ValueError, match="ambiguous"):
        _coerce_prediction_batch(
            predictions,
            batch_size=1,
            tile_shape=torch.Size([2, 2]),
            channels=2,
        )


def test_run_level_inference_batches_tiles_and_coerces_nchw_predictions(tmp_path: Path) -> None:
    slide = FakeSlide(
        level_dimensions=[(896, 224)],
        rgb_tiles={
            (0, 0): np.ones((224, 224, 3), dtype=np.float32),
            (224, 0): np.zeros((224, 224, 3), dtype=np.float32),
            (448, 0): np.zeros((224, 224, 3), dtype=np.float32),
            (672, 0): np.zeros((224, 224, 3), dtype=np.float32),
        },
    )
    model = FakeModel(fill_value=0.0, out_channels=2)
    batch_sizes: list[int] = []

    def fake_read_tile(
        current_slide: FakeSlide,
        level: int,
        x: int,
        y: int,
        width: int,
        height: int,
    ) -> np.ndarray:
        assert level == 0
        assert (width, height) == (224, 224)
        return current_slide.rgb_tiles[(x, y)]

    def fake_run_model(current_model: FakeModel, batch: dict[str, object]) -> np.ndarray:
        batch_input = batch["input"]
        assert isinstance(batch_input, torch.Tensor)
        batch_sizes.append(batch_input.shape[0])

        output = np.zeros(
            (batch_input.shape[0], current_model.out_channels, 224, 224),
            dtype=np.float32,
        )
        if batch_input.shape[0] == 2:
            output[0, 0, :, :] = 1.0
            output[0, 1, :, :] = 10.0
            output[1, 0, :, :] = 2.0
            output[1, 1, :, :] = 20.0
        else:
            output[0, 0, :, :] = 3.0
            output[0, 1, :, :] = 30.0

        return output

    result = run_level_inference(
        slide=slide,
        level=0,
        biomarkers=["A", "B"],
        model=model,
        output_dir=tmp_path,
        tile_size=224,
        stride=224,
        white_threshold=0.95,
        batch_size=2,
        read_tile_fn=fake_read_tile,
        run_model_fn=fake_run_model,
    )

    arr = np.load(result.raw_npy_path, mmap_mode="r")

    assert batch_sizes == [2, 1]
    assert arr.shape == (224, 896, 2)
    assert np.all(np.isnan(arr[:, 0:224]))
    assert np.all(arr[:, 224:448, 0] == 1.0)
    assert np.all(arr[:, 224:448, 1] == 10.0)
    assert np.all(arr[:, 448:672, 0] == 2.0)
    assert np.all(arr[:, 448:672, 1] == 20.0)
    assert np.all(arr[:, 672:896, 0] == 3.0)
    assert np.all(arr[:, 672:896, 1] == 30.0)
    assert result.total_tiles == 4
    assert result.skipped_tiles == 1
    assert result.predicted_tiles == 3
    assert result.batches_run == 2


def test_run_level_inference_rejects_invalid_tiling_before_creating_outputs(tmp_path: Path) -> None:
    slide = FakeSlide(
        level_dimensions=[(224, 224)],
        rgb_tiles={(0, 0): np.zeros((224, 224, 3), dtype=np.float32)},
    )

    with pytest.raises(ValueError, match="tile_size"):
        run_level_inference(
            slide=slide,
            level=0,
            biomarkers=["A"],
            model=object(),
            output_dir=tmp_path,
            tile_size=0,
            stride=224,
            white_threshold=0.95,
            read_tile_fn=lambda *_args: np.zeros((224, 224, 3), dtype=np.float32),
            run_model_fn=lambda *_args: np.zeros((1, 224, 224, 1), dtype=np.float32),
        )

    assert not (tmp_path / "level_0").exists()


def test_run_level_inference_rejects_overlapping_stride_before_creating_outputs(tmp_path: Path) -> None:
    slide = FakeSlide(
        level_dimensions=[(224, 224)],
        rgb_tiles={(0, 0): np.zeros((224, 224, 3), dtype=np.float32)},
    )

    with pytest.raises(ValueError, match="greater than or equal"):
        run_level_inference(
            slide=slide,
            level=0,
            biomarkers=["A"],
            model=object(),
            output_dir=tmp_path,
            tile_size=224,
            stride=112,
            white_threshold=0.95,
            read_tile_fn=lambda *_args: np.zeros((224, 224, 3), dtype=np.float32),
            run_model_fn=lambda *_args: np.zeros((1, 224, 224, 1), dtype=np.float32),
        )

    assert not (tmp_path / "level_0").exists()


def test_run_level_inference_rejects_multiple_marker_out_groups(tmp_path: Path) -> None:
    slide = FakeSlide(
        level_dimensions=[(224, 224)],
        rgb_tiles={(0, 0): np.zeros((224, 224, 3), dtype=np.float32)},
    )

    def fake_build_inputs(
        tile_rgb: np.ndarray,
        biomarkers: list[str],
        patch_size: int,
    ) -> dict[str, object]:
        return {
            "input": np.zeros((224, 224, 1), dtype=np.float32),
            "marker_in": ["A"],
            "marker_out": [["A"], ["B"]],
            "infer_mask": torch.zeros((1, 784), dtype=torch.float32),
        }

    with pytest.raises(ValueError, match="exactly one output marker group"):
        run_level_inference(
            slide=slide,
            level=0,
            biomarkers=["A"],
            model=object(),
            output_dir=tmp_path,
            tile_size=224,
            stride=224,
            white_threshold=1.0,
            read_tile_fn=lambda *_args: np.zeros((224, 224, 3), dtype=np.float32),
            run_model_fn=lambda *_args: np.zeros((1, 224, 224, 1), dtype=np.float32),
            build_inputs_fn=fake_build_inputs,
        )


def test_run_level_inference_rejects_levels_smaller_than_tile_size(tmp_path: Path) -> None:
    slide = FakeSlide(
        level_dimensions=[(128, 224)],
        rgb_tiles={},
    )

    with pytest.raises(ValueError, match="fit within"):
        run_level_inference(
            slide=slide,
            level=0,
            biomarkers=["A"],
            model=object(),
            output_dir=tmp_path,
            tile_size=224,
            stride=224,
            white_threshold=0.95,
            read_tile_fn=lambda *_args: np.zeros((224, 224, 3), dtype=np.float32),
            run_model_fn=lambda *_args: np.zeros((1, 224, 224, 1), dtype=np.float32),
        )

    assert not (tmp_path / "level_0").exists()


def test_run_level_inference_calls_build_inputs_with_positional_patch_size(tmp_path: Path) -> None:
    slide = FakeSlide(
        level_dimensions=[(224, 224)],
        rgb_tiles={(0, 0): np.zeros((224, 224, 3), dtype=np.float32)},
    )
    observed_patch_sizes: list[int] = []

    def fake_build_inputs(
        tile_rgb: np.ndarray,
        biomarkers: list[str],
        tile_patch_size: int,
    ) -> dict[str, object]:
        observed_patch_sizes.append(tile_patch_size)
        return {
            "input": np.zeros((224, 224, 1), dtype=np.float32),
            "marker_in": ["A"],
            "marker_out": [["A"]],
            "infer_mask": torch.zeros((1, 784), dtype=torch.float32),
        }

    result = run_level_inference(
        slide=slide,
        level=0,
        biomarkers=["A"],
        model=object(),
        output_dir=tmp_path,
        tile_size=224,
        stride=224,
        white_threshold=1.0,
        patch_size=16,
        read_tile_fn=lambda *_args: np.zeros((224, 224, 3), dtype=np.float32),
        run_model_fn=lambda *_args: np.zeros((1, 224, 224, 1), dtype=np.float32),
        build_inputs_fn=fake_build_inputs,
    )

    assert observed_patch_sizes == [16]
    assert result.predicted_tiles == 1


def test_run_level_inference_rejects_mismatched_infer_masks_within_batch(tmp_path: Path) -> None:
    slide = FakeSlide(
        level_dimensions=[(448, 224)],
        rgb_tiles={
            (0, 0): np.zeros((224, 224, 3), dtype=np.float32),
            (224, 0): np.zeros((224, 224, 3), dtype=np.float32),
        },
    )
    call_count = {"count": 0}

    def fake_build_inputs(
        tile_rgb: np.ndarray,
        biomarkers: list[str],
        tile_patch_size: int,
    ) -> dict[str, object]:
        del tile_rgb, biomarkers, tile_patch_size
        call_count["count"] += 1
        infer_mask = torch.zeros((1, 784), dtype=torch.float32)
        infer_mask[:, :] = float(call_count["count"] - 1)
        return {
            "input": np.zeros((224, 224, 1), dtype=np.float32),
            "marker_in": ["A"],
            "marker_out": [["A"]],
            "infer_mask": infer_mask,
        }

    with pytest.raises(ValueError, match="share the same infer_mask"):
        run_level_inference(
            slide=slide,
            level=0,
            biomarkers=["A"],
            model=object(),
            output_dir=tmp_path,
            tile_size=224,
            stride=224,
            white_threshold=1.0,
            batch_size=2,
            read_tile_fn=lambda *_args: np.zeros((224, 224, 3), dtype=np.float32),
            run_model_fn=lambda *_args: np.zeros((2, 224, 224, 1), dtype=np.float32),
            build_inputs_fn=fake_build_inputs,
        )
