from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


def _load_cli_module():
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "run_wsi_inference.py"
    spec = importlib.util.spec_from_file_location("run_wsi_inference", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cli_parser_accepts_level_zero_defaults() -> None:
    module = _load_cli_module()
    parser = module.build_arg_parser()

    args = parser.parse_args(["--slide-path", "sample.svs", "--output-dir", "out"])

    assert args.levels == [0]
    assert args.tile_size == 224
    assert args.stride == 224
    assert args.white_threshold == pytest.approx(200.0 / 255.0)
    assert args.ome_quant_mode == "global"


def test_cli_validate_args_accepts_fixed_quantization_bounds() -> None:
    module = _load_cli_module()
    parser = module.build_arg_parser()

    args = parser.parse_args(
        [
            "--slide-path",
            "sample.svs",
            "--output-dir",
            "out",
            "--quant-min",
            "0.0",
            "--quant-max",
            "1.0",
        ]
    )

    module._validate_args(args)


def test_cli_validate_args_warns_that_fixed_bounds_are_ignored_in_tile_mode() -> None:
    module = _load_cli_module()
    parser = module.build_arg_parser()

    args = parser.parse_args(
        [
            "--slide-path",
            "sample.svs",
            "--output-dir",
            "out",
            "--ome-quant-mode",
            "tile",
            "--quant-min",
            "0.0",
            "--quant-max",
            "1.0",
        ]
    )

    with pytest.warns(UserWarning, match="ignored"):
        module._validate_args(args)


def test_cli_validate_args_rejects_non_finite_quantization_bounds() -> None:
    module = _load_cli_module()
    parser = module.build_arg_parser()

    args = parser.parse_args(
        [
            "--slide-path",
            "sample.svs",
            "--output-dir",
            "out",
            "--quant-min",
            "nan",
            "--quant-max",
            "1.0",
        ]
    )

    with pytest.raises(ValueError, match="finite"):
        module._validate_args(args)


def test_load_biomarkers_rejects_non_1d_arrays(tmp_path: Path) -> None:
    module = _load_cli_module()
    biomarker_path = tmp_path / "bad_biomarkers.npy"
    np.save(biomarker_path, np.array([["A"], ["B"]], dtype="<U1"))

    with pytest.raises(ValueError, match="1D"):
        module.load_biomarkers(biomarker_path)


def test_cli_validate_args_rejects_negative_levels() -> None:
    module = _load_cli_module()
    parser = module.build_arg_parser()

    args = parser.parse_args(["--slide-path", "sample.svs", "--output-dir", "out", "--levels", "0", "-1"])

    with pytest.raises(ValueError, match="non-negative"):
        module._validate_args(args)


def test_cli_validate_args_rejects_overlapping_stride() -> None:
    module = _load_cli_module()
    parser = module.build_arg_parser()

    args = parser.parse_args(
        ["--slide-path", "sample.svs", "--output-dir", "out", "--tile-size", "224", "--stride", "112"]
    )

    with pytest.raises(ValueError, match="greater than or equal"):
        module._validate_args(args)


def test_main_runs_requested_levels_and_closes_slide(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_cli_module()
    output_dir = tmp_path / "out"
    run_calls: list[dict[str, object]] = []

    class FakeSlide:
        def __init__(self) -> None:
            self.closed = False

        def close(self) -> None:
            self.closed = True

    fake_slide = FakeSlide()

    monkeypatch.setattr(
        module.OmegaConf,
        "load",
        lambda path: module.OmegaConf.create({"ds": {"token_size": 8, "patch_size": 224}}),
    )
    monkeypatch.setattr(module, "load_biomarkers", lambda path: ["A", "B"])
    monkeypatch.setattr(module, "load_from_checkpoint", lambda path, config, device: "model")
    monkeypatch.setattr(module, "_open_slide", lambda path: fake_slide)
    monkeypatch.setattr(module.torch.cuda, "is_available", lambda: False)

    def fake_run_level_inference(**kwargs):
        run_calls.append(kwargs)
        level = kwargs["level"]
        level_dir = Path(kwargs["output_dir"]) / f"level_{level}"
        return module.LevelInferenceResult(
            level=level,
            level_shape=(224, 224),
            level_output_dir=level_dir,
            raw_npy_path=level_dir / "predictions.npy",
            ome_tiff_path=level_dir / "predictions.ome.tiff",
            total_tiles=1,
            skipped_tiles=0,
            predicted_tiles=1,
            batches_run=1,
        )

    monkeypatch.setattr(module, "run_level_inference", fake_run_level_inference)

    exit_code = module.main(
        [
            "--slide-path",
            "sample.svs",
            "--output-dir",
            str(output_dir),
            "--levels",
            "0",
            "2",
            "--ome-quant-mode",
            "tile",
            "--quant-min",
            "0.0",
            "--quant-max",
            "1.0",
        ]
    )

    assert exit_code == 0
    assert fake_slide.closed is True
    assert [call["level"] for call in run_calls] == [0, 2]
    assert all(call["slide"] is fake_slide for call in run_calls)
    assert all(call["model"] == "model" for call in run_calls)
    assert all(call["biomarkers"] == ["A", "B"] for call in run_calls)
    assert all(call["patch_size"] == 8 for call in run_calls)
    assert all(call["device"] == "cpu" for call in run_calls)
    assert all(call["quant_min"] is None for call in run_calls)
    assert all(call["quant_max"] is None for call in run_calls)
    assert all(call["ome_quant_mode"] == "tile" for call in run_calls)


def test_load_patch_size_rejects_invalid_config() -> None:
    module = _load_cli_module()

    with pytest.raises(ValueError, match="ds.token_size"):
        module._load_patch_size(module.OmegaConf.create({"ds": {"token_size": True}}))

    with pytest.raises(ValueError, match="ds.token_size"):
        module._load_patch_size(module.OmegaConf.create({"ds": {}}))


def test_validate_model_geometry_rejects_mismatched_tile_size() -> None:
    module = _load_cli_module()
    parser = module.build_arg_parser()
    args = parser.parse_args(["--slide-path", "sample.svs", "--output-dir", "out", "--tile-size", "256"])
    config = module.OmegaConf.create({"ds": {"patch_size": 224}})

    with pytest.raises(ValueError, match="ds.patch_size"):
        module._validate_model_geometry(args, config)
