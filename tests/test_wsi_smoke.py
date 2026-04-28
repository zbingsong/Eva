from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET

import numpy as np
import pytest
import tifffile
import torch

from utils.wsi_eva import build_virtual_stain_inputs
from utils.wsi_quant import quantize_uint16

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BIOMARKERS_PATH = REPO_ROOT / "examples" / "biomarkers.npy"
SMOKE_SLIDE_ENV = "EVA_WSI_SMOKE_SVS"
OME_NS = {"ome": "http://www.openmicroscopy.org/Schemas/OME/2016-06"}


class SmokeModel:
    """Deterministic stand-in that keeps the smoke test fast."""

    @staticmethod
    def expected_output(input_tensor: torch.Tensor, channel_count: int) -> np.ndarray:
        """Map Eva input tensors to a deterministic, spatially varying NHWC output."""

        # Input shape is ``(batch, height, width, channels)`` with inverted H&E in
        # the final three channels. Reusing that image content makes spatial
        # placement bugs visible in the smoke test.
        base = input_tensor[..., -3:].mean(dim=-1, keepdim=True).detach().cpu().numpy()
        channel_offsets = np.linspace(0.0, 0.2, channel_count, dtype=np.float32).reshape(1, 1, 1, channel_count)
        return np.clip(base * 0.75 + channel_offsets, 0.0, 1.0).astype(np.float32, copy=False)

    def __call__(
        self,
        input_tensor: torch.Tensor,
        *,
        marker_in: list[list[str]],
        marker_out: list[list[str]],
        infer_mask: torch.Tensor,
        channel_mask: object | None = None,
    ) -> np.ndarray:
        """Return a stable NHWC prediction tensor for the requested biomarkers."""

        del marker_in, infer_mask, channel_mask

        channel_count = len(marker_out[0])
        return self.expected_output(input_tensor, channel_count)


def _load_cli_module() -> Any:
    """Load the WSI inference CLI module without importing the package root."""

    module_path = REPO_ROOT / "scripts" / "run_wsi_inference.py"
    spec = importlib.util.spec_from_file_location("run_wsi_inference_smoke", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _resolve_fixture_path(env_name: str, default: Path | None = None) -> Path:
    """Resolve a smoke-test fixture path from an environment variable or default."""

    raw_value = os.environ.get(env_name)
    if raw_value:
        path = Path(raw_value).expanduser()
        if path.is_file():
            return path
        pytest.skip(f"{env_name} points to a missing file: {path}")

    if default is None:
        pytest.skip(f"Set {env_name} to a small .svs fixture to enable the WSI smoke test")

    if default.is_file():
        return default

    pytest.skip(f"default fixture is missing: {default}")


def test_smoke_region_export(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Run a cheap real-slide smoke path when a sample SVS fixture is available."""

    pytest.importorskip("openslide")

    slide_path = _resolve_fixture_path(SMOKE_SLIDE_ENV)
    module = _load_cli_module()
    biomarkers = module.load_biomarkers(DEFAULT_BIOMARKERS_PATH)
    output_dir = tmp_path / "wsi-smoke"

    monkeypatch.setattr(module, "load_from_checkpoint", lambda path, config, device: SmokeModel())
    monkeypatch.setattr(module.torch.cuda, "is_available", lambda: False)

    exit_code = module.main(
        [
            "--slide-path",
            str(slide_path),
            "--output-dir",
            str(output_dir),
            "--levels",
            "0",
            "--tile-size",
            "224",
            "--stride",
            "224",
            "--batch-size",
            "1",
            "--white-threshold",
            "0.95",
            "--quant-min",
            "0.0",
            "--quant-max",
            "1.0",
        ]
    )

    assert exit_code == 0

    raw_path = output_dir / "level_0" / "predictions.npy"
    ome_path = output_dir / "level_0" / "predictions.ome.tiff"
    assert raw_path.is_file()
    assert ome_path.is_file()

    openslide = pytest.importorskip("openslide")
    slide = openslide.OpenSlide(str(slide_path))
    try:
        level_width, level_height = slide.level_dimensions[0]
        raw = np.load(raw_path, mmap_mode="r")
        assert raw.shape == (level_height, level_width, len(biomarkers))
        assert raw.shape[2] == len(biomarkers)
        predicted_positions = np.argwhere(np.any(np.isfinite(raw), axis=-1))
        assert predicted_positions.size > 0

        y, x = predicted_positions[0]
        tile_y = (int(y) // 224) * 224
        tile_x = (int(x) // 224) * 224
        tile_rgba = np.asarray(slide.read_region((tile_x, tile_y), 0, (224, 224)), dtype=np.uint8)
        tile_rgb = tile_rgba[:, :, :3].astype(np.float32) / 255.0
        payload = build_virtual_stain_inputs(tile_rgb, biomarkers, patch_size=8)
        expected_tile = SmokeModel.expected_output(
            torch.from_numpy(payload["input"]).unsqueeze(0).float(),
            channel_count=len(biomarkers),
        )[0]

        local_y = int(y) - tile_y
        local_x = int(x) - tile_x
        assert np.allclose(raw[y, x, :], expected_tile[local_y, local_x, :])
        assert np.allclose(
            raw[tile_y : tile_y + 8, tile_x : tile_x + 8, 0],
            expected_tile[:8, :8, 0],
        )

        with tifffile.TiffFile(ome_path) as tif:
            series = tif.series[0]
            ome_root = ET.fromstring(tif.ome_metadata)
            channels = ome_root.findall("ome:Image/ome:Pixels/ome:Channel", OME_NS)
            assert series.axes == "CYX"
            assert series.shape == (len(biomarkers), level_height, level_width)
            assert [channel.attrib["Name"] for channel in channels] == biomarkers

            ome_data = series.asarray()
            expected_quantized = quantize_uint16(expected_tile[local_y, local_x, :], quant_min=0.0, quant_max=1.0)
            assert np.array_equal(ome_data[:, y, x], expected_quantized)
    finally:
        slide.close()
