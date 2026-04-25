from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import tifffile

from utils.wsi_inference import run_level_inference
from utils.wsi_ome import write_level_ome_tiff
from utils.wsi_quant import quantize_uint16

OME_NS = {"ome": "http://www.openmicroscopy.org/Schemas/OME/2016-06"}


@dataclass
class FakeSlide:
    level_dimensions: list[tuple[int, int]]
    rgb_tiles: dict[tuple[int, int], np.ndarray]


@dataclass
class FakeModel:
    tile_outputs: dict[float, np.ndarray]


def test_write_level_ome_tiff_creates_multichannel_file(tmp_path: Path) -> None:
    raw_path = tmp_path / "pred.npy"
    raw = np.lib.format.open_memmap(raw_path, mode="w+", dtype=np.float32, shape=(3, 4, 2))
    raw[:, :, 0] = np.array(
        [
            [0.0, 0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6, 0.7],
            [0.8, 0.9, 1.0, 1.1],
        ],
        dtype=np.float32,
    )
    raw[:, :, 1] = np.array(
        [
            [10.0, 11.0, 12.0, 13.0],
            [14.0, 15.0, 16.0, 17.0],
            [18.0, 19.0, 20.0, 21.0],
        ],
        dtype=np.float32,
    )
    raw.flush()

    ome_path = write_level_ome_tiff(
        raw_npy_path=raw_path,
        ome_path=tmp_path / "pred.tiff",
        channel_names=["A", "B"],
        level=0,
        channel_chunk_size=1,
        quant_mode="global",
        tile_size=2,
        ome_dtype="uint16",
    )

    assert ome_path.exists()

    with tifffile.TiffFile(ome_path) as tif:
        series = tif.series[0]
        data = series.asarray()
        ome_root = ET.fromstring(tif.ome_metadata)
        image = ome_root.find("ome:Image", OME_NS)
        pixels = ome_root.find("ome:Image/ome:Pixels", OME_NS)
        channels = ome_root.findall("ome:Image/ome:Pixels/ome:Channel", OME_NS)

        assert tif.is_ome
        assert series.shape == (2, 3, 4)
        assert series.axes == "CYX"
        assert data.dtype == np.uint16
        assert image is not None
        assert image.attrib["Name"] == "level0"
        assert pixels is not None
        assert pixels.attrib["DimensionOrder"] == "XYCZT"
        assert [channel.attrib["Name"] for channel in channels] == ["A", "B"]
        assert data[0, 0, 0] == 0
        assert data[0, -1, -1] == np.iinfo(np.uint16).max
        assert data[1, 0, 0] == 0
        assert data[1, -1, -1] == np.iinfo(np.uint16).max


def test_run_level_inference_writes_raw_and_ome_outputs(tmp_path: Path) -> None:
    slide = FakeSlide(
        level_dimensions=[(448, 224)],
        rgb_tiles={
            (0, 0): np.ones((224, 224, 3), dtype=np.float32),
            (224, 0): np.zeros((224, 224, 3), dtype=np.float32),
        },
    )
    model = FakeModel(
        tile_outputs={
            0.0: np.stack(
                [
                    np.linspace(0.0, 1.0, 224 * 224, dtype=np.float32).reshape(224, 224),
                    np.linspace(0.0, 1.0, 224 * 224, dtype=np.float32).reshape(224, 224),
                ],
                axis=-1,
            ),
        }
    )

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
        fill_key = float(batch_input[0, 0, 0, 0].item())
        tile_output = current_model.tile_outputs[fill_key]
        return np.expand_dims(tile_output, axis=0)

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
        ome_dtype="uint16",
    )

    raw = np.load(result.raw_npy_path, mmap_mode="r")

    assert raw.shape == (224, 448, 2)
    assert np.all(raw[:, :224] == 0.0)
    assert np.isclose(raw[0, 224, 0], 0.0)
    assert np.isclose(raw[-1, -1, 0], 1.0)
    assert np.isclose(raw[0, 224, 1], 0.0)
    assert np.isclose(raw[-1, -1, 1], 1.0)

    with tifffile.TiffFile(result.ome_tiff_path) as tif:
        data = tif.series[0].asarray()
        ome_root = ET.fromstring(tif.ome_metadata)
        image = ome_root.find("ome:Image", OME_NS)
        channels = ome_root.findall("ome:Image/ome:Pixels/ome:Channel", OME_NS)

        assert tif.is_ome
        assert tif.series[0].axes == "CYX"
        assert data.shape == (2, 224, 448)
        assert data.dtype == np.uint16
        assert image is not None
        assert image.attrib["Name"] == "level0"
        assert [channel.attrib["Name"] for channel in channels] == ["A", "B"]
        assert np.all(data[:, :, :224] == 0)
        assert data[0, 0, 224] == 0
        assert data[0, -1, -1] == np.iinfo(np.uint16).max
        assert data[1, 0, 224] == 0
        assert data[1, -1, -1] == np.iinfo(np.uint16).max


def test_run_level_inference_forwards_ome_writer_settings(tmp_path: Path) -> None:
    slide = FakeSlide(
        level_dimensions=[(224, 224)],
        rgb_tiles={(0, 0): np.zeros((224, 224, 3), dtype=np.float32)},
    )
    forwarded_calls: list[
        tuple[Path, Path, list[str], int, int, float | None, float | None, str, int, str | None]
    ] = []

    def fake_read_tile(
        current_slide: FakeSlide,
        level: int,
        x: int,
        y: int,
        width: int,
        height: int,
    ) -> np.ndarray:
        return current_slide.rgb_tiles[(x, y)]

    def fake_run_model(current_model: object, batch: dict[str, object]) -> np.ndarray:
        return np.zeros((1, 224, 224, 2), dtype=np.float32)

    def fake_write_ome(
        raw_npy_path: str | Path,
        ome_path: str | Path,
        channel_names: list[str],
        level: int,
        channel_chunk_size: int,
        quant_min: float | None,
        quant_max: float | None,
        quant_mode: str,
        tile_size: int,
        ome_dtype: str | None,
    ) -> Path:
        raw_path = Path(raw_npy_path)
        requested_output_path = Path(ome_path)
        actual_output_path = tmp_path / "custom" / "writer-output.ome.tiff"
        forwarded_calls.append(
            (
                raw_path,
                requested_output_path,
                channel_names,
                level,
                channel_chunk_size,
                quant_min,
                quant_max,
                quant_mode,
                tile_size,
                ome_dtype,
            )
        )
        actual_output_path.parent.mkdir(parents=True, exist_ok=True)
        actual_output_path.write_bytes(b"fake-ome")
        return actual_output_path

    result = run_level_inference(
        slide=slide,
        level=0,
        biomarkers=["A", "B"],
        model=object(),
        output_dir=tmp_path,
        tile_size=224,
        stride=224,
        white_threshold=1.0,
        read_tile_fn=fake_read_tile,
        run_model_fn=fake_run_model,
        write_ome_fn=fake_write_ome,
        ome_channel_chunk_size=7,
        quant_min=-1.0,
        quant_max=3.0,
        ome_quant_mode="tile",
        ome_dtype="uint16",
    )

    assert forwarded_calls == [
        (
            tmp_path / "level_0" / "predictions.npy",
            tmp_path / "level_0" / "predictions.ome.tiff",
            ["A", "B"],
            0,
            7,
            -1.0,
            3.0,
            "tile",
            224,
            "uint16",
        )
    ]
    assert result.ome_tiff_path == tmp_path / "custom" / "writer-output.ome.tiff"


def test_write_level_ome_tiff_respects_fixed_quantization_bounds(tmp_path: Path) -> None:
    raw_path = tmp_path / "fixed-pred.npy"
    raw = np.lib.format.open_memmap(raw_path, mode="w+", dtype=np.float32, shape=(2, 2, 1))
    raw[:, :, 0] = np.array([[0.0, 5.0], [10.0, 20.0]], dtype=np.float32)
    raw.flush()

    ome_path = write_level_ome_tiff(
        raw_npy_path=raw_path,
        ome_path=tmp_path / "fixed-pred.ome.tiff",
        channel_names=["A"],
        level=0,
        channel_chunk_size=1,
        quant_min=0.0,
        quant_max=20.0,
        quant_mode="global",
        tile_size=2,
        ome_dtype="uint16",
    )

    with tifffile.TiffFile(ome_path) as tif:
        data = tif.series[0].asarray()

    assert np.array_equal(np.squeeze(data), quantize_uint16(raw[:, :, 0], quant_min=0.0, quant_max=20.0))


def test_write_level_ome_tiff_tile_quantizes_each_tile_independently(tmp_path: Path) -> None:
    raw_path = tmp_path / "tile-pred.npy"
    raw = np.lib.format.open_memmap(raw_path, mode="w+", dtype=np.float32, shape=(2, 4, 1))
    raw[:, 0:2, 0] = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=np.float32)
    raw[:, 2:4, 0] = np.array([[10.0, 20.0], [10.0, 20.0]], dtype=np.float32)
    raw.flush()

    ome_path = write_level_ome_tiff(
        raw_npy_path=raw_path,
        ome_path=tmp_path / "tile-pred.ome.tiff",
        channel_names=["A"],
        level=0,
        channel_chunk_size=1,
        quant_mode="tile",
        tile_size=2,
        ome_dtype="uint16",
    )

    with tifffile.TiffFile(ome_path) as tif:
        data = np.squeeze(tif.series[0].asarray())

    expected = np.array(
        [
            [0, np.iinfo(np.uint16).max, 0, np.iinfo(np.uint16).max],
            [0, np.iinfo(np.uint16).max, 0, np.iinfo(np.uint16).max],
        ],
        dtype=np.uint16,
    )
    assert np.array_equal(data, expected)


def test_write_level_ome_tiff_none_mode_preserves_float32_values(tmp_path: Path) -> None:
    raw_path = tmp_path / "float-pred.npy"
    raw = np.lib.format.open_memmap(raw_path, mode="w+", dtype=np.float32, shape=(2, 3, 1))
    raw[:, :, 0] = np.array(
        [
            [-1.5, 0.0, 1.5],
            [2.25, -0.75, 3.5],
        ],
        dtype=np.float32,
    )
    raw.flush()

    ome_path = write_level_ome_tiff(
        raw_npy_path=raw_path,
        ome_path=tmp_path / "float-pred.ome.tiff",
        channel_names=["A"],
        level=0,
        channel_chunk_size=1,
        quant_mode="none",
        tile_size=2,
        ome_dtype="float32",
    )

    with tifffile.TiffFile(ome_path) as tif:
        data = np.squeeze(tif.series[0].asarray())
        pixels = ET.fromstring(tif.ome_metadata).find("ome:Image/ome:Pixels", OME_NS)

    assert data.dtype == np.float32
    np.testing.assert_allclose(data, raw[:, :, 0])
    assert pixels is not None
    assert pixels.attrib["Type"] == "float"
