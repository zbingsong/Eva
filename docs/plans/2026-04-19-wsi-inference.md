# Eva WSI Inference Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a level-aware WSI inference pipeline for Eva that reads an H&E `.svs`, runs level-0 tiled H&E-to-spatial-proteomics inference, skips white background tiles, writes raw `float32` `.npy`, and writes a quantized multichannel `ome.tiff`.

**Architecture:** Add a thin orchestration layer outside `Eva/` that uses OpenSlide for tile reads, existing `EvaMAE` inference for per-tile predictions, a memmap-backed raw writer for out-of-core assembly, and a `tifffile` writer for multichannel OME-TIFF export. Structure the code around `levels=[...]` even though the first release only processes level `0`.

**Tech Stack:** Python, OpenSlide, NumPy, PyTorch, OmegaConf, tifffile, pytest

---

### Task 1: Add WSI runtime dependencies and test scaffold

**Files:**
- Modify: `env.yaml`
- Modify: `setup.py`
- Create: `tests/conftest.py`

**Step 1: Write the failing test**

```python
def test_runtime_deps_listed():
    text = Path("env.yaml").read_text()
    assert "tifffile" in text
    assert "openslide" in text or "openslide-python" in text
    assert "pytest" in text
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/conftest.py -v`
Expected: FAIL because the file and dependency entries do not exist yet.

**Step 3: Write minimal implementation**

- Add `tifffile`, `openslide-python`, and `pytest` to `env.yaml`.
- Add runtime dependencies needed by the new helpers to `setup.py`.
- Create `tests/conftest.py` with shared fixtures for temporary output dirs.

**Step 4: Run test to verify it passes**

Run: `pytest tests/conftest.py -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add env.yaml setup.py tests/conftest.py
git commit -m "build: add WSI inference dependencies"
```

### Task 2: Add tile grid planning and white-background detection

**Files:**
- Create: `utils/wsi_background.py`
- Create: `utils/wsi_tiling.py`
- Create: `tests/test_wsi_background.py`
- Create: `tests/test_wsi_tiling.py`

**Step 1: Write the failing test**

```python
def test_is_near_white_tile_detects_blank_rgb():
    tile = np.ones((224, 224, 3), dtype=np.float32)
    assert is_near_white_tile(tile, threshold=0.95) is True

def test_is_near_white_tile_rejects_tissue():
    tile = np.ones((224, 224, 3), dtype=np.float32)
    tile[0, 0] = [0.2, 0.3, 0.4]
    assert is_near_white_tile(tile, threshold=0.95) is False

def test_iter_level_tiles_yields_non_overlapping_windows():
    windows = list(iter_level_tiles((500, 500), tile_size=224, stride=224))
    assert windows == [(0, 0, 224, 224), (224, 0, 224, 224), (0, 224, 224, 224), (224, 224, 224, 224)]
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_wsi_background.py tests/test_wsi_tiling.py -v`
Expected: FAIL with import errors.

**Step 3: Write minimal implementation**

- `utils/wsi_background.py`: implement `is_near_white_tile(tile, threshold)` using “all pixels, all channels above threshold”.
- `utils/wsi_tiling.py`: implement `iter_level_tiles(level_size, tile_size, stride)` yielding only full windows.

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_wsi_background.py tests/test_wsi_tiling.py -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add utils/wsi_background.py utils/wsi_tiling.py tests/test_wsi_background.py tests/test_wsi_tiling.py
git commit -m "feat: add WSI tile planning and background filtering"
```

### Task 3: Add Eva tile-input construction and output metadata helpers

**Files:**
- Create: `utils/wsi_eva.py`
- Create: `tests/test_wsi_eva.py`

**Step 1: Write the failing test**

```python
def test_build_virtual_stain_inputs_shapes():
    tile = np.zeros((224, 224, 3), dtype=np.float32)
    biomarkers = ["A", "B"]
    payload = build_virtual_stain_inputs(tile, biomarkers)
    assert payload["input"].shape == (224, 224, 5)
    assert payload["marker_in"] == ["A", "B", "HECHA1", "HECHA2", "HECHA3"]
    assert payload["marker_out"] == [biomarkers]
    assert payload["infer_mask"].shape == (5, 784)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_wsi_eva.py -v`
Expected: FAIL because the helper module does not exist.

**Step 3: Write minimal implementation**

- Implement `build_virtual_stain_inputs(tile_rgb, biomarkers, patch_size=8)`.
- Reverse H&E as `1 - tile`.
- Create zero-valued biomarker placeholders.
- Build `marker_in`, `marker_out`, and `infer_mask` exactly as in `tutorials/virtual_stain.ipynb`.

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_wsi_eva.py -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add utils/wsi_eva.py tests/test_wsi_eva.py
git commit -m "feat: add Eva WSI input construction helpers"
```

### Task 4: Add raw memmap writer for slide-level assembly

**Files:**
- Create: `utils/wsi_raw_writer.py`
- Create: `tests/test_wsi_raw_writer.py`

**Step 1: Write the failing test**

```python
def test_raw_writer_persists_tile_into_memmap(tmp_path):
    path = tmp_path / "pred.npy"
    writer = RawPredictionWriter(path=path, shape=(448, 448, 2), dtype=np.float32)
    tile = np.ones((224, 224, 2), dtype=np.float32)
    writer.write_tile(224, 0, tile)
    writer.close()
    arr = np.load(path, mmap_mode="r")
    assert arr.shape == (448, 448, 2)
    assert np.all(arr[0:224, 224:448] == 1.0)
    assert np.all(arr[224:, :] == 0.0)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_wsi_raw_writer.py -v`
Expected: FAIL because the writer does not exist.

**Step 3: Write minimal implementation**

- Implement `RawPredictionWriter` using `numpy.lib.format.open_memmap`.
- Zero-initialize the full array.
- Add `write_tile(x, y, tile)` and `close()`.

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_wsi_raw_writer.py -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add utils/wsi_raw_writer.py tests/test_wsi_raw_writer.py
git commit -m "feat: add raw WSI prediction writer"
```

### Task 5: Add quantization helper and OME metadata builder

**Files:**
- Create: `utils/wsi_quant.py`
- Create: `utils/wsi_ome.py`
- Create: `tests/test_wsi_quant.py`

**Step 1: Write the failing test**

```python
def test_quantize_uint16_maps_range():
    arr = np.array([0.0, 0.5, 1.0], dtype=np.float32)
    out = quantize_uint16(arr, quant_min=0.0, quant_max=1.0)
    assert out.dtype == np.uint16
    assert out[0] == 0
    assert out[-1] == 65535

def test_build_ome_metadata_contains_channel_names():
    md = build_ome_metadata(["A", "B"], level=0, level_shape=(10, 20))
    assert md["axes"] == "CYX"
    assert md["Channel"]["Name"] == ["A", "B"]
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_wsi_quant.py -v`
Expected: FAIL with import errors.

**Step 3: Write minimal implementation**

- Implement `quantize_uint16`.
- Implement `build_ome_metadata(channel_names, level, level_shape, extra_metadata=None)`.

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_wsi_quant.py -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add utils/wsi_quant.py utils/wsi_ome.py tests/test_wsi_quant.py
git commit -m "feat: add OME quantization helpers"
```

### Task 6: Add WSI orchestration with injectable slide reader and model

**Files:**
- Create: `utils/wsi_inference.py`
- Create: `tests/test_wsi_inference.py`

**Step 1: Write the failing test**

```python
def test_run_level_inference_skips_blank_tiles_and_writes_outputs(tmp_path):
    slide = FakeSlide(level_dimensions=[(448, 224)], rgb_tiles={
        (0, 0): np.ones((224, 224, 3), dtype=np.float32),
        (224, 0): np.zeros((224, 224, 3), dtype=np.float32),
    })
    model = FakeModel(fill_value=3.0, out_channels=2)
    result = run_level_inference(
        slide=slide,
        level=0,
        biomarkers=["A", "B"],
        model=model,
        output_dir=tmp_path,
        tile_size=224,
        stride=224,
        white_threshold=0.95,
    )
    arr = np.load(result.raw_npy_path, mmap_mode="r")
    assert np.all(arr[:, :224] == 0.0)
    assert np.all(arr[:, 224:] == 3.0)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_wsi_inference.py -v`
Expected: FAIL because orchestration code does not exist.

**Step 3: Write minimal implementation**

- Implement `run_level_inference(...)` with dependency injection for slide reads and model calls.
- Use tile iteration, background filtering, Eva input construction, batch execution, and raw writer integration.
- Return output paths and simple counters.

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_wsi_inference.py -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add utils/wsi_inference.py tests/test_wsi_inference.py
git commit -m "feat: add WSI inference orchestration"
```

### Task 7: Add OME-TIFF writer integration

**Files:**
- Modify: `utils/wsi_inference.py`
- Modify: `utils/wsi_ome.py`
- Create: `tests/test_wsi_ome_writer.py`

**Step 1: Write the failing test**

```python
def test_write_level_ome_tiff_creates_multichannel_file(tmp_path):
    raw = np.lib.format.open_memmap(tmp_path / "pred.npy", mode="w+", dtype=np.float32, shape=(16, 16, 2))
    raw[:] = 1.0
    raw.flush()
    ome_path = write_level_ome_tiff(
        raw_npy_path=tmp_path / "pred.npy",
        ome_path=tmp_path / "pred.ome.tiff",
        channel_names=["A", "B"],
        quant_min=0.0,
        quant_max=1.0,
        level=0,
    )
    assert ome_path.exists()
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_wsi_ome_writer.py -v`
Expected: FAIL because the writer is incomplete.

**Step 3: Write minimal implementation**

- Implement a level-0 OME-TIFF writer using `tifffile`.
- Read raw predictions via memmap.
- Quantize per block or per channel chunk.
- Write `CYX` output with biomarker metadata.
- Wire OME export into `run_level_inference(...)`.

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_wsi_ome_writer.py tests/test_wsi_inference.py -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add utils/wsi_inference.py utils/wsi_ome.py tests/test_wsi_ome_writer.py
git commit -m "feat: add OME-TIFF export for WSI predictions"
```

### Task 8: Add CLI entry point for real `.svs` runs

**Files:**
- Create: `scripts/run_wsi_inference.py`
- Modify: `README.md`

**Step 1: Write the failing test**

```python
def test_cli_parser_accepts_level_zero_defaults():
    parser = build_arg_parser()
    args = parser.parse_args(["--slide-path", "sample.svs", "--output-dir", "out"])
    assert args.levels == [0]
    assert args.tile_size == 224
    assert args.stride == 224
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_cli_wsi.py -v`
Expected: FAIL because the CLI does not exist.

**Step 3: Write minimal implementation**

- Add CLI parsing for slide path, output dir, checkpoint, config, levels, batch size, white threshold, device, and quantization args.
- Load `config.yaml`, biomarkers, and `Eva_ft.ckpt`.
- Call `run_level_inference(...)`.
- Add README usage for one-slide level-0 export.

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_cli_wsi.py -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add scripts/run_wsi_inference.py README.md tests/test_cli_wsi.py
git commit -m "feat: add CLI for Eva WSI inference"
```

### Task 9: Add smoke validation path for a small real slide region

**Files:**
- Create: `tests/test_wsi_smoke.py`
- Modify: `README.md`

**Step 1: Write the failing test**

```python
def test_smoke_region_export(tmp_path):
    pytest.skip("Enable only when sample SVS fixture is available")
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_wsi_smoke.py -v`
Expected: SKIP or FAIL until the fixture policy is documented.

**Step 3: Write minimal implementation**

- Document a smoke-test command using a small `.svs` sample.
- Add assertions for output file existence, output shape, and channel count when a fixture is available.

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_wsi_smoke.py -v`
Expected: SKIP by default, PASS when fixture is provided.

**Step 5: Commit**

```bash
git add tests/test_wsi_smoke.py README.md
git commit -m "test: add smoke validation for WSI inference"
```

### Task 10: Final verification

**Files:**
- Modify: none
- Test: `tests/`

**Step 1: Run focused unit and integration tests**

Run: `pytest tests/test_wsi_background.py tests/test_wsi_tiling.py tests/test_wsi_eva.py tests/test_wsi_raw_writer.py tests/test_wsi_quant.py tests/test_wsi_inference.py tests/test_wsi_ome_writer.py tests/test_cli_wsi.py -v`
Expected: PASS.

**Step 2: Run smoke validation if a sample slide is available**

Run: `pytest tests/test_wsi_smoke.py -v`
Expected: SKIP or PASS depending on fixture availability.

**Step 3: Verify CLI help**

Run: `python scripts/run_wsi_inference.py --help`
Expected: usage output listing slide path, output dir, levels, tile size, stride, batch size, white threshold, and quantization flags.

**Step 4: Commit**

```bash
git add tests README.md scripts/run_wsi_inference.py utils env.yaml setup.py
git commit -m "feat: add level-aware WSI inference pipeline"
```
