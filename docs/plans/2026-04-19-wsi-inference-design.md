# Eva WSI Inference Design

## Goal

Design a level-aware whole-slide inference pipeline for Eva that reads an H&E `.svs` slide, runs the fine-tuned H&E-to-spatial-proteomics model on non-overlapping tiles, skips near-white background tiles, and writes:

- one canonical multichannel `ome.tiff` with predicted biomarker channels
- one raw `.npy` tensor per requested level

Version 1 targets only level `0`, but the code should be structured so `levels=[...]` can later support multiple native slide levels with minimal refactoring.

## Scope Decisions

The approved scope for the first implementation is:

- Input slide format: `.svs`
- Inference levels: level `0` only for now
- Future extensibility: level list parameter from day one
- Output channel set: full biomarker set from `examples/biomarkers.npy`
- Current biomarker count: `52`
- Tile size: `224 x 224`
- Stride: `224` (non-overlapping)
- Background handling: skip tiles that are all white or near-white by threshold
- Raw output: per-level `float32` `.npy`
- OME-TIFF output: quantized representation for efficiency
- Memory strategy: stream writes and avoid keeping full-slide prediction tensors in RAM

## Recommended Approach

Use `OpenSlide + EvaMAE + NumPy memmap + tifffile`.

Why this approach:

- `OpenSlide` is the most direct fit for `.svs` level-aware reads.
- Eva already operates on fixed-size patch tensors and does not need model-side changes.
- `np.memmap` provides a practical out-of-core backing store for large raw prediction arrays.
- `tifffile` is a straightforward path for writing multichannel OME-TIFF with channel metadata.
- The resulting architecture is simple enough for a level-0 first version and cleanly extensible to multiple levels later.

Rejected alternatives for v1:

- `pyvips` end-to-end: attractive for large image I/O, but adds complexity around multichannel tensor assembly and dual-output writing.
- `zarr` intermediate: best long-term for resumable multiscale workflows, but unnecessary complexity for a first single-level implementation.

## Existing Repo Constraints

Observed from the current repo:

- Eva inference is patch-based, not WSI-native.
- The implemented H&E-to-MIF path is in `tutorials/virtual_stain.ipynb`.
- The fine-tuned checkpoint is `Eva_ft.ckpt`.
- The model expects patch input shaped `[B, H, W, C]`.
- H&E channels are represented as the last three channels and are reversed during this task (`1 - H&E`) to match the training convention shown in the notebook.

Strongly inferred implications:

- WSI inference must be an orchestration layer outside the core model files.
- The first implementation should be a new script plus helper modules rather than modifications to `Eva/mae.py` or `Eva/eva.py`.

## End-to-End Pipeline

### 1. Slide inspection

Open the `.svs` via OpenSlide and read:

- level count
- dimensions per level
- downsample factors

Even though v1 runs only level `0`, the API should normalize this into an iterable `levels=[0]`.

### 2. Tile planning

For each requested level:

- compute a regular grid of `224 x 224` windows
- use stride `224`
- allow partial edge tiles by padding or by restricting to full tiles

Recommended v1 behavior:

- iterate only full tiles first
- if edge coverage is required later, add a padded-edge option explicitly rather than silently resizing tiles

### 3. Background filtering

Before model inference, evaluate each tile for white-background skipping.

Approved rule:

- if every pixel is white or near-white, do not run inference

Recommended practical implementation:

- convert tile RGB values to normalized float
- define near-white as all channels above a configurable threshold such as `0.95` or `242/255`
- skip tile if the full tile satisfies that condition

For skipped tiles:

- write zeros into raw and OME outputs
- optionally record a skipped-tile count in logs

### 4. Model input construction

For each retained H&E tile:

- normalize RGB to `[0, 1]`
- reverse H&E intensities as `1 - tile`
- create zero placeholders for all MIF/spatial-proteomics channels
- concatenate placeholders with reversed H&E to form the model input

Follow the notebook contract:

- `marker_in = biomarkers + ['HECHA1', 'HECHA2', 'HECHA3']`
- `marker_out = [biomarkers]`
- `infer_mask[:len(biomarkers), :] = 1.0`

This yields one predicted image per biomarker for the tile.

### 5. Batched inference

Accumulate retained tiles into batches and run inference through `EvaMAE.from_checkpoint(...)`.

Expected tile output shape:

- per tile: `224 x 224 x 52`

Expected slide output shape per level:

- `[H_level, W_level, 52]`

### 6. Streaming raw output

For each requested level, allocate a disk-backed raw output array.

Recommended implementation:

- create a `.npy`-compatible `float32` memmap-backed array sized to `[H_level, W_level, 52]`
- write predicted tiles into the correct spatial windows in-place
- initialize the array to zeros so skipped tiles need no extra handling

This raw array is the authoritative quantitative artifact.

### 7. Streaming OME-TIFF output

Write a separate OME-TIFF artifact for the same level using streamed or blockwise writes.

Recommended semantics for v1:

- one OME-TIFF containing level `0` only
- channel axis contains the 52 biomarkers
- channel names recorded from `examples/biomarkers.npy`
- stored as quantized integer data for efficiency

Future multi-level support should be isolated behind the writer abstraction so the project can later choose between:

- separate series per level
- true pyramid/subIFD output
- one file per level

## Output Contract

For an input slide `sample.svs`, level `0`, the pipeline should produce conceptually:

- `sample.level0.predictions.npy`
- `sample.level0.predictions.ome.tiff`

Suggested output metadata:

- slide path
- source level
- level dimensions
- tile size
- stride
- biomarker names
- checkpoint path
- quantization policy
- white-threshold

## Quantization Policy

The user requested both:

- raw `.npy`
- quantized OME-TIFF

This means the OME-TIFF path must define a deterministic float-to-integer mapping.

Recommended v1 policy:

- raw `.npy` remains `float32`
- OME-TIFF uses `uint16`
- quantization policy is explicitly recorded in metadata

The implementation should make the policy pluggable. A reasonable first version is:

- compute or accept per-channel scaling parameters
- clamp values to a configured min/max range
- linearly map to `[0, 65535]`

Important note:

- the quantized OME-TIFF is for efficient storage and interoperability
- the raw `.npy` remains the canonical scientific output

## Proposed Code Structure

Recommended new files:

- `scripts/run_wsi_inference.py`
  - CLI entry point
- `utils/wsi_inference.py`
  - orchestration, tiling loop, batching, output coordination
- `utils/wsi_io.py`
  - slide-level reading helpers and OME/raw writers

Possible optional helper:

- `utils/wsi_background.py`
  - white/near-white tile detection

Keep existing Eva model code unchanged unless a tiny adapter materially reduces duplication.

## Configuration Surface

The first version should expose a small but extensible configuration surface:

- `--slide-path`
- `--output-dir`
- `--checkpoint-path`
- `--config-path`
- `--levels` default `[0]`
- `--tile-size` default `224`
- `--stride` default `224`
- `--batch-size`
- `--white-threshold`
- `--device`
- `--ome-dtype` default `uint16`
- `--quant-min`
- `--quant-max`

Future-ready options that can exist now or be added later:

- biomarker subset selection
- edge padding strategy
- multi-level output layout mode

## Error Handling

The pipeline should fail early and clearly if:

- the slide path does not exist
- OpenSlide cannot open the file
- requested level is missing
- `Eva_ft.ckpt` is missing
- `marker_embeddings/GenePT_embedding.pkl` is missing
- biomarker list cannot be loaded
- output directory is not writable

The pipeline should log:

- number of tiles planned
- number of tiles skipped as background
- number of tiles inferred
- per-level output paths

## RAM and Performance Strategy

The user explicitly requested stream-oriented saving to reduce RAM use.

The implementation should therefore avoid:

- building a full-slide raw array in memory before write
- building a second full-slide quantized array in memory

Instead:

- read tiles incrementally
- batch only enough tiles for efficient GPU inference
- write completed predictions directly into memmap-backed storage
- quantize and flush OME blocks incrementally

This keeps memory bounded by:

- one inference batch
- one small amount of writer-side buffering

## Testing Strategy

The first implementation should be tested at three layers.

### Unit tests

- white/near-white tile detection
- tile-grid generation
- quantization mapping
- channel metadata generation

### Integration tests

- synthetic fake slide read into tiles
- mock model returning deterministic outputs
- stitching and writeback into raw level array

### Smoke test

- one small real or fixture-backed slide region through the full script
- verify:
  - `.npy` file exists
  - OME-TIFF exists
  - output shapes and channel counts match expectation

## Open Risks

1. OME-TIFF layout for future multi-level export remains intentionally deferred.
   - v1 should isolate this behind a writer abstraction.

2. Quantization policy may need revision once real output ranges are observed.
   - raw `.npy` prevents information loss in the meantime.

3. Edge-tile handling is a likely follow-up requirement for exact full-slide coverage.
   - v1 should document whether it writes only full tiles or pads edges.

## Recommended Next Step

Create a detailed implementation plan that breaks this into:

- script and helper scaffolding
- tiling and background filtering
- model inference wrapper
- raw memmap writer
- OME-TIFF writer
- tests and smoke validation
