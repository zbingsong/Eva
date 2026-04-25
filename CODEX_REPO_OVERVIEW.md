# CODEX_REPO_OVERVIEW

## 1. Summary

Observed: This repository is a compact Python package plus tutorial notebooks for Eva, a pretrained masked-autoencoder-style vision transformer that works on multiplex immunofluorescence or spatial proteomics channels together with H&E channels. The shipped code is centered on inference, feature extraction, masked reconstruction, image translation, and a fine-tuned virtual staining example rather than on end-to-end training.

Observed: The codebase is structurally simple. Most core behavior lives in `Eva/mae.py`, `Eva/eva.py`, `Eva/layers.py`, and `Eva/utils.py`; notebooks in `tutorials/` are the main user-facing entry points; `utils/` contains small helper and visualization utilities; `downstream/` contains lightweight analysis heads that are not connected to the main runtime.

Strongly inferred: The primary workflow is notebook-driven inference from pretrained checkpoints, especially loading a checkpoint, preparing a `[B, H, W, C]` patch tensor plus marker names, then calling `EvaMAE` for either embedding extraction or masked/cross-modality reconstruction. This is supported by the absence of CLI/train entry points and by all documented usage flowing through `tutorials/basic.ipynb`, `tutorials/masked_prediction.ipynb`, and `tutorials/virtual_stain.ipynb`.

## 2. Architecture

Observed: Eva is organized as a small library around a two-stage transformer autoencoder in `Eva/mae.py`.

- `Eva/mae.py` is the architectural center. `MaskedAutoencoderViT` builds the channel encoder, patch encoder, and decoder; performs masking; loads marker embeddings from `marker_embeddings/GenePT_embedding.pkl`; and exposes `forward_encoder`, `forward_decoder`, and full reconstruction `forward`.
- `Eva/eva.py` wraps the MAE in `EvaMAE`, adapts between external patch layout `[B, H, W, C]` and the MAE’s internal `[B, C, H, W]`, reconstructs decoded patch tokens back to image space, and provides checkpoint loading plus feature extraction helpers.
- `Eva/layers.py` supplies the main building blocks: channel-agnostic patch embedding, masked attention/transformer blocks, and marker embedding lookup through gene-aligned embeddings.
- `Eva/masking.py` implements masking strategies used by the core model runtime, including random, patch, channel, H&E-only, MIF-only, and specified-channel masking.
- `Eva/utils.py` is a thin convenience layer for Hugging Face checkpoint download, local checkpoint loading, and standalone feature extraction.
- `tutorials/` is the practical orchestration layer. The notebooks show how the model is intended to be driven for embeddings, masked prediction, MIF-to-H&E translation, and H&E-to-MIF virtual staining.

Observed: There is no dedicated training script, CLI, service layer, or configuration framework beyond `OmegaConf.load("config.yaml")` in notebooks. The repo behaves more like an inference library with executable examples than a full pipeline application.

## 3. Key Components

- `MaskedAutoencoderViT` — two-stage model core. Role: patchify inputs per channel, inject marker embeddings, apply masking, mix across channels and then across patches, then decode requested output markers. Main file: `Eva/mae.py`. Interacts with `Eva/layers.py`, `Eva/masking.py`, `Eva/pos_embed.py`, and `utils/constant.py`. Observed.
- `EvaMAE` — inference wrapper around the MAE. Role: checkpoint loading, tensor layout conversion, image reconstruction reshaping, and feature extraction. Main file: `Eva/eva.py`. Interacts with `MaskedAutoencoderViT` and is the object used directly in notebooks. Observed.
- `Checkpoint/model loading helpers` — convenience entry points for users. Role: download `Eva_model.ckpt` from Hugging Face or load a local checkpoint and return an `EvaMAE` instance. Main file: `Eva/utils.py`. Observed.
- `Marker embedding subsystem` — converts marker names to dense vectors. Role: map protein marker names to genes via `utils/constant.py`, look up precomputed GenePT embeddings when available, and fall back to learned embeddings for unknown markers. Main files: `Eva/layers.py`, `utils/constant.py`, `marker_embeddings/GenePT_embedding.pkl`. Observed.
- `Notebook inference flows` — practical demos of the intended runtime. Role: load config/checkpoints/examples, build masks, and call the model for embeddings or cross-modality prediction. Main files: `tutorials/basic.ipynb`, `tutorials/masked_prediction.ipynb`, `tutorials/virtual_stain.ipynb`. Observed.
- `Visualization helpers` — overlay selected predicted channels for inspection. Main file: `utils/overlay.py`. Used in `tutorials/basic.ipynb` and `tutorials/virtual_stain.ipynb`. Observed.
- `Downstream analysis heads` — optional models for slide-level tasks after feature extraction. Main files: `downstream/ABMIL.py`, `downstream/survival.py`. They do not appear to be invoked by the tutorials or core Eva runtime. Observed.

## 4. Dependency Graph

- `tutorials/basic.ipynb` -> `config.yaml`, `Eva/utils.py`, `utils/overlay.py`, `examples/*.npy`
- `tutorials/masked_prediction.ipynb` -> `config.yaml`, `Eva/utils.py`, `Eva/eva.py`, `utils/helpers.py`, `examples/*.npy`
- `tutorials/virtual_stain.ipynb` -> `config.yaml`, `Eva/eva.py`, `utils/overlay.py`, `Eva_ft.ckpt`, `examples/*.npy`
- `Eva/utils.py` -> `huggingface_hub.hf_hub_download`, `Eva/eva.py`
- `Eva/eva.py` -> `Eva/mae.py`, `einops.rearrange`
- `Eva/mae.py` -> `Eva/layers.py`, `Eva/masking.py`, `Eva/pos_embed.py`, `marker_embeddings/GenePT_embedding.pkl`
- `Eva/layers.py` -> `timm`, `utils/constant.py`
- `downstream/survival.py` -> `torch`, `lifelines`, `sklearn`
- `downstream/ABMIL.py` -> `torch`

## 5. Execution Flow

### Primary flow

1. Load config and checkpoint.
   Observed: Notebooks load `config.yaml` through `OmegaConf.load(...)`. Model weights come either from Hugging Face via `Eva.utils.load_from_hf(...)` or from a local checkpoint via `EvaMAE.from_checkpoint(...)`.

2. Prepare an image patch tensor and marker names.
   Observed: External inputs are expected as patches shaped `[B, H, W, C]`, where channels correspond to MIF/spatial-proteomics markers and optionally H&E channels. The tutorials treat H&E as the last three channels with marker names `HECHA1`, `HECHA2`, `HECHA3`.

3. Enter `EvaMAE.forward(...)` or `extract_features(...)`.
   Observed: `EvaMAE.forward` permutes inputs to `[B, C, H, W]` and forwards into `MaskedAutoencoderViT`. `extract_features` calls the encoder only and either uses the patch-average or CLS representation.

4. Run the channel encoder.
   Observed: `MaskedAutoencoderViT.channel_forward(...)` patchifies each channel independently, projects per-channel patch tokens, adds marker embeddings, applies a chosen or provided mask, inserts a marker CLS token, and runs masked transformer blocks across channels for each patch position.

5. Run the patch encoder.
   Observed: `patch_forward(...)` keeps only the channel-level CLS stream, projects it to patch-mixer dimension, adds a patch CLS token and fixed 2D positional embeddings, then applies transformer blocks across spatial patches.

6. Decode requested output markers.
   Observed: `forward_decoder(...)` projects encoder latents to decoder dimension, repeats them across the requested output marker set, re-injects marker embeddings for `marker_out`, applies decoder transformer blocks, and predicts `token_size**2` values per patch token.

7. Reassemble image-space outputs.
   Observed: `EvaMAE.forward(...)` removes the decoder CLS token and rearranges predicted patch tokens back into `[B, H, W, C_out]`.

### Secondary flows

- `Feature extraction`.
  Observed: `Eva/utils.py:extract_features` and `EvaMAE.extract_features` run `forward_encoder(...)` only and flatten either patch-averaged or CLS features into `[B, D]`.

- `Masked reconstruction`.
  Observed: `tutorials/masked_prediction.ipynb` creates `infer_mask` tensors and calls `model(...)` to reconstruct randomly masked, patch-masked, or specified masked channels.

- `MIF -> H&E image translation`.
  Observed: `tutorials/masked_prediction.ipynb` concatenates real MIF channels with zero-valued H&E placeholders, masks the last three H&E channels in `infer_mask`, sets `marker_out` to H&E marker names, and decodes predicted H&E.

- `H&E -> MIF virtual staining`.
  Observed: `tutorials/virtual_stain.ipynb` loads `Eva_ft.ckpt`, concatenates zero-valued MIF placeholders with reversed H&E channels, masks all MIF channels in `infer_mask`, sets `marker_out` to the biomarker list, and decodes predicted MIF/spatial-proteomics channels.
  Strongly inferred: This is the repo’s clearest implementation of “generate spatial proteomics from H&E” because it is the only path that explicitly starts from H&E-only content and reconstructs the multiplex marker channels.

### Notebooks

- `tutorials/basic.ipynb` — embedding extraction from MIF-only, H&E-only, and concatenated multimodal patches. Observed.
- `tutorials/masked_prediction.ipynb` — masked reconstruction, channel prediction, and MIF-to-H&E translation. Observed.
- `tutorials/virtual_stain.ipynb` — fine-tuned H&E-to-MIF virtual staining plus overlay visualization. Observed.

## 6. Data Flow

Observed: External image data enters as patch tensors shaped `[B, H, W, C]`, not as whole-slide image readers or tiling pipelines. The repository ships example `.npy` patch arrays in `examples/`.

Observed: Inside `EvaMAE`, the tensor is converted to `[B, C, H, W]`. `PatchEmbedChannelFree` in `Eva/layers.py` applies the same convolution independently to each channel, yielding per-channel, per-patch token blocks. These are projected and fused with marker embeddings derived from marker names.

Observed: Masking is channel-by-patch, represented as `[C, N]` in core inference examples. `infer_mask` can be supplied explicitly to control which modalities or channels are reconstructed.

Observed: The channel encoder mixes information across channels within each patch location, while the patch encoder mixes spatial information across patch positions using the channel CLS summary. The decoder then conditions on `marker_out` so the same latent can be decoded into any requested target marker set.

Observed: Outputs leave the model as reconstructed patch images `[B, H, W, C_out]` or flattened embeddings `[B, D]`. Notebook visualization either displays raw single channels or builds RGB overlays using `utils/overlay.py`.

Strongly inferred: Whole-slide inference is not implemented directly in this repository. The user request mentions WSIs, but the observable code operates on already prepared 224×224 patches and does not contain WSI readers, tissue detection, slide tiling, or slide-level stitching. The likely WSI workflow is external tiling into patches, per-patch Eva inference, then optional external aggregation.

## 7. Configuration

Observed: The explicit configuration source is `config.yaml`, loaded through OmegaConf in the notebooks.

Observed: `config.yaml` is divided into:

- `ds` — patch/image and masking defaults such as `patch_size`, `token_size`, `marker_dim`, `mask_strategy`, and `mask_ratio`
- `cm` — channel-mixer width, depth, heads, and MLP ratio
- `pm` — patch-mixer width, depth, heads, MLP ratio, and output dimension
- `de` — decoder width, marker dimension, depth, heads, and MLP ratio

Observed: Runtime behavior also depends on non-YAML inputs:

- marker name lists passed as `marker_in` and `marker_out`
- optional `infer_mask` tensors passed directly to `EvaMAE.forward(...)`
- `channel_mode` in feature extraction helpers, which assumes H&E channels occupy the last three positions when selecting `"HE"` or `"MIF"`
- checkpoint path or Hugging Face repo ID

Strongly inferred: There is no layered config precedence system beyond “load YAML, then pass runtime arguments directly.” This is supported by the absence of Hydra composition, argparse, environment-variable parsing, or config merging code.

## 8. Environment Setup

### Documented

Observed: `README.md` documents:

- `conda env create -f env.yaml`
- `conda activate Eva`
- `pip install -e .`

Observed: The current user also stated that a repo-specific conda environment is already available and can be activated with `conda activate Eva`.

Observed: `env.yaml` specifies Python 3.12 and pip-installed dependencies including `torch`, `torchvision`, `einops`, `timm`, `huggingface-hub`, `numpy`, `omegaconf`, `pandas`, `lifelines`, and `scikit-learn`.

Observed: Core inference additionally depends on local marker embeddings at `marker_embeddings/GenePT_embedding.pkl`; the basic notebook states these should be downloaded from Zenodo if missing. In this checkout, that file is present.

Observed: `tutorials/virtual_stain.ipynb` expects a local fine-tuned checkpoint `Eva_ft.ckpt`. In this checkout, that file is present at the repo root.

### Inferred

Strongly inferred: Running notebook examples from the repo root is the most reliable path because tutorials import `Eva.*` and `utils.*` by source path and reference relative files such as `config.yaml`, `examples/*.npy`, and `marker_embeddings/GenePT_embedding.pkl`.

Unknown: The packaging/install story is somewhat ambiguous. `setup.py` uses `setuptools.find_packages()`, but this checkout has no `__init__.py` under `Eva/`, `find_packages()` returns `[]` in static inspection, and `eva.egg-info/top_level.txt` is empty. It is therefore unclear whether `pip install -e .` is functionally required, functionally sufficient, or mostly incidental to the documented workflow.

## 9. Optional Pipelines

- `Embedding pipeline`.
  Observed: Implemented through `Eva/utils.py:extract_features`, `EvaMAE.extract_features`, and `tutorials/basic.ipynb`.

- `Masked reconstruction / translation pipeline`.
  Observed: Implemented in `EvaMAE.forward(...)` and demonstrated in `tutorials/masked_prediction.ipynb`.

- `Virtual staining pipeline`.
  Observed: Implemented as a notebook workflow in `tutorials/virtual_stain.ipynb` using a fine-tuned checkpoint rather than separate model code.

- `Downstream slide/patient modeling`.
  Observed: `downstream/ABMIL.py` and `downstream/survival.py` provide generic MIL and survival-analysis modules, but no orchestration code in this repo wires them to Eva features.

- `Training pipeline`.
  Unknown: No training entry point or script was found in the inspected files.

## 10. Other Notes

Observed: The tutorials contain hard-coded `os.chdir(...)` statements pointing to author-local absolute paths in two notebooks and to the current repo path in `tutorials/virtual_stain.ipynb`. They are usage conveniences, not reusable orchestration.

Observed: `utils/helpers.py` defines notebook-side masking and patchify utilities separate from the production masking implementation in `Eva/masking.py`. The notebooks therefore use a mix of helper-side mask generation and model-side `infer_mask` handling.

Observed: H&E is treated as a special three-channel modality represented by synthetic marker names `HECHA1`, `HECHA2`, `HECHA3`, and some notebooks invert H&E intensities with `1 - he_patch_np` because training used reversed H&E colors.

## 11. Uncertainties

- Unknown: Whether editable installation is actually necessary or sufficient.
  Evidence gap: static inspection shows `find_packages()` discovers no packages in this checkout, but the notebooks may still work when executed from the repo root because Python can import from the source tree directly.

- Unknown: How the authors perform real WSI-scale inference outside the provided patch demos.
  Evidence gap: no slide reader, tiler, or stitcher is present; only patch-based examples are implemented.

- Unknown: Whether there are additional unpublished checkpoints or config variants for different tasks.
  Evidence gap: the repo exposes one base config, Hugging Face loading for `Eva_model.ckpt`, and one local fine-tuned checkpoint `Eva_ft.ckpt`, but no checkpoint registry or task catalog.

## 12. Coverage Report

- docs read: `README.md`, `setup.py`, `env.yaml`, `config.yaml`
- core code inspected: `Eva/eva.py`, `Eva/mae.py`, `Eva/layers.py`, `Eva/masking.py`, `Eva/pos_embed.py`, `Eva/utils.py`
- utility code inspected: `utils/constant.py`, `utils/helpers.py`, `utils/overlay.py`
- notebooks inspected: `tutorials/basic.ipynb`, `tutorials/masked_prediction.ipynb`, `tutorials/virtual_stain.ipynb`
- optional components inspected: `downstream/ABMIL.py`, `downstream/survival.py`
- static metadata inspected: `eva.egg-info/*`, file tree, checkpoint and marker-embedding presence
- skipped or lightly sampled: binary assets and data files such as `examples/*.npy`, `Eva_ft.ckpt`, `figures/model_structure.png`, compiled `__pycache__` files
