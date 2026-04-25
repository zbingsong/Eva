# Eva

## Overview ✨

Eva (**E**ncoding of **v**isual **a**tlas) is a foundation model for tissue imaging data that learns complex spatial representations of tissues at the molecular, cellular, and patient levels. Eva uses a novel vision transformer architecture and is pre-trained on masked image reconstruction of spatial proteomics and matched histopathology. 

### Model Architecture
<img src="figures/model_structure.png" width="80%">

## Installation ⚙️

```bash
git clone https://github.com/YAndrewL/Eva.git
cd Eva

conda env create -f env.yaml
conda activate Eva

pip install -e .  # ~10min
```

## Getting Started 🚀

👉 **Start with the [tutorials](https://github.com/YAndrewL/Eva/tree/main/tutorials)** for examples and visualizations.

Model weights are on the HuggingFace Hub: https://huggingface.co/yandrewl/Eva

They walk through:
- Loading the model from HuggingFace Hub
- Downloading marker embeddings
- Extracting embeddings
- Working with multi-modality inputs
- Masked prediction (random-, patch-, and channel-masking)
- Image translation (MIF -> H&E) and virtual staining (H&E -> MIF)

A minimal quick start:
```python
from Eva.utils import load_from_hf, extract_features
from omegaconf import OmegaConf
import torch

conf = OmegaConf.load("config.yaml")
device = "cuda" if torch.cuda.is_available() else "cpu"
model = load_from_hf(repo_id="yandrewl/Eva", conf=conf, device=device)

patch = torch.randn(1, 224, 224, 6)
biomarkers = ["DAPI", "CD3e", "CD20", "CD4", "CD8", "PanCK"]
features = extract_features(
    patch=patch,
    bms=[biomarkers],
    model=model,
    device=device,
    cls=False,
    channel_mode="full",
)
```


## Configuration 🛠️

The model requires a configuration file (YAML format) that specifies:
- Dataset parameters (patch_size, token_size, marker_dim, etc.)
- Channel mixer parameters (dim, n_layers, n_heads, etc.)
- Patch mixer parameters (dim, n_layers, n_heads, etc.)
- Decoder parameters (dim, n_layers, n_heads, etc.)

See `config.yaml` for an example configuration.

## WSI Inference

Run one-slide level-0 virtual staining and export both the raw `float32` memmap and the quantized OME-TIFF:

```bash
python scripts/run_wsi_inference.py \
  --slide-path /path/to/sample.svs \
  --output-dir outputs/wsi \
  --levels 0 \
  --biomarkers-path examples/biomarkers.npy
```

By default the CLI loads `config.yaml`, `examples/biomarkers.npy`, and `Eva_ft.ckpt` from the repo root, then writes:
- `outputs/wsi/level_0/predictions.npy`
- `outputs/wsi/level_0/predictions.ome.tiff`

OME-TIFF quantization modes:
- `global` (default): quantize each biomarker using the min/max over the whole level image
- `tile`: quantize each full inference tile independently before stitching into the OME-TIFF

In `tile` mode, `--quant-min` and `--quant-max` are ignored with a warning.

Smoke validation stays opt-in so it remains cheap by default. Point `EVA_WSI_SMOKE_SVS` at a small `.svs` sample, then run:

```bash
EVA_WSI_SMOKE_SVS=/path/to/small_sample.svs pytest tests/test_wsi_smoke.py -v
```

For a manual CLI spot-check on the same sample:

```bash
python scripts/run_wsi_inference.py \
  --slide-path /path/to/small_sample.svs \
  --output-dir outputs/wsi-smoke \
  --levels 0 \
  --tile-size 224 \
  --stride 224 \
  --batch-size 1 \
  --white-threshold 0.7843137254901961 \
  --quant-min 0.0 \
  --quant-max 1.0
```


## Citation 📚
Please check Eva paper at [bioRxiv](https://www.biorxiv.org/content/10.64898/2025.12.10.693553v1), and please cite as:

```
@article {Liu2025.12.10.693553,
	author = {Liu, Yufan and Sharma, Rishabh and Bieniosek, Matthew and Kang, Amy and Wu, Eric and Chou, Peter and Li, Irene and Rahim, Maha and Bauer, Erica and Ji, Ran and Duan, Wei and Qian, Li and Luo, Ruibang and Sharma, Padmanee and Dhanasekaran, Renu and Sch{\"u}rch, Christian M. and Charville, Gregory and Mayer, Aaron T. and Zou, James and Trevino, Alexandro E. and Wu, Zhenqin},
	title = {Modeling patient tissues at molecular resolution with Eva},
	elocation-id = {2025.12.10.693553},
	year = {2025},
	doi = {10.64898/2025.12.10.693553},
	publisher = {Cold Spring Harbor Laboratory},
	URL = {https://www.biorxiv.org/content/early/2025/12/12/2025.12.10.693553},
	eprint = {https://www.biorxiv.org/content/early/2025/12/12/2025.12.10.693553.full.pdf},
	journal = {bioRxiv}
}
```
