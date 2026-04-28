# How to

First go to Eva's HuggingFace repo and request access to the model. Once you are granted access, download the model's weights:
```bash
# assume you are inside Eva/
wget --header="Authorization: Bearer <your huggingface token>" https://huggingface.co/yandrewl/Eva/resolve/main/Eva_ft.ckpt
```

Run the following:

```bash
conda env create -f env.yaml
conda activate Eva
pip install -e .
pip install openslide-python tifffile

python scripts/run_wsi_inference.py \
    --slide-path <path to .svs> \
    --output-dir <output directory> \
    --levels <comma-separated list of WSI levels to run predictions> \
    --ome-quant-mode <global/tile/none>
```

`--ome-quant-mode` affects how the raw prediction logits are converted to the `ome.tiff` file:
- `tile`: logits are normalized per tile without considering other tiles;
- `global`: logits are normalized using global max/min logit values;
- `none`: do not perform quantization and save float values into the `ome.tiff` file directly.

# Important Things
- The WSI inference pipeline first split the input WSI level into non-overlapping tiles, and then identify background tiles using a heuristic: tiles where 99.9% of pixels have RGB values greater than 200 in all 3 channels are considered background (this means this tile is almost purely white).
- The prediction array is first initialized with `NaN`. Background tiles are skipped during inference. After model inference, background tiles are filled with global minimum for `global` and `none` mode, or directly made black for `tile` mode.

