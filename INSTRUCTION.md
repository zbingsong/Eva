# How to Run

```
python scripts/run_wsi_inference.py 
    --slide-path <path to .svs> 
    --output-dir <output directory> 
    --levels <comma-separated list of WSI levels to run predictions> 
    --ome-quant-mode <global/tile/none>
```
`--ome-quant-mode` affects how the raw prediction logits are converted to the `ome.tiff` file:
- `tile`: logits are normalized per tile without considering other tiles;
- `global`: logits are normalized using global max/min logit values;
- `none`: do not perform quantization and save float values into the `ome.tiff` file directly.

